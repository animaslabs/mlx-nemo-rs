#!/usr/bin/env python3
"""
NeMo Streaming ASR using Cache-Aware Pipeline

This script demonstrates real-time streaming inference using NVIDIA NeMo's
cache-aware pipeline API. It uses CacheAwarePipelineBuilder to create either
a CacheAwareRNNTPipeline or CacheAwareCTCPipeline for streaming transcription.

The cache-aware approach is based on:
"Stateful FastConformer with Cache-based Inference for Streaming Automatic
Speech Recognition" (Noroozi et al., ICASSP 2024)

Usage:
    python scripts/streaming.py \
        --model stt_en_fastconformer_hybrid_large_streaming_multi \
        --lookahead 80 samples/Speaker26_004_16k.wav

    python scripts/streaming.py \
        --model nvidia/nemotron-speech-streaming-en-0.6b \
        --decoder rnnt samples/Speaker26_004_16k.wav

Requirements:
    cd scripts && ./setup.sh

    Requires NeMo 2.2+ with the new inference API:
    - nemo.collections.asr.inference.factory.pipeline_builder
    - nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Encoder step length is 80ms for FastConformer models
ENCODER_STEP_LENGTH = 80  # ms
SAMPLE_RATE = 16000  # Hz

# Valid lookahead sizes for multi-lookahead models (in ms)
VALID_LOOKAHEADS = {
    0: 0,  # No lookahead
    80: 1,  # 1 frame = 80ms
    480: 6,  # 6 frames = 480ms
    1040: 13,  # 13 frames = 1040ms
}


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    for module in ["numpy", "torch", "soundfile", "nemo", "omegaconf", "hydra"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("Error: Missing required dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print("  cd scripts && ./setup.sh")
        sys.exit(1)

    # Check for the new inference API
    try:
        from nemo.collections.asr.inference.factory.pipeline_builder import (
            PipelineBuilder,
        )
    except ImportError:
        print("Error: NeMo's new inference API not found.")
        print("This script requires NeMo 2.2+ with the cache-aware pipeline API.")
        print("\nThe following modules are required:")
        print("  - nemo.collections.asr.inference.factory.pipeline_builder")
        print("  - nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline")
        print("\nPlease upgrade NeMo or check your installation.")
        sys.exit(1)


def lookahead_to_att_context(lookahead_ms: int, left_context: int = 70) -> list:
    """
    Convert lookahead in milliseconds to attention context size.

    Args:
        lookahead_ms: Lookahead size in milliseconds (0, 80, 480, or 1040)
        left_context: Left context size (default: 70 for FastConformer)

    Returns:
        [left_context, right_context] list for att_context_size config
    """
    if lookahead_ms in VALID_LOOKAHEADS:
        right_context = VALID_LOOKAHEADS[lookahead_ms]
    else:
        # For non-standard lookahead, compute frames (80ms per frame)
        right_context = lookahead_ms // ENCODER_STEP_LENGTH
        print(
            f"Warning: Non-standard lookahead {lookahead_ms}ms -> {right_context} frames"
        )
        print(
            f"Valid options for multi-lookahead models: {list(VALID_LOOKAHEADS.keys())} ms"
        )

    return [left_context, right_context]


def load_audio(audio_path: str, target_sample_rate: int = 16000):
    """Load and optionally resample audio file."""
    import soundfile as sf
    import numpy as np

    data, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sample_rate != target_sample_rate:
        try:
            import librosa

            print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
            data = librosa.resample(
                data, orig_sr=sample_rate, target_sr=target_sample_rate
            )
        except ImportError:
            print(
                f"Warning: librosa not installed, cannot resample from {sample_rate} Hz"
            )
            print("Audio must be 16kHz for accurate transcription.")

    return data, target_sample_rate


def build_config(args):
    """
    Build the pipeline configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        OmegaConf DictConfig for the pipeline
    """
    from omegaconf import OmegaConf

    # Load base config
    config_path = Path(__file__).parent / "streaming_config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    cfg.asr.model_name = args.model
    cfg.asr.device = args.device

    # Set compute dtype based on device
    if args.device == "cuda":
        cfg.asr.compute_dtype = "bfloat16"
        cfg.asr.use_amp = True
    else:
        cfg.asr.compute_dtype = "float32"
        cfg.asr.use_amp = False

    # Set attention context size based on lookahead
    cfg.streaming.att_context_size = lookahead_to_att_context(args.lookahead)

    # Set decoder type
    cfg.asr_decoding_type = args.decoder

    # Set log level
    if args.quiet:
        cfg.log_level = 40  # ERROR
    elif args.verbose:
        cfg.log_level = 10  # DEBUG
    else:
        cfg.log_level = 30  # WARNING

    return cfg


def run_streaming_inference(
    pipeline, audio_path: str, sample_rate: int, verbose: bool = False
):
    """
    Run streaming inference on an audio file with partial result display.

    This function processes audio through the pipeline and prints partial
    transcriptions as each chunk is processed.

    Args:
        pipeline: The cache-aware streaming pipeline
        audio_path: Path to the audio file
        sample_rate: Audio sample rate
        verbose: If True, print debug information

    Returns:
        tuple: (final_text, elapsed_time, chunks_processed, audio_duration)
    """
    import soundfile as sf

    # Get audio duration
    info = sf.info(audio_path)
    audio_duration = info.duration

    print(f"Audio duration: {audio_duration:.2f}s")

    # Get the request generator from the pipeline
    request_generator = pipeline.get_request_generator()

    # Create streaming options
    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    options = [ASRRequestOptions()]

    # Set up the request generator with audio files
    request_generator.set_audio_filepaths([audio_path], options)

    # Start timing
    start_time = time.time()
    chunks_processed = 0
    last_partial = ""
    accumulated_text = ""

    print("\n--- Streaming Transcription ---")

    # Open a session for streaming
    pipeline.open_session()

    try:
        # Process audio through the request generator
        # The generator yields batches of Request objects (Frame or FeatureBuffer)
        for requests in request_generator:
            # Use transcribe_step which handles both frames and feature buffers
            step_outputs = pipeline.transcribe_step(requests)
            chunks_processed += 1

            # Process step outputs
            for step_output in step_outputs:
                # Print final transcript segments as they're produced
                if step_output.final_transcript:
                    final = step_output.final_transcript.strip()
                    if final:
                        print(f"[Final] {final}")
                        if accumulated_text:
                            accumulated_text += " " + final
                        else:
                            accumulated_text = final

                # Print partial transcript updates (shows what's being recognized)
                partial = step_output.partial_transcript.strip()
                if partial and partial != last_partial:
                    if verbose:
                        print(f"[Partial] {partial}")
                    last_partial = partial

    finally:
        pipeline.close_session()

    elapsed = time.time() - start_time

    # Final text is the accumulated finalized transcripts
    final_text = accumulated_text if accumulated_text else last_partial

    return final_text, elapsed, chunks_processed, audio_duration


def run_batch_inference(pipeline, audio_path: str, verbose: bool = False):
    """
    Run batch inference on an audio file (simpler, no streaming display).

    Args:
        pipeline: The cache-aware streaming pipeline
        audio_path: Path to the audio file
        verbose: If True, print debug information

    Returns:
        tuple: (final_text, elapsed_time, audio_duration)
    """
    import soundfile as sf

    # Get audio duration
    info = sf.info(audio_path)
    audio_duration = info.duration

    print(f"Audio duration: {audio_duration:.2f}s")
    print("\n--- Processing ---")

    # Run the pipeline
    start_time = time.time()
    output = pipeline.run([audio_path])
    elapsed = time.time() - start_time

    # Extract transcription
    if output and len(output) > 0:
        result = output[0]
        # The result can be a dict or an object depending on output format
        if isinstance(result, dict):
            final_text = result.get("text", str(result))
        elif hasattr(result, "text"):
            final_text = result.text
        elif hasattr(result, "transcript"):
            final_text = result.transcript
        else:
            final_text = str(result)
    else:
        final_text = ""

    return final_text, elapsed, audio_duration


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ASR transcription using NeMo Cache-Aware Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic streaming transcription
    python scripts/streaming.py \\
        --model stt_en_fastconformer_hybrid_large_streaming_multi \\
        samples/Speaker26_004_16k.wav

    # With specific lookahead (latency vs accuracy tradeoff)
    python scripts/streaming.py \\
        --model nvidia/nemotron-speech-streaming-en-0.6b \\
        --lookahead 480 samples/Speaker26_004_16k.wav

    # Using CTC decoder instead of RNNT
    python scripts/streaming.py \\
        --model stt_en_fastconformer_hybrid_large_streaming_multi \\
        --decoder ctc samples/Speaker26_004_16k.wav

Lookahead options for multi-lookahead models:
    0ms    - Lowest latency, lower accuracy
    80ms   - Default, good balance (1 frame)
    480ms  - Higher accuracy (6 frames)
    1040ms - Highest accuracy, highest latency (13 frames)
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model name (HuggingFace/NGC) or path to .nemo file. "
        "Recommended: stt_en_fastconformer_hybrid_large_streaming_multi "
        "or nvidia/nemotron-speech-streaming-en-0.6b",
    )
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Path to the audio file (.wav, 16kHz mono recommended)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=80,
        help="Lookahead size in milliseconds (default: 80). "
        "Valid options for multi-lookahead models: 0, 80, 480, or 1040.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="rnnt",
        choices=["ctc", "rnnt"],
        help="Decoder type (default: rnnt). RNNT typically has higher accuracy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode (process entire file at once, no streaming display)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress logging output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug output"
    )

    args = parser.parse_args()

    # Check for audio file
    if not args.audio:
        print("Error: No audio file specified.")
        parser.print_help()
        return 1

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Suppress logging if quiet mode
    if args.quiet:
        import logging

        for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Check dependencies
    check_dependencies()

    # Import pipeline builder after dependency check
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    # Build configuration
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Decoder: {args.decoder.upper()}")
    print(f"Lookahead: {args.lookahead}ms")

    cfg = build_config(args)

    # Build the pipeline
    print("\nBuilding pipeline...")
    build_start = time.time()
    pipeline = PipelineBuilder.build_pipeline(cfg)
    print(f"Pipeline built in {time.time() - build_start:.2f}s")

    # Run inference
    print(f"\nProcessing: {audio_path}")

    if args.batch:
        # Batch mode - process entire file at once
        final_text, elapsed, audio_duration = run_batch_inference(
            pipeline, str(audio_path), args.verbose
        )
        chunks_processed = 1
    else:
        # Streaming mode - show partial results
        final_text, elapsed, chunks_processed, audio_duration = run_streaming_inference(
            pipeline, str(audio_path), SAMPLE_RATE, args.verbose
        )

    # Print final results
    print("\n--- Final Result ---")
    print(f"Text: {final_text}")

    # Statistics
    print("\n--- Statistics ---")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Real-time factor: {elapsed / audio_duration:.2f}x")
    if not args.batch:
        print(f"Chunks processed: {chunks_processed}")
    print(f"Lookahead: {args.lookahead}ms")
    print(f"Attention context: {cfg.streaming.att_context_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
