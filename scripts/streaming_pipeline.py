#!/usr/bin/env python3
"""
Streaming ASR using CacheAwareRNNTPipeline directly.

This script demonstrates streaming transcription from an audio file using
NeMo's CacheAwareRNNTPipeline class. It shows:

1. Direct pipeline construction without PipelineBuilder
2. Frame-by-frame streaming with partial transcript updates
3. Cache-aware encoder processing
4. Real-time transcription output

Usage:
    cd scripts
    source .venv/bin/activate
    python streaming_pipeline.py ../samples/2086-149220-0033.wav
    python streaming_pipeline.py --lookahead 480 audio.wav
    python streaming_pipeline.py --model path/to/model.nemo audio.wav

Requirements:
    cd scripts && ./setup.sh
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import logging
for name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
    logging.getLogger(name).setLevel(logging.ERROR)
os.environ["NEMO_LOG_LEVEL"] = "ERROR"


# Encoder step length is 80ms for FastConformer models
ENCODER_STEP_LENGTH = 80  # ms
SAMPLE_RATE = 16000  # Hz

# Valid lookahead sizes for multi-lookahead models (in ms)
VALID_LOOKAHEADS = {
    0: 0,     # No lookahead
    80: 1,    # 1 frame = 80ms (default, lowest latency)
    480: 6,   # 6 frames = 480ms
    1040: 13, # 13 frames = 1040ms (highest accuracy)
}


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
            print(f"Warning: librosa not installed, cannot resample from {sample_rate} Hz")

    return data, target_sample_rate


def build_config(model_path: str, device: str, lookahead_ms: int):
    """
    Build the pipeline configuration.

    Args:
        model_path: Path to .nemo model file or model name
        device: Device to run on (cpu, cuda, mps)
        lookahead_ms: Lookahead in milliseconds (0, 80, 480, or 1040)

    Returns:
        OmegaConf DictConfig for the pipeline
    """
    from omegaconf import OmegaConf

    # Load base config
    config_path = Path(__file__).parent / "streaming_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Model settings
    cfg.asr.model_name = model_path
    cfg.asr.device = device

    # Compute dtype based on device
    if device == "cuda":
        cfg.asr.compute_dtype = "bfloat16"
        cfg.asr.use_amp = True
    else:
        cfg.asr.compute_dtype = "float32"
        cfg.asr.use_amp = False

    # Attention context size based on lookahead
    if lookahead_ms in VALID_LOOKAHEADS:
        right_context = VALID_LOOKAHEADS[lookahead_ms]
    else:
        right_context = lookahead_ms // ENCODER_STEP_LENGTH
        print(f"Warning: Non-standard lookahead {lookahead_ms}ms -> {right_context} frames")

    cfg.streaming.att_context_size = [70, right_context]  # [left_context, right_context]
    cfg.asr_decoding_type = "rnnt"
    cfg.log_level = 40  # ERROR

    return cfg


def load_asr_model(cfg):
    """
    Load the ASR model and wrap it for cache-aware inference.

    Args:
        cfg: Pipeline configuration

    Returns:
        CacheAwareRNNTInferenceWrapper instance
    """
    from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
        CacheAwareRNNTInferenceWrapper,
    )

    print(f"Loading model: {cfg.asr.model_name}")

    # The wrapper handles model loading internally
    wrapper = CacheAwareRNNTInferenceWrapper(
        model_name=cfg.asr.model_name,
        device=cfg.asr.device,
        device_id=cfg.asr.device_id,
        compute_dtype=cfg.asr.compute_dtype,
        use_amp=cfg.asr.use_amp,
        decoding_cfg=cfg.asr.decoding,
    )

    return wrapper


def create_pipeline(cfg, asr_model):
    """
    Create the CacheAwareRNNTPipeline directly.

    Args:
        cfg: Pipeline configuration
        asr_model: CacheAwareRNNTInferenceWrapper instance

    Returns:
        CacheAwareRNNTPipeline instance
    """
    from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import (
        CacheAwareRNNTPipeline,
    )

    pipeline = CacheAwareRNNTPipeline(
        cfg=cfg,
        asr_model=asr_model,
        itn_model=None,  # No inverse text normalization
        nmt_model=None,  # No translation
    )

    return pipeline


def run_streaming_transcription(pipeline, audio_path: str, verbose: bool = False):
    """
    Run streaming transcription on an audio file.

    This demonstrates the frame-by-frame streaming approach using
    CacheAwareRNNTPipeline.

    Args:
        pipeline: CacheAwareRNNTPipeline instance
        audio_path: Path to audio file
        verbose: Print detailed output

    Returns:
        tuple: (final_text, elapsed_time, chunks_processed, audio_duration)
    """
    import soundfile as sf
    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    # Get audio duration
    info = sf.info(audio_path)
    audio_duration = info.duration

    print(f"\n{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Duration: {audio_duration:.2f}s")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"{'='*60}")

    # Get the request generator from the pipeline
    # This handles audio chunking and mel spectrogram computation
    request_generator = pipeline.get_request_generator()

    # Create streaming options (default settings)
    options = [ASRRequestOptions()]

    # Set up the request generator with the audio file
    request_generator.set_audio_filepaths([audio_path], options)

    # Start timing
    start_time = time.time()
    chunks_processed = 0
    last_partial = ""
    accumulated_text = ""

    print(f"\n--- Streaming Transcription ---\n")

    # Open a streaming session
    pipeline.open_session()

    try:
        # Process audio through the request generator
        # Each iteration yields a batch of Frame/FeatureBuffer requests
        for requests in request_generator:
            # Debug: Print request details
            print(f"\n{'─'*40}")
            print(f"[Chunk {chunks_processed}] Processing {len(requests)} request(s)")
            for i, req in enumerate(requests):
                req_type = type(req).__name__
                print(f"  Request[{i}]: type={req_type}")
                if hasattr(req, 'features') and req.features is not None:
                    feat = req.features
                    print(f"    features: shape={feat.shape}, dtype={feat.dtype}")
                    print(f"    features min/max: {feat.min():.4f} / {feat.max():.4f}")
                if hasattr(req, 'audio') and req.audio is not None:
                    audio = req.audio
                    print(f"    audio: shape={audio.shape}, dtype={audio.dtype}")
                if hasattr(req, 'is_final'):
                    print(f"    is_final: {req.is_final}")
                if hasattr(req, 'stream_id'):
                    print(f"    stream_id: {req.stream_id}")

            # transcribe_step processes the requests and updates internal state
            step_outputs = pipeline.transcribe_step(requests)
            chunks_processed += 1

            # Debug: Print step output details
            print(f"  Step outputs: {len(step_outputs)} output(s)")

            # Process outputs for each stream (we have a single stream)
            for idx, step_output in enumerate(step_outputs):
                print(f"  Output[{idx}]:")
                print(f"    partial_transcript: '{step_output.partial_transcript}'")
                print(f"    final_transcript: '{step_output.final_transcript}'")
                if hasattr(step_output, 'tokens') and step_output.tokens:
                    print(f"    tokens: {step_output.tokens}")
                if hasattr(step_output, 'timestamps') and step_output.timestamps:
                    print(f"    timestamps: {step_output.timestamps}")
                if hasattr(step_output, 'confidence') and step_output.confidence is not None:
                    print(f"    confidence: {step_output.confidence}")

                # Final transcript - complete segments after EOU detection
                if step_output.final_transcript:
                    final = step_output.final_transcript.strip()
                    if final:
                        # Print finalized segment
                        print(f"\r[Final]   {final}")
                        if accumulated_text:
                            accumulated_text += " " + final
                        else:
                            accumulated_text = final

                # Partial transcript - current recognition hypothesis
                partial = step_output.partial_transcript.strip()
                if partial and partial != last_partial:
                    # Show partial transcript with carriage return for updates
                    if verbose:
                        print(f"\r[Partial] {partial}", end="", flush=True)
                    last_partial = partial

    finally:
        pipeline.close_session()

    elapsed = time.time() - start_time

    # Final text is accumulated finals, or last partial if no finals emitted
    final_text = accumulated_text if accumulated_text else last_partial

    return final_text, elapsed, chunks_processed, audio_duration


def print_pipeline_info(pipeline, cfg):
    """Print information about the pipeline configuration."""
    print(f"\n{'='*60}")
    print("Pipeline Configuration")
    print(f"{'='*60}")
    print(f"Sample rate: {pipeline.sample_rate} Hz")
    print(f"Model chunk size: {pipeline.model_chunk_size} mel frames")
    print(f"Chunk duration: {pipeline.chunk_size_in_secs:.3f}s")
    print(f"Buffer duration: {pipeline.buffer_size_in_secs:.3f}s")
    print(f"Pre-encode cache: {pipeline.pre_encode_cache_size} frames")
    print(f"Tokens per frame: {pipeline.tokens_per_frame}")
    print(f"Subsampling factor: {pipeline.subsampling_factor}")
    print(f"Use cache: {pipeline.use_cache}")
    print(f"Use feature cache: {pipeline.use_feat_cache}")
    print(f"Attention context: {cfg.streaming.att_context_size}")
    print(f"Vocabulary size: {len(pipeline.vocabulary)}")
    print(f"Blank ID: {pipeline.blank_id}")
    print(f"Device: {pipeline.device}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ASR using CacheAwareRNNTPipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default model
    python streaming_pipeline.py audio.wav

    # Specify a model
    python streaming_pipeline.py --model path/to/model.nemo audio.wav

    # Use different lookahead (latency vs accuracy tradeoff)
    python streaming_pipeline.py --lookahead 480 audio.wav

Lookahead options:
    0ms    - No lookahead (lowest latency)
    80ms   - Default (1 frame)
    480ms  - Medium (6 frames)
    1040ms - Full (13 frames, highest accuracy)
        """,
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file (.wav, 16kHz mono recommended)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
        help="Path to .nemo model file or pretrained model name",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=80,
        help="Lookahead in milliseconds (default: 80). Options: 0, 80, 480, 1040",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show partial transcripts in real-time",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print detailed pipeline information",
    )

    args = parser.parse_args()

    # Check audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Build configuration
    cfg = build_config(args.model, args.device, args.lookahead)

    # Load ASR model
    print("Loading model...")
    load_start = time.time()
    asr_model = load_asr_model(cfg)
    print(f"Model loaded in {time.time() - load_start:.2f}s")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline(cfg, asr_model)

    # Print pipeline info if requested
    # if args.info:
    print_pipeline_info(pipeline, cfg)

    # Run streaming transcription
    final_text, elapsed, chunks, audio_duration = run_streaming_transcription(
        pipeline, str(audio_path), args.verbose
    )

    # Print results
    print(f"\n{'='*60}")
    print("Final Transcription")
    print(f"{'='*60}")
    print(f"{final_text}")

    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Real-time factor: {elapsed / audio_duration:.2f}x")
    print(f"Chunks processed: {chunks}")
    print(f"Lookahead: {args.lookahead}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
