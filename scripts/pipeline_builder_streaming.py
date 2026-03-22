#!/usr/bin/env python3
"""
Streaming ASR using PipelineBuilder.

This script demonstrates streaming transcription using NeMo's PipelineBuilder
which handles all pipeline construction automatically.

Usage:
    cd scripts
    source .venv/bin/activate
    python pipeline_builder_streaming.py ../samples/2086-149220-0033.wav
    python pipeline_builder_streaming.py --lookahead 480 audio.wav
    python pipeline_builder_streaming.py --model path/to/model.nemo audio.wav

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

# Valid lookahead sizes for multi-lookahead models (in ms)
VALID_LOOKAHEADS = {
    0: 0,     # No lookahead
    80: 1,    # 1 frame = 80ms (default, lowest latency)
    480: 6,   # 6 frames = 480ms
    1040: 13, # 13 frames = 1040ms (highest accuracy)
}


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


def run_streaming_transcription(pipeline, audio_path: str, verbose: bool = False):
    """
    Run streaming transcription on an audio file.

    Args:
        pipeline: Pipeline from PipelineBuilder
        audio_path: Path to audio file
        verbose: Print detailed output

    Returns:
        tuple: (final_text, elapsed_time, audio_duration)
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
    request_generator = pipeline.get_request_generator()

    # Create streaming options
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
        for requests in request_generator:
            step_outputs = pipeline.transcribe_step(requests)
            chunks_processed += 1

            # Process outputs for each stream
            for step_output in step_outputs:
                # Final transcript - complete segments after EOU detection
                if step_output.final_transcript:
                    final = step_output.final_transcript.strip()
                    if final:
                        print(f"\r[Final]   {final}")
                        if accumulated_text:
                            accumulated_text += " " + final
                        else:
                            accumulated_text = final

                # Partial transcript - current recognition hypothesis
                partial = step_output.partial_transcript.strip()
                if partial and partial != last_partial:
                    if verbose:
                        print(f"\r[Partial] {partial}", end="", flush=True)
                    last_partial = partial

    finally:
        pipeline.close_session()

    elapsed = time.time() - start_time

    # Final text is accumulated finals, or last partial if no finals emitted
    final_text = accumulated_text if accumulated_text else last_partial

    return final_text, elapsed, chunks_processed, audio_duration


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ASR using PipelineBuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default model
    python pipeline_builder_streaming.py audio.wav

    # Specify a model
    python pipeline_builder_streaming.py --model path/to/model.nemo audio.wav

    # Use different lookahead (latency vs accuracy tradeoff)
    python pipeline_builder_streaming.py --lookahead 480 audio.wav

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

    args = parser.parse_args()

    # Check audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Build configuration
    cfg = build_config(args.model, args.device, args.lookahead)

    # Build pipeline using PipelineBuilder
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    print("Building pipeline...")
    build_start = time.time()
    pipeline = PipelineBuilder.build_pipeline(cfg)
    print(f"Pipeline built in {time.time() - build_start:.2f}s")

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
