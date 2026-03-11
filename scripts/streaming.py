#!/usr/bin/env python3
"""
NeMo Streaming ASR Example

This script demonstrates real-time streaming inference using NVIDIA NeMo,
processing audio in chunks and displaying partial transcriptions as they
become available.

Usage:
    python scripts/streaming.py \
        --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
        --mode 0 --chunk-ms 100 samples/Speaker26_004_16k.wav

Requirements:
    cd scripts && ./setup.sh
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


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    for module in ["numpy", "torch", "soundfile", "nemo"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("Error: Missing required dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print("  cd scripts && ./setup.sh")
        sys.exit(1)


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
        import librosa

        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    return data, sample_rate


def list_streaming_modes(model):
    """List available streaming modes from the model's encoder."""
    if (
        not hasattr(model.encoder, "streaming_cfg")
        or model.encoder.streaming_cfg is None
    ):
        model.encoder.setup_streaming_params()

    cfg = model.encoder.streaming_cfg
    if cfg is None:
        print("No streaming configuration available.")
        return

    # Try to get att_context_size which defines streaming modes
    if hasattr(model.encoder, "att_context_size"):
        att_context = model.encoder.att_context_size
        print(f"\nAttention context size: {att_context}")

    print("\nStreaming configuration:")
    print(f"  Chunk size: {cfg.chunk_size}")
    print(f"  Shift size: {cfg.shift_size}")
    print(f"  Valid output length: {cfg.valid_out_len}")
    if hasattr(cfg, "pre_encode_cache_size"):
        print(f"  Pre-encode cache size: {cfg.pre_encode_cache_size}")
    if hasattr(cfg, "last_channel_cache_size"):
        print(f"  Last channel cache size: {cfg.last_channel_cache_size}")
    if hasattr(cfg, "last_time_cache_size"):
        print(f"  Last time cache size: {cfg.last_time_cache_size}")
    if hasattr(cfg, "drop_extra_pre_encoded"):
        print(f"  Drop extra pre-encoded: {cfg.drop_extra_pre_encoded}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ASR transcription using NVIDIA NeMo"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Path to the .nemo model file"
    )
    parser.add_argument(
        "audio", type=str, nargs="?", help="Path to the audio file (.wav)"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Streaming mode index (0 = lowest latency, default: 0)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Audio chunk size in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available streaming modes and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress NeMo logging output"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each transcription update on a new line (for debugging)",
    )

    args = parser.parse_args()

    # Suppress NeMo logging if requested
    if args.quiet:
        import logging

        for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Check dependencies
    check_dependencies()

    # Import heavy dependencies after arg parsing
    import torch
    import numpy as np

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Loading model from: {model_path}")
    load_start = time.time()

    # Import NeMo and load model
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(
        str(model_path), map_location=args.device
    )
    model.eval()

    print(f"Model loaded in {time.time() - load_start:.2f}s")
    print(f"Model type: {type(model).__name__}")
    sample_rate = model.cfg.preprocessor.sample_rate
    print(f"Sample rate: {sample_rate} Hz")

    # Check if model supports streaming
    from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder

    if not isinstance(model.encoder, StreamingEncoder):
        print("Error: Model does not support streaming inference.")
        print("Use a streaming-capable model like Nemotron.")
        return 1

    # Setup streaming parameters
    # The mode is an index into att_context_size_all (list of [left, right] context sizes)
    # e.g., [[70, 13], [70, 6], [70, 1], [70, 0]] for nemotron-speech-streaming
    if hasattr(model.encoder, "att_context_size_all"):
        att_context_sizes = model.encoder.att_context_size_all
        if args.mode < 0 or args.mode >= len(att_context_sizes):
            print(
                f"Error: Invalid mode {args.mode}. Available modes: 0-{len(att_context_sizes) - 1}"
            )
            print(f"  Mode mappings: {list(enumerate(att_context_sizes))}")
            return 1
        att_context_size = att_context_sizes[args.mode]
        if hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size)
        else:
            model.encoder.att_context_size = att_context_size
            model.encoder.setup_streaming_params()
    else:
        model.encoder.setup_streaming_params()

    # List modes if requested
    if args.list_modes:
        list_streaming_modes(model)
        return 0

    # Check for audio file
    if not args.audio:
        print("Error: No audio file specified.")
        parser.print_help()
        return 1

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    samples, file_sample_rate = load_audio(str(audio_path), sample_rate)
    audio_duration = len(samples) / file_sample_rate
    print(
        f"Audio duration: {audio_duration:.2f}s ({len(samples)} samples at {file_sample_rate} Hz)"
    )

    # Setup streaming buffer
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    streaming_cfg = model.encoder.streaming_cfg
    print(f"\nInitializing streaming (mode {args.mode})")
    if isinstance(streaming_cfg.chunk_size, list):
        print(
            f"Chunk size: {streaming_cfg.chunk_size[0]} frames (first), {streaming_cfg.chunk_size[1]} frames (subsequent)"
        )
    else:
        print(f"Chunk size: {streaming_cfg.chunk_size} frames")

    if isinstance(streaming_cfg.shift_size, list):
        print(
            f"Shift size: {streaming_cfg.shift_size[0]} frames (first), {streaming_cfg.shift_size[1]} frames (subsequent)"
        )
    else:
        print(f"Shift size: {streaming_cfg.shift_size} frames")

    print(f"Valid output length: {streaming_cfg.valid_out_len} frames")
    if hasattr(streaming_cfg, "last_channel_cache_size"):
        cache_time = getattr(streaming_cfg, "last_time_cache_size", "N/A")
        print(
            f"Cache sizes: channel={streaming_cfg.last_channel_cache_size}, time={cache_time}"
        )

    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * args.chunk_ms / 1000)
    print(f"Processing in {args.chunk_ms} ms chunks ({chunk_samples} samples)")

    # Process audio using streaming buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False
    )

    # Add the audio file to the buffer
    processed_signal, processed_signal_length, stream_id = (
        streaming_buffer.append_audio_file(str(audio_path), stream_id=-1)
    )

    # Get initial cache state
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )

    # Process chunks
    print("\n--- Streaming Transcription ---")
    start = time.time()
    last_text = ""
    chunks_processed = 0
    previous_hypotheses = None
    previous_pred_out = None

    streaming_buffer_iter = iter(streaming_buffer)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = (
            streaming_cfg.drop_extra_pre_encoded if step_num != 0 else 0
        )

        with torch.inference_mode():
            result = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
                return_transcription=True,
                return_log_probs=False,
            )

            (
                previous_pred_out,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = result

        chunks_processed += 1

        # Display partial result if text changed
        if transcribed_texts:
            # For RNNT models, transcribed_texts contains Hypothesis objects
            if hasattr(transcribed_texts[0], "text"):
                current_text = transcribed_texts[0].text
            else:
                current_text = str(transcribed_texts[0])

            if current_text != last_text:
                if args.verbose:
                    # Print on new line for verbose/debug mode
                    print(current_text)
                else:
                    # Clear line and print new text (in-place update)
                    print(f"\r\033[K{current_text}", end="", flush=True)
                last_text = current_text

    elapsed = time.time() - start

    # Print final newline
    print()

    # Final result
    print("\n--- Final Result ---")
    print(f"Text: {last_text}")

    # Statistics
    print("\n--- Statistics ---")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Real-time factor: {elapsed / audio_duration:.2f}x")
    print(f"Chunks processed: {chunks_processed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
