#!/usr/bin/env python3
"""Debug script to dump NeMo mel spectrogram frames for comparison - detailed version."""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="Chunk size in milliseconds"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=3, help="Number of chunks to process"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}", file=sys.stderr)
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    model.eval()
    print(f"Model loaded", file=sys.stderr)

    # Setup encoder for streaming
    model.encoder.setup_streaming_params()
    streaming_cfg = model.encoder.streaming_cfg

    print(f"\nStreaming config:", file=sys.stderr)
    print(f"  chunk_size: {streaming_cfg.chunk_size}", file=sys.stderr)
    print(f"  shift_size: {streaming_cfg.shift_size}", file=sys.stderr)
    print(
        f"  pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}",
        file=sys.stderr,
    )

    # Load audio
    import soundfile as sf

    print(f"\nLoading audio: {args.audio}", file=sys.stderr)
    audio, sr = sf.read(args.audio, dtype="float32")
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    print(f"Audio duration: {len(audio) / sr:.2f}s", file=sys.stderr)

    # Initialize streaming
    chunk_samples = int(args.chunk_ms * sr / 1000)
    print(
        f"\nProcessing in {args.chunk_ms}ms chunks ({chunk_samples} samples)",
        file=sys.stderr,
    )

    # Compute features for full audio and process in chunks
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    preprocessor = model.preprocessor

    # Buffer for mel frames
    mel_buffer = None
    mel_idx = 0
    audio_idx = 0

    # Get chunk parameters
    chunk_size_first = (
        streaming_cfg.chunk_size[0]
        if isinstance(streaming_cfg.chunk_size, list)
        else streaming_cfg.chunk_size
    )
    chunk_size_rest = (
        streaming_cfg.chunk_size[1]
        if isinstance(streaming_cfg.chunk_size, list)
        else streaming_cfg.chunk_size
    )
    shift_size_first = (
        streaming_cfg.shift_size[0]
        if isinstance(streaming_cfg.shift_size, list)
        else streaming_cfg.shift_size
    )
    shift_size_rest = (
        streaming_cfg.shift_size[1]
        if isinstance(streaming_cfg.shift_size, list)
        else streaming_cfg.shift_size
    )
    pre_encode_first = (
        streaming_cfg.pre_encode_cache_size[0]
        if isinstance(streaming_cfg.pre_encode_cache_size, list)
        else 0
    )
    pre_encode_rest = (
        streaming_cfg.pre_encode_cache_size[1]
        if isinstance(streaming_cfg.pre_encode_cache_size, list)
        else streaming_cfg.pre_encode_cache_size
    )

    chunk_num = 0
    while audio_idx < len(audio) and chunk_num < args.num_chunks:
        # Add audio chunk
        end_idx = min(audio_idx + chunk_samples, len(audio))
        audio_chunk = audio_tensor[:, audio_idx:end_idx]
        audio_idx = end_idx

        # Compute mel for this chunk
        length = torch.tensor([audio_chunk.shape[1]])
        with torch.no_grad():
            processed_signal, _ = preprocessor(input_signal=audio_chunk, length=length)

        print(
            f"\n--- Audio chunk {(audio_idx - 1) // chunk_samples}, computed mel shape: {processed_signal.shape}",
            file=sys.stderr,
        )

        # Append to mel buffer
        if mel_buffer is None:
            mel_buffer = processed_signal
        else:
            mel_buffer = torch.cat([mel_buffer, processed_signal], dim=2)

        print(f"    mel_buffer shape after append: {mel_buffer.shape}", file=sys.stderr)

        # Check if we have enough mel frames
        chunk_size = chunk_size_first if chunk_num == 0 else chunk_size_rest
        pre_encode_cache = pre_encode_first if chunk_num == 0 else pre_encode_rest

        total_needed = chunk_size + pre_encode_cache
        available = mel_buffer.shape[2] - mel_idx

        if available < total_needed:
            continue

        # Extract mel chunk
        start = mel_idx - pre_encode_cache if chunk_num > 0 else mel_idx
        end = start + total_needed
        mel_chunk = mel_buffer[:, :, max(0, start) : end]

        # Advance buffer index
        shift_size = shift_size_first if chunk_num == 0 else shift_size_rest
        mel_idx += shift_size

        print(f"\n=== Chunk {chunk_num + 1} ===", file=sys.stderr)
        print(f"  mel_chunk shape: {mel_chunk.shape}", file=sys.stderr)
        print(f"  mel extract range: [{start}, {end}) in mel_buffer", file=sys.stderr)
        print(f"  mel_idx after shift: {mel_idx}", file=sys.stderr)

        # Print first few mel frames
        mel_np = mel_chunk[0].numpy()  # [F, T]

        print(f"  First 10 mel frames (first 3 freq bins each):", file=sys.stderr)
        for t in range(min(10, mel_np.shape[1])):
            vals = [mel_np[f, t] for f in range(3)]
            is_zero = all(abs(v) < 1e-6 for v in vals)
            zero_marker = " <-- ZEROS!" if is_zero else ""
            print(
                f"    frame {t}: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}, ...]{zero_marker}",
                file=sys.stderr,
            )

        chunk_num += 1

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
