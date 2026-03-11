#!/usr/bin/env python3
"""Debug script to check raw mel buffer contents."""

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
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}", file=sys.stderr)
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    model.eval()
    print(f"Model loaded", file=sys.stderr)

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
    audio_idx = 0

    # Process 20 audio chunks to build up buffer
    for i in range(20):
        end_idx = min(audio_idx + chunk_samples, len(audio))
        audio_chunk = audio_tensor[:, audio_idx:end_idx]
        audio_idx = end_idx

        # Compute mel for this chunk
        length = torch.tensor([audio_chunk.shape[1]])
        with torch.no_grad():
            processed_signal, _ = preprocessor(input_signal=audio_chunk, length=length)

        # Append to mel buffer
        if mel_buffer is None:
            mel_buffer = processed_signal
        else:
            mel_buffer = torch.cat([mel_buffer, processed_signal], dim=2)

    print(f"\nFinal mel_buffer shape: {mel_buffer.shape}", file=sys.stderr)

    # Check frames around index 96-100
    print(
        f"\nMel buffer frames around 96-100 (first 3 freq bins each):", file=sys.stderr
    )
    for idx in range(94, 102):
        vals = [mel_buffer[0, f, idx].item() for f in range(3)]
        is_zero = all(abs(v) < 1e-6 for v in vals)
        zero_marker = " <-- ZEROS!" if is_zero else ""
        print(
            f"  buffer[{idx}]: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}, ...]{zero_marker}",
            file=sys.stderr,
        )

    # Also check the raw buffer for any zeros
    print(f"\nChecking for zero frames in mel_buffer:", file=sys.stderr)
    mel_np = mel_buffer[0].numpy()  # [128, T]
    zero_frames = []
    for t in range(mel_np.shape[1]):
        if np.allclose(mel_np[:, t], 0):
            zero_frames.append(t)
    if zero_frames:
        print(
            f"  Found {len(zero_frames)} zero frames at indices: {zero_frames[:20]}...",
            file=sys.stderr,
        )
    else:
        print(f"  No zero frames found!", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
