#!/usr/bin/env python3
"""
Debug script to compare subsampling in batch (non-streaming) mode.

This processes the entire audio at once and compares with Rust batch mode.

Run with:
    cd scripts && uv run python debug_batch_subsampling.py --model ../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo ../samples/2086-149220-0033.wav
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf


def build_config(model_path: str):
    """Build the pipeline configuration."""
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent / "streaming_config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = model_path
    cfg.asr.device = "cpu"
    cfg.asr.compute_dtype = "float32"
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 1]
    cfg.asr_decoding_type = "rnnt"
    cfg.log_level = 40

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Debug batch subsampling comparison")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to .nemo model")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--chunk-ms", type=int, default=1120, help="Chunk size in milliseconds")
    args = parser.parse_args()

    # Suppress logging
    import logging
    for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Load audio
    print(f"\n=== Loading Audio: {args.audio} ===")
    audio, sr = sf.read(args.audio, dtype="float32")
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio) / sr:.2f}s ({len(audio)} samples)")

    # Build pipeline
    print(f"\n=== Building Pipeline: {args.model} ===")
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    cfg = build_config(args.model)
    pipeline = PipelineBuilder.build_pipeline(cfg)

    # Get the model
    wrapper = pipeline.asr_model
    model = wrapper.asr_model
    preprocessor = model.preprocessor
    encoder = model.encoder
    pre_encode = encoder.pre_encode

    print(f"\n=== Processing first {args.chunk_ms}ms chunk in BATCH mode ===")

    # Process chunk
    chunk_samples = int(args.chunk_ms * sr / 1000)
    chunk = audio[:chunk_samples]

    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(chunk)])

    # Get MEL features
    with torch.no_grad():
        mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

    print(f"\nMEL output:")
    print(f"  shape: {list(mel.shape)} (batch, n_mels, time)")
    print(f"  mel_len: {mel_len.tolist()}")

    # Transpose for encoder: [B, F, T] -> [B, T, F]
    mel_transposed = mel.transpose(1, 2)
    print(f"\nMEL transposed for encoder:")
    print(f"  shape: {list(mel_transposed.shape)} (batch, time, n_mels)")

    # Run through pre_encode directly (batch mode)
    with torch.no_grad():
        subsampled, sub_len = pre_encode(mel_transposed, mel_len)

    print(f"\n=== Subsampling Output ===")
    print(f"  input shape: {list(mel_transposed.shape)}")
    print(f"  output shape: {list(subsampled.shape)}")
    print(f"  output_len: {sub_len.tolist()}")

    # Print first 5 values
    out_flat = subsampled.flatten()
    n = min(5, out_flat.numel())
    out_samples = [f"{out_flat[i].item():.6f}" for i in range(n)]
    print(f"  output first 5: [{', '.join(out_samples)}]")

    # Process multiple chunks
    print(f"\n=== Processing all chunks ===")
    offset = 0
    chunk_idx = 0
    while offset < len(audio):
        chunk = audio[offset:offset + chunk_samples]
        if len(chunk) < chunk_samples:
            # Pad the last chunk
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        chunk_len = torch.tensor([len(chunk)])

        with torch.no_grad():
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
            mel_transposed = mel.transpose(1, 2)
            subsampled, sub_len = pre_encode(mel_transposed, mel_len)

        out_flat = subsampled.flatten()
        n = min(5, out_flat.numel())
        out_samples = [f"{out_flat[i].item():.6f}" for i in range(n)]

        print(f"\nChunk {chunk_idx}: input={list(mel_transposed.shape)} output={list(subsampled.shape)}")
        print(f"  first 5: [{', '.join(out_samples)}]")

        offset += chunk_samples
        chunk_idx += 1


if __name__ == "__main__":
    main()
