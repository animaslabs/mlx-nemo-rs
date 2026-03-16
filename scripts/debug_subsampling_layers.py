#!/usr/bin/env python3
"""
Debug script to compare subsampling layer outputs between NeMo and Rust implementations.

Computes MEL for streaming-chunk-sized audio blocks (not 100ms chunks) to match
the proper streaming approach where MEL boundaries align with encoder chunk boundaries.

Run with:
    cd scripts && uv run python debug_subsampling_layers.py --model ../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo ../samples/2086-149220-0033.wav
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
    parser = argparse.ArgumentParser(description="Debug subsampling layer comparison")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to .nemo model")
    parser.add_argument("audio", type=str, help="Path to audio file")
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

    # Streaming parameters from model config
    HOP_SAMPLES = 160  # 10ms at 16kHz
    CHUNK_SIZE = 112  # MEL frames per encoder chunk
    PRE_ENCODE_CACHE = 9  # subsampling_factor + 1 = 8 + 1 = 9
    SHIFT_SIZE = 112  # How much buffer_idx advances per chunk

    # Audio samples needed for one streaming chunk worth of MEL frames
    # First chunk needs (CHUNK_SIZE + PRE_ENCODE_CACHE) frames = 121 frames
    # Each subsequent chunk needs SHIFT_SIZE new frames = 112 frames
    FIRST_CHUNK_AUDIO_SAMPLES = (CHUNK_SIZE + PRE_ENCODE_CACHE) * HOP_SAMPLES  # 121 * 160 = 19360
    SUBSEQUENT_CHUNK_AUDIO_SAMPLES = SHIFT_SIZE * HOP_SAMPLES  # 112 * 160 = 17920

    print(f"\n=== Streaming Config ===")
    print(f"  Chunk size: {CHUNK_SIZE} MEL frames")
    print(f"  Pre-encode cache: {PRE_ENCODE_CACHE} frames")
    print(f"  First chunk audio: {FIRST_CHUNK_AUDIO_SAMPLES} samples ({FIRST_CHUNK_AUDIO_SAMPLES/sr*1000:.1f}ms)")
    print(f"  Subsequent chunk audio: {SUBSEQUENT_CHUNK_AUDIO_SAMPLES} samples ({SUBSEQUENT_CHUNK_AUDIO_SAMPLES/sr*1000:.1f}ms)")

    # Process audio in streaming-chunk-sized blocks
    print(f"\n=== Processing Audio (streaming-chunk-sized MEL) ===")

    audio_offset = 0
    encoder_call = 0
    buffer_idx = 0

    while audio_offset < len(audio):
        # Determine how much audio we need for this chunk
        if encoder_call == 0:
            audio_needed = FIRST_CHUNK_AUDIO_SAMPLES
        else:
            audio_needed = SUBSEQUENT_CHUNK_AUDIO_SAMPLES

        # Get audio for this chunk
        chunk_end = min(audio_offset + audio_needed, len(audio))
        chunk = audio[audio_offset:chunk_end]

        # Check if we have enough audio (skip if too short for meaningful MEL)
        if len(chunk) < HOP_SAMPLES:
            break

        # Pad if needed for last chunk
        if len(chunk) < audio_needed:
            chunk = np.pad(chunk, (0, audio_needed - len(chunk)))

        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        chunk_len = torch.tensor([len(chunk)])

        # Compute MEL for this streaming-chunk-sized audio block
        with torch.no_grad():
            mel_chunk, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

        num_mel_frames = mel_chunk.shape[2]
        valid_frames = mel_len.item()

        # Transpose for encoder: [B, F, T] -> [B, T, F]
        mel_transposed = mel_chunk.transpose(1, 2)
        mel_len_tensor = torch.tensor([valid_frames])

        # Run through subsampling
        with torch.no_grad():
            subsampled, sub_len = pre_encode(mel_transposed, mel_len_tensor)

        # Print results
        out_flat = subsampled.flatten()
        total_elements = out_flat.numel()

        # First 5 values
        n = min(5, total_elements)
        first_samples = [f"{out_flat[i].item():.6f}" for i in range(n)]

        # Middle 5 values
        mid_start = total_elements // 2
        mid_samples = [f"{out_flat[mid_start + i].item():.6f}" for i in range(min(5, total_elements - mid_start))]

        # Last 5 values
        last_start = max(0, total_elements - 5)
        last_samples = [f"{out_flat[last_start + i].item():.6f}" for i in range(min(5, total_elements - last_start))]

        # Statistics
        out_np = subsampled.detach().numpy().flatten()
        out_min = out_np.min()
        out_max = out_np.max()
        out_mean = out_np.mean()
        out_std = out_np.std()

        print(f"\nEncoder call {encoder_call}:")
        print(f"  Audio samples: {len(chunk)} ({len(chunk)/sr*1000:.1f}ms)")
        print(f"  MEL frames: {num_mel_frames} (valid: {valid_frames})")
        print(f"  MEL input shape: {list(mel_transposed.shape)}")
        print(f"  Subsampling output shape: {list(subsampled.shape)} (total: {total_elements})")
        print(f"  first 5: [{', '.join(first_samples)}]")
        print(f"  mid 5 (@{mid_start}): [{', '.join(mid_samples)}]")
        print(f"  last 5 (@{last_start}): [{', '.join(last_samples)}]")
        print(f"  stats: min={out_min:.6f}, max={out_max:.6f}, mean={out_mean:.6f}, std={out_std:.6f}")

        audio_offset += audio_needed
        encoder_call += 1

        # Stop after 8 calls for comparison
        if encoder_call >= 8:
            break

    print(f"\n=== Done: {encoder_call} encoder calls ===")


if __name__ == "__main__":
    main()
