#!/usr/bin/env python3
"""
Detailed comparison script for subsampling outputs.
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf


def build_config(model_path: str):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("audio", type=str)
    args = parser.parse_args()

    import logging
    for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Load audio
    audio, sr = sf.read(args.audio, dtype="float32")
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # Build pipeline
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    cfg = build_config(args.model)
    pipeline = PipelineBuilder.build_pipeline(cfg)

    wrapper = pipeline.asr_model
    model = wrapper.asr_model
    preprocessor = model.preprocessor
    encoder = model.encoder
    pre_encode = encoder.pre_encode

    # Process first chunk (19360 samples)
    chunk_samples = 19360
    chunk = audio[:chunk_samples]
    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(chunk)])

    with torch.no_grad():
        mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

    print(f"\nMEL output shape: {list(mel.shape)}")
    print(f"MEL len: {mel_len.item()}")

    # Transpose: [B, F, T] -> [B, T, F]
    mel_transposed = mel.transpose(1, 2)
    print(f"MEL transposed shape: {list(mel_transposed.shape)}")

    # Check MEL values at specific positions
    mel_np = mel_transposed[0].numpy()
    print(f"\nMEL frame 0, first 5 features: {mel_np[0, :5]}")
    print(f"MEL frame 60, first 5 features: {mel_np[60, :5]}")
    print(f"MEL frame 121 (last), first 5 features: {mel_np[121, :5]}")

    # Run subsampling
    with torch.no_grad():
        subsampled, sub_len = pre_encode(mel_transposed, mel_len)

    print(f"\nSubsampling output shape: {list(subsampled.shape)}")
    print(f"Subsampling output len: {sub_len.item()}")

    # Detailed output comparison
    out_np = subsampled[0].numpy()  # [T, D]
    print(f"\nOutput tensor shape: {out_np.shape}")

    # First time step
    print(f"\nTime step 0:")
    print(f"  features 0-4: {out_np[0, :5]}")
    print(f"  features 512-516: {out_np[0, 512:517]}")
    print(f"  features 1019-1023: {out_np[0, 1019:1024]}")

    # Middle time step
    mid_t = out_np.shape[0] // 2
    print(f"\nTime step {mid_t}:")
    print(f"  features 0-4: {out_np[mid_t, :5]}")
    print(f"  features 512-516: {out_np[mid_t, 512:517]}")
    print(f"  features 1019-1023: {out_np[mid_t, 1019:1024]}")

    # Last time step
    last_t = out_np.shape[0] - 1
    print(f"\nTime step {last_t}:")
    print(f"  features 0-4: {out_np[last_t, :5]}")
    print(f"  features 512-516: {out_np[last_t, 512:517]}")
    print(f"  features 1019-1023: {out_np[last_t, 1019:1024]}")

    # Dump all values for first chunk comparison
    print(f"\n=== All first 50 values (flattened) ===")
    flat = out_np.flatten()
    for i in range(50):
        print(f"  [{i}] = {flat[i]:.6f}")


if __name__ == "__main__":
    main()
