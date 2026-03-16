#!/usr/bin/env python3
"""Debug mel length computation."""

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
import numpy as np

# Suppress NeMo logging
import logging
for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
os.environ["NEMO_LOG_LEVEL"] = "ERROR"

from pathlib import Path
from omegaconf import OmegaConf

def build_config(model_path: str):
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

# Build pipeline to get preprocessor
model_path = "../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
cfg = build_config(model_path)

from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
pipeline = PipelineBuilder.build_pipeline(cfg)
preprocessor = pipeline.asr_model.asr_model.preprocessor

print("Preprocessor config:")
print(f"  hop_length: {preprocessor.hop_length}")
print(f"  n_fft: {preprocessor.n_fft}")
print(f"  win_length: {preprocessor.win_length}")
print()

# Test different audio lengths
test_lengths = [19360, 17920, 16000, 1600, 3200]

for audio_len in test_lengths:
    # Create dummy audio
    audio = torch.randn(1, audio_len)
    length = torch.tensor([audio_len])

    with torch.no_grad():
        mel, mel_len = preprocessor(input_signal=audio, length=length)

    # Expected values
    expected_by_nemo = int(np.ceil(audio_len / preprocessor.hop_length))
    expected_with_center = audio_len // preprocessor.hop_length + 1

    print(f"Audio length: {audio_len}")
    print(f"  Actual MEL shape: {list(mel.shape)} (frames: {mel.shape[2]})")
    print(f"  MEL len returned: {mel_len.item()}")
    print(f"  ceil(audio_len/hop): {expected_by_nemo}")
    print(f"  audio_len/hop + 1: {expected_with_center}")
    print()
