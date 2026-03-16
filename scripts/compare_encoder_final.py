#!/usr/bin/env python3
"""
Final comparison of full encoder output between Python and Rust.
"""

import os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

audio, sr = sf.read('../samples/2086-149220-0033.wav', dtype='float32')
config_path = Path('streaming_config.yaml')
cfg = OmegaConf.load(config_path)
cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
cfg.asr.device = 'cpu'
cfg.asr.compute_dtype = 'float32'
cfg.asr.use_amp = False
cfg.streaming.att_context_size = [70, 1]
cfg.asr_decoding_type = 'rnnt'
cfg.log_level = 40

pipeline = PipelineBuilder.build_pipeline(cfg)
preprocessor = pipeline.asr_model.asr_model.preprocessor
encoder = pipeline.asr_model.asr_model.encoder

chunk = audio[:19360]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)

    encoded_t = encoded.transpose(1, 2)
    print(f"=== Python Full Encoder Output ===")
    print(f"Shape: {encoded_t.shape}, encoded_len: {encoded_len.item()}")
    for t in range(min(8, encoded_t.shape[1])):
        vals = encoded_t[0, t, :5].tolist()
        print(f"  T{t:02d} [:5]: [{', '.join(f'{v:.6f}' for v in vals)}]")
    if encoded_t.shape[1] > 8:
        print("  ...")
        t = encoded_len.item() - 1
        vals = encoded_t[0, t, :5].tolist()
        print(f"  T{t:02d} [:5]: [{', '.join(f'{v:.6f}' for v in vals)}]")

    # Rust values from earlier run:
    rust_values = [
        [-0.041328, -0.026628, 0.057415, -0.015438, -0.068392],
        [-0.050317, -0.031904, 0.060536, -0.011644, -0.061440],
        [-0.107618, 0.043788, -0.017621, -0.081881, -0.074037],
        [-0.084262, 0.063997, -0.013424, -0.090807, -0.070128],
        [-0.059877, 0.087548, 0.016241, -0.078298, -0.066933],
        [0.002169, 0.105739, 0.027543, -0.049664, -0.059549],
        [0.104615, 0.065348, -0.004508, -0.002019, -0.006552],
        [0.125606, 0.038368, -0.002571, 0.017080, -0.016992],
    ]

    print(f"\n=== Comparison with Rust ===")
    for t in range(min(8, len(rust_values))):
        py_vals = encoded_t[0, t, :5].tolist()
        rs_vals = rust_values[t]
        diff = [abs(p - r) for p, r in zip(py_vals, rs_vals)]
        max_diff = max(diff)
        print(f"  T{t:02d}: Python=[{', '.join(f'{v:.6f}' for v in py_vals)}]")
        print(f"       Rust  =[{', '.join(f'{v:.6f}' for v in rs_vals)}]")
        print(f"       MaxDiff={max_diff:.6f}")
