#!/usr/bin/env python3
"""
Compare encoder output with and without pad_mask.
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
    mel_transposed = mel.transpose(1, 2)
    x, lengths = encoder.pre_encode(mel_transposed, mel_len)
    x_pe, pos_emb = encoder.pos_enc(x=x, cache_len=0)

    # Run through ALL layers manually WITHOUT pad_mask
    print("=== Running through all layers WITHOUT pad_mask ===")
    current_x = x_pe
    for i, layer in enumerate(encoder.layers):
        current_x = layer(x=current_x, att_mask=None, pos_emb=pos_emb, pad_mask=None)
        if i < 3 or i >= 21:  # Print first 3 and last 3 layers
            print(f"After layer {i:2d} T00[:5]: {current_x[0, 0, :5].tolist()}")
        elif i == 3:
            print("...")

    print(f"\n=== Final output WITHOUT pad_mask ===")
    print(f"  T00 [:5]: {current_x[0, 0, :5].tolist()}")
    print(f"  T01 [:5]: {current_x[0, 1, :5].tolist()}")
    print(f"  T02 [:5]: {current_x[0, 2, :5].tolist()}")

    # Compare with actual encoder output (WITH pad_mask)
    encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)
    encoded_t = encoded.transpose(1, 2)
    print(f"\n=== Actual encoder output (WITH pad_mask) ===")
    print(f"  T00 [:5]: {encoded_t[0, 0, :5].tolist()}")
    print(f"  T01 [:5]: {encoded_t[0, 1, :5].tolist()}")
    print(f"  T02 [:5]: {encoded_t[0, 2, :5].tolist()}")

    # Rust values
    print(f"\n=== Rust encoder output (NO pad_mask in conv) ===")
    print(f"  T00 [:5]: [-0.041328, -0.026628, 0.057415, -0.015438, -0.068392]")
    print(f"  T01 [:5]: [-0.050317, -0.031904, 0.060536, -0.011644, -0.061440]")
    print(f"  T02 [:5]: [-0.107618, 0.043788, -0.017621, -0.081881, -0.074037]")
