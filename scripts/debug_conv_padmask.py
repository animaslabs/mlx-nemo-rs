#!/usr/bin/env python3
"""
Debug pad_mask convention in NeMo conv module.
"""

import os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
import inspect

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
encoder = pipeline.asr_model.asr_model.encoder
layer0 = encoder.layers[0]
conv = layer0.conv

# Check the conv forward code to understand pad_mask
print("=== ConformerConvolution forward code ===")
forward_src = inspect.getsource(conv.forward)
print(forward_src)

# Check the layer0 forward code
print("\n=== ConformerLayer forward code ===")
layer_forward_src = inspect.getsource(layer0.forward)
# Print just the conv part
lines = layer_forward_src.split('\n')
for i, line in enumerate(lines):
    if 'conv' in line.lower():
        print(f"Line {i}: {line}")

# Check how encoder creates pad_mask
print("\n=== Encoder forward_internal code (pad_mask creation) ===")
encoder_forward = inspect.getsource(encoder.forward_internal)
lines = encoder_forward.split('\n')
for i, line in enumerate(lines):
    if 'pad_mask' in line.lower() or 'mask' in line.lower():
        # Print surrounding context
        start = max(0, i-2)
        end = min(len(lines), i+3)
        for j in range(start, end):
            print(f"Line {j}: {lines[j]}")
        print("---")
