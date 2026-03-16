#!/usr/bin/env python3
"""
Debug MaskedConvSequential and apply_channel_mask.
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
subsampling = encoder.pre_encode

# Get the apply_channel_mask function
from nemo.collections.asr.parts.submodules.subsampling import apply_channel_mask, calculate_conv_output_size
print(f"=== apply_channel_mask source ===")
print(inspect.getsource(apply_channel_mask))

print(f"\n=== calculate_conv_output_size source ===")
print(inspect.getsource(calculate_conv_output_size))

# Get _create_mask method
print(f"\n=== MaskedConvSequential._create_mask source ===")
print(inspect.getsource(subsampling.conv._create_mask))
