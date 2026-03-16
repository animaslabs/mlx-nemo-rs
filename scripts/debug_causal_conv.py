#!/usr/bin/env python3
"""
Debug CausalConv2D to understand how it handles padding.
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
preprocessor = pipeline.asr_model.asr_model.preprocessor
encoder = pipeline.asr_model.asr_model.encoder

chunk = audio[:19360]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    mel_transposed = mel.transpose(1, 2)

subsampling = encoder.pre_encode

# Get the CausalConv2D forward source
causal_conv = subsampling.conv[0]
print(f"CausalConv2D type: {type(causal_conv)}")
print(f"\n=== CausalConv2D forward source ===")
forward_src = inspect.getsource(causal_conv.forward)
print(forward_src)

# Check if the Sequential-like conv has a forward method that handles lengths
print(f"\n=== subsampling.conv type: {type(subsampling.conv)} ===")
if hasattr(subsampling.conv, 'forward'):
    conv_forward = inspect.getsource(subsampling.conv.forward)
    print(conv_forward)

# Check the CausalConv2D class
print(f"\n=== CausalConv2D class definition ===")
class_src = inspect.getsource(type(causal_conv))
# Print first 3000 chars
print(class_src[:3000])
