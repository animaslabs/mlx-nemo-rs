#!/usr/bin/env python3
"""
Debug ConvolutionModule step by step.
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
    x_pe, pos_emb = encoder.pos_enc(x)

# Create pad_mask
pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

# Get layer 0
layer0 = encoder.layers[0]

# Trace to get input to conv
with torch.no_grad():
    residual = x.clone()
    x_normed = layer0.norm_feed_forward1(x)
    ff1_out = layer0.feed_forward1(x_normed)
    x_after_ff1 = residual + 0.5 * ff1_out

    residual = x_after_ff1.clone()
    x_attn_normed = layer0.norm_self_att(x_after_ff1)
    attn_out = layer0.self_attn(query=x_attn_normed, key=x_attn_normed, value=x_attn_normed, pos_emb=pos_emb, mask=None)
    x_after_attn = residual + attn_out

    # Now trace through conv
    print(f"Input to conv (after attn+residual) T00[:5]: {x_after_attn[0, 0, :5].tolist()}")

    x_conv_normed = layer0.norm_conv(x_after_attn)
    print(f"After norm_conv T00[:5]: {x_conv_normed[0, 0, :5].tolist()}")

    # Check the conv module structure
    conv = layer0.conv
    print(f"\nConv module type: {type(conv).__name__}")
    print(f"Conv module children:")
    for name, child in conv.named_children():
        print(f"  {name}: {type(child).__name__}")

    # Check conv module attributes
    if hasattr(conv, 'pointwise_conv1'):
        print(f"\nPointwise conv1: {conv.pointwise_conv1}")
    if hasattr(conv, 'depthwise_conv'):
        print(f"Depthwise conv: {conv.depthwise_conv}")
    if hasattr(conv, 'pointwise_conv2'):
        print(f"Pointwise conv2: {conv.pointwise_conv2}")

    # Trace through conv module manually
    print(f"\n=== Tracing through conv module ===")

    # Check if it has a forward method we can trace
    import inspect
    forward_src = inspect.getsource(conv.forward)
    print(f"Conv forward method (first 1500 chars):\n{forward_src[:1500]}")
