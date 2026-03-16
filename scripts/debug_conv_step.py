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
import torch.nn as nn
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

    x_conv_normed = layer0.norm_conv(x_after_attn)
    print(f"Input to conv (after conv_norm) T00[:5]: {x_conv_normed[0, 0, :5].tolist()}")

    # Now trace through conv module manually
    conv = layer0.conv

    # Step 1: Transpose to [B, D, T]
    x = x_conv_normed.transpose(1, 2)
    print(f"After transpose to [B, D, T], shape: {list(x.shape)}")
    print(f"  x[0, :5, 0] (first 5 channels, T=0): {x[0, :5, 0].tolist()}")

    # Step 2: Pointwise conv1
    x = conv.pointwise_conv1(x)
    print(f"\nAfter pointwise_conv1, shape: {list(x.shape)}")
    print(f"  x[0, :5, 0] (first 5 channels, T=0): {x[0, :5, 0].tolist()}")

    # Step 3: GLU
    x = nn.functional.glu(x, dim=1)
    print(f"\nAfter GLU, shape: {list(x.shape)}")
    print(f"  x[0, :5, 0] (first 5 channels, T=0): {x[0, :5, 0].tolist()}")

    # Step 4: Mask (if pad_mask provided)
    # NeMo convention: pad_mask True = padding position to mask out
    # My pad_mask: True = valid, so need to invert
    nemo_pad_mask = ~pad_mask  # Invert: now True means padding
    if nemo_pad_mask is not None:
        print(f"\nApplying NeMo pad_mask (True=padding): {nemo_pad_mask[0].tolist()}")
        x = x.masked_fill(nemo_pad_mask.unsqueeze(1), 0.0)
        print(f"After masking, x[0, :5, 0]: {x[0, :5, 0].tolist()}")
        print(f"After masking, x[0, :5, 16]: {x[0, :5, 16].tolist()}")  # T=16 should be masked

    # Step 5: Depthwise conv (CausalConv1D)
    print(f"\nDepthwise conv module: {conv.depthwise_conv}")
    x = conv.depthwise_conv(x, cache=None)
    print(f"After depthwise_conv, shape: {list(x.shape)}")
    print(f"  x[0, :5, 0] (first 5 channels, T=0): {x[0, :5, 0].tolist()}")

    # Step 6: Batch norm (which is actually LayerNorm)
    print(f"\nNorm type: {type(conv.batch_norm)}")
    x = x.transpose(1, 2)  # [B, T, D] for LayerNorm
    x = conv.batch_norm(x)
    x = x.transpose(1, 2)  # Back to [B, D, T]
    print(f"After norm, shape: {list(x.shape)}")
    print(f"  x[0, :5, 0]: {x[0, :5, 0].tolist()}")

    # Step 7: Activation (Swish)
    x = conv.activation(x)
    print(f"\nAfter activation (Swish), x[0, :5, 0]: {x[0, :5, 0].tolist()}")

    # Step 8: Pointwise conv2
    x = conv.pointwise_conv2(x)
    print(f"\nAfter pointwise_conv2, shape: {list(x.shape)}")
    print(f"  x[0, :5, 0]: {x[0, :5, 0].tolist()}")

    # Step 9: Transpose back to [B, T, D]
    x = x.transpose(1, 2)
    print(f"\nFinal output (after transpose), shape: {list(x.shape)}")
    print(f"  x[0, 0, :5]: {x[0, 0, :5].tolist()}")

    # Compare with actual conv output - need to use NeMo convention (True=padding)
    actual_out = conv(x_conv_normed, pad_mask=nemo_pad_mask)
    print(f"\nActual conv output T00[:5] (with NeMo pad_mask): {actual_out[0, 0, :5].tolist()}")

    # Also try without pad_mask to see if that's the issue
    actual_out_no_mask = conv(x_conv_normed, pad_mask=None)
    print(f"Actual conv output T00[:5] (no pad_mask): {actual_out_no_mask[0, 0, :5].tolist()}")
