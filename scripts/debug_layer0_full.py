#!/usr/bin/env python3
"""
Debug ConformerLayer 0 step by step through all components.
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

print(f"Input to layer 0:")
print(f"  x shape: {list(x.shape)}, lengths: {lengths.item()}")
print(f"  x[0, 0, :5]: {x[0, 0, :5].tolist()}")

# Create pad_mask - NeMo convention: True = padding position to mask out
# My convention: True = valid
my_pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
pad_mask = ~my_pad_mask  # NeMo convention: True = padding
att_mask = None
print(f"pad_mask (NeMo convention, True=padding): {pad_mask[0].tolist()}")

# Get layer 0
layer0 = encoder.layers[0]
print(f"\nLayer 0 type: {type(layer0).__name__}")

# Trace through layer 0 manually
with torch.no_grad():
    residual = x.clone()

    # FF1 (half-step residual)
    print(f"\n=== FF1 ===")
    x_normed = layer0.norm_feed_forward1(x)
    print(f"After norm_feed_forward1 T00[:5]: {x_normed[0, 0, :5].tolist()}")

    ff1_out = layer0.feed_forward1(x_normed)
    print(f"After feed_forward1 T00[:5]: {ff1_out[0, 0, :5].tolist()}")

    x_after_ff1 = residual + 0.5 * ff1_out
    print(f"After FF1+residual T00[:5]: {x_after_ff1[0, 0, :5].tolist()}")
    print(f"After FF1+residual T16[:5]: {x_after_ff1[0, 16, :5].tolist()}")

    # Self-attention
    print(f"\n=== Self-Attention ===")
    residual = x_after_ff1.clone()
    x_attn_normed = layer0.norm_self_att(x_after_ff1)
    print(f"After norm_self_att T00[:5]: {x_attn_normed[0, 0, :5].tolist()}")
    print(f"After norm_self_att T16[:5]: {x_attn_normed[0, 16, :5].tolist()}")

    # Self attention - check the signature
    import inspect
    sig = inspect.signature(layer0.self_attn.forward)
    print(f"self_attn.forward signature: {sig}")

    # Try calling with correct args
    attn_out = layer0.self_attn(query=x_attn_normed, key=x_attn_normed, value=x_attn_normed, pos_emb=pos_emb, mask=att_mask)
    print(f"After self_attn T00[:5]: {attn_out[0, 0, :5].tolist()}")

    x_after_attn = residual + attn_out
    print(f"After self_attn+residual T00[:5]: {x_after_attn[0, 0, :5].tolist()}")

    # Convolution
    print(f"\n=== Convolution ===")
    residual = x_after_attn.clone()
    x_conv_normed = layer0.norm_conv(x_after_attn)
    print(f"After norm_conv T00[:5]: {x_conv_normed[0, 0, :5].tolist()}")

    conv_out = layer0.conv(x_conv_normed, pad_mask=pad_mask)
    print(f"After conv T00[:5]: {conv_out[0, 0, :5].tolist()}")

    x_after_conv = residual + conv_out
    print(f"After conv+residual T00[:5]: {x_after_conv[0, 0, :5].tolist()}")

    # FF2 (half-step residual)
    print(f"\n=== FF2 ===")
    residual = x_after_conv.clone()
    x_ff2_normed = layer0.norm_feed_forward2(x_after_conv)
    print(f"After norm_feed_forward2 T00[:5]: {x_ff2_normed[0, 0, :5].tolist()}")

    ff2_out = layer0.feed_forward2(x_ff2_normed)
    print(f"After feed_forward2 T00[:5]: {ff2_out[0, 0, :5].tolist()}")

    x_after_ff2 = residual + 0.5 * ff2_out
    print(f"After FF2+residual T00[:5]: {x_after_ff2[0, 0, :5].tolist()}")

    # Final norm
    print(f"\n=== Final Norm ===")
    x_final = layer0.norm_out(x_after_ff2)
    print(f"After norm_out T00[:5]: {x_final[0, 0, :5].tolist()}")

    print(f"\n=== Layer 0 output ===")
    # Now run the actual layer forward to compare
    out0 = layer0(x=x_pe, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
    print(f"Actual layer output T00[:5]: {out0[0, 0, :5].tolist()}")
