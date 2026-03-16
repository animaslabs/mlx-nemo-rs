#!/usr/bin/env python3
"""
Debug attention layer step by step.
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
    print(f"MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
    mel_transposed = mel.transpose(1, 2)
    x, lengths = encoder.pre_encode(mel_transposed, mel_len)
    print(f"After pre_encode: x shape: {list(x.shape)}, lengths: {lengths.item()}")
    x_pe, pos_emb = encoder.pos_enc(x)

print("Input to layer 0:")
print(f"  x shape: {list(x.shape)}")
print(f"  x[0, 0, :5]: {x[0, 0, :5].tolist()}")
print(f"  x[0, 16, :5]: {x[0, 16, :5].tolist()}")  # Last position
print(f"  pos_emb shape: {list(pos_emb.shape)}")

pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

# Get layer 0
layer0 = encoder.layers[0]
self_attn = layer0.self_attn

# Print attention module info
print(f"\nSelf-attention module: {type(self_attn).__name__}")
print(f"  n_heads (h): {self_attn.h}")
print(f"  head_dim (d_k): {self_attn.d_k}")

# Step through attention manually
with torch.no_grad():
    # FF1 first (Macaron style)
    residual = x.clone()

    if hasattr(layer0, 'norm_feed_forward1'):
        x_normed = layer0.norm_feed_forward1(x)
        ff1_out = layer0.feed_forward1(x_normed)
        x_after_ff1 = residual + 0.5 * ff1_out
        print(f"\nAfter FF1 + half residual:")
        print(f"  T00[:5]: {x_after_ff1[0, 0, :5].tolist()}")
        print(f"  T16[:5]: {x_after_ff1[0, 16, :5].tolist()}")
    else:
        x_after_ff1 = x

    # Self-attention norm
    if hasattr(layer0, 'norm_self_att'):
        x_attn_input = layer0.norm_self_att(x_after_ff1)
    else:
        x_attn_input = x_after_ff1

    print(f"\nInput to self-attention (after norm):")
    print(f"  shape: {list(x_attn_input.shape)}")
    print(f"  T00[:5]: {x_attn_input[0, 0, :5].tolist()}")
    print(f"  T16[:5]: {x_attn_input[0, 16, :5].tolist()}")

    # Q, K, V projections
    q = self_attn.linear_q(x_attn_input)
    k = self_attn.linear_k(x_attn_input)
    v = self_attn.linear_v(x_attn_input)

    print(f"\nQ, K, V projections:")
    print(f"  Q[0, 0, :5]: {q[0, 0, :5].tolist()}")
    print(f"  Q[0, 1, :5]: {q[0, 1, :5].tolist()}")
    print(f"  K[0, 0, :5]: {k[0, 0, :5].tolist()}")
    print(f"  K[0, 16, :5]: {k[0, 16, :5].tolist()}")  # Last position
    print(f"  V[0, 0, :5]: {v[0, 0, :5].tolist()}")

    # Position biases
    print(f"\nPosition biases:")
    print(f"  pos_bias_u shape: {list(self_attn.pos_bias_u.shape)}")
    print(f"  pos_bias_u[0, :5]: {self_attn.pos_bias_u[0, :5].tolist()}")
    print(f"  pos_bias_v[0, :5]: {self_attn.pos_bias_v[0, :5].tolist()}")

    # Reshape Q, K, V to multi-head
    B, T, D = q.shape
    n_heads = self_attn.h  # NeMo calls it 'h'
    head_dim = self_attn.d_k  # NeMo calls it 'd_k'

    q_heads = q.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, T, D/H]
    k_heads = k.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)
    v_heads = v.view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)

    # Position encoding projection
    p = self_attn.linear_pos(pos_emb)
    p_heads = p.view(1, -1, n_heads, head_dim).permute(0, 2, 1, 3)

    # Add position biases
    pos_bias_u = self_attn.pos_bias_u.view(1, n_heads, 1, head_dim)
    pos_bias_v = self_attn.pos_bias_v.view(1, n_heads, 1, head_dim)

    q_with_bias_u = q_heads + pos_bias_u
    q_with_bias_v = q_heads + pos_bias_v

    # Content attention: (Q + u) @ K^T
    matrix_ac = torch.matmul(q_with_bias_u, k_heads.transpose(-2, -1))

    # Position attention: (Q + v) @ P^T
    matrix_bd = torch.matmul(q_with_bias_v, p_heads.transpose(-2, -1))

    print(f"\nBefore rel_shift:")
    print(f"  matrix_ac shape: {list(matrix_ac.shape)}")
    print(f"  matrix_bd shape: {list(matrix_bd.shape)}")
    print(f"  matrix_ac[0,0,0,:5]: {matrix_ac[0, 0, 0, :5].tolist()}")
    print(f"  matrix_bd[0,0,0,:5]: {matrix_bd[0, 0, 0, :5].tolist()}")
    print(f"  matrix_bd[0,0,0,-5:]: {matrix_bd[0, 0, 0, -5:].tolist()}")

    # NeMo's rel_shift (pad and reshape)
    def rel_shift(x):
        """NeMo's rel_shift implementation."""
        B, H, T, P = x.shape
        # Pad left with zeros
        x_padded = torch.nn.functional.pad(x, (1, 0))  # [B, H, T, P+1]
        # Reshape and slice
        x_reshaped = x_padded.view(B, H, P + 1, T)
        x_sliced = x_reshaped[:, :, 1:, :]  # Skip first row
        x_shifted = x_sliced.view(B, H, T, P)
        return x_shifted

    matrix_bd_shifted = rel_shift(matrix_bd)
    print(f"\nAfter rel_shift (before truncation):")
    print(f"  matrix_bd_shifted shape: {list(matrix_bd_shifted.shape)}")
    print(f"  matrix_bd_shifted[0,0,0,:5]: {matrix_bd_shifted[0, 0, 0, :5].tolist()}")
    print(f"  matrix_bd_shifted[0,0,0,12:17]: {matrix_bd_shifted[0, 0, 0, 12:17].tolist()}")

    matrix_bd_truncated = matrix_bd_shifted[:, :, :, :T]

    print(f"\nAfter truncation:")
    print(f"  matrix_bd shape: {list(matrix_bd_truncated.shape)}")
    print(f"  matrix_ac[0,0,0,:5]: {matrix_ac[0, 0, 0, :5].tolist()}")
    print(f"  matrix_bd[0,0,0,:5]: {matrix_bd_truncated[0, 0, 0, :5].tolist()}")
    print(f"  matrix_ac[0,0,0,12:17]: {matrix_ac[0, 0, 0, 12:17].tolist()}")
    print(f"  matrix_bd[0,0,0,12:17]: {matrix_bd_truncated[0, 0, 0, 12:17].tolist()}")

    # Combine and scale
    scale = head_dim ** 0.5
    scores = (matrix_ac + matrix_bd_truncated) / scale

    print(f"\nScaled scores:")
    print(f"  scores shape: {list(scores.shape)}")
    print(f"  scores[0,0,0,:5]: {scores[0, 0, 0, :5].tolist()}")
    print(f"  scores[0,0,0,-5:]: {scores[0, 0, 0, -5:].tolist()}")
    print(f"  scale: {scale}")

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    print(f"\nAttention weights:")
    print(f"  attn_weights[0,0,0,:5]: {attn_weights[0, 0, 0, :5].tolist()}")

    # Apply attention
    context = torch.matmul(attn_weights, v_heads)  # [B, H, T, D/H]
    context = context.permute(0, 2, 1, 3).contiguous().view(B, T, D)

    # Output projection
    output = self_attn.linear_out(context)

    print(f"\nSelf-attention output:")
    print(f"  output[0, 0, :5]: {output[0, 0, :5].tolist()}")
    print(f"  output[0, 1, :5]: {output[0, 1, :5].tolist()}")
