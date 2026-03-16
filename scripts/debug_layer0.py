#!/usr/bin/env python3
"""
Debug ConformerLayer 0 step by step.
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

pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

print("Input to layer 0:")
print(f"  x shape: {x.shape}")
print(f"  x[0, 0, :5]: {x[0, 0, :5].tolist()}")
print(f"  x[0, 1, :5]: {x[0, 1, :5].tolist()}")
print(f"  pos_emb shape: {pos_emb.shape}")
print(f"  pad_mask: {pad_mask[0].tolist()}")

# Get layer 0
layer0 = encoder.layers[0]
print(f"\nLayer 0 type: {type(layer0).__name__}")
print(f"Layer 0 components:")
for name, child in layer0.named_children():
    print(f"  {name}: {type(child).__name__}")

# Step through layer 0 manually
# NeMo ConformerLayer has: ff1_norm, ff1, self_attn_norm, self_attn, conv_norm, conv, ff2_norm, ff2, norm_out
# But this model uses Macaron-style: norm_feed_forward1, feed_forward1, norm_self_att, self_att, ...

# Check what attributes exist
print(f"\nLayer 0 attributes:")
attrs = [a for a in dir(layer0) if not a.startswith('_')]
ff_attrs = [a for a in attrs if 'ff' in a.lower() or 'feed' in a.lower() or 'norm' in a.lower()]
print(f"  FF/norm attrs: {ff_attrs}")

# Try to trace through
with torch.no_grad():
    residual = x.clone()

    # Check for first normalization
    if hasattr(layer0, 'norm_feed_forward1'):
        x_normed = layer0.norm_feed_forward1(x)
        print(f"\nAfter norm_feed_forward1:")
        print(f"  shape: {x_normed.shape}")
        print(f"  T00[0:5]: {x_normed[0, 0, :5].tolist()}")

        if hasattr(layer0, 'feed_forward1'):
            ff1_out = layer0.feed_forward1(x_normed)
            print(f"\nAfter feed_forward1:")
            print(f"  shape: {ff1_out.shape}")
            print(f"  T00[0:5]: {ff1_out[0, 0, :5].tolist()}")

            # Half residual
            x_after_ff1 = residual + 0.5 * ff1_out
            print(f"\nAfter FF1 + half residual:")
            print(f"  T00[0:5]: {x_after_ff1[0, 0, :5].tolist()}")

    elif hasattr(layer0, 'norm_feed_forward'):
        # Different naming convention
        x_normed = layer0.norm_feed_forward(x)
        print(f"\nAfter norm_feed_forward:")
        print(f"  shape: {x_normed.shape}")
        print(f"  T00[0:5]: {x_normed[0, 0, :5].tolist()}")
