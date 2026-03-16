#!/usr/bin/env python3
"""
Debug encoder layer by layer to find divergence.
"""

import os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

# Load audio
audio_path = '../samples/2086-149220-0033.wav'
audio, sr = sf.read(audio_path, dtype='float32')
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Build pipeline
config_path = Path(__file__).parent / 'streaming_config.yaml'
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

# Process first chunk
chunk = audio[:19360]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

print(f"MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")

# Step through encoder manually
# 1. Subsampling
with torch.no_grad():
    # NeMo encoder expects [B, F, T] input
    # pre_encode expects [B, T, F] input
    mel_transposed = mel.transpose(1, 2)  # [B, T, F]
    x, lengths = encoder.pre_encode(mel_transposed, mel_len)

print(f"\n1. After subsampling:")
print(f"   Shape: {list(x.shape)}, lengths: {lengths.item()}")
x_np = x[0].numpy()
print(f"   T00[0:5]: [{', '.join(f'{v:.6f}' for v in x_np[0, :5])}]")
print(f"   T01[0:5]: [{', '.join(f'{v:.6f}' for v in x_np[1, :5])}]")
print(f"   Stats: min={x_np.min():.6f}, max={x_np.max():.6f}, mean={x_np.mean():.6f}")

# 2. Positional encoding
with torch.no_grad():
    pos_emb = encoder.pos_enc(x)

print(f"\n2. Positional encoding:")
print(f"   Shape: {list(pos_emb.shape)}")
pe_np = pos_emb[0].numpy()
print(f"   PE[0][0:5]: [{', '.join(f'{v:.6f}' for v in pe_np[0, :5])}]")
print(f"   PE[1][0:5]: [{', '.join(f'{v:.6f}' for v in pe_np[1, :5])}]")
print(f"   Stats: min={pe_np.min():.6f}, max={pe_np.max():.6f}, mean={pe_np.mean():.6f}")

# 3. After first conformer block
# Create attention mask
pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
print(f"\n3. Pad mask shape: {list(pad_mask.shape)}, sum: {pad_mask.sum().item()}")

with torch.no_grad():
    # Run first layer
    x_after_layer0 = encoder.layers[0](
        x=x,
        att_mask=None,
        pos_emb=pos_emb,
        pad_mask=pad_mask,
        cache_last_channel=None,
        cache_last_time=None,
    )
    if isinstance(x_after_layer0, tuple):
        x_after_layer0 = x_after_layer0[0]

print(f"\n4. After layer 0:")
print(f"   Shape: {list(x_after_layer0.shape)}")
l0_np = x_after_layer0[0].numpy()
print(f"   T00[0:5]: [{', '.join(f'{v:.6f}' for v in l0_np[0, :5])}]")
print(f"   T01[0:5]: [{', '.join(f'{v:.6f}' for v in l0_np[1, :5])}]")
print(f"   Stats: min={l0_np.min():.6f}, max={l0_np.max():.6f}, mean={l0_np.mean():.6f}")

# Continue through more layers
with torch.no_grad():
    x_curr = x_after_layer0
    for i in range(1, min(5, len(encoder.layers))):
        x_curr = encoder.layers[i](
            x=x_curr,
            att_mask=None,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
            cache_last_channel=None,
            cache_last_time=None,
        )
        if isinstance(x_curr, tuple):
            x_curr = x_curr[0]

print(f"\n5. After layer 4:")
l4_np = x_curr[0].numpy()
print(f"   T00[0:5]: [{', '.join(f'{v:.6f}' for v in l4_np[0, :5])}]")
print(f"   Stats: min={l4_np.min():.6f}, max={l4_np.max():.6f}, mean={l4_np.mean():.6f}")
