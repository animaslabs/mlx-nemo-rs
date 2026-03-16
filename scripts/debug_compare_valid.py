#!/usr/bin/env python3
"""Compare valid frames only."""

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
audio_path = sys.argv[1] if len(sys.argv) > 1 else '../samples/2086-149220-0033.wav'
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
pre_encode = pipeline.asr_model.asr_model.encoder.pre_encode

# Process first chunk
chunk = audio[:19360]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    mel_transposed = mel.transpose(1, 2)  # [B, F, T] -> [B, T, F]
    subsampled, sub_len = pre_encode(mel_transposed, mel_len)

print(f"MEL: shape={list(mel.shape)}, mel_len={mel_len.item()}")
print(f"Subsampled: shape={list(subsampled.shape)}, sub_len={sub_len.item()}")
print()

# Print values at specific positions (only within valid range)
out_np = subsampled[0].numpy()  # [T, D]
valid_t = sub_len.item()

print(f"=== Valid output (time steps 0 to {valid_t-1}) ===")
for t in range(valid_t):
    vals = out_np[t, :5]
    print(f"T{t:02d} [0:5]: [{', '.join(f'{v:.6f}' for v in vals)}]")
