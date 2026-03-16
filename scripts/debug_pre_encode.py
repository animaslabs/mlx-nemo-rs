#!/usr/bin/env python3
"""
Debug pre_encode output at all positions.
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
    # Check last few MEL frames
    print(f"MEL frames 119-121 (last 3):")
    for i in range(119, min(122, mel.shape[2])):
        vals = mel[0, :5, i].tolist()
        valid = "VALID" if i < mel_len.item() else "PADDING"
        print(f"  Frame {i} [{valid}]: [{', '.join(f'{v:.6f}' for v in vals)}]")

    mel_transposed = mel.transpose(1, 2)
    x, lengths = encoder.pre_encode(mel_transposed, mel_len)

    print(f"\nAfter pre_encode:")
    print(f"  x shape: {list(x.shape)}, lengths: {lengths.item()}")

    x_np = x[0].numpy()
    print(f"\n  Stats: min={x_np.min():.6f}, max={x_np.max():.6f}, mean={x_np.mean():.6f}")

    # Print all positions
    print(f"\n  All time steps:")
    for t in range(x.shape[1]):
        vals = x[0, t, :5].tolist()
        valid = "VALID" if t < lengths.item() else "PADDING"
        print(f"  T{t:02d} [{valid:7s}]: [{', '.join(f'{v:.6f}' for v in vals)}]")
