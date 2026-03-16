#!/usr/bin/env python3
"""
Debug what pad_mask the encoder actually passes to layers.
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
    print(f"mel.shape: {mel.shape}, mel_len: {mel_len}")

    # Run full encoder and capture what pad_mask it creates internally
    # Patch the layer to capture pad_mask
    original_forward = encoder.layers[0].forward

    captured_pad_mask = [None]

    def patched_forward(x, att_mask, pos_emb, pad_mask, **kwargs):
        captured_pad_mask[0] = pad_mask
        return original_forward(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask, **kwargs)

    encoder.layers[0].forward = patched_forward

    # Run encoder
    encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)

    # Restore original forward
    encoder.layers[0].forward = original_forward

    print(f"\nCaptured pad_mask from encoder:")
    print(f"  pad_mask: {captured_pad_mask[0]}")
    if captured_pad_mask[0] is not None:
        print(f"  pad_mask shape: {captured_pad_mask[0].shape}")
        print(f"  pad_mask[0]: {captured_pad_mask[0][0].tolist()}")
    else:
        print("  pad_mask is None!")

    print(f"\nEncoder output:")
    encoded_t = encoded.transpose(1, 2)
    print(f"  T00[:5]: {encoded_t[0, 0, :5].tolist()}")
