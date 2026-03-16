#!/usr/bin/env python3
"""
Debug the actual pad_mask values that NeMo uses.
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

    # Check the _create_masks function
    print(f"\n=== Checking _create_masks ===")
    create_masks_src = inspect.getsource(encoder._create_masks)
    print(create_masks_src[:2000])

    # Run through pre_encode and pos_enc
    mel_transposed = mel.transpose(1, 2)
    x, lengths = encoder.pre_encode(mel_transposed, mel_len)
    print(f"\nAfter pre_encode: x.shape={x.shape}, lengths={lengths}")

    x_pe, pos_emb = encoder.pos_enc(x=x, cache_len=0)

    # Create masks like the encoder does
    # For batch mode (no streaming), padding_length should be lengths
    pad_mask, att_mask = encoder._create_masks(
        att_context_size=encoder.att_context_size,
        padding_length=lengths,
        max_audio_length=x.shape[1],
        offset=torch.tensor([0]),
        device=x.device,
    )
    print(f"\npad_mask from _create_masks: {pad_mask}")
    print(f"pad_mask shape: {pad_mask.shape}")
    print(f"pad_mask dtype: {pad_mask.dtype}")
    print(f"pad_mask[0]: {pad_mask[0].tolist()}")

    # Compare with my pad_mask
    my_pad_mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    print(f"\nMy pad_mask: {my_pad_mask[0].tolist()}")

    print(f"\nConvention check:")
    print(f"  NeMo pad_mask True at position 0: {pad_mask[0, 0].item()}")
    print(f"  NeMo pad_mask True at position 15 (last valid): {pad_mask[0, 15].item()}")
    print(f"  NeMo pad_mask True at position 16 (padding): {pad_mask[0, 16].item()}")
