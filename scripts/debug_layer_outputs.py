#!/usr/bin/env python3
"""
Debug layer-by-layer encoder outputs.
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
    print(f"After pre_encode: x shape: {list(x.shape)}, lengths: {lengths.item()}")
    print(f"  T00[0:5]: {x[0, 0, :5].tolist()}")
    print(f"  T01[0:5]: {x[0, 1, :5].tolist()}")
    print(f"  T16[0:5]: {x[0, 16, :5].tolist()}")

    # Position encoding
    x_pe, pos_emb = encoder.pos_enc(x)
    print(f"\nAfter pos_enc: x_pe shape: {list(x_pe.shape)}")
    print(f"  T00[0:5]: {x_pe[0, 0, :5].tolist()}")
    print(f"  T01[0:5]: {x_pe[0, 1, :5].tolist()}")
    print(f"  pos_emb shape: {list(pos_emb.shape)}")
    print(f"  pos_emb[0, 0, :5]: {pos_emb[0, 0, :5].tolist()}")
    print(f"  pos_emb[0, 1, :5]: {pos_emb[0, 1, :5].tolist()}")

    # Create pad_mask
    max_len = x.shape[1]
    pad_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    att_mask = None  # For batch processing, no attention mask

    # Run through layer 0
    layer0 = encoder.layers[0]

    # NeMo's ConformerLayer forward needs: x, att_mask, pos_emb, pad_mask
    # But the actual signature might be different - check what it expects
    print(f"\nLayer 0 type: {type(layer0)}")

    # Try to call layer forward
    try:
        out0 = layer0(x=x_pe, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        print(f"\nAfter layer 0: shape: {list(out0.shape)}")
        print(f"  T00[0:5]: {out0[0, 0, :5].tolist()}")
        print(f"  T01[0:5]: {out0[0, 1, :5].tolist()}")
        print(f"  T16[0:5]: {out0[0, 16, :5].tolist()}")
    except Exception as e:
        print(f"Error calling layer 0: {e}")
        import traceback
        traceback.print_exc()

    # Run through layer 1
    if 'out0' in locals():
        out1 = layer0(x=out0, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        print(f"\nAfter layer 1: shape: {list(out1.shape)}")
        print(f"  T00[0:5]: {out1[0, 0, :5].tolist()}")
        print(f"  T01[0:5]: {out1[0, 1, :5].tolist()}")
