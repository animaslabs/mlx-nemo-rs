#!/usr/bin/env python3
"""Trace GLU output for padding frame."""

import os
import warnings
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


def main():
    audio_path = '../samples/2086-149220-0033.wav'

    config_path = Path('streaming_config.yaml')
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
    cfg.asr.device = 'cpu'
    cfg.asr.compute_dtype = 'float32'
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 1]
    cfg.asr_decoding_type = 'rnnt'
    cfg.log_level = 40

    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    pipeline = PipelineBuilder.build_pipeline(cfg)

    asr_model = pipeline.asr_model.asr_model
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder

    streaming_cfg = encoder.streaming_cfg
    if streaming_cfg is None:
        encoder.setup_streaming_params()
        streaming_cfg = encoder.streaming_cfg

    audio, sr = sf.read(audio_path, dtype='float32')

    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]

    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        hop_length = int(16000 * 0.01)

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    # Process chunk 0
    audio_chunk = audio[0:audio_chunk_samples]
    chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(audio_chunk)])

    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

    # Patch the conv module to trace values
    layer0 = encoder.layers[0]
    conv = layer0.conv
    original_forward = conv.forward

    def traced_forward(x, pad_mask=None, cache=None):
        print(f"=== Conv Module Forward (Chunk 0) ===")
        print(f"Input x shape: {list(x.shape)}")  # [B, T, D]

        x_t = x.transpose(1, 2)  # [B, D, T]
        print(f"After transpose: {list(x_t.shape)}")

        # Pointwise conv 1
        x_pw1 = conv.pointwise_conv1(x_t)
        print(f"After pointwise1: {list(x_pw1.shape)}")
        print(f"  Last frame (T=-1), first 5 channels: {x_pw1[0, :5, -1].tolist()}")

        # GLU
        x_glu = torch.nn.functional.glu(x_pw1, dim=1)
        print(f"After GLU: {list(x_glu.shape)}")
        print(f"  Last frame (T=-1), first 5 channels: {x_glu[0, :5, -1].tolist()}")
        print(f"  Last frame norm: {torch.norm(x_glu[0, :, -1]).item():.6f}")

        # Now continue with original forward
        return original_forward(x, pad_mask, cache)

    conv.forward = traced_forward

    with torch.no_grad():
        (encoded, encoded_len,
         cache_last_channel, cache_last_time,
         cache_last_channel_len) = encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_len,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=0,
        )


if __name__ == '__main__':
    main()
