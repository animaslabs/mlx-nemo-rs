#!/usr/bin/env python3
"""Trace conv cache initialization and chunk 0 update."""

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

    print("=== Initial Conv Cache (cache_last_time) ===")
    print(f"Shape: {list(cache_last_time.shape)}")
    print(f"Layer 0 cache shape: {list(cache_last_time[0].shape)}")
    print(f"Layer 0 cache[0, 0, :]: {cache_last_time[0, 0, 0, :].tolist()}")
    print(f"Layer 0 cache norm: {torch.norm(cache_last_time[0]).item():.6f}")

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
    if len(audio_chunk) < audio_chunk_samples:
        audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

    chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(audio_chunk)])

    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

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

    print("\n=== After Chunk 0: Conv Cache (cache_last_time) ===")
    print(f"Shape: {list(cache_last_time.shape)}")
    print(f"Layer 0 cache shape: {list(cache_last_time[0].shape)}")
    print(f"Layer 0 cache[0, 0, :]: {cache_last_time[0, 0, 0, :].tolist()}")
    print(f"Layer 0 cache[0, :10, 0]: {cache_last_time[0, 0, :10, 0].tolist()}")
    print(f"Layer 0 cache[0, :10, -1]: {cache_last_time[0, 0, :10, -1].tolist()}")
    print(f"Layer 0 cache norm: {torch.norm(cache_last_time[0]).item():.6f}")


if __name__ == '__main__':
    main()
