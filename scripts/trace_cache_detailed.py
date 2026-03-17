#!/usr/bin/env python3
"""Trace cache values in detail through NeMo encoder."""

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

    # Get initial cache
    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    # Streaming parameters
    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]

    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        hop_length = int(16000 * 0.01)

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    # Patch the attention to trace cache
    original_update_cache = encoder.layers[0].self_attn.update_cache.__func__

    def traced_update_cache(self, key, value, query, cache):
        nonlocal chunk_idx
        print(f"\n    === Layer 0 update_cache (chunk {chunk_idx}) ===")
        print(f"    query shape: {list(query.shape)}")
        if cache is not None:
            print(f"    cache shape: {list(cache.shape)}")
            cache_len = cache.shape[1]
            q_keep_size = query.shape[1] - self.cache_drop_size
            print(f"    cache_drop_size: {self.cache_drop_size}")
            print(f"    q_keep_size: {q_keep_size}")

            # Before concatenation - trace values
            print(f"    cache[:, 0, :10]: {cache[0, 0, :10].tolist()[:5]}...")
            print(f"    cache[:, -1, :10]: {cache[0, -1, :10].tolist()[:5]}...")
            print(f"    query[:, 0, :10]: {query[0, 0, :10].tolist()[:5]}...")
            print(f"    query[:, -1, :10]: {query[0, -1, :10].tolist()[:5]}...")

        result = original_update_cache(self, key, value, query, cache)
        key, value, query, new_cache = result

        if new_cache is not None:
            print(f"    new_cache shape: {list(new_cache.shape)}")
            print(f"    new_cache[:, 0, :10]: {new_cache[0, 0, :10].tolist()[:5]}...")
            print(f"    new_cache[:, -1, :10]: {new_cache[0, -1, :10].tolist()[:5]}...")

        return result

    # Patch
    encoder.layers[0].self_attn.update_cache = traced_update_cache.__get__(
        encoder.layers[0].self_attn, type(encoder.layers[0].self_attn)
    )

    print("=== Detailed Cache Trace ===")
    print(f"cache_last_channel shape: {list(cache_last_channel.shape)}")
    print()

    audio_pos = 0
    max_chunks = 3

    with torch.no_grad():
        for chunk_idx in range(max_chunks):
            if audio_pos >= len(audio):
                break

            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_idx}")
            print(f"{'='*60}")
            print(f"mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
            print(f"Input cache_last_channel_len: {cache_last_channel_len.item()}")

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            (encoded, encoded_len,
             cache_last_channel, cache_last_time,
             cache_last_channel_len) = encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=True,
                drop_extra_pre_encoded=drop_extra,
            )

            print(f"\nOutput:")
            print(f"  encoded shape: {list(encoded.shape)}, len: {encoded_len.item()}")
            print(f"  cache_last_channel_len: {cache_last_channel_len.item()}")
            print(f"  encoder frame 0 norm: {torch.norm(encoded[0, 0]).item():.6f}")
            print(f"  encoder frame 0 [:10]: {encoded[0, 0, :10].tolist()}")

            audio_pos = end_pos


if __name__ == '__main__':
    main()
