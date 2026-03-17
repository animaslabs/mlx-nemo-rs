#!/usr/bin/env python3
"""Trace encoder for specific chunks to compare with Rust."""

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
    target_chunks = list(range(20))  # Trace first 20 chunks for cache comparison

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

    print(f"Streaming config:")
    print(f"  pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"  model_chunk_size: {model_chunk_size}")
    print(f"  buffer_mel_frames: {buffer_mel_frames}")
    print(f"  audio_chunk_samples: {audio_chunk_samples}")
    print(f"  valid_out_len: {streaming_cfg.valid_out_len}")
    print(f"  drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print()

    audio_pos = 0
    chunk_idx = 0

    with torch.no_grad():
        while audio_pos < len(audio):
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            # Get cache info before processing
            if cache_last_channel is not None:
                cache_shape = [list(c.shape) if c is not None else None for c in cache_last_channel]
                filled_lens = cache_last_channel_len.tolist() if cache_last_channel_len is not None else None
            else:
                cache_shape = None
                filled_lens = None

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

            if chunk_idx in target_chunks:
                enc_np = encoded.numpy()
                if enc_np.shape[1] == 1024:  # [B, D, T]
                    frame0 = enc_np[0, :, 0]
                    frame1 = enc_np[0, :, 1] if enc_np.shape[2] > 1 else enc_np[0, :, 0]
                else:  # [B, T, D]
                    frame0 = enc_np[0, 0, :]
                    frame1 = enc_np[0, 1, :] if enc_np.shape[1] > 1 else enc_np[0, 0, :]

                print(f"=== CHUNK {chunk_idx} ===")
                print(f"  mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
                print(f"  drop_extra: {drop_extra}")
                print(f"  encoded shape: {list(encoded.shape)}, encoded_len: {encoded_len.item()}")
                print(f"  cache_filled_lens: {filled_lens}")
                print(f"  Frame 0 norm: {np.linalg.norm(frame0):.6f}")
                print(f"  Frame 0 first 10: [{', '.join(f'{v:.6f}' for v in frame0[:10])}]")
                print(f"  Frame 1 norm: {np.linalg.norm(frame1):.6f}")
                print(f"  Frame 1 first 10: [{', '.join(f'{v:.6f}' for v in frame1[:10])}]")
                print()

            audio_pos = end_pos
            chunk_idx += 1


if __name__ == '__main__':
    main()
