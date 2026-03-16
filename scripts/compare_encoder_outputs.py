#!/usr/bin/env python3
"""Compare Python NeMo encoder outputs with expected Rust values."""

import os
import sys
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
    subsampling_factor = encoder.subsampling_factor
    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]

    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        hop_length = int(16000 * 0.01)

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    print("=== Detailed Encoder Output Comparison ===")
    print(f"pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"model_chunk_size: {model_chunk_size}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print()

    audio_pos = 0
    max_chunks = 5

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

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            print(f"--- Chunk {chunk_idx} ---")
            print(f"Mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
            print(f"drop_extra: {drop_extra}")

            # Print mel values for first few frames
            print(f"Mel frame 0 [:5]: {mel[0, :5, 0].tolist()}")
            print(f"Mel frame 0 L2 norm: {torch.norm(mel[0, :, 0]).item():.6f}")

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

            encoded_t = encoded.transpose(1, 2)  # [B, T, D]

            print(f"Encoded shape: {list(encoded_t.shape)}, len: {encoded_len.item()}")
            print(f"cache_last_channel_len: {cache_last_channel_len.item()}")

            # Print all encoder frames
            for frame_idx in range(min(encoded_t.shape[1], 5)):
                frame = encoded_t[0, frame_idx]
                print(f"  Frame {frame_idx} [:10]: {[round(v, 6) for v in frame[:10].tolist()]}")
                print(f"  Frame {frame_idx} L2 norm: {torch.norm(frame).item():.6f}")

            print()
            audio_pos = end_pos


if __name__ == '__main__':
    main()
