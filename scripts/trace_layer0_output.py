#!/usr/bin/env python3
"""Trace layer 0 output values for chunk 1."""

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

    # Patch layer 0 to trace its output
    layer0 = encoder.layers[0]
    original_forward = layer0.forward

    chunk_idx = [0]

    def traced_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if chunk_idx[0] == 1:
            if isinstance(result, tuple):
                output = result[0]
            else:
                output = result
            print(f"\n=== CHUNK 1 LAYER 0 OUTPUT ===")
            print(f"Layer 0 output shape: {list(output.shape)}")
            print(f"Layer 0 output[0, 0, :10]: {output[0, 0, :10].tolist()}")
            print(f"Layer 0 output[0, 0] norm: {torch.norm(output[0, 0]).item():.6f}")
        return result

    layer0.forward = traced_forward

    print("=== Processing Chunks ===")

    audio_pos = 0
    max_chunks = 2

    with torch.no_grad():
        for ci in range(max_chunks):
            chunk_idx[0] = ci
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
            print(f"CHUNK {ci}")
            print(f"{'='*60}")

            is_first = (ci == 0)
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

            print(f"\n=== CHUNK {ci} ENCODER OUTPUT ===")
            print(f"Encoder output shape: {list(encoded.shape)}")
            print(f"Encoder frame 0 [:10]: {encoded[0, :10, 0].tolist()}")
            print(f"Encoder frame 0 norm: {torch.norm(encoded[0, :, 0]).item():.6f}")

            audio_pos = end_pos


if __name__ == '__main__':
    main()
