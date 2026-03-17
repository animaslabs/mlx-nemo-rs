#!/usr/bin/env python3
"""Compare encoder outputs for all chunks between Python NeMo and Rust."""

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

    audio_pos = 0
    chunk_idx = 0

    # Collect all encoder outputs
    all_outputs = []

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

            enc_np = encoded.numpy()
            enc_len = int(encoded_len.item())

            # Shape is [B, D, T] where D=1024 features, T=time frames
            # To get frame 0, we need enc_np[0, :, 0] (all features at time 0)
            frame0 = enc_np[0, :10, 0].tolist() if enc_np.shape[2] > 0 else []
            frame0_norm = np.linalg.norm(enc_np[0, :, 0]) if enc_np.shape[2] > 0 else 0

            all_outputs.append({
                'chunk': chunk_idx,
                'shape': list(enc_np.shape),
                'enc_len': enc_len,
                'frame0': frame0,
                'frame0_norm': frame0_norm,
            })

            audio_pos = end_pos
            chunk_idx += 1

    # Print all outputs in a format easy to compare
    print("PYTHON_ENCODER_OUTPUTS = [")
    for out in all_outputs:
        print(f"    {{'chunk': {out['chunk']}, 'shape': {out['shape']}, 'enc_len': {out['enc_len']}, 'frame0_norm': {out['frame0_norm']:.6f}, 'frame0': {out['frame0']}}},")
    print("]")


if __name__ == '__main__':
    main()
