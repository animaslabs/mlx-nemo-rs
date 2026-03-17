#!/usr/bin/env python3
"""Trace MHA internals for layer 1, chunk 1."""

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

    # Get layer 1 and its self_attn module
    layer1 = encoder.layers[1]
    mha = layer1.self_attn

    # Hook storage
    mha_internals = {}

    def hook_linear_q(module, input, output):
        mha_internals['q'] = output.detach().clone()

    def hook_linear_k(module, input, output):
        mha_internals['k'] = output.detach().clone()

    def hook_linear_v(module, input, output):
        mha_internals['v'] = output.detach().clone()

    def hook_linear_pos(module, input, output):
        mha_internals['pos_proj'] = output.detach().clone()

    def hook_linear_out(module, input, output):
        mha_internals['out'] = output.detach().clone()
        # Also capture input to output projection (which is the context after attention)
        if isinstance(input, tuple) and len(input) > 0:
            mha_internals['context'] = input[0].detach().clone()

    # Register hooks
    hooks = []
    hooks.append(mha.linear_q.register_forward_hook(hook_linear_q))
    hooks.append(mha.linear_k.register_forward_hook(hook_linear_k))
    hooks.append(mha.linear_v.register_forward_hook(hook_linear_v))
    hooks.append(mha.linear_pos.register_forward_hook(hook_linear_pos))
    hooks.append(mha.linear_out.register_forward_hook(hook_linear_out))

    audio_pos = 0
    chunk_idx = 0

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx <= 1:
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            mha_internals.clear()

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

            if chunk_idx == 1:
                print(f"\n{'='*60}")
                print(f"CHUNK 1 - Layer 1 MHA Internals")
                print(f"{'='*60}")

                for name in ['q', 'k', 'v', 'pos_proj', 'context', 'out']:
                    if name in mha_internals:
                        tensor = mha_internals[name]
                        print(f"\n{name}:")
                        print(f"  shape: {list(tensor.shape)}")
                        if len(tensor.shape) == 3:  # [B, T, D]
                            frame0 = tensor[0, 0, :]
                            print(f"  frame0 norm: {torch.norm(frame0).item():.6f}")
                            print(f"  frame0 [:5]: {frame0[:5].tolist()}")
                        elif len(tensor.shape) == 4:  # [B, H, T, D]
                            print(f"  [0,0,0,:5]: {tensor[0, 0, 0, :5].tolist()}")

                # Also print cache info
                print(f"\ncache_last_channel_len: {cache_last_channel_len.item() if cache_last_channel_len is not None else 'None'}")
                print(f"cache_last_channel shape: {list(cache_last_channel.shape)}")

            audio_pos = end_pos
            chunk_idx += 1

    # Remove hooks
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
