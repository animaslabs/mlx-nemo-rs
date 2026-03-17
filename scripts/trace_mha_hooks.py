#!/usr/bin/env python3
"""Trace MHA internals using hooks for layer 1, chunk 1."""

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

    # Get layer 1's self_attn
    layer1 = encoder.layers[1]
    mha = layer1.self_attn

    # Storage for captured data
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                captured[f'{name}_input'] = inp[0].detach().clone() if isinstance(inp[0], torch.Tensor) else None
            if isinstance(out, torch.Tensor):
                captured[f'{name}_output'] = out.detach().clone()
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                captured[f'{name}_output'] = out[0].detach().clone()
        return hook

    # Register hooks on the linear layers
    hooks = []
    hooks.append(mha.linear_q.register_forward_hook(make_hook('linear_q')))
    hooks.append(mha.linear_k.register_forward_hook(make_hook('linear_k')))
    hooks.append(mha.linear_v.register_forward_hook(make_hook('linear_v')))
    hooks.append(mha.linear_pos.register_forward_hook(make_hook('linear_pos')))
    hooks.append(mha.linear_out.register_forward_hook(make_hook('linear_out')))

    # Also hook the full MHA to get input/output
    hooks.append(mha.register_forward_hook(make_hook('mha')))

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

            captured.clear()

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

            # Trace chunk 0 layer 1 frame 4 (will be at cache pos 69 for chunk 1)
            if chunk_idx == 0 and captured:
                if 'linear_k_input' in captured and captured['linear_k_input'] is not None:
                    t = captured['linear_k_input']
                    seq_len = t.shape[1]
                    # For chunk 0: kv_input = concat(zeros(70), input(5)) = 75
                    # Input frames are at positions 70-74
                    # Frame 4 of input is at position 74
                    cache_size = 70  # from config
                    input_frame4_pos = cache_size + 4  # Position of input frame 4 in kv_input
                    print(f"\nCHUNK 0 - Layer 1 kv_input shape: {list(t.shape)}")
                    if seq_len > input_frame4_pos:
                        frame4 = t.numpy()[0, input_frame4_pos, :]
                        print(f"  [0,{input_frame4_pos},:10] (input frame 4): {frame4[:10].tolist()}")
                        print(f"  [0,{input_frame4_pos}] norm: {np.linalg.norm(frame4):.6f}")
                    # Also trace the last position
                    if seq_len > 0:
                        last_frame = t.numpy()[0, seq_len-1, :]
                        print(f"  [0,{seq_len-1},:10] (last frame): {last_frame[:10].tolist()}")
                        print(f"  [0,{seq_len-1}] norm: {np.linalg.norm(last_frame):.6f}")

            if chunk_idx == 1 and captured:
                print(f"\n{'='*70}")
                print(f"CHUNK 1 - Layer 1 MHA Hook Trace")
                print(f"{'='*70}")

                def print_tensor(name, t):
                    print(f"\n{name}:")
                    print(f"  shape: {list(t.shape)}")
                    t_np = t.numpy()
                    if len(t.shape) == 3:  # [B, T, D]
                        frame0 = t_np[0, 0, :]
                        print(f"  [0,0,:10]: {frame0[:10].tolist()}")
                        print(f"  [0,0] norm: {np.linalg.norm(frame0):.6f}")
                        # Also print frame at position 65 if available (first filled cache position)
                        if t.shape[1] > 65:
                            frame65 = t_np[0, 65, :]
                            print(f"  [0,65,:10]: {frame65[:10].tolist()}")
                            print(f"  [0,65] norm: {np.linalg.norm(frame65):.6f}")
                        # Also print frame at position 69 (divergence point!)
                        if t.shape[1] > 69:
                            frame69 = t_np[0, 69, :]
                            print(f"  [0,69,:10]: {frame69[:10].tolist()}")
                            print(f"  [0,69] norm: {np.linalg.norm(frame69):.6f}")
                        # Also print positions 70-72 (current chunk positions)
                        for pos in [70, 71, 72]:
                            if t.shape[1] > pos:
                                frame = t_np[0, pos, :]
                                print(f"  [0,{pos},:10]: {frame[:10].tolist()}")
                                print(f"  [0,{pos}] norm: {np.linalg.norm(frame):.6f}")
                    elif len(t.shape) == 2:  # [H, D]
                        print(f"  [:5,:5]: {t_np[:5, :5].tolist()}")
                        print(f"  norm: {np.linalg.norm(t_np):.6f}")

                # Print captured tensors
                for key in sorted(captured.keys()):
                    if captured[key] is not None:
                        print_tensor(key, captured[key])

                # Also print pos_bias_u and pos_bias_v
                print(f"\npos_bias_u:")
                pu = mha.pos_bias_u.detach().numpy()
                print(f"  shape: {list(pu.shape)}")
                print(f"  [0,:5]: {pu[0, :5].tolist()}")
                print(f"  norm: {np.linalg.norm(pu):.6f}")

                print(f"\npos_bias_v:")
                pv = mha.pos_bias_v.detach().numpy()
                print(f"  shape: {list(pv.shape)}")
                print(f"  [0,:5]: {pv[0, :5].tolist()}")
                print(f"  norm: {np.linalg.norm(pv):.6f}")

            audio_pos = end_pos
            chunk_idx += 1

    # Remove hooks
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
