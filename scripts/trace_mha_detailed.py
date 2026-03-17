#!/usr/bin/env python3
"""Trace MHA internals in detail for layer 1, chunk 1."""

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

    # Patch the MHA forward to capture internals
    original_forward = mha.forward

    def patched_forward(query, key, value, mask=None, pos_emb=None, cache=None, cache_len=None):
        n_batch = query.size(0)
        n_head = mha.h
        d_head = mha.d_k
        d_model = n_head * d_head

        # Store inputs
        captured['query_input'] = query.detach().clone()
        captured['key_input'] = key.detach().clone()
        captured['value_input'] = value.detach().clone()
        if pos_emb is not None:
            captured['pos_emb'] = pos_emb.detach().clone()
        if mask is not None:
            captured['mask'] = mask.detach().clone()

        # Linear projections
        q = mha.linear_q(query)
        k = mha.linear_k(key)
        v = mha.linear_v(value)

        captured['q_proj'] = q.detach().clone()
        captured['k_proj'] = k.detach().clone()
        captured['v_proj'] = v.detach().clone()

        # Reshape to multi-head
        q = q.view(n_batch, -1, n_head, d_head).transpose(1, 2)
        k = k.view(n_batch, -1, n_head, d_head).transpose(1, 2)
        v = v.view(n_batch, -1, n_head, d_head).transpose(1, 2)

        captured['q_reshaped'] = q.detach().clone()
        captured['k_reshaped'] = k.detach().clone()
        captured['v_reshaped'] = v.detach().clone()

        # Position embedding projection
        p = mha.linear_pos(pos_emb)
        p = p.view(1, -1, n_head, d_head).transpose(1, 2)
        captured['p_proj'] = p.detach().clone()

        # Position biases
        q_with_bias_u = q + mha.pos_bias_u.view(1, n_head, 1, d_head)
        q_with_bias_v = q + mha.pos_bias_v.view(1, n_head, 1, d_head)

        captured['pos_bias_u'] = mha.pos_bias_u.detach().clone()
        captured['pos_bias_v'] = mha.pos_bias_v.detach().clone()
        captured['q_with_bias_u'] = q_with_bias_u.detach().clone()
        captured['q_with_bias_v'] = q_with_bias_v.detach().clone()

        # Content attention: matrix_ac
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        captured['matrix_ac'] = matrix_ac.detach().clone()

        # Position attention: matrix_bd
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        captured['matrix_bd_before_shift'] = matrix_bd.detach().clone()

        # rel_shift
        matrix_bd = mha.rel_shift(matrix_bd)
        captured['matrix_bd_after_shift'] = matrix_bd.detach().clone()

        # Combine and scale
        scores = (matrix_ac + matrix_bd) / (d_head ** 0.5)
        captured['scores_before_mask'] = scores.detach().clone()

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, min_val)
        captured['scores_after_mask'] = scores.detach().clone()

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        captured['attn_weights'] = attn_weights.detach().clone()

        # Apply to values
        context = torch.matmul(attn_weights, v)
        captured['context_before_reshape'] = context.detach().clone()

        context = context.transpose(1, 2).contiguous().view(n_batch, -1, d_model)
        captured['context_after_reshape'] = context.detach().clone()

        # Output projection
        output = mha.linear_out(context)
        captured['output'] = output.detach().clone()

        return output

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

            # Temporarily patch
            mha.forward = patched_forward

            try:
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
            finally:
                mha.forward = original_forward

            if chunk_idx == 1 and captured:
                print(f"\n{'='*70}")
                print(f"CHUNK 1 - Layer 1 MHA Detailed Trace")
                print(f"{'='*70}")

                def print_tensor(name, t, positions=None):
                    print(f"\n{name}:")
                    print(f"  shape: {list(t.shape)}")
                    t_np = t.numpy()
                    if len(t.shape) == 3:  # [B, T, D]
                        frame0 = t_np[0, 0, :]
                        print(f"  [0,0,:5]: {frame0[:5].tolist()}")
                        print(f"  [0,0] norm: {np.linalg.norm(frame0):.6f}")
                        if positions:
                            for pos in positions:
                                if pos < t.shape[1]:
                                    frame = t_np[0, pos, :]
                                    print(f"  [0,{pos},:5]: {frame[:5].tolist()}")
                                    print(f"  [0,{pos}] norm: {np.linalg.norm(frame):.6f}")
                    elif len(t.shape) == 4:  # [B, H, T, D] or [B, H, Q, K]
                        print(f"  [0,0,0,:5]: {t_np[0, 0, 0, :5].tolist()}")
                        print(f"  [0,0,0] norm: {np.linalg.norm(t_np[0, 0, 0]):.6f}")
                        if positions:
                            for pos in positions:
                                if pos < t.shape[2]:
                                    print(f"  [0,0,{pos},:5]: {t_np[0, 0, pos, :5].tolist()}")

                # Print key tensors
                for name in ['query_input', 'key_input', 'value_input', 'pos_emb']:
                    if name in captured:
                        print_tensor(name, captured[name])

                for name in ['q_proj', 'k_proj', 'v_proj']:
                    if name in captured:
                        print_tensor(name, captured[name], positions=[0, 65])

                for name in ['pos_bias_u', 'pos_bias_v']:
                    if name in captured:
                        t = captured[name]
                        print(f"\n{name}:")
                        print(f"  shape: {list(t.shape)}")
                        print(f"  [:5]: {t.numpy().flatten()[:5].tolist()}")
                        print(f"  norm: {torch.norm(t).item():.6f}")

                for name in ['matrix_ac', 'matrix_bd_before_shift', 'matrix_bd_after_shift',
                             'scores_before_mask', 'scores_after_mask', 'attn_weights']:
                    if name in captured:
                        t = captured[name]
                        print(f"\n{name}:")
                        print(f"  shape: {list(t.shape)}")
                        t_np = t.numpy()
                        print(f"  [0,0,0,:5]: {t_np[0, 0, 0, :5].tolist()}")
                        print(f"  [0,0,0,65:70]: {t_np[0, 0, 0, 65:70].tolist()}")
                        if 'attn' in name:
                            # Show where attention is focusing
                            row = t_np[0, 0, 0, :]
                            top_idx = np.argsort(row)[-5:][::-1]
                            print(f"  [0,0,0] top5 positions: {top_idx.tolist()}")
                            print(f"  [0,0,0] top5 values: {row[top_idx].tolist()}")

                for name in ['context_after_reshape', 'output']:
                    if name in captured:
                        print_tensor(name, captured[name])

                # Also print mask info
                if 'mask' in captured:
                    mask = captured['mask']
                    print(f"\nmask:")
                    print(f"  shape: {list(mask.shape)}")
                    mask_np = mask.numpy()
                    true_count = mask_np.sum()
                    print(f"  total True: {true_count}")
                    # First query's visible positions
                    q0_mask = mask_np[0, 0, :]
                    visible = np.where(q0_mask)[0]
                    print(f"  query 0 visible positions: {visible.tolist()}")

            audio_pos = end_pos
            chunk_idx += 1


if __name__ == '__main__':
    main()
