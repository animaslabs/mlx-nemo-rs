#!/usr/bin/env python3
"""Trace attention scores and weights in detail for layer 1, chunk 1."""

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

    # Get the original forward and patch it to capture internals
    original_forward = mha.forward

    def patched_forward(query, key, value, mask, pos_emb, cache=None):
        # Call update_cache like the original
        key, value, query, new_cache = mha.update_cache(key=key, value=value, query=query, cache=cache)

        n_head = mha.h
        d_head = mha.d_k
        d_model = n_head * d_head

        # Get Q, K, V projections
        q, k, v = mha.forward_qkv(query, key, value)
        # Note: forward_qkv returns [batch, head, time, d_k]
        # But we need to add biases differently

        # In NeMo's RelPositionMultiHeadAttention.forward:
        # q = q.transpose(1, 2)  # [batch, time1, head, d_k]
        # Then q_with_bias_u = (q + pos_bias_u).transpose(1, 2)  # [batch, head, time1, d_k]

        # Let's redo this manually to match NeMo exactly
        n_batch = query.size(0)
        q_linear = mha.linear_q(query).view(n_batch, -1, n_head, d_head)  # [batch, time, head, d_k]
        k_linear = mha.linear_k(key).view(n_batch, -1, n_head, d_head)
        v_linear = mha.linear_v(value).view(n_batch, -1, n_head, d_head)

        # Transpose Q to [batch, time, head, d_k] then add bias
        q_for_bias = q_linear  # Already [batch, time, head, d_k]
        q_with_bias_u = (q_for_bias + mha.pos_bias_u).transpose(1, 2)  # [batch, head, time, d_k]
        q_with_bias_v = (q_for_bias + mha.pos_bias_v).transpose(1, 2)  # [batch, head, time, d_k]

        # K, V need to be [batch, head, time, d_k]
        k_t = k_linear.transpose(1, 2)  # [batch, head, time, d_k]
        v_t = v_linear.transpose(1, 2)  # [batch, head, time, d_k]

        # Position embedding projection
        n_batch_pos = pos_emb.size(0)
        p = mha.linear_pos(pos_emb).view(n_batch_pos, -1, n_head, d_head)
        p = p.transpose(1, 2)  # [batch, head, time, d_k]

        # Content attention: matrix_ac
        matrix_ac = torch.matmul(q_with_bias_u, k_t.transpose(-2, -1))  # [batch, head, time1, time2]

        # Position attention: matrix_bd
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = mha.rel_shift(matrix_bd)
        # Truncate to match k's length
        matrix_bd = matrix_bd[:, :, :, :k_t.size(2)]

        # Store for debugging
        captured['matrix_ac'] = matrix_ac.detach().clone()
        captured['matrix_bd'] = matrix_bd.detach().clone()

        # Combine and scale
        scores = (matrix_ac + matrix_bd) / mha.s_d_k
        captured['scores'] = scores.detach().clone()

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # [batch, 1, time1, time2]
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        captured['scores_after_mask'] = scores.detach().clone()

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)
        captured['attn_weights'] = attn_weights.detach().clone()

        # Apply to values
        context = torch.matmul(attn_weights, v_t)  # [batch, head, time1, d_k]
        context = context.transpose(1, 2).reshape(n_batch, -1, d_model)  # [batch, time1, d_model]
        captured['context'] = context.detach().clone()

        # Output projection
        output = mha.linear_out(context)
        captured['output'] = output.detach().clone()

        if new_cache is None:
            return output
        else:
            return output, new_cache

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
                print(f"CHUNK 1 - Layer 1 Attention Details")
                print(f"{'='*70}")

                # Print matrix_ac, matrix_bd, scores at [0,0,0,65:70]
                if 'matrix_ac' in captured:
                    ac = captured['matrix_ac'][0, 0, 0, 65:70].numpy()
                    print(f"\nmatrix_ac [0,0,0,65:70]: {ac.tolist()}")

                if 'matrix_bd' in captured:
                    bd = captured['matrix_bd'][0, 0, 0, 65:70].numpy()
                    print(f"matrix_bd [0,0,0,65:70]: {bd.tolist()}")

                if 'scores' in captured:
                    sc = captured['scores'][0, 0, 0, 65:70].numpy()
                    print(f"scores [0,0,0,65:70] (scale={mha.s_d_k}): {sc.tolist()}")

                if 'attn_weights' in captured:
                    aw = captured['attn_weights'][0, 0, 0, :].numpy()
                    aw_65_72 = aw[65:73].tolist()
                    top_idx = np.argsort(aw)[-5:][::-1]
                    top5 = [(int(i), float(aw[i])) for i in top_idx]
                    print(f"\nattn_weights [0,0,0,65:72]: {aw_65_72}")
                    print(f"attn_weights top5: {top5}")

                if 'context' in captured:
                    ctx = captured['context'][0, 0, :10].numpy()
                    ctx_norm = np.linalg.norm(captured['context'][0, 0, :].numpy())
                    print(f"\ncontext [0,0,:10]: {ctx.tolist()}")
                    print(f"context norm: {ctx_norm}")

                if 'output' in captured:
                    out = captured['output'][0, 0, :10].numpy()
                    out_norm = np.linalg.norm(captured['output'][0, 0, :].numpy())
                    print(f"\noutput [0,0,:10]: {out.tolist()}")
                    print(f"output norm: {out_norm}")

            audio_pos = end_pos
            chunk_idx += 1


if __name__ == '__main__':
    main()
