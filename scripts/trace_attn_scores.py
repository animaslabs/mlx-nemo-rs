#!/usr/bin/env python3
"""Trace attention scores for layer 1, chunk 1."""

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

    # Storage for scores
    scores_data = {}

    # Hook to capture intermediate values
    # We need to patch the forward method to get matrix_ac, matrix_bd
    original_forward = mha.forward.__func__ if hasattr(mha.forward, '__func__') else mha.forward

    def patched_forward(self, query, key, value, mask=None, pos_emb=None, cache=None, cache_len=None):
        # This is a simplified version to capture the scores
        # Based on NeMo's RelPositionMultiHeadAttention.forward

        n_batch = query.size(0)
        d_model = self.d_model
        n_head = self.n_head
        d_head = d_model // n_head

        # Linear projections
        q = self.linear_q(query).view(n_batch, -1, n_head, d_head)
        k = self.linear_k(key).view(n_batch, -1, n_head, d_head)
        v = self.linear_v(value).view(n_batch, -1, n_head, d_head)

        q = q.transpose(1, 2)  # [batch, n_head, time, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Positional embedding
        p = self.linear_pos(pos_emb).view(1, -1, n_head, d_head)
        p = p.transpose(1, 2)  # [1, n_head, pos_len, d_head]

        # Position biases
        q_with_bias_u = q + self.pos_bias_u.view(1, n_head, 1, d_head)
        q_with_bias_v = q + self.pos_bias_v.view(1, n_head, 1, d_head)

        # Content attention
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # [batch, n_head, q_len, k_len]

        # Position attention
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # [batch, n_head, q_len, pos_len]

        # rel_shift
        matrix_bd = self.rel_shift(matrix_bd)

        # Store for debugging
        scores_data['matrix_ac'] = matrix_ac.detach().clone()
        scores_data['matrix_bd'] = matrix_bd.detach().clone()

        # Combine and scale
        scores = (matrix_ac + matrix_bd) / (d_head ** 0.5)
        scores_data['scores'] = scores.detach().clone()
        scores_data['scale'] = d_head ** 0.5

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, q_len, k_len]
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, min_val)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        scores_data['attn_weights'] = attn_weights.detach().clone()

        # Apply to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(n_batch, -1, d_model)

        # Output projection
        output = self.linear_out(context)

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

            scores_data.clear()

            # Temporarily patch the forward method
            bound_method = patched_forward.__get__(mha, type(mha))
            mha.forward = bound_method

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
                # Restore original forward
                mha.forward = original_forward.__get__(mha, type(mha)) if hasattr(original_forward, '__get__') else original_forward

            if chunk_idx == 1 and scores_data:
                print(f"\n{'='*60}")
                print(f"CHUNK 1 - Layer 1 Attention Scores")
                print(f"{'='*60}")

                # Print scores for positions 65-72, head 0, query 0
                if 'matrix_ac' in scores_data:
                    ac = scores_data['matrix_ac'][0, 0, 0, 65:73].numpy()
                    print(f"matrix_ac (pos 65-72): [{', '.join(f'{v:.4f}' for v in ac)}]")

                if 'matrix_bd' in scores_data:
                    bd = scores_data['matrix_bd'][0, 0, 0, 65:73].numpy()
                    print(f"matrix_bd (pos 65-72): [{', '.join(f'{v:.4f}' for v in bd)}]")

                if 'scores' in scores_data:
                    sc = scores_data['scores'][0, 0, 0, 65:73].numpy()
                    print(f"scores (pos 65-72): [{', '.join(f'{v:.4f}' for v in sc)}]")

                if 'scale' in scores_data:
                    print(f"scale: {scores_data['scale']:.6f}")

                if 'attn_weights' in scores_data:
                    aw = scores_data['attn_weights'][0, 0, 0, 65:73].numpy()
                    print(f"attn_weights (pos 65-72): [{', '.join(f'{v:.6f}' for v in aw)}]")

            audio_pos = end_pos
            chunk_idx += 1


if __name__ == '__main__':
    main()
