#!/usr/bin/env python3
"""Trace decoder values for comparison with Rust implementation.

This script traces:
1. Encoder frame values at each timestep
2. Prediction network (LSTM) inputs and outputs
3. Joint network outputs
4. Token predictions

Run with: python trace_decoder.py
"""

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


def l2_norm(tensor):
    """Compute L2 norm of a tensor."""
    return float(torch.norm(tensor.float()).item())


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
    decoder = asr_model.decoder  # Prediction network
    joint = asr_model.joint      # Joint network

    # Get blank ID
    blank_id = len(joint.vocabulary)
    print(f"Blank ID: {blank_id}")
    print(f"Vocab size: {len(joint.vocabulary)}")

    streaming_cfg = encoder.streaming_cfg
    if streaming_cfg is None:
        encoder.setup_streaming_params()
        streaming_cfg = encoder.streaming_cfg

    audio, sr = sf.read(audio_path, dtype='float32')
    print(f"Audio duration: {len(audio)/sr:.2f}s")

    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')

    # Initialize encoder cache
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    # Initialize decoder state (LSTM hidden and cell states)
    dec_state = decoder.initialize_state(torch.zeros(batch_size))
    print(f"Decoder state type: {type(dec_state)}")
    if isinstance(dec_state, tuple):
        print(f"  Hidden shape: {dec_state[0].shape if dec_state[0] is not None else 'None'}")
        print(f"  Cell shape: {dec_state[1].shape if dec_state[1] is not None else 'None'}")

    # Track last token for prediction network
    last_token = None

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]
    hop_length = 160

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    audio_pos = 0
    chunk_idx = 0
    max_chunks = 5  # Only trace first few chunks for comparison
    global_step = 0

    all_tokens = []

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx < max_chunks:
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            # Run encoder
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

            # encoded: [B, D, T] -> [B, T, D]
            encoded = encoded.transpose(1, 2)
            num_frames = encoded.shape[1]

            print(f"\n=== Chunk {chunk_idx} ===")
            print(f"Encoder output shape: {list(encoded.shape)}, valid_frames: {encoded_len.item()}")

            # Decode frame by frame (mirroring streaming decoder)
            valid_frames = min(int(encoded_len.item()), num_frames)

            for t in range(valid_frames):
                # Get encoder frame: [1, 1, D]
                enc_frame = encoded[:, t:t+1, :]
                enc_flat = enc_frame.flatten()
                enc_norm = l2_norm(enc_flat)
                enc_first10 = enc_flat[:10].tolist()

                print(f"\nFrame {t}:")
                print(f"  encoder_frame_norm: {enc_norm:.6f}")
                print(f"  encoder_frame[:10]: {[f'{v:.6f}' for v in enc_first10]}")

                symbols_emitted = 0
                max_symbols = 10

                while symbols_emitted < max_symbols:
                    # Build input for prediction network
                    if last_token is None:
                        # First prediction - use None (will trigger SOS/blank embedding)
                        y = None
                        print(f"  Step {global_step}: pred_input=None (SOS)")
                    else:
                        y = torch.tensor([[last_token]], dtype=torch.long)
                        print(f"  Step {global_step}: pred_input={last_token}")

                    # Run prediction network
                    # decoder.predict returns (g, state) where g: [B, 1, pred_hidden]
                    pred_output, new_dec_state = decoder.predict(
                        y,
                        state=dec_state,
                        add_sos=False,
                        batch_size=1
                    )

                    pred_flat = pred_output.flatten()
                    pred_norm = l2_norm(pred_flat)
                    pred_first10 = pred_flat[:10].tolist()

                    print(f"  Step {global_step}: pred_output_norm: {pred_norm:.6f}")
                    print(f"  Step {global_step}: pred_output[:10]: {[f'{v:.6f}' for v in pred_first10]}")

                    # Run joint network
                    # enc_frame: [1, 1, enc_hidden], pred_output: [1, 1, pred_hidden]
                    # Joint expects [B, T, H] and [B, U, H]
                    logits = joint.joint(enc_frame, pred_output)  # [B, 1, 1, vocab+1]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs.squeeze()  # [vocab+1]

                    # Get prediction
                    pred_token = int(torch.argmax(log_probs).item())
                    pred_prob = float(log_probs[pred_token].item())
                    blank_prob = float(log_probs[blank_id].item())

                    # Get top-5
                    top5_vals, top5_idx = torch.topk(log_probs, 5)
                    top5 = [(int(idx), float(val)) for idx, val in zip(top5_idx, top5_vals)]

                    print(f"  Step {global_step}: pred_token={pred_token}, pred_prob={pred_prob:.6f}, blank_prob={blank_prob:.6f}")
                    print(f"  Step {global_step}: top5={[(i, f'{p:.4f}') for i, p in top5]}")
                    print(f"  Step {global_step}: is_blank={pred_token == blank_id}")

                    global_step += 1

                    if pred_token == blank_id:
                        # Blank - advance to next frame
                        break
                    else:
                        # Non-blank - emit token, update state
                        all_tokens.append(pred_token)
                        last_token = pred_token
                        dec_state = new_dec_state
                        symbols_emitted += 1
                        print(f"  -> Emitted token {pred_token}, total tokens: {len(all_tokens)}")

            audio_pos = end_pos
            chunk_idx += 1

    print(f"\n=== Summary ===")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Tokens: {all_tokens[:50]}{'...' if len(all_tokens) > 50 else ''}")


if __name__ == '__main__':
    main()
