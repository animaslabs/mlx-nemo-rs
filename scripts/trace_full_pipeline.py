#!/usr/bin/env python3
"""Trace full pipeline values for comparison with Rust implementation.

This script traces:
1. Mel spectrogram values at each chunk
2. Encoder outputs at each chunk
3. Decoder state and predictions

Run with: python trace_full_pipeline.py
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
    audio_path = '../samples/Speaker26_004_16k.wav'

    config_path = Path('streaming_config.yaml')
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
    cfg.asr.device = 'cpu'
    cfg.asr.compute_dtype = 'float32'
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 13]  # Mode 0: 1040ms lookahead
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

    print(f"\n=== Streaming Config ===")
    print(f"chunk_size: {streaming_cfg.chunk_size}")
    print(f"shift_size: {streaming_cfg.shift_size}")
    print(f"valid_out_len: {streaming_cfg.valid_out_len}")
    print(f"pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")

    audio, sr = sf.read(audio_path, dtype='float32')
    print(f"\nAudio duration: {len(audio)/sr:.2f}s ({len(audio)} samples)")

    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')

    # Initialize encoder cache
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    # Initialize decoder state (LSTM hidden and cell states)
    dec_state = decoder.initialize_state(torch.zeros(batch_size))

    # Track last token for prediction network
    last_token = None

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]
    hop_length = 160

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    print(f"\n=== Chunk Processing ===")
    print(f"pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"model_chunk_size: {model_chunk_size}")
    print(f"buffer_mel_frames: {buffer_mel_frames}")
    print(f"audio_chunk_samples: {audio_chunk_samples}")

    audio_pos = 0
    chunk_idx = 0
    max_chunks = 10  # Trace first N chunks

    all_tokens = []

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx < max_chunks:
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            # Compute mel spectrogram
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            mel_flat = mel.flatten()
            mel_norm = l2_norm(mel_flat)
            mel_first10 = mel[0, :, 0].tolist()[:10]  # First frame, first 10 features

            print(f"\n=== Chunk {chunk_idx} ===")
            print(f"Audio pos: {audio_pos} - {end_pos} ({end_pos - audio_pos} samples)")
            print(f"Mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
            print(f"Mel frame 0 first 10: {[f'{v:.6f}' for v in mel_first10]}")
            print(f"Mel L2 norm: {mel_norm:.6f}")

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
            valid_frames = min(int(encoded_len.item()), num_frames)

            enc_flat = encoded.flatten()
            enc_norm = l2_norm(enc_flat)
            enc_frame0 = encoded[0, 0, :10].tolist() if num_frames > 0 else []

            print(f"\nEncoder output shape: {list(encoded.shape)}")
            print(f"Encoder valid_frames: {valid_frames}, raw encoded_len: {encoded_len.item()}")
            print(f"Encoder frame 0 first 10: {[f'{v:.6f}' for v in enc_frame0]}")
            print(f"Encoder L2 norm: {enc_norm:.6f}")

            # Decode frame by frame
            chunk_tokens = []
            for t in range(valid_frames):
                enc_frame = encoded[:, t:t+1, :]

                symbols_emitted = 0
                max_symbols = 10

                while symbols_emitted < max_symbols:
                    if last_token is None:
                        y = None
                    else:
                        y = torch.tensor([[last_token]], dtype=torch.long)

                    pred_output, new_dec_state = decoder.predict(
                        y,
                        state=dec_state,
                        add_sos=False,
                        batch_size=1
                    )

                    logits = joint.joint(enc_frame, pred_output)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs.squeeze()

                    pred_token = int(torch.argmax(log_probs).item())

                    if pred_token == blank_id:
                        break
                    else:
                        chunk_tokens.append(pred_token)
                        all_tokens.append(pred_token)
                        last_token = pred_token
                        dec_state = new_dec_state
                        symbols_emitted += 1

            print(f"\nChunk {chunk_idx} tokens: {chunk_tokens}")
            print(f"Total tokens so far: {len(all_tokens)}")

            audio_pos = end_pos
            chunk_idx += 1

    print(f"\n=== Summary ===")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"First 50 tokens: {all_tokens[:50]}")


if __name__ == '__main__':
    main()
