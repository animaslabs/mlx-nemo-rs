#!/usr/bin/env python3
"""
Detailed trace of NeMo streaming pipeline with intermediate values.

This script hooks into NeMo's streaming infrastructure to capture
encoder/decoder outputs at each step for comparison with Rust.
"""

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
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]

    config_path = Path('streaming_config.yaml')
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
    cfg.asr.device = 'cpu'
    cfg.asr.compute_dtype = 'float32'
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 1]  # mode 2: 80ms lookahead
    cfg.asr_decoding_type = 'rnnt'
    cfg.log_level = 40

    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    print("Building pipeline...")
    pipeline = PipelineBuilder.build_pipeline(cfg)

    asr_model = pipeline.asr_model.asr_model
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint
    blank_id = decoder.vocab_size

    print(f"\n=== Model Info ===")
    print(f"Blank ID: {blank_id}")
    print(f"Encoder d_model: {encoder.d_model}")
    print(f"Subsampling: {encoder.subsampling_factor}")
    print(f"Vocab size: {decoder.vocab_size}")

    # Get streaming config
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
    print(f"cache_drop_size: {streaming_cfg.cache_drop_size}")
    print(f"last_channel_cache_size: {streaming_cfg.last_channel_cache_size}")
    print(f"last_time_cache_size: {getattr(streaming_cfg, 'last_time_cache_size', 'N/A')}")

    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    print(f"\n=== Audio ===")
    print(f"Path: {audio_path}")
    print(f"Duration: {len(audio)/sr:.2f}s")
    print(f"Samples: {len(audio)}")

    # Get initial cache (returns tuple: cache_last_channel, cache_last_time, cache_last_channel_len)
    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0  # max_dim=0 for zeros init
    )

    print(f"\n=== Initial Cache ===")
    print(f"cache_last_channel: {list(cache_last_channel.shape)}")
    print(f"cache_last_time: {list(cache_last_time.shape)}")
    print(f"cache_last_channel_len: {cache_last_channel_len.item()}")

    # Initialize decoder state (add_sos=True)
    print(f"\n=== Decoder Initialization (add_sos=True) ===")
    pred_result = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
    pred_out, dec_state = pred_result[0], pred_result[1]
    pred_out = pred_out[:, -1:, :]  # Take last timestep (warmed up)
    print(f"Pred output shape: {list(pred_out.shape)}")
    print(f"Pred output[:10]: {pred_out[0, 0, :10].tolist()}")
    print(f"State hidden[0,0,:5]: {dec_state[0][0, 0, :5].tolist()}")

    # Streaming parameters
    # The model_chunk_size[1] is what NeMo uses for ALL chunks in CacheAwareRNNTPipeline
    subsampling_factor = encoder.subsampling_factor
    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]  # mel frames per chunk

    # Convert to audio samples
    # Try different attribute paths for hop_length
    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        # Fallback: compute from sample_rate and window_stride
        hop_length = int(16000 * 0.01)  # 10ms hop at 16kHz

    # Audio samples per chunk (like Rust's audio_chunk_size)
    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    print(f"\n=== Chunk Parameters ===")
    print(f"pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"model_chunk_size: {model_chunk_size}")
    print(f"hop_length: {hop_length}")
    print(f"audio_chunk_samples: {audio_chunk_samples}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print(f"valid_out_len: {streaming_cfg.valid_out_len}")

    # Process audio in streaming chunks
    print(f"\n=== Processing Audio in Streaming Chunks ===")

    audio_pos = 0
    chunk_idx = 0
    all_tokens = []
    max_chunks = 30

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx < max_chunks:
            # Get audio chunk
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]

            # Pad if needed
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            # Get mel spectrogram
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            # Run encoder with cache
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
                keep_all_outputs=True,  # Get all outputs for comparison
                drop_extra_pre_encoded=drop_extra,
            )

            encoded_t = encoded.transpose(1, 2)  # [B, T, D]

            print(f"\n--- Chunk {chunk_idx} ---")
            print(f"Audio: [{audio_pos}:{end_pos}] ({end_pos - audio_pos} samples)")
            print(f"Mel shape: {list(mel.shape)}, len: {mel_len.item()}")
            print(f"drop_extra: {drop_extra}")
            print(f"Encoded shape: {list(encoded_t.shape)}, len: {encoded_len.item()}")
            print(f"cache_last_channel_len: {cache_last_channel_len.item()}")

            # Print encoder frame values
            enc_frames = encoded_t.shape[1]
            if enc_frames > 0:
                frame0 = encoded_t[0, 0, :10].tolist()
                print(f"Encoder frame 0 [:10]: {[round(v, 6) for v in frame0]}")
                norm = torch.norm(encoded_t[0, 0]).item()
                print(f"Encoder frame 0 L2 norm: {norm:.6f}")

                if enc_frames > 1:
                    frame1 = encoded_t[0, 1, :10].tolist()
                    print(f"Encoder frame 1 [:10]: {[round(v, 6) for v in frame1]}")
                    norm1 = torch.norm(encoded_t[0, 1]).item()
                    print(f"Encoder frame 1 L2 norm: {norm1:.6f}")

            # Get valid output length (clip to available frames)
            valid_out = min(streaming_cfg.valid_out_len, enc_frames)
            print(f"valid_out_len: {valid_out} (of {enc_frames} available)")

            # Decode frames using greedy RNN-T
            chunk_tokens = []
            for frame_idx in range(valid_out):
                enc_frame = encoded_t[:, frame_idx:frame_idx+1, :]

                # Inner RNN-T loop for this frame
                max_symbols_per_frame = 10
                symbols_this_frame = 0

                while symbols_this_frame < max_symbols_per_frame:
                    # Run joint network
                    enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
                    pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out
                    combined = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
                    joint_out = joint.joint_net(combined)
                    logits = joint_out[0, 0, 0, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    best_token = torch.argmax(log_probs).item()

                    if best_token == blank_id:
                        break  # Move to next frame

                    # Emit token
                    chunk_tokens.append(best_token)
                    all_tokens.append(best_token)
                    symbols_this_frame += 1

                    # Update decoder state with emitted token
                    y_new = torch.tensor([[best_token]], dtype=torch.long)
                    new_pred_result = decoder.predict(y=y_new, state=dec_state, add_sos=False)
                    pred_out = new_pred_result[0]
                    dec_state = new_pred_result[1]

            # Decode tokens to text
            if chunk_tokens and hasattr(asr_model, 'tokenizer') and asr_model.tokenizer:
                chunk_text = asr_model.tokenizer.ids_to_text(chunk_tokens)
                print(f"Chunk tokens: {chunk_tokens}")
                print(f"Chunk text: '{chunk_text}'")
            elif chunk_tokens:
                print(f"Chunk tokens: {chunk_tokens}")
            else:
                print(f"Chunk tokens: (none)")

            audio_pos = end_pos
            chunk_idx += 1

    # Final transcription
    print(f"\n=== Final Transcription ===")
    print(f"All tokens ({len(all_tokens)}): {all_tokens[:50]}{'...' if len(all_tokens) > 50 else ''}")
    if all_tokens and hasattr(asr_model, 'tokenizer') and asr_model.tokenizer:
        final_text = asr_model.tokenizer.ids_to_text(all_tokens)
        print(f"Text: '{final_text}'")


if __name__ == '__main__':
    main()
