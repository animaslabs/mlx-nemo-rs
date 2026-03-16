#!/usr/bin/env python3
"""
Trace streaming pipeline values for comparison with Rust implementation.

This script processes audio through the NeMo streaming pipeline and outputs
intermediate values at each chunk for direct comparison with the Rust implementation.

Values traced:
1. Mel spectrogram per chunk
2. Encoder output per chunk
3. Decoder prediction output
4. Joint network output
5. Predicted tokens

Usage:
    cd scripts
    source .venv/bin/activate
    python trace_streaming_pipeline.py ../samples/2086-149220-0033.wav

The output format is designed to be easily parsed and compared with Rust output.
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


def format_values(tensor, n=10):
    """Format first n values of a tensor for display."""
    flat = tensor.flatten().tolist()
    return [round(v, 6) for v in flat[:n]]


def main():
    if len(sys.argv) < 2:
        print("Usage: python trace_streaming_pipeline.py <audio_file>")
        print("       python trace_streaming_pipeline.py ../samples/2086-149220-0033.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Load config
    config_path = Path('streaming_config.yaml')
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
    cfg.asr.device = 'cpu'
    cfg.asr.compute_dtype = 'float32'
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 1]  # 80ms lookahead
    cfg.asr_decoding_type = 'rnnt'
    cfg.log_level = 40

    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    print("Building pipeline...")
    pipeline = PipelineBuilder.build_pipeline(cfg)

    # Get model components
    asr_model = pipeline.asr_model.asr_model
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint

    # Compute streaming parameters from model config
    # These are computed based on att_context_size and model architecture
    att_context = cfg.streaming.att_context_size  # [left, right] e.g., [70, 1] for 80ms lookahead
    left_context = att_context[0]
    right_context = att_context[1]  # This is the lookahead in encoder frames

    # The chunk_size is computed as: right_context + 1 (for the current frame)
    # shift_size = right_context + 1 (same as chunk_size for this model)
    # pre_encode_cache_size = left_context
    chunk_size = right_context + 1
    shift_size = chunk_size
    pre_encode_cache_size = left_context
    valid_out_len = right_context + 1  # Number of valid output frames

    print(f"\n=== Streaming Configuration ===")
    print(f"att_context_size: {att_context}")
    print(f"chunk_size: {chunk_size} (encoder frames)")
    print(f"shift_size: {shift_size} (encoder frames)")
    print(f"pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"valid_out_len: {valid_out_len}")
    print(f"subsampling: {encoder.subsampling_factor}")

    blank_id = decoder.vocab_size
    print(f"\n=== Model Configuration ===")
    print(f"Blank ID: {blank_id}")
    print(f"Vocab size: {blank_id + 1}")
    print(f"Pred hidden: {decoder.pred_hidden}")
    print(f"Encoder hidden: {encoder.d_model}")

    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    print(f"\n=== Audio ===")
    print(f"Path: {audio_path}")
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio)/sr:.2f}s")
    print(f"Samples: {len(audio)}")

    # Streaming parameters were computed above
    sample_rate = 16000
    # chunk_size, shift_size, pre_encode_cache_size, valid_out_len already defined

    # Samples per chunk
    # chunk_size is in mel frames (10ms per frame for 80 mel hop)
    hop_length = preprocessor.hop_length  # Usually 160 samples
    chunk_samples = chunk_size * hop_length
    shift_samples = shift_size * hop_length

    print(f"\n=== Chunk Parameters ===")
    print(f"Hop length: {hop_length} samples")
    print(f"Chunk samples: {chunk_samples}")
    print(f"Shift samples: {shift_samples}")

    # Process chunks
    print(f"\n=== Processing Chunks ===")

    # Initialize decoder state
    with torch.no_grad():
        # Initialize prediction network state
        pred_result = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
        pred_out, state = pred_result[0], pred_result[1]
        # Use last timestep for decoding
        pred_out = pred_out[:, -1:, :]  # [1, 1, pred_hidden]

        print(f"\nInitial pred output[:5]: {format_values(pred_out)}")
        print(f"Initial state hidden[0,0,:5]: {format_values(state[0][0, 0])}")

        # Track encoder cache (simplified - actual cache management is more complex)
        total_emitted = []
        chunk_idx = 0
        start_sample = 0

        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]

            # Pad chunk if necessary (for last chunk)
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(chunk)])

            print(f"\n--- Chunk {chunk_idx} ---")
            print(f"Samples: {start_sample}-{end_sample}")
            print(f"Chunk length: {len(chunk)}")

            # Get mel spectrogram
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
            print(f"Mel shape: {list(mel.shape)}")
            print(f"Mel[:5] (first frame): {format_values(mel[0, :5, 0])}")

            # Run encoder (note: this doesn't handle caching properly for true streaming)
            # For accurate comparison, we'd need to replicate the exact cache handling
            encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)
            encoded_t = encoded.transpose(1, 2)  # [B, T, D]

            print(f"Encoded shape: {list(encoded_t.shape)}")
            print(f"Encoded frames: {encoded_t.shape[1]}")

            # Get valid frames for this chunk
            num_valid = min(valid_out_len, encoded_t.shape[1])
            print(f"Valid frames: {num_valid}")

            if encoded_t.shape[1] > 0:
                print(f"Encoder out[:5] (frame 0): {format_values(encoded_t[0, 0])}")
                if encoded_t.shape[1] > 1:
                    print(f"Encoder out[:5] (frame 1): {format_values(encoded_t[0, 1])}")

            # Process frames through decoder
            chunk_tokens = []
            for t in range(num_valid):
                enc_frame = encoded_t[:, t:t+1, :]

                # Run joint network
                enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
                pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out

                enc_proj_exp = enc_proj.unsqueeze(2)
                pred_proj_exp = pred_proj.unsqueeze(1)
                combined = enc_proj_exp + pred_proj_exp
                joint_out = joint.joint_net(combined)

                logits = joint_out[0, 0, 0, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                best_token = torch.argmax(log_probs).item()
                best_prob = log_probs[best_token].item()

                if t == 0:  # Only print first frame details
                    print(f"Frame {t}: enc_proj[:5]={format_values(enc_proj)}")
                    print(f"Frame {t}: pred_proj[:5]={format_values(pred_proj)}")
                    print(f"Frame {t}: log_probs[:5]={format_values(log_probs)}")

                if best_token == blank_id:
                    print(f"Frame {t}: BLANK (prob={best_prob:.6f})")
                else:
                    print(f"Frame {t}: TOKEN {best_token} (prob={best_prob:.6f})")
                    chunk_tokens.append(best_token)
                    total_emitted.append(best_token)

                    # Update decoder state
                    y_new = torch.tensor([[best_token]], dtype=torch.long)
                    pred_out, state = decoder.predict(y=y_new, state=state, add_sos=False)[:2]

                    # Inner loop - keep emitting while not blank
                    inner_count = 0
                    max_inner = 5
                    while inner_count < max_inner:
                        inner_count += 1

                        enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
                        pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out
                        enc_proj_exp = enc_proj.unsqueeze(2)
                        pred_proj_exp = pred_proj.unsqueeze(1)
                        combined = enc_proj_exp + pred_proj_exp
                        joint_out = joint.joint_net(combined)

                        logits = joint_out[0, 0, 0, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        best_token = torch.argmax(log_probs).item()
                        best_prob = log_probs[best_token].item()

                        if best_token == blank_id:
                            break
                        else:
                            print(f"Frame {t}[inner]: TOKEN {best_token} (prob={best_prob:.6f})")
                            chunk_tokens.append(best_token)
                            total_emitted.append(best_token)

                            y_new = torch.tensor([[best_token]], dtype=torch.long)
                            pred_out, state = decoder.predict(y=y_new, state=state, add_sos=False)[:2]

            print(f"Chunk tokens: {chunk_tokens}")

            # Advance to next chunk
            start_sample += shift_samples
            chunk_idx += 1

            # Limit chunks for testing
            if chunk_idx >= 10:
                print("\n(Stopping after 10 chunks for testing)")
                break

    print(f"\n=== Summary ===")
    print(f"Total tokens emitted: {len(total_emitted)}")
    print(f"Tokens: {total_emitted}")

    # Decode to text
    if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
        text = asr_model.tokenizer.ids_to_text(total_emitted)
        print(f"Text: '{text}'")


if __name__ == '__main__':
    main()
