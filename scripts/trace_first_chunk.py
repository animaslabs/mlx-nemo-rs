#!/usr/bin/env python3
"""
Trace the first chunk through the streaming pipeline.
This uses the proper NeMo streaming pipeline to get values that match Rust.
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
    cfg.streaming.att_context_size = [70, 1]  # 80ms lookahead
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

    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    print(f"\n=== Audio ===")
    print(f"Path: {audio_path}")
    print(f"Duration: {len(audio)/sr:.2f}s")
    print(f"Samples: {len(audio)}")

    # Process first chunk through streaming pipeline
    # Simulate what the Rust pipeline does

    # Rust uses chunk_ms = 100ms default, which is 1600 samples at 16kHz
    # But the actual chunk size depends on the streaming config
    # Let's use the same chunk size as the Rust streaming example

    chunk_ms = 100
    chunk_samples = int(sr * chunk_ms / 1000)

    print(f"\n=== First Chunk Processing ===")
    print(f"Chunk size: {chunk_ms}ms = {chunk_samples} samples")

    chunk = audio[:chunk_samples]
    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(chunk)])

    with torch.no_grad():
        # Get mel spectrogram
        mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
        print(f"\nMel shape: {list(mel.shape)}")
        print(f"Mel frames: {mel.shape[2]}")
        print(f"Mel[:5, frame 0]: {mel[0, :5, 0].tolist()}")

        # Run encoder
        encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)
        encoded_t = encoded.transpose(1, 2)  # [B, T, D]
        print(f"\nEncoded shape: {list(encoded_t.shape)}")
        print(f"Encoder frames: {encoded_t.shape[1]}")
        print(f"Encoder[:10, frame 0]: {encoded_t[0, 0, :10].tolist()}")
        if encoded_t.shape[1] > 1:
            print(f"Encoder[:10, frame 1]: {encoded_t[0, 1, :10].tolist()}")

        # Initialize decoder with warmup (add_sos=True)
        print(f"\n=== Decoder Initialization (add_sos=True) ===")
        pred_result = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
        pred_out, state = pred_result[0], pred_result[1]
        pred_out = pred_out[:, -1:, :]  # Take last timestep
        print(f"Pred output shape: {list(pred_out.shape)}")
        print(f"Pred output[:10]: {pred_out[0, 0, :10].tolist()}")
        print(f"State hidden[0,0,:5]: {state[0][0, 0, :5].tolist()}")

        # Decode first frame
        print(f"\n=== Decoding First Frame ===")
        enc_frame = encoded_t[:, 0:1, :]

        # Run joint network
        enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
        pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out
        enc_proj_exp = enc_proj.unsqueeze(2)
        pred_proj_exp = pred_proj.unsqueeze(1)
        combined = enc_proj_exp + pred_proj_exp
        joint_out = joint.joint_net(combined)
        logits = joint_out[0, 0, 0, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        print(f"Enc_proj[:10]: {enc_proj[0, 0, :10].tolist()}")
        print(f"Pred_proj[:10]: {pred_proj[0, 0, :10].tolist()}")
        print(f"Log_probs[:5]: {log_probs[:5].tolist()}")
        print(f"Blank log_prob: {log_probs[blank_id].item():.6f}")

        best_token = torch.argmax(log_probs).item()
        print(f"Best token: {best_token}, log_prob: {log_probs[best_token].item():.6f}")

        if best_token == blank_id:
            print("=> BLANK")
        else:
            print(f"=> TOKEN {best_token}")
            if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
                text = asr_model.tokenizer.ids_to_text([best_token])
                print(f"   Text: '{text}'")

        # Continue decoding from first frame like RNN-T streaming
        print(f"\n=== Continuing RNN-T Inner Loop ===")
        tokens_emitted = [best_token] if best_token != blank_id else []
        max_symbols = 10
        current_pred = pred_out
        current_state = state

        while best_token != blank_id and len(tokens_emitted) < max_symbols:
            # Update decoder with emitted token
            y_new = torch.tensor([[best_token]], dtype=torch.long)
            new_pred_result = decoder.predict(y=y_new, state=current_state, add_sos=False)
            current_pred = new_pred_result[0]
            current_state = new_pred_result[1]

            # Run joint with same encoder frame
            enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
            pred_proj = joint.pred(current_pred) if joint.pred is not None else current_pred
            enc_proj_exp = enc_proj.unsqueeze(2)
            pred_proj_exp = pred_proj.unsqueeze(1)
            combined = enc_proj_exp + pred_proj_exp
            joint_out = joint.joint_net(combined)
            logits = joint_out[0, 0, 0, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            best_token = torch.argmax(log_probs).item()
            best_prob = log_probs[best_token].item()

            if best_token == blank_id:
                print(f"  Token {len(tokens_emitted)+1}: BLANK (prob={best_prob:.6f}) - stop")
            else:
                tokens_emitted.append(best_token)
                if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
                    text = asr_model.tokenizer.ids_to_text([best_token])
                    print(f"  Token {len(tokens_emitted)}: {best_token} '{text}' (prob={best_prob:.6f})")
                else:
                    print(f"  Token {len(tokens_emitted)}: {best_token} (prob={best_prob:.6f})")

        print(f"\nTotal tokens from frame 0: {len(tokens_emitted)}")
        if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
            full_text = asr_model.tokenizer.ids_to_text(tokens_emitted)
            print(f"Text: '{full_text}'")


if __name__ == '__main__':
    main()
