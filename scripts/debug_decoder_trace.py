#!/usr/bin/env python3
"""
Trace RNNT decoder (prediction network + joint) step by step.

This script traces through the decoder components to compare with Rust:
1. Prediction Network (LSTM) - produces decoder hidden states
2. Joint Network - combines encoder + decoder outputs
3. Greedy decoding - selects tokens
"""

import os, warnings
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

# Load config
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
print("Building pipeline...")
pipeline = PipelineBuilder.build_pipeline(cfg)

# Get model components
asr_model = pipeline.asr_model.asr_model
encoder = asr_model.encoder
decoder = asr_model.decoder  # Prediction network (RNNTDecoder)
joint = asr_model.joint

print("\n=== Decoder (Prediction Network) Configuration ===")
print(f"Type: {type(decoder).__name__}")
print(f"Vocab size: {decoder.vocab_size}")
print(f"Hidden size: {decoder.pred_hidden}")
print(f"Num layers: {decoder.pred_rnn_layers}")
blank_id = decoder.vocab_size
print(f"Blank token ID: {blank_id}")

print("\n=== Joint Network Configuration ===")
print(f"Type: {type(joint).__name__}")
print(f"Encoder hidden: {joint.encoder_hidden}")
print(f"Pred hidden: {joint.pred_hidden}")
print(f"Joint hidden: {joint.joint_hidden}")
print(f"Num classes (incl blank): {joint.num_classes_with_blank}")

# Load audio and process first chunk through encoder
audio_path = '../samples/2086-149220-0033.wav'
audio, sr = sf.read(audio_path, dtype='float32')
preprocessor = asr_model.preprocessor

# Process a chunk
chunk_samples = 19360
chunk = audio[:chunk_samples]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

print("\n=== Processing First Chunk ===")
with torch.no_grad():
    # Get mel spectrogram
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    print(f"MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")

    # Run encoder
    encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)
    print(f"Encoded shape: {list(encoded.shape)}, encoded_len: {encoded_len.item()}")

    # Transpose to [B, T, D]
    encoded_t = encoded.transpose(1, 2)
    print(f"Encoded transposed shape: {list(encoded_t.shape)}")

    # Show encoder output for first few frames
    print(f"\nEncoder output (first 3 frames, first 5 dims):")
    for t in range(min(3, encoded_t.shape[1])):
        vals = encoded_t[0, t, :5].tolist()
        print(f"  T{t:02d}: [{', '.join(f'{v:.6f}' for v in vals)}]")

print("\n\n=== Tracing Greedy RNNT Decoding ===")

# The decoder.predict() function signature from NeMo:
# def predict(self, y: Optional[torch.Tensor], state: Optional[Tuple[torch.Tensor, torch.Tensor]], add_sos: bool = True) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
# y: [B, U] token indices (or None for initial call with add_sos=True)
# state: tuple of (hidden, cell) each [num_layers, B, hidden_size]
# returns: (g, hid, y) where g is [B, U, hidden_size]

with torch.no_grad():
    # Initialize: call predict with y=None, add_sos=True to get initial prediction
    # This uses the blank/sos embedding
    print("Initial decoder call (y=None, add_sos=True):")
    result = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
    print(f"predict() returned {len(result)} values")
    if len(result) == 2:
        pred_out_full, state = result
    else:
        pred_out_full, state, _ = result
    print(f"  Pred output full shape: {list(pred_out_full.shape)}")
    # When add_sos=True, output is [B, 2, hidden] - take last timestep for decoding
    pred_out = pred_out_full[:, -1:, :]  # [B, 1, hidden]
    print(f"  Pred output (last timestep) shape: {list(pred_out.shape)}")
    print(f"  Pred output [:5]: {pred_out[0, 0, :5].tolist()}")
    print(f"  State hidden shape: {state[0].shape}")
    print(f"  State cell shape: {state[1].shape}")
    print(f"  State hidden[0,0,:5]: {state[0][0, 0, :5].tolist()}")
    print(f"  State cell[0,0,:5]: {state[1][0, 0, :5].tolist()}")

    # Now trace frame by frame
    last_token = None  # Will use blank initially
    tokens_emitted = []

    num_frames_to_process = min(8, encoded_t.shape[1])

    for frame_idx in range(num_frames_to_process):
        print(f"\n--- Frame {frame_idx} ---")

        # Get encoder output for this frame: [B, 1, D]
        enc_frame = encoded_t[:, frame_idx:frame_idx+1, :]
        print(f"Encoder frame shape: {list(enc_frame.shape)}")
        print(f"Encoder frame [:5]: {enc_frame[0, 0, :5].tolist()}")
        print(f"Pred_out shape: {list(pred_out.shape)}")

        # Check joint network encoder projection
        if hasattr(joint, 'enc') and joint.enc is not None:
            print(f"Joint encoder proj weight shape: {list(joint.enc.weight.shape)}")
        if hasattr(joint, 'pred') and joint.pred is not None:
            print(f"Joint pred proj weight shape: {list(joint.pred.weight.shape)}")

        # Run joint network manually to debug
        # Joint expects encoder [B, T, D] and pred [B, U, D]
        # First project encoder and pred, then combine
        enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
        print(f"Enc projected shape: {list(enc_proj.shape)}")
        pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out
        print(f"Pred projected shape: {list(pred_proj.shape)}")

        # Joint after projection does: enc + pred (with broadcasting) -> activation -> linear
        # enc_proj: [B, T, joint_hidden] -> [B, T, 1, joint_hidden]
        # pred_proj: [B, U, joint_hidden] -> [B, 1, U, joint_hidden]
        # sum: [B, T, U, joint_hidden]
        enc_proj_exp = enc_proj.unsqueeze(2)  # [B, T, 1, D]
        pred_proj_exp = pred_proj.unsqueeze(1)  # [B, 1, U, D]
        combined = enc_proj_exp + pred_proj_exp  # [B, T, U, D]
        print(f"Combined shape: {list(combined.shape)}")

        # Apply joint_net (ReLU -> Dropout -> Linear)
        joint_out = joint.joint_net(combined)
        print(f"Joint output shape: {list(joint_out.shape)}")
        print(f"Joint output shape: {list(joint_out.shape)}")

        # Get logits [B, T, U, vocab] -> squeeze to [vocab]
        logits = joint_out[0, 0, 0, :]  # [vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
        print(f"Logits [:5]: {logits[:5].tolist()}")
        print(f"Log probs [:5]: {log_probs[:5].tolist()}")
        print(f"Log prob[blank={blank_id}]: {log_probs[blank_id].item():.6f}")

        # Greedy: select best token
        best_token = torch.argmax(log_probs).item()
        best_prob = log_probs[best_token].item()
        print(f"Best token: {best_token}, log_prob: {best_prob:.6f}")

        if best_token == blank_id:
            print(f"  -> BLANK (no emission, no state update)")
            # Don't update state or pred_out for blank
        else:
            print(f"  -> Emitting token {best_token}")
            tokens_emitted.append(best_token)

            # Update decoder with new token
            y_new = torch.tensor([[best_token]], dtype=torch.long)
            pred_out, state = decoder.predict(y=y_new, state=state, add_sos=False)[:2]
            print(f"  New pred output [:5]: {pred_out[0, 0, :5].tolist()}")
            print(f"  New state hidden[0,0,:5]: {state[0][0, 0, :5].tolist()}")

            # Inner loop: keep emitting while not blank
            inner_loop = 0
            max_inner = 5
            while inner_loop < max_inner:
                inner_loop += 1

                # Re-run joint with same encoder frame, new pred_out (manual computation)
                enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
                pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out
                enc_proj_exp = enc_proj.unsqueeze(2)
                pred_proj_exp = pred_proj.unsqueeze(1)
                combined = enc_proj_exp + pred_proj_exp
                joint_out = joint.joint_net(combined)

                logits = joint_out[0, 0, 0, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                best_token = torch.argmax(log_probs).item()
                print(f"  [Inner {inner_loop}] Best token: {best_token}, log_prob: {log_probs[best_token].item():.6f}")

                if best_token == blank_id:
                    print(f"    -> BLANK (stop inner loop)")
                    break
                else:
                    print(f"    -> Emitting token {best_token}")
                    tokens_emitted.append(best_token)

                    y_new = torch.tensor([[best_token]], dtype=torch.long)
                    pred_out, state = decoder.predict(y=y_new, state=state, add_sos=False)[:2]

print(f"\n\n=== Summary ===")
print(f"Tokens emitted: {tokens_emitted}")

# Decode tokens to text
if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
    text = asr_model.tokenizer.ids_to_text(tokens_emitted)
    print(f"Decoded text: '{text}'")


print("\n\n=== Key Component Values for Rust Comparison ===")

# Embedding weights
embed = decoder.prediction['embed']
print(f"\n--- Embedding ---")
print(f"Shape: {list(embed.weight.shape)}")
print(f"Weight[0, :5]: {embed.weight[0, :5].tolist()}")
print(f"Weight[1, :5]: {embed.weight[1, :5].tolist()}")
print(f"Weight[blank={blank_id}, :5]: {embed.weight[blank_id, :5].tolist()}")

# LSTM weights
rnn = decoder.prediction['dec_rnn'].lstm
print(f"\n--- LSTM Layer 0 ---")
print(f"weight_ih_l0 shape: {list(rnn.weight_ih_l0.shape)}")
print(f"weight_ih_l0[0, :5]: {rnn.weight_ih_l0[0, :5].tolist()}")
print(f"weight_hh_l0 shape: {list(rnn.weight_hh_l0.shape)}")
print(f"weight_hh_l0[0, :5]: {rnn.weight_hh_l0[0, :5].tolist()}")
print(f"bias_ih_l0[:5]: {rnn.bias_ih_l0[:5].tolist()}")
print(f"bias_hh_l0[:5]: {rnn.bias_hh_l0[:5].tolist()}")

# Joint network
print(f"\n--- Joint Network ---")
# The joint network may have encoder_proj, pred_proj, and joint_net
if joint.encoder_proj is not None:
    print(f"encoder_proj weight shape: {list(joint.encoder_proj.weight.shape)}")
    print(f"encoder_proj weight[0, :5]: {joint.encoder_proj.weight[0, :5].tolist()}")
else:
    print("encoder_proj: None")

if joint.pred_proj is not None:
    print(f"pred_proj weight shape: {list(joint.pred_proj.weight.shape)}")
    print(f"pred_proj weight[0, :5]: {joint.pred_proj.weight[0, :5].tolist()}")
else:
    print("pred_proj: None")

# joint_net is Sequential: ReLU -> Dropout -> Linear
for name, module in joint.joint_net.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"joint_net Linear weight shape: {list(module.weight.shape)}")
        print(f"joint_net Linear weight[0, :5]: {module.weight[0, :5].tolist()}")
        print(f"joint_net Linear bias[:5]: {module.bias[:5].tolist()}")
