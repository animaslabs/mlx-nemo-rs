#!/usr/bin/env python3
"""
Compare LSTM decoder first step between Python and Rust.

This script outputs the exact values from Python that should match Rust.
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

asr_model = pipeline.asr_model.asr_model
decoder = asr_model.decoder
joint = asr_model.joint

blank_id = decoder.vocab_size
print(f"Blank ID: {blank_id}")
print(f"Vocab size: {blank_id + 1}")
print(f"Pred hidden: {decoder.pred_hidden}")
print(f"Num LSTM layers: {decoder.pred_rnn_layers}")

# Get embedding and LSTM weights
embed = decoder.prediction['embed']
rnn = decoder.prediction['dec_rnn'].lstm

print("\n=== Embedding Weights ===")
print(f"Shape: {list(embed.weight.shape)}")
print(f"Blank embedding[0:10]: {embed.weight[blank_id, 0:10].tolist()}")

print("\n=== LSTM Layer 0 Weights ===")
print(f"weight_ih_l0[0, 0:5]: {rnn.weight_ih_l0[0, 0:5].tolist()}")
print(f"weight_hh_l0[0, 0:5]: {rnn.weight_hh_l0[0, 0:5].tolist()}")
print(f"bias_ih_l0[0:5]: {rnn.bias_ih_l0[0:5].tolist()}")
print(f"bias_hh_l0[0:5]: {rnn.bias_hh_l0[0:5].tolist()}")

# Manual first step computation
print("\n\n=== Manual First Step Computation ===")
print("Input: blank token embedding (should be zeros)")

with torch.no_grad():
    # Embed blank token
    blank_input = torch.tensor([[blank_id]], dtype=torch.long)
    embedded = embed(blank_input)  # [1, 1, 640]
    print(f"Embedded shape: {list(embedded.shape)}")
    print(f"Embedded[0,0,0:10]: {embedded[0, 0, 0:10].tolist()}")

    # Initialize LSTM state (zeros)
    h0 = torch.zeros(2, 1, 640)  # [num_layers, batch, hidden]
    c0 = torch.zeros(2, 1, 640)

    # Run through LSTM manually for layer 0
    x = embedded.squeeze(0)  # [1, 640]
    h = h0[0]  # [1, 640]
    c = c0[0]  # [1, 640]

    # LSTM cell: gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
    print("\n--- Layer 0 Manual LSTM ---")
    gates_ih = x @ rnn.weight_ih_l0.T + rnn.bias_ih_l0
    gates_hh = h @ rnn.weight_hh_l0.T + rnn.bias_hh_l0
    gates = gates_ih + gates_hh
    print(f"gates_ih[0,0:5]: {gates_ih[0, 0:5].tolist()}")
    print(f"gates_hh[0,0:5]: {gates_hh[0, 0:5].tolist()}")
    print(f"gates[0,0:5]: {gates[0, 0:5].tolist()}")

    # Split into 4 gates (i, f, g, o)
    hidden_size = 640
    i_gate = torch.sigmoid(gates[:, 0:hidden_size])
    f_gate = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
    g_gate = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
    o_gate = torch.sigmoid(gates[:, 3*hidden_size:])

    print(f"i_gate[0,0:5]: {i_gate[0, 0:5].tolist()}")
    print(f"f_gate[0,0:5]: {f_gate[0, 0:5].tolist()}")
    print(f"g_gate[0,0:5]: {g_gate[0, 0:5].tolist()}")
    print(f"o_gate[0,0:5]: {o_gate[0, 0:5].tolist()}")

    # New cell and hidden
    new_c = f_gate * c + i_gate * g_gate
    new_h = o_gate * torch.tanh(new_c)
    print(f"new_c[0,0:5]: {new_c[0, 0:5].tolist()}")
    print(f"new_h[0,0:5]: {new_h[0, 0:5].tolist()}")

    # Using actual LSTM module
    print("\n--- Using PyTorch LSTM ---")
    rnn.eval()
    lstm_out, (hn, cn) = rnn(embedded, (h0, c0))
    print(f"lstm_out shape: {list(lstm_out.shape)}")
    print(f"lstm_out[0,0,0:5]: {lstm_out[0, 0, 0:5].tolist()}")
    print(f"hn[0,0,0:5]: {hn[0, 0, 0:5].tolist()}")
    print(f"hn[1,0,0:5]: {hn[1, 0, 0:5].tolist()}")
    print(f"cn[0,0,0:5]: {cn[0, 0, 0:5].tolist()}")
    print(f"cn[1,0,0:5]: {cn[1, 0, 0:5].tolist()}")

    # Compare with decoder.predict
    print("\n--- Using decoder.predict ---")
    result = decoder.predict(y=None, state=None, add_sos=True, batch_size=1)
    pred_out, state = result[0], result[1]
    print(f"pred_out shape: {list(pred_out.shape)}")
    # pred_out has shape [1, 2, 640] because add_sos=True prepends SOS
    print(f"pred_out[0,0,0:10] (SOS): {pred_out[0, 0, 0:10].tolist()}")
    print(f"pred_out[0,1,0:10] (after SOS): {pred_out[0, 1, 0:10].tolist()}")
    print(f"state hidden[0,0,0:10]: {state[0][0, 0, 0:10].tolist()}")
    print(f"state hidden[1,0,0:10]: {state[0][1, 0, 0:10].tolist()}")
    print(f"state cell[0,0,0:10]: {state[1][0, 0, 0:10].tolist()}")
    print(f"state cell[1,0,0:10]: {state[1][1, 0, 0:10].tolist()}")

# Now compute joint output with first encoder frame
print("\n\n=== Joint Network First Step ===")

# Load audio and get encoder output
audio_path = '../samples/2086-149220-0033.wav'
audio, sr = sf.read(audio_path, dtype='float32')
preprocessor = asr_model.preprocessor
encoder = asr_model.encoder

chunk_samples = 19360
chunk = audio[:chunk_samples]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)
    encoded_t = encoded.transpose(1, 2)  # [B, T, D]

print(f"Encoded shape: {list(encoded_t.shape)}")
print(f"Encoder frame 0 [0:10]: {encoded_t[0, 0, 0:10].tolist()}")

with torch.no_grad():
    # Get prediction output (last timestep from add_sos=True call)
    pred_for_joint = pred_out[:, -1:, :]  # [1, 1, 640]
    print(f"pred_for_joint shape: {list(pred_for_joint.shape)}")
    print(f"pred_for_joint[0,0,0:10]: {pred_for_joint[0, 0, 0:10].tolist()}")

    # Get first encoder frame
    enc_frame = encoded_t[:, 0:1, :]  # [1, 1, 1024]
    print(f"enc_frame shape: {list(enc_frame.shape)}")

    # Project encoder through joint.enc
    enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
    print(f"enc_proj shape: {list(enc_proj.shape)}")
    print(f"enc_proj[0,0,0:10]: {enc_proj[0, 0, 0:10].tolist()}")

    # Project pred through joint.pred
    pred_proj = joint.pred(pred_for_joint) if joint.pred is not None else pred_for_joint
    print(f"pred_proj shape: {list(pred_proj.shape)}")
    print(f"pred_proj[0,0,0:10]: {pred_proj[0, 0, 0:10].tolist()}")

    # Combine: [1, 1, 1, 640]
    enc_exp = enc_proj.unsqueeze(2)
    pred_exp = pred_proj.unsqueeze(1)
    combined = enc_exp + pred_exp
    print(f"combined shape: {list(combined.shape)}")
    print(f"combined[0,0,0,0:10]: {combined[0, 0, 0, 0:10].tolist()}")

    # Apply ReLU and output projection
    activated = torch.relu(combined)
    print(f"After ReLU[0,0,0,0:10]: {activated[0, 0, 0, 0:10].tolist()}")

    # Get joint output layer
    joint_linear = None
    for module in joint.joint_net.modules():
        if isinstance(module, torch.nn.Linear):
            joint_linear = module
            break

    if joint_linear:
        joint_out = joint_linear(activated)
        print(f"joint_out shape: {list(joint_out.shape)}")
        print(f"joint_out logits[0,0,0,0:5]: {joint_out[0, 0, 0, 0:5].tolist()}")
        print(f"joint_out logits[0,0,0,-5:]: {joint_out[0, 0, 0, -5:].tolist()}")
        print(f"joint_out logits[0,0,0,blank={blank_id}]: {joint_out[0, 0, 0, blank_id].item()}")

        # Log softmax
        log_probs = torch.log_softmax(joint_out, dim=-1)
        print(f"log_probs[0,0,0,0:5]: {log_probs[0, 0, 0, 0:5].tolist()}")
        print(f"log_probs[0,0,0,blank={blank_id}]: {log_probs[0, 0, 0, blank_id].item()}")

        # Argmax
        best_token = torch.argmax(log_probs).item()
        print(f"Best token: {best_token}, log_prob: {log_probs[0, 0, 0, best_token].item():.6f}")

print("\n=== Expected values for Rust comparison ===")
print(f"1. Blank embedding should be all zeros")
print(f"2. First LSTM output (layer 0) should match: {new_h[0, 0:5].tolist()}")
print(f"3. Full LSTM output should match: {lstm_out[0, 0, 0:5].tolist()}")
print(f"4. Encoder projection should match: {enc_proj[0, 0, 0:5].tolist()}")
print(f"5. Pred projection should match: {pred_proj[0, 0, 0:5].tolist()}")
print(f"6. Combined before ReLU should match: {combined[0, 0, 0, 0:5].tolist()}")
print(f"7. Best token should be: {best_token}")
