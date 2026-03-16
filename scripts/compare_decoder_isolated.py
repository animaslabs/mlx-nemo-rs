#!/usr/bin/env python3
"""
Test decoder in isolation with known inputs.
This compares the LSTM decoder and joint network output given the same inputs.
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
print(f"Pred hidden: {decoder.pred_hidden}")

print("\n=== Test 1: LSTM with zeros input and zeros state ===")
print("Input: blank token (embedding = zeros)")
print("State: h=zeros, c=zeros")

with torch.no_grad():
    embed = decoder.prediction['embed']
    rnn = decoder.prediction['dec_rnn'].lstm

    # Blank embedding
    blank_input = torch.tensor([[blank_id]], dtype=torch.long)
    embedded = embed(blank_input)
    print(f"Embedded[:5]: {embedded[0, 0, :5].tolist()}")
    assert all(v == 0 for v in embedded[0, 0].tolist()), "Blank embedding should be zeros!"

    # Run LSTM with zeros state
    h0 = torch.zeros(2, 1, 640)
    c0 = torch.zeros(2, 1, 640)
    lstm_out, (hn, cn) = rnn(embedded, (h0, c0))

    print(f"\nLSTM output (layer 1 hidden):")
    print(f"  [:10]: {lstm_out[0, 0, :10].tolist()}")

    print(f"\nNew hidden state:")
    print(f"  Layer 0 [:10]: {hn[0, 0, :10].tolist()}")
    print(f"  Layer 1 [:10]: {hn[1, 0, :10].tolist()}")

    print(f"\nNew cell state:")
    print(f"  Layer 0 [:10]: {cn[0, 0, :10].tolist()}")
    print(f"  Layer 1 [:10]: {cn[1, 0, :10].tolist()}")

print("\n" + "="*60)
print("=== Test 2: Joint network with known encoder input ===")
print("="*60)

with torch.no_grad():
    # Create a known encoder output (use simple values for reproducibility)
    # This represents a single encoder frame
    enc_frame = torch.zeros(1, 1, 1024)
    enc_frame[0, 0, :5] = torch.tensor([-0.04, -0.03, 0.06, -0.02, -0.04])

    # Pred output from LSTM with zeros (from Test 1)
    pred_out = lstm_out  # [1, 1, 640]

    print(f"Encoder frame[:5]: {enc_frame[0, 0, :5].tolist()}")
    print(f"Pred output[:5]: {pred_out[0, 0, :5].tolist()}")

    # Project through joint network
    enc_proj = joint.enc(enc_frame) if joint.enc is not None else enc_frame
    pred_proj = joint.pred(pred_out) if joint.pred is not None else pred_out

    print(f"\nEnc projection[:5]: {enc_proj[0, 0, :5].tolist()}")
    print(f"Pred projection[:5]: {pred_proj[0, 0, :5].tolist()}")

    # Compute enc_proj and pred_proj norms
    enc_norm = torch.norm(enc_proj).item()
    pred_norm = torch.norm(pred_proj).item()
    print(f"\nEnc projection L2 norm: {enc_norm:.6f}")
    print(f"Pred projection L2 norm: {pred_norm:.6f}")
    print(f"Ratio (enc/pred): {enc_norm/pred_norm:.6f}")

    # Combine
    combined = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
    print(f"\nCombined (before ReLU)[:5]: {combined[0, 0, 0, :5].tolist()}")

    activated = torch.relu(combined)
    print(f"After ReLU[:5]: {activated[0, 0, 0, :5].tolist()}")

    # Joint output
    joint_linear = None
    for module in joint.joint_net.modules():
        if isinstance(module, torch.nn.Linear):
            joint_linear = module
            break

    joint_out = joint_linear(activated)
    log_probs = torch.log_softmax(joint_out, dim=-1)

    print(f"\nLogits[:5]: {joint_out[0, 0, 0, :5].tolist()}")
    print(f"Log probs[:5]: {log_probs[0, 0, 0, :5].tolist()}")
    print(f"Log prob[blank={blank_id}]: {log_probs[0, 0, 0, blank_id].item():.6f}")

    best_token = torch.argmax(log_probs).item()
    print(f"\nBest token: {best_token}")
    print(f"Log prob: {log_probs[0, 0, 0, best_token].item():.6f}")

print("\n" + "="*60)
print("=== Expected Values for Rust ===")
print("="*60)
print("""
These values should match in Rust when running:
1. Blank embedding through LSTM with zeros state
2. Joint network with zeros encoder input and LSTM output from step 1

LSTM output (first 10 values):
""")
print(f"  {lstm_out[0, 0, :10].tolist()}")
print(f"""
Hidden state layer 0 (first 10 values):
  {hn[0, 0, :10].tolist()}
Hidden state layer 1 (first 10 values):
  {hn[1, 0, :10].tolist()}
Cell state layer 0 (first 10 values):
  {cn[0, 0, :10].tolist()}
Cell state layer 1 (first 10 values):
  {cn[1, 0, :10].tolist()}

With zeros encoder input, joint network should predict: {best_token}
""")
