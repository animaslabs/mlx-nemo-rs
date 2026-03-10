#!/usr/bin/env python3
"""
Statistical comparison of NeMo vs Rust intermediate outputs.
"""

import torch
import numpy as np
import soundfile as sf
import nemo.collections.asr as nemo_asr

# Load audio
data, sr = sf.read('../samples/2086-149220-0033.wav', dtype='float32')
waveform = torch.tensor(data).unsqueeze(0)
lengths = torch.tensor([len(data)])

print("=" * 60)
print("  NeMo Reference Statistics")
print("=" * 60)

# Load model
model = nemo_asr.models.ASRModel.restore_from(
    '../models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo',
    map_location='cpu'
)
model.eval()

# Storage for intermediate outputs
intermediate = {}

def preprocessor_hook(module, input, output):
    intermediate['mel'], intermediate['mel_len'] = output[0].clone(), output[1].clone()

def encoder_hook(module, input, output):
    intermediate['encoder'], intermediate['encoder_len'] = output[0].clone(), output[1].clone()

def decoder_hook(module, input, output):
    intermediate['logits'] = output.clone()

# Register hooks
h1 = model.preprocessor.register_forward_hook(preprocessor_hook)
h2 = model.encoder.register_forward_hook(encoder_hook)
h3 = model.decoder.register_forward_hook(decoder_hook)

with torch.no_grad():
    transcriptions = model.transcribe([str('../samples/2086-149220-0033.wav')], batch_size=1)

h1.remove()
h2.remove()
h3.remove()

def print_stats(name, tensor):
    arr = tensor.detach().cpu().numpy()
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Min: {arr.min():.6f}")
    print(f"  Max: {arr.max():.6f}")
    print(f"  Mean: {arr.mean():.6f}")
    print(f"  Std: {arr.std():.6f}")
    print(f"  Median: {np.median(arr):.6f}")
    
    # Check for specific positions
    if len(arr.shape) == 3:
        # First few values at [0, 0, :5]
        print(f"  [0,0,:5]: {arr[0, 0, :5].tolist()}")
        # Middle position
        mid_c = arr.shape[1] // 2
        print(f"  [0,{mid_c},:5]: {arr[0, mid_c, :5].tolist()}")

print_stats("Mel Spectrogram", intermediate['mel'])
print_stats("Encoder Output", intermediate['encoder'])
print_stats("Decoder Logits", intermediate['logits'])

print(f"\nTranscription: \"{transcriptions[0]}\"")

# Also print the per-channel statistics for mel
mel = intermediate['mel'].squeeze().numpy()
print("\nMel per-channel (mel bin) statistics:")
print(f"  Channel means range: [{mel.mean(axis=1).min():.4f}, {mel.mean(axis=1).max():.4f}]")
print(f"  Channel stds range: [{mel.std(axis=1).min():.4f}, {mel.std(axis=1).max():.4f}]")

# Encoder per-channel stats
enc = intermediate['encoder'].squeeze().numpy()
print("\nEncoder per-channel (d_model dim) statistics:")
print(f"  Channel means range: [{enc.mean(axis=1).min():.4f}, {enc.mean(axis=1).max():.4f}]")
print(f"  Channel stds range: [{enc.std(axis=1).min():.4f}, {enc.std(axis=1).max():.4f}]")

