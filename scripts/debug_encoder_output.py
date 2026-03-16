#!/usr/bin/env python3
"""
Compare full encoder outputs between Python and Rust.
"""

import os, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

# Load audio
audio_path = sys.argv[1] if len(sys.argv) > 1 else '../samples/2086-149220-0033.wav'
audio, sr = sf.read(audio_path, dtype='float32')
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)
print(f"Audio: {len(audio)} samples at {sr} Hz")

# Build pipeline
config_path = Path(__file__).parent / 'streaming_config.yaml'
cfg = OmegaConf.load(config_path)
cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
cfg.asr.device = 'cpu'
cfg.asr.compute_dtype = 'float32'
cfg.asr.use_amp = False
cfg.streaming.att_context_size = [70, 1]
cfg.asr_decoding_type = 'rnnt'
cfg.log_level = 40

pipeline = PipelineBuilder.build_pipeline(cfg)
preprocessor = pipeline.asr_model.asr_model.preprocessor
encoder = pipeline.asr_model.asr_model.encoder

print(f"\nEncoder config:")
print(f"  d_model: {encoder.d_model}")
print(f"  n_layers: {len(encoder.layers)}")
print(f"  subsampling_factor: {encoder.pre_encode.subsampling_factor}")

# Streaming parameters
HOP_SAMPLES = 160
CHUNK_SIZE = 112
PRE_ENCODE_CACHE = 9
FIRST_CHUNK_AUDIO = (CHUNK_SIZE + PRE_ENCODE_CACHE) * HOP_SAMPLES  # 19360
SUBSEQUENT_CHUNK_AUDIO = CHUNK_SIZE * HOP_SAMPLES  # 17920

print(f"\n=== Processing Audio ===")

audio_offset = 0
encoder_call = 0

while audio_offset < len(audio) and encoder_call < 3:
    # Determine audio needed
    audio_needed = FIRST_CHUNK_AUDIO if encoder_call == 0 else SUBSEQUENT_CHUNK_AUDIO

    # Get audio chunk
    chunk_end = min(audio_offset + audio_needed, len(audio))
    chunk = audio[audio_offset:chunk_end]

    if len(chunk) < HOP_SAMPLES:
        break

    # Pad if needed
    if len(chunk) < audio_needed:
        chunk = np.pad(chunk, (0, audio_needed - len(chunk)))

    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    chunk_len = torch.tensor([len(chunk)])

    # Compute MEL
    with torch.no_grad():
        mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

    # Run full encoder (batch mode, no streaming cache)
    with torch.no_grad():
        encoded, encoded_len = encoder(audio_signal=mel, length=mel_len)

    # encoded shape is [B, D, T] in NeMo
    print(f"\nEncoder call {encoder_call}:")
    print(f"  Audio samples: {len(chunk)}")
    print(f"  MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
    print(f"  Encoded shape: {list(encoded.shape)}, encoded_len: {encoded_len.item()}")

    # Transpose to [B, T, D] for easier comparison
    encoded_t = encoded.transpose(1, 2)
    enc_np = encoded_t[0].numpy()  # [T, D]
    valid_t = encoded_len.item()

    print(f"  Valid time steps: {valid_t}")

    # Statistics
    print(f"  Stats: min={enc_np.min():.6f}, max={enc_np.max():.6f}, mean={enc_np.mean():.6f}, std={enc_np.std():.6f}")

    # Print first few time steps
    print(f"\n  === Valid encoder output ===")
    for t in range(min(valid_t, 8)):
        vals = enc_np[t, :5]
        print(f"  T{t:02d} [0:5]: [{', '.join(f'{v:.6f}' for v in vals)}]")

    if valid_t > 8:
        print(f"  ...")
        # Last valid time step
        t = valid_t - 1
        vals = enc_np[t, :5]
        print(f"  T{t:02d} [0:5]: [{', '.join(f'{v:.6f}' for v in vals)}]")

    audio_offset += audio_needed
    encoder_call += 1

print(f"\n=== Done: {encoder_call} encoder calls ===")
