#!/usr/bin/env python3
"""
Debug MEL frames near the boundary to understand subsampling input.
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
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

audio, sr = sf.read('../samples/2086-149220-0033.wav', dtype='float32')
config_path = Path('streaming_config.yaml')
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

chunk = audio[:19360]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    print(f"MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")

    # Print last several MEL frames before and after masking boundary
    print(f"\nMEL frames near boundary (mel_len={mel_len.item()}):")
    for i in range(115, min(125, mel.shape[2])):
        vals = mel[0, :5, i].tolist()
        is_valid = "VALID" if i < mel_len.item() else "PADDING"
        all_zero = all(abs(v) < 1e-9 for v in mel[0, :, i].tolist())
        zero_status = " [ALL ZEROS]" if all_zero else ""
        print(f"  Frame {i:3d} [{is_valid:7s}]: [{', '.join(f'{v:.6f}' for v in vals)}]{zero_status}")

    # Check if NeMo masks beyond mel_len
    print(f"\nChecking if NeMo masks MEL frames beyond mel_len:")
    mel_frame_121_sum = mel[0, :, 121].sum().item()
    print(f"  Sum of MEL frame 121: {mel_frame_121_sum}")

    # Transpose for encoder: [B, n_mels, T] -> [B, T, n_mels]
    mel_transposed = mel.transpose(1, 2)
    print(f"\nMEL transposed shape: {list(mel_transposed.shape)}")

    # Get the subsampling module
    subsampling = encoder.pre_encode
    print(f"\nSubsampling module: {type(subsampling).__name__}")

    # Check the convolution layers
    print(f"\nSubsampling structure:")
    for name, module in subsampling.named_modules():
        if 'conv' in name.lower() or isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'kernel_size'):
                print(f"  {name}: kernel={module.kernel_size}, stride={module.stride}, padding={module.padding}")

    # Manually trace through the subsampling to see what happens
    print(f"\n=== Tracing through subsampling ===")

    # NeMo's ConvSubsampling.forward
    # Input: [B, T, n_mels] -> [B, 1, T, n_mels]
    x = mel_transposed.unsqueeze(1)
    print(f"Input to subsampling: {list(x.shape)}")

    # Print subsampling conv structure
    print(f"\nSubsampling conv modules:")
    for i, module in enumerate(subsampling.conv):
        print(f"  [{i}] {type(module).__name__}")
        if hasattr(module, 'conv'):
            print(f"      conv: kernel={module.conv.kernel_size}, stride={module.conv.stride}, padding={module.conv.padding}")

    # Run through subsampling and check output
    x_out, lengths_out = subsampling(mel_transposed, mel_len)
    print(f"\nSubsampling output: {list(x_out.shape)}, lengths: {lengths_out.item()}")

    # Print all time positions
    print(f"\nAll subsampling output positions:")
    for t in range(x_out.shape[1]):
        vals = x_out[0, t, :5].tolist()
        is_valid = "VALID" if t < lengths_out.item() else "PADDING"
        print(f"  T{t:02d} [{is_valid:7s}]: [{', '.join(f'{v:.6f}' for v in vals)}]")

    # Check if there's any masking done after convolutions
    print(f"\n=== Checking ConvSubsampling source code ===")
    import inspect
    # Print the forward method
    forward_src = inspect.getsource(subsampling.forward)
    print(forward_src[:2000] if len(forward_src) > 2000 else forward_src)
