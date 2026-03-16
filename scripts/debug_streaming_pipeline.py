#!/usr/bin/env python3
"""
Debug NeMo CacheAwareRNNTPipeline to compare with Rust implementation.

This script traces through the NeMo streaming pipeline step-by-step to
capture intermediate values for comparison with the Rust implementation.
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
cfg.streaming.att_context_size = [70, 1]  # 80ms lookahead
cfg.asr_decoding_type = 'rnnt'
cfg.log_level = 40

# Build pipeline
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
print("Building pipeline...")
pipeline = PipelineBuilder.build_pipeline(cfg)

# Get the underlying ASR model and components
asr_model = pipeline.asr_model
encoder = asr_model.asr_model.encoder
preprocessor = asr_model.asr_model.preprocessor

print("\n=== Encoder Configuration ===")
print(f"d_model: {encoder.d_model}")
print(f"n_layers: {encoder.n_layers}")
print(f"subsampling_factor: {encoder.subsampling_factor}")
print(f"att_context_size: {encoder.att_context_size}")
print(f"att_context_style: {encoder.att_context_style}")

# Get streaming params
print("\n=== Streaming Configuration ===")
# NeMo stores streaming params in the encoder after setup
if hasattr(encoder, 'streaming_params'):
    sp = encoder.streaming_params
    print(f"model_chunk_size: {sp.get('model_chunk_size', 'N/A')}")
    print(f"shift_size: {sp.get('shift_size', 'N/A')}")
    print(f"pre_encode_cache_size: {sp.get('pre_encode_cache_size', 'N/A')}")
    print(f"drop_extra_pre_encoded: {sp.get('drop_extra_pre_encoded', 'N/A')}")
    print(f"cache_drop_size: {sp.get('cache_drop_size', 'N/A')}")
    print(f"last_channel_cache_size: {sp.get('last_channel_cache_size', 'N/A')}")
    print(f"last_time_cache_size: {sp.get('last_time_cache_size', 'N/A')}")
else:
    print("No streaming_params found on encoder")

# Check asr_model for streaming config
if hasattr(asr_model, 'streaming_cfg'):
    sc = asr_model.streaming_cfg
    print(f"\nasr_model.streaming_cfg:")
    print(f"  chunk_size: {getattr(sc, 'chunk_size', 'N/A')}")
    print(f"  shift_size: {getattr(sc, 'shift_size', 'N/A')}")

# Get the cache-aware pipeline's internal config
print("\n=== Pipeline Configuration ===")
if hasattr(pipeline, 'streaming_cfg'):
    sc = pipeline.streaming_cfg
    for k in dir(sc):
        if not k.startswith('_'):
            v = getattr(sc, k)
            if not callable(v):
                print(f"  {k}: {v}")

# Load test audio
audio_path = '../samples/2086-149220-0033.wav'
audio, sr = sf.read(audio_path, dtype='float32')
print(f"\n=== Audio Info ===")
print(f"Path: {audio_path}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f}s ({len(audio)} samples)")

# Process audio through the pipeline manually to trace values
print("\n=== Processing First Chunk ===")

# Manually compute mel spectrogram for first chunk
chunk_samples = 19360  # First chunk size matching debug scripts
chunk = audio[:chunk_samples]
chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
chunk_len = torch.tensor([len(chunk)])

with torch.no_grad():
    mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
    print(f"MEL shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
    print(f"MEL frames: {mel.shape[2]}")

    # Show MEL values at frame boundaries
    print(f"\nMEL T00 [:5]: {mel[0, :5, 0].tolist()}")
    print(f"MEL T-1 [:5]: {mel[0, :5, -1].tolist()}")

# Now trace through the streaming pipeline itself
print("\n=== Tracing CacheAwareRNNTPipeline ===")

# Get the request generator and set up for streaming
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
request_generator = pipeline.get_request_generator()
options = [ASRRequestOptions()]
request_generator.set_audio_filepaths([audio_path], options)

# Open session
pipeline.open_session()

# Process chunks and capture intermediate values
chunk_idx = 0
print(f"\nProcessing audio in streaming chunks...")

for requests in request_generator:
    print(f"\n--- Chunk {chunk_idx} ---")

    # Inspect the request
    for i, req in enumerate(requests):
        if hasattr(req, 'features'):
            print(f"  Request {i}: features shape = {req.features.shape if hasattr(req.features, 'shape') else 'N/A'}")
        if hasattr(req, 'is_first'):
            print(f"  Request {i}: is_first = {req.is_first}")
        if hasattr(req, 'is_last'):
            print(f"  Request {i}: is_last = {req.is_last}")

    # Get step outputs
    step_outputs = pipeline.transcribe_step(requests)

    for step_output in step_outputs:
        partial = step_output.partial_transcript.strip() if step_output.partial_transcript else ""
        final = step_output.final_transcript.strip() if step_output.final_transcript else ""

        print(f"  Partial: '{partial}'")
        if final:
            print(f"  Final: '{final}'")

    chunk_idx += 1

    # Limit output for debugging
    if chunk_idx > 5:
        print(f"\n... (stopping after 5 chunks for debugging)")
        break

pipeline.close_session()

# Now let's look at what the encoder actually does in streaming mode
print("\n\n=== Deep Dive: Encoder Streaming Behavior ===")

# Reset state and manually trace encoder
if hasattr(encoder, 'reset_streaming_state'):
    encoder.reset_streaming_state()

# First, let's understand the mel buffer/chunking
print("\nNeMo Mel Chunking Configuration:")

# The key config values come from setup_streaming_params
# Let's compute them ourselves to compare with Rust
subsampling_factor = 8
n_layers = 24
conv_kernel = 9  # From config
right_context = 1  # From att_context_size[1]
left_context = 70  # From att_context_size[0]
att_context_style = "chunked_limited"

# Compute streaming params like NeMo does
if att_context_style == "chunked_limited":
    lookahead_steps = max(right_context, 0)
    cache_drop_size = 0
else:
    conv_context_right = 0 if True else (conv_kernel - 1) // 2  # causal conv
    lookahead_steps = right_context * n_layers + conv_context_right * n_layers
    cache_drop_size = lookahead_steps

print(f"\nComputed streaming params:")
print(f"  lookahead_steps: {lookahead_steps}")
print(f"  cache_drop_size: {cache_drop_size}")

# Sampling frames from dw_striding subsampling: [1, subsampling_factor]
sampling_frames = (1, subsampling_factor)

# Chunk size (mel frames)
chunk_size_first = sampling_frames[0] + subsampling_factor * lookahead_steps
chunk_size_subseq = sampling_frames[1] + subsampling_factor * lookahead_steps
print(f"  chunk_size: ({chunk_size_first}, {chunk_size_subseq})")

# Shift size (mel frames)
shift_first = sampling_frames[0] + sampling_frames[1] * (lookahead_steps - cache_drop_size)
shift_subseq = sampling_frames[1] + sampling_frames[1] * (lookahead_steps - cache_drop_size)
print(f"  shift_size: ({shift_first}, {shift_subseq})")

# Pre-encode cache (for dw_striding: [0, subsampling_factor + 1])
pre_encode_cache = (0, subsampling_factor + 1)
print(f"  pre_encode_cache_size: {pre_encode_cache}")

# Drop extra pre-encoded
drop_extra = 1 + (pre_encode_cache[1] - 1) // subsampling_factor if pre_encode_cache[1] >= 1 else 0
print(f"  drop_extra_pre_encoded: {drop_extra}")

# Valid output length
valid_out_len = (shift_subseq - sampling_frames[1]) // subsampling_factor + 1 if sampling_frames[1] > 0 else shift_subseq // subsampling_factor
print(f"  valid_out_len: {valid_out_len}")

# Cache sizes
last_channel_cache = left_context if left_context >= 0 else 10000  # max_context
last_time_cache = conv_kernel - 1
print(f"  last_channel_cache_size: {last_channel_cache}")
print(f"  last_time_cache_size: {last_time_cache}")

print("\n\nThese values should match the Rust StreamingConfig!")
print("Compare with: cargo run --example streaming -- --model path.nemo --list-modes")
