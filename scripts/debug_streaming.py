#!/usr/bin/env python3
"""
Debug script to understand NeMo's streaming buffer behavior.
Prints detailed information about chunk sizes, mel frames, etc.
"""

import os
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch


def main():
    model_path = "../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
    audio_path = "../samples/Speaker26_004_16k.wav"
    mode = 0

    # Suppress NeMo logging
    import logging

    for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    print(f"Loading model from: {model_path}")
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location="cpu")
    model.eval()

    sample_rate = model.cfg.preprocessor.sample_rate
    print(f"Sample rate: {sample_rate}")

    # Setup streaming mode
    if hasattr(model.encoder, "att_context_size_all"):
        att_context_sizes = model.encoder.att_context_size_all
        att_context_size = att_context_sizes[mode]
        if hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size)
        else:
            model.encoder.att_context_size = att_context_size
            model.encoder.setup_streaming_params()

    streaming_cfg = model.encoder.streaming_cfg

    print("\n=== Streaming Configuration ===")
    print(f"chunk_size: {streaming_cfg.chunk_size}")
    print(f"shift_size: {streaming_cfg.shift_size}")
    print(f"valid_out_len: {streaming_cfg.valid_out_len}")
    print(f"pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print(f"last_channel_cache_size: {streaming_cfg.last_channel_cache_size}")
    if hasattr(streaming_cfg, "last_time_cache_size"):
        print(f"last_time_cache_size: {streaming_cfg.last_time_cache_size}")
    if hasattr(streaming_cfg, "cache_drop_size"):
        print(f"cache_drop_size: {streaming_cfg.cache_drop_size}")

    # Get preprocessor info
    preproc_cfg = model.cfg.preprocessor
    print("\n=== Preprocessor Configuration ===")
    print(
        f"window_size: {preproc_cfg.window_size} ({preproc_cfg.window_size * sample_rate:.0f} samples)"
    )
    print(
        f"window_stride: {preproc_cfg.window_stride} ({preproc_cfg.window_stride * sample_rate:.0f} samples)"
    )
    print(f"features: {preproc_cfg.features}")

    # Load audio
    import soundfile as sf

    audio, _ = sf.read(audio_path, dtype="float32")
    print(f"\n=== Audio ===")
    print(f"Total samples: {len(audio)}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")

    # Create streaming buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False
    )

    # Add audio to buffer - this pre-computes ALL mel frames
    processed_signal, processed_signal_length, stream_id = (
        streaming_buffer.append_audio_file(audio_path, stream_id=-1)
    )

    print(f"\n=== Pre-computed Mel Buffer ===")
    print(f"Buffer shape: {streaming_buffer.buffer.shape}")
    print(f"Buffer (mel frames): {streaming_buffer.buffer.shape[2]}")
    print(f"Total mel frames computed upfront: {processed_signal_length.item()}")

    # Calculate expected mel frames
    hop_samples = int(preproc_cfg.window_stride * sample_rate)
    window_samples = int(preproc_cfg.window_size * sample_rate)
    expected_frames = (len(audio) - window_samples) // hop_samples + 1
    print(f"Expected mel frames from formula: {expected_frames}")

    print("\n=== Chunk Iteration ===")
    streaming_buffer_iter = iter(streaming_buffer)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        # Calculate what chunk/shift sizes are being used
        chunk_size = (
            streaming_cfg.chunk_size[0]
            if step_num == 0
            else streaming_cfg.chunk_size[1]
        )
        shift_size = (
            streaming_cfg.shift_size[0]
            if step_num == 0
            else streaming_cfg.shift_size[1]
        )
        pre_encode_cache = (
            streaming_cfg.pre_encode_cache_size[0]
            if step_num == 0
            else streaming_cfg.pre_encode_cache_size[1]
        )
        drop_extra = streaming_cfg.drop_extra_pre_encoded if step_num != 0 else 0

        print(f"\nStep {step_num}:")
        print(f"  chunk_size={chunk_size}, shift_size={shift_size}")
        print(f"  pre_encode_cache={pre_encode_cache}, drop_extra={drop_extra}")
        print(f"  chunk_audio shape: {chunk_audio.shape}")
        print(f"  chunk_lengths: {chunk_lengths.item()}")
        print(f"  buffer_idx after: {streaming_buffer.buffer_idx}")

        if step_num >= 5:
            print("\n  ... (truncated)")
            break

    print(f"\n=== Summary ===")

    # Show how NeMo computes these values
    print("\n=== NeMo Config Computation ===")
    subsampling_factor = model.encoder.subsampling_factor
    print(f"subsampling_factor: {subsampling_factor}")

    # Get the sampling frames from pre_encode
    if hasattr(model.encoder, "pre_encode"):
        sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        print(f"sampling_frames (from pre_encode): {sampling_frames}")
        pre_encode_cache_size = model.encoder.pre_encode.get_streaming_cache_size()
        print(f"pre_encode streaming_cache_size: {pre_encode_cache_size}")


if __name__ == "__main__":
    main()
