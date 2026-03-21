#!/usr/bin/env python3
"""
Detailed streaming ASR using CacheAwareRNNTPipeline with intermediate outputs.

This script demonstrates the internal workings of cache-aware streaming:

1. How audio is chunked into frames
2. How mel spectrograms are computed and buffered
3. How the encoder cache is maintained across chunks
4. How RNN-T decoding produces tokens incrementally
5. How partial and final transcripts are generated

Useful for debugging and comparing with the Rust implementation.

Usage:
    cd scripts
    source .venv/bin/activate
    python streaming_pipeline_detailed.py ../samples/2086-149220-0033.wav
    python streaming_pipeline_detailed.py --max-chunks 5 audio.wav
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import (
    CacheAwareRNNTPipeline,
)
from nemo.collections.asr.inference.streaming.framing.request_options import (
    ASRRequestOptions,
)
from nemo.collections.asr.inference.streaming.framing.request import Frame

import logging
for name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
    logging.getLogger(name).setLevel(logging.ERROR)


SAMPLE_RATE = 16000
VALID_LOOKAHEADS = {0: 0, 80: 1, 480: 6, 1040: 13}


def build_config(model_path: str, device: str, lookahead_ms: int):
    """Build pipeline configuration."""
    config_path = Path(__file__).parent / "streaming_config.yaml"
    cfg = OmegaConf.load(config_path)

    cfg.asr.model_name = model_path
    cfg.asr.device = device
    cfg.asr.device_id = 0
    cfg.asr.compute_dtype = "float32"
    cfg.asr.use_amp = False

    right_context = VALID_LOOKAHEADS.get(lookahead_ms, lookahead_ms // 80)
    cfg.streaming.att_context_size = [70, right_context]
    cfg.asr_decoding_type = "rnnt"
    cfg.log_level = 40

    return cfg


def load_model_and_pipeline(cfg):
    """Load model and create pipeline."""

    # The wrapper handles model loading internally
    wrapper = CacheAwareRNNTInferenceWrapper(
        model_name=cfg.asr.model_name,
        device=cfg.asr.device,
        device_id=cfg.asr.device_id,
        compute_dtype=cfg.asr.compute_dtype,
        use_amp=cfg.asr.use_amp,
        decoding_cfg=cfg.asr.decoding,
    )

    pipeline = CacheAwareRNNTPipeline(
        cfg=cfg,
        asr_model=wrapper,
        itn_model=None,
        nmt_model=None,
    )

    # Get the underlying ASR model for detailed inspection
    asr_model = wrapper.asr_model

    return pipeline, wrapper, asr_model


def print_model_info(pipeline, wrapper, asr_model):
    """Print detailed model and pipeline information."""
    encoder = asr_model.encoder
    decoder = asr_model.decoder

    # Get streaming config
    streaming_cfg = encoder.streaming_cfg
    if streaming_cfg is None:
        encoder.setup_streaming_params()
        streaming_cfg = encoder.streaming_cfg

    print(f"\n{'='*70}")
    print("Model Configuration")
    print(f"{'='*70}")
    print(f"Model type: {type(asr_model).__name__}")
    print(f"Encoder d_model: {encoder.d_model}")
    print(f"Subsampling factor: {encoder.subsampling_factor}")
    print(f"Vocabulary size: {decoder.vocab_size}")
    print(f"Blank ID: {decoder.vocab_size}")

    print(f"\n{'='*70}")
    print("Encoder Streaming Config")
    print(f"{'='*70}")
    print(f"chunk_size: {streaming_cfg.chunk_size}")
    print(f"shift_size: {streaming_cfg.shift_size}")
    print(f"valid_out_len: {streaming_cfg.valid_out_len}")
    print(f"pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print(f"cache_drop_size: {streaming_cfg.cache_drop_size}")
    print(f"last_channel_cache_size: {streaming_cfg.last_channel_cache_size}")
    if hasattr(streaming_cfg, 'last_time_cache_size'):
        print(f"last_time_cache_size: {streaming_cfg.last_time_cache_size}")

    print(f"\n{'='*70}")
    print("Pipeline Configuration")
    print(f"{'='*70}")
    print(f"Sample rate: {pipeline.sample_rate} Hz")
    print(f"Model chunk size: {pipeline.model_chunk_size} mel frames")
    print(f"Chunk duration: {pipeline.chunk_size_in_secs:.4f}s ({pipeline.chunk_size_in_secs*1000:.1f}ms)")
    print(f"Buffer duration: {pipeline.buffer_size_in_secs:.4f}s ({pipeline.buffer_size_in_secs*1000:.1f}ms)")
    print(f"Pre-encode cache: {pipeline.pre_encode_cache_size} frames")
    print(f"Pre-encode cache duration: {pipeline.pre_encode_cache_size_in_secs:.4f}s")
    print(f"Tokens per frame: {pipeline.tokens_per_frame}")
    print(f"Use cache: {pipeline.use_cache}")
    print(f"Use feature cache: {pipeline.use_feat_cache}")


def run_detailed_streaming(pipeline, wrapper, asr_model, audio_path: str, max_chunks: int = None):
    """
    Run streaming with detailed output at each step.

    Shows:
    - Audio chunk boundaries
    - Mel spectrogram shapes
    - Encoder output shapes and values
    - Decoder tokens emitted
    - Cache state
    """

    # Load audio info
    info = sf.info(audio_path)
    audio_duration = info.duration

    print(f"\n{'='*70}")
    print("Audio File")
    print(f"{'='*70}")
    print(f"Path: {audio_path}")
    print(f"Duration: {audio_duration:.3f}s ({int(audio_duration * SAMPLE_RATE)} samples)")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Channels: {info.channels}")

    # Get request generator
    request_generator = pipeline.get_request_generator()
    options = [ASRRequestOptions()]
    request_generator.set_audio_filepaths([audio_path], options)

    # Tracking
    chunks_processed = 0
    all_tokens = []
    accumulated_final = ""
    last_partial = ""

    print(f"\n{'='*70}")
    print("Streaming Processing")
    print(f"{'='*70}")

    pipeline.open_session()

    try:
        for requests in request_generator:
            if max_chunks and chunks_processed >= max_chunks:
                print(f"\n[Stopping after {max_chunks} chunks]")
                break

            # Print chunk header
            print(f"\n{'─'*70}")
            print(f"Chunk {chunks_processed}")
            print(f"{'─'*70}")

            # Examine requests
            for i, request in enumerate(requests):
                is_frame = isinstance(request, Frame)
                print(f"Request {i}: stream_id={request.stream_id}, "
                      f"is_first={request.is_first}, is_last={request.is_last}, "
                      f"type={'Frame' if is_frame else 'FeatureBuffer'}")

                if is_frame:
                    samples = request.samples
                    print(f"  Audio samples: {samples.shape}, "
                          f"duration: {len(samples)/SAMPLE_RATE*1000:.1f}ms")
                    print(f"  First 5 samples: {samples[:5].tolist()}")
                else:
                    # FeatureBuffer
                    features = request.features
                    print(f"  Features shape: {features.shape}")
                    print(f"  Valid size: {request.valid_size}")

            # Run transcribe step
            step_outputs = pipeline.transcribe_step(requests)
            chunks_processed += 1

            # Examine outputs
            for step_output in step_outputs:
                print(f"\nStep Output (stream_id={step_output.stream_id}):")

                # Partial transcript
                partial = step_output.partial_transcript.strip()
                if partial:
                    # Show what changed from last partial
                    if partial != last_partial:
                        print(f"  Partial: '{partial}'")
                        last_partial = partial
                    else:
                        print(f"  Partial: (unchanged)")
                else:
                    print(f"  Partial: (empty)")

                # Final transcript (after EOU detection)
                final = step_output.final_transcript.strip()
                if final:
                    print(f"  FINAL: '{final}'")
                    accumulated_final += (" " if accumulated_final else "") + final

                # Current step transcript (new tokens this step)
                step_text = step_output.current_step_transcript
                if step_text:
                    print(f"  Current step: '{step_text}'")

            # Get internal state for more details
            for request in requests:
                state = pipeline.get_state(request.stream_id)
                if state:
                    tokens = state.tokens
                    if tokens:
                        print(f"\n  State tokens ({len(tokens)}): {tokens[-20:]}"
                              f"{'...' if len(tokens) > 20 else ''}")

    finally:
        pipeline.close_session()

    # Final results
    print(f"\n{'='*70}")
    print("Final Results")
    print(f"{'='*70}")
    print(f"Chunks processed: {chunks_processed}")
    print(f"Accumulated final: '{accumulated_final}'")
    print(f"Last partial: '{last_partial}'")

    # The final text
    final_text = accumulated_final if accumulated_final else last_partial
    print(f"\nTranscription: '{final_text}'")

    return final_text, chunks_processed


def main():
    parser = argparse.ArgumentParser(
        description="Detailed streaming ASR with CacheAwareRNNTPipeline",
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
        help="Path to .nemo model file",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=80,
        help="Lookahead in ms (default: 80)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to process (default: all)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Build config
    cfg = build_config(args.model, args.device, args.lookahead)

    # Load model and pipeline
    print("Loading model and pipeline...")
    start = time.time()
    pipeline, wrapper, asr_model = load_model_and_pipeline(cfg)
    print(f"Loaded in {time.time() - start:.2f}s")

    # Print model info
    print_model_info(pipeline, wrapper, asr_model)

    # Run streaming
    final_text, chunks = run_detailed_streaming(
        pipeline, wrapper, asr_model,
        str(audio_path), args.max_chunks
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
