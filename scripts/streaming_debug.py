#!/usr/bin/env python3
"""
NeMo Streaming ASR with detailed encoder debugging for comparison with Rust.

This script adds hooks to log intermediate values from the FastConformer encoder
to compare against the Rust implementation.

Usage:
    source scripts/.venv/bin/activate
    python scripts/streaming_debug.py --model ../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo --lookahead 1040 ../samples/2086-149220-0033.wav
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

ENCODER_STEP_LENGTH = 80  # ms
SAMPLE_RATE = 16000  # Hz

VALID_LOOKAHEADS = {
    0: 0,
    80: 1,
    480: 6,
    1040: 13,
}


def lookahead_to_att_context(lookahead_ms: int, left_context: int = 70) -> list:
    """Convert lookahead in milliseconds to attention context size."""
    if lookahead_ms in VALID_LOOKAHEADS:
        right_context = VALID_LOOKAHEADS[lookahead_ms]
    else:
        right_context = lookahead_ms // ENCODER_STEP_LENGTH
    return [left_context, right_context]


def tensor_stats(tensor, name: str, step: int):
    """Return formatted tensor statistics string."""
    if tensor is None:
        return f"[Step {step}] {name}: None"

    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy().flatten()
    else:
        arr = np.array(tensor).flatten().astype(np.float32)

    return (
        f"[Step {step}] {name}: shape={list(tensor.shape)}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )


def install_encoder_hooks(encoder, verbose=True):
    """Install hooks on the encoder to log intermediate values.

    Returns a dict with step counter and collected stats.
    """
    import torch

    state = {
        "step": 0,
        "stats": [],
    }

    # Hook pre_encode (subsampling) if it exists
    if hasattr(encoder, "pre_encode") and encoder.pre_encode is not None:
        pre_encode = encoder.pre_encode
        original_pre_encode_forward = pre_encode.forward

        def hooked_pre_encode(x=None, lengths=None, **kwargs):
            step = state["step"]
            if verbose:
                print(tensor_stats(x, "pre_encode input", step))
                print(f"[Step {step}] pre_encode input length: {lengths}")

            out, out_len = original_pre_encode_forward(x=x, lengths=lengths, **kwargs)

            if verbose:
                print(tensor_stats(out, "pre_encode output", step))
                print(f"[Step {step}] pre_encode output length: {out_len}")

            return out, out_len

        pre_encode.forward = hooked_pre_encode

    # Hook the main encoder forward_for_export (streaming)
    original_forward_for_export = encoder.forward_for_export

    def hooked_forward_for_export(
        audio_signal,
        length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ):
        step = state["step"]

        print(f"\n{'=' * 70}")
        print(f"[Step {step}] ENCODER forward_for_export")
        print(f"{'=' * 70}")

        if verbose:
            print(tensor_stats(audio_signal, "encoder input (mel)", step))
            print(f"[Step {step}] input length: {length}")
            print(
                f"[Step {step}] cache_last_channel shape: {list(cache_last_channel.shape)}"
            )
            print(f"[Step {step}] cache_last_time shape: {list(cache_last_time.shape)}")
            print(f"[Step {step}] cache_last_channel_len: {cache_last_channel_len}")

        # Call original
        result = original_forward_for_export(
            audio_signal,
            length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        )

        out, out_len, new_cache_ch, new_cache_t, new_cache_len = result

        if verbose:
            print(tensor_stats(out, "encoder output", step))
            print(f"[Step {step}] encoder output length: {out_len}")
            print(f"[Step {step}] new cache_len: {new_cache_len}")

        state["step"] += 1
        return result

    encoder.forward_for_export = hooked_forward_for_export

    # Also hook regular forward for non-streaming comparisons
    original_forward = encoder.forward

    def hooked_forward(audio_signal=None, length=None, **kwargs):
        step = state["step"]

        # Check if this is streaming mode (has cache args)
        is_streaming = "cache_last_channel" in kwargs
        mode_str = "streaming" if is_streaming else "non-streaming"

        print(f"\n{'=' * 70}")
        print(f"[Step {step}] ENCODER forward ({mode_str})")
        print(f"{'=' * 70}")

        if verbose:
            print(tensor_stats(audio_signal, "encoder input (mel)", step))
            print(f"[Step {step}] input length: {length}")
            if is_streaming:
                cache_ch = kwargs.get("cache_last_channel")
                cache_t = kwargs.get("cache_last_time")
                cache_len = kwargs.get("cache_last_channel_len")
                if cache_ch is not None:
                    print(
                        f"[Step {step}] cache_last_channel shape: {list(cache_ch.shape)}"
                    )
                if cache_t is not None:
                    print(f"[Step {step}] cache_last_time shape: {list(cache_t.shape)}")
                print(f"[Step {step}] cache_last_channel_len: {cache_len}")

        result = original_forward(audio_signal=audio_signal, length=length, **kwargs)

        # Result is (out, out_len) for non-streaming,
        # or (out, out_len, cache_ch, cache_t, cache_len) for streaming
        if isinstance(result, tuple) and len(result) >= 2:
            out, out_len = result[0], result[1]
        else:
            out, out_len = result, None

        if verbose:
            print(tensor_stats(out, "encoder output", step))
            print(f"[Step {step}] encoder output length: {out_len}")
            if is_streaming and len(result) >= 5:
                print(f"[Step {step}] new cache_len: {result[4]}")

        state["step"] += 1
        return result

    encoder.forward = hooked_forward

    return state


def load_audio(audio_path: str, target_sample_rate: int = 16000):
    """Load and optionally resample audio file."""
    import soundfile as sf

    data, sample_rate = sf.read(audio_path, dtype="float32")

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if sample_rate != target_sample_rate:
        import librosa

        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)

    return data, target_sample_rate


def build_config(args):
    """Build the pipeline configuration."""
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent / "streaming_config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = args.model
    cfg.asr.device = args.device

    if args.device == "cuda":
        cfg.asr.compute_dtype = "bfloat16"
        cfg.asr.use_amp = True
    else:
        cfg.asr.compute_dtype = "float32"
        cfg.asr.use_amp = False

    cfg.streaming.att_context_size = lookahead_to_att_context(args.lookahead)
    cfg.asr_decoding_type = args.decoder
    cfg.log_level = 30  # WARNING

    return cfg


def run_streaming_debug(pipeline, audio_path: str, verbose: bool = True):
    """Run streaming inference with debug logging."""
    import soundfile as sf

    info = sf.info(audio_path)
    audio_duration = info.duration
    print(f"Audio duration: {audio_duration:.2f}s")

    # Install hooks on the encoder
    # pipeline.asr_model is a wrapper, the actual model is wrapper.asr_model
    wrapper = pipeline.asr_model
    model = wrapper.asr_model
    encoder = model.encoder
    hook_state = install_encoder_hooks(encoder, verbose=verbose)

    # Get request generator
    request_generator = pipeline.get_request_generator()

    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    options = [ASRRequestOptions()]
    request_generator.set_audio_filepaths([audio_path], options)

    print("\n--- Streaming Transcription (Debug Mode) ---")

    pipeline.open_session()

    accumulated_text = ""
    chunks_processed = 0

    try:
        for requests in request_generator:
            step_outputs = pipeline.transcribe_step(requests)
            chunks_processed += 1

            for step_output in step_outputs:
                if step_output.final_transcript:
                    final = step_output.final_transcript.strip()
                    if final:
                        print(f"\n[TRANSCRIPTION Final] {final}")
                        if accumulated_text:
                            accumulated_text += " " + final
                        else:
                            accumulated_text = final

    finally:
        pipeline.close_session()

    return accumulated_text, chunks_processed, audio_duration


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ASR with debug logging for Rust comparison"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model name or path to .nemo file",
    )
    parser.add_argument("audio", type=str, help="Path to the audio file")
    parser.add_argument(
        "--lookahead",
        type=int,
        default=1040,
        help="Lookahead size in milliseconds (default: 1040)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="rnnt",
        choices=["ctc", "rnnt"],
        help="Decoder type (default: rnnt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce logging verbosity"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Suppress NeMo logging
    import logging

    for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Decoder: {args.decoder.upper()}")
    print(f"Lookahead: {args.lookahead}ms")

    cfg = build_config(args)

    print("\nBuilding pipeline...")
    pipeline = PipelineBuilder.build_pipeline(cfg)
    print("Pipeline built!")

    print(f"Attention context: {cfg.streaming.att_context_size}")

    print(f"\nProcessing: {audio_path}")

    final_text, chunks, audio_duration = run_streaming_debug(
        pipeline, str(audio_path), verbose=not args.quiet
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Text: {final_text}")
    print(f"\nChunks processed: {chunks}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
