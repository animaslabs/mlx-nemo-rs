#!/usr/bin/env python3
"""
Debug script to inspect mel spectrogram values from NeMo's CacheAwareRNNTPipeline.

This script extracts mel values from the pipeline to compare with Rust implementation.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from nemo.collections.asr.inference.factory.pipeline_builder import (
            PipelineBuilder,
        )
    except ImportError:
        print("Error: NeMo's new inference API not found.")
        sys.exit(1)


def load_audio(audio_path: str, target_sample_rate: int = 16000):
    """Load audio file."""
    import soundfile as sf

    data, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sample_rate != target_sample_rate:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)

    return data, target_sample_rate


def build_config(model_path: str, lookahead: int):
    """Build pipeline configuration."""
    from omegaconf import OmegaConf

    # Valid lookahead sizes
    VALID_LOOKAHEADS = {
        0: 0,
        80: 1,
        480: 6,
        1040: 13,
    }

    right_context = VALID_LOOKAHEADS.get(lookahead, lookahead // 80)
    att_context_size = [70, right_context]

    config_path = Path(__file__).parent / "streaming_config.yaml"
    cfg = OmegaConf.load(config_path)

    cfg.asr.model_name = model_path
    cfg.asr.device = "cpu"
    cfg.asr.compute_dtype = "float32"
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = att_context_size
    cfg.asr_decoding_type = "rnnt"
    cfg.log_level = 40  # ERROR

    return cfg


def inspect_pipeline(pipeline):
    """Inspect the pipeline's preprocessor settings."""
    print("\n=== Pipeline Inspection ===")

    # Get the ASR model
    if hasattr(pipeline, "asr_model"):
        asr_model = pipeline.asr_model
        print(f"ASR Model type: {type(asr_model).__name__}")

        # Get preprocessor
        if hasattr(asr_model, "preprocessor"):
            preprocessor = asr_model.preprocessor
            print(f"\nPreprocessor type: {type(preprocessor).__name__}")

            # Print preprocessor config
            if hasattr(preprocessor, "_cfg"):
                cfg = preprocessor._cfg
                print("\nPreprocessor config:")
                for key in [
                    "sample_rate",
                    "window_size",
                    "window_stride",
                    "features",
                    "n_fft",
                    "dither",
                    "preemph",
                    "log",
                    "normalize",
                    "window",
                ]:
                    if hasattr(cfg, key):
                        print(f"  {key}: {getattr(cfg, key)}")

        # Get encoder streaming config
        if hasattr(asr_model, "encoder"):
            encoder = asr_model.encoder
            if hasattr(encoder, "streaming_cfg"):
                print(f"\nEncoder streaming config: {encoder.streaming_cfg}")


def extract_mel_from_pipeline(pipeline, audio_path: str):
    """Extract mel spectrograms by hooking into the pipeline."""
    print("\n=== Extracting Mel from Pipeline ===")

    # Get the request generator
    request_generator = pipeline.get_request_generator()

    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    options = [ASRRequestOptions()]
    request_generator.set_audio_filepaths([audio_path], options)

    pipeline.open_session()

    mel_chunks = []
    chunk_idx = 0

    try:
        for requests in request_generator:
            for req in requests:
                # The request should contain mel features
                print(f"\nChunk {chunk_idx}:")
                print(f"  Request type: {type(req).__name__}")

                if hasattr(req, "features"):
                    features = req.features
                    if isinstance(features, torch.Tensor):
                        feat_np = features.cpu().numpy()
                    else:
                        feat_np = np.array(features)

                    print(f"  Features shape: {feat_np.shape}")
                    print(f"  Features min: {feat_np.min():.6f}")
                    print(f"  Features max: {feat_np.max():.6f}")
                    print(f"  Features mean: {feat_np.mean():.6f}")
                    print(f"  Positive values: {np.sum(feat_np > 0)}")

                    mel_chunks.append(feat_np)

                if hasattr(req, "audio"):
                    audio = req.audio
                    if isinstance(audio, torch.Tensor):
                        audio_np = audio.cpu().numpy()
                    else:
                        audio_np = np.array(audio)
                    print(f"  Audio shape: {audio_np.shape}")
                    print(f"  Audio min: {audio_np.min():.6f}")
                    print(f"  Audio max: {audio_np.max():.6f}")

                chunk_idx += 1

            # Also run transcribe_step to get the actual processing
            step_outputs = pipeline.transcribe_step(requests)

    finally:
        pipeline.close_session()

    return mel_chunks


def compute_mel_directly(audio_path: str, model_path: str):
    """Compute mel spectrogram directly using NeMo's preprocessor."""
    print("\n=== Computing Mel Directly ===")

    import nemo.collections.asr as nemo_asr

    # Load the model to get its preprocessor
    print(f"Loading model: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        model_path, map_location="cpu"
    )

    # Get preprocessor
    preprocessor = model.preprocessor
    print(f"Preprocessor type: {type(preprocessor).__name__}")

    # Print config
    if hasattr(preprocessor, "_cfg"):
        cfg = preprocessor._cfg
        print("\nPreprocessor config from model:")
        for key in [
            "sample_rate",
            "window_size",
            "window_stride",
            "features",
            "n_fft",
            "dither",
            "preemph",
            "log",
            "normalize",
            "window",
            "lowfreq",
            "highfreq",
            "pad_to",
            "pad_value",
        ]:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                print(f"  {key}: {val}")

    # Load audio
    audio, sr = load_audio(audio_path)
    print(f"\nAudio: {len(audio)} samples, {sr} Hz, duration: {len(audio) / sr:.2f}s")
    print(f"Audio min: {audio.min():.6f}, max: {audio.max():.6f}")

    # Convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_length = torch.tensor([len(audio)], dtype=torch.long)

    print(f"Audio tensor shape: {audio_tensor.shape}")

    # Compute mel using the preprocessor
    with torch.no_grad():
        mel, mel_length = preprocessor(input_signal=audio_tensor, length=audio_length)

    mel_np = mel.squeeze(0).cpu().numpy()
    print(f"\nMel shape: {mel_np.shape}")
    print(f"Mel min: {mel_np.min():.6f}")
    print(f"Mel max: {mel_np.max():.6f}")
    print(f"Mel mean: {mel_np.mean():.6f}")
    print(f"Positive values: {np.sum(mel_np > 0)}")

    # Save mel to file for comparison
    np.save("python_mel_full.npy", mel_np)
    print("\nSaved mel to python_mel_full.npy")

    # Print first few frames
    print("\nFirst 5 frames (first 10 mel bins):")
    for i in range(min(5, mel_np.shape[1])):
        print(f"  Frame {i}: {mel_np[:10, i]}")

    return mel_np


def compute_mel_streaming(audio_path: str, model_path: str, lookahead: int = 1040):
    """Compute mel spectrogram in streaming mode."""
    print("\n=== Computing Mel in Streaming Mode ===")

    import nemo.collections.asr as nemo_asr

    # Load the model
    print(f"Loading model: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        model_path, map_location="cpu"
    )

    # Get preprocessor
    preprocessor = model.preprocessor

    # Load audio
    audio, sr = load_audio(audio_path)
    print(f"Audio: {len(audio)} samples, {sr} Hz")

    # Get streaming config from encoder
    encoder = model.encoder

    # Get streaming modes
    if hasattr(encoder, "get_streaming_cache_cfg"):
        cache_cfg = encoder.get_streaming_cache_cfg()
        print(f"Cache config: {cache_cfg}")

    # Calculate chunk size based on streaming config
    # This matches NeMo's CacheAwareRNNTPipeline logic
    hop_samples = int(0.01 * sr)  # 10ms hop = 160 samples
    win_samples = int(0.025 * sr)  # 25ms window = 400 samples

    # Get the model's att_context_size for the given lookahead
    VALID_LOOKAHEADS = {0: 0, 80: 1, 480: 6, 1040: 13}
    right_context = VALID_LOOKAHEADS.get(lookahead, lookahead // 80)
    att_context_size = [70, right_context]

    print(f"\nAttention context size: {att_context_size}")

    # Get streaming parameters from encoder
    if hasattr(encoder, "setup_streaming_params"):
        # This sets up the streaming configuration
        encoder.setup_streaming_params(
            decoding_cfg=None,
            cache_aware_cfg={
                "att_context_size": att_context_size,
            },
        )

    # Simulate streaming by processing audio in chunks
    # The chunk size calculation follows NeMo's logic:
    # chunk_audio_samples = (chunk_mel_frames + cache_frames) * hop_samples

    # Streaming mode 0 for 1040ms lookahead: chunk_size=112 mel frames
    chunk_mel_frames = 112  # This is model_chunk_size[1]
    cache_mel_frames = 9  # This is pre_encode_cache_size[1]

    total_mel_frames = chunk_mel_frames + cache_mel_frames
    chunk_audio_samples = total_mel_frames * hop_samples

    print(f"Chunk audio samples: {chunk_audio_samples}")
    print(f"Chunk mel frames: {chunk_mel_frames}")
    print(f"Cache mel frames: {cache_mel_frames}")

    # Process audio in chunks
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    all_mels = []
    pos = 0
    chunk_idx = 0

    while pos < len(audio):
        end = min(pos + chunk_audio_samples, len(audio))
        chunk_audio = audio[pos:end]

        # Pad if needed
        if len(chunk_audio) < chunk_audio_samples:
            chunk_audio = np.pad(
                chunk_audio, (0, chunk_audio_samples - len(chunk_audio))
            )

        chunk_tensor = torch.tensor(chunk_audio, dtype=torch.float32).unsqueeze(0)
        chunk_length = torch.tensor([len(chunk_audio)], dtype=torch.long)

        with torch.no_grad():
            mel, mel_length = preprocessor(
                input_signal=chunk_tensor, length=chunk_length
            )

        mel_np = mel.squeeze(0).cpu().numpy()

        print(f"\nChunk {chunk_idx} (pos={pos}, samples={end - pos}):")
        print(f"  Mel shape: {mel_np.shape}")
        print(f"  Mel min: {mel_np.min():.6f}")
        print(f"  Mel max: {mel_np.max():.6f}")
        print(f"  Positive values: {np.sum(mel_np > 0)}")

        all_mels.append(mel_np)

        pos = end
        chunk_idx += 1

    return all_mels


def main():
    parser = argparse.ArgumentParser(
        description="Debug mel spectrogram from NeMo pipeline"
    )
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("audio", type=str, help="Audio file path")
    parser.add_argument("--lookahead", type=int, default=1040, help="Lookahead in ms")
    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        choices=["direct", "streaming", "pipeline"],
        help="Mode: direct (full audio), streaming (chunked), pipeline (via CacheAwarePipeline)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    check_dependencies()

    if args.mode == "direct":
        compute_mel_directly(str(audio_path), args.model)
    elif args.mode == "streaming":
        compute_mel_streaming(str(audio_path), args.model, args.lookahead)
    elif args.mode == "pipeline":
        from nemo.collections.asr.inference.factory.pipeline_builder import (
            PipelineBuilder,
        )

        print(f"Model: {args.model}")
        print(f"Lookahead: {args.lookahead}ms")

        cfg = build_config(args.model, args.lookahead)

        print("\nBuilding pipeline...")
        pipeline = PipelineBuilder.build_pipeline(cfg)

        inspect_pipeline(pipeline)
        extract_mel_from_pipeline(pipeline, str(audio_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
