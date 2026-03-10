#!/usr/bin/env python3
"""
NeMo Parakeet CTC Model Verification Script

This script runs the NVIDIA NeMo Parakeet CTC model and outputs intermediate
values at each pipeline stage for comparison with the mlx-nemo-rs Rust implementation.

Usage:
    python scripts/verify_nemo.py \
        --model models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \
        --audio samples/2086-149220-0033.wav

Requirements:
    cd scripts && ./setup.sh
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy warnings from NeMo and its dependencies
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    for module in ["numpy", "torch", "soundfile", "nemo"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("Error: Missing required dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print("  pip install -r scripts/requirements.txt")
        sys.exit(1)


def print_separator(title: str) -> None:
    """Print a section separator with title."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_tensor_stats(name: str, tensor, show_samples: bool = True) -> None:
    """Print statistics and sample values for a tensor."""
    # Convert to numpy for easier handling
    arr = tensor.detach().cpu().numpy()

    print(f"\n--- {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(
        f"Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}, Std: {arr.std():.6f}"
    )

    if show_samples and arr.size > 0:
        # Show first few values (flattened view for simplicity)
        flat = arr.flatten()
        n_samples = min(10, len(flat))
        print(f"First {n_samples} values: {flat[:n_samples].tolist()}")


def load_audio_info(audio_path: str) -> dict:
    """Load audio file and return info without using NeMo's loader."""
    import soundfile as sf

    data, sample_rate = sf.read(audio_path, dtype="float32")

    return {
        "waveform": data,
        "sample_rate": sample_rate,
        "duration": len(data) / sample_rate,
        "num_samples": len(data),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify NeMo Parakeet CTC model outputs for comparison with Rust implementation"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Path to the .nemo model file"
    )
    parser.add_argument(
        "--audio", "-a", type=str, required=True, help="Path to the audio file (.wav)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show more detailed output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress NeMo logging output"
    )

    args = parser.parse_args()

    # Suppress NeMo logging if requested (must be done before importing nemo)
    if args.quiet:
        import logging

        # Set up logging suppression before any NeMo imports
        for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        # Also suppress the NeMo logging module specifically
        os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Check dependencies before proceeding
    check_dependencies()

    # Import heavy dependencies after arg parsing (so --help works without them)
    import torch

    # Validate paths
    model_path = Path(args.model)
    audio_path = Path(args.audio)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    print_separator("NeMo Parakeet CTC Verification")
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"Device: {args.device}")

    # Load audio info first (before NeMo import which is slow)
    print_separator("Audio File Info")
    audio_info = load_audio_info(str(audio_path))
    print(f"Sample rate: {audio_info['sample_rate']} Hz")
    print(f"Duration: {audio_info['duration']:.3f} seconds")
    print(f"Num samples: {audio_info['num_samples']}")
    print(f"Waveform shape: {audio_info['waveform'].shape}")
    print(f"Waveform dtype: {audio_info['waveform'].dtype}")
    print(
        f"Waveform range: [{audio_info['waveform'].min():.6f}, {audio_info['waveform'].max():.6f}]"
    )
    if args.verbose:
        print(f"First 10 samples: {audio_info['waveform'][:10].tolist()}")

    # Import NeMo (slow due to dependencies)
    print_separator("Loading NeMo")
    print("Importing nemo_toolkit (this may take a moment)...")
    import_start = time.time()

    import nemo.collections.asr as nemo_asr

    print(f"Import completed in {time.time() - import_start:.2f}s")

    # Load model
    print_separator("Loading Model")
    load_start = time.time()

    # Restore the model from .nemo file
    model = nemo_asr.models.ASRModel.restore_from(
        str(model_path), map_location=args.device
    )
    model.eval()

    print(f"Model loaded in {time.time() - load_start:.2f}s")
    print(f"Model type: {type(model).__name__}")
    print(f"Sample rate: {model.cfg.preprocessor.sample_rate} Hz")
    print(f"Vocabulary size: {model.decoder.num_classes_with_blank}")

    if args.verbose:
        print(f"\nModel config:")
        print(f"  Preprocessor: {model.cfg.preprocessor._target_}")
        print(f"  Encoder: {model.cfg.encoder._target_}")
        print(f"  Decoder: {model.cfg.decoder._target_}")

    # Storage for intermediate outputs
    intermediate_outputs = {}

    # Hook to capture preprocessor output (mel spectrogram)
    def preprocessor_hook(module, input, output):
        processed_signal, processed_signal_length = output
        intermediate_outputs["mel_spectrogram"] = processed_signal.clone()
        intermediate_outputs["mel_spectrogram_length"] = processed_signal_length.clone()

    # Hook to capture encoder output
    def encoder_hook(module, input, output):
        encoder_output, encoder_length = output
        intermediate_outputs["encoder_output"] = encoder_output.clone()
        intermediate_outputs["encoder_length"] = encoder_length.clone()

    # Hook to capture decoder output (logits)
    def decoder_hook(module, input, output):
        intermediate_outputs["decoder_logits"] = output.clone()

    # Register hooks
    preprocessor_handle = model.preprocessor.register_forward_hook(preprocessor_hook)
    encoder_handle = model.encoder.register_forward_hook(encoder_hook)
    decoder_handle = model.decoder.register_forward_hook(decoder_hook)

    try:
        # Run transcription
        print_separator("Running Inference")
        inference_start = time.time()

        with torch.no_grad():
            transcriptions = model.transcribe([str(audio_path)], batch_size=1)

        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.3f}s")

    finally:
        # Remove hooks
        preprocessor_handle.remove()
        encoder_handle.remove()
        decoder_handle.remove()

    # Print intermediate outputs
    print_separator("Mel Spectrogram (Preprocessor Output)")
    if "mel_spectrogram" in intermediate_outputs:
        mel = intermediate_outputs["mel_spectrogram"]
        mel_len = intermediate_outputs["mel_spectrogram_length"]
        print_tensor_stats("Mel Spectrogram", mel, show_samples=args.verbose)
        print(f"Mel length: {mel_len.item()}")

        # Show specific positions for comparison
        print(f"\nSample values for comparison:")
        print(f"  mel[0, 0, :5] = {mel[0, 0, :5].tolist()}")
        print(f"  mel[0, :5, 0] = {mel[0, :5, 0].tolist()}")
        print(f"  mel[0, 39, :5] = {mel[0, 39, :5].tolist()}")  # Middle mel bin
        print(f"  mel[0, 79, :5] = {mel[0, 79, :5].tolist()}")  # Last mel bin

    print_separator("Encoder Output")
    if "encoder_output" in intermediate_outputs:
        enc_out = intermediate_outputs["encoder_output"]
        enc_len = intermediate_outputs["encoder_length"]
        print_tensor_stats("Encoder Output", enc_out, show_samples=args.verbose)
        print(f"Encoder output length: {enc_len.item()}")

        # Show specific positions for comparison
        print(f"\nSample values for comparison:")
        print(f"  encoder[0, 0, :5] = {enc_out[0, 0, :5].tolist()}")
        print(f"  encoder[0, :5, 0] = {enc_out[0, :5, 0].tolist()}")

    print_separator("Decoder Logits")
    if "decoder_logits" in intermediate_outputs:
        logits = intermediate_outputs["decoder_logits"]
        print_tensor_stats(
            "Decoder Logits (Log Probabilities)", logits, show_samples=args.verbose
        )

        # Get predicted tokens via argmax
        predicted_tokens = torch.argmax(logits, dim=-1)
        print(f"\nPredicted token IDs (first 20): {predicted_tokens[0, :20].tolist()}")

        # Show log probabilities at specific positions
        print(f"\nSample log probabilities:")
        print(f"  logits[0, 0, :5] = {logits[0, 0, :5].tolist()}")
        print(f"  logits[0, 0, -5:] = {logits[0, 0, -5:].tolist()}")

        # Show top-k tokens at first position
        probs = torch.softmax(logits[0, 0], dim=-1)
        top_k = torch.topk(probs, k=5)
        print(f"\nTop 5 tokens at position 0:")
        for i, (idx, prob) in enumerate(
            zip(top_k.indices.tolist(), top_k.values.tolist())
        ):
            print(f"  {i + 1}. Token {idx}: {prob:.4f}")

    print_separator("Final Transcription")
    if isinstance(transcriptions, list):
        transcription = transcriptions[0]
    else:
        transcription = transcriptions

    # Handle different return types from NeMo
    if hasattr(transcription, "text"):
        text = transcription.text
    else:
        text = str(transcription)

    print(f'Text: "{text}"')
    print(f"Length: {len(text)} characters")

    print_separator("Verification Summary")
    print(f"Audio duration: {audio_info['duration']:.3f}s")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Real-time factor: {inference_time / audio_info['duration']:.3f}x")

    if "mel_spectrogram" in intermediate_outputs:
        mel = intermediate_outputs["mel_spectrogram"]
        print(f"\nKey shapes for Rust comparison:")
        print(f"  Input waveform: ({audio_info['num_samples']},)")
        print(f"  Mel spectrogram: {tuple(mel.shape)}")
        if "encoder_output" in intermediate_outputs:
            print(
                f"  Encoder output: {tuple(intermediate_outputs['encoder_output'].shape)}"
            )
        if "decoder_logits" in intermediate_outputs:
            print(
                f"  Decoder logits: {tuple(intermediate_outputs['decoder_logits'].shape)}"
            )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
