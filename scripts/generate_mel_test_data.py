#!/usr/bin/env python3
"""Generate golden test data for mel_spectrogram.rs unit tests.

This script generates reference values from librosa/PyTorch/NeMo
that can be used to validate the Rust implementation.

Usage:
    python scripts/generate_mel_test_data.py

Output:
    testdata/mel_spectrogram/*.json
"""

import json
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("testdata/mel_spectrogram")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_window_data():
    """Generate window function reference values."""
    import scipy.signal as signal

    data = {}

    # Standard window lengths used in ASR
    for length in [400, 512]:
        # Hann window (periodic=False for 'symmetric' which is scipy default)
        hann = signal.windows.hann(length, sym=True)
        data[f"hann_{length}"] = {
            "length": length,
            "values": hann.tolist(),
            "first": float(hann[0]),
            "last": float(hann[-1]),
            "middle": float(hann[length // 2]),
            "max": float(hann.max()),
            "sum": float(hann.sum()),
        }

        # Hamming window
        hamming = signal.windows.hamming(length, sym=True)
        data[f"hamming_{length}"] = {
            "length": length,
            "values": hamming.tolist(),
            "first": float(hamming[0]),
            "last": float(hamming[-1]),
            "middle": float(hamming[length // 2]),
            "max": float(hamming.max()),
            "sum": float(hamming.sum()),
        }

    with open(OUTPUT_DIR / "windows.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'windows.json'}")


def generate_mel_filterbank_data():
    """Generate mel filterbank reference values using librosa."""
    import librosa

    # Standard NeMo/Parakeet configuration
    configs = [
        {"n_mels": 80, "n_fft": 512, "sr": 16000, "fmin": 0.0, "fmax": None},
    ]

    data = {}

    for cfg in configs:
        n_fft = cfg["n_fft"]
        sr = cfg["sr"]
        n_mels = cfg["n_mels"]
        fmin = cfg["fmin"]
        fmax = cfg["fmax"] if cfg["fmax"] else sr / 2

        # librosa uses Slaney normalization by default (norm='slaney')
        # htk=False means Slaney mel scale (not HTK)
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=False,
            norm="slaney",
        )

        key = f"mel_{n_mels}_{n_fft}_{sr}"
        data[key] = {
            "n_mels": n_mels,
            "n_fft": n_fft,
            "sample_rate": sr,
            "fmin": fmin,
            "fmax": fmax,
            "shape": list(mel_basis.shape),
            # Store select rows to keep file size reasonable
            "row_0": mel_basis[0].tolist(),  # First filter
            "row_39": mel_basis[39].tolist(),  # Middle filter
            "row_79": mel_basis[79].tolist(),  # Last filter
            # Store a few key statistics
            "min": float(mel_basis.min()),
            "max": float(mel_basis.max()),
            "row_sums": mel_basis.sum(axis=1).tolist(),  # Sum of each filter
        }

    with open(OUTPUT_DIR / "mel_filterbank.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'mel_filterbank.json'}")


def generate_preemphasis_data():
    """Generate preemphasis reference values."""
    data = {}

    # Test case 1: Simple impulse
    x1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    coef = 0.97
    # y[n] = x[n] - coef * x[n-1], with x[-1] = 0
    y1 = np.zeros_like(x1)
    y1[0] = x1[0]  # x[-1] = 0
    for i in range(1, len(x1)):
        y1[i] = x1[i] - coef * x1[i - 1]

    data["impulse"] = {
        "input": x1.tolist(),
        "coef": coef,
        "output": y1.tolist(),
    }

    # Test case 2: Ramp signal
    x2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    y2 = np.zeros_like(x2)
    y2[0] = x2[0]
    for i in range(1, len(x2)):
        y2[i] = x2[i] - coef * x2[i - 1]

    data["ramp"] = {
        "input": x2.tolist(),
        "coef": coef,
        "output": y2.tolist(),
    }

    # Test case 3: Sine wave
    t = np.linspace(0, 1, 100, dtype=np.float32)
    x3 = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    y3 = np.zeros_like(x3)
    y3[0] = x3[0]
    for i in range(1, len(x3)):
        y3[i] = x3[i] - coef * x3[i - 1]

    data["sine"] = {
        "input": x3.tolist(),
        "coef": coef,
        "output": y3.tolist(),
    }

    with open(OUTPUT_DIR / "preemphasis.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'preemphasis.json'}")


def generate_output_length_data():
    """Generate output length calculation reference values."""
    # NeMo default config
    sample_rate = 16000
    window_size = 0.025  # 25ms
    window_stride = 0.01  # 10ms
    n_fft = 512

    win_length = int(window_size * sample_rate)  # 400
    hop_length = int(window_stride * sample_rate)  # 160

    # Formula with center padding:
    # padded_len = input_len + n_fft
    # n_frames = (padded_len - n_fft) / hop_length + 1
    #          = input_len / hop_length + 1

    test_lengths = [1600, 8000, 16000, 24000, 32000, 48000, 160, 320, 480]

    data = {
        "config": {
            "sample_rate": sample_rate,
            "window_size": window_size,
            "window_stride": window_stride,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
        "cases": [],
    }

    for input_len in test_lengths:
        # Using floor division to match Rust behavior
        n_frames = input_len // hop_length + 1
        data["cases"].append(
            {
                "input_length": input_len,
                "output_length": n_frames,
            }
        )

    with open(OUTPUT_DIR / "output_lengths.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'output_lengths.json'}")


def generate_stft_data():
    """Generate STFT reference values using PyTorch (matches NeMo)."""
    import torch

    data = {}

    # Config matching NeMo defaults
    n_fft = 512
    win_length = 400
    hop_length = 160

    # Create centered Hann window (same as NeMo)
    window = torch.hann_window(win_length)

    # Test case 1: Short signal (1 second at 16kHz)
    torch.manual_seed(42)
    x1 = torch.randn(1, 16000)

    # PyTorch STFT with center padding (default), pad_mode='constant' (zeros)
    stft1 = torch.stft(
        x1.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )

    # Power spectrum
    power1 = torch.abs(stft1) ** 2

    data["short_signal"] = {
        "input_length": 16000,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "stft_shape": list(stft1.shape),
        "power_shape": list(power1.shape),
        "n_frames": stft1.shape[1],
        "n_freqs": stft1.shape[0],
        # Store statistics for validation
        "power_mean": float(power1.mean()),
        "power_std": float(power1.std()),
        "power_max": float(power1.max()),
        # Store first frame power spectrum for detailed comparison
        "power_frame_0": power1[:, 0].tolist(),
    }

    # Test case 2: Sine wave (deterministic)
    sr = 16000
    freq = 440  # A4 note
    duration = 0.1  # 100ms
    t = torch.linspace(0, duration, int(sr * duration))
    x2 = torch.sin(2 * np.pi * freq * t).unsqueeze(0)

    stft2 = torch.stft(
        x2.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power2 = torch.abs(stft2) ** 2

    # Find the bin with max energy (should be near 440 Hz)
    freq_bins = torch.fft.rfftfreq(n_fft, 1 / sr)
    expected_bin = int(round(freq * n_fft / sr))

    data["sine_440hz"] = {
        "frequency": freq,
        "sample_rate": sr,
        "duration": duration,
        "input_length": len(t),
        "stft_shape": list(stft2.shape),
        "n_frames": stft2.shape[1],
        "expected_peak_bin": expected_bin,
        "expected_peak_freq": float(freq_bins[expected_bin]),
        # Average power spectrum across frames
        "avg_power_spectrum": power2.mean(dim=1).tolist(),
    }

    with open(OUTPUT_DIR / "stft.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'stft.json'}")


def generate_normalization_data():
    """Generate per-feature normalization reference values."""
    np.random.seed(42)

    # Shape: [batch, n_mels, frames]
    batch_size = 2
    n_mels = 80
    n_frames = 100

    features = np.random.randn(batch_size, n_mels, n_frames).astype(np.float32)

    # Per-feature normalization: normalize each mel bin to zero mean, unit variance
    # Mean and std computed along the time axis (axis=2)
    mean = features.mean(axis=2, keepdims=True)
    std = features.std(axis=2, keepdims=True)
    # Add epsilon for numerical stability (matches Rust implementation)
    eps = 1e-5
    normalized = (features - mean) / np.sqrt(std**2 + eps)

    data = {
        "input_shape": [batch_size, n_mels, n_frames],
        "epsilon": eps,
        # Store subset of data to keep file size reasonable
        "input_sample": features[0, :5, :10].tolist(),  # First 5 mels, 10 frames
        "output_sample": normalized[0, :5, :10].tolist(),
        # Statistics for validation
        "output_mean_per_feature": normalized.mean(axis=2).tolist(),  # Should be ~0
        "output_var_per_feature": normalized.var(axis=2).tolist(),  # Should be ~1
    }

    with open(OUTPUT_DIR / "normalization.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'normalization.json'}")


def generate_full_pipeline_data():
    """Generate full mel spectrogram pipeline reference using NeMo."""
    try:
        from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
        import torch

        HAS_NEMO = True
    except ImportError:
        HAS_NEMO = False
        print("Warning: NeMo not installed, using librosa for pipeline data")

    import librosa

    data = {}

    # Generate a simple test signal
    sr = 16000
    duration = 1.0
    np.random.seed(42)

    # Test signal: chirp + noise
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * np.linspace(100, 1000, len(t)) * t)
    signal += 0.1 * np.random.randn(len(t)).astype(np.float32)

    if HAS_NEMO:
        # Use NeMo's FilterbankFeatures for reference
        preprocessor = FilterbankFeatures(
            sample_rate=sr,
            n_window_size=int(0.025 * sr),  # 400 samples
            n_window_stride=int(0.01 * sr),  # 160 samples
            window="hann",
            normalize="per_feature",
            n_fft=512,
            preemph=0.97,
            nfilt=80,
            lowfreq=0.0,
            highfreq=sr / 2,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2**-24,
            dither=0.0,  # Disable dither for deterministic output
            pad_to=0,
            frame_splicing=1,
            exact_pad=False,
            stft_exact_pad=False,
            stft_conv=False,
        )

        audio_tensor = torch.from_numpy(signal).unsqueeze(0)
        length_tensor = torch.tensor([len(signal)])

        with torch.no_grad():
            features, feat_len = preprocessor(audio_tensor, length_tensor)

        features = features.numpy()
        feat_len = feat_len.numpy()

        data["nemo_pipeline"] = {
            "input_length": len(signal),
            "sample_rate": sr,
            "output_shape": list(features.shape),
            "output_length": int(feat_len[0]),
            # Store statistics
            "output_mean": float(features.mean()),
            "output_std": float(features.std()),
            "output_min": float(features.min()),
            "output_max": float(features.max()),
            # Store subset for detailed comparison
            "output_frame_0": features[0, :, 0].tolist(),  # First frame, all mels
            "output_frame_50": features[0, :, 50].tolist()
            if features.shape[2] > 50
            else None,
            "output_mel_0": features[0, 0, :20].tolist(),  # First mel, first 20 frames
        }
    else:
        # Fallback to librosa
        # Apply preemphasis
        preemph_signal = librosa.effects.preemphasis(signal, coef=0.97)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=preemph_signal,
            sr=sr,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window="hann",
            center=True,
            pad_mode="constant",
            n_mels=80,
            fmin=0.0,
            fmax=sr / 2,
            htk=False,
            norm="slaney",
        )

        # Apply log with guard (NeMo style)
        log_mel = np.log(mel_spec + 2**-24)

        # Per-feature normalization
        mean = log_mel.mean(axis=1, keepdims=True)
        std = log_mel.std(axis=1, keepdims=True)
        normalized = (log_mel - mean) / (std + 1e-5)

        data["librosa_pipeline"] = {
            "input_length": len(signal),
            "sample_rate": sr,
            "output_shape": list(normalized.shape),
            "output_mean": float(normalized.mean()),
            "output_std": float(normalized.std()),
            "output_min": float(normalized.min()),
            "output_max": float(normalized.max()),
            "output_frame_0": normalized[:, 0].tolist(),
            "output_mel_0": normalized[0, :20].tolist(),
        }

    # Also store the input signal for reproducibility
    data["input_signal"] = {
        "sample_rate": sr,
        "duration": duration,
        "length": len(signal),
        # Store first 1000 samples
        "samples_head": signal[:1000].tolist(),
    }

    with open(OUTPUT_DIR / "full_pipeline.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated: {OUTPUT_DIR / 'full_pipeline.json'}")


def main():
    print("Generating mel spectrogram test data...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()

    generate_window_data()
    generate_mel_filterbank_data()
    generate_preemphasis_data()
    generate_output_length_data()
    generate_stft_data()
    generate_normalization_data()
    generate_full_pipeline_data()

    print()
    print("Done! Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")


if __name__ == "__main__":
    main()
