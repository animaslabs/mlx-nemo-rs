# NeMo Verification Script

This document describes the Python verification script used to run NVIDIA NeMo Parakeet CTC models and capture intermediate values for comparison with the mlx-nemo-rs Rust implementation.

## Overview

The `scripts/verify_nemo.py` script loads a NeMo ASR model and processes audio files, outputting detailed information at each pipeline stage:

1. **Audio loading** - Raw waveform statistics
2. **Mel spectrogram** - Preprocessor output (STFT → Mel filterbank → Normalization)
3. **Encoder output** - Conformer encoder hidden states
4. **Decoder logits** - CTC log probabilities
5. **Final transcription** - Decoded text

This enables numerical verification of the Rust implementation against the reference Python/PyTorch implementation.

## Prerequisites

- **Python 3.10-3.12** (NeMo 2.x is incompatible with Python 3.13+)
- **uv** package manager (recommended) or pip
- Downloaded Parakeet model (via `cargo xtask download`)

## Setup

### Quick Setup (Recommended)

```bash
cd scripts
./setup.sh
```

The setup script will:
1. Create a Python 3.12 virtual environment
2. Install PyTorch, NeMo, and all dependencies
3. Create mock `nv_one_logger` package (NVIDIA internal, not on PyPI)
4. Verify the installation

### Manual Setup

If you prefer manual installation:

```bash
cd scripts

# Create Python 3.12 virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install core dependencies first (order matters for resolver)
uv pip install torch numpy soundfile numba

# Install NeMo without extras to avoid dependency conflicts
uv pip install nemo_toolkit --no-deps

# Install remaining dependencies
uv pip install -r requirements.txt
```

You'll also need to create the `nv_one_logger` mock package - see `setup.sh` for details.

## Usage

### Basic Usage

```bash
# Activate the environment
source scripts/.venv/bin/activate

# Run verification
python scripts/verify_nemo.py \
    --model models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \
    --audio samples/2086-149220-0033.wav
```

### Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Path to `.nemo` model file (required) |
| `--audio` | `-a` | Path to audio file (required) |
| `--device` | | Device for inference: `cpu`, `cuda`, `mps` (default: `cpu`) |
| `--verbose` | `-v` | Show additional sample values |
| `--quiet` | `-q` | Suppress NeMo logging output |
| `--help` | `-h` | Show help message |

### Examples

```bash
# Basic transcription verification
python scripts/verify_nemo.py \
    -m models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \
    -a samples/2086-149220-0033.wav

# Verbose output with more sample values
python scripts/verify_nemo.py \
    -m models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \
    -a samples/2086-149220-0033.wav \
    --verbose

# Run on GPU (if available)
python scripts/verify_nemo.py \
    -m models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \
    -a samples/2086-149220-0033.wav \
    --device cuda
```

## Output Explanation

### Audio File Info

```
Sample rate: 16000 Hz
Duration: 7.435 seconds
Num samples: 118960
Waveform shape: (118960,)
Waveform dtype: float32
Waveform range: [-0.536407, 0.781830]
```

Basic audio file properties. The Rust implementation should load identical values.

### Mel Spectrogram

```
Shape: (1, 80, 744)
Dtype: float32
Min: -1.583432, Max: 5.317983, Mean: 0.000000, Std: 0.998652
Mel length: 743

Sample values for comparison:
  mel[0, 0, :5] = [-1.506, -1.278, -1.264, -1.281, -1.432]
  mel[0, :5, 0] = [-1.506, -1.481, -1.446, -1.356, -1.364]
  mel[0, 39, :5] = [-1.280, -1.302, -1.306, -1.303, -1.300]
  mel[0, 79, :5] = [-1.166, -1.127, -1.256, -1.020, -1.142]
```

- **Shape**: `(batch, mel_bins, time_frames)` = `(1, 80, 744)`
- **80 mel bins**: Standard for NeMo ASR models
- **744 frames**: `num_samples / hop_length` ≈ `118960 / 160 = 743.5`
- **Normalized**: Mean ≈ 0, Std ≈ 1 (per-feature normalization)

### Encoder Output

```
Shape: (1, 1024, 93)
Dtype: float32
Min: -9.334983, Max: 4.213620, Mean: -0.008121, Std: 0.626286
Encoder output length: 93

Sample values for comparison:
  encoder[0, 0, :5] = [0.102, 0.187, 0.203, 0.171, 0.219]
  encoder[0, :5, 0] = [0.102, -0.632, 0.566, -0.170, -0.227]
```

- **Shape**: `(batch, d_model, time_frames)` = `(1, 1024, 93)`
- **1024**: Hidden dimension of the Conformer encoder
- **93 frames**: 8x subsampling from mel frames (`744 / 8 ≈ 93`)

### Decoder Logits

```
Shape: (1, 93, 1025)
Dtype: float32
Min: -175.465332, Max: -0.000004, Mean: -23.266014, Std: 5.508220

Predicted token IDs (first 20): [1024, 1024, 1024, 1024, 1024, 1024, 298, ...]

Top 5 tokens at position 0:
  1. Token 1024: 1.0000
  2. Token 741: 0.0000
  ...
```

- **Shape**: `(batch, time_frames, vocab_size)` = `(1, 93, 1025)`
- **1025 vocab**: 1024 BPE tokens + 1 CTC blank token (index 1024)
- **Log probabilities**: Values are in log space (negative)

### Final Transcription

```
Text: "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait"
Length: 117 characters
```

The final decoded text after CTC greedy decoding and SentencePiece detokenization.

### Verification Summary

```
Audio duration: 7.435s
Inference time: 1.463s
Real-time factor: 0.197x

Key shapes for Rust comparison:
  Input waveform: (118960,)
  Mel spectrogram: (1, 80, 744)
  Encoder output: (1, 1024, 93)
  Decoder logits: (1, 93, 1025)
```

## Reference Values

Reference output for `samples/2086-149220-0033.wav` with `parakeet-ctc-0.6b`:

| Stage | Shape | Key Values |
|-------|-------|------------|
| Waveform | `(118960,)` | range: `[-0.536, 0.782]` |
| Mel spectrogram | `(1, 80, 744)` | mean: `0.0`, std: `~1.0` |
| Encoder output | `(1, 1024, 93)` | mean: `~0.0`, std: `~0.6` |
| Decoder logits | `(1, 93, 1025)` | log probs, blank=1024 |
| Transcription | 117 chars | "well i don't wish to see it..." |

## Troubleshooting

### Python Version Errors

```
RuntimeError: Cannot install on Python version 3.14.2; only versions >=3.6,<3.10 are supported.
```

NeMo 2.x requires Python 3.10-3.12. Use the setup script which creates a Python 3.12 environment:

```bash
cd scripts && ./setup.sh
```

### Missing nv_one_logger

```
ModuleNotFoundError: No module named 'nv_one_logger'
```

This is an NVIDIA internal package not available on PyPI. The `setup.sh` script creates a mock package. If you installed manually, run the setup script or copy the mock creation commands from it.

### numba/numpy Compatibility

```
ImportError: Numba needs NumPy 2.3 or less
```

Install a compatible numpy version:

```bash
uv pip install "numpy>=1.24.0,<2.4"
```

### Slow Import Times

NeMo has many dependencies and takes 5-10 seconds to import. This is normal. The script shows import progress:

```
Importing nemo_toolkit (this may take a moment)...
Import completed in 5.61s
```

### MPS Device Warnings

```
'pin_memory' argument is set as true but not supported on MPS now
```

This warning can be ignored - it doesn't affect results. Use `--device cpu` to avoid it.

## Files

| File | Description |
|------|-------------|
| `scripts/verify_nemo.py` | Main verification script |
| `scripts/requirements.txt` | Python dependencies |
| `scripts/setup.sh` | Automated setup script |
| `scripts/.venv/` | Python virtual environment (created by setup) |
