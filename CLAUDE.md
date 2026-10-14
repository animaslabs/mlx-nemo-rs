# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-voice-rs is a Rust implementation of NVIDIA NeMo speech AI models for Apple Silicon, built on the MLX framework. It provides ASR (Automatic Speech Recognition) capabilities including CTC and RNN-T/Transducer models with streaming support.

## Build Commands

```bash
# Build the library
cargo build

# Build in release mode
cargo build --release

# Run tests (most are #[ignore] by default - require model files)
# IMPORTANT: Must use --test-threads=1 to avoid Metal/GPU command buffer crashes
cargo test -- --test-threads=1

# Run a specific ignored test (requires model files in models/)
cargo test test_load_parakeet_rnnt_checkpoint -- --ignored

# Run an example
cargo run --example transcribe_rnnt -- --model models/nvidia/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b.nemo samples/christmas2025_nightbeforechristmas_kri_64kb.wav
cargo run --example streaming -- --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo samples/Speaker26_004_16k.wav

# Download models from Hugging Face
cargo xtask download nvidia/parakeet-ctc-0.6b
cargo xtask download nvidia/nemotron-speech-streaming-en-0.6b --token $HF_TOKEN
```

## Architecture

### Core Modules (`src/core/`)
- `checkpoint.rs`: Handles `.nemo` file loading (TAR archives containing config YAML, weights, tokenizers)
- `pytorch.rs`: PyTorch pickle checkpoint loading for model weights
- `config.rs`: YAML config parsing utilities
- `error.rs`: Custom error types

### ASR Collection (`src/asr/`)

**Models** (`models/`):
- `EncDecCTCModel`: CTC-based ASR (e.g., parakeet-ctc-0.6b)
- `EncDecRNNTModel`: RNN-T/Transducer ASR (e.g., parakeet-rnnt-0.6b)

**Modules** (`modules/`):
- `MelSpectrogram`: Audio preprocessing (mel filterbank features)
- `ConformerEncoder`: FastConformer encoder with multi-head attention and convolution
- `CTCDecoder`: CTC output projection layer
- `RNNTJoint`: Joint network for RNN-T
- `PredictionNetwork`: RNN-T decoder/prediction network

**Pipeline** (`pipeline/`):
- `CacheAwareRNNTPipeline`: Main streaming pipeline for RNN-T models, handles audio buffering, cache-aware encoder processing, and incremental decoding

**Streaming** (`streaming/`):
- `EncoderCache`: Manages attention and convolution caches for streaming
- `MelBuffer`/`FeatureBuffer`: Audio chunk buffering
- `StreamingConfig`: Streaming mode configuration (chunk sizes, context windows)

**Decoding** (`decoding/`):
- `greedy.rs`: Greedy CTC decoding
- `rnnt_greedy.rs`, `rnnt_streaming.rs`: RNN-T greedy decoding
- `merge.rs`: Token merging and LCS-based buffer merging for streaming

### Data Flow

1. Audio (f32 samples) → `MelSpectrogram` → mel features [batch, n_mels, frames]
2. Mel features → `ConformerEncoder` → encoder output [batch, d_model, time']
3. Encoder output → `CTCDecoder` or `RNNTJoint` → logits
4. Logits → greedy/beam decoding → token IDs → tokenizer → text

For streaming:
- Audio accumulated in `MelBuffer` until enough frames
- Encoder uses `EncoderCache` for attention/conv context
- RNN-T decodes incrementally with `DecoderState`

## Model Files

Models are stored in `models/nvidia/` and downloaded via xtask:
- `parakeet-ctc-0.6b/`: Non-streaming CTC model
- `parakeet-rnnt-0.6b/`: Non-streaming RNN-T model
- `nemotron-speech-streaming-en-0.6b/`: Streaming RNN-T model (gated, requires HF token)

`.nemo` files are uncompressed TAR archives containing:
- `model_config.yaml`: Full model configuration
- `model_weights.ckpt`: PyTorch checkpoint with weights
- `*.model`: SentencePiece tokenizer files

## Key Types

- `AsrModelConfig`: Top-level config with preprocessor, encoder, decoder sections
- `NemoCheckpoint`: Handles .nemo archive extraction and loading
- `StreamingModeInfo`: Describes streaming mode latency/context settings
- `TranscribeOutput`: Streaming transcription result with partial text and tokens
