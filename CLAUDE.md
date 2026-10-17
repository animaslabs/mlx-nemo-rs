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
cargo test --features internals test_load_parakeet_rnnt_checkpoint -- --ignored

# `--features hub` lets from_pretrained("owner/name") download from the HF Hub.
# Internals (encoder layers, decoders, streaming caches, checkpoint parsing) are
# crate-private unless the `internals` feature is on. The debug_*/compare_encoder/
# mel_test/trace_streaming examples and tests/pytorch_loading.rs require it.
cargo check --all-targets --features internals

# Run an example
cargo run --example transcribe_rnnt -- --model models/nvidia/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b.nemo samples/christmas2025_nightbeforechristmas_kri_64kb.wav
cargo run --example streaming -- --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo samples/Speaker26_004_16k.wav
cargo run --release --example multi_talker -- samples/Speaker26_004_16k.wav  # auto-detects models in models/

# Multi-talker model tests (need the two models below)
cargo test --release --test multitalker -- --ignored --test-threads=1 --nocapture

# Download models from Hugging Face
cargo xtask download nvidia/parakeet-ctc-0.6b
cargo xtask download nvidia/nemotron-speech-streaming-en-0.6b --token $HF_TOKEN
cargo xtask download nvidia/nemotron-3.5-asr-streaming-0.6b --file nemotron-3.5-asr-streaming-0.6b.nemo
cargo xtask download nvidia/multitalker-parakeet-streaming-0.6b-v1
cargo xtask download nvidia/diar_streaming_sortformer_4spk-v2.1

# WER eval: shared private dataset animaslabs/asr-eval → samples/eval/, scored in ../mlx-models
cargo xtask download-eval
cargo run --release --example eval -- --model <model.nemo> --out hypotheses.jsonl
(cd ../mlx-models && uv run samples/score.py ../mlx-voice-rs/hypotheses.jsonl)
# eval also takes --offline (whole-file RNN-T), --dtype bfloat16 (offline encoder), --warmup N, --repeats N

# Speed + WER benchmark vs parakeet-mlx, mlx-audio and NeMo (lives in mlx-models; builds and runs examples/eval)
(cd ../mlx-models && uv run bench/bench.py)
```

The eval dataset revision is pinned in both `xtask/src/main.rs` and mlx-models `samples/score.py` (`EVAL_REVISION`); bump both together.

## Architecture

### Core Modules (`src/core/`)
- `checkpoint.rs`: Handles `.nemo` file loading (TAR archives containing config YAML, weights, tokenizers)
- `pytorch.rs`: PyTorch pickle checkpoint loading for model weights
- `config.rs`: YAML config parsing utilities
- `error.rs`: Custom error types
- `loading.rs`: `LoadOptions` / `LoadReport` and the `Weights` context layer loaders take. Loads are strict by default (a missing tensor is an error); `strict(false)` keeps initial values and reports them

### ASR Collection (`src/asr/`)

**Models** (`models/`):
- `EncDecCtcModel`: CTC-based ASR (e.g., parakeet-ctc-0.6b)
- `EncDecRnntModel`: RNN-T/Transducer ASR (e.g., parakeet-rnnt-0.6b)

**Modules** (`modules/`):
- `MelSpectrogram`: Audio preprocessing (mel filterbank features)
- `ConformerEncoder`: FastConformer encoder with multi-head attention and convolution
- `CtcDecoder`: CTC output projection layer
- `RnntJoint`: Joint network for RNN-T
- `SpeakerKernels`: Multitalker speaker kernels, applied by `ConformerEncoder` before selected layers when speaker targets are set
- `SortformerModules`: SortFormer speaker head (diarization)
- `LanguagePrompt`: Language-ID prompt (`prompt_kernel`) of multilingual prompt models; `RnntJoint::project_encoder` applies it before the encoder projection, so every decode path is conditioned
- `PredictionNetwork`: RNN-T decoder/prediction network

**Pipeline** (`pipeline/`):
- `StreamingSession`: Cache-aware streaming for RNN-T models (`model.stream(options)` borrows the model, `StreamingSession::new(model, options)` owns it); handles resampling, audio buffering, cache-aware encoder processing, and incremental decoding
- `MultiTalkerPipeline`: Wraps an owned `StreamingSession` with a streaming SortFormer diarizer fed the same mel chunks. With a multitalker checkpoint (has `spk_kernels.*` weights) it runs one ASR instance per speaker as a batched encoder call (parallel mode); otherwise it attributes one transcript to speakers (serial mode)

**Streaming** (`streaming/`):
- `EncoderCache`: Manages attention and convolution caches for streaming
- `MelBuffer`/`FeatureBuffer`: Audio chunk buffering
- `StreamingConfig`: Streaming mode configuration (chunk sizes, context windows)
- `StreamingSortformerState`: SortFormer speaker cache / FIFO of pre-encode embeddings (port of NeMo's `streaming_update` and `_compress_spkcache`)

**Decoding** (`decoding/`):
- `greedy.rs`: Greedy CTC decoding
- `rnnt_greedy.rs`, `rnnt_streaming.rs`: RNN-T greedy decoding
- `merge.rs`: Token merging and LCS-based buffer merging for streaming

### Data Flow

1. Audio (f32 samples) → `MelSpectrogram` → mel features [batch, n_mels, frames]
2. Mel features → `ConformerEncoder` → encoder output [batch, d_model, time']
3. Encoder output → `CtcDecoder` or `RnntJoint` → logits
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
- `nemotron-3.5-asr-streaming-0.6b/`: Multilingual streaming RNN-T with a language-ID prompt (`set_language`, default `auto`; `<xx-XX>` tags stripped from text by default)
- `multitalker-parakeet-streaming-0.6b-v1/`: Streaming multitalker RNN-T (Nemotron streaming + speaker kernels)
- `diar_streaming_sortformer_4spk-v2.1/`: Streaming SortFormer diarizer (up to 4 speakers)

`.nemo` files are TAR archives (usually uncompressed; some, like nemotron-3.5, are gzip-compressed) containing:
- `model_config.yaml`: Full model configuration
- `model_weights.ckpt`: PyTorch checkpoint with weights
- `*.model`: SentencePiece tokenizer files

## Key Types

- `AsrModelConfig`: Top-level config with preprocessor, encoder, decoder sections
- `NemoCheckpoint`: Handles .nemo archive extraction and loading
- `StreamingModeInfo`: Describes streaming mode latency/context settings
- `Transcript` / `Token` / `Word`: Offline transcription result with token and word timestamps (`asr::transcript`), from `transcribe(inputs)` / `transcribe_with(inputs, &TranscribeOptions)`; inputs are `AudioInput` (paths or `(samples, rate)`)
- `StreamUpdate`: One streaming step's new tokens (with times) and appended text (`delta`)
