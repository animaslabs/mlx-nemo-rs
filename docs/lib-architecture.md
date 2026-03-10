# mlx-nemo-rs Architecture Overview

## High-Level Structure

```
mlx-nemo-rs/
├── src/
│   ├── lib.rs              # Entry point, re-exports Error/Result
│   ├── core/               # Core utilities (always available)
│   │   ├── error.rs        # Error types with 15+ variants
│   │   ├── config.rs       # YAML parsing, FromConfig trait
│   │   ├── checkpoint.rs   # .nemo file handling
│   │   └── pytorch.rs      # Native .ckpt loader (no Python!)
│   ├── asr/                # ASR module (feature-gated)
│   │   ├── config.rs       # NeMo-compatible configs
│   │   ├── models/         # High-level models
│   │   │   └── enc_dec_ctc.rs  # Main CTC model
│   │   ├── modules/        # Neural network components
│   │   │   ├── preprocessors/mel_spectrogram.rs
│   │   │   ├── encoders/conformer.rs
│   │   │   └── decoders/ctc.rs
│   │   └── decoding/       # Decoding strategies
│   │       └── greedy.rs
│   ├── audio.rs            # Audio I/O, resampling, features
│   ├── mlx.rs              # MLX utilities (device, tensor ops)
│   ├── math.rs             # Basic math utilities
│   └── utils.rs            # Constants, string/collection helpers
```

## Key Types & Their Locations

### Core Error Handling
| Type | File | Line | Purpose |
|------|------|------|---------|
| `Error` | `src/core/error.rs` | 10-89 | Main error enum (15 variants) |
| `Result<T>` | `src/core/error.rs` | 7 | Type alias for `std::result::Result<T, Error>` |

### Configuration System
| Type | File | Line | Purpose |
|------|------|------|---------|
| `FromConfig` trait | `src/core/config.rs` | 14-20 | Build types from config |
| `NemoCheckpoint` | `src/core/checkpoint.rs` | 17-31 | Extracted .nemo contents |
| `AsrModelConfig` | `src/asr/config.rs` | 9-43 | Top-level ASR configuration |
| `PreprocessorConfig` | `src/asr/config.rs` | 50-107 | Mel spectrogram settings |
| `EncoderConfig` | `src/asr/config.rs` | 135-216 | Conformer encoder settings |
| `DecoderConfig` | `src/asr/config.rs` | 256-273 | CTC decoder settings |

### ASR Models & Modules
| Type | File | Line | Purpose |
|------|------|------|---------|
| `EncDecCTCModel` | `src/asr/models/enc_dec_ctc.rs` | 36-44 | Main ASR model |
| `MelSpectrogram` | `src/asr/modules/preprocessors/mel_spectrogram.rs` | 15-25 | Audio feature extraction |
| `ConformerEncoder` | `src/asr/modules/encoders/conformer.rs` | 67-78 | Conformer encoder |
| `ConformerBlock` | `src/asr/modules/encoders/conformer.rs` | 271-293 | Single conformer layer |
| `CTCDecoder` | `src/asr/modules/decoders/ctc.rs` | 15-22 | Linear projection to vocab |

### Key Functions
| Function | File | Line | Signature |
|----------|------|------|-----------|
| `EncDecCTCModel::restore` | `enc_dec_ctc.rs` | 76-120 | `fn restore(path) -> Result<Self>` |
| `EncDecCTCModel::transcribe` | `enc_dec_ctc.rs` | 153-167 | `fn transcribe(&mut self, paths) -> Result<Vec<String>>` |
| `greedy_decode` | `decoding/greedy.rs` | 19-66 | CTC greedy decoding |
| `NemoCheckpoint::open` | `checkpoint.rs` | 39-83 | Extract .nemo archive |
| `pytorch::load` | `pytorch.rs` | 45-78 | Load .ckpt without Python |

## Data Flow Diagram

```
┌─────────────────┐
│  Audio Files    │
│    (.wav)       │
└────────┬────────┘
         │ load_wav() + resample()
         ▼
┌─────────────────┐
│  MelSpectrogram │  STFT → Mel filterbank → Log → Normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ConformerEncoder │  Subsampling → N×ConformerBlock → LayerNorm
│                 │  (FF→MHSA→Conv→FF per block)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CTCDecoder    │  Linear projection → Log softmax
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  greedy_decode  │  Argmax → CTC collapse
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SentencePiece  │  Token IDs → Text
└─────────────────┘
```

## Architectural Patterns

1. **Configuration-driven instantiation**: Models built from NeMo-compatible YAML via the `FromConfig` pattern

2. **Weight loading via string prefixes**: Weights loaded by matching keys like `"encoder.layers.0.ff1.linear1.weight"`

3. **Native PyTorch checkpoint loading**: Custom pickle protocol 2 parser in `pytorch.rs` (893 lines) - no Python dependency

4. **Feature-gated compilation**: `asr` feature gates audio dependencies (`hound`, `rubato`) and ASR code

5. **Layered module hierarchy**: 
   - `models/` - High-level API (`transcribe`, `restore`)
   - `modules/` - NN components (`forward`, `load_weights`)
   - `decoding/` - Post-processing strategies

6. **Error propagation**: All functions return `Result<T>` with rich error context via helper constructors (`Error::config()`, `Error::checkpoint_at()`, etc.)

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Modules | 4 main (`core`, `asr`, `audio`, `mlx`) |
| Error variants | 15+ |
| Config structs | 6 major |
| NN modules | 6 (MelSpec, Conformer, ConformerBlock, FeedForward, MHSA, Conv) |
| Total LoC (approx) | ~3,500 |

---

Would you like me to now fetch information about the Python NeMo library architecture so we can compare them directly? I can research specific aspects such as:
- Module organization
- Configuration system
- Model class hierarchy  
- Decoding strategies
- Preprocessing pipeline

---

