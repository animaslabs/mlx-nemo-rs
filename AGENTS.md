# AGENTS.md - AI Agent Guidelines for mlx-nemo-rs

Guidelines for AI coding agents working in this Rust implementation of NVIDIA NeMo for Apple Silicon (MLX).

## Build Commands

```bash
cargo build                    # Build the library
cargo build --all-features     # Build with all features (default includes 'asr')
cargo build --release          # Build in release mode
cargo check                    # Quick compilation check without building
```

## Test Commands

**IMPORTANT:** Tests must run single-threaded due to MLX runtime constraints.

```bash
cargo test -- --test-threads=1                     # Run all tests (single-threaded)
cargo test test_name -- --test-threads=1           # Run tests matching "test_name"
cargo test --exact test_name -- --test-threads=1   # Run a single test with exact name match
cargo test module::tests -- --test-threads=1       # Run tests in a specific module (e.g., math::tests)
cargo test -- --test-threads=1 --nocapture         # Run tests with stdout/stderr visible
cargo test --doc                                   # Run documentation tests only
```

## Lint and Format Commands

```bash
cargo fmt                      # Format code
cargo fmt --check              # Check formatting without changes
cargo clippy                   # Run clippy lints
cargo clippy -- -D warnings    # Fail on any warnings
```

## Development Tasks (xtask)

```bash
cargo xtask download nvidia/parakeet-ctc-0.6b                    # Download models from HF
cargo xtask download nvidia/parakeet-ctc-0.6b --output ./models  # Custom output directory
cargo xtask download nvidia/parakeet-ctc-0.6b --token <token>    # Use HF token for gated models
```

## Running Rust Examples

```bash
cargo run --example transcribe -- --model path/to/model.nemo audio.wav
```

## Running Python Examples

```bash
source scripts/.venv/bin/activate && python scripts/streaming.py --model stt_en_fastconformer_hybrid_large_streaming_multi -q samples/2086-149220-0033.wav 2>&1
```

## Code Style Guidelines

### Import Organization
Organize imports in three groups separated by blank lines: (1) std, (2) external crates, (3) crate modules.

```rust
use std::path::Path;

use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::core::error::{Error, Result};
```

### Naming Conventions
- **Structs/Enums**: `PascalCase` - `EncDecCTCModel`, `WindowType`
- **Functions/methods**: `snake_case` - `greedy_decode`, `load_audio_batch`
- **Constants**: `SCREAMING_SNAKE_CASE` - `PI`, `GOLDEN_RATIO`
- **Modules/files**: `snake_case` - `mel_spectrogram.rs`, `enc_dec_ctc.rs`

### Error Handling
Use the crate's `Error` type and `Result<T>` alias from `crate::core::error`:

```rust
use crate::core::error::{Error, Result};

// Use ? operator for propagation
fn load_model(path: &Path) -> Result<Model> {
    let config = load_config(path)?;
    Ok(Model::new(config)?)
}

// Use map_err for context; use ok_or_else for Option -> Result
let file = std::fs::read(path)
    .map_err(|e| Error::audio(format!("Failed to read {}: {}", path.display(), e)))?;

// Error helper constructors (prefer these over direct enum construction)
Error::config("Invalid configuration")
Error::checkpoint_at("Failed to load", path)
Error::model("Unsupported architecture")
Error::tokenizer("Missing vocabulary")
Error::audio("Invalid sample rate")
Error::not_implemented("Feature X")
```

### Feature Flags

```rust
#[cfg(feature = "asr")]
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> { ... }

#[cfg(not(feature = "asr"))]
pub fn load_wav(_path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    Err(Error::not_implemented("Audio loading requires the 'asr' feature"))
}
```

### Type Flexibility
Use `impl Into<T>` and `impl AsRef<Path>` for flexible function parameters.

### Serde Patterns

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub optional_field: Option<String>,
    #[serde(rename = "sample_rate")]
    pub sr: u32,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}
```

### Testing Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_name() {
        let result = function_under_test();
        assert_eq!(result, expected);
        assert!(condition, "helpful error message");
    }

    #[test]
    #[ignore]  // For tests requiring external resources (models, etc.)
    fn test_requires_model() { ... }
}
```

### Logging
Use `tracing` macros: `info!`, `debug!`, `warn!`, `error!` with structured fields.

```rust
info!("Loading model from {}", path.display());
debug!(sample_rate = %sr, "Audio loaded");
error!(?err, "Failed to process");
```

## Project Structure

```
mlx-nemo-rs/
├── src/
│   ├── lib.rs              # Library entry point, re-exports
│   ├── audio.rs            # Audio loading, resampling, preprocessing
│   ├── math.rs             # Mathematical utilities
│   ├── mlx.rs              # MLX integration layer
│   ├── core/               # Core utilities
│   │   ├── error.rs        # Error types (Error enum, Result alias)
│   │   ├── config.rs       # Configuration traits (FromConfig)
│   │   └── checkpoint.rs   # NeMo checkpoint handling
│   └── asr/                # ASR module (feature-gated with "asr")
│       ├── config.rs       # ASR configuration types
│       ├── models/         # High-level model implementations
│       ├── modules/        # Neural network modules (encoders, decoders, etc.)
│       └── decoding/       # Decoding strategies (greedy, beam search)
├── examples/               # Example applications (transcribe.rs)
├── xtask/                  # Development task runner (model downloads)
├── models/                 # Downloaded models (gitignored)
└── samples/                # Sample audio files for testing
```

## Key Dependencies

- `mlx-rs` (0.25) - MLX bindings for Apple Silicon
- `thiserror` (2.0) - Error derive macros
- `serde` / `serde_yaml` / `serde_json` - Serialization
- `hound` / `rubato` - Audio I/O and resampling (optional, "asr" feature)
- `sentencepiece` - Tokenization
- `tracing` - Structured logging
