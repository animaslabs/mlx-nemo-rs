# AGENTS.md - AI Agent Guidelines for mlx-nemo-rs

This document provides guidelines for AI coding agents working in this repository.

## Project Overview

mlx-nemo-rs is a Rust implementation of NVIDIA NeMo for Apple Silicon, built on MLX.
It provides speech AI capabilities including ASR (Automatic Speech Recognition).

## Build Commands

```bash
cargo build                    # Build the library
cargo build --all-features     # Build with all features (default includes 'asr')
cargo build --release          # Build in release mode
cargo check                    # Check for compilation errors without building
```

## Test Commands

```bash
cargo test                     # Run all tests
cargo test test_name           # Run a single test by name
cargo test --exact test_add    # Run a single test with exact match
cargo test math::tests         # Run tests in a specific module
cargo test core::config::tests # Run tests in a specific file (by module path)
cargo test -- --nocapture      # Run tests with output visible
cargo test --doc               # Run doc tests only
```

## Lint and Format Commands

```bash
cargo fmt                      # Format code (uses default rustfmt settings)
cargo fmt --check              # Check formatting without changing files
cargo clippy                   # Run clippy lints
cargo clippy --all-features    # Run clippy with all features
cargo clippy -- -D warnings    # Run clippy and fail on warnings
```

## Development Tasks (xtask)

```bash
cargo xtask download nvidia/parakeet-ctc-0.6b                    # Download models from HF
cargo xtask download nvidia/parakeet-ctc-0.6b --output ./models  # Custom output directory
cargo xtask download nvidia/parakeet-ctc-0.6b --token <token>    # Use HF token for gated models
```

## Running Examples

```bash
cargo run --example transcribe -- --model path/to/model.nemo audio.wav
```

## Code Style Guidelines

### Import Organization

Organize imports in three groups, separated by blank lines:
1. Standard library (`std::*`)
2. External crates
3. Internal crate modules (`crate::*`)

```rust
use std::path::Path;

use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::core::error::{Error, Result};
use crate::asr::config::AsrModelConfig;
```

### Naming Conventions

- **Structs/Enums**: `PascalCase` - `EncDecCTCModel`, `AsrModelConfig`
- **Enum variants**: `PascalCase` - `Error::ShapeMismatch`, `WindowType::Hann`
- **Functions/methods**: `snake_case` - `greedy_decode`, `load_audio_batch`
- **Constants**: `SCREAMING_SNAKE_CASE` - `PI`, `GOLDEN_RATIO`
- **Modules**: `snake_case` - `mel_spectrogram.rs`, `enc_dec_ctc.rs`

### Error Handling

Use the crate's `Error` type and `Result<T>` alias from `crate::core::error`:

```rust
use crate::core::error::{Error, Result};

// Use ? operator for propagation
fn load_model(path: &Path) -> Result<Model> {
    let config = load_config(path)?;
    Ok(Model::new(config)?)
}

// Use map_err for context
let file = std::fs::read(path)
    .map_err(|e| Error::audio(format!("Failed to read {}: {}", path.display(), e)))?;

// Use ok_or_else for Option -> Result
let value = optional_value
    .ok_or_else(|| Error::config("Value is required"))?;

// Use helper constructors on Error
Error::config("Invalid configuration")
Error::checkpoint_at("Failed to load", path)
Error::model("Unsupported architecture")
Error::tokenizer("Missing vocabulary")
Error::audio("Invalid sample rate")
```

### Other Patterns

- **Documentation**: Use `//!` for module docs, `///` for items with `# Arguments`, `# Returns`, `# Errors`, `# Example` sections
- **Type flexibility**: Use `impl Into<T>` and `impl AsRef<Path>` for flexible parameters
- **Module organization**: Use `mod.rs` for aggregation, re-export types, use feature flags `#[cfg(feature = "...")]`
- **Logging**: Use `tracing` macros (`info!`, `debug!`, `warn!`, `error!`)
- **Serde**: Use `#[serde(default)]`, `#[serde(rename = "...")]`, `#[serde(flatten)]` as needed

### Testing Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_name() {
        assert_eq!(result, expected);
        assert!(condition);
    }
}
```

## Project Structure

```
mlx-nemo-rs/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── audio.rs            # Audio processing utilities
│   ├── math.rs             # Mathematical utilities
│   ├── mlx.rs              # MLX integration layer
│   ├── core/               # Core utilities
│   │   ├── error.rs        # Error types (Error, Result)
│   │   ├── config.rs       # Configuration traits
│   │   └── checkpoint.rs   # NeMo checkpoint handling
│   └── asr/                # ASR module (feature-gated)
│       ├── config.rs       # ASR configuration types
│       ├── models/         # High-level model implementations
│       ├── modules/        # Neural network modules
│       └── decoding/       # Decoding strategies
├── examples/               # Example applications
├── xtask/                  # Development task runner
├── models/                 # Downloaded models (gitignored)
└── samples/                # Sample audio files
```

## Key Dependencies

- `mlx-rs` (0.25) - MLX bindings for Apple Silicon
- `thiserror` (2.0) - Error handling derive macros
- `serde` / `serde_yaml` / `serde_json` - Serialization
- `hound` / `rubato` - Audio processing (optional, asr feature)
- `sentencepiece` - Tokenization
- `tracing` - Logging
