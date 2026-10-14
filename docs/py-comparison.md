# Architecture Comparison: mlx-voice-rs vs NVIDIA NeMo Python

## Overview

| Aspect | **mlx-voice-rs (Your Rust Implementation)** | **NVIDIA NeMo Python** |
|--------|---------------------------------------------|------------------------|
| Language | Rust | Python |
| ML Backend | mlx-rs (Apple Silicon) | PyTorch + CUDA |
| Scope | ASR (CTC-based Conformer) | ASR, TTS, NLP, LLMs, VLMs, etc. |
| LoC (ASR) | ~3,500 | ~100,000+ |
| Training | Not implemented | Full training support |
| Inference | Supported | Supported |

---

## Module Structure Comparison

### High-Level Organization

| Category | **mlx-voice-rs** | **NeMo Python** |
|----------|-----------------|-----------------|
| Entry point | `src/lib.rs` | `nemo/collections/asr/__init__.py` |
| Models | `src/asr/models/` | `nemo/collections/asr/models/` |
| NN Modules | `src/asr/modules/` | `nemo/collections/asr/parts/submodules/` |
| Decoding | `src/asr/decoding/` | `nemo/collections/asr/parts/submodules/` |
| Config | `src/asr/config.rs` | YAML files + OmegaConf |
| Preprocessing | `src/asr/modules/preprocessors/` | `nemo/collections/asr/parts/preprocessing/` |

### NeMo Python ASR Directory Structure
```
nemo/collections/asr/
├── models/
│   ├── ctc_models.py          # EncDecCTCModel (base)
│   ├── ctc_bpe_models.py      # EncDecCTCModelBPE
│   ├── rnnt_models.py         # Transducer models
│   ├── rnnt_bpe_models.py     # EncDecRNNTBPEModel
│   ├── hybrid_rnnt_ctc*.py    # Hybrid models
│   └── aed_models.py          # Encoder-Decoder (AED)
├── parts/
│   ├── submodules/
│   │   ├── conformer_modules.py  # ConformerLayer, etc.
│   │   ├── multi_head_attention.py
│   │   ├── ctc_decoding.py
│   │   └── rnnt_decoding.py
│   └── preprocessing/
│       └── features.py        # Mel spectrogram
├── data/                      # Dataset classes
├── losses/                    # CTCLoss, RNNTLoss
└── metrics/                   # WER, etc.
```

---

## Model Classes Comparison

### CTC Models

| Feature | **mlx-voice-rs** | **NeMo Python** |
|---------|-----------------|-----------------|
| Main class | `EncDecCTCModel` | `EncDecCTCModel` / `EncDecCTCModelBPE` |
| Location | `src/asr/models/enc_dec_ctc.rs:36` | `ctc_models.py` / `ctc_bpe_models.py` |
| Inheritance | None (standalone struct) | `ASRModel` → `ModelPT` → `LightningModule` |
| Config-driven | Yes (YAML) | Yes (OmegaConf + YAML) |
| Tokenizer | SentencePiece only | BPE, WPE, Char, Aggregate |

### Key Method Comparison

| Method | **mlx-voice-rs** | **NeMo Python** |
|--------|-----------------|-----------------|
| Load model | `EncDecCTCModel::restore(path)` | `ASRModel.from_pretrained(name)` / `.restore_from(path)` |
| Transcribe | `model.transcribe(&paths)` | `model.transcribe(paths)` |
| Forward | `model.forward(&audio, lengths)` | `model.forward(input_signal, input_signal_length)` |
| Load weights | `model.load_weights(&weights)` | Automatic via PyTorch |
| Change vocab | Not implemented | `model.change_vocabulary(...)` |
| Change decoding | Not implemented | `model.change_decoding_strategy(...)` |

---

## Encoder Architecture Comparison

### Conformer Encoder

| Component | **mlx-voice-rs** | **NeMo Python** |
|-----------|-----------------|-----------------|
| Main struct/class | `ConformerEncoder` | `ConformerEncoder` |
| Location | `src/asr/modules/encoders/conformer.rs:67` | `conformer_encoder.py` |
| Block class | `ConformerBlock` | `ConformerLayer` |
| FeedForward | `FeedForward` | `ConformerFeedForward` |
| Attention | `MultiHeadAttention` | `MultiHeadAttention`, `RelPositionMultiHeadAttention` |
| Convolution | `ConvolutionModule` | `ConformerConvolution` |
| Subsampling | `Subsampling` (Conv-based) | `ConvSubsampling` / `StackingSubsampling` |

### Conformer Block Structure (Both implement the same architecture)

```
Input
  │
  ├──► FeedForward (half-step) ──► residual
  │
  ├──► MultiHeadSelfAttention ──► residual
  │
  ├──► ConvolutionModule ──► residual
  │
  ├──► FeedForward (half-step) ──► residual
  │
  └──► LayerNorm
  │
Output
```

### Attention Variants

| Variant | **mlx-voice-rs** | **NeMo Python** |
|---------|-----------------|-----------------|
| Absolute Position | Implemented | `MultiHeadAttention` |
| Relative Position (XL) | Not implemented | `RelPositionMultiHeadAttention` |
| Longformer-style | Not implemented | `RelPositionMultiHeadAttentionLongformer` |
| Streaming/Cache | Not implemented | Full support |

---

## Decoder Comparison

### CTC Decoder

| Feature | **mlx-voice-rs** | **NeMo Python** |
|---------|-----------------|-----------------|
| Class | `CTCDecoder` | `ConvASRDecoder` |
| Location | `src/asr/modules/decoders/ctc.rs:15` | `conv_asr.py` |
| Structure | Linear → LogSoftmax | Linear (configurable) |
| Vocabulary | From config | Dynamic, changeable |

### Decoding Strategies

| Strategy | **mlx-voice-rs** | **NeMo Python** |
|----------|-----------------|-----------------|
| Greedy | `greedy_decode()` | `CTCGreedyDecoding` |
| Beam Search | Not implemented | `CTCBeamDecoding` (multiple variants) |
| LM Fusion | Not implemented | N-gram, Neural LM support |
| With Timestamps | `greedy_decode_with_timestamps()` | Full timestamp support |

---

## Preprocessing Comparison

### Mel Spectrogram

| Feature | **mlx-voice-rs** | **NeMo Python** |
|---------|-----------------|-----------------|
| Class | `MelSpectrogram` | `AudioToMelSpectrogramPreprocessor` |
| Location | `preprocessors/mel_spectrogram.rs:15` | `features.py` |
| STFT | Custom implementation | `torch.stft` / torchaudio |
| Window types | Hann, Hamming, Blackman | Multiple via torchaudio |
| Normalization | Per-feature | Per-feature / Batch |
| Dithering | Not implemented | Supported |
| Speed/Tempo perturb | Not implemented | Supported |

---

## Configuration System Comparison

| Aspect | **mlx-voice-rs** | **NeMo Python** |
|--------|-----------------|-----------------|
| Format | YAML (yaml_serde) | YAML + OmegaConf |
| Config struct | `AsrModelConfig` | `DictConfig` (dynamic) |
| Target resolution | `target_class_name()` | Hydra `_target_` field |
| Type safety | Compile-time (Rust structs) | Runtime (OmegaConf) |
| Nested configs | `PreprocessorConfig`, `EncoderConfig`, etc. | Nested DictConfig |

### Config Structure Comparison

**mlx-voice-rs** (`src/asr/config.rs`):
```rust
pub struct AsrModelConfig {
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub tokenizer: Option<TokenizerConfig>,
    pub decoding: Option<DecodingConfig>,
}
```

**NeMo Python** (typical YAML):
```yaml
model:
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    ...
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    ...
  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoder
    ...
```

---

## Checkpoint Handling Comparison

| Feature | **mlx-voice-rs** | **NeMo Python** |
|---------|-----------------|-----------------|
| Format | `.nemo` (TAR), `.ckpt`, `.safetensors` | `.nemo` (TAR), `.ckpt` |
| Config extraction | `NemoCheckpoint::open()` | `restore_from()` |
| Weight loading | `pytorch::load()` (custom pickle parser!) | PyTorch `torch.load()` |
| Safetensors | Supported | Supported |
| Artifact handling | `checkpoint.artifacts` HashMap | Automatic |

### Your Native PyTorch Checkpoint Loader

A notable achievement in your implementation is the **native Rust pickle parser** (`src/core/pytorch.rs`, 893 lines) that loads PyTorch `.ckpt` files without Python dependency. This is quite impressive!

---

## Features Gap Analysis

### What mlx-voice-rs Has That Works Well

| Feature | Status | Notes |
|---------|--------|-------|
| NeMo checkpoint loading | ✅ Complete | Both TAR and gzip |
| Native .ckpt loading | ✅ Complete | No Python needed |
| Safetensors support | ✅ Complete | Via mlx-rs |
| CTC greedy decoding | ✅ Complete | With timestamps |
| Conformer encoder | ✅ Complete | Basic architecture |
| Mel spectrogram | ✅ Complete | Custom STFT |
| SentencePiece tokenizer | ✅ Complete | |
| Audio loading/resampling | ✅ Complete | hound + rubato |

### What mlx-voice-rs is Missing

| Feature | NeMo Python | Priority |
|---------|-------------|----------|
| **Decoder Types** | | |
| RNN-T / Transducer | `EncDecRNNTBPEModel` | High |
| TDT (Token-Duration Transducer) | `EncDecRNNTBPEModel` variants | Medium |
| AED (Attention Encoder-Decoder) | `EncDecMultiTaskModel` | Medium |
| **Decoding** | | |
| Beam search | `CTCBeamDecoding`, `BeamRNNTInfer` | High |
| LM fusion (N-gram, Neural) | Extensive support | Medium |
| Word/segment timestamps | Full support | Low (partial) |
| **Encoder Features** | | |
| Relative positional encoding | `RelPositionMultiHeadAttention` | High |
| Longformer attention | `RelPositionMultiHeadAttentionLongformer` | Low |
| Cache-aware streaming | Full streaming support | Medium |
| FastConformer optimizations | 2.4x faster than Conformer | Medium |
| **Training** | | |
| Training loop | PyTorch Lightning | High (if needed) |
| Loss functions | CTCLoss, RNNTLoss | High (if needed) |
| Data augmentation | SpecAugment, speed perturb | Medium |
| **Other** | | |
| Multi-language support | 14+ languages | Low |
| Hybrid CTC-RNNT | `EncDecHybridRNNTCTCBPEModel` | Low |

---

## API Style Comparison

### Transcription API

**mlx-voice-rs:**
```rust
let model = EncDecCTCModel::restore("path/to/model.nemo")?;
let transcriptions = model.transcribe(&["audio.wav"])?;
```

**NeMo Python:**
```python
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-0.6b")
transcriptions = model.transcribe(["audio.wav"])
```

Both are quite similar in API design, which is good for users transitioning between them.

---

## Summary

Your **mlx-voice-rs** implementation is a solid foundation that correctly mirrors the core NeMo architecture for CTC-based ASR models. The key architectural decisions align well:

1. **Module hierarchy** matches NeMo's `models/` → `modules/` → `decoding/` pattern
2. **Configuration system** uses compatible YAML structure
3. **Conformer architecture** follows the same block structure
4. **Checkpoint compatibility** is excellent (including native .ckpt loading)

**Main gaps** to consider closing if you want parity:
1. **Relative positional encoding** in attention (used by all modern NeMo models)
2. **Beam search decoding** for better accuracy
3. **RNN-T/Transducer decoder** (Parakeet models use this)
4. **Streaming/caching** for real-time inference

Would you like me to dive deeper into any specific area, or help plan implementation of any missing features?

---

