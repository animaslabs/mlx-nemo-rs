# NVIDIA NeMo ASR Models: Technical Architecture and Implementation Guide

This document provides a comprehensive technical reference for NVIDIA's Automatic Speech Recognition (ASR) models in the NeMo Framework. It covers the underlying neural network architectures, decoder algorithms, Python implementations, and configuration details needed for training, fine-tuning, and deploying these models.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FastConformer Encoder](#fastconformer-encoder)
3. [Decoder Architectures](#decoder-architectures)
4. [Model Families](#model-families)
5. [Python Implementation](#python-implementation)
6. [Configuration Reference](#configuration-reference)
7. [Streaming ASR](#streaming-asr)
8. [Training and Fine-Tuning](#training-and-fine-tuning)
9. [Resources and Links](#resources-and-links)

---

## Architecture Overview

NVIDIA NeMo ASR models follow an encoder-decoder architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ASR Model Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Audio Input (16kHz mono)                                               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  Preprocessor (AudioToMelSpectrogramPreprocessor)   │               │
│  │  • 80/128 mel filterbank features                   │               │
│  │  • 25ms window, 10ms stride                         │               │
│  │  • Per-feature normalization                        │               │
│  └─────────────────────────────────────────────────────┘               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  SpecAugment (Optional)                             │               │
│  │  • Frequency masking (2 masks, width 27)            │               │
│  │  • Time masking (10 masks, 5% width)                │               │
│  └─────────────────────────────────────────────────────┘               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  FastConformer Encoder                              │               │
│  │  • 8x depthwise-separable subsampling               │               │
│  │  • 17-24 conformer layers                           │               │
│  │  • Self-attention + Convolution modules             │               │
│  └─────────────────────────────────────────────────────┘               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  Decoder (CTC / RNNT / TDT / AED)                   │               │
│  │  • Greedy or beam search decoding                   │               │
│  │  • Optional language model fusion                   │               │
│  └─────────────────────────────────────────────────────┘               │
│       │                                                                 │
│       ▼                                                                 │
│  Text Output (with optional timestamps)                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Papers

| Paper | Description | Link |
|-------|-------------|------|
| **Fast Conformer** | 8x downsampling, 2.8x faster than Conformer | [arXiv:2305.05084](https://arxiv.org/abs/2305.05084) |
| **TDT (Token-Duration Transducer)** | Joint token and duration prediction | [arXiv:2304.06795](https://arxiv.org/abs/2304.06795) |
| **Cache-aware Streaming** | Stateful streaming with activation caching | [arXiv:2312.17279](https://arxiv.org/abs/2312.17279) |
| **Granary Dataset** | 670K+ hours across 25 EU languages | [arXiv:2505.13404](https://arxiv.org/abs/2505.13404) |

---

## FastConformer Encoder

FastConformer is an optimized Conformer variant designed for efficiency without sacrificing accuracy.

### Key Innovations

1. **8x Depthwise-Separable Subsampling**: Reduces sequence length 8x (vs 4x in standard Conformer)
2. **Reduced Kernel Size**: Convolution kernel size of 9 (vs 31 in original Conformer)
3. **256 Subsampling Channels**: Optimal balance of speed and quality

### Architecture Details

```python
# FastConformer encoder configuration
encoder_config = {
    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
    "feat_in": 80,                    # Input mel features
    "n_layers": 17,                   # Number of conformer blocks
    "d_model": 512,                   # Model dimension
    
    # 8x Subsampling (key FastConformer innovation)
    "subsampling": "dw_striding",     # Depthwise-separable striding
    "subsampling_factor": 8,          # 8x downsampling
    "subsampling_conv_channels": 256, # Subsampling channels
    
    # Attention configuration
    "self_attention_model": "rel_pos", # Relative positional encoding
    "n_heads": 8,
    "att_context_size": [-1, -1],     # Full context (streaming: [70, 13])
    
    # Convolution module
    "conv_kernel_size": 9,            # Reduced from 31
    "conv_norm_type": "batch_norm",
    
    # Regularization
    "dropout": 0.1,
    "dropout_att": 0.1,
}
```

### Conformer Block Structure

Each conformer block consists of:

```
Input
  │
  ▼
┌────────────────────────────────────────┐
│  Feed-Forward Module (FFN)             │
│  • Linear(d_model, 4*d_model)          │
│  • Swish activation                    │
│  • Dropout                             │
│  • Linear(4*d_model, d_model)          │
│  • Residual connection (0.5x)          │
└────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────┐
│  Multi-Head Self-Attention (MHSA)      │
│  • Relative positional encoding        │
│  • 8 attention heads                   │
│  • Residual connection                 │
└────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────┐
│  Convolution Module                    │
│  • Pointwise Conv (d_model, 2*d_model) │
│  • GLU activation                      │
│  • Depthwise Conv (kernel=9)           │
│  • BatchNorm                           │
│  • Swish + Pointwise Conv              │
│  • Residual connection                 │
└────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────┐
│  Feed-Forward Module (FFN)             │
│  • Same as first FFN                   │
│  • Residual connection (0.5x)          │
└────────────────────────────────────────┘
  │
  ▼
LayerNorm → Output
```

### Model Size Variants

| Model | d_model | n_heads | n_layers | Parameters |
|-------|---------|---------|----------|------------|
| Small | 176 | 4 | 16 | 14M |
| Medium | 256 | 4 | 16 | 32M |
| Large | 512 | 8 | 17 | 120M |
| XLarge | 1024 | 8 | 24 | 616M |
| XXLarge | 1024 | 8 | 42 | 1.2B |

---

## Decoder Architectures

### CTC (Connectionist Temporal Classification)

Non-autoregressive decoder with frame-independent predictions.

```python
decoder_config = {
    "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
    "feat_in": 512,       # Matches encoder d_model
    "num_classes": 1024,  # Vocabulary size
    "vocabulary": [],     # Filled from tokenizer
}
```

**Loss Function:**
```python
# CTC loss with blank token handling
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean_batch', zero_infinity=True)
```

**Decoding:**
```python
# Greedy decoding
predictions = torch.argmax(log_probs, dim=-1)
# Remove blanks and repeated tokens
decoded = ctc_decode(predictions)
```

### RNN-Transducer (RNNT)

Autoregressive decoder with joint network combining encoder and prediction outputs.

```python
# Prediction network (language model)
prednet_config = {
    "pred_hidden": 640,
    "pred_rnn_layers": 1,
    "dropout": 0.2,
}

# Joint network
joint_config = {
    "_target_": "nemo.collections.asr.modules.RNNTJoint",
    "jointnet": {
        "joint_hidden": 640,
        "activation": "relu",
        "dropout": 0.2,
    },
    "fuse_loss_wer": True,
    "fused_batch_size": 4,
}
```

**Architecture:**
```
Encoder Output: [B, T, D_enc]
      │
      └───────────────────┐
                          │
                          ▼
Prediction Network ──► Joint Network ──► Output [B, T, U, V]
      │                   │
      ▼                   │
Previous Token ◄──────────┘
```

### TDT (Token-and-Duration Transducer)

NVIDIA's innovation that jointly predicts tokens AND durations, enabling frame skipping during inference.

```python
# TDT-specific configuration
model_defaults = {
    "tdt_durations": [0, 1, 2, 3, 4],  # Possible frame skips
    "num_tdt_durations": 5,
}

# Joint network with duration outputs
joint_config = {
    "_target_": "nemo.collections.asr.modules.RNNTJoint",
    "num_extra_outputs": 5,  # Duration logits
}

# TDT loss function
loss_config = {
    "loss_name": "tdt",
    "tdt_kwargs": {
        "durations": [0, 1, 2, 3, 4],
        "sigma": 0.02,    # Under-normalization hyperparameter
        "omega": 0.1,     # Weight for regular RNN-T loss
        "fastemit_lambda": 0.0,  # FastEmit for low-latency streaming
    },
}
```

**TDT Inference Speedup:**
- Standard RNNT: Process every encoder frame
- TDT: Skip frames based on predicted duration (up to 2.82x faster)

```python
# TDT decoding strategy
decoding_config = {
    "strategy": "greedy_batch",
    "model_type": "tdt",
    "durations": [0, 1, 2, 3, 4],
    "greedy": {
        "max_symbols": 10,
    },
}
```

### AED (Attention Encoder-Decoder)

Used in Canary models for multitask ASR and translation.

```python
# Transformer decoder for AED models
decoder_config = {
    "_target_": "nemo.collections.asr.modules.TransformerDecoder",
    "num_layers": 8,
    "d_model": 512,
    "num_attention_heads": 8,
    "ff_expansion_factor": 4,
    "dropout": 0.1,
}
```

---

## Model Families

### Parakeet Models

**Parakeet TDT 0.6B v2** - English ASR leader ([HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2))

| Spec | Value |
|------|-------|
| Architecture | FastConformer-TDT |
| Parameters | 600M |
| Encoder Layers | 24 |
| d_model | 1024 |
| Training Data | 120K hours (Granary) |
| Average WER | 6.05% |
| RTFx | 3380 |

**Python Usage:**
```python
import nemo.collections.asr as nemo_asr

# Load pretrained model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Basic transcription
output = asr_model.transcribe(['audio.wav'])
print(output[0].text)

# Transcription with timestamps
output = asr_model.transcribe(['audio.wav'], timestamps=True)
word_timestamps = output[0].timestamp['word']
segment_timestamps = output[0].timestamp['segment']

for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
```

**Parakeet TDT 0.6B v3** - Multilingual ([HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3))

| Spec | Value |
|------|-------|
| Languages | 25 EU languages |
| Training Data | 670K hours |
| Average WER (FLEURS) | 11.97% |

```python
# Multilingual transcription (auto language detection)
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)
output = asr_model.transcribe(['german_audio.wav'])
```

**Model Variants:**

| Model | Decoder | Params | Avg WER | Use Case |
|-------|---------|--------|---------|----------|
| [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | TDT | 600M | 6.05% | Best English accuracy |
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TDT | 600M | 6.34% | Multilingual (25 langs) |
| [parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CTC | 1.1B | 7.1% | High-throughput batch |
| [parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) | RNNT | 1.1B | 6.9% | Best streaming accuracy |

### Canary Models

Encoder-decoder architecture for multitask ASR + Translation.

**Canary 1B v2** ([HuggingFace](https://huggingface.co/nvidia/canary-1b-v2))

| Spec | Value |
|------|-------|
| Architecture | FastConformer Encoder + Transformer Decoder |
| Parameters | 978M (32 encoder + 8 decoder layers) |
| Languages | 25 EU languages |
| Tasks | ASR, X→EN translation, EN→X translation |
| Training Data | 1.7M hours |

```python
from nemo.collections.asr.models import ASRModel

# Load Canary model
asr_ast_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

# Transcription
output = asr_ast_model.transcribe(
    ['audio.wav'], 
    source_lang='en', 
    target_lang='en'
)
print(output[0].text)

# Translation (English to French)
output = asr_ast_model.transcribe(
    ['english_audio.wav'],
    source_lang='en',
    target_lang='fr'
)
print(output[0].text)

# Translation with timestamps
output = asr_ast_model.transcribe(
    ['audio.wav'],
    source_lang='en',
    target_lang='fr',
    timestamps=True
)
for stamp in output[0].timestamp['segment']:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
```

### Nemotron Speech ASR

Cache-aware streaming model for real-time voice agents.

**Nemotron Speech Streaming 0.6B** ([HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b))

| Spec | Value |
|------|-------|
| Architecture | FastConformer Cache-Aware RNNT |
| Parameters | 600M |
| Training Data | 285K hours |
| Latency Modes | 80ms, 160ms, 560ms, 1120ms |
| Max Concurrency (H100) | 560 streams |

```python
import nemo.collections.asr as nemo_asr

# Load streaming model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)

# Configure streaming latency
# att_context_size = [left_context, right_context] in 80ms frames
# [70, 0]  -> 80ms latency
# [70, 1]  -> 160ms latency  
# [70, 6]  -> 560ms latency
# [70, 13] -> 1120ms latency
```

**Streaming Inference Script:**
```bash
python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=nvidia/nemotron-speech-streaming-en-0.6b \
    dataset_manifest=test.json \
    batch_size=16 \
    att_context_size="[70,13]" \
    output_path=output/
```

### Multitalker Models

Multi-speaker transcription with speaker-specific output.

**Multitalker Parakeet Streaming 0.6B** ([HuggingFace](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1))

```python
from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel
import torch

# Load diarization and ASR models
diar_model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1"
).eval().to("cuda")

asr_model = ASRModel.from_pretrained(
    "nvidia/multitalker-parakeet-streaming-0.6b-v1"  
).eval().to("cuda")

# Multi-speaker transcription
# Each speaker gets a separate model instance with speaker-specific kernels
```

---

## Python Implementation

### Installation

```bash
# Install NeMo with ASR support
pip install nemo_toolkit['asr']

# Or install from source
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Basic Transcription

```python
import nemo.collections.asr as nemo_asr

# Load any NeMo ASR model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Single file transcription
output = asr_model.transcribe(['audio.wav'])
print(output[0].text)

# Batch transcription
outputs = asr_model.transcribe(['audio1.wav', 'audio2.wav', 'audio3.wav'])
for out in outputs:
    print(out.text)
```

### Transcription with Timestamps

```python
# Enable timestamp extraction
output = asr_model.transcribe(['audio.wav'], timestamps=True)

# Access different timestamp levels
char_timestamps = output[0].timestamp['char']
word_timestamps = output[0].timestamp['word']
segment_timestamps = output[0].timestamp['segment']

# Print word-level timestamps
for stamp in word_timestamps:
    print(f"{stamp['start']:.2f}s - {stamp['end']:.2f}s: {stamp['word']}")
```

### Long-Form Audio Transcription

```python
# Enable local attention for long audio (up to 11 hours on A100 80GB)
asr_model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[256, 256]  # 256 frames left and right context
)

# Enable chunking for subsampling module
asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

# Transcribe very long audio
output = asr_model.transcribe(['long_audio.wav'])  # Up to 11+ hours
```

### Decoding Configuration

```python
from omegaconf import OmegaConf, open_dict

# Access current decoding config
decoding_cfg = asr_model.cfg.decoding

# Modify decoding parameters
with open_dict(decoding_cfg):
    decoding_cfg.strategy = "greedy_batch"
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    decoding_cfg.segment_seperators = [".", "?", "!"]
    decoding_cfg.word_seperator = " "
    
asr_model.change_decoding_strategy(decoding_cfg)
```

### Model Classes

| Task | Model Class | Example Model |
|------|-------------|---------------|
| CTC (char) | `EncDecCTCModel` | conformer_ctc_char |
| CTC (BPE) | `EncDecCTCModelBPE` | parakeet-ctc-1.1b |
| RNNT (char) | `EncDecRNNTModel` | conformer_transducer_char |
| RNNT (BPE) | `EncDecRNNTBPEModel` | parakeet-rnnt-1.1b |
| TDT (BPE) | `EncDecRNNTBPEModel` | parakeet-tdt-0.6b-v2 |
| Hybrid | `EncDecHybridRNNTCTCBPEModel` | hybrid_transducer_ctc |
| AED | `EncDecMultiTaskModel` | canary-1b-v2 |

```python
# Instantiate specific model classes
from nemo.collections.asr.models import (
    EncDecCTCModelBPE,
    EncDecRNNTBPEModel,
    EncDecHybridRNNTCTCBPEModel,
)

# CTC model
ctc_model = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-1.1b")

# RNNT model
rnnt_model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-rnnt-1.1b")

# Hybrid model (switch between CTC and RNNT decoding)
hybrid_model = EncDecHybridRNNTCTCBPEModel.from_pretrained("model.nemo")
hybrid_model.change_decoding_strategy(decoder_type='ctc')   # Use CTC decoder
hybrid_model.change_decoding_strategy(decoder_type='rnnt')  # Use RNNT decoder
```

---

## Configuration Reference

### Full FastConformer-CTC Config

```yaml
name: "FastConformer-CTC-BPE"

model:
  sample_rate: 16000
  log_prediction: true
  ctc_reduction: 'mean_volume'
  
  # Tokenizer
  tokenizer:
    dir: ???  # Path to tokenizer directory
    type: bpe  # 'bpe' (SentencePiece) or 'wpe' (WordPiece)
  
  # Audio Preprocessing  
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    normalize: "per_feature"
    window_size: 0.025
    window_stride: 0.01
    window: "hann"
    features: 80
    n_fft: 512
    dither: 0.00001
    
  # SpecAugment
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2
    time_masks: 10
    freq_width: 27
    time_width: 0.05

  # FastConformer Encoder
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    n_layers: 17
    d_model: 512
    
    # 8x Subsampling
    subsampling: dw_striding
    subsampling_factor: 8
    subsampling_conv_channels: 256
    
    # Attention
    self_attention_model: rel_pos
    n_heads: 8
    att_context_size: [-1, -1]
    xscaling: true
    untie_biases: true
    pos_emb_max_len: 5000
    
    # Convolution
    conv_kernel_size: 9
    conv_norm_type: batch_norm
    
    # Regularization
    dropout: 0.1
    dropout_att: 0.1

  # CTC Decoder
  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoder
    feat_in: null  # Auto-set from encoder
    num_classes: -1  # Auto-set from tokenizer
    vocabulary: []

  # Optimizer
  optim:
    name: adamw
    lr: 1e-3
    betas: [0.9, 0.98]
    weight_decay: 1e-3
    sched:
      name: CosineAnnealing
      warmup_steps: 15000
      min_lr: 1e-4

trainer:
  devices: -1
  max_epochs: 1000
  precision: bf16
  gradient_clip_val: 1.0
```

### FastConformer-TDT Config

```yaml
name: "FastConformer-Hybrid-TDT-CTC-BPE"

model:
  model_defaults:
    enc_hidden: 512
    pred_hidden: 640
    joint_hidden: 640
    tdt_durations: [0, 1, 2, 3, 4]
    num_tdt_durations: 5

  # Prediction Network
  decoder:
    _target_: nemo.collections.asr.modules.RNNTDecoder
    blank_as_pad: true
    prednet:
      pred_hidden: 640
      pred_rnn_layers: 1
      dropout: 0.2

  # Joint Network
  joint:
    _target_: nemo.collections.asr.modules.RNNTJoint
    fuse_loss_wer: true
    fused_batch_size: 4
    jointnet:
      joint_hidden: 640
      activation: "relu"
      dropout: 0.2
    num_extra_outputs: 5  # Duration logits

  # TDT Decoding
  decoding:
    strategy: "greedy_batch"
    model_type: "tdt"
    durations: [0, 1, 2, 3, 4]
    greedy:
      max_symbols: 10

  # TDT Loss
  loss:
    loss_name: "tdt"
    tdt_kwargs:
      durations: [0, 1, 2, 3, 4]
      sigma: 0.02
      omega: 0.1
      fastemit_lambda: 0.0

  # Auxiliary CTC decoder
  aux_ctc:
    ctc_loss_weight: 0.3
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
```

### Cache-Aware Streaming Config

```yaml
model:
  encoder:
    # Streaming attention context
    # [left_context, right_context] in frames (80ms each)
    att_context_size: [70, 13]  # 5.6s left, 1.04s right
    att_context_style: regular
    
    # Causal convolutions for streaming
    conv_context_size: causal
    causal_downsampling: true
    
  # Multiple latency modes (optional)
  # att_context_size: [[70,13], [70,6], [70,1], [70,0]]
  # att_context_probs: [0.25, 0.25, 0.25, 0.25]
```

---

## Streaming ASR

### Cache-Aware vs Buffered Streaming

| Feature | Buffered Streaming | Cache-Aware Streaming |
|---------|-------------------|----------------------|
| Computation | Overlapping windows | Non-overlapping chunks |
| Memory | Grows with overlap | Fixed cache size |
| Latency drift | Yes, under load | No |
| Concurrency | ~180 streams (H100) | ~560 streams (H100) |

### Cache-Aware Architecture

```
Traditional Buffered:
┌───────────────────────────────────────────────┐
│ Audio: [====chunk1====][===overlap===][====chunk2====]
│                         ↑ Re-computed
└───────────────────────────────────────────────┘

Cache-Aware:
┌───────────────────────────────────────────────┐
│ Audio: [====chunk1====][====chunk2====][====chunk3====]
│              ↓              ↓              ↓
│           Cache→        Update→        Update→
│        (stored)      (reused)       (reused)
└───────────────────────────────────────────────┘
```

### Streaming Inference Python

```python
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

# Configure streaming buffer
streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=asr_model,
    online_normalization=True,
    pad_and_drop_preencoded=True,
)

# Append audio chunks
streaming_buffer.append_audio_file(
    audio_filepath="audio.wav",
    stream_id=-1
)

# Process chunks
for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
    with torch.inference_mode():
        # Process chunk through model
        outputs = asr_model.forward(
            input_signal=chunk_audio,
            input_signal_length=chunk_lengths,
        )
        # Get partial transcription
        partial_text = asr_model.decoding.decode(outputs)
```

### Runtime Latency Configuration

```python
# Switch latency modes at runtime (no retraining needed)
asr_model.encoder.set_default_att_context_size([70, 0])   # 80ms (lowest latency)
asr_model.encoder.set_default_att_context_size([70, 1])   # 160ms
asr_model.encoder.set_default_att_context_size([70, 6])   # 560ms
asr_model.encoder.set_default_att_context_size([70, 13])  # 1120ms (best accuracy)
```

---

## Training and Fine-Tuning

### Training Script

```bash
# Train FastConformer-CTC
python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path=conf/fastconformer \
    --config-name=fast-conformer_ctc_bpe.yaml \
    model.train_ds.manifest_filepath=train_manifest.json \
    model.validation_ds.manifest_filepath=val_manifest.json \
    model.tokenizer.dir=/path/to/tokenizer \
    trainer.devices=8 \
    trainer.max_epochs=100 \
    trainer.precision=bf16

# Train FastConformer-TDT
python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=conf/fastconformer/hybrid_transducer_ctc \
    --config-name=fastconformer_hybrid_tdt_ctc_bpe.yaml \
    model.train_ds.manifest_filepath=train_manifest.json \
    model.validation_ds.manifest_filepath=val_manifest.json \
    model.tokenizer.dir=/path/to/tokenizer
```

### Manifest Format

```json
{"audio_filepath": "/path/to/audio1.wav", "text": "transcription text", "duration": 10.5}
{"audio_filepath": "/path/to/audio2.wav", "text": "another transcription", "duration": 8.2}
```

### Fine-Tuning Pretrained Models

```python
import nemo.collections.asr as nemo_asr

# Load pretrained model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Update dataset configs
asr_model.cfg.train_ds.manifest_filepath = "train_manifest.json"
asr_model.cfg.validation_ds.manifest_filepath = "val_manifest.json"

# Setup data loaders
asr_model.setup_training_data(asr_model.cfg.train_ds)
asr_model.setup_validation_data(asr_model.cfg.validation_ds)

# Fine-tune with PyTorch Lightning
import pytorch_lightning as pl
trainer = pl.Trainer(
    devices=1,
    max_epochs=10,
    precision="bf16-mixed",
)
trainer.fit(asr_model)

# Save fine-tuned model
asr_model.save_to("finetuned_model.nemo")
```

### Building Custom Tokenizer

```bash
# Create SentencePiece BPE tokenizer
python scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest=train_manifest.json \
    --vocab_size=1024 \
    --tokenizer=spe \
    --spe_type=unigram \
    --output_dir=/path/to/tokenizer
```

---

## Resources and Links

### Official Documentation
- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [NeMo ASR Models Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [NeMo ASR Configuration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html)

### Source Code
- [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [ASR Examples](https://github.com/NVIDIA/NeMo/tree/main/examples/asr)
- [ASR Configs](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf)

### HuggingFace Models
| Model | Link |
|-------|------|
| Parakeet TDT 0.6B v2 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 |
| Parakeet TDT 0.6B v3 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| Parakeet CTC 1.1B | https://huggingface.co/nvidia/parakeet-ctc-1.1b |
| Parakeet RNNT 1.1B | https://huggingface.co/nvidia/parakeet-rnnt-1.1b |
| Canary 1B v2 | https://huggingface.co/nvidia/canary-1b-v2 |
| Nemotron Speech 0.6B | https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |
| Multitalker Parakeet | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 |

### HuggingFace Datasets
- [NVIDIA Granary](https://huggingface.co/datasets/nvidia/Granary) - 670K+ hours across 25 EU languages

### Research Papers
| Paper | Topic | Link |
|-------|-------|------|
| Fast Conformer | 8x downsampling, efficient attention | [arXiv:2305.05084](https://arxiv.org/abs/2305.05084) |
| TDT | Token-and-Duration Transducer | [arXiv:2304.06795](https://arxiv.org/abs/2304.06795) |
| Cache-aware Streaming | Stateful streaming ASR | [arXiv:2312.17279](https://arxiv.org/abs/2312.17279) |
| Granary Dataset | Multilingual speech dataset | [arXiv:2505.13404](https://arxiv.org/abs/2505.13404) |
| Conformer | Original architecture | [arXiv:2005.08100](https://arxiv.org/abs/2005.08100) |

### Tutorials
- [ASR with NeMo](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb)
- [Introduction to Transducers](https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Intro_to_Transducers.ipynb)
- [Cache-Aware Streaming Demo](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb)
- [Canary Multitask Model](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Canary_Multitask_Speech_Model.ipynb)
- [Streaming Multitalker ASR](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb)

### Deployment
- [NVIDIA Riva](https://developer.nvidia.com/riva) - Production deployment SDK
- [NVIDIA NIM](https://build.nvidia.com/explore/speech) - Inference microservices
- [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

### Developer Blog Posts
- [NVIDIA Speech AI Models Deliver Industry-Leading Accuracy](https://developer.nvidia.com/blog/nvidia-speech-ai-models-deliver-industry-leading-accuracy-and-performance/)
- [Pushing the Boundaries with Parakeet ASR Models](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/)
- [Cache-Aware Streaming ASR for Voice Agents](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)

---

*Last updated: January 2026*
