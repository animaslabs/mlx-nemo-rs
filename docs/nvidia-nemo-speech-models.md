# NVIDIA NeMo Speech-to-Text Models: Architecture Guide

This document provides a comprehensive overview of NVIDIA's NeMo speech-to-text models, focusing on model architectures, encoders (Conformer/FastConformer), decoders (CTC, RNN-T, TDT), and streaming capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Model Families](#model-families)
3. [Encoder Architectures](#encoder-architectures)
4. [Decoder Architectures](#decoder-architectures)
5. [Streaming and Real-Time Models](#streaming-and-real-time-models)
6. [Model Comparison](#model-comparison)
7. [References](#references)

---

## Overview

NVIDIA NeMo provides state-of-the-art Automatic Speech Recognition (ASR) models through its open-source toolkit. The models follow a naming convention: `stt_{language}_{encoder}_{decoder}_{size}{_optional_descriptor}`.

Key model families include:
- **Nemotron Speech**: The latest streaming-first ASR models optimized for real-time voice agents
- **Parakeet**: High-performance English ASR models with FastConformer encoder
- **Canary**: Multilingual, multitask models supporting ASR and speech translation
- **Conformer/FastConformer**: Various encoder-decoder combinations for different use cases

---

## Model Families

### Nemotron Speech (NEW - January 2026)

**Nemotron Speech** is NVIDIA's newest model family, purpose-built for real-time voice agent applications. The flagship model `nemotron-speech-streaming-en-0.6b` introduces **cache-aware streaming** architecture that fundamentally changes how streaming ASR works.

| Model | Parameters | Decoder | Key Features |
|-------|------------|---------|--------------|
| nemotron-speech-streaming-en-0.6b | 600M | Cache-Aware RNN-T | Native streaming, configurable latency |
| parakeet_realtime_eou_120m-v1 | 120M | RNN-T + EOU | End-of-utterance detection for turn-taking |

**Why Nemotron Speech is Different:**

Traditional "streaming" ASR uses **buffered inference** - repeatedly re-processing overlapping audio windows. This wastes computation and causes **latency drift** at scale. Nemotron Speech uses **cache-aware streaming** where each audio frame is processed exactly once:

```
┌─────────────────────────────────────────────────────────────┐
│           Traditional Buffered Streaming (Inefficient)      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Window 1: [████████████]                                   │
│  Window 2:     [████████████]      ← Overlap recomputed!   │
│  Window 3:         [████████████]  ← More overlap!         │
│                                                             │
│  Problem: Same audio processed multiple times               │
│  Result: Latency drift under load, wasted GPU compute      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           Cache-Aware Streaming (Nemotron Speech)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chunk 1: [████]                                            │
│  Chunk 2:      [████]  ← Only new audio, cache reused      │
│  Chunk 3:           [████]  ← Zero overlap computation     │
│                                                             │
│  Caches: Self-attention + convolution states preserved     │
│  Result: Linear scaling, stable latency at high concurrency│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovations:**

1. **Cache-Aware Architecture**: Maintains internal cache of encoder representations across all self-attention and convolution layers
2. **Dynamic Latency Modes**: Runtime-configurable chunk sizes (80ms, 160ms, 560ms, 1120ms) - no retraining required
3. **3x Higher Throughput**: Supports 560 concurrent streams vs 180 baseline on H100 (at 320ms chunk)
4. **Zero Latency Drift**: Stable performance even at peak capacity
5. **24ms Time-to-Final**: Median transcription finalization time

**Performance by Latency Mode:**

| Chunk Size | Latency | Avg WER | Use Case |
|------------|---------|---------|----------|
| 80ms | Ultra-low | 8.53% | Real-time voice agents |
| 160ms | Low | 7.84% | Interactive applications |
| 560ms | Medium | 7.22% | Balanced accuracy/latency |
| 1120ms | Higher | 7.16% | Best accuracy |

**Real-World Validation:**
- **Modal**: 127 concurrent WebSocket clients with stable 182ms median delay over 3-minute streams
- **Daily**: Full voice-to-voice loop under 900ms with Nemotron Speech + Nemotron LLM + Magpie TTS

### Parakeet Models

Parakeet is NVIDIA's flagship ASR model family, featuring FastConformer encoders with various decoder options. The **Parakeet-TDT-0.6B-V2** currently sits at the top of the HuggingFace Open ASR Leaderboard.

| Model | Parameters | Decoder | Key Features |
|-------|------------|---------|--------------|
| parakeet-tdt-0.6b-v2 | 600M | TDT | Best accuracy, word-level timestamps |
| parakeet-ctc-1.1b | 1.1B | CTC | Fast inference, good accuracy |
| parakeet-rnnt-1.1b | 1.1B | RNN-T | Streaming capable |
| parakeet-tdt-1.1b | 1.1B | TDT | High accuracy with timestamps |
| parakeet-tdt_ctc-110m | 110M | Hybrid TDT-CTC | Lightweight, dual decoder |

**Key Features:**
- Automatic punctuation and capitalization
- Word-level timestamp predictions
- Support for audio up to 24 minutes in a single pass
- RTFx of ~3380 on HF-Open-ASR leaderboard (batch size 128)

### Canary Models

Canary models are encoder-decoder architectures designed for multilingual ASR and speech translation (AST).

| Model | Parameters | Languages | Tasks |
|-------|------------|-----------|-------|
| canary-1b-v2 | 978M | 25 EU languages | ASR, X→En, En→X translation |
| canary-1b-flash | 1B | 4 languages | ASR, translation |
| canary-180m-flash | 180M | 4 languages | Lightweight ASR/translation |

**Architecture:**
- FastConformer Encoder (32 layers)
- Transformer Decoder (8 layers)
- Unified SentencePiece tokenizer (16,384 tokens)

---

## Encoder Architectures

### Conformer

The Conformer architecture, introduced by Google in 2020, combines the strengths of self-attention (Transformers) and convolutions to capture both global and local audio patterns.

```
┌─────────────────────────────────────────────────────────────┐
│                    Conformer Block                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Feed-Forward Module (½)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Multi-Head Self-Attention Module            │   │
│  │    (with relative positional encoding)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Convolution Module                     │   │
│  │    (Depthwise separable convolution)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Feed-Forward Module (½)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Layer Norm                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**
1. **Multi-Head Self-Attention**: Captures long-range global context
2. **Convolution Module**: Captures local patterns efficiently
3. **Feed-Forward Modules**: Non-linear transformations (split in half, before and after attention)
4. **Positional Encoding**: Supports both absolute and relative (Transformer-XL style)

### FastConformer

FastConformer is NVIDIA's optimized version of Conformer, achieving **2.8x faster** training and inference without significant accuracy loss.

**Key Optimizations:**

1. **8x Depthwise Convolutional Downsampling**
   - Original Conformer: 4x downsampling
   - FastConformer: 8x downsampling with 256 channels
   - Reduces sequence length early, improving efficiency

2. **Reduced Convolution Kernel Size**
   - Kernel size reduced from 31 to 9 in conformer blocks
   - Maintains accuracy while improving speed

3. **Limited Context Attention (Optional)**
   - Longformer-style attention for very long audio (up to 11 hours)
   - Global tokens aggregate information from beyond local context
   - Enables transcription of audio >1 hour with 256 subsampling channels

```
┌─────────────────────────────────────────────────────────────┐
│                  FastConformer Encoder                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Input (16kHz)                                        │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     8x Depthwise Conv Downsampling (256 ch)         │   │
│  │     (vs 4x in original Conformer)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     N × FastConformer Blocks                        │   │
│  │     - Multi-Head Attention (full or limited)        │   │
│  │     - Conv kernel size: 9 (vs 31 in Conformer)      │   │
│  │     - Feed-Forward modules                          │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  Encoded Features                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Performance Comparison:**

| Metric | Conformer | FastConformer |
|--------|-----------|---------------|
| Training Speed | 1x | 2.8x |
| Inference Speed | 1x | 2.8x |
| Max Audio Length | ~40 min | 11+ hours (with limited attention) |
| Downsampling | 4x | 8x |
| Conv Kernel Size | 31 | 9 |

---

## Decoder Architectures

NeMo supports four primary decoder types for ASR:

### 1. CTC (Connectionist Temporal Classification)

CTC is a **non-autoregressive** decoder that independently predicts tokens at each encoder output frame.

```
┌─────────────────────────────────────────────────────────────┐
│                      CTC Decoder                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Encoder Output: [T frames × D dimensions]                  │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Linear Projection                       │   │
│  │         (D → Vocabulary Size + blank)               │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Softmax (per frame)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  Frame predictions: [h, h, e, e, -, l, l, l, o, -]         │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         CTC Decoding (collapse & remove blank)       │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  Output: "hello"                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- **Non-autoregressive**: All frames decoded in parallel
- **Fast inference**: O(T) complexity
- **Conditional independence**: Each frame prediction is independent
- **Blank token**: Special token to handle alignment and repetitions

**Pros:**
- Very fast inference
- Simple architecture
- Good for streaming with greedy decoding

**Cons:**
- No language model integration during decoding
- Conditional independence assumption limits accuracy

### 2. RNN-T (RNN Transducer / Transducer)

RNN-T is an **autoregressive** decoder that conditions predictions on both acoustic features and previously predicted tokens.

```
┌─────────────────────────────────────────────────────────────┐
│                    RNN-T Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                    ┌─────────────┐        │
│  │   Encoder   │                    │  Prediction │        │
│  │(FastConformer)│                   │   Network   │        │
│  │             │                    │   (LSTM)    │        │
│  └──────┬──────┘                    └──────┬──────┘        │
│         │                                  │                │
│         │  Encoder Output                  │ Label Embedding│
│         │  [T × D_enc]                     │ [U × D_pred]   │
│         ↓                                  ↓                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Joint Network                      │   │
│  │                                                      │   │
│  │   joint(t,u) = f(encoder[t], prediction[u])         │   │
│  │                                                      │   │
│  │   Output: [T × U × Vocab+blank]                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│            Transducer Loss / Greedy Decoding               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Decoding Process:**
1. Start at position (t=0, u=0)
2. Compute joint output for current (t, u)
3. If blank predicted: move to (t+1, u) - advance in time
4. If token predicted: move to (t, u+1) - advance in label sequence
5. Repeat until end of encoder output

**Characteristics:**
- **Autoregressive**: Each prediction depends on previous tokens
- **Streaming capable**: Can emit tokens as audio arrives
- **Joint network**: Combines acoustic and language information

**Pros:**
- Better accuracy than CTC (language model effect)
- Natural streaming support
- Handles alignment implicitly

**Cons:**
- Slower than CTC (sequential decoding)
- More complex training

### 3. TDT (Token-and-Duration Transducer)

TDT is NVIDIA's innovation that **jointly predicts tokens AND durations**, enabling significant speedups over standard RNN-T.

```
┌─────────────────────────────────────────────────────────────┐
│                    TDT Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                    ┌─────────────┐        │
│  │   Encoder   │                    │  Prediction │        │
│  │(FastConformer)│                   │   Network   │        │
│  └──────┬──────┘                    └──────┬──────┘        │
│         │                                  │                │
│         ↓                                  ↓                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Joint Network                      │   │
│  │                                                      │   │
│  │        ┌──────────────┬──────────────┐              │   │
│  │        │              │              │              │   │
│  │        ↓              ↓              │              │   │
│  │   ┌─────────┐   ┌──────────┐        │              │   │
│  │   │  Token  │   │ Duration │        │              │   │
│  │   │  Head   │   │   Head   │        │              │   │
│  │   └────┬────┘   └────┬─────┘        │              │   │
│  │        │             │              │              │   │
│  │        ↓             ↓              │              │   │
│  │   P(token|t,u)  P(duration|t,u)     │              │   │
│  │                                      │              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Inference: Skip 'duration' frames after each token        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation:**
- Two output heads: **Token** and **Duration**
- Duration predicts how many encoder frames the token covers
- During inference, skip frames based on predicted duration

**Speed Improvements:**

| Task | TDT vs RNN-T Speedup |
|------|---------------------|
| Speech Recognition | Up to 2.82x |
| Speech Translation | Up to 2.27x |
| Intent Classification | Up to 1.28x |

**Characteristics:**
- **Accurate timestamps**: Duration predictions enable precise word timing
- **Faster inference**: Frame skipping reduces computation
- **Better accuracy**: Often matches or exceeds RNN-T

### 4. AED (Attention-based Encoder-Decoder)

AED uses a Transformer decoder with cross-attention to the encoder, similar to seq2seq models.

```
┌─────────────────────────────────────────────────────────────┐
│                    AED Architecture                         │
│               (Used in Canary models)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastConformer Encoder                   │   │
│  │                  (32 layers)                         │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         │ Encoded Audio                     │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Transformer Decoder (8 layers)            │   │
│  │                                                      │   │
│  │    [Task Tokens] → [<src_lang>] → [<tgt_lang>]      │   │
│  │                         ↓                            │   │
│  │    ┌─────────────────────────────────────────┐      │   │
│  │    │     Masked Self-Attention               │      │   │
│  │    └─────────────────────────────────────────┘      │   │
│  │                         ↓                            │   │
│  │    ┌─────────────────────────────────────────┐      │   │
│  │    │     Cross-Attention to Encoder          │      │   │
│  │    └─────────────────────────────────────────┘      │   │
│  │                         ↓                            │   │
│  │    ┌─────────────────────────────────────────┐      │   │
│  │    │     Feed-Forward Network                │      │   │
│  │    └─────────────────────────────────────────┘      │   │
│  │                         ↓                            │   │
│  │              Token Predictions                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Used For:**
- Multilingual ASR
- Speech Translation (X→En, En→X)
- Multitask models (Canary)

**Characteristics:**
- Flexible for multiple tasks via prompt tokens
- Full attention over encoder output
- Supports beam search decoding

### 5. Hybrid Transducer-CTC

Hybrid models combine both CTC and RNN-T/TDT decoders, enabling:
- Faster CTC training convergence
- Flexibility to use either decoder at inference
- Single model for different latency requirements

```
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Transducer-CTC Architecture             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastConformer Encoder                   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │                     │                        │
│              ↓                     ↓                        │
│  ┌─────────────────┐   ┌─────────────────────────────┐    │
│  │   CTC Decoder   │   │  Transducer/TDT Decoder     │    │
│  │   (Linear +     │   │  (Prediction Net +          │    │
│  │    Softmax)     │   │   Joint Network)            │    │
│  └─────────────────┘   └─────────────────────────────┘    │
│              │                     │                        │
│              ↓                     ↓                        │
│         CTC Loss              Transducer Loss              │
│                                                             │
│  Training: Combined loss (weighted sum)                    │
│  Inference: Choose decoder based on requirements           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Streaming and Real-Time Models

### Nemotron Speech Cache-Aware Streaming (Recommended)

The **Nemotron Speech** model family represents NVIDIA's state-of-the-art approach to streaming ASR, designed specifically for voice agent workloads at scale.

**Architecture:**
- FastConformer encoder with 24 layers
- RNN-T decoder with cache-aware design
- 8x depthwise-separable convolutional downsampling
- 600M parameters

```
┌─────────────────────────────────────────────────────────────┐
│           Nemotron Speech Streaming Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Stream (16kHz) ───→ Audio Buffer                    │
│                               │                             │
│                               ↓                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Cache-Aware FastConformer Encoder            │   │
│  │                                                      │   │
│  │   ┌──────────────────────────────────────────┐      │   │
│  │   │  Self-Attention Cache (all 24 layers)    │      │   │
│  │   │  - Left context: 70 frames (5.6s)        │      │   │
│  │   │  - Cached activations reused each step   │      │   │
│  │   └──────────────────────────────────────────┘      │   │
│  │                                                      │   │
│  │   ┌──────────────────────────────────────────┐      │   │
│  │   │  Convolution Cache (all layers)          │      │   │
│  │   │  - Causal conv states preserved          │      │   │
│  │   │  - No redundant computation              │      │   │
│  │   └──────────────────────────────────────────┘      │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                               │                             │
│                               ↓                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RNN-T Decoder                           │   │
│  │   - Prediction network (LSTM)                       │   │
│  │   - Joint network                                   │   │
│  │   - Greedy or beam search decoding                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                               │                             │
│                               ↓                             │
│              Streaming Transcriptions                       │
│              (with punctuation & capitalization)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Configuring Latency Modes:**

Latency is controlled by `att_context_size = [left_context, right_context]` measured in 80ms frames:

```python
# Ultra-low latency (80ms chunks)
att_context_size = [70, 0]   # chunk_size = 1 frame

# Low latency (160ms chunks)  
att_context_size = [70, 1]   # chunk_size = 2 frames

# Balanced (560ms chunks)
att_context_size = [70, 6]   # chunk_size = 7 frames

# Best accuracy (1120ms chunks)
att_context_size = [70, 13]  # chunk_size = 14 frames
```

**Scaling Benefits:**

| Hardware | Baseline (Buffered) | Nemotron Speech | Improvement |
|----------|---------------------|-----------------|-------------|
| NVIDIA H100 | 180 streams | 560 streams | 3x |
| NVIDIA RTX A5000 | ~50 streams | 250+ streams | 5x |
| NVIDIA DGX B200 | Varies | 2x baseline | 2x |

### Legacy Cache-Aware Streaming Conformer

For applications not requiring the newest Nemotron Speech models, NeMo also provides cache-aware streaming with earlier Parakeet models:

**Context Modes:**

| Mode | Description | Latency | Accuracy |
|------|-------------|---------|----------|
| Fully Causal | Zero look-ahead | Lowest | Lower |
| Regular Look-ahead | Small fixed right context | Medium | Medium |
| Chunk-aware | Context within chunks | Low | High |

**Configuration Example:**
```yaml
# Chunk-aware streaming config
encoder:
  att_context_size: [70, 13]  # [left, right] in frames
  conv_context_size: causal
```

### Canary Streaming Modes

Canary models support two streaming policies:

**1. Wait-K Policy**
- Predicts one token per new audio chunk
- Higher latency, higher accuracy
- Best with large/infinite left context

**2. AlignAtt Policy**
- Uses cross-attention to determine when to emit tokens
- Lower latency, suitable for production
- Can predict multiple tokens per chunk

```
┌─────────────────────────────────────────────────────────────┐
│              Canary Streaming Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Stream ─────→ [chunk_1][chunk_2][chunk_3]...        │
│                            │                                │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Sliding Window Encoder                     │   │
│  │   left_context_secs: 10.0                           │   │
│  │   chunk_secs: 1.0-2.0                               │   │
│  │   right_context_secs: 0.5-2.0                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Streaming Policy                           │   │
│  │   - Wait-K: emit after K chunks                     │   │
│  │   - AlignAtt: emit based on attention threshold     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ↓                                │
│               Partial Transcriptions                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Multitalker Streaming ASR

NeMo also provides streaming models for multi-speaker scenarios:
- One model instance per speaker
- Speaker kernel injection for adaptation
- Handles fully overlapped speech

---

## Model Comparison

### Accuracy vs Speed Trade-offs

| Model | Decoder | Avg WER (OpenASR) | RTFx | Streaming | Best For |
|-------|---------|-------------------|------|-----------|----------|
| parakeet-tdt-0.6b-v2 | TDT | 6.05% | 3380 | No | Best offline accuracy |
| nemotron-speech-streaming-en-0.6b | Cache-RNN-T | 7.16% | High | **Native** | Voice agents at scale |
| parakeet-ctc-1.1b | CTC | ~6.5% | Higher | Partial | Fast batch inference |
| parakeet-rnnt-1.1b | RNN-T | ~6.3% | Lower | Yes | Streaming (legacy) |
| canary-1b-v2 | AED | 7.15% | 749 | Yes | Multilingual/Translation |

### Decoder Selection Guide

| Requirement | Recommended Model/Decoder |
|-------------|---------------------------|
| Voice agents at scale | **Nemotron Speech** (Cache-aware RNN-T) |
| Best offline accuracy | Parakeet TDT |
| Fastest batch inference | Parakeet CTC |
| Word timestamps | Parakeet TDT |
| Multilingual/Translation | Canary (AED) |
| Low latency streaming | Nemotron Speech or Canary with AlignAtt |
| Turn-taking detection | parakeet_realtime_eou (RNN-T + EOU) |

### Training Data Scale

| Dataset | Hours | Description |
|---------|-------|-------------|
| NeMo ASR Set 3.0 | ~10,000 | Human-transcribed, high quality |
| Granary | ~120,000 | Pseudo-labeled + human mix |
| Full Training (Parakeet v2) | ~120,000 | Combined dataset |
| Nemotron Speech Training | ~285,000 | Granary + additional corpora |

---

## References

### Papers

1. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
   - arXiv: 2005.08100

2. **FastConformer**: Rekesh et al., "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition" (2023)
   - arXiv: 2305.05084
   - Key contribution: 2.8x speedup, 8x downsampling, limited context attention

3. **TDT**: Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations" (2023)
   - arXiv: 2304.06795
   - Key contribution: Joint token-duration prediction, up to 2.82x faster inference

4. **Cache-Aware Streaming**: "Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition" (2023)
   - arXiv: 2312.17279
   - Key contribution: Cache-aware architecture for efficient streaming without redundant computation

5. **Granary Dataset**: "Granary: Speech Recognition and Translation Dataset in 25 European Languages" (2025)
   - arXiv: 2505.13404

### Model Links

- [Nemotron-Speech-Streaming-En-0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) - **NEW: Native streaming ASR**
- [Parakeet-TDT-0.6B-V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - Best offline accuracy
- [Canary-1B-V2](https://huggingface.co/nvidia/canary-1b-v2) - Multilingual ASR/Translation
- [Parakeet Realtime EOU 120M](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) - End-of-utterance detection
- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)

### Blog Posts

- [Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [NVIDIA Speech AI Models Deliver Industry-Leading Accuracy and Performance](https://developer.nvidia.com/blog/nvidia-speech-ai-models-deliver-industry-leading-accuracy-and-performance/)

### Quick Start

**Offline Transcription (Best Accuracy):**
```python
import nemo.collections.asr as nemo_asr

# Load Parakeet TDT for best offline accuracy
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# Transcribe audio
output = asr_model.transcribe(['audio.wav'])
print(output[0].text)

# With timestamps
output = asr_model.transcribe(['audio.wav'], timestamps=True)
for stamp in output[0].timestamp['word']:
    print(f"{stamp['start']}s - {stamp['end']}s: {stamp['word']}")
```

**Streaming Transcription (Voice Agents):**
```python
import nemo.collections.asr as nemo_asr

# Load Nemotron Speech for streaming
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")

# Configure latency mode (560ms chunks for balanced accuracy/latency)
# att_context_size = [70, 6]  # Set via config or inference script

# For streaming inference, use the cache-aware streaming script:
# python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
#     model_path=nvidia/nemotron-speech-streaming-en-0.6b \
#     dataset_manifest=<manifest> \
#     att_context_size="[70,6]" \
#     output_path=<output>
```

**Pipeline-based Streaming with PnC and ITN:**
```python
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from omegaconf import OmegaConf

# Load streaming config
cfg = OmegaConf.load('cache_aware_rnnt.yaml')

# Build pipeline with punctuation, capitalization, and ITN
pipeline = PipelineBuilder.build_pipeline(cfg)

# Run inference
output = pipeline.run(['/path/to/audio.wav'])
print(output[0]['text'])
```

---

*Document generated based on NVIDIA NeMo documentation and model cards as of January 2026.*
