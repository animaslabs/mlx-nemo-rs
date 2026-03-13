# NVIDIA NeMo Streaming ASR Models: Technical Architecture and Implementation Guide

This document provides a comprehensive technical reference for NVIDIA's streaming Automatic Speech Recognition (ASR) models in the NeMo Framework. It covers the cache-aware streaming architecture, neural network designs, Python implementations, and configuration details for building real-time voice applications.

## Table of Contents

1. [Introduction to Streaming ASR](#introduction-to-streaming-asr)
2. [Cache-Aware Streaming Architecture](#cache-aware-streaming-architecture)
3. [Streaming Model Families](#streaming-model-families)
4. [Cache-Aware FastConformer Encoder](#cache-aware-fastconformer-encoder)
5. [Streaming Inference Pipeline](#streaming-inference-pipeline)
6. [Python Implementation](#python-implementation)
7. [Configuration Reference](#configuration-reference)
8. [Multitalker Streaming ASR](#multitalker-streaming-asr)
9. [Streaming Speaker Diarization](#streaming-speaker-diarization)
10. [Performance and Benchmarks](#performance-and-benchmarks)
11. [Resources and Links](#resources-and-links)

---

## Introduction to Streaming ASR

Streaming ASR enables real-time transcription of audio as it arrives, essential for voice agents, live captioning, and conversational AI systems. NVIDIA's streaming models are designed to address the fundamental challenges of real-time speech recognition.

### Streaming vs Offline ASR

| Aspect | Offline ASR | Streaming ASR |
|--------|-------------|---------------|
| Input | Complete audio file | Continuous audio chunks |
| Latency | High (full file processing) | Low (chunk-by-chunk) |
| Context | Full bidirectional | Limited look-ahead |
| Use Case | Batch transcription | Real-time applications |

### Key Challenges in Streaming ASR

1. **Limited Future Context**: Models cannot see future audio frames
2. **Latency vs Accuracy Trade-off**: More look-ahead improves accuracy but increases latency
3. **Computational Efficiency**: Redundant computation in overlapping windows
4. **Latency Drift**: Performance degradation under high concurrency

### Key Papers

| Paper | Description | Link |
|-------|-------------|------|
| **Stateful Conformer** | Cache-aware streaming with activation caching | [arXiv:2312.17279](https://arxiv.org/abs/2312.17279) |
| **Fast Conformer** | 8x downsampling, efficient attention | [arXiv:2305.05084](https://arxiv.org/abs/2305.05084) |
| **Streaming Sortformer** | Speaker cache-based online diarization | [arXiv:2507.18446](https://arxiv.org/abs/2507.18446) |
| **Self-Speaker Adaptation** | Multitalker ASR via speaker kernels | [arXiv:2506.22646](https://arxiv.org/abs/2506.22646) |

---

## Cache-Aware Streaming Architecture

Cache-aware streaming is NVIDIA's solution to efficient real-time ASR. Unlike traditional buffered streaming, it eliminates redundant computation by caching intermediate activations.

### Buffered vs Cache-Aware Streaming

```
Traditional Buffered Streaming:
┌───────────────────────────────────────────────────────────────────┐
│ Audio: [====chunk1====][===overlap===][====chunk2====]            │
│                         ↑ Re-computed every step                  │
│ Problem: Redundant computation, latency drift under load          │
└───────────────────────────────────────────────────────────────────┘

Cache-Aware Streaming:
┌───────────────────────────────────────────────────────────────────┐
│ Audio: [====chunk1====][====chunk2====][====chunk3====]           │
│              ↓              ↓              ↓                      │
│           Cache→        Update→        Update→                    │
│        (stored)      (reused)       (reused)                      │
│ Benefit: Zero redundant computation, linear scaling               │
└───────────────────────────────────────────────────────────────────┘
```

### Why Cache-Aware Streaming?

| Feature | Buffered Streaming | Cache-Aware Streaming |
|---------|-------------------|----------------------|
| Computation | Overlapping windows | Non-overlapping chunks |
| Memory Growth | Grows with overlap | Fixed cache size |
| Latency Drift | Yes, under load | No |
| Concurrency (H100) | ~180 streams | ~560 streams |
| GPU Efficiency | Lower | Up to 3x higher |

### Caching Mechanism

The cache-aware encoder maintains caches for:
1. **Self-attention layers**: Cached key/value states from previous chunks
2. **Convolution layers**: Cached activations for causal convolutions
3. **Subsampling layers**: Context from downsampling operations

```
┌─────────────────────────────────────────────────────────────────┐
│                  Cache-Aware Encoder Layer                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  New Audio Chunk                                                 │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Subsampling (8x downsampling)                      │        │
│  │  • Causal depthwise-separable convolution           │        │
│  │  • Cache: Previous frame context                    │        │
│  └─────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Self-Attention with Limited Context                │        │
│  │  • att_context_size = [left, right]                 │        │
│  │  • Cache: Key/Value states from left context        │        │
│  └─────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Convolution Module (Causal)                        │        │
│  │  • conv_context_size = causal                       │        │
│  │  • Cache: Left context activations                  │        │
│  └─────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  Output + Updated Caches                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Streaming Model Families

### Nemotron Speech ASR (Primary Streaming Model)

**[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)**

The flagship cache-aware streaming model designed for real-time voice agents.

| Specification | Value |
|---------------|-------|
| Architecture | FastConformer Cache-Aware RNNT |
| Parameters | 600M |
| Encoder Layers | 24 |
| d_model | 1024 |
| Training Data | 285K hours |
| Languages | English |
| Latency Modes | 80ms, 160ms, 560ms, 1120ms |
| Max Concurrency (H100) | 560 streams |

**Key Features:**
- Native streaming architecture with cache-aware design
- Runtime-configurable latency modes (no retraining needed)
- Built-in punctuation and capitalization
- 24ms median time-to-final transcription

### Multitalker Parakeet Streaming

**[nvidia/multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)**

Multi-speaker streaming ASR with speaker-specific transcription.

| Specification | Value |
|---------------|-------|
| Architecture | FastConformer with Speaker Kernel Injection |
| Parameters | 600M |
| Max Speakers | 4 (one instance per speaker) |
| Features | Speaker-separated transcription |

### Streaming Speaker Diarization

**[nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)**

Real-time speaker diarization using Streaming Sortformer.

| Specification | Value |
|---------------|-------|
| Architecture | FastConformer + Transformer (Sortformer) |
| Parameters | 117M |
| Max Speakers | 4 |
| Latency Modes | 1.04s (low), 30.4s (high accuracy) |

---

## Cache-Aware FastConformer Encoder

The streaming encoder is based on FastConformer with modifications for cache-aware operation.

### Architecture Modifications for Streaming

```python
# Cache-aware streaming encoder configuration
encoder_config = {
    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
    "feat_in": 80,
    "n_layers": 24,
    "d_model": 1024,
    
    # 8x Subsampling (same as offline FastConformer)
    "subsampling": "dw_striding",
    "subsampling_factor": 8,
    "subsampling_conv_channels": 256,
    
    # Streaming attention context
    # [left_context, right_context] in 80ms frames
    "att_context_size": [70, 13],  # 5.6s left, 1.04s right (1120ms latency)
    "att_context_style": "regular",
    
    # Causal convolutions for streaming
    "conv_context_size": "causal",
    "causal_downsampling": True,
    
    # Standard attention config
    "self_attention_model": "rel_pos",
    "n_heads": 8,
    "conv_kernel_size": 9,
    
    # Regularization
    "dropout": 0.1,
    "dropout_att": 0.1,
}
```

### Latency Configuration

The `att_context_size` parameter controls streaming latency:

| att_context_size | Chunk Size | Latency | Use Case |
|------------------|------------|---------|----------|
| [70, 0] | 1 frame | 80ms | Ultra-low latency |
| [70, 1] | 2 frames | 160ms | Low latency |
| [70, 6] | 7 frames | 560ms | Balanced |
| [70, 13] | 14 frames | 1120ms | Best accuracy |

**Latency Calculation:**
- Each frame = 80ms (after 8x downsampling with 10ms stride)
- Chunk size = current frame + right context
- Latency = chunk_size × 80ms

### Context Types

**Regular Look-ahead:**
```yaml
# Fixed right context for all positions
att_context_size: [70, 13]
att_context_style: regular
```

**Chunk-aware Look-ahead:**
```yaml
# All tokens in chunk can see each other
# More efficient caching, similar accuracy
att_context_size: [70, 13]
att_context_style: chunk
```

### Multi-Latency Training

Train a single model that supports multiple latency modes:

```yaml
model:
  encoder:
    # Multiple context sizes for training
    att_context_size: [[70,13], [70,6], [70,1], [70,0]]
    att_context_probs: [0.25, 0.25, 0.25, 0.25]
```

Switch at inference time:
```python
# No retraining required
asr_model.encoder.set_default_att_context_size([70, 0])   # 80ms
asr_model.encoder.set_default_att_context_size([70, 1])   # 160ms
asr_model.encoder.set_default_att_context_size([70, 6])   # 560ms
asr_model.encoder.set_default_att_context_size([70, 13])  # 1120ms
```

---

## Streaming Inference Pipeline

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Streaming ASR Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Audio Stream (16kHz)                                                │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  CacheAwareStreamingAudioBuffer                         │        │
│  │  • Buffers incoming audio                               │        │
│  │  • Handles chunk alignment                              │        │
│  │  • Online normalization (optional)                      │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Mel Spectrogram Preprocessor                           │        │
│  │  • 80 mel filterbank features                           │        │
│  │  • 25ms window, 10ms stride                             │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Cache-Aware FastConformer Encoder                      │        │
│  │  • Processes chunk with cached context                  │        │
│  │  • Updates caches for next chunk                        │        │
│  │  • 8x downsampling                                      │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  RNNT Decoder                                           │        │
│  │  • Prediction network (LSTM)                            │        │
│  │  • Joint network                                        │        │
│  │  • Greedy or beam search decoding                       │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  Partial/Final Transcription                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Cache Management

```python
# Cache structure per encoder layer
class EncoderCache:
    # Self-attention cache: stores key/value from left context
    attn_cache: Tensor  # [batch, left_context, d_model]
    
    # Convolution cache: stores activations for causal conv
    conv_cache: Tensor  # [batch, kernel_size-1, d_model]
    
    # Subsampling cache: context for downsampling
    subsample_cache: Tensor
```

---

## Python Implementation

### Installation

```bash
# Install NeMo with ASR support
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Loading Streaming Models

```python
import nemo.collections.asr as nemo_asr

# Load Nemotron Speech streaming model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)
asr_model.eval()
```

### Basic Streaming Inference

```python
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
import torch

# Configure streaming buffer
streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=asr_model,
    online_normalization=True,
    pad_and_drop_preencoded=True,
)

# Load audio file into buffer
streaming_buffer.append_audio_file(
    audio_filepath="audio.wav",
    stream_id=-1
)

# Process chunks
transcripts = []
for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
    with torch.inference_mode():
        # Get partial transcription
        outputs = asr_model.transcribe_step(
            input_signal=chunk_audio,
            input_signal_length=chunk_lengths,
        )
        transcripts.append(outputs)
        
print("Final transcription:", transcripts[-1])
```

### Setting Latency Mode

```python
# Configure for specific latency
# att_context_size = [left_context, right_context] in 80ms frames

# Ultra-low latency (80ms)
asr_model.encoder.set_default_att_context_size([70, 0])

# Low latency (160ms)
asr_model.encoder.set_default_att_context_size([70, 1])

# Balanced (560ms)
asr_model.encoder.set_default_att_context_size([70, 6])

# Best accuracy (1120ms)
asr_model.encoder.set_default_att_context_size([70, 13])
```

### Using the Pipeline Builder

```python
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load('cache_aware_rnnt.yaml')

# Build pipeline with optional PnC, ITN
pipeline = PipelineBuilder.build_pipeline(cfg)

# Run inference
audios = ['/path/to/audio1.wav', '/path/to/audio2.wav']
output = pipeline.run(audios)

for entry in output:
    print(entry['text'])
```

### Streaming Inference Script

```bash
# Using NeMo's streaming inference script
python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=nvidia/nemotron-speech-streaming-en-0.6b \
    dataset_manifest=test.json \
    batch_size=16 \
    att_context_size="[70,13]" \
    output_path=output/
```

---

## Configuration Reference

### Full Cache-Aware Streaming Config

```yaml
name: "FastConformer-CacheAware-RNNT-Streaming"

model:
  sample_rate: 16000
  
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

  # Cache-Aware FastConformer Encoder
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    n_layers: 24
    d_model: 1024
    
    # 8x Subsampling
    subsampling: dw_striding
    subsampling_factor: 8
    subsampling_conv_channels: 256
    causal_downsampling: true
    
    # Streaming attention context
    att_context_size: [70, 13]  # [left, right] in 80ms frames
    att_context_style: regular
    
    # Causal convolutions
    conv_context_size: causal
    conv_kernel_size: 9
    
    # Attention
    self_attention_model: rel_pos
    n_heads: 8
    xscaling: true
    
    # Regularization
    dropout: 0.1
    dropout_att: 0.1

  # RNNT Decoder
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
    jointnet:
      joint_hidden: 640
      activation: "relu"
      dropout: 0.2
    fuse_loss_wer: true
    fused_batch_size: 4

  # Decoding
  decoding:
    strategy: "greedy_batch"
    greedy:
      max_symbols: 10

  # Tokenizer
  tokenizer:
    dir: ???  # Path to tokenizer
    type: bpe
```

### Multi-Latency Training Config

```yaml
model:
  encoder:
    # Train with multiple latency modes
    att_context_size: [[70,13], [70,6], [70,1], [70,0]]
    att_context_probs: [0.25, 0.25, 0.25, 0.25]
    
    # Shared parameters
    att_context_style: regular
    conv_context_size: causal
```

### Streaming Buffer Config

```yaml
# For streaming inference
streaming:
  chunk_size: 14        # Frames per chunk (14 * 80ms = 1.12s)
  shift_size: 14        # Non-overlapping
  left_context: 70      # 5.6s left context
  right_context: 13     # 1.04s look-ahead
  online_normalization: true
  pad_and_drop_preencoded: true
```

---

## Multitalker Streaming ASR

The Multitalker Parakeet model enables speaker-separated transcription in streaming mode.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│             Multitalker Streaming ASR Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Mixed Audio Input (multiple speakers)                               │
│       │                                                              │
│       ├──────────────────┬──────────────────┐                       │
│       ▼                  ▼                  ▼                       │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐                  │
│  │Instance 1│      │Instance 2│      │Instance N│                  │
│  │(Speaker1)│      │(Speaker2)│      │(SpeakerN)│                  │
│  └──────────┘      └──────────┘      └──────────┘                  │
│       │                  │                  │                       │
│       ▼                  ▼                  ▼                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Speaker Kernel Injection (Pre-encode layer)            │        │
│  │  • Learnable speaker kernels from diarization           │        │
│  │  • Dynamic adaptation to target speaker                 │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                  │                  │                       │
│       ▼                  ▼                  ▼                       │
│  Transcript 1       Transcript 2       Transcript N                 │
│  (Speaker 1)        (Speaker 2)        (Speaker N)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Speaker Kernel Injection

Unlike target-speaker ASR requiring enrollment audio, the multitalker model uses:
1. **Speaker diarization output** to identify speakers
2. **Learnable speaker kernels** injected at the pre-encode layer
3. **One model instance per speaker** for parallel processing

### Python Implementation

```python
from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel
import torch

# Load diarization model
diar_model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1"
).eval().to("cuda")

# Load multitalker ASR model
asr_model = ASRModel.from_pretrained(
    "nvidia/multitalker-parakeet-streaming-0.6b-v1"
).eval().to("cuda")

# Configure streaming
from multitalker_transcript_config import MultitalkerTranscriptionConfig
from omegaconf import OmegaConf

cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
cfg.audio_file = "/path/to/audio.wav"
cfg.output_path = "/path/to/output.json"

# Initialize diarization model with streaming params
diar_model = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar_model)

# Setup streaming buffer
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=asr_model,
    online_normalization=cfg.online_normalization,
    pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
)
streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)

# Process with multispeaker handler
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR

multispk_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
    with torch.inference_mode():
        multispk_streamer.perform_parallel_streaming_stt_spk(
            step_num=step_num,
            chunk_audio=chunk_audio,
            chunk_lengths=chunk_lengths,
            is_buffer_empty=streaming_buffer.is_buffer_empty(),
        )

# Get speaker-tagged transcripts
multispk_streamer.generate_seglst_dicts_from_parallel_streaming(samples=[{'audio_filepath': cfg.audio_file}])
print(multispk_streamer.instance_manager.seglst_dict_list)
```

### Script-Based Inference

```bash
python examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py \
    asr_model="/path/to/multitalker-parakeet-streaming-0.6b-v1.nemo" \
    diar_model="/path/to/diar_streaming_sortformer_4spk-v2.nemo" \
    att_context_size="[70,13]" \
    audio_file="/path/to/audio.wav" \
    output_path="/path/to/output.json"
```

---

## Streaming Speaker Diarization

### Streaming Sortformer Architecture

The Streaming Sortformer uses an **Arrival-Order Speaker Cache (AOSC)** to maintain speaker identity across streaming chunks.

```
┌─────────────────────────────────────────────────────────────────────┐
│              Streaming Sortformer Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Audio Chunk                                                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  NEST Encoder (FastConformer, 17 layers)                │        │
│  │  • Generates frame-level acoustic embeddings            │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Arrival-Order Speaker Cache (AOSC)                     │        │
│  │  • Stores embeddings ordered by speaker arrival time    │        │
│  │  • Dynamic update based on prediction scores            │        │
│  │  • Speaker tracking across chunks                       │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Transformer Decoder (18 layers)                        │        │
│  │  • Cross-attention with speaker cache                   │        │
│  │  • Frame-level speaker activity prediction              │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  Speaker Activity Probabilities [T × 4]                              │
│  (4 speakers max, per-frame probabilities)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Streaming Configuration Parameters

| Parameter | Description | Low Latency | High Accuracy |
|-----------|-------------|-------------|---------------|
| CHUNK_SIZE | Frames per processing chunk | 6 | 340 |
| RIGHT_CONTEXT | Future frames after chunk | 7 | 40 |
| FIFO_SIZE | Previous frames before chunk | 188 | 40 |
| UPDATE_PERIOD | Frames for cache update | 144 | 300 |
| SPEAKER_CACHE_SIZE | Total cache frames | 188 | 188 |
| **Latency** | Input buffer latency | 1.04s | 30.4s |
| **RTF** | Real-time factor | 0.093 | 0.002 |

### Python Implementation

```python
from nemo.collections.asr.models import SortformerEncLabelModel

# Load streaming diarization model
diar_model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1"
)
diar_model.eval()

# Configure for low latency (1.04s)
diar_model.sortformer_modules.chunk_len = 6
diar_model.sortformer_modules.chunk_right_context = 7
diar_model.sortformer_modules.fifo_len = 188
diar_model.sortformer_modules.spkcache_update_period = 144
diar_model.sortformer_modules.spkcache_len = 188
diar_model.sortformer_modules._check_streaming_parameters()

# Or configure for high accuracy (30.4s latency)
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300
diar_model.sortformer_modules.spkcache_len = 188
diar_model.sortformer_modules._check_streaming_parameters()

# Run diarization
predicted_segments = diar_model.diarize(
    audio=["/path/to/audio.wav"], 
    batch_size=1
)

# Print results
for segment in predicted_segments[0]:
    print(segment)  # (start_time, end_time, speaker_id)
```

### Output Format

```python
# Speaker segments output
# Format: (begin_seconds, end_seconds, speaker_index)
[
    (0.0, 2.5, 0),    # Speaker 0 from 0.0s to 2.5s
    (2.3, 5.1, 1),    # Speaker 1 from 2.3s to 5.1s (overlap)
    (5.0, 8.2, 0),    # Speaker 0 from 5.0s to 8.2s
    ...
]

# Or get raw probabilities
predicted_segments, predicted_probs = diar_model.diarize(
    audio=["/path/to/audio.wav"],
    batch_size=1,
    include_tensor_outputs=True
)
# predicted_probs: [T × 4] tensor of speaker activity probabilities
```

---

## Performance and Benchmarks

### Nemotron Speech ASR Performance

#### Word Error Rate by Latency Mode

| Latency | AMI | Earnings22 | Gigaspeech | LS-clean | LS-other | SPGI | TEDLIUM | VoxPopuli | **Avg** |
|---------|-----|------------|------------|----------|----------|------|---------|-----------|---------|
| 1.12s | 11.58 | 12.48 | 11.45 | 2.31 | 4.75 | 2.62 | 4.50 | 7.57 | **7.16** |
| 0.56s | 11.69 | 12.61 | 11.43 | 2.40 | 4.97 | 2.64 | 4.46 | 7.59 | **7.22** |
| 0.16s | 13.88 | 13.61 | 12.12 | 2.43 | 5.33 | 2.82 | 4.80 | 7.72 | **7.84** |
| 0.08s | 16.05 | 14.60 | 12.92 | 2.55 | 5.79 | 3.01 | 5.07 | 8.23 | **8.53** |

#### Throughput Comparison

| Hardware | Buffered Streaming | Cache-Aware | Improvement |
|----------|-------------------|-------------|-------------|
| NVIDIA H100 | ~180 streams | ~560 streams | **3.1x** |
| NVIDIA RTX A5000 | ~50 streams | ~250 streams | **5x** |
| NVIDIA DGX B200 | - | 2x baseline | **2x** |

#### Time-to-Final Transcription

| Model Type | Median Latency |
|------------|----------------|
| Nemotron Speech ASR | 24ms |
| Alternative (local GPU) | 90ms |
| Alternative (API-based) | 200ms+ |

### Streaming Sortformer Diarization Performance

#### Diarization Error Rate (DER)

**Telephonic Speech (1.04s latency):**

| Dataset | DER |
|---------|-----|
| DIHARD III Eval (1-4 spk) | 15.09% |
| CALLHOME-part2 (full) | 11.19% |
| CH109 | 5.09% |

**Meeting Speech (1.04s latency):**

| Dataset | DER |
|---------|-----|
| AMI Test IHM | 16.67% |
| AMI Test SDM | 20.57% |
| AliMeeting Test near | 12.60% |

### Multitalker ASR Performance (cpWER)

| Dataset | cpWER (1.12s latency) |
|---------|----------------------|
| AMI IHM | 21.26% |
| AMI SDM | 37.44% |
| CH109 | 15.81% |
| Mixer 6 | 23.81% |

---

## Resources and Links

### HuggingFace Models

| Model | Use Case | Link |
|-------|----------|------|
| Nemotron Speech 0.6B | Streaming ASR | [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) |
| Multitalker Parakeet 0.6B | Multi-speaker streaming | [nvidia/multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) |
| Streaming Sortformer v2.1 | Speaker diarization | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) |

### Interactive Demos

- [Nemotron Speech Streaming Demo](https://huggingface.co/spaces/nvidia/nemotron-speech-streaming-en-0.6b)

### Research Papers

| Paper | Topic | Link |
|-------|-------|------|
| Stateful Conformer | Cache-aware streaming | [arXiv:2312.17279](https://arxiv.org/abs/2312.17279) |
| Fast Conformer | Efficient encoder | [arXiv:2305.05084](https://arxiv.org/abs/2305.05084) |
| Streaming Sortformer | Online diarization | [arXiv:2507.18446](https://arxiv.org/abs/2507.18446) |
| Self-Speaker Adaptation | Multitalker ASR | [arXiv:2506.22646](https://arxiv.org/abs/2506.22646) |
| NEST | Self-supervised encoder | [arXiv:2408.13106](https://arxiv.org/abs/2408.13106) |

### NeMo Resources

- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [Streaming ASR Examples](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/asr_cache_aware_streaming)
- [Streaming Configs](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/fastconformer/cache_aware_streaming)
- [Cache-Aware Streaming Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb)
- [Multitalker ASR Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb)

### Deployment

- [Modal Deployment](https://github.com/modal-projects/modal-nvidia-asr) - Cloud deployment example
- [Daily/Pipecat Integration](https://github.com/pipecat-ai/nemotron-january-2026) - Voice agent framework
- [NVIDIA Riva](https://developer.nvidia.com/riva) - Production deployment SDK
- [NVIDIA NIM](https://build.nvidia.com/explore/speech) - Inference microservices

### Developer Blog

- [Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)

---

*Last updated: January 2026*
