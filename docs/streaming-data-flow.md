# Streaming ASR Data Flow: From Audio to Tokens

This document provides a detailed walkthrough of how data flows through the mlx-voice-rs streaming ASR pipeline, including tensor shapes, caching mechanisms, and the decoding loop.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Streaming Modes and Configuration](#streaming-modes-and-configuration)
3. [Audio Chunking](#audio-chunking)
4. [Mel Spectrogram Computation](#mel-spectrogram-computation)
5. [Mel Buffer Management](#mel-buffer-management)
6. [Streaming Encoder](#streaming-encoder)
7. [Cache-Aware Attention](#cache-aware-attention)
8. [RNN-T Decoding](#rnn-t-decoding)
9. [Complete Example](#complete-example)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STREAMING ASR DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Raw Audio (16kHz, f32)                                                         │
│   ════════════════════                                                           │
│        │                                                                         │
│        │  Chunk: ~160ms = 2,560 samples                                          │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                    │
│   │         Audio Buffer                     │                                    │
│   │   Accumulates samples until enough      │                                    │
│   │   for mel computation                   │                                    │
│   └─────────────────────────────────────────┘                                    │
│        │                                                                         │
│        │  STFT: 25ms window, 10ms stride                                         │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                    │
│   │      Mel Spectrogram Preprocessor       │                                    │
│   │   [batch, n_mels=80, time]              │                                    │
│   └─────────────────────────────────────────┘                                    │
│        │                                                                         │
│        │  Store in mel buffer                                                    │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                    │
│   │          Mel Frame Buffer               │                                    │
│   │   Stores frames, provides chunks        │                                    │
│   │   with pre_encode_cache overlap         │                                    │
│   └─────────────────────────────────────────┘                                    │
│        │                                                                         │
│        │  Extract: chunk_size + pre_encode_cache frames                          │
│        │  Advance: shift_size frames                                             │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐     ┌─────────────────────────┐    │
│   │      Cache-Aware Conformer Encoder      │ ◄── │    Encoder Caches       │    │
│   │   • Subsampling (8x)                    │     │  • Attention K/V        │    │
│   │   • 24 Conformer layers                 │     │  • Conv left context    │    │
│   │   • Attention + Conv + FFN              │ ──► │    [layers, B, T, D]    │    │
│   └─────────────────────────────────────────┘     └─────────────────────────┘    │
│        │                                                                         │
│        │  Output: [batch, d_model=1024, valid_out_len]                           │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐     ┌─────────────────────────┐    │
│   │        RNN-T Greedy Decoding            │ ◄── │    Decoder State        │    │
│   │   • Prediction network (LSTM)           │     │  • LSTM hidden/cell     │    │
│   │   • Joint network                       │     │  • Last token           │    │
│   │   • Emit tokens until blank             │ ──► │    Per batch item       │    │
│   └─────────────────────────────────────────┘     └─────────────────────────┘    │
│        │                                                                         │
│        │  Token IDs                                                              │
│        ▼                                                                         │
│   ┌─────────────────────────────────────────┐                                    │
│   │       SentencePiece Tokenizer           │                                    │
│   │   Decode token IDs to text              │                                    │
│   └─────────────────────────────────────────┘                                    │
│        │                                                                         │
│        ▼                                                                         │
│   Transcription Text                                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Streaming Modes and Configuration

The streaming pipeline is highly configurable. The primary control is the **streaming mode**, which sets the latency/accuracy trade-off. Additional parameters control cache sizes, attention context, and subsampling behavior.

### Selecting a Streaming Mode

Models like Nemotron Speech support multiple streaming modes, selected by `mode_idx`:

```rust
// Initialize streaming with a specific mode
let session = model.init_streaming(mode_idx, batch_size)?;

// Query available modes
for mode in model.streaming_modes() {
    println!("Mode {}: {}ms latency - {}", 
             mode.mode_idx, mode.latency_ms, mode.description);
}
```

### Mode Configuration (Nemotron Speech Example)

The model's `att_context_size` config defines available modes:

```yaml
# Model config (model_config.yaml)
encoder:
  att_context_size: [[70,13], [70,6], [70,1], [70,0]]  # 4 modes
  att_context_style: "regular"
```

| Mode | att_context_size | Left Context | Right Context | Latency | WER Impact |
|------|------------------|--------------|---------------|---------|------------|
| 0 | [70, 13] | 70 frames (5.6s) | 13 frames | **1120ms** | Best accuracy |
| 1 | [70, 6] | 70 frames (5.6s) | 6 frames | **560ms** | Balanced |
| 2 | [70, 1] | 70 frames (5.6s) | 1 frame | **160ms** | Low latency |
| 3 | [70, 0] | 70 frames (5.6s) | 0 frames | **80ms** | Ultra-low latency |

**Latency Calculation**: `latency = (right_context + 1) × 80ms` (80ms per frame after 8x subsampling)

### How Mode Selection Affects the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MODE SELECTION IMPACT ON PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  att_context_size = [left, right]                                                │
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  lookahead_steps = right × n_layers + conv_right × n_layers                 ││
│  │                  = right × 24 + 0 × 24  (causal conv has 0 right context)   ││
│  │                                                                              ││
│  │  Mode 0 (right=13): lookahead = 13 × 24 = 312                               ││
│  │  Mode 3 (right=0):  lookahead = 0 × 24 = 0                                  ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  chunk_size = subsampling_factor + subsampling_factor × lookahead           ││
│  │             = 8 + 8 × lookahead                                              ││
│  │                                                                              ││
│  │  Mode 0: chunk_size = 8 + 8×312 = 2504 mel frames                           ││
│  │  Mode 3: chunk_size = 8 + 8×0 = 8 mel frames                                ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  shift_size = subsampling_factor (for "regular" style)                      ││
│  │             = 8 mel frames (always, regardless of mode)                      ││
│  │                                                                              ││
│  │  This means Mode 0 has MASSIVE overlap (2504 - 8 = 2496 frames)             ││
│  │  while Mode 3 has NO overlap (8 - 8 = 0 frames)                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Attention mask window = [left_context, right_context]                       ││
│  │                                                                              ││
│  │  Mode 0: each query sees 70 past + 13 future frames                         ││
│  │  Mode 3: each query sees 70 past + 0 future frames (fully causal)           ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Derived Configuration Parameters

All streaming parameters are computed from the base config:

```rust
pub struct StreamingConfig {
    // From mode selection
    pub left_context: i32,           // att_context_size[0]
    pub right_context: i32,          // att_context_size[1]
    
    // Computed from right_context and architecture
    pub lookahead_frames: usize,     // right × n_layers (for "regular" style)
    pub chunk_size: (usize, usize),  // (first_chunk, subsequent_chunk)
    pub shift_size: (usize, usize),  // (first_shift, subsequent_shift)
    
    // Cache configuration
    pub last_channel_cache_size: usize,  // = left_context (attention K/V)
    pub last_time_cache_size: usize,     // = conv_kernel - 1 (conv state)
    pub cache_drop_size: usize,          // = lookahead (for "regular" style)
    
    // Output handling
    pub valid_out_len: usize,            // Frames to keep from encoder output
    pub pre_encode_cache_size: (usize, usize),  // Subsampling overlap
    pub drop_extra_pre_encoded: usize,   // Redundant frames to drop
}
```

### Complete Mode Comparison Table

For a 24-layer model with 8x subsampling, conv_kernel=9:

| Parameter | Mode 0 (1120ms) | Mode 1 (560ms) | Mode 2 (160ms) | Mode 3 (80ms) |
|-----------|-----------------|----------------|----------------|---------------|
| `right_context` | 13 | 6 | 1 | 0 |
| `left_context` | 70 | 70 | 70 | 70 |
| `lookahead_steps` | 312 | 144 | 24 | 0 |
| `chunk_size` (subsequent) | 2504 | 1160 | 200 | 8 |
| `shift_size` | 8 | 8 | 8 | 8 |
| `mel_overlap` | 2496 | 1152 | 192 | 0 |
| `cache_drop_size` | 312 | 144 | 24 | 0 |
| `valid_out_len` | 1 | 1 | 1 | 1 |
| `attention_cache_size` | 70 | 70 | 70 | 70 |
| `conv_cache_size` | 8 | 8 | 8 | 8 |

### Attention Context Styles

Two styles affect how lookahead and caching work:

#### "regular" Style (Default)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REGULAR ATTENTION STYLE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  • Lookahead accumulates through layers                                      │
│  • lookahead_steps = right_context × n_layers                                │
│  • cache_drop_size = lookahead_steps (drops all lookahead from cache)        │
│  • Large mel chunks, but shift_size stays small                              │
│                                                                              │
│  Attention window per position (left=70, right=13):                          │
│                                                                              │
│  Frame positions:  ... -3  -2  -1   0   1   2   3  ...  13                   │
│                        ├───────────┼───────────────────┤                     │
│                        │   left    │      right        │                     │
│                        │  context  │     context       │                     │
│                        │  (70 max) │    (13 frames)    │                     │
│                                                                              │
│  Each query at position t can attend to:                                     │
│    - Up to 70 frames in the past (from cache)                                │
│    - Current frame                                                           │
│    - Up to 13 frames in the future (from lookahead)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### "chunked_limited" Style

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHUNKED LIMITED ATTENTION STYLE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  • Lookahead is just right_context (not multiplied by layers)                │
│  • cache_drop_size = 0 (keep all cache)                                      │
│  • Smaller chunks, but shift_size equals chunk_size (overlap from cache)     │
│  • Within each chunk, all positions can see each other                       │
│                                                                              │
│  Chunk-based attention (right=13 means chunk_size=14):                       │
│                                                                              │
│  Chunk N:        [frame 0, frame 1, ..., frame 13]                           │
│                  └────────── all visible ──────────┘                         │
│                                                                              │
│  Chunk N+1:      [frame 14, frame 15, ..., frame 27]                         │
│                  └────────── all visible ──────────┘                         │
│                                                                              │
│  Cross-chunk: Chunk N+1 can see left_chunks × chunk_size past frames         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Impact Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHAT EACH PARAMETER CONTROLS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LATENCY & ACCURACY                                                          │
│  ─────────────────                                                           │
│  right_context ↑  →  latency ↑, accuracy ↑, chunk_size ↑                     │
│  right_context ↓  →  latency ↓, accuracy ↓, chunk_size ↓                     │
│                                                                              │
│  MEMORY                                                                      │
│  ──────                                                                      │
│  left_context ↑   →  attention_cache_size ↑, more past context               │
│  conv_kernel ↑    →  conv_cache_size ↑ (kernel - 1)                          │
│                                                                              │
│  THROUGHPUT                                                                  │
│  ──────────                                                                  │
│  Mode 3 (right=0) →  No overlap, minimal computation per step                │
│  Mode 0 (right=13)→  Massive overlap, but encoder caches prevent recompute   │
│                                                                              │
│  AUDIO BUFFERING                                                             │
│  ───────────────                                                             │
│  chunk_size ↑     →  Need more mel frames before processing                  │
│  Mode 0: ~20s of audio buffered before first output (2504 × 10ms)            │
│  Mode 3: ~80ms of audio buffered (8 × 10ms)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Practical Mode Selection Guide

| Use Case | Recommended Mode | Rationale |
|----------|------------------|-----------|
| Voice agents / real-time | Mode 3 (80ms) | Minimum latency for responsive interaction |
| Live captioning | Mode 2 (160ms) | Balance of latency and accuracy |
| Meeting transcription | Mode 1 (560ms) | Good accuracy with acceptable delay |
| Batch/offline streaming | Mode 0 (1120ms) | Best accuracy when latency doesn't matter |

### Changing Modes at Runtime

```rust
// Start with low latency
let mut session = model.init_streaming(3, 1)?;  // 80ms mode

// Process audio...
let output = model.process_chunk(&mut session, &audio_chunk)?;

// To change modes, must create new session (caches are mode-dependent)
let session = model.init_streaming(0, 1)?;  // Switch to 1120ms mode
```

---

## Audio Chunking

### Input Format

```
Audio Source (microphone, file, etc.)
    │
    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Raw Audio Chunk                                                            │
│  ══════════════                                                             │
│  • Sample rate: 16,000 Hz                                                   │
│  • Format: f32 normalized [-1.0, 1.0]                                       │
│  • Typical chunk: 160ms = 2,560 samples                                     │
│  • Shape: [num_samples] (flat array)                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### Audio Buffer

The `StreamingMelBuffer` accumulates audio samples:

```rust
// Add audio chunk to buffer
session.mel_buffer.add_audio(audio_chunk);  // audio_chunk: &[f32]

// Internal state:
audio_buffer: Vec<f32>           // Accumulated samples
total_audio_samples: usize       // Cumulative count (never decreases)
```

### Chunking Strategy

```
Time ────────────────────────────────────────────────────────────────────────►

Audio Stream:
═══════════════════════════════════════════════════════════════════════════

Chunk 1          Chunk 2          Chunk 3          Chunk 4
[2560 samples]   [2560 samples]   [2560 samples]   [2560 samples]
│                │                │                │
└───160ms────────┴───160ms────────┴───160ms────────┴───160ms───►

After 4 chunks: 10,240 samples = 640ms of audio
```

---

## Mel Spectrogram Computation

### STFT Parameters (NeMo defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_size` | 0.025s (25ms) | STFT window length |
| `window_stride` | 0.01s (10ms) | Hop between windows |
| `n_fft` | 512 | FFT size |
| `n_mels` | 80 | Mel filterbank bins |
| `sample_rate` | 16000 Hz | Audio sample rate |

### Sample to Frame Conversion

```
Window samples = 0.025 × 16000 = 400 samples
Hop samples    = 0.01  × 16000 = 160 samples

For N audio samples (with center padding):
  n_frames = N / hop_samples + 1
  
Example: 2560 samples → 2560/160 + 1 = 17 frames
```

### Mel Computation Pipeline

```
Audio: [batch=1, time=2560]
    │
    │ (1) Optional preemphasis: y[n] = x[n] - α·x[n-1]
    ▼
Audio: [1, 2560]
    │
    │ (2) STFT with center padding
    │     • Pad: n_fft/2 on each side = 256 samples
    │     • Window: Hann, 400 samples
    │     • Hop: 160 samples
    ▼
Complex STFT: [1, n_fft/2+1=257, frames=17]
    │
    │ (3) Power spectrum: |STFT|²
    ▼
Power: [1, 257, 17]
    │
    │ (4) Mel filterbank: einsum("mf,bft->bmt", mel_fb, power)
    │     mel_fb: [80, 257]
    ▼
Mel: [1, 80, 17]
    │
    │ (5) Log with guard: log(x + 2⁻²⁴)
    ▼
Log-Mel: [1, 80, 17]  ← Output to mel buffer
```

### Tensor Shape Progression

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage              │ Shape              │ Notes                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Raw audio          │ [2560]             │ 160ms @ 16kHz                 │
│ Batched audio      │ [1, 2560]          │ Add batch dimension           │
│ After preemph      │ [1, 2560]          │ High-freq emphasis            │
│ STFT (complex)     │ [1, 257, 17]       │ Freq bins × frames            │
│ Power spectrum     │ [1, 257, 17]       │ Real-valued                   │
│ Mel spectrogram    │ [1, 80, 17]        │ 80 mel bins                   │
│ Log-mel            │ [1, 80, 17]        │ Final features                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Mel Buffer Management

The mel buffer stores computed mel frames and provides chunks to the encoder with proper overlap for subsampling continuity.

### Buffer Structure

```rust
pub struct StreamingMelBuffer {
    mel_buffer: Vec<f32>,        // [time, n_mels] flattened storage
    n_mels: usize,               // 80
    total_frames: usize,         // Total frames computed
    buffer_idx: usize,           // Current read position
    mel_buffer_offset: usize,    // Frames trimmed from start
    // ... audio buffer fields
}
```

### Chunk Extraction with pre_encode_cache

```
Mel Buffer State:
═════════════════

                    buffer_idx
                        │
Frame indices:   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
                 ├──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┤
Mel frames:      │F0│F1│F2│F3│F4│F5│F6│F7│F8│F9│F10│F11│F12│F13│...│
                 └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┴───┴───┴───┴...┘
                          ▲                 ▲
                          │                 │
                 pre_encode_cache     chunk_size
                    (9 frames)        (8 frames)
                          │                 │
                          └────────┬────────┘
                                   │
                            Total extract:
                             17 frames


extract_chunk(chunk_size=8, shift_size=8, pre_encode_cache=9):
════════════════════════════════════════════════════════════════

1. Start position: buffer_idx - pre_encode_cache = 12 - 9 = 3
2. Extract: frames 3..20 (17 frames total)
3. Advance: buffer_idx += shift_size → 12 + 8 = 20
4. Output shape: [n_mels=80, time=17]
```

### First vs Subsequent Chunks

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Parameter          │ First Chunk     │ Subsequent Chunks               │
├─────────────────────────────────────────────────────────────────────────┤
│ chunk_size         │ Smaller (1)     │ Larger (8) = subsampling_factor │
│ shift_size         │ Smaller (1)     │ Same as chunk_size (8)          │
│ pre_encode_cache   │ 0               │ 9 (subsampling_factor + 1)      │
│ drop_extra         │ 0               │ 2                               │
└─────────────────────────────────────────────────────────────────────────┘

Mode 0 (80ms latency, right_context=0):
  chunk_size = (1, 8)
  shift_size = (1, 8)
  pre_encode_cache = (0, 9)
  
Mode 3 (1120ms latency, right_context=13):
  chunk_size = (1 + 8×312, 8 + 8×312) = (2497, 2504)
  shift_size = (1, 8)  -- same, because cache_drop_size = lookahead_steps
  pre_encode_cache = (0, 9)
```

---

## Streaming Encoder

### Encoder Architecture

```
Input Mel Chunk: [batch=1, n_mels=80, time=17]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONFORMER ENCODER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              Subsampling (dw_striding, 8x)                         │ │
│  │  ════════════════════════════════════════                          │ │
│  │  Input:  [1, 80, 17]                                               │ │
│  │  • 2x CausalConv2D + pooling                                       │ │
│  │  • Reduces time by 8x, projects to d_model                         │ │
│  │  Output: [1, 1024, 2] (17/8 ≈ 2 frames)                            │ │
│  │                                                                     │ │
│  │  drop_extra_pre_encoded: drops first N output frames               │ │
│  │  (removes frames from pre_encode_cache overlap)                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                   │
│      │ Transpose to [1, 2, 1024]                                        │
│      ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              Positional Encoding                                    │ │
│  │  ════════════════════════════                                       │ │
│  │  • RelPositionalEncoding                                            │ │
│  │  • Adds cache_len offset for correct positions                      │ │
│  │  Output: x [1, 2, 1024], pos_emb [1, T_total, 1024]                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                   │
│      ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              Conformer Layers × 24                                  │ │
│  │  ════════════════════════════════                                   │ │
│  │                                                                     │ │
│  │  For each layer:                                                    │ │
│  │    ┌──────────────────────────────────────────────────────────────┐│ │
│  │    │ Input x: [1, seq_len, 1024]                                  ││ │
│  │    │                                                               ││ │
│  │    │ (1) FFN (first half)                                          ││ │
│  │    │     x = x + 0.5 × FFN(LayerNorm(x))                          ││ │
│  │    │                                                               ││ │
│  │    │ (2) Multi-Head Self-Attention (with cache)                   ││ │
│  │    │     • Concat cache + x for K,V                               ││ │
│  │    │     • Apply attention mask (context window)                  ││ │
│  │    │     • Update cache with new K,V                              ││ │
│  │    │     x = x + Attention(LayerNorm(x))                          ││ │
│  │    │                                                               ││ │
│  │    │ (3) Convolution Module (with cache)                          ││ │
│  │    │     • Concat time_cache + x                                  ││ │
│  │    │     • Depthwise conv (kernel=9)                              ││ │
│  │    │     • Update time_cache                                      ││ │
│  │    │     x = x + Conv(LayerNorm(x))                               ││ │
│  │    │                                                               ││ │
│  │    │ (4) FFN (second half)                                         ││ │
│  │    │     x = x + 0.5 × FFN(LayerNorm(x))                          ││ │
│  │    │                                                               ││ │
│  │    │ (5) Final LayerNorm                                           ││ │
│  │    │     x = LayerNorm(x)                                          ││ │
│  │    │                                                               ││ │
│  │    │ Output x: [1, seq_len, 1024]                                  ││ │
│  │    │ Cache updates: channel [1, cache_size, 1024]                  ││ │
│  │    │                time [1, 1024, 8]                              ││ │
│  │    └──────────────────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                   │
│      │ Truncate to valid_out_len                                        │
│      │ Transpose to [1, 1024, valid_out_len]                            │
│      ▼                                                                   │
│  Output: [1, 1024, 1]  (1 valid output frame after truncation)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Subsampling Detail

```
Input: [batch, n_mels=80, time=17]

Reshape to 4D: [batch, time, freq, channels=1]
             = [1, 17, 80, 1]

CausalConv2D Layer 1:
  • kernel=3×3, stride=2×2, out_channels=256
  • Causal padding: left=2, right=1 (time & freq)
  • Input:  [1, 17, 80, 1]
  • Output: [1, 9, 40, 256]  (time/2, freq/2)

CausalConv2D Layer 2:
  • kernel=3×3, stride=2×2, out_channels=256
  • Causal padding: left=2, right=1
  • Input:  [1, 9, 40, 256]
  • Output: [1, 5, 20, 256]  (time/2, freq/2)

Reshape: [batch, time, freq × channels]
       = [1, 5, 5120]

Linear projection: 5120 → 1024
  • Output: [1, 5, 1024]

After drop_extra_pre_encoded (drop first 2):
  • Output: [1, 3, 1024]

Transpose: [1, 1024, 3]  ← To encoder
```

---

## Cache-Aware Attention

### Cache Structure

```rust
pub struct EncoderCache {
    /// Attention K/V cache: [num_layers=24, batch=1, cache_size=70, d_model=1024]
    pub last_channel: Array,
    
    /// Convolution cache: [num_layers=24, batch=1, d_model=1024, kernel-1=8]
    pub last_time: Array,
    
    /// Filled cache length: [batch=1] (how many frames are valid)
    pub last_channel_len: Array,  // i64
}
```

### Attention with Cache (per layer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CACHE-AWARE SELF-ATTENTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Current input x: [batch=1, seq_len=2, d_model=1024]                         │
│  Cache (filled portion): [batch=1, filled=35, d_model=1024]                  │
│                                                                              │
│  Step 1: Concatenate cache + input for K, V                                  │
│  ═══════════════════════════════════════════                                 │
│                                                                              │
│       ┌─────────────────────┬──────────┐                                     │
│       │  Cache (35 frames)  │ x (2 fr) │  = K, V source: 37 frames           │
│       └─────────────────────┴──────────┘                                     │
│                                                                              │
│  Step 2: Compute Q, K, V                                                     │
│  ═══════════════════════                                                     │
│                                                                              │
│       Q = Linear(x)           → [1, 2, 1024]  (query only for new frames)    │
│       K = Linear(cache + x)   → [1, 37, 1024] (keys for all visible frames)  │
│       V = Linear(cache + x)   → [1, 37, 1024] (values for all visible)       │
│                                                                              │
│  Step 3: Reshape for multi-head                                              │
│  ══════════════════════════════                                              │
│                                                                              │
│       Q: [1, 2, 8, 128]   (8 heads, 128 dim each)                            │
│       K: [1, 37, 8, 128]                                                     │
│       V: [1, 37, 8, 128]                                                     │
│                                                                              │
│  Step 4: Attention with mask                                                 │
│  ═══════════════════════════                                                 │
│                                                                              │
│       scores = Q @ K.T / sqrt(128)  → [1, 8, 2, 37]                          │
│                                                                              │
│       Mask (for regular style, left=70, right=0):                            │
│       ┌───────────────────────────────────────────────────────────────┐      │
│       │  Position 0 (query): can see positions max(0, 0-70) to 0       │      │
│       │  Position 1 (query): can see positions max(0, 1-70) to 1       │      │
│       │                                                                │      │
│       │  Also mask unfilled cache positions (offset masking)           │      │
│       └───────────────────────────────────────────────────────────────┘      │
│                                                                              │
│       attn = softmax(masked_scores) @ V  → [1, 8, 2, 128]                    │
│                                                                              │
│  Step 5: Output projection                                                   │
│  ═════════════════════════                                                   │
│                                                                              │
│       output = Linear(concat_heads(attn))  → [1, 2, 1024]                    │
│                                                                              │
│  Step 6: Update cache (sliding window)                                       │
│  ═════════════════════════════════════                                       │
│                                                                              │
│       new_cache = concat(cache[cache_drop:], x[:keep])                       │
│                 = concat(cache[0:35], x[0:2])                                │
│                 → [1, 37, 1024]  (grows until cache_size=70)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Convolution with Cache (per layer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CAUSAL CONVOLUTION WITH CACHE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Conv kernel size: 9                                                         │
│  Time cache size: kernel - 1 = 8                                             │
│                                                                              │
│  Input x: [1, seq_len=2, d_model=1024]                                       │
│  Time cache: [1, d_model=1024, cache=8]                                      │
│                                                                              │
│  Step 1: Pointwise conv (expand)                                             │
│  ═════════════════════════════════                                           │
│       x → [1, 2, 2048]  (2× expansion)                                       │
│                                                                              │
│  Step 2: GLU activation                                                      │
│  ════════════════════════                                                    │
│       x → [1, 2, 1024]                                                       │
│                                                                              │
│  Step 3: Depthwise conv with cache                                           │
│  ═══════════════════════════════════                                         │
│                                                                              │
│       Transpose: [1, 1024, 2]                                                │
│                                                                              │
│       Concat cache:                                                          │
│       ┌────────────────────────┬────────┐                                    │
│       │ time_cache (8 frames)  │ x (2)  │  = [1, 1024, 10]                   │
│       └────────────────────────┴────────┘                                    │
│                                                                              │
│       Depthwise conv (kernel=9, groups=1024):                                │
│       • No padding (causal via cache)                                        │
│       • Output: [1, 1024, 2]  (10 - 9 + 1 = 2)                               │
│                                                                              │
│       Update cache: last 8 frames of (cache + x)                             │
│       • new_cache = [1, 1024, 8]                                             │
│                                                                              │
│  Step 4: Batch norm + activation                                             │
│  ═══════════════════════════════                                             │
│       BatchNorm → SiLU → [1, 1024, 2]                                        │
│                                                                              │
│  Step 5: Pointwise conv (project back)                                       │
│  ══════════════════════════════════════                                      │
│       Transpose: [1, 2, 1024]                                                │
│       Linear: [1, 2, 1024]                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## RNN-T Decoding

### RNN-T Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RNN-T DECODING LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Encoder output: [batch=1, d_model=1024, time=1]                             │
│  (Transposed to [1, 1, 1024] for decoding)                                   │
│                                                                              │
│  For each encoder frame t:                                                   │
│  ══════════════════════════                                                  │
│                                                                              │
│      enc_frame = encoder_output[:, t:t+1, :]  → [1, 1, 1024]                │
│                                                                              │
│      symbols_emitted = 0                                                     │
│      while symbols_emitted < MAX_SYMBOLS_PER_STEP (10):                      │
│                                                                              │
│          ┌────────────────────────────────────────────────────────────────┐ │
│          │              PREDICTION NETWORK (LSTM)                         │ │
│          │  ══════════════════════════════════                            │ │
│          │                                                                 │ │
│          │  Input: last_token or blank (for SOS)                          │ │
│          │  State: (hidden, cell) per LSTM layer                          │ │
│          │                                                                 │ │
│          │  last_token: [1, 1] (token index)                              │ │
│          │       │                                                         │ │
│          │       ▼                                                         │ │
│          │  Embedding lookup: [1, 1, 640]                                  │ │
│          │       │                                                         │ │
│          │       ▼                                                         │ │
│          │  LSTM Layer 1:                                                  │ │
│          │    input: [1, 1, 640]                                           │ │
│          │    hidden_in: [1, 640]                                          │ │
│          │    cell_in: [1, 640]                                            │ │
│          │       │                                                         │ │
│          │       ▼                                                         │ │
│          │    output: [1, 1, 640]                                          │ │
│          │    hidden_out: [1, 640]                                         │ │
│          │    cell_out: [1, 640]                                           │ │
│          │       │                                                         │ │
│          │       ▼                                                         │ │
│          │  LSTM Layer 2:                                                  │ │
│          │    (same structure)                                             │ │
│          │       │                                                         │ │
│          │       ▼                                                         │ │
│          │  pred_output: [1, 1, 640]                                       │ │
│          │  new_state: (hidden [2, 1, 640], cell [2, 1, 640])              │ │
│          └────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│          ┌────────────────────────────────────────────────────────────────┐ │
│          │              JOINT NETWORK                                      │ │
│          │  ═════════════════════                                          │ │
│          │                                                                 │ │
│          │  enc_frame: [1, 1, 1024]                                        │ │
│          │  pred_output: [1, 1, 640]                                       │ │
│          │                                                                 │ │
│          │  enc_proj = Linear(enc_frame)    → [1, 1, 640]                  │ │
│          │  pred_proj = Linear(pred_output) → [1, 1, 640]                  │ │
│          │                                                                 │ │
│          │  joint = ReLU(enc_proj + pred_proj)  → [1, 1, 640]              │ │
│          │                                                                 │ │
│          │  logits = Linear(joint)  → [1, 1, vocab_size+1]                 │ │
│          │                           = [1, 1, 1025]                        │ │
│          │                                                                 │ │
│          │  log_probs = log_softmax(logits)  → [1, 1, 1025]                │ │
│          └────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│          pred_token = argmax(log_probs)                                      │
│                                                                              │
│          if pred_token == BLANK_ID (1024):                                   │
│              break  // Move to next encoder frame                            │
│          else:                                                               │
│              emit(pred_token)                                                │
│              update decoder_state = new_state                                │
│              last_token = pred_token                                         │
│              symbols_emitted += 1                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decoding State Management

```rust
pub struct RnntStreamingSession {
    // Decoder states for each batch item
    pub decoder_states: Vec<DecoderState>,  // LSTM hidden/cell
    
    // Last emitted token (for prediction network input)
    pub last_tokens: Vec<Option<usize>>,
    
    // All tokens emitted so far (for final transcription)
    pub accumulated_tokens: Vec<Vec<usize>>,
    
    // ... other fields
}

// LSTM state structure
pub struct LstmState {
    pub hidden: Array,  // [num_layers=2, batch=1, hidden=640]
    pub cell: Array,    // [num_layers=2, batch=1, hidden=640]
}
```

---

## Complete Example

### Processing 1 Second of Audio

```
═══════════════════════════════════════════════════════════════════════════════
                        STREAMING 1 SECOND OF AUDIO
═══════════════════════════════════════════════════════════════════════════════

Input: 1 second = 16,000 samples

Chunking (160ms chunks):
  Chunk 1: samples[0:2560]     → 160ms
  Chunk 2: samples[2560:5120]  → 160ms  
  Chunk 3: samples[5120:7680]  → 160ms
  Chunk 4: samples[7680:10240] → 160ms
  Chunk 5: samples[10240:12800]→ 160ms
  Chunk 6: samples[12800:15360]→ 160ms
  (remaining 640 samples handled in finalize)

═══════════════════════════════════════════════════════════════════════════════

Processing Chunk 1 (first chunk):
────────────────────────────────

Audio: [2560 samples]
   │
   ▼
Mel frames: 2560/160 + 1 = 17 frames
   │
   │ chunk_size=1, shift_size=1, pre_encode_cache=0 (first chunk)
   │ Need 1 frame, have 17 → can process
   ▼
Extract mel chunk: [80, 1]
   │
   ▼
Encoder:
  • Subsampling: [80, 1] → insufficient (need at least 8 for 8x subsample)
  • Output: 0 encoder frames
   │
   ▼
No tokens emitted (no encoder output)

Buffer state after:
  • mel_buffer: 17 frames
  • buffer_idx: 1
  • remaining: 16 frames

═══════════════════════════════════════════════════════════════════════════════

Processing Chunk 2:
───────────────────

Audio: [2560 samples] (new)
   │
   ▼
Mel frames: +16 new frames (total: 33)
   │
   │ chunk_size=8, shift_size=8, pre_encode_cache=9 (subsequent)
   │ Need 8, have 32 remaining → can process
   ▼
Extract mel chunk: [80, 17]  (9 cache + 8 new)
   │
   ▼
Encoder:
  • Subsampling: [80, 17] → [1024, 2] (after drop_extra)
  • 24 Conformer layers with caching
  • valid_out_len truncation → [1024, 1]
   │
   ▼
RNN-T Decoding (1 encoder frame):
  • enc_frame[0]: predict → "▁The" (token 450)
  • enc_frame[0]: predict → BLANK
  • Move to next frame (none)
   │
   ▼
Output: ["▁The"]

═══════════════════════════════════════════════════════════════════════════════

Processing Chunks 3-6:
──────────────────────

Each chunk adds ~16 mel frames
Each extraction: 9 cache + 8 new = 17 mel frames
Encoder output: 1 frame per chunk
Tokens: gradually builds transcription

Example progression:
  Chunk 3: "▁The" → "▁The▁quick"
  Chunk 4: "▁The▁quick" → "▁The▁quick▁brown"  
  Chunk 5: "▁The▁quick▁brown" → "▁The▁quick▁brown▁fox"
  Chunk 6: "▁The▁quick▁brown▁fox" → "▁The▁quick▁brown▁fox▁jumps"

═══════════════════════════════════════════════════════════════════════════════

Final State:
────────────

Encoder caches (per layer):
  • Attention cache: [24, 1, 70, 1024] → filled ~6 frames
  • Conv cache: [24, 1, 1024, 8] → fully populated

Decoder state:
  • LSTM hidden: [2, 1, 640] → encodes label history
  • LSTM cell: [2, 1, 640]
  • last_token: "jumps" (or its token ID)

Accumulated tokens: [450, 234, 891, 102, 567, ...]
                    "The quick brown fox jumps..."

═══════════════════════════════════════════════════════════════════════════════
```

### Tensor Shapes Summary

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| Audio chunk | `audio` | `[2560]` | 160ms @ 16kHz |
| Mel buffer | `mel_buffer` | `[frames, 80]` | Internal storage |
| Mel chunk | `mel_chunk` | `[1, 80, 17]` | To encoder |
| After subsample | `encoded` | `[1, 1024, 2]` | Before drop_extra |
| After drop | `encoded` | `[1, 1024, 1]` | Before layers |
| Pos embedding | `pos_emb` | `[1, 71, 1024]` | cache + seq |
| Per-layer input | `x` | `[1, 1, 1024]` | [B, T, D] |
| Attention Q | `q` | `[1, 8, 1, 128]` | [B, heads, T, dim] |
| Attention K,V | `k,v` | `[1, 8, 36, 128]` | cache + T |
| Conv input | `conv_in` | `[1, 1024, 9]` | cache + T |
| Encoder output | `enc_out` | `[1, 1024, 1]` | [B, D, T] |
| For decoding | `enc_frame` | `[1, 1, 1024]` | [B, T, D] |
| Pred embedding | `emb` | `[1, 1, 640]` | Token embedding |
| LSTM output | `pred_out` | `[1, 1, 640]` | Label encoding |
| Joint logits | `logits` | `[1, 1, 1025]` | vocab + blank |
| Attention cache | `last_channel` | `[24, 1, 70, 1024]` | Per layer |
| Conv cache | `last_time` | `[24, 1, 1024, 8]` | Per layer |
| Cache fill | `cache_len` | `[1]` i64 | Filled frames |
| LSTM state | `hidden/cell` | `[2, 1, 640]` | Per layer |

---

## Key Insights

### Why Mel Frames Overlap

For streaming modes with right context > 0, mel frame chunks overlap significantly:

```
Mode 3 (right_context=13):
  chunk_size = 2504 frames (~20s of mel features)
  shift_size = 8 frames
  overlap = 2496 frames per step
```

This is **by design**. The overlap provides the encoder with lookahead context. The encoder caches prevent redundant computation:
- Attention K/V from overlapping frames are cached
- Only new frames trigger new computations
- `drop_extra_pre_encoded` removes redundant output frames

### Cache Size Constraints

| Cache | Size | Purpose |
|-------|------|---------|
| Attention (channel) | 70 frames | Left context window |
| Convolution (time) | 8 frames | kernel_size - 1 |
| Pre-encode | 9 frames | Subsampling continuity |

### Memory Bounds

The streaming implementation maintains O(1) memory:
- Audio buffer: ~window_samples (400 samples ≈ 1.6KB)
- Mel buffer: ~pre_encode_cache × n_mels (9 × 80 × 4 ≈ 2.9KB)  
- Encoder caches: Fixed size (24 × 70 × 1024 × 4 ≈ 6.7MB for attention)
- Decoder state: Fixed size (2 × 640 × 4 × 2 ≈ 10KB)

---

## Configuration-to-Pipeline Mapping

This section shows exactly how configuration changes affect tensor shapes and data flow at each stage.

### Example: Mode 3 (80ms, Ultra-Low Latency)

```
Configuration:
  att_context_size: [70, 0]
  att_context_style: "regular"
  subsampling_factor: 8
  n_layers: 24
  conv_kernel_size: 9

Derived Parameters:
  lookahead_steps = 0 × 24 = 0
  chunk_size = (1, 8)           # First: 1, Subsequent: 8
  shift_size = (1, 8)           # Same as chunk_size (no overlap!)
  pre_encode_cache = (0, 9)
  drop_extra = 2
  valid_out_len = 1
  cache_drop_size = 0
```

```
PIPELINE TRACE (Mode 3, processing 160ms audio):
════════════════════════════════════════════════

Audio: 2560 samples (160ms)
    │
    ▼ Mel spectrogram
Mel: 17 frames
    │
    ▼ Extract chunk (subsequent): chunk_size=8, pre_encode_cache=9
Mel chunk: [80, 17]  (9 cache + 8 new)
    │
    ▼ Subsampling (8x)
Subsampled: [1024, 2]
    │
    ▼ drop_extra_pre_encoded (drop first 2)
After drop: [1024, 0]  ← No output! Need more frames
    │
    ▼ (wait for next audio chunk)
    ...

After 2nd 160ms chunk:
Mel: 33 total frames, extract next 17
After subsampling + drop: [1024, 1]  ← 1 encoder output frame
    │
    ▼ valid_out_len truncation (keep 1)
Encoder output: [1024, 1]
    │
    ▼ RNN-T decode
Tokens emitted: depends on content
```

### Example: Mode 0 (1120ms, Best Accuracy)

```
Configuration:
  att_context_size: [70, 13]
  att_context_style: "regular"
  subsampling_factor: 8
  n_layers: 24
  conv_kernel_size: 9

Derived Parameters:
  lookahead_steps = 13 × 24 = 312
  chunk_size = (1 + 8×312, 8 + 8×312) = (2497, 2504)
  shift_size = (1, 8)           # Small! Massive overlap
  pre_encode_cache = (0, 9)
  drop_extra = 2
  valid_out_len = 1
  cache_drop_size = 312
```

```
PIPELINE TRACE (Mode 0, first output):
══════════════════════════════════════

Need chunk_size = 2504 mel frames before first encoder output
  = 2504 × 10ms = 25.04 seconds of audio!
  = 25.04s × 16000 Hz = 400,640 samples

Audio accumulation:
  Chunk 1:  2,560 samples →  17 mel frames (need 2504, waiting...)
  Chunk 2:  5,120 samples →  33 mel frames (need 2504, waiting...)
  ...
  Chunk 157: 400,640 samples → 2504+ mel frames (can process!)

First extraction:
  Mel chunk: [80, 2504]  (0 cache + 2504 new, first chunk)
      │
      ▼ Subsampling (8x)
  Subsampled: [1024, 313]  (2504/8 ≈ 313)
      │
      ▼ drop_extra_pre_encoded (drop first 2)
  After drop: [1024, 311]
      │
      ▼ valid_out_len truncation (keep 1)
  Encoder output: [1024, 1]  ← Finally! First output after 25s

Subsequent chunks (every 80ms = 8 mel frames):
  Extract: [80, 2513]  (9 cache + 2504 new)
  But only advance by shift_size = 8
  So 2496 frames overlap with previous chunk!
  
  The encoder caches K/V for these overlapping frames,
  so attention computation is still O(1) per new frame.
```

### How Overlap Affects Computation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODE 0 vs MODE 3: OVERLAP COMPARISON                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODE 3 (right_context=0): NO OVERLAP                                        │
│  ════════════════════════════════════                                        │
│                                                                              │
│  Time →                                                                      │
│  Mel frames:     [████████][████████][████████][████████]                    │
│                  chunk 1   chunk 2   chunk 3   chunk 4                       │
│                                                                              │
│  Each chunk is independent. Caches provide left context only.                │
│  Minimal computation per chunk.                                              │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  MODE 0 (right_context=13): MASSIVE OVERLAP                                  │
│  ═══════════════════════════════════════════                                 │
│                                                                              │
│  Time →                                                                      │
│  Mel frames:     [████████████████████████████████...████████]               │
│                  └──────────── chunk 1 (2504 frames) ─────────┘              │
│                          [████████████████████████████████...████████]       │
│                          └──────────── chunk 2 (2504 frames) ─────────┘      │
│                  ▲       ▲                                                   │
│                  │       └─ shift = 8 frames                                 │
│                  └───────── overlap = 2496 frames!                           │
│                                                                              │
│  The overlapping frames are NOT recomputed!                                  │
│  • Attention K/V for old frames → retrieved from cache                       │
│  • Conv activations for old frames → retrieved from cache                    │
│  • Only NEW 8 frames trigger computation                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Encoder Output Rate by Mode

| Mode | Audio per Output Frame | Output Rate | Notes |
|------|------------------------|-------------|-------|
| Mode 3 (80ms) | 80ms | 12.5 Hz | After initial 2 chunks (~160ms startup) |
| Mode 2 (160ms) | 80ms | 12.5 Hz | After ~200ms startup |
| Mode 1 (560ms) | 80ms | 12.5 Hz | After ~1.2s startup |
| Mode 0 (1120ms) | 80ms | 12.5 Hz | After ~25s startup |

**Key insight**: Once streaming is primed, ALL modes output at the same rate (1 encoder frame per 8 mel frames = 80ms). The difference is startup latency and accuracy.

### Cache Behavior by Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CACHE FILL PROGRESSION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Attention Cache (size = left_context = 70 frames):                          │
│                                                                              │
│  Step 0:  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  filled=0    │
│  Step 1:  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██]  filled=1    │
│  Step 2:  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████]  filled=2    │
│  ...                                                                         │
│  Step 70: [██████████████████████████████████████████████████]  filled=70   │
│  Step 71: [██████████████████████████████████████████████████]  filled=70   │
│           └─ oldest frame dropped, newest added (sliding window)             │
│                                                                              │
│  Conv Cache (size = kernel-1 = 8 frames):                                    │
│                                                                              │
│  Fills completely after first chunk, then slides.                            │
│  Always fully populated after step 1.                                        │
│                                                                              │
│  Offset Masking:                                                             │
│                                                                              │
│  Before cache is full, attention masks out unfilled positions.               │
│  offset = cache_size - filled = 70 - filled                                  │
│  Positions [0, offset) are masked (hidden from attention).                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Customizing Beyond Modes

While modes are the primary control, you can also affect streaming via:

```yaml
# In model config (requires retraining for changes)
encoder:
  # Context window
  att_context_size: [[70,13], [70,6], [70,1], [70,0]]
  att_context_style: "regular"  # or "chunked_limited"
  
  # Architecture (affects cache sizes)
  n_layers: 24                  # More layers = more lookahead accumulation
  conv_kernel_size: 9           # Larger = bigger conv cache (kernel-1)
  subsampling_factor: 8         # Affects chunk_size calculation
  
  # Attention
  n_heads: 8                    # Doesn't affect streaming params
  d_model: 1024                 # Affects cache memory, not streaming logic
```

### Runtime vs Config-Time Parameters

| Parameter | When Set | Can Change at Runtime? |
|-----------|----------|------------------------|
| `mode_idx` | `init_streaming()` | Yes (new session) |
| `batch_size` | `init_streaming()` | Yes (new session) |
| `att_context_size` | Model config | No (baked into weights) |
| `att_context_style` | Model config | No |
| `n_layers` | Model config | No |
| `conv_kernel_size` | Model config | No |
| `subsampling_factor` | Model config | No |