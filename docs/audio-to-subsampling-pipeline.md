# Raw Audio to Subsampling Output: A Complete Technical Reference

This document provides a detailed technical reference for transforming raw audio into subsampled encoder input for NVIDIA's NemoTron/FastConformer streaming ASR models. It covers every stage of the pipeline including framing, windowing, mel spectrogram computation, normalization, caching, and subsampling—with specific attention to streaming considerations.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Raw Audio Input](#raw-audio-input)
3. [Framing and Windowing](#framing-and-windowing)
4. [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
5. [Mel Filterbank](#mel-filterbank)
6. [Log Compression](#log-compression)
7. [Dithering](#dithering)
8. [Pre-emphasis Filtering](#pre-emphasis-filtering)
9. [Normalization](#normalization)
10. [Streaming Mel Buffer](#streaming-mel-buffer)
11. [Subsampling Module](#subsampling-module)
12. [Subsampling Cache for Streaming](#subsampling-cache-for-streaming)
13. [Output Format](#output-format)
14. [Complete Pipeline Summary](#complete-pipeline-summary)
15. [Implementation Notes](#implementation-notes)

---

## Pipeline Overview

The transformation from raw audio to subsampled encoder input follows this sequence:

```
Raw Audio (16 kHz, mono)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Optional: Pre-emphasis Filter (y[n] = x[n] - α·x[n-1]) │
│  Coefficient: α = 0.97                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Optional: Dithering (add small noise for quantization) │
│  Amplitude: ~1e-5                                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Framing: Extract overlapping windows                    │
│  • Window size: 25ms (400 samples @ 16kHz)               │
│  • Hop/stride: 10ms (160 samples)                        │
│  • Window function: Hann                                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STFT: FFT per frame                                     │
│  • FFT size: 512 (257 frequency bins)                    │
│  • Power spectrum: |FFT|²                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Mel Filterbank: Project to mel scale                    │
│  • 80 or 128 triangular filters                          │
│  • Frequency range: 0 Hz to 8000 Hz (Nyquist)            │
│  • Slaney normalization                                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Log Compression                                         │
│  • log(mel + ε) where ε = 2⁻²⁴ ≈ 5.96e-8                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Normalization                                           │
│  • Offline: per_feature (zero mean, unit var per bin)    │
│  • Streaming: running stats or skip                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Mel Spectrogram: [batch, n_mels, frames] @ 100 Hz
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  8x Subsampling (3 conv stages)                          │
│  • Stage 1: Conv2D(1→256, k=3, s=2) + ReLU   → 50 Hz    │
│  • Stage 2: DWConv2D + PWConv2D + ReLU       → 25 Hz    │
│  • Stage 3: DWConv2D + PWConv2D + ReLU       → 12.5 Hz  │
│  • Linear projection to d_model                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Encoder Input: [batch, time/8, d_model] @ 12.5 Hz (80ms/frame)
```

---

## Raw Audio Input

### Requirements

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 16,000 Hz | Standard for ASR; resample if different |
| Channels | 1 (mono) | Stereo → average channels |
| Bit Depth | 16-bit or float | Normalize to [-1.0, 1.0] range |
| Format | PCM | WAV, FLAC, or raw samples |

### Loading and Resampling

```rust
// Load WAV file, convert stereo to mono
let (samples, sample_rate) = load_wav("audio.wav")?;

// Resample if needed
let samples = if sample_rate != 16000 {
    resample(&samples, sample_rate, 16000)?
} else {
    samples
};
```

**Resampling Notes:**
- Use high-quality sinc interpolation (e.g., `rubato` crate)
- Anti-aliasing filter cutoff: 0.95 × Nyquist
- Oversampling factor: 256 for quality

---

## Framing and Windowing

### Frame Extraction

Audio is divided into overlapping frames for short-time analysis:

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Window Size | 25 ms | 400 samples @ 16 kHz |
| Window Stride (Hop) | 10 ms | 160 samples @ 16 kHz |
| Frame Rate | 100 Hz | 1000 ms / 10 ms stride |
| Overlap | 60% | (400 - 160) / 400 |

**Frame Count Formula:**
```
num_frames = floor((audio_len - window_samples) / hop_samples) + 1

With center padding:
num_frames = floor(audio_len / hop_samples) + 1
```

### Window Function

The **Hann window** is applied to each frame to reduce spectral leakage:

```
w[n] = 0.5 × (1 - cos(2π × n / (N - 1)))
```

where `N = 400` (window length).

**Properties:**
- Smooth taper to zero at endpoints
- Main lobe width: 4 bins
- First sidelobe: -31 dB
- Symmetric: `w[n] = w[N-1-n]`

```rust
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| {
            0.5 * (1.0 - (2.0 * PI * n as f32 / (length - 1) as f32).cos())
        })
        .collect()
}
```

### Center Padding

NeMo uses **reflection padding** to center frames over the audio:

```
Original:     [s0, s1, s2, s3, ..., sN-1, sN]
Padded:   [s256, ..., s1 | s0, s1, s2, ..., sN | sN-1, ..., sN-255]
                ↑ reflect        audio         reflect ↑
```

- Pad size: `n_fft / 2 = 256` samples on each side
- First frame is centered at sample 0
- Avoids edge artifacts

---

## Short-Time Fourier Transform (STFT)

### Process

For each windowed frame:

1. **Apply window:** `frame_windowed[n] = frame[n] × window[n]`
2. **Zero-pad to FFT size:** `[400 samples, 112 zeros]` → 512 samples
3. **Compute FFT:** Complex spectrum of 512 points
4. **Take positive frequencies:** First 257 bins (DC to Nyquist)
5. **Compute power:** `|X[k]|² = real[k]² + imag[k]²`

### Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| FFT Size (n_fft) | 512 | Power of 2 for efficiency |
| Frequency Bins | 257 | n_fft/2 + 1 |
| Frequency Resolution | 31.25 Hz | sample_rate / n_fft |

### Frequency Bin Mapping

```
bin[k] → frequency = k × (sample_rate / n_fft)
       → frequency = k × 31.25 Hz

bin[0]   → 0 Hz (DC)
bin[128] → 4000 Hz
bin[256] → 8000 Hz (Nyquist)
```

---

## Mel Filterbank

### Mel Scale

The mel scale models human perception of pitch:

**Slaney Formula (NeMo default):**
```
For f < 1000 Hz:  mel = f / (200/3)
For f ≥ 1000 Hz:  mel = 15.0 + ln(f / 1000) / 0.0687517...
```

This is a hybrid linear-log scale with breakpoint at 1000 Hz.

### Triangular Filters

80 overlapping triangular filters span the mel scale:

```
         Filter m-1    Filter m    Filter m+1
              ▲           ▲           ▲
             /│\         /│\         /│\
            / │ \       / │ \       / │ \
           /  │  \     /  │  \     /  │  \
          /   │   \   /   │   \   /   │   \
─────────┴────┴────┴─┴────┴────┴─┴────┴────┴─────
        f_l   f_c  f_r f_l  f_c  f_r
```

Each filter:
- Rises linearly from left edge to center
- Falls linearly from center to right edge
- Overlaps 50% with neighbors
- Peak amplitude normalized (Slaney normalization)

### Filterbank Matrix

Shape: `[n_mels, n_fft/2+1]` = `[80, 257]`

```rust
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, f_min: f32, f_max: f32) -> Array {
    // 1. Compute mel points (n_mels + 2 to include edges)
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    
    // 2. Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    
    // 3. Build triangular filters
    let n_freqs = n_fft / 2 + 1;
    let mut fb = vec![0.0f32; n_mels * n_freqs];
    
    for m in 0..n_mels {
        let (f_left, f_center, f_right) = (hz_points[m], hz_points[m+1], hz_points[m+2]);
        
        for k in 0..n_freqs {
            let freq = k as f32 * (sr / 2.0) / (n_freqs - 1) as f32;
            
            // Triangular filter response
            let response = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            
            fb[m * n_freqs + k] = response;
        }
    }
    
    Array::from_slice(&fb, &[n_mels as i32, n_freqs as i32])
}
```

### Application

```
mel_spec[m, t] = Σₖ filterbank[m, k] × power_spec[k, t]
```

This is a matrix multiplication: `mel = filterbank @ power_spec`

---

## Log Compression

### Purpose

Log compression serves two purposes:
1. **Dynamic range reduction:** Speech spans ~60 dB; log compresses this
2. **Perceptual alignment:** Human loudness perception is approximately logarithmic

### Formula

```
log_mel[m, t] = log(mel[m, t] + guard)
```

**Guard value:** `ε = 2⁻²⁴ ≈ 5.96 × 10⁻⁸`

This prevents `log(0) = -∞` for silent frames while being small enough not to affect non-silent values.

### Implementation

```rust
fn log_compress(mel: &Array, guard: f32) -> Array {
    let guard_arr = Array::from_f32(guard);
    let mel_safe = ops::add(mel, &guard_arr)?;
    ops::log(&mel_safe)?
}

// NeMo default
let log_mel = log_compress(&mel, 2.0_f32.powi(-24))?;
```

---

## Dithering

### Purpose

Dithering adds small random noise to break up quantization patterns and improve model robustness to silence.

### When to Apply

| Mode | Dithering |
|------|-----------|
| Training | Enabled (default: 1e-5) |
| Inference | Typically disabled |
| Streaming | Disabled (determinism) |

### Implementation

```rust
fn apply_dither(audio: &[f32], dither_amount: f32) -> Vec<f32> {
    if dither_amount <= 0.0 {
        return audio.to_vec();
    }
    
    let mut rng = rand::thread_rng();
    audio.iter()
        .map(|&s| s + rng.gen_range(-dither_amount..dither_amount))
        .collect()
}
```

### NeMo Configuration

```yaml
preprocessor:
  dither: 0.00001  # Training
  # dither: 0.0    # Inference
```

---

## Pre-emphasis Filtering

### Purpose

Pre-emphasis boosts high-frequency content to:
1. Compensate for spectral tilt in speech
2. Improve consonant recognition
3. Balance energy across frequencies

### Filter Definition

First-order high-pass filter:
```
y[n] = x[n] - α × x[n-1]
```

**Coefficient:** `α = 0.97` (NeMo default)

### Frequency Response

```
H(ω) = 1 - α × e^(-jω)
|H(ω)|² = 1 + α² - 2α cos(ω)
```

- At DC (ω=0): `|H(0)| = 1 - α = 0.03` (attenuation)
- At Nyquist (ω=π): `|H(π)| = 1 + α = 1.97` (boost)
- Crossover (~1.5 kHz): Unity gain

### Batch vs Streaming

**Batch Mode:**
```rust
fn preemphasis(audio: &Array, coef: f32) -> Array {
    // First sample unchanged
    let first = audio.index((.., ..1));
    // y[n] = x[n] - coef * x[n-1] for n > 0
    let current = audio.index((.., 1..));
    let previous = audio.index((.., ..-1));
    let filtered = ops::subtract(&current, &(coef * &previous))?;
    ops::concatenate_axis(&[&first, &filtered], 1)?
}
```

**Streaming Mode (with state):**
```rust
fn preemphasis_streaming(
    audio: &[f32], 
    coef: f32, 
    prev_sample: f32
) -> (Vec<f32>, f32) {
    let mut output = Vec::with_capacity(audio.len());
    
    // First sample uses prev_sample from previous chunk
    output.push(audio[0] - coef * prev_sample);
    
    // Subsequent samples use previous sample in current chunk
    for i in 1..audio.len() {
        output.push(audio[i] - coef * audio[i - 1]);
    }
    
    // Return output and last sample for next chunk
    (output, *audio.last().unwrap())
}
```

**Critical for Streaming:** The `prev_sample` state must be maintained across chunks to avoid discontinuities at chunk boundaries.

---

## Normalization

### Per-Feature Normalization (Offline)

Normalize each mel bin independently across time:

```rust
fn normalize_per_feature(features: &Array) -> Array {
    // features: [batch, n_mels, frames]
    
    // Mean per mel bin: [batch, n_mels, 1]
    let mean = ops::mean_axis(features, 2, true)?;
    let centered = ops::subtract(features, &mean)?;
    
    // Std per mel bin: [batch, n_mels, 1]
    let variance = ops::mean_axis(&ops::square(&centered)?, 2, true)?;
    let std = ops::sqrt(&(variance + 1e-5))?;
    
    ops::divide(&centered, &std)?
}
```

### Streaming Normalization Options

For streaming, full-utterance statistics aren't available. Options:

#### Option 1: Skip Normalization
```yaml
preprocessor:
  normalize: "NA"
```
- Pros: Simple, no state
- Cons: Distribution shift from training

#### Option 2: Running Statistics
```rust
struct RunningNormalizer {
    sum: Vec<f32>,      // Running sum per mel bin
    sum_sq: Vec<f32>,   // Running sum of squares
    count: usize,       // Number of frames seen
}

impl RunningNormalizer {
    fn update(&mut self, frame: &[f32]) {
        for (i, &val) in frame.iter().enumerate() {
            self.sum[i] += val;
            self.sum_sq[i] += val * val;
        }
        self.count += 1;
    }
    
    fn get_stats(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.count as f32;
        let mean: Vec<f32> = self.sum.iter().map(|&s| s / n).collect();
        let std: Vec<f32> = self.sum.iter().zip(&self.sum_sq)
            .map(|(&s, &sq)| ((sq / n) - (s / n).powi(2)).max(1e-5).sqrt())
            .collect();
        (mean, std)
    }
}
```

#### Option 3: Fixed Statistics
Use pre-computed dataset statistics:
```rust
let (dataset_mean, dataset_std) = load_normalization_stats("stats.json")?;
let normalized = normalize_with_stats(&features, &dataset_mean, &dataset_std)?;
```

### NeMo's Online Normalization

NeMo's `CacheAwareStreamingAudioBuffer` supports online normalization:

```python
streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=asr_model,
    online_normalization=True,  # Use running stats
    pad_and_drop_preencoded=True,
)
```

---

## Streaming Mel Buffer

### Purpose

The streaming mel buffer manages:
1. Audio sample accumulation
2. Incremental mel computation
3. Overlap handling for STFT continuity
4. Chunk extraction with pre-encode cache

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    StreamingMelBuffer                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Audio Buffer (sliding window)                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ [overlap samples | new unprocessed samples]            │   │
│  │  ← window_samples →                                    │   │
│  └────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│              ┌─────────────────────────┐                      │
│              │   Mel Computation       │                      │
│              │   (STFT + filterbank)   │                      │
│              └─────────────────────────┘                      │
│                           │                                    │
│                           ▼                                    │
│  Mel Buffer (frame storage)                                    │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ [frame0, frame1, frame2, ..., frameN]                  │   │
│  │  ← n_mels features per frame →                         │   │
│  └────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  Chunk Extraction:                                             │
│  ┌──────────────────────┬────────────────────────────────┐   │
│  │ pre_encode_cache     │ chunk_size frames               │   │
│  │ (9 frames for 8x)    │ (varies by latency mode)        │   │
│  └──────────────────────┴────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Key State Variables

```rust
struct StreamingMelBuffer {
    // Mel frame storage [time, n_mels] format internally
    mel_buffer: Vec<f32>,
    n_mels: usize,
    total_frames: usize,
    
    // Current read position (analogous to NeMo's buffer_idx)
    buffer_idx: usize,
    
    // Audio accumulation
    audio_buffer: Vec<f32>,
    total_audio_samples: usize,
    
    // STFT parameters
    sample_rate: u32,
    window_samples: usize,  // 400 @ 16kHz
    hop_samples: usize,     // 160 @ 16kHz
    n_fft: usize,           // 512
    
    // Memory management
    mel_buffer_offset: usize,  // Frames trimmed from start
}
```

### Operations

#### Add Audio
```rust
fn add_audio(&mut self, samples: &[f32]) {
    self.audio_buffer.extend_from_slice(samples);
    self.total_audio_samples += samples.len();
}
```

#### Check if Ready
```rust
fn has_pending_audio(&self) -> bool {
    self.audio_buffer.len() >= self.window_samples  // Need at least one frame
}
```

#### Append Computed Mel Frames
```rust
fn append_mel(&mut self, mel: &[f32], num_frames: usize) {
    // Input: [n_mels, time] from preprocessor
    // Store as [time, n_mels] for efficient frame access
    for frame_idx in 0..num_frames {
        for mel_idx in 0..self.n_mels {
            let src_idx = mel_idx * num_frames + frame_idx;
            self.mel_buffer.push(mel[src_idx]);
        }
    }
    self.total_frames += num_frames;
}
```

#### Extract Chunk
```rust
fn extract_chunk(
    &mut self,
    chunk_size: usize,
    shift_size: usize,
    pre_encode_cache: usize,
) -> Option<(Vec<f32>, usize)> {
    // Check if enough frames
    if self.buffer_idx + chunk_size > self.total_frames {
        return None;
    }
    
    // Include pre_encode_cache frames from before buffer_idx
    let cache_start = self.buffer_idx.saturating_sub(pre_encode_cache);
    let total_extract = (self.buffer_idx - cache_start) + chunk_size;
    
    // Extract and transpose to [n_mels, time]
    let mut chunk = vec![0.0f32; self.n_mels * total_extract];
    for frame_offset in 0..total_extract {
        let frame_idx = cache_start + frame_offset;
        for mel_idx in 0..self.n_mels {
            let src = (frame_idx - self.mel_buffer_offset) * self.n_mels + mel_idx;
            let dst = mel_idx * total_extract + frame_offset;
            chunk[dst] = self.mel_buffer[src];
        }
    }
    
    // Advance position
    self.buffer_idx += shift_size;
    
    Some((chunk, total_extract))
}
```

#### Drain Processed Audio
```rust
fn drain_processed_audio(&mut self) {
    // Keep enough for STFT overlap (one window worth)
    let keep = self.window_samples;
    if self.audio_buffer.len() > keep {
        self.audio_buffer.drain(..self.audio_buffer.len() - keep);
    }
}
```

#### Trim Old Frames
```rust
fn trim_old_frames(&mut self, keep_frames: usize) {
    // Remove frames that are no longer needed for pre_encode_cache
    let can_trim = self.buffer_idx.saturating_sub(keep_frames + self.mel_buffer_offset);
    if can_trim > 0 {
        self.mel_buffer.drain(..can_trim * self.n_mels);
        self.mel_buffer_offset += can_trim;
    }
}
```

---

## Subsampling Module

### Purpose

Reduce the mel spectrogram frame rate by 8x using convolutional downsampling:
- Input: 100 Hz (10ms per frame)
- Output: 12.5 Hz (80ms per frame)

This dramatically reduces the sequence length for efficient attention computation.

### FastConformer Architecture (dw_striding)

```
Input: [B, T, F=80, C=1]  (mel spectrogram with channel dim)
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │  Stage 1: 2x Downsample (100 Hz → 50 Hz)             │
    │  Conv2D(1 → 256, kernel=3×3, stride=2×2)             │
    │  + ReLU                                              │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │  Stage 2: 2x Downsample (50 Hz → 25 Hz)              │
    │  DepthwiseConv2D(256, kernel=3×3, stride=2×2)        │
    │  + PointwiseConv2D(256 → 256, kernel=1×1)            │
    │  + ReLU                                              │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │  Stage 3: 2x Downsample (25 Hz → 12.5 Hz)            │
    │  DepthwiseConv2D(256, kernel=3×3, stride=2×2)        │
    │  + PointwiseConv2D(256 → 256, kernel=1×1)            │
    │  + ReLU                                              │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │  Reshape & Project                                    │
    │  [B, T/8, F/8, 256] → [B, T/8, 256 × F/8]            │
    │  Linear(256 × F/8 → d_model)                         │
    └──────────────────────────────────────────────────────┘
         │
         ▼
Output: [B, T/8, d_model=1024]
```

### Depthwise Separable Convolution

**Depthwise Conv:** Each channel convolved independently
- Groups = input channels
- No channel mixing

**Pointwise Conv:** 1×1 convolution for channel mixing
- Mixes information across channels
- Much fewer parameters than full 3×3

```
Standard Conv2D(256→256, 3×3): 256 × 256 × 9 = 589,824 params
Depthwise + Pointwise:
  DW: 256 × 1 × 9 = 2,304 params
  PW: 256 × 256 × 1 = 65,536 params
  Total: 67,840 params (8.7x reduction)
```

### Causal Padding for Streaming

For streaming, convolutions must be **causal** (no future context):

**Standard (non-causal) padding:**
```
Pad: (kernel-1)/2 on each side
Example (k=3): [pad, input, pad] = [1, T, 1]
Output centered over input
```

**Causal padding:**
```
Pad: kernel-1 on left, 0 on right
Example (k=3): [pad, pad, input] = [2, T]
Output uses only past context
```

NeMo's `CausalConv2D` padding (used in streaming):
```
left_pad = kernel_size - 1 = 2
right_pad = (kernel_size - 1) // 2 = 1

Applied to BOTH time and frequency dimensions
```

### Output Size Calculation

**Causal mode:**
```
output_size = floor(input_size / stride) + 1

For 8x subsampling (3 stages, stride=2 each):
T → floor(T/2)+1 → floor((floor(T/2)+1)/2)+1 → ...
```

**Frequency dimension:**
```
F=80 → 80/2+1=41 → 41/2+1=21 → 21/2+1=11

Linear input: 256 × 11 = 2816
```

### Implementation

```rust
struct Subsampling {
    first_conv: Conv2D,           // 1 → 256 channels
    depthwise_convs: Vec<Conv2D>, // Depthwise (groups=256)
    pointwise_convs: Vec<Conv2D>, // Pointwise (1×1)
    out_proj: Linear,             // Project to d_model
    is_causal: bool,
    kernel_size: usize,
    stride: usize,
}

impl Subsampling {
    fn forward(&mut self, x: &Array, lengths: &Array) -> (Array, Array) {
        // x: [B, T, F] → [B, T, F, 1]
        let mut x = x.reshape(&[-1, x.dim(1), x.dim(2), 1])?;
        
        // Stage 1
        if self.is_causal {
            x = causal_pad_2d(&x, self.kernel_size, self.stride)?;
        }
        x = relu(self.first_conv.forward(&x)?)?;
        
        // Stages 2-3
        for (dw, pw) in self.depthwise_convs.iter()
                            .zip(self.pointwise_convs.iter()) {
            if self.is_causal {
                x = causal_pad_2d(&x, self.kernel_size, self.stride)?;
            }
            x = relu(pw.forward(&dw.forward(&x)?)?)?;
        }
        
        // Reshape: [B, T', F', C] → [B, T', C×F']
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        x = x.transpose(&[0, 1, 3, 2])?;  // [B, T', C, F']
        x = x.reshape(&[b, t, c * f])?;
        
        // Project to d_model
        let x = self.out_proj.forward(&x)?;
        
        // Update lengths
        let new_lengths = compute_output_lengths(lengths, self.is_causal);
        
        (x, new_lengths)
    }
}
```

---

## Subsampling Cache for Streaming

### Pre-Encode Cache

For streaming, subsampling needs context from previous chunks to maintain continuity:

```
┌─────────────────────────────────────────────────────────────┐
│                Chunk Boundary Problem                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Without cache:                                              │
│  Chunk 1: [mel0, mel1, ..., mel111]                         │
│  Chunk 2: [mel112, mel113, ...]   ← Missing context!        │
│                                                              │
│  With pre_encode_cache=9:                                    │
│  Chunk 1: [mel0, mel1, ..., mel111]                         │
│  Chunk 2: [mel103..mel111 | mel112, mel113, ...]            │
│            ↑ cache (9)      ↑ new frames                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Cache Size Calculation

NeMo computes `pre_encode_cache_size` based on subsampling type:

```python
# For dw_striding subsampling:
pre_encode_cache_size = (0, subsampling_factor + 1)
# First chunk: 0 (no cache)
# Subsequent: 8 + 1 = 9 frames
```

### Drop Extra Pre-Encoded

After subsampling with cache, some output frames may be duplicates that need dropping:

```python
drop_extra_pre_encoded = 1 + (pre_encode_cache_size[1] - 1) // subsampling_factor
# = 1 + (9 - 1) // 8 = 1 + 1 = 2
```

This ensures only `valid_out_len` frames are kept per chunk.

### Streaming Configuration

```rust
struct StreamingConfig {
    // Chunk sizes in mel frames
    chunk_size: (usize, usize),      // (first_chunk, subsequent)
    shift_size: (usize, usize),      // How much to advance
    
    // Cache sizes
    pre_encode_cache_size: (usize, usize),  // (first=0, subsequent=9)
    drop_extra_pre_encoded: usize,          // Frames to drop post-subsample
    
    // Output
    valid_out_len: usize,  // Valid encoder output frames per chunk
}

// Example for 1120ms latency mode (lookahead=13):
// chunk_size = (1, 8) + 8 * 13 = (1, 112) mel frames
// shift_size = 112 mel frames
// pre_encode_cache = 9 mel frames
// valid_out_len = 14 encoder frames
```

---

## Output Format

### Mel Spectrogram Output

```
Shape: [batch, n_mels, frames]
       [1, 80, T] or [1, 128, T]

Data type: float32
Value range: ~[-10, 10] after log + normalization
Frame rate: 100 Hz (10ms per frame)
```

### Subsampled Encoder Input

```
Shape: [batch, time, d_model]
       [1, T/8, 1024]

Data type: float32
Value range: Depends on model scaling
Frame rate: 12.5 Hz (80ms per frame)
```

### Time Relationships

| Audio Duration | Audio Samples | Mel Frames | Encoder Frames |
|----------------|---------------|------------|----------------|
| 80 ms | 1,280 | 8 | 1 |
| 1 second | 16,000 | 100 | ~12-13 |
| 10 seconds | 160,000 | 1,000 | ~125 |

**Exact formula (with center padding, causal):**
```
mel_frames = floor(audio_samples / hop_samples) + 1
encoder_frames ≈ mel_frames / 8  (with adjustments for padding)
```

---

## Complete Pipeline Summary

### Offline (Batch) Processing

```rust
fn process_audio_offline(audio: &[f32], config: &PreprocessorConfig) -> Array {
    // 1. Pre-emphasis (optional)
    let audio = preemphasis(&audio, config.preemph.unwrap_or(0.0))?;
    
    // 2. STFT with center padding
    let window = hann_window(config.window_samples);
    let power = stft_power(&audio, config.n_fft, config.hop_samples, 
                           config.window_samples, &window, true)?;
    
    // 3. Mel filterbank
    let filterbank = mel_filterbank(config.n_mels, config.n_fft, 
                                    config.sample_rate, 0.0, 8000.0)?;
    let mel = filterbank @ power;  // Matrix multiply
    
    // 4. Log compression
    let log_mel = log(mel + 2e-24);
    
    // 5. Per-feature normalization
    let normalized = normalize_per_feature(&log_mel)?;
    
    normalized  // [1, n_mels, frames]
}
```

### Streaming Processing

```rust
struct StreamingMelPipeline {
    processor: MelProcessor,
    mel_buffer: StreamingMelBuffer,
    preemph_state: f32,
}

impl StreamingMelPipeline {
    fn process_chunk(&mut self, audio: &[f32]) -> Option<Array> {
        // 1. Pre-emphasis with state
        let (preemph_audio, new_state) = 
            preemphasis_streaming(audio, 0.97, self.preemph_state)?;
        self.preemph_state = new_state;
        
        // 2. Add to buffer
        self.mel_buffer.add_audio(&preemph_audio);
        
        // 3. Compute mel if enough samples
        if !self.mel_buffer.has_pending_audio() {
            return None;
        }
        
        let pending = self.mel_buffer.get_pending_audio();
        let mel = self.processor.forward_unnormalized(&pending)?;
        
        // 4. Append to mel buffer
        let mel_data: &[f32] = mel.as_slice();
        let num_frames = mel.dim(2);
        self.mel_buffer.append_mel(mel_data, num_frames);
        
        // 5. Clean up audio buffer
        self.mel_buffer.drain_processed_audio();
        
        // 6. Extract chunk with pre-encode cache
        let (chunk, len) = self.mel_buffer.extract_chunk(
            self.chunk_size,
            self.shift_size,
            self.pre_encode_cache,
        )?;
        
        Some(Array::from_slice(&chunk, &[1, self.n_mels, len]))
    }
}
```

---

## Implementation Notes

### Memory Efficiency

1. **Audio Buffer:** Keep only `window_samples` overlap (~400 samples = 1.6 KB)
2. **Mel Buffer:** Trim old frames, keep only `pre_encode_cache` + current chunk
3. **Total streaming memory:** O(window_samples + n_mels × pre_encode_cache) ≈ 5 KB

### Numerical Stability

1. **Log guard:** Always use `log(x + ε)` with `ε = 2⁻²⁴`
2. **Normalization:** Add `1e-5` to std before division
3. **Pre-emphasis:** Handle first sample specially in streaming

### Alignment Considerations

1. **Frame boundaries:** Audio chunks should align to `hop_samples` boundaries
2. **Subsampling:** Mel chunks should be multiples of 8 for clean subsampling
3. **Causality:** All streaming operations must be causal (no future lookahead in preprocessing)

### Testing

```rust
#[test]
fn test_streaming_matches_offline() {
    let config = PreprocessorConfig::default();
    let audio = load_test_audio();
    
    // Offline processing
    let offline_mel = process_offline(&audio, &config)?;
    
    // Streaming processing (concatenate all chunks)
    let mut stream = StreamingMelPipeline::new(config)?;
    let mut streaming_mel = Vec::new();
    for chunk in audio.chunks(1600) {  // 100ms chunks
        if let Some(mel) = stream.process_chunk(chunk)? {
            streaming_mel.extend(mel.as_slice::<f32>());
        }
    }
    if let Some(final_mel) = stream.flush()? {
        streaming_mel.extend(final_mel.as_slice::<f32>());
    }
    
    // Compare (allowing for boundary effects)
    assert_mel_approximately_equal(&offline_mel, &streaming_mel, tolerance=1e-4);
}
```

---

## References

1. **Fast Conformer Paper:** [arXiv:2305.05084](https://arxiv.org/abs/2305.05084)
2. **Stateful Conformer Paper:** [arXiv:2312.17279](https://arxiv.org/abs/2312.17279)
3. **NeMo ASR Documentation:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
4. **Mel Scale (Slaney):** Auditory Toolbox, Malcolm Slaney, 1998
