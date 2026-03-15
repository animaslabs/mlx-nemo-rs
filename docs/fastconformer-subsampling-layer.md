# FastConformer Subsampling Layer: Complete Technical Reference

> A comprehensive guide to the depthwise-separable striding subsampling architecture used in NVIDIA's Nemotron Speech Streaming ASR and FastConformer models.

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
   - [2.1 The Subsampling Problem](#21-the-subsampling-problem)
   - [2.2 Layer-by-Layer Architecture](#22-layer-by-layer-architecture)
   - [2.3 Comparison with Other Subsampling Methods](#23-comparison-with-other-subsampling-methods)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - [3.1 Output Length Calculations](#31-output-length-calculations)
   - [3.2 Temporal Resolution and Frame Alignment](#32-temporal-resolution-and-frame-alignment)
   - [3.3 Receptive Field Analysis](#33-receptive-field-analysis)
4. [Causal Mode and Streaming Inference](#4-causal-mode-and-streaming-inference)
   - [4.1 Causal Padding Mechanics](#41-causal-padding-mechanics)
   - [4.2 Cache Requirements Derivation](#42-cache-requirements-derivation)
   - [4.3 Streaming Inference Algorithm](#43-streaming-inference-algorithm)
   - [4.4 Chunk Boundary Handling](#44-chunk-boundary-handling)
5. [Implementation Details](#5-implementation-details)
   - [5.1 Configuration Parameters](#51-configuration-parameters)
   - [5.2 Weight Initialization](#52-weight-initialization)
   - [5.3 Integration with Encoder Caching](#53-integration-with-encoder-caching)
6. [Practical Considerations](#6-practical-considerations)
   - [6.1 Memory and Compute Trade-offs](#61-memory-and-compute-trade-offs)
   - [6.2 First Chunk Handling](#62-first-chunk-handling)
   - [6.3 Variable Length Sequences](#63-variable-length-sequences)
7. [References](#7-references)

---

## 1. Overview

The subsampling layer is the entry point of the FastConformer encoder, responsible for reducing the temporal resolution of mel-spectrogram features before they enter the main encoder blocks. In the **Nemotron Speech Streaming** model, this layer performs **8x temporal downsampling** using a stack of depthwise-separable striding convolutions.

### Key Specifications (Nemotron Speech Streaming)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `subsampling` | `dw_striding` | Depthwise-separable striding convolutions |
| `subsampling_factor` | 8 | Total temporal reduction factor |
| `subsampling_conv_channels` | 256 | Intermediate channel dimension |
| `causal_downsampling` | `true` | Enables streaming inference |
| `feat_in` | 128 | Input mel-spectrogram bins |
| `d_model` | 1024 | Output dimension to encoder |

### What This Layer Does

```
Input:  Mel-spectrogram [B, 1, T, F]     →  [B, 1, T, 128]  @ 10ms per frame
Output: Encoder input   [B, T', D]       →  [B, T/8, 1024]  @ 80ms per frame
```

The 8x downsampling is critical for computational efficiency: it reduces the sequence length that all subsequent self-attention layers must process, cutting their quadratic complexity by a factor of 64 (8²).

---

## 2. Architecture Deep Dive

### 2.1 The Subsampling Problem

Speech signals sampled at 16kHz, when converted to mel-spectrograms with a hop size of 160 samples (10ms), produce approximately **100 frames per second**. For a Conformer with 24 self-attention layers, processing these directly would be computationally prohibitive.

The original Conformer used **4x downsampling**, yielding frames at 40ms intervals. FastConformer increases this to **8x downsampling** (80ms intervals), which:

1. Reduces self-attention compute by 4x compared to 4x downsampling
2. Reduces memory requirements proportionally
3. Enables longer maximum audio duration on fixed hardware
4. Maintains accuracy when using subword tokenization (BPE)

> **Note on CTC Constraint**: The CTC loss requires `output_length >= target_length`. With 8x downsampling, character-level tokenization often violates this constraint. Nemotron Speech uses **1024-token BPE vocabulary** for RNNT, making 8x downsampling viable. Models using CTC with 8x downsampling typically use 128-1024 BPE tokens.

### 2.2 Layer-by-Layer Architecture

The `dw_striding` subsampling module consists of **3 convolutional stages** (since log₂(8) = 3), each performing 2x temporal and frequency reduction.

#### Complete Layer Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUBSAMPLING MODULE (dw_striding)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [B, 1, T, 128]  (mel-spectrogram)                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Standard Conv2D                                 │   │
│  │   Conv2D(1 → 256, k=3×3, s=2, groups=1)                 │   │
│  │   ReLU                                                   │   │
│  │   Output: [B, 256, T/2, 64]                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Stage 2: Depthwise Separable Conv2D                      │   │
│  │   DepthwiseConv2D(256 → 256, k=3×3, s=2, groups=256)    │   │
│  │   PointwiseConv2D(256 → 256, k=1×1, s=1, groups=1)      │   │
│  │   ReLU                                                   │   │
│  │   Output: [B, 256, T/4, 32]                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Stage 3: Depthwise Separable Conv2D                      │   │
│  │   DepthwiseConv2D(256 → 256, k=3×3, s=2, groups=256)    │   │
│  │   PointwiseConv2D(256 → 256, k=1×1, s=1, groups=1)      │   │
│  │   ReLU                                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Reshape + Linear Projection                              │   │
│  │   Reshape: [B, 256, T/8, F'] → [B, T/8, 256 * F']       │   │
│  │   Linear(256 * F' → 1024)                                │   │
│  │   Output: [B, T/8, 1024]                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Detailed Layer Parameters

| Stage | Layer Type | In Channels | Out Channels | Kernel | Stride | Padding* | Groups |
|-------|------------|-------------|--------------|--------|--------|----------|--------|
| 1 | Conv2D | 1 | 256 | 3×3 | 2 | 1 | 1 |
| 1 | ReLU | - | - | - | - | - | - |
| 2 | Conv2D (DW) | 256 | 256 | 3×3 | 2 | 1 | 256 |
| 2 | Conv2D (PW) | 256 | 256 | 1×1 | 1 | 0 | 1 |
| 2 | ReLU | - | - | - | - | - | - |
| 3 | Conv2D (DW) | 256 | 256 | 3×3 | 2 | 1 | 256 |
| 3 | Conv2D (PW) | 256 | 256 | 1×1 | 1 | 0 | 1 |
| 3 | ReLU | - | - | - | - | - | - |
| - | Linear | 256 × F' | 1024 | - | - | - | - |

*\* Padding shown is for non-causal mode. Causal mode uses asymmetric padding (see Section 4).*

#### Why Depthwise Separable?

A standard 3×3 convolution with C input and C output channels requires C² × 9 parameters and C² × 9 × H × W multiply-accumulate operations.

Depthwise separable convolutions factorize this into:
- **Depthwise**: C × 9 parameters (one 3×3 filter per channel)
- **Pointwise**: C² × 1 parameters (channel mixing via 1×1 conv)

For C = 256:
- Standard Conv: 256² × 9 = 589,824 parameters
- Depthwise Separable: 256 × 9 + 256² = 67,840 parameters (**8.7x reduction**)

> **Model Variant Note**: The `subsampling_conv_channels` parameter controls the intermediate channel dimension. Smaller models may use 128 or 192 channels, while larger models (XL, XXL) may use 512. The Nemotron Speech 0.6B model uses 256.

### 2.3 Comparison with Other Subsampling Methods

| Method | Architecture | Final Factor | Compute Balance | Used In |
|--------|-------------|--------------|-----------------|---------|
| `striding` | Standard Conv2D stack | 4x or 8x | All layers equal | Original Conformer |
| `dw_striding` | Depthwise-sep Conv2D | 4x or 8x | All layers equal | **FastConformer, Nemotron** |
| `vggnet` | VGG-style convolutions | 4x | All layers equal | Some Conformer variants |
| Progressive | Conv at multiple depths | 8x (2→4→8) | Early layers heavier | EfficientConformer |
| U-Net | Progressive + upsampling | 4x output | Middle layers light | Squeezeformer |

**FastConformer's design choice**: By performing all 8x downsampling at the input, all encoder layers operate on the same sequence length. This provides uniform compute distribution across layers, unlike progressive schemes where early layers process 4x longer sequences.

---

## 3. Mathematical Foundations

### 3.1 Output Length Calculations

For a convolutional layer with:
- Input length: L_in
- Kernel size: k
- Stride: s
- Padding: p (total, left + right)
- Ceil mode: false (floor division)

The output length is:

```
L_out = floor((L_in + p - k) / s) + 1
```

For the `dw_striding` subsampling with k=3, s=2:

**Non-causal mode** (symmetric padding, p = 2):
```
L_out = floor((L_in + 2 - 3) / 2) + 1 = floor((L_in - 1) / 2) + 1
```

**Causal mode** (asymmetric padding, p_left = 2, p_right = 1, total p = 3):
```
L_out = floor((L_in + 3 - 3) / 2) + 1 = floor(L_in / 2) + 1
```

#### Multi-Stage Length Propagation

For 3 stages of stride-2 convolutions (non-causal):

```
Stage 1: L₁ = floor((L₀ - 1) / 2) + 1
Stage 2: L₂ = floor((L₁ - 1) / 2) + 1
Stage 3: L₃ = floor((L₂ - 1) / 2) + 1
```

**Example**: Input of 100 mel frames (1 second of audio):
```
L₀ = 100
L₁ = floor((100 - 1) / 2) + 1 = floor(99/2) + 1 = 49 + 1 = 50
L₂ = floor((50 - 1) / 2) + 1 = floor(49/2) + 1 = 24 + 1 = 25
L₃ = floor((25 - 1) / 2) + 1 = floor(24/2) + 1 = 12 + 1 = 13
```

So 100 input frames → 13 output frames (approximately 100/8 = 12.5, rounded up due to padding).

#### Frequency Dimension

The same striding applies to the frequency dimension:
```
F₀ = 128 (mel bins)
F₁ = floor((128 - 1) / 2) + 1 = 64
F₂ = floor((64 - 1) / 2) + 1 = 32
F₃ = floor((32 - 1) / 2) + 1 = 16
```

The final linear projection input dimension is: 256 channels × 16 frequency bins = 4096.

> **Model Variant Note**: If `feat_in` differs (e.g., 80 mel bins), F₃ will differ. For 80 mel bins: 80 → 40 → 20 → 10, giving 256 × 10 = 2560 as the linear input dimension.

### 3.2 Temporal Resolution and Frame Alignment

#### Mel-Spectrogram to Time Mapping

For 16kHz audio with standard mel extraction parameters:
- Window size: 400 samples (25ms)
- Hop size: 160 samples (10ms)

Each mel frame m[t] represents audio centered at time:
```
time(t) = (t × hop_size + window_size/2) / sample_rate
        = (t × 160 + 200) / 16000
        = (t × 10 + 12.5) ms
```

The frame covers the interval:
```
[t × 10ms, t × 10ms + 25ms]
```

#### Encoder Frame to Time Mapping

After 8x subsampling, encoder frame e[t'] corresponds to mel frames:
```
mel_start(t') ≈ t' × 8
mel_end(t') ≈ t' × 8 + receptive_field - 1
```

Each encoder frame e[t'] represents approximately:
```
time_center(t') ≈ t' × 80ms + offset
duration ≈ 80ms per frame
```

The exact offset depends on the receptive field and padding mode.

### 3.3 Receptive Field Analysis

The receptive field determines how many input mel frames contribute to each output encoder frame.

#### Per-Layer Receptive Field Growth

For a single conv layer with kernel k and stride s, if the input receptive field is R_in:
```
R_out = R_in + (k - 1) × s^(layer_depth - 1)
```

More directly, for our architecture:

| Layer | Kernel | Stride | Receptive Field (in mel frames) |
|-------|--------|--------|--------------------------------|
| Input | - | - | 1 |
| Stage 1 | 3 | 2 | 3 |
| Stage 2 | 3 | 2 | 3 + 2×2 = 7 |
| Stage 3 | 3 | 2 | 7 + 2×4 = 15 |

**Total receptive field: 15 mel frames = 150ms of audio**

#### Derivation

The receptive field R after n layers of stride-s, kernel-k convolutions:

```
R = k + (k-1) × (s + s² + ... + s^(n-1))
  = k + (k-1) × (s^n - s) / (s - 1)
```

For k=3, s=2, n=3:
```
R = 3 + 2 × (8 - 2) / 1 = 3 + 12 = 15
```

This means each encoder output frame "sees" 15 consecutive mel frames (150ms of audio), centered (in non-causal mode) or left-aligned (in causal mode) relative to its temporal position.

---

## 4. Causal Mode and Streaming Inference

### 4.1 Causal Padding Mechanics

In streaming inference, we cannot access future audio frames. The subsampling layer must be **causal**: each output frame depends only on current and past input frames.

#### Standard (Non-Causal) Padding

For k=3, s=2: symmetric padding of 1 on each side.
```
Input:   [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
Padded:  [0, x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇, 0]
Output:  [y₀,     y₁,     y₂,     y₃]
              ↑       ↑       ↑       ↑
         uses    uses    uses    uses
         0,x₀,x₁ x₁,x₂,x₃ x₃,x₄,x₅ x₅,x₆,x₇
```

Here, y₁ depends on x₃, which is "future" relative to the center of the receptive field.

#### Causal Padding

For k=3, s=2: left padding of k-1=2, right padding of s-1=1.
```
Input:   [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
Padded:  [0, 0, x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇, 0]
Output:  [y₀,     y₁,     y₂,     y₃,     y₄]
              ↑       ↑       ↑       ↑       ↑
         uses    uses    uses    uses    uses
         0,0,x₀  x₀,x₁,x₂ x₂,x₃,x₄ x₄,x₅,x₆ x₆,x₇,0
```

**Key insight**: The right padding of s-1=1 ensures proper alignment at chunk boundaries during streaming. Without it, the last valid output would require one future frame.

#### CausalConv2D Implementation

```python
class CausalConv2D(nn.Conv2d):
    """
    2D convolution where each output location depends only on 
    current and past input locations (in both time and frequency).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        # No padding in base Conv2d - we handle it manually
        super().__init__(in_channels, out_channels, kernel_size, 
                        stride=stride, padding=0, groups=groups)
        
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        s = stride if isinstance(stride, int) else stride[0]
        
        self._left_padding = k - 1   # = 2 for k=3
        self._right_padding = s - 1  # = 1 for s=2
    
    def forward(self, x):
        # x shape: [B, C, T, F]
        # Pad format: (left_F, right_F, left_T, right_T)
        x = F.pad(x, (self._left_padding, self._right_padding,   # frequency
                      self._left_padding, self._right_padding))  # time
        return super().forward(x)
```

### 4.2 Cache Requirements Derivation

Unlike the encoder's self-attention and convolution layers, the subsampling module **does not maintain activation caches**. Instead, it requires **mel frame prepending**.

#### Why Not Cache Activations?

1. **Small kernel size**: With k=3, only 2 frames of context are needed per layer
2. **Striding discards frames**: Cached intermediate activations would be misaligned after striding
3. **Recomputation is cheap**: The subsampling module is computationally lightweight compared to encoder layers

#### Derivation of Required Mel Frame Overlap

From the Stateful Conformer paper (arXiv:2312.17279):

> "The downsampling module uses striding convolutions and can also benefit from caching. However, due to a small kernel size (typically 3), it would be a small cache and can be ignored. Instead, we can simply concatenate the last `log(Dr)*2+1` mel-spectrogram feature frames to each chunk where Dr is the downsampling rate."

For **Dr = 8** (subsampling_factor):
```
frames_to_prepend = log₂(Dr) × 2 + 1
                  = log₂(8) × 2 + 1
                  = 3 × 2 + 1
                  = 7 mel frames
```

#### Mathematical Justification

Consider what happens at chunk boundaries with stride-2 convolutions:

**Stage 1 (stride 2)**:
- To produce output at position 0 of chunk N, we need input positions [-2, -1, 0]
- Positions -2 and -1 come from the previous chunk
- Need: 2 frames from previous chunk

**Stage 2 (stride 2, operating on Stage 1 output)**:
- To produce output at position 0, we need Stage 1 outputs at [-2, -1, 0]
- Stage 1 position -1 requires input positions [-4, -3, -2]
- Stage 1 position -2 requires input positions [-6, -5, -4]
- Need: 4 additional input frames (total: 6)

**Stage 3 (stride 2, operating on Stage 2 output)**:
- Similar analysis adds 2 more at the input level due to effective stride of 4
- But the asymmetric causal padding (right_pad = s-1 = 1) adds 1 more

**Total**: 2 + 2×2 + 1 = 7 frames, matching the formula.

> **Model Variant Note**: For 4x subsampling (2 stages), the formula gives log₂(4)×2+1 = 5 frames.

### 4.3 Streaming Inference Algorithm

```python
class StreamingSubsampler:
    def __init__(self, subsampling_factor=8):
        self.subsampling_factor = subsampling_factor
        self.cache_size = int(np.log2(subsampling_factor)) * 2 + 1  # = 7
        self.mel_cache = None  # Shape: [B, 1, cache_size, F]
        
    def reset(self):
        """Call at the start of each new utterance."""
        self.mel_cache = None
    
    def process_chunk(self, mel_chunk, subsampler_module):
        """
        Process a chunk of mel frames through the subsampler.
        
        Args:
            mel_chunk: [B, 1, T_chunk, F] - New mel frames
            subsampler_module: The ConvSubsampling module
            
        Returns:
            encoder_input: [B, T_out, D] - Subsampled features
        """
        B, _, T_chunk, F = mel_chunk.shape
        
        if self.mel_cache is None:
            # First chunk: pad with zeros
            self.mel_cache = torch.zeros(B, 1, self.cache_size, F, 
                                         device=mel_chunk.device)
        
        # Prepend cached frames
        mel_input = torch.cat([self.mel_cache, mel_chunk], dim=2)
        # mel_input shape: [B, 1, cache_size + T_chunk, F]
        
        # Update cache with last `cache_size` frames from input
        self.mel_cache = mel_input[:, :, -self.cache_size:, :].clone()
        
        # Run through subsampler
        encoder_input, lengths = subsampler_module(mel_input)
        # encoder_input shape: [B, (cache_size + T_chunk) // 8, D]
        
        # Calculate how many output frames came from the cache
        # For causal mode: cache_size // subsampling_factor (approximately)
        # The first frame is partially from cache, so we keep it
        # but subsequent chunks need frame alignment
        
        # In practice, NeMo handles this via drop_extra_pre_encoded
        cache_output_frames = self.cache_size // self.subsampling_factor  # = 0
        
        return encoder_input[:, cache_output_frames:, :]
```

### 4.4 Chunk Boundary Handling

A critical challenge in streaming is ensuring that output frames align correctly across chunk boundaries.

#### The Alignment Problem

With stride-2 convolutions, the output grid is determined by the cumulative input position:
```
output_index = floor(input_index / stride)
```

If chunks don't align to stride boundaries, output frames will shift:

**Chunk A** (indices 0-15):
```
Outputs at: 0, 2, 4, 6, 8, 10, 12, 14 → output indices 0-7
```

**Chunk B** (indices 16-31):
```
Outputs at: 16, 18, 20, 22, 24, 26, 28, 30 → output indices 8-15
```

This works because 16 is a multiple of 2.

**Problem case** - Chunk B starts at index 17:
```
Outputs at: 17(invalid), 18, 20, ... 
```

#### Solution: Chunk Size Constraints

For correct streaming with subsampling_factor = 8:
- Chunk sizes (in mel frames) should be multiples of 8
- After initial cache warmup, each chunk produces exactly `chunk_size / 8` encoder frames

The Nemotron model uses chunk sizes defined in encoder frames (not mel frames):
```yaml
att_context_size:
  - [70, 13]   # chunk = 14 encoder frames = 112 mel frames
  - [70, 6]    # chunk = 7 encoder frames = 56 mel frames
  - [70, 1]    # chunk = 2 encoder frames = 16 mel frames
  - [70, 0]    # chunk = 1 encoder frame = 8 mel frames
```

These are all multiples of 8 mel frames, ensuring alignment.

#### Frame Drop Mechanism

NeMo's streaming implementation uses `drop_extra_pre_encoded` to handle the overlap:

```python
# In streaming config setup
streaming_cfg.drop_extra_pre_encoded = (
    streaming_cfg.pre_encode_cache_size[1] // self.subsampling_factor
)
```

For cache_size = 7 and subsampling_factor = 8:
```
drop_extra_pre_encoded = 7 // 8 = 0
```

This means the cached mel frames produce 0 "extra" encoder frames that need to be discarded (the cache is just enough to provide context without generating duplicate outputs).

---

## 5. Implementation Details

### 5.1 Configuration Parameters

Complete configuration for Nemotron Speech Streaming encoder:

```yaml
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  
  # Subsampling configuration
  feat_in: 128                    # Mel-spectrogram bins
  feat_out: -1                    # -1 means use d_model
  subsampling: dw_striding        # Depthwise separable striding
  subsampling_factor: 8           # 8x temporal reduction
  subsampling_conv_channels: 256  # Intermediate channels
  causal_downsampling: true       # Enable streaming mode
  
  # Encoder configuration
  n_layers: 24
  d_model: 1024
  
  # Attention configuration (for context, not subsampling)
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
    - [70, 13]  # Layers 0-5:  1.12s chunks
    - [70, 6]   # Layers 6-11: 0.56s chunks  
    - [70, 1]   # Layers 12-17: 0.16s chunks
    - [70, 0]   # Layers 18-23: 0.08s chunks
  att_context_style: chunked_limited
  
  # Convolution configuration (encoder blocks, not subsampling)
  conv_kernel_size: 9
  conv_norm_type: layer_norm
  conv_context_size: causal       # Causal encoder convolutions
  
  # Other
  use_bias: false
  ff_expansion_factor: 4
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_att: 0.1
```

> **Model Variant Note**: 
> - Smaller models (e.g., 120M params) may use `d_model: 512`, `n_layers: 17`, `subsampling_conv_channels: 256`
> - Larger models (e.g., 1B params) may use `d_model: 1024`, `n_layers: 42`, `subsampling_conv_channels: 512`
> - 4x subsampling models use `subsampling_factor: 4` (2 conv stages instead of 3)

### 5.2 Weight Initialization

The `dw_striding` subsampling uses specific initialization for stable training:

```python
def reset_parameters(self):
    """
    Initialize weights for dw_striding subsampling.
    Uses uniform initialization scaled by layer dimensions.
    """
    with torch.no_grad():
        k = self._kernel_size  # 3
        C = self._conv_channels  # 256
        
        # Scaling factors
        scale = 1.0 / k                    # = 0.333
        dw_max = (k ** 2) ** -0.5          # = 1/9 ≈ 0.111
        pw_max = C ** -0.5                  # = 1/16 = 0.0625
        
        # First conv layer (standard conv)
        # Layer index 0 in self.conv Sequential
        nn.init.uniform_(self.conv[0].weight, -scale, scale)
        nn.init.uniform_(self.conv[0].bias, -scale, scale)
        
        # Depthwise + Pointwise layers
        # Structure: [Conv, ReLU, DWConv, PWConv, ReLU, DWConv, PWConv, ReLU]
        # Indices:   [0,    1,    2,      3,      4,    5,      6,      7   ]
        for idx in range(2, len(self.conv), 3):  # idx = 2, 5
            # Depthwise conv
            nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
            nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
            
            # Pointwise conv
            nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
            nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)
        
        # Final linear projection
        # Scaled by output and effective input dimensions
        fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
        nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
        nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)
```

### 5.3 Integration with Encoder Caching

The subsampling module provides cache size information to the encoder's streaming configuration:

```python
class ConvSubsampling(nn.Module):
    def get_streaming_cache_size(self):
        """
        Returns cache sizes for streaming.
        Format: [left_cache, right_cache]
        
        For causal subsampling, we need to prepend mel frames,
        not cache intermediate activations.
        """
        if self.is_causal:
            # right_cache = frames needed from previous chunk
            return [0, self.subsampling_factor + 1]  # [0, 9] for 8x
        else:
            return [0, 0]
    
    def get_sampling_frames(self):
        """
        Returns [output_frames_per_input, input_frames_per_output].
        Used for chunk size calculations.
        """
        return [1, self.subsampling_factor]  # [1, 8]
```

The encoder uses this to calculate streaming parameters:

```python
class ConformerEncoder(nn.Module):
    def setup_streaming_params(self, ...):
        # Get subsampler's cache requirements
        pre_encode_cache = self.pre_encode.get_streaming_cache_size()
        # pre_encode_cache = [0, 9] for 8x causal subsampling
        
        # Calculate frames to drop after subsampling
        # (frames that came from the cache and shouldn't be re-processed)
        drop_extra = pre_encode_cache[1] // self.subsampling_factor
        # drop_extra = 9 // 8 = 1
        
        streaming_cfg.pre_encode_cache_size = pre_encode_cache
        streaming_cfg.drop_extra_pre_encoded = drop_extra
```

---

## 6. Practical Considerations

### 6.1 Memory and Compute Trade-offs

#### Compute Cost Breakdown (per forward pass)

For a 30-second audio sample (3000 mel frames → 375 encoder frames):

| Component | Operations | % of Total |
|-----------|-----------|------------|
| Subsampling (3 conv stages + linear) | ~50 GMACS | ~2% |
| Encoder (24 layers × attention + conv + FFN) | ~2400 GMACS | ~98% |

The subsampling module is **not the bottleneck**. This justifies using mel frame prepending (with recomputation) rather than complex activation caching.

#### Memory Footprint

| Component | Memory (fp16, batch=1) |
|-----------|----------------------|
| Mel frame cache (7 × 128) | ~1.8 KB |
| Subsampling activations | ~2 MB |
| Encoder attention cache (24 layers × 70 frames × 1024) | ~3.4 MB |
| Encoder conv cache (24 layers × 8 frames × 1024) | ~0.4 MB |

The mel frame cache is negligible compared to encoder caches.

### 6.2 First Chunk Handling

The first chunk of audio has no previous frames to prepend. Two approaches:

#### Zero Padding (Default)

```python
if self.mel_cache is None:
    self.mel_cache = torch.zeros(B, 1, self.cache_size, F, device=device)
```

**Effect**: First ~7 frames of output have reduced accuracy due to missing left context.

#### Warmup Period

Some applications discard the first N milliseconds of transcription or use a "warmup" buffer:

```python
# Accumulate minimum audio before starting inference
MIN_AUDIO_MS = 200  # 200ms warmup
min_mel_frames = MIN_AUDIO_MS // 10  # 20 frames

if accumulated_frames < min_mel_frames:
    buffer.append(mel_chunk)
    return None  # No output yet
else:
    # Process with full context
    ...
```

### 6.3 Variable Length Sequences

The subsampling module handles variable-length sequences through length tracking:

```python
def forward(self, x, lengths):
    """
    Args:
        x: [B, C, T, F] - Input mel-spectrogram (C=1)
        lengths: [B] - Valid length of each sequence in batch
        
    Returns:
        out: [B, T', D] - Subsampled features
        out_lengths: [B] - Valid lengths after subsampling
    """
    # Process through conv layers
    x = self.conv(x)  # [B, C', T', F']
    
    # Calculate output lengths
    out_lengths = self.calc_output_lengths(lengths)
    
    # Reshape: [B, C', T', F'] → [B, T', C' * F']
    b, c, t, f = x.shape
    x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
    
    # Project to d_model
    x = self.out(x)  # [B, T', D]
    
    return x, out_lengths

def calc_output_lengths(self, lengths):
    """Calculate output lengths after subsampling."""
    for _ in range(self._sampling_num):  # 3 stages
        lengths = torch.div(
            lengths + self._left_padding + self._right_padding - self._kernel_size,
            self._stride,
            rounding_mode='floor'
        ) + 1
    return lengths.to(torch.int)
```

---

## 7. References

### Papers

1. **Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition**
   - Rekesh et al., ASRU 2023
   - arXiv:2305.05084
   - Introduces 8x downsampling with depthwise separable convolutions

2. **Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition**
   - Noroozi et al., ICASSP 2024
   - arXiv:2312.17279
   - Details causal subsampling and cache requirements for streaming

3. **Conformer: Convolution-augmented Transformer for Speech Recognition**
   - Gulati et al., Interspeech 2020
   - Original Conformer architecture with 4x subsampling

4. **Efficient Conformer: Progressive Downsampling and Grouped Attention**
   - Burchi & Vielzeuf, ASRU 2021
   - Alternative progressive downsampling approach

### Code References

- **NeMo ConvSubsampling**: `nemo/collections/asr/parts/submodules/subsampling.py`
- **NeMo ConformerEncoder**: `nemo/collections/asr/modules/conformer_encoder.py`
- **Streaming Inference**: `nemo/collections/asr/parts/submodules/conformer_modules.py`

### Models

- **Nemotron Speech Streaming 0.6B**: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- **FastConformer Hybrid Large**: [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)

---

## Appendix A: Quick Reference

### Subsampling Formulas

| Formula | Value (Nemotron) |
|---------|------------------|
| Number of conv stages | log₂(subsampling_factor) = 3 |
| Mel frames to cache | log₂(factor) × 2 + 1 = 7 |
| Output frames per chunk | chunk_mel_frames / factor |
| Receptive field | k + (k-1)(s^n - s)/(s-1) = 15 frames |
| Causal left padding | k - 1 = 2 |
| Causal right padding | s - 1 = 1 |

### Streaming Chunk Sizes (Nemotron)

| Latency Mode | Right Context | Encoder Frames | Mel Frames | Audio Duration |
|--------------|---------------|----------------|------------|----------------|
| 1.12s | 13 | 14 | 112 | 1.12s |
| 0.56s | 6 | 7 | 56 | 0.56s |
| 0.16s | 1 | 2 | 16 | 0.16s |
| 0.08s | 0 | 1 | 8 | 0.08s |

### Parameter Variations by Model Size

| Model | d_model | n_layers | subsampling_channels | subsampling_factor |
|-------|---------|----------|---------------------|-------------------|
| Small (~30M) | 256 | 12 | 128 | 4 or 8 |
| Medium (~120M) | 512 | 17 | 256 | 8 |
| Large (~600M) | 1024 | 24 | 256 | 8 |
| XL (~1B) | 1024 | 42 | 512 | 8 |
