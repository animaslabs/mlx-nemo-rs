```
(pre_encode): ConvSubsampling(
    (conv): Sequential(
        (0): CausalConv2D(1, 256, kernel_size=(3, 3), stride=(2, 2))
        (1): ReLU(inplace=True)
        (2): CausalConv2D(256, 256, kernel_size=(3, 3), stride=(2, 2), groups=256)  # depthwise
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))                    # pointwise
        (4): ReLU(inplace=True)
        (5): CausalConv2D(256, 256, kernel_size=(3, 3), stride=(2, 2), groups=256)  # depthwise
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))                    # pointwise
        (7): ReLU(inplace=True)
    )
    (out): Linear(in_features=???, out_features=1024, bias=True)
)
```

**Key observations:**
- 3 strided CausalConv2D layers, each with stride=(2,2)
- Total time-axis downsampling: 2 × 2 × 2 = **8x**
- The stride applies to **both** time and frequency axes
- Layers 2-3 and 5-6 form depthwise separable convolutions (depthwise + pointwise 1×1)

### CausalConv2D Padding

For NeMo's CausalConv2D with kernel_size=3, stride=2:
```python
left_pad = kernel_size - 1 = 2   # for causality
right_pad = stride - 1 = 1       # for stride alignment
```
The **same padding is applied to both axes** (time and frequency):
- **Left padding (time axis):** 2
- **Right padding (time axis):** 1
- **Top padding (freq axis):** 2
- **Bottom padding (freq axis):** 1

In PyTorch F.pad order: `(left=2, right=1, top=2, bottom=1)`

Total padding per axis: 3 (left+right = top+bottom = 2+1 = 3)

### Let's Trace Through with Concrete Numbers

**Scenario:** Process a 560ms audio chunk for the [70, 6] configuration

#### Step 0: Audio to Mel Spectrogram
```
Audio: 560ms at 16kHz = 8,960 samples
Window: 25ms (400 samples)
Stride: 10ms (160 samples)

Number of mel frames = floor((8960 - 400) / 160) + 1 = floor(8560/160) + 1 = 53 + 1 = 54 frames

But wait - for streaming we typically compute:
mel_frames ≈ audio_ms / stride_ms = 560 / 10 = 56 frames
```

Let's use **56 mel frames** for this example.

**Input tensor shape:** `[B, T, F] = [B, 56, 128]`

For Conv2D, we need to reshape to `[B, C, F, T] = [B, 1, 128, 56]`
(1 channel, 128 frequency bins, 56 time steps)

---

#### Layer 0: CausalConv2D(1→256, K=3×3, S=2×2)

**Input:** `[B, 1, 128, 56]` (channels, freq, time)

**Padding (causal):**
- Top (freq): 2
- Bottom (freq): 1
- Left (time): 2
- Right (time): 1

**After padding:** `[B, 1, 131, 59]`

**Output size formula:**
```
out_size = floor((in_size + total_padding - kernel_size) / stride) + 1
         = floor((in_size + 3 - 3) / 2) + 1
         = floor(in_size / 2) + 1
```

**Frequency dimension:**
```
out_F = floor(128 / 2) + 1 = 64 + 1 = 65
```

**Time dimension:**
```
out_T = floor(56 / 2) + 1 = 28 + 1 = 29
```

**Output after Layer 0:** `[B, 256, 65, 29]`

---

#### Layer 1: ReLU

**Output:** `[B, 256, 65, 29]` (unchanged shape)

---

#### Layer 2: CausalConv2D(256→256, K=3×3, S=2×2, groups=256) - Depthwise

**Input:** `[B, 256, 65, 29]`

**Padding:** Same as Layer 0 (top=2, bottom=1, left=2, right=1)

**Frequency dimension:**
```
out_F = floor(65 / 2) + 1 = 32 + 1 = 33
```

**Time dimension:**
```
out_T = floor(29 / 2) + 1 = 14 + 1 = 15
```

**Output after Layer 2:** `[B, 256, 33, 15]`

---

#### Layer 3: Conv2d(256→256, K=1×1, S=1×1) - Pointwise

**Input:** `[B, 256, 33, 15]`

No padding needed for 1×1 convolution.

**Output after Layer 3:** `[B, 256, 33, 15]` (unchanged shape, mixes channels)

---

#### Layer 4: ReLU

**Output:** `[B, 256, 33, 15]`

---

#### Layer 5: CausalConv2D(256→256, K=3×3, S=2×2, groups=256) - Depthwise

**Input:** `[B, 256, 33, 15]`

**Padding:** Same as Layer 0 (top=2, bottom=1, left=2, right=1)

**Frequency dimension:**
```
out_F = floor(33 / 2) + 1 = 16 + 1 = 17
```

**Time dimension:**
```
out_T = floor(15 / 2) + 1 = 7 + 1 = 8
```

**Output after Layer 5:** `[B, 256, 17, 8]`

---

#### Layer 6: Conv2d(256→256, K=1×1, S=1×1) - Pointwise

**Output:** `[B, 256, 17, 8]`

---

#### Layer 7: ReLU

**Output:** `[B, 256, 17, 8]`

---

#### Reshape and Linear Projection

Now we flatten the frequency dimension into features and project to d_model:

**Reshape:** `[B, 256, 17, 8]` → `[B, 8, 256 × 17]` = `[B, 8, 4352]`

**Linear projection:** `Linear(4352 → 1024)`

**Final output:** `[B, 8, 1024]`

---

### Summary Table

| Stage | Shape [B, C, F, T] | Time Steps | Freq Dim | Time Resolution |
|-------|-------------------|------------|----------|-----------------|
| Input mel | [B, 1, 128, 56] | 56 | 128 | 10ms |
| After Conv0 (S=2) | [B, 256, 65, 29] | 29 | 65 | 20ms |
| After Conv2 (S=2) | [B, 256, 33, 15] | 15 | 33 | 40ms |
| After Conv5 (S=2) | [B, 256, 17, 8] | 8 | 17 | 80ms |
| After Linear | [B, 8, 1024] | 8 | - | 80ms |

**Total downsampling:** 56 → 8 ≈ **7x** (not exactly 8x due to padding)

**Linear layer input features:** 256 × 17 = **4352**

---

### Verifying with att_context_size = [70, 6]

For the [70, 6] configuration:
- **Chunk size** = 1 (current) + 6 (lookahead) = **7 encoder tokens**
- **Mel frames needed** ≈ 7 × 8 = **56 mel frames**
- **Audio duration** = 56 × 10ms = **560ms**

Note: Due to the padding formula `floor(in/2) + 1`, the actual output is 8 frames
from 56 input frames. The subsampling is slightly less than 8x.

---

### What Happens at the Boundaries (Causal Padding Detail)

Let me show exactly what the causal padding looks like for the time axis:
```
Input time axis (56 mel frames):
[m0][m1][m2][m3][m4][m5]...[m53][m54][m55]

After causal padding (left=2, right=1):
[0][0][m0][m1][m2][m3][m4][m5]...[m53][m54][m55][0]
 ↑  ↑                                          ↑
 left padding                              right padding

Convolution with kernel=3, stride=2:

Position 0: kernel sees [0, 0, m0]      → output o0 (aligned with m0)
Position 2: kernel sees [m0, m1, m2]    → output o1 (aligned with m2)
Position 4: kernel sees [m2, m3, m4]    → output o2 (aligned with m4)
...
Position 54: kernel sees [m52, m53, m54] → output o27 (aligned with m54)
Position 56: kernel sees [m54, m55, 0]  → output o28 (with right padding)

So we get 29 outputs from 56 inputs: floor(56/2) + 1 = 29
