# mlx-voice-rs vs mlx-audio: benchmark gap analysis

Findings from the `verify` benchmark run (mlx-models `bench/results/verify`), and what was changed as a result (see [Results](#results-after-the-changes) first).

| benchmark | mlx-voice-rs | mlx-audio | gap |
|---|---|---|---|
| parakeet-rnnt offline, 870 s audio | 8.93 s | 6.58 s | rs is 36% slower |
| nemotron streaming mode 0, 4591 s audio | 94.8 s, 22.5 ms/chunk | 86.0 s, 20.7 ms/chunk | rs is 10% slower |
| nemotron streaming WER (all) | 7.60% | 7.33% (NeMo 7.35%) | rs differs from NeMo |

## Results after the changes

Same clips, same machine, measured with `examples/eval` (times exclude model load, like the bench):

| benchmark | before | after | mlx-audio | NeMo |
|---|---|---|---|---|
| parakeet-rnnt offline, 870 s audio | 8.93 s | **2.79 s** (2.18 s with `--dtype bfloat16`) | 6.58 s | 10.66 s |
| parakeet-rnnt offline WER (all) | 5.80% | **6.71%** (bf16: 6.75%) | 6.83% | 6.75% |
| nemotron streaming, 4591 s audio | 97 s, ~23 ms/chunk | **65 s, ~15.5 ms/chunk** | 86.0 s, 20.7 ms/chunk | 150.9 s |
| nemotron streaming WER (all) | 7.60% | **7.34%** | 7.33% | 7.35% |

Both paths now track NeMo's transcripts as closely as mlx-audio does (word-level diff
opcodes against NeMo per clip: offline 0/1/0, streaming 0/3/0/13/1/0). The offline WER
went *up* from 5.80% to 6.71%. The old number was not better accuracy: the crimetown
clip's 18.83% came from a broken encoder (see the BatchNorm finding below) and
didn't match NeMo's 25.43%.

### What changed

- **Streaming chunking follows NeMo** (§2a–2c): first chunk is 105 frames with no pre-encode cache and no frame drop, later chunks 9 cache + 112 new frames with 2 dropped, the last chunk is shorter instead of padded, and the silence flush is gone. Encoder lengths are shortened by the dropped frames and the cache fill level grows by the frames actually kept, as in NeMo. The multitalker pipeline and the legacy `EncDecRNNTModel` streaming session use the same rules.
- **Mel frames are centred like NeMo's STFT** (§2b): `StreamingMel` starts with `win/2` zeros and flushes with `win/2` samples that are zero after preemphasis, so streaming reproduces the whole-file mel (new unit test). The batch `MelProcessor` reports NeMo's `samples / hop` valid frames, zeroes the extra frame, and normalises per feature over valid frames with NeMo's unbiased std + 1e-5.
- **Subsampling length fix**: the causal length formula was missing NeMo's `+1`, so it reported one frame fewer than the tensor held.
- **Decode loop** (§1a): one shared windowed greedy decoder (`RnntStreamingDecoder::decode_projected`) for the offline model, the streaming pipeline, the multitalker pipeline and the legacy session. Encoder frames are projected once, the prediction network only runs after a token, and the joint scores up to 32 frames per GPU round trip. Blank threshold and punctuation masking are folded into the same on-GPU argmax. The LSTM cell is one fused matmul with pre-summed biases. Tokens are identical to the old loop.
- **Per-chunk overhead** (§3.1–3.2, 1d): debug readbacks removed; lengths and cache fill levels are host arrays, so reading them never waits on the GPU; attention masks are memoized and dropped once nothing is hidden; streaming text is detokenized incrementally (only the last word is re-decoded); `std::env::var` debug switches are read once.
- **Fused attention** (§1c, 3.4): offline and streaming attention use `fast::scaled_dot_product_attention` with the position scores as the additive mask; offline query blocks use `async_eval` instead of draining the GPU; streaming caches the projected position embeddings per layer.
- **Opt-in bf16 encoder** (§1b): `EncDecRNNTModel::set_encoder_dtype(Dtype::Bfloat16)` and `examples/eval --offline --dtype bfloat16`.
- **Dither off** for the offline RNN-T, CTC and SortFormer models (§1e).
- **Bench label** (mlx-models `bench/runners/mlx_audio.py`): reports the dtype the model computes in (float32 for the mlx-audio Parakeet run) and records the input dtype separately.

### Corrections to the original findings

- **The offline divergence from NeMo was not dither.** Turning dither off changed no output. The cause was that mlx-rs `nn::BatchNorm` defaults to training mode, and the conformer conv modules never switched it off. Offline, every conv module normalised with the clip's own statistics instead of the checkpoint's running mean/var (and overwrote the running stats on each call). A long clip's statistics are close to the trained ones, which is why WER only looked a little off. The encoder output correlated 0.89 with mlx-audio's on identical mel input; with inference-mode BatchNorm it's 0.99999. Streaming models were not affected (they use LayerNorm in the conv module). The SortFormer diarizer's encoder uses BatchNorm convs, so diarization should improve too.
- **Per-layer cache `Vec` (§3.3) was not done.** It would only save one ~7 MB `stack` copy per chunk (tens of µs against ~15 ms per chunk) and would change `EncoderCache`'s public layout across six files.

## TL;DR (original findings)

1. **The offline speed gap is entirely in the greedy RNN-T decode loop, not the encoder.** We do 2 GPU syncs per decode step and mlx-audio does 1. We also re-run the LSTM on every blank frame. A scratch prototype of a windowed greedy decode produced identical tokens and cut decode time from 2.65 s to 0.58 s on a 345 s clip. That would make mlx-voice-rs about 2× faster than mlx-audio offline.
2. **The streaming WER gap is a correctness bug in how we build the first chunk.** It shifts the 8× subsampling grid by 7 mel frames for the whole file. Running mlx-audio's model with our chunking reproduces our WER (7.73%), and running it with NeMo's chunking reproduces NeMo's (7.33%). The encoder layers, caches and decoder already match NeMo.
3. **The streaming speed gap is per-chunk overhead.** It comes from debug code that runs even when logging is off, host syncs, cache re-stacking, unfused attention, and the same decode-loop issue as offline.
4. **The "bfloat16" label on mlx-audio is misleading.** `mlx_audio.stt.load()` keeps the weights in f32. Only the audio and mel are bf16, and the first conv promotes everything back to f32. Both libraries effectively ran in f32, so dtype is not why mlx-audio is faster. Real bf16 weights would cut encoder time from about 0.78 s to 0.44 s per 345 s clip for either library.

---

## 1. Offline parakeet-rnnt speed

### Where the time goes (ted_gary_flake, 345 s, T' = 4316 encoder frames)

| stage | mlx-voice-rs | mlx-audio |
|---|---|---|
| mel | 6 ms | (included below) |
| mel + encoder | 0.75–0.83 s | 0.79 s |
| greedy decode | **2.8 s** | **1.95 s** |
| total | 3.58 s | 2.74 s |

The decode loop takes about 75% of wall time in both libraries.

### 1a. Decode loop (high impact: the whole gap and more)

At `src/asr/models/enc_dec_rnnt.rs:712-728`, each step does the following:

- runs `predict_one` (2-layer LSTM), even when the previous step emitted blank and the state didn't change;
- computes the joint and `log_softmax`, then calls `log_probs.eval()`, which is sync #1;
- calls `argmax`, `prediction.eval()` and `to_vec`, which is sync #2.

That is 2 command-buffer round trips across about 6000 steps per clip. mlx-audio (`parakeet.py:~681-701`) syncs once per step. It also re-runs the LSTM on blanks.

These decode variants were measured on the same encoder output with 1672 tokens. All of them produced output identical to the current code:

| variant | decode time |
|---|---|
| A: current (2 evals/step) | 2.61–2.70 s |
| B: 1 eval/step (argmax only), no `log_softmax` | 1.85 s (≈ mlx-audio) |
| C: B + cache prediction-net output across blanks | 1.31–1.41 s |
| D: C + joint over a window of W frames, first non-blank found on host | 0.60–0.75 s (W=8), 0.58 s (W=32) |

Variant D is exact greedy, including `max_symbols`. While the prediction output is fixed, it computes `joint(enc[t..t+W], pred)`, takes the argmax over the window in one sync, and jumps to the first non-blank frame. The number of syncs then scales with tokens + T/W instead of frames + tokens.

Smaller follow-ons:
- Precompute the encoder projection of the joint for all frames in one matmul.
- Pre-sum `bias_ih + bias_hh` and use `addmm` in the LSTM cell. We issue 5 kernels per gate computation (`rnnt.rs:596-600`) where MLX's `nn.LSTM` issues 2.
- Stop re-transposing weights on every call (`rnnt.rs:596-597`, `quantization.rs:101`).
- Hoist the per-step `std::env::var("DEBUG_JOINT")` call out of the loop (`rnnt.rs:983`).
- Optionally `mx.compile` the predict + joint step.

The same two-eval pattern exists in the streaming per-chunk decode (`enc_dec_rnnt.rs:1255-1270`, `rnnt_streaming.rs:203-218`), so this fix helps streaming too.

### 1b. bf16 encoder weights (medium impact, opt-in)

We create every weight in f32 and have no lower-precision path. With real bf16 weights, the encoder ran in 0.44 s instead of 0.78 s on the test clip. Make this an opt-in cast of the encoder weights at load time, keeping mel, LSTM state and joint in f32, and check WER before enabling it by default.

### 1c. Attention (low impact now, more after 1a/1b)

We compute `ac` and `bd` as separate matmuls, then do the rel-shift, softmax and a final matmul (`attention.rs:363-397`). mlx-audio uses `mx.fast.scaled_dot_product_attention(q_u, k, v, mask=matrix_bd)`, which fuses the `ac` / softmax / `·V` part. At f32 this makes no measurable difference today.

Two suggested changes:
- Use `fast::scaled_dot_product_attention` per query block, with `bd` as the additive mask. This keeps the memory bound from 288d892.
- Replace the per-block `context.eval()` with `async_eval`, or raise the block budget. At T'=4316 the current code drains the GPU 5 blocks × 24 layers = 120 times.

### 1d. Minor overheads

- `EncDecRNNTModel::forward` (`enc_dec_rnnt.rs:769-813`) evaluates the mel and encoder output and copies both to the host (about 18 MB each) to compute min/max/mean for a `tracing::debug!` call, even when debug logging is off. Gate it with `tracing::enabled!(Level::DEBUG)`.
- Subsampling evaluates once per 4096-frame chunk (`conformer.rs:1456-1487`). This costs very little.

### 1e. Offline correctness side finding: dither is on

`EncDecRNNTModel::new` (`enc_dec_rnnt.rs:259`) builds the preprocessor from the config as is, so `dither = 1e-5` stays on. Offline transcripts are therefore randomly perturbed and nondeterministic between runs, which accounts for some of the small differences from NeMo. The streaming pipeline (`cache_aware_rnnt.rs:274, 525`) and the classification model already set `dither = 0.0`. Do the same here.

### Checked and equivalent offline (not causes)

- **Chunking:** neither library chunks the audio. Both run whole-file attention.
- **Mel:** both compute the FFT on the GPU, in about 6 ms.
- **Subsampling:** both use dw_striding conv.
- **LayerNorm:** both use `fast::layer_norm`.
- **BatchNorm:** unfused on both sides.
- **`mx.compile`:** neither library uses it for the encoder or the RNN-T step.
- **max_symbols:** 10 on both sides.

### Projected offline result

Applying 1a gives about 0.8 s encoder + 0.6 s decode ≈ 1.45 s per 345 s clip, against 3.58 s now and 2.74 s for mlx-audio. Over the 870 s bench set that is about 3.5–3.7 s (RTF ≈ 0.004), against 6.58 s for mlx-audio. Adding bf16 would take it to roughly 1.1 s per clip.

---

## 2. Streaming nemotron WER

### What the transcript diffs look like

Word-level diff opcodes against NeMo:

| clip | mlx-audio | mlx-voice-rs |
|---|---|---|
| ted_gary_flake | 0–9 | 15 |
| earnings21_cumulus_media | 0–9 | 67 |
| earnings21_zagg | 0–9 | 94 |
| rev16_crimetown_trailer | 0–9 | 17 |
| rev16_the_daily_wildfires | 0–9 | 30 |

mlx-audio is within floating-point noise of NeMo; mlx-voice-rs is not. Our differences are spread evenly through each file rather than grouped at chunk boundaries. Most are punctuation and casing (for example "Covid"→"COVID", "quarter, as"→"quarter. As"), which the scorer normalizes away. That is why zagg has 94 diffs and no WER change. There are also some real word errors:

- cumulus: "improves sequentially"→"improvedi" and "podcasting"→"podcastercasting".
- cumulus: "has put us in a fantastic position"→"has." (6 words dropped).
- crimetown: "Swami Kilpatrick I saw" dropped.

This pattern of slightly perturbed acoustics throughout the file is what a systematic input misalignment would produce.

### Reproduction

The streaming agent re-ran mlx-audio's own model modules and changed only the chunking:

| chunking | WER all | cumulus | crimetown |
|---|---|---|---|
| NeMo grid | 7.33% | 5.14% | 16.63% |
| mlx-voice-rs grid + mel offset | 7.73% | 5.76% | 17.36% |
| actual mlx-voice-rs | 7.60% | 6.05% | 17.60% |

The NeMo-grid run reproduces NeMo's text almost word for word, so the transformer layers, cache updates and RNN-T decoding are not at fault.

### 2a. Root cause: first-chunk handling shifts the subsampling grid (main)

NeMo's `ConvSubsampling.get_sampling_frames()` returns `[1, 8]` and `get_streaming_cache_size()` returns `[0, 9]`. As a result, the first chunk differs from the rest:

| | first chunk | later chunks |
|---|---|---|
| mel frames | 105 real frames, no pre-encode cache | 9 cache + 112 new |
| `drop_extra_pre_encoded` | 0 | 2 |
| encoder frames out | 14 | 14 |

This puts encoder frame *j*'s newest mel frame at 8*j*, the same grid as offline training. mlx-audio reaches the same grid by a different route (`streaming.py:94-117`).

Our code uses the steady-state values for the first chunk too. The code comments cite the right NeMo values, but the code uses the second element for both:

- `src/asr/streaming/core.rs:644`: `sampling_frames = (subsampling_factor, subsampling_factor)`, where NeMo uses `(1, 8)`. So `chunk_size` and `shift_size` are `(112, 112)` instead of `(105, 112)`.
- `core.rs:671`: `pre_encode_cache = (9, 9)`, where NeMo uses `(0, 9)`.
- `core.rs:676`: `drop_extra = 2` is hardcoded, and `cache_aware_rnnt.rs:~1312` applies it to every chunk, including the first.
- `cache_aware_rnnt.rs:1083-1092`: the first chunk is built as 9 `LOG_MEL_ZERO` (−16.6) frames + 112 real frames, where NeMo relies on the conv's zero padding.

Net effect: our encoder frame *m* is centred on mel frame 8*m*+7, so every frame of the file is phase-shifted against what the model was trained on.

**Fix:**
- Derive the first-chunk values from NeMo's formulas: `chunk_size`/`shift_size` = (105, 112), `pre_encode_cache` = (0, 9), and `drop_extra` = 0 on the first chunk and `1 + (9−1)/8` = 2 after.
- Make `StreamingMelConfig` (`mel_spectrogram.rs:~816-823`) honour the first-chunk sizes; it currently uses only `.1`.
- Remove the silence prepend.

### 2b. Mel frame offset (minor)

`StreamingMel` starts with a 240-sample zero lookback (`mel_spectrogram.rs:~917`) and frames 400 samples zero-padded to 512 (`dsp.rs:~321-349`). This centres frame *t* at sample 160*t* − 40. NeMo and mlx-audio use a centred STFT (n_fft/2 = 256 padding), which centres it at 160*t*.

**Fix:** start with 256 lookback zeros and 512-sample windows, or equivalently 200 zeros and 400-sample windows. On its own this moved the text by only 1–5 opcodes, but it should match.

### 2c. End-of-file handling (minor)

- `try_extract_chunk` pads the last chunk but always reports `valid_size = chunk_size` (`cache_aware_rnnt.rs:~1150`). NeMo passes the real remaining length.
- `flush_decode_with_silence` (`:~1226`, called at `:~1525`) re-runs a silence chunk against the encoder cache from *before* the last chunk, even though the decoder has already consumed that chunk. This produces trailing junk: "grid."→"grid in" and "app"→"apply".

**Fix:** pass the true last-chunk length and drop the silence flush, matching NeMo.

### Checked and matching NeMo

- chunked_limited attention mask: 70/14 = 5 left chunks, with the unfilled-cache offset
- attention cache update: `concat(cache[14:], x[:14])`
- conv cache: k−1 = 8
- rel-pos slice and `rel_shift`
- causal dw-striding padding
- Slaney mel and symmetric Hann window
- log guard 2⁻²⁴, normalize NA, dither 0
- `max_symbols` = 10
- LSTM state carried across chunks

---

## 3. Streaming nemotron speed (~2 ms/chunk over ~4100 chunks)

These are ranked by estimated impact from reading the code; none of them have been profiled individually.

1. **Debug work runs every chunk even with logging off.**
   - `cache_aware_rnnt.rs:~1367-1395` evaluates the next cache and copies `cache_channel_next` (24×70×1024 f32, about 6.9 MB) to a host `Vec` just to compute a norm for a `debug!` call.
   - `conformer.rs:~989-1040` evaluates the attention mask, copies it to the host and builds an 84-column ASCII grid string on every call.

   Remove both or gate them behind `tracing::enabled!`. This is probably worth about 1 ms/chunk.
2. **Extra host syncs.** These run on every chunk:
   - `lengths.eval()` and `cache_last_channel_len.eval()` (`conformer.rs:~926, 931`)
   - `cache_channel_len_next.eval()` + `to_vec` (`:~1162`)

   Track the filled cache length as a host integer instead. Build the chunked mask once per fill level and reuse it; once the cache is full, the mask is constant.
3. **Cache copies.** Caches are stored as stacked `[L, …]` tensors: each chunk indexes them per layer and then `ops::stack`s all 24 layers back together (`conformer.rs:~1134, 1141`), copying about 7 MB plus zero-pad concats. mlx-audio keeps a list per layer. Use a `Vec<Array>` per layer instead.
4. **Unfused attention** (`attention.rs:~590-670`). We run separate matmuls, two `select` masks and a softmax. mlx-audio calls `mx.fast.scaled_dot_product_attention` with the `bd` term as an additive mask. Also, `linear_pos(pos_emb)` is recomputed for every layer on every chunk even though the length (84) is constant; precompute it once per layer.
5. **Decode loop** (`rnnt_streaming.rs:203-218`). This is the same issue as §1a: the LSTM runs on every frame and the encoder frame is re-projected inside the joint. Cache the prediction output, project the chunk's encoder frames once, and sync once per symbol (or use the windowed variant D).
6. **Smaller items:**
   - The full transcript is detokenized on every step (`cache_aware_rnnt.rs:~1550`), which is O(N²) over a file.
   - `StreamingMel` transposes on the CPU in loops and syncs per mel chunk.

dtype and `mx.compile` are not factors here: both libraries run f32 and neither compiles.

---

## 4. Suggested order of work

| # | change | area | expected effect |
|---|---|---|---|
| 1 | Fix first-chunk grid (2a), mel offset (2b) and EOF handling (2c) | streaming correctness | WER 7.60% → ≈7.33% (match NeMo) |
| 2 | Zero dither in `EncDecRNNTModel` (1e) | offline correctness | deterministic output, closer to NeMo |
| 3 | Decode loop: 1 sync per step, cache prediction across blanks, windowed joint (1a, 3.5) | offline + streaming speed | offline decode ~4.5× faster; rs ≈ 2× faster than mlx-audio overall |
| 4 | Remove or gate per-chunk debug work and host syncs (1d, 3.1, 3.2) | speed | about 1–2 ms/chunk in streaming |
| 5 | Per-layer cache `Vec`, precomputed `linear_pos`, fast SDPA (3.3, 3.4, 1c) | speed | streaming chunk latency below mlx-audio |
| 6 | Opt-in bf16 encoder weights (1b) | speed | encoder about 40% faster; needs WER check |

Also fix the bench report: the mlx-audio dtype column says `bfloat16`, but its weights and encoder actually run in f32. Either record the real weight dtype, or load through `from_pretrained(dtype=…)` so the label is true.

## Appendix: evidence

The scratch files are session-local, not in the repo:
- Decode variants A–D: scratchpad `rtime/src/bin/dec.rs`, and encoder/decode timing split in `rtime/src/main.rs`.
- mlx-audio timing splits: scratchpad `mta.py`, `mta3.py`, `enc.py`.
- Grid simulations: scratchpad `sim.py`, `sim2.py`, `sim3.py`, with outputs `hyp_nemo.jsonl` and `hyp_rs_s40.jsonl`.

Line numbers marked `~` are approximate (taken from agent reports). Unmarked ones were checked directly.
