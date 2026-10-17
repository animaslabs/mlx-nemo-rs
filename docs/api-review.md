# Public API Review and Proposed Improvements

A review of the `mlx-voice` public API as of v0.1.1 (commit `d2e88d9`), with
suggested changes. It weighs the API against the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/), against
common practice in ML inference libraries (Hugging Face `transformers` /
`candle`, `whisper-rs`, `ort`), and against the two ecosystems this crate
bridges: MLX (`mlx-rs`, `mlx-lm`, `parakeet-mlx`, `mlx-audio`) and NVIDIA NeMo.

The crate is pre-1.0 and has few downstream users, so this is the cheapest point
to make breaking changes. Most of the proposals below are breaking. The
[migration plan](#12-migration-plan) at the end puts them in order.

## Summary

The core works well. The models match NeMo, streaming is cache-aware, and
multilingual and multi-talker support are in. The public API grew alongside the
implementation, and it shows in three ways:

1. **The surface is very large.** About 900 public items, most of them model
   internals (`RelPositionMultiHeadAttention`, `StreamingMelBuffer` with all
   fields `pub`, `reset_encoder_counter`, debug structs). Any refactor of
   internals is a breaking change.
2. **One task, several entry points.** Streaming RNN-T has two independent APIs
   (`EncDecRNNTModel::init_streaming/process_chunk` and
   `CacheAwareRNNTPipeline`). Models load through three constructors each, and
   every example rebuilds the same "directory or `.nemo`?" dispatch.
3. **Some API promises are not kept.** `batch_size > 1` streaming sessions take
   a single flat audio slice, and what streaming mode 0 means depends on the
   model. Two more were fixed with this review: `BeamSearch` was ignored, and a
   failed weight load left a model that still ran (§1.1, §1.2).

### Top priorities

| # | Change | Why | Breaking |
|---|--------|-----|----------|
| 1 | ~~Make weight and tokenizer loading strict~~ (done, §1.2) | Silent random-weight models | Behavior |
| 2 | ~~Wire up or remove `BeamSearch`~~ (done, §1.1) | Setting it did nothing | Yes |
| 3 | One loader: `from_pretrained(path_or_repo)` plus `LoadOptions` | Three constructors per type, dispatch repeated in every example | Yes |
| 4 | Rich result types (`Transcript`, `Token`, `Word` with timestamps and confidence) | `Vec<String>` drops the information users need | Yes |
| 5 | Accept in-memory audio everywhere (`AudioInput`) | Offline models only take file paths | Additive |
| 6 | Merge the two streaming APIs into one session type | Duplicate code paths that drift apart | Yes |
| 7 | Shrink the public surface (`pub(crate)`, `internals` feature) | Locks in internals, hard to learn | Yes |
| 8 | Typed config enums in place of `String` fields | Typos fail at runtime, deep inside a forward pass | Yes |
| 9 | `#[non_exhaustive]` on `Error`, configs and outputs | Lets fields and variants be added later without a breaking release | Yes, once |
| 10 | Public `Tokenizer` wrapper | Token ids can't be turned back into text | Additive |

---

## 1. Correctness issues the API hides

These are bugs, or near-bugs, that come from how the API is shaped. They are
worth fixing before any redesign.

### 1.1 `BeamSearch` was a no-op (fixed)

`EncDecRNNTModel::set_decoding_strategy(RnntDecodingStrategy::BeamSearch { .. })`
stored the strategy, and `new()` also picks it up from `decoding.strategy: beam`
in the YAML, but `transcribe` always ran greedy decoding. The `transcribe_rnnt`
example's `--beam` flag did nothing. The public `rnnt_beam_decode` only accepted
a `&StatelessTransducerDecoder`, so it couldn't serve LSTM prediction networks
such as parakeet-rnnt-0.6b, and no model called it.

**Done:** `rnnt_beam_decode` is now a port of NeMo's default beam search
(`BeamRNNTInfer`, `search_type: default`, length-normalized final choice). It
takes a `&PredictionNetwork`, so it works with stateless and LSTM decoders and
with language-prompt models, and `transcribe` dispatches on the strategy. Beam
search applies to offline `transcribe` only; streaming stays greedy, and the
enum's docs now say so. On the 60 s sample, WER with beam 4 or 8 is equal to
greedy on parakeet-rnnt and nemotron-speech, and lower on the 8-bit nemotron
(0.110–0.116 vs 0.123).

**Still open:** `rnnt_greedy_decode{,_batched}` still take a
`&StatelessTransducerDecoder` and are unused by the models. Generalize them the
same way or stop exporting them.

### 1.2 Lenient weight loading (fixed)

All four `restore` functions caught a failed `load_weights()` and logged
`tracing::warn!("Could not load weights")`, and did the same when the archive
had no weights file. Missing tensors were skipped quietly: `load_layer_norm` and
the conv and BatchNorm helpers returned without a word, and `load_linear_layer`
only warned. Pipelines turned a failed tokenizer load into a warning, after which
every transcript came back empty.

**Done:**

- `restore` fails if the checkpoint has no weights or they don't load.
- `restore`, `restore_from_directory` and both pipeline constructors fail if
  there's no SentencePiece tokenizer or it doesn't load. `restore_safetensors`
  still allows `None`, since the caller chooses that explicitly.
- Every tensor the model has is required. LayerNorm, Conv1d/2d and BatchNorm
  parameters are required whenever the layer has them, and linear, embedding,
  LSTM, joint, CTC, Sortformer, transformer, Jasper and speaker-kernel weights
  must all be present. A missing tensor is an `Error::Checkpoint` naming the key
  (or every alternative key the loader tried).

Turning this on found two real bugs, both now fixed:

- **The conformer depthwise conv always had a bias,** even for `use_bias: false`
  checkpoints (every nemotron and multitalker model). It stayed at its zero
  init, so outputs were unaffected, but strict loading needed the layer built
  from the config. A dead `encoder.final_norm` layer, which NeMo doesn't have
  and the forward pass never used, is gone too.
- **The Sortformer diarizer applied an untrained LayerNorm to its output.** The
  port created the transformer's `final_layer_norm` for `pre_ln: false`, but
  NeMo only creates it for `pre_ln: true` (`transformer_encoders.py`), so the
  checkpoint has no weights for it. After the fix, offline diarizer
  probabilities are 8× closer to NeMo on the 60 s sample (mean |diff| 0.00005,
  was 0.00041; max 0.033, was 0.079). Multi-talker transcripts change slightly
  as a result. For example, the sample's "The Mice in Counciled a council" now
  comes out as "The Mice in Council" / "summoned a council…".

All other local models (parakeet-rnnt MLX, nemotron `.nemo` and 8-bit MLX,
nemotron-3.5, MarbleNet VAD) produce output identical to before.

**Opting out:** `LoadOptions::default().strict(false)` makes a load leave
missing tensors at their initial values and list them in a `LoadReport`. It
also lets `restore` go ahead without a tokenizer, with a warning. The options
reach every entry point:

```rust
use mlx_voice::LoadOptions;

let lenient = LoadOptions::default().strict(false);
let mut model = EncDecRNNTModel::restore_with("partial.nemo", lenient)?;
let report = model.load_weights_with(&weights, None, lenient)?;   // report.missing
let pipeline = CacheAwareRNNTPipeline::restore_from_directory_with("dir/", 0, lenient)?;
let mut config = MultiTalkerConfig::default();
config.load = lenient;
let mt = MultiTalkerPipeline::restore_with_config(asr, diar, 0, config)?;
```

Layer loaders take a `core::Weights` (the checkpoint's tensors plus the
options) in place of a bare `HashMap`. `Weights::from(&map)` loads strictly.

**Still open:** loading reports missing tensors, but not *unexpected* ones
(checkpoint keys nothing consumed). That needs key-consumption tracking, which
`ModuleParameters` integration (§7.1) would provide, and would add an
`unexpected` list to `LoadReport`. `restore_safetensors` has no `_with` variant;
to load those files non-strictly, call `load_weights_with` yourself.

### 1.3 Streaming `batch_size` is not real

`EncDecRNNTModel::init_streaming(mode_idx, batch_size)` and the CTC version
allocate caches and decoder states for `batch_size` streams. `process_chunk`
then takes `audio_chunk: &[f32]` and reshapes it to `[1, len]`
(`enc_dec_ctc.rs:572`). The CTC doc comment says "`[batch, samples]` or
`[samples]`", which the type can't express. Any `batch_size > 1` is either wrong
or unsupported.

**Suggestion:** drop `batch_size` from the public signature until batched
streaming works. When it does, take `&[&[f32]]` (one slice per stream), as
`MultiTalkerPipeline` already does internally with its batched encoder cache.

### 1.4 What streaming mode 0 means was contradictory (docs fixed)

**Done:** every `mode_idx` doc, `AttContextSize::get_default` and the README now say the same thing: the index into the checkpoint's list, where mode 0 is the default and has the most lookahead. The type-safe `StreamingMode` below is still open.

`CacheAwareRNNTPipeline::restore` documents `mode_idx` as "0 = highest latency",
and `EncDecCTCModel::init_streaming` says "0 = default, typically lowest
latency". In fact the order is whatever the checkpoint's `att_context_size` list
holds, so it varies by model. See §4.4 for a type-safe alternative.

### 1.5 Duplicated audio loading (fixed)

**Done:** the four private `load_wav` copies and the pipeline's `resample_audio` wrapper are gone; everything uses `crate::audio`.

`enc_dec_rnnt.rs`, `enc_dec_ctc.rs`, `enc_dec_classification.rs` and
`cache_aware_rnnt.rs` each have a private `load_wav` copy alongside the public
`crate::audio::load_wav`, and the pipeline has its own `resample_audio`. These copies can drift, for example on mono downmix or 24-bit
scaling. Route everything through `crate::audio`.

---

## 2. Loading models

**Done (§2.1–2.2):** every model and both pipelines load through
`from_pretrained(source)` and `from_pretrained_with(source, LoadOptions)`.
`source` is anything that converts into `ModelSource`: a `.nemo` path, a model
directory, a Hub repo id string, or a `HubRepo` with a revision or token.
Hub downloads use a new `hub` feature (`hf-hub`'s blocking ureq client). They
fetch a repository's `.nemo` archive if it has one, otherwise its safetensors,
config and tokenizer files, into the Hugging Face cache. `restore*` is gone;
`#[doc(alias = "restore")]` keeps it searchable. `restore_safetensors` is now
`from_safetensors`. The examples no longer branch on directory vs. file.
`MultiTalkerPipeline::from_pretrained(asr, diar, mode)` takes two sources.

Still open: `dtype` in `LoadOptions` (§7.2), building pipelines from a loaded
model (§2.3, part of the streaming redesign in §4), and model auto-detection
(§2.4).

### 2.1 Today

```rust
EncDecRNNTModel::restore(path)                                  // .nemo
EncDecRNNTModel::restore_safetensors(w, cfg, Some(tok), fmt)    // loose files
EncDecRNNTModel::restore_from_directory(dir)                    // MLX dir
CacheAwareRNNTPipeline::restore(path, mode_idx)                 // .nemo only
CacheAwareRNNTPipeline::restore_from_directory(dir, mode_idx)
MultiTalkerPipeline::restore(asr, diar, mode_idx)               // .nemo or dir
EncDecClassificationModel::restore(path)                        // .nemo only
```

Six examples carry the same `if path.is_dir() { restore_from_directory } else { restore }`
branch (`examples/eval.rs:113`, `streaming.rs:144`, `transcribe.rs:116`,
`transcribe_rnnt.rs:174`, `live_pipeline.rs:141`, `multi_talker.rs:209`).
`MultiTalkerPipeline::restore` already does the dispatch internally, so the
three types don't even agree with each other.

### 2.2 Proposal

One constructor per type that takes a path or a Hub repo id, plus an options
struct for everything optional. This mirrors `from_pretrained` in NeMo,
`transformers`, `parakeet-mlx` and `mlx-lm`'s `load`:

```rust
let model = EncDecRNNTModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;      // Hub (feature "hub")
let model = EncDecRNNTModel::from_pretrained("models/parakeet.nemo")?;            // .nemo
let model = EncDecRNNTModel::from_pretrained("models/parakeet-mlx/")?;            // MLX dir

let model = EncDecRNNTModel::from_pretrained_with(
    "nvidia/nemotron-speech-streaming-en-0.6b",
    LoadOptions::default()
        .dtype(Dtype::Bfloat16)          // §7.2
        .revision("main")
        .token(std::env::var("HF_TOKEN").ok())
        .strict(true),
)?;
```

- Decide the source in one place, `ModelSource::resolve(&str) -> ModelSource { Nemo(PathBuf), Dir(PathBuf), Files { .. }, Hub { repo, revision } }`.
  Keep `restore_safetensors` only as a lower-level `from_parts(config, weights, tokenizer)`.
- **Hub support behind a `hub` feature.** `hf-hub` is already a workspace
  dependency for `xtask`, so moving download logic into the library is cheap,
  and it's what users of every comparable library expect.
- **Keep `restore` as a `#[doc(alias)]`** so NeMo users searching the docs for
  `restore_from` still find `from_pretrained`.

### 2.3 Build pipelines from models

`CacheAwareRNNTPipeline::restore` repeats about 250 lines of
`EncDecRNNTModel::restore` (`cache_aware_rnnt.rs:272-520`). A user who wants
both offline and streaming transcription loads the weights twice.

```rust
let model = EncDecRNNTModel::from_pretrained(path)?;
let mut stream = model.stream(StreamingOptions::default().latency_ms(560))?;  // borrows the model
// or, to own it:
let pipeline = CacheAwareRNNTPipeline::new(model, StreamingOptions::default())?;
```

The pipeline then holds an `EncDecRNNTModel` (or an `Arc` of one) rather than
copies of its fields. This also removes the parallel `set_language`,
`languages`, `detected_language`, `set_strip_lang_tags`, `sample_rate`,
`blank_id`, `encoder`, `joint` and `prediction_network` methods that both types
carry today.

### 2.4 Auto-detect the model class

NeMo's `ASRModel.restore_from` reads `target` from the config and builds the
right class. A small dispatcher saves users from knowing whether a checkpoint is
CTC, RNN-T, prompt-conditioned or multitalker:

```rust
pub enum AnyAsrModel { Ctc(EncDecCTCModel), Rnnt(EncDecRNNTModel) }
let model = mlx_voice::asr::load("model.nemo")?;   // -> AnyAsrModel, which implements AsrModel (§6.1)
```

`AsrModelConfig::is_rnnt()` and `is_ctc()` already exist to back it.

---

## 3. Offline inference

### 3.1 Accept audio, not only paths (partly done)

**Done:** `EncDecCTCModel` and `EncDecRNNTModel` have `transcribe_samples(&[f32], sample_rate) -> String`, which resamples to the model's rate. The `AudioInput` design below, which would unify paths and samples in `transcribe`, is still open.

`EncDecCTCModel::transcribe` and `EncDecRNNTModel::transcribe` only take file
paths (`&[impl AsRef<Path>]`). There's no supported way to transcribe samples
already in memory, from a microphone, a decoder, or a network buffer, without
building the padded `Array` and calling `forward` plus a decoder by hand.
(`EncDecClassificationModel` alone has `infer_samples`.) Most inference
libraries take raw arrays first and files second: NeMo's `transcribe` takes
paths or numpy arrays, `whisper-rs` takes `&[f32]`, and `parakeet-mlx`
`transcribe` takes a path but exposes `generate(mel)`.

```rust
pub enum AudioInput<'a> {
    Path(&'a Path),
    Samples { data: &'a [f32], sample_rate: u32 },
}
impl<'a> From<&'a Path> for AudioInput<'a> { /* ... */ }
impl<'a> From<&'a str> for AudioInput<'a> { /* path */ }
impl<'a> From<(&'a [f32], u32)> for AudioInput<'a> { /* samples */ }

pub fn transcribe<'a, I>(&self, inputs: I) -> Result<Vec<Transcript>>
where I: IntoIterator, I::Item: Into<AudioInput<'a>>;

pub fn transcribe_one<'a>(&self, input: impl Into<AudioInput<'a>>) -> Result<Transcript>;
```

Resampling happens inside, as it already does for files. Requiring the sample
rate with `Samples` removes the most common bug: feeding 44.1 kHz or 48 kHz audio
to a 16 kHz model.

### 3.2 Return structured results

`transcribe` returns `Vec<String>`. Token ids, timestamps and confidences are
all computed or computable (`DecodeStepOutput::timesteps` already tracks frame
indices; `MultiTalkerPipeline` already builds per-word timing) but get thrown
away. NeMo returns `Hypothesis` objects with `timestamp={'word','segment','char'}`
when `timestamps=True`. `parakeet-mlx` returns `AlignedResult` with sentences and
tokens.

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<Token>,
    pub language: Option<String>,   // from the last language tag, for prompt models
}

#[non_exhaustive]
pub struct Token { pub id: TokenId, pub text: String, pub start: f32, pub end: f32, pub confidence: Option<f32> }

impl Transcript {
    pub fn words(&self) -> impl Iterator<Item = Word> + '_;     // grouped by SentencePiece word start
    pub fn sentences(&self) -> impl Iterator<Item = Segment> + '_;
}
impl Display for Transcript { /* writes text */ }
```

`SpeakerWord`/`SpeakerSegment` can then become `Word`/`Segment` with
`speaker: Option<usize>`, so both pipelines share one result vocabulary.

### 3.3 Inference options

Per-call knobs are currently setters on the model (`set_decoding_strategy`,
`set_strip_lang_tags`, `set_language`, `set_encoder_dtype`). Setters that
change shared state make a model hard to share between call sites. A
`TranscribeOptions` with a `Default` makes each call self-contained:

```rust
model.transcribe_with(inputs, &TranscribeOptions {
    decoding: Decoding::Greedy { max_symbols: 10 },
    language: Some("de-DE".into()),
    timestamps: true,
    batch_size: 8,               // NeMo's transcribe(batch_size=...); bounds peak memory
    strip_lang_tags: true,
    ..Default::default()
})?;
```

Keep model-level defaults, set at load time, for anything that changes weights
or compiled state, such as dtype or quantization.

### 3.4 `&mut self` on inference

`transcribe` and `forward` take `&mut self`, partly because `ConformerEncoder::forward`
does and partly for leftover debug counters (`chunk_count`, the global
`ENCODER_CALL_COUNT` reset by the public `reset_encoder_counter`). Offline
inference has no logical state. Making it `&self` lets one loaded model serve
many requests behind an `Arc` (with a `Mutex` only where MLX requires it; see
§7.3). Move the counters into `tracing` spans or drop them.

---

## 4. Streaming

### 4.1 Two streaming APIs

| | `EncDecRNNTModel` + `RnntStreamingSession` | `CacheAwareRNNTPipeline` |
|---|---|---|
| Input | `process_chunk(&mut session, &[f32])` | `add_audio(&[f32])` + `transcribe_step()` |
| Mel | Batch mel per chunk + `StreamingMelBuffer` | `StreamingMel` (stateful, exact) |
| Decoding | Per-model loop | `RnntStreamingDecoder` |
| Output | `StreamingOutput { tokens: Vec<Vec<_>>, text: Vec<String> }` | `TranscribeOutput { text, new_tokens, ... }` |
| Extras | — | Blank/silence thresholds, punctuation suppression, incremental detokenizer |

These are two implementations of NeMo's cache-aware streaming that have already
drifted apart. The model path still recomputes mel per chunk "statelessly" while
the pipeline uses the exact streaming mel. `EncDecCTCModel` has only the first.
The eval example, the README and `MultiTalkerPipeline` all use the pipeline.

**Suggestion:** keep one design. A session object that borrows (or owns) the
model and holds every piece of streaming state:

```rust
let mut stream = model.stream(StreamingOptions::default())?;   // works for CTC and RNN-T
stream.push_audio(&samples, 48_000)?;                           // resamples internally
while let Some(update) = stream.next_update()? {
    print!("{}", update.delta);           // text appended since the last update
}
let transcript: Transcript = stream.finish()?;                   // flushes, consumes the stream
```

- `push_audio` / `finish` replace `add_audio` / `end_stream` / `is_done` /
  `run`. `finish` consuming `self` makes "push after end" a compile error.
- File mode (`set_audio_file`, `set_audio_samples`) belongs in a helper,
  `model.transcribe_streaming(input, opts) -> Transcript`, not in the stream's
  state machine. Today `set_audio_file` silently `reset()`s a live stream.
- Hold model state and stream state separately, so one model can run several
  concurrent streams. NeMo's own `CacheAwareStreamingAudioBuffer` and
  `parakeet-mlx`'s `transcribe_stream()` context manager both do this.
- Make `StreamingSession` fields private. Today `RnntStreamingSession` exposes
  `encoder_cache`, `mel_buffer`, `decoder_states`, `last_tokens` and
  `is_first_chunk` as `pub` fields that users can corrupt.

### 4.2 Clear output semantics

`TranscribeOutput` has `text` ("accumulated since stream start"), `new_tokens`,
`tokens` (empty except on the final step) and `token_count`. The pipeline's own
doc example in `pipeline/mod.rs` prints `output.text` on every step, which
repeats the whole transcript each time. A field that is only filled in
sometimes is a trap, and its name should say which kind of text it holds.

```rust
#[non_exhaustive]
pub struct StreamUpdate {
    pub delta: String,         // stable text appended in this step
    pub tokens: Vec<Token>,    // tokens emitted in this step, with timestamps
    pub is_final: bool,
}
impl StreamingSession { pub fn transcript(&self) -> &str; }   // full text, borrowed, no O(n²) cloning
```

### 4.3 Resampling and input format for live audio

Both live examples carry their own `rubato` resampler and about 60 lines of
glue (`examples/live.rs:451-505`, `live_pipeline.rs:537-570`), because
`add_audio` assumes model-rate mono `f32`. The library already depends on
`rubato`. A stateful streaming resampler inside `push_audio(samples, rate)`
removes the most error-prone part of every real-time integration. Consider
interleaved multi-channel input too (`push_interleaved(samples, rate, channels)`)
to match what `cpal` delivers.

### 4.4 Latency modes as a type

`mode_idx: usize` is an index into a list whose order depends on the checkpoint
(§1.4). `StreamingModeInfo` already carries `latency_ms`, `left_context` and
`right_context`. Let users choose by the quantity they care about:

```rust
pub struct StreamingMode { /* private: index, left, right, latency_ms */ }
impl EncDecRNNTModel {
    pub fn streaming_modes(&self) -> &[StreamingMode];
}
StreamingOptions::default()
    .mode(StreamingMode::LowestLatency)      // or ::HighestAccuracy, ::Latency(Duration), ::Index(n)
```

Put `Default` on the checkpoint's own default (`att_context_size` entry 0, the
one NeMo uses), not on index 0 of a list that changes from model to model.
Allow switching modes on an existing model without reloading.

### 4.5 Smaller cleanups

- `set_blank_threshold` rebuilds `RnntStreamingDecoder` from scratch, which
  discards any other decoder setting. It will bite once `max_symbols` becomes
  configurable. Move all such knobs into `StreamingOptions`.
- `flush_audio_to_mel` and `streaming_mel().latest_frame_bins()` exist to drive a
  spectrogram view in a live UI. Expose that as an explicit observer
  (`StreamUpdate::mel_frames` or a callback), not by letting callers reach into
  the pipeline's preprocessor.
- `compute_streaming_config(..., 10000)` has a magic maximum length. Name it, or
  derive it from `pos_emb_max_len`.

---

## 5. Configuration types

### 5.1 Stringly-typed fields

Many config fields are `String` where the set of valid values is closed:

| Field | Values |
|---|---|
| `PreprocessorConfig::window` / `MelConfig::window` | `hann`, `hamming`, ... |
| `PreprocessorConfig::normalize` | `per_feature`, `all_features`, `NA` |
| `EncoderConfig::subsampling` | `dw_striding`, `striding`, `vggnet`, ... |
| `EncoderConfig::self_attention_model` | `rel_pos`, `abs_pos`, `rel_pos_local_attn` |
| `EncoderConfig::att_context_style` | `regular`, `chunked_limited` |
| `EncoderConfig::conv_norm_type` | `batch_norm`, `layer_norm` |
| `JointConfig::activation` | `relu`, `tanh`, `sigmoid` |
| `DecodingConfig::strategy` | `greedy`, `greedy_batch`, `beam`, `tsl`, `maes`, ... |
| `dsp::create_window(window_type: &str, ..)` | `hann`, `hamming` |

A typo or an unsupported variant currently surfaces as a runtime error or a
silent fallback somewhere in a forward pass. `JointActivation::from_str`
(`rnnt.rs:811`) falls back silently, and its inherent `from_str` shadows the
standard `FromStr` trait (clippy `should_implement_trait`).

**Suggestion:** use serde-backed enums with an escape hatch that keeps
unrecognized NeMo values, so that new NeMo releases still parse:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Normalize {
    PerFeature,
    AllFeatures,
    #[serde(alias = "NA")] None,
    #[serde(untagged)] Other(String),
}
```

Then `validate()` turns `Other(..)` into a clear `Error::Unsupported { field, value }`
at load time.

### 5.2 Option fields with `effective_*` helpers

`EncoderConfig` has `xscaling: Option<bool>`, `untie_biases: Option<bool>`,
`use_bias: Option<bool>` and more, plus `effective_use_bias()`,
`is_causal_conv()`, `JointConfig::effective_joint_hidden()`, etc. The defaults
live in two places: `#[serde(default)]` and the helpers. Apply NeMo's defaults
during deserialization (`#[serde(default = "...")]`) and store plain `bool` and
`usize` values. Keep `Option` only where "absent" means something different
from any default.

### 5.3 Types and naming

- `DecoderConfig::num_classes: i32` and `EncoderConfig::feat_out: i32` use `-1`
  as a sentinel. Use `Option<usize>`, with a custom deserializer that maps
  `-1` to `None`.
- `AttContextSize::get_default` and `get_mode` break the API guidelines' rule
  against `get_` prefixes (C-GETTER). Rename them `default_mode` and `mode`.
- `MelConfig` and `PreprocessorConfig` describe the same thing, and `fmin`/`fmax`
  duplicate `lowfreq`/`highfreq`. Keep `PreprocessorConfig` as the
  NeMo-faithful serde type and make `MelConfig` the validated runtime type,
  converted with `TryFrom` rather than an infallible `From` that can't reject
  bad values.
- `FromConfig` (`core/config.rs:14`) is public but has no implementations.
  Either implement it for every model and module (so
  `ConformerEncoder::from_config(cfg)` works everywhere) or remove it.
- Make configs `#[non_exhaustive]` and give them `Default`, so that adding a NeMo
  field is not a breaking change. Users then build them with
  `EncoderConfig { d_model: 512, ..Default::default() }`.
- `pub use config::*` in `asr/mod.rs` puts about 20 config types at
  `mlx_voice::asr::` root. Leave them under `asr::config`.

---

## 6. Traits and naming

### 6.1 Shared traits

Every model reimplements the same inherent methods (`sample_rate`, `config`,
`transcribe`, `forward`, `supports_streaming`, `streaming_modes`, `blank_id`,
language methods). Generic code, such as the eval example's
`enum Model { Offline(..), Streaming(..) }` (`examples/eval.rs:113`), has to
wrap each type by hand. A few small traits would cover it:

```rust
pub trait AsrModel {
    fn sample_rate(&self) -> u32;
    fn transcribe_with<'a>(&self, inputs: &[AudioInput<'a>], opts: &TranscribeOptions) -> Result<Vec<Transcript>>;
    fn tokenizer(&self) -> &Tokenizer;
}
pub trait StreamingAsrModel: AsrModel {
    type Stream<'m>: StreamingSession where Self: 'm;
    fn streaming_modes(&self) -> &[StreamingMode];
    fn stream(&self, opts: StreamingOptions) -> Result<Self::Stream<'_>>;
}
pub trait Multilingual {
    fn languages(&self) -> &[Language];
    fn set_default_language(&mut self, lang: Language) -> Result<()>;
}
```

The decoder side benefits the same way. A `PredictionNet` trait implemented by
`StatelessTransducerDecoder` and `RNNTDecoder` would let decoders such as
`rnnt_greedy_decode` work with both (§1.1) and replace the `PredictionNetwork`
and `DecoderState` enums that `rnnt_beam_decode` now matches on.

### 6.2 Naming conventions

The Rust API guidelines (C-CASE) treat acronyms as words: `Rnnt`, `Ctc`, `Vad`.
The crate mixes both styles: `EncDecRNNTModel` and `RnntStreamingSession`,
`CTCDecoder` and `CtcStreamingSession`, `RNNTJoint` and `RnntDecodingStrategy`.

NeMo class names are useful for discoverability, so:

- Pick Rust casing for every type (`EncDecRnntModel`, `CtcDecoder`, `RnntJoint`).
- Add `#[doc(alias = "EncDecRNNTModel")]` so a docs.rs search for the NeMo name
  still lands on the right type.
- Keep a one-release `#[deprecated] pub type EncDecRNNTModel = EncDecRnntModel;`.

Other naming inconsistencies:

| Now | Issue | Suggestion |
|---|---|---|
| `blank_id()` / `blank_idx()` | Both are used | `blank_id` |
| `transcribe` / `infer` / `diarize` / `forward` | `EncDecClassificationModel::infer` returns `Vec<Vec<f32>>` with no labels | `classify` → `Vec<FrameLabels>`; `diarize` → `Vec<SpeakerActivity>` |
| `language()` → `Option<&str>`, `detected_language()` → `Option<String>`, `languages()` → `Vec<&str>` | Three return shapes for one concept | A `Language` newtype (BCP-47-ish) and `&[Language]` |
| `MelSpectrogram` / `MelSpectrogramConfig` aliases | "Backwards-compatible" aliases in a 0.1 crate | Remove |
| `get_streaming_modes`, `compute_streaming_config` | Free functions with a `get_` prefix | Methods on the model or `EncoderConfig` |
| `transcribe_step` | Doesn't always transcribe; returns `None` when starved | `next_update` / `poll` |
| `DebugOutput` | Top-level `asr::DebugOutput` but CTC-only | Behind the `internals` feature (§8) |

### 6.3 Token ids

Token ids are `usize` everywhere, lengths are `i32` inside `Array`s, and shapes
are `i32`. `usize` is reasonable, but a `pub type TokenId = u32;` (or a newtype)
documents intent, halves memory for long transcripts, and matches SentencePiece
and Hugging Face `tokenizers`.

---

## 7. MLX-specific conventions

### 7.1 Integrate with `mlx-rs` modules

`mlx-rs` provides `Module<Input>` and `ModuleParameters` (with a derive macro)
modeled on `mlx.nn.Module`. They give `parameters()`, `update()`,
`load_safetensors()`, `quantize`, freezing, and `eval` control without per-layer
code. This crate hand-writes `load_weights(&HashMap<String, Array>, prefix)` on
every layer and wraps quantization in its own `LinearLayer`/`EmbeddingLayer`
enums.

Moving to `#[derive(ModuleParameters)]` with NeMo key remapping in one place
(the way `mlx-lm`'s `sanitize()` works) would:

- make strict loading (§1.2) fall out for free, since missing and unexpected
  keys come from comparing two parameter trees;
- let users quantize any model through `mlx-rs`'s `Quantizable` trait
  (`nn::quantize(model, group_size, bits)`) and save it back with
  `save_safetensors`, matching the `mlx-community` workflow;
- remove the public `load_weights`, `load_linear_layer`, `load_embedding_layer`
  and `LinearLayer`/`EmbeddingLayer` from the API.

This is the largest change in this document. Treat it as a direction, not a
blocker for the rest.

### 7.2 Dtype and precision

`set_encoder_dtype` exists only on `EncDecRNNTModel`, applies only to offline
`transcribe` and `forward` (its docs say so), and isn't available at load time.
MLX libraries usually choose dtype at load time and apply it to the whole model
(`mlx_lm.load(..., dtype=...)`, `parakeet-mlx`'s `from_pretrained(dtype=mx.bfloat16)`).
Put `LoadOptions::dtype` (§2.2) in charge. Where a component must stay in
`f32` (mel, the decoder's LSTM), document that once in the type docs and keep
the logic internal.

### 7.3 Threading and devices

`mlx_rs::Array` is `Send` but not `Sync`, and MLX's default stream is global.
The crate doesn't say which of its types are `Send` or `Sync`, and the live
examples work around that with a manual audio thread plus a `ringbuf`. Suggestions:

- Add compile-time assertions (`const _: () = assert_send::<EncDecRnntModel>();`)
  so that a dependency upgrade can't silently change thread safety.
- Document the intended pattern, "one model per thread, or `Mutex<Model>`", in
  the crate docs.
- Accept an optional `mlx_rs::StreamOrDevice` in `LoadOptions` and
  `TranscribeOptions`, so users can choose CPU or GPU, or run two models on
  separate streams. Every `mlx-rs` op already takes one.
- Expose MLX memory controls (`mlx_rs::memory::set_cache_limit` and friends)
  through the re-exported `mlx_rs`, and mention them in docs. Long-running
  streaming services need them.

### 7.4 Tensor layout

Public tensor layouts aren't consistent:

- `ConformerEncoder::forward` returns `[B, D, T]` (NeMo's channels-first).
- `EncDecRNNTModel::forward` transposes to `[B, T, D]`.
- `EncDecCTCModel::forward` returns log-probs as `[B, T, V]`.
- `DebugOutput::encoder_output` is `[B, D, T]`.
- `StreamingMelBuffer` stores mel as `[n_mels, T]`.

MLX convolutions are channels-last (`NLC`), and `mlx-lm` and `mlx-whisper` use
`[B, T, D]` throughout. Pick channels-last for every public tensor, transpose at
the NeMo weight-import boundary only, and state shapes in doc comments in one
consistent format (`[batch, time, dim]`). Lengths should be `&[usize]` or
`Vec<usize>` in high-level APIs, not an `Array` users must build.

---

## 8. Shrinking the public surface (done)

**Done:** the documented public API went from about 900 items to about 55.

- `asr::{decoding, dsp, modules, streaming}` and
  `core::{loading, pytorch, quantization, weights}` are crate-private unless
  the `internals` feature is on. The stable types they contain
  (`MelProcessor`, `MelConfig`, `StreamingMel`, `SpectrumTransform`,
  `StreamingConfig`, `StreamingModeInfo`, `StreamingOutput`, `LoadOptions`,
  `LoadReport`, `QuantizationConfig`, `WeightFormat`) are re-exported from
  `asr` and `core`. `asr::config::*` is no longer glob-re-exported.
- `PredictionNetwork` and `DecoderState` moved into `modules::decoders`.
  The `encoder()`, `prediction_network()` and `joint()` accessors exist only
  with `internals`, and Sortformer's streaming step is crate-private.
- Removed the `MelSpectrogram`/`MelSpectrogramConfig` aliases, the stateless mel
  API (`MelStreamState`, `frames_available`, `process_audio_streaming`,
  `process_frame`), `reset_encoder_counter`, `DecoderState::compute_norm`,
  `DebugOutput`, `FromConfig`, and the public `NemoCheckpoint::open_gzip`.
- Made the fields of `NemoCheckpoint`, `SafetensorsLoader` and the streaming
  sessions private, with getters where they're read.
- Added `#[non_exhaustive]` to `Error`, every config struct, `MelConfig`,
  `QuantizationConfig`, the output types and the public enums.
- The debug examples and `tests/pytorch_loading.rs` require `internals`.
  Doctests pass with and without the feature.

The `cargo public-api` snapshot is still open.

About 900 `pub` items are reachable. Under SemVer each of them is a promise. The
Rust API guidelines (C-STRUCT-PRIVATE, future proofing) recommend private
fields and a small, deliberate facade.

**Proposed tiers:**

1. **Stable facade** at `mlx_voice::` and `mlx_voice::asr::`: models, pipelines and
   sessions, options, result types, `Tokenizer`, `Error`, `audio::{load, resample}`,
   `prelude`.
2. **Building blocks** in `mlx_voice::asr::{modules, decoding, streaming}`, public
   but `#[non_exhaustive]` with private fields: `MelProcessor`, `StreamingMel`,
   `SpectrumTransform`, `ConformerEncoder`, decode functions, `VadPostProcessor`.
3. **Internals** behind a `features = ["internals"]` flag (or `#[doc(hidden)]`),
   with no SemVer guarantee: `RelPositionMultiHeadAttention`,
   `RelPositionalEncoding`, `AttentionContext`, `EncoderCache`,
   `StreamingMelBuffer`, `FeatureBuffer`, `StreamingSortformerState`,
   `SpeakerKernels`, `LanguagePrompt`, `core::pytorch`, `core::weights`,
   `quantization::*`, `forward_debug`, `forward_subsampling_only`,
   `forward_from_embeddings`, `DebugOutput`, `DebugEncoderOutput`.

**Remove:**

- `reset_encoder_counter` (debug-only global state).
- `DecoderState::compute_norm` (debug helper that swallows errors with `unwrap_or_default`).
- `StreamingMelBuffer::memory_stats`.
- The legacy stateless mel streaming API (`MelStreamState`, `StreamingMelConfig`,
  `frames_available`, `process_audio_streaming`). The module comment already
  calls it legacy, so this is a good time to drop it.
- `NemoCheckpoint::open_gzip` as a public API (`open` detects gzip), and the
  redundant `.or_else(open_gzip)` fallbacks in every `restore`.

**Make private:** every `pub` field on stateful types (`RnntStreamingSession`,
`CtcStreamingSession`, `EncoderCache`, `StreamingMelBuffer`, `NemoCheckpoint`,
`SafetensorsLoader`, `StreamingSortformerState`, `SortformerModules::encoder_proj`,
`RelPositionMultiHeadAttention::cache_drop_size`), with getters where reads are
needed.

Tools to keep this in check:

- `cargo public-api` snapshot test in CI, so any surface change is visible in review.
- `cargo semver-checks` before each release.
- `#![warn(missing_docs, unreachable_pub)]` and
  `#![deny(rustdoc::broken_intra_doc_links)]` in `lib.rs`.

---

## 9. Errors

`Error` is a reasonable `thiserror` enum. Improvements:

- **`#[non_exhaustive]`**, so new variants aren't breaking.
- **Structured variants for failures users act on.** Most errors are
  `Model(String)` or `Config { message }`, which callers can only match by
  string. Candidates:
  - `UnsupportedModel { target: String }`
  - `MissingWeights { missing: Vec<String>, unexpected: Vec<String> }`
  - `StreamingNotSupported`
  - `UnknownLanguage { requested: String, available: Vec<String> }`
  - `SampleRate { expected: u32, actual: u32 }`
  - `Unsupported { field: &'static str, value: String }`
- **Keep MLX errors as MLX errors.** `ArrayExt::to_vec_row_major` maps an MLX
  exception to `Error::model(e.to_string())` (`core/array.rs:21`), which loses
  the source. Use `?` with the existing `From<Exception>`.
- **Keep error sources.** `Audio(String)` wraps `hound` and `rubato` errors
  with `format!`, which drops `source()`. Use
  `Audio { message, #[source] source: Option<Box<dyn Error + Send + Sync>> }`,
  like `Config` already does.
- **Leave app errors to apps.** The examples use `mlx_voice::Error::audio(..)`
  for `cpal` and `ctrlc` errors. That's fine in an example, but the public
  `Error::audio` and `Error::model` constructors invite it. Consider making the
  constructors `pub(crate)` and showing `anyhow` in examples.
- **Note the SemVer coupling.** `Error::Mlx(mlx_rs::error::Exception)` puts
  `mlx-rs` (0.x) types in this crate's public API, and the crate re-exports
  `mlx_rs` too. That's a sensible choice, but every `mlx-rs` minor bump is a
  breaking release of `mlx-voice`. Say so in the README.

---

## 10. Tokenizer, audio and features

### 10.1 Public `Tokenizer` (done)

**Done:** `mlx_voice::asr::Tokenizer` has `decode`, `encode`, `piece_to_id`, `vocab_size`, `is_word_start`, `is_lang_tag` and `detected_language`, and `tokenizer()` is available on the CTC and RNN-T models and on both pipelines. Token ids stay `usize` for now (no `TokenId` alias). The free functions in `asr::decoding` that take a `SentencePieceProcessor` are still public.

The models hold a `sentencepiece::SentencePieceProcessor`, but no model exposes
it, and the crate doesn't re-export `sentencepiece`. Users can get token ids
(`TranscribeOutput::new_tokens`, `StreamingOutput::tokens`) but can't turn them
into text, inspect the vocabulary, or call `decoding::decode_text` and
`IncrementalDetokenizer::update`, which take a `&SentencePieceProcessor` they
have no way to name.

```rust
pub struct Tokenizer { /* SentencePiece + word_start + lang_tag tables */ }
impl Tokenizer {
    pub fn decode(&self, ids: &[TokenId]) -> String;
    pub fn id_to_piece(&self, id: TokenId) -> Option<&str>;
    pub fn vocab_size(&self) -> usize;
    pub fn is_word_start(&self, id: TokenId) -> bool;
    pub fn language_of(&self, id: TokenId) -> Option<&str>;
}
impl EncDecRnntModel { pub fn tokenizer(&self) -> &Tokenizer; }
```

This also hides the C++ `sentencepiece` dependency behind a type the crate
controls. A pure-Rust backend (for example `tokenizers`' Unigram model) could
replace it later without a breaking change.

### 10.2 `audio` module and features

- `audio::load_wav` and `audio::resample` compile without the `asr` feature but
  return `NotImplemented` at runtime. Gate them at compile time
  (`#[cfg(feature = "audio")]`) so misuse fails to build. Split the features:
  `audio = ["dep:hound", "dep:rubato"]` and `asr = ["audio"]`.
- `load_wav` returns `(Vec<f32>, u32)`. A small `struct Audio { samples, sample_rate }`
  (or `AudioInput` from §3.1) is self-describing and saves users from mixing up
  the tuple fields.
- Offer `audio::load(path)` with an optional `symphonia` feature for
  FLAC/MP3/OGG. NeMo, `parakeet-mlx` and `mlx-audio` all accept common formats
  via ffmpeg or soundfile, and WAV-only is a frequent stumbling block.
- Add a `serde` feature that derives `Serialize`/`Deserialize` on result types
  (`Transcript`, `Word`, `Segment`, `SpeechSegment`, `StreamingMode`). The
  multi-talker output is a SegLST, a JSON format by definition.
- `NemoCheckpoint::open` extracts the whole archive to a temp directory and
  keeps `artifacts` in memory as `HashMap<String, Vec<u8>>`. Reading TAR entries
  straight into memory (or memory-mapping the uncompressed TAR) would avoid
  doubling disk use for 2+ GB checkpoints. Done well, `tempfile` could also stop
  being a runtime dependency.

---

## 11. Derives and small conventions (partly done)

**Done:** every public type implements `Debug`. Layers and states derive it, and models, pipelines and mel processors have short manual impls that don't print weights. Output types (`TranscribeOutput`, `SpeakerWord`, `SpeakerSegment`, `MultiTalkerOutput`, `MultiTalkerConfig`, `StreamingModeInfo`, `DecodeStepOutput`, `VadPostProcessConfig`) derive `PartialEq`, and `SpeechSegment` derives `Copy`. `RnntDecodingStrategy` derives `Default`. `lib.rs` sets `#![warn(missing_docs, missing_debug_implementations)]`. `#[must_use]` and the `with_*`/`set_*` cleanup are still open.

From the Rust API guidelines interoperability checklist (C-COMMON-TRAITS, C-DEBUG):

- **`Debug` on every public type.** Missing on `EncDecCTCModel`, `EncDecRNNTModel`,
  `EncDecClassificationModel`, `SortformerEncLabelModel`, `CacheAwareRNNTPipeline`,
  `MultiTalkerPipeline`, `PredictionNetwork`, `DecoderState`, `LstmState`,
  `SpeakerTargets`, `MelProcessor`, `StreamingMel`, `VadPostProcessor` and the
  encoder and decoder modules. A manual impl that prints the config and parameter
  count is enough; `mlx-rs`'s `Module` trait even requires `Debug`.
- **`PartialEq`** on output types (`TranscribeOutput`, `SpeakerWord`,
  `SpeakerSegment`, `StreamingModeInfo`) so tests can `assert_eq!` them.
- **`#[must_use]`** on pure getters and on `StreamUpdate` and `Transcript` returns.
- **`Default`** derives in place of manual impls where the values are zero or
  empty (`RnntDecodingStrategy` can use `#[derive(Default)]` with `#[default]`).
- **`Copy` on small value types** (`SpeechSegment`, `LcsAlignmentResult` already has it).
- **Builder `with_*` methods vs. `set_*` setters.** `RnntStreamingDecoder` and
  `FeatureBuffer` use `with_*`, the pipelines use `set_*`. Use options structs
  for configuration and keep `set_*` for runtime-mutable state only.
- **Doc examples should match behavior.** `pipeline/mod.rs` prints the
  accumulated `output.text` every step (§4.2), and
  `RnntDecodingStrategy::BeamSearch` is described as "higher quality" but has no
  effect (§1.1).

---

## 12. Migration plan

These changes are ordered to land as small PRs, each one leaving the crate
working. Steps 1 and 2 are mostly additive. Step 3 is where the breaking
release happens.

**0.1.x (no breaking changes)**

1. ~~Wire up `BeamSearch` (§1.1). Make missing weights and a missing
   tokenizer an error, with an opt-out (§1.2). Fix the mode-0 docs (§1.4) and
   remove the duplicate `load_wav` copies (§1.5).~~ Done.
2. ~~Add the `Tokenizer` wrapper and `model.tokenizer()` (§10.1),
   `transcribe_samples(&[f32], sample_rate)` on CTC/RNN-T (§3.1),
   `Debug`/`PartialEq` derives (§11), and `#![warn(missing_docs)]`.~~ Done.
   Still to do: a `cargo public-api` snapshot test, which needs a nightly
   toolchain in CI.

**0.2.0 (breaking, one release)**

3. ~~Add `#[non_exhaustive]` to `Error`, configs and outputs. Make internals
   `pub(crate)` or put them behind `internals` (§8). Remove the legacy aliases
   and the stateless mel API.~~ Done.
4. ~~Add `from_pretrained` + `LoadOptions`, and replace `restore*` (§2).~~ Done.
5. Add `Transcript`/`Token`/`Word` results and `TranscribeOptions` (§3.2–3.3).
6. Add one `StreamingSession` built from a model, and turn
   `CacheAwareRNNTPipeline` into a thin wrapper or remove it (§4). Remove
   `init_streaming`/`process_chunk`/`finalize_streaming`.
7. Add typed config enums (§5) and Rust-cased type names with
   `#[doc(alias)]` (§6.2).

**Later**

8. `AsrModel`/`StreamingAsrModel` traits and `asr::load` auto-detection (§2.4, §6.1).
9. `ModuleParameters` integration, `nn::quantize`, and channels-last public tensors (§7).
10. The `hub`, `serde` and `symphonia` features (§2.2, §10.2).

---

## Appendix: target API at a glance

```rust
use mlx_voice::prelude::*;

// Offline
let model = EncDecRnntModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;
let t = model.transcribe_one("audio.wav")?;
println!("{t}");
for w in t.words() {
    println!("{:>6.2}–{:<6.2} {}", w.start, w.end, w.text);
}

// In-memory samples at any rate
let t = model.transcribe_one((&samples[..], 48_000))?;

// Streaming (same model, no reload)
let model = EncDecRnntModel::from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")?;
let mut stream = model.stream(
    StreamingOptions::default()
        .mode(StreamingMode::Latency(Duration::from_millis(560)))
        .language("auto"),
)?;
for chunk in mic_chunks {
    stream.push_audio(&chunk, 48_000)?;
    while let Some(u) = stream.next_update()? {
        print!("{}", u.delta);
    }
}
let transcript = stream.finish()?;

// Multi-talker
let mut mt = MultiTalkerPipeline::from_pretrained(
    "nvidia/multitalker-parakeet-streaming-0.6b-v1",
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    MultiTalkerOptions::default(),
)?;
let segments: Vec<Segment> = mt.transcribe_one("meeting.wav")?.segments().collect();
```
