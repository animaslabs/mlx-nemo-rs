# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/) (before 1.0, a minor release may
break the API).

## Unreleased

### Added

- TDT (token-and-duration transducer) models such as
  `nvidia/parakeet-tdt-0.6b-v3` load and transcribe with greedy offline
  decoding, matching NeMo's `greedy_batch` decoder. Token times span each
  token's predicted duration, and punctuation is timed at the end of the word
  before it, as in NeMo. `EncDecRnntModel::is_tdt` tells them apart. Beam
  search and streaming sessions return a not-implemented error for TDT models.
- `Tokenizer::is_punctuation`.

### Fixed

- Offline encoding keeps padded encoder frames from affecting valid ones, as
  NeMo does: padding is hidden from attention and zeroed before each
  convolution. Before, the last second or so of every file, and clips padded
  to a longer clip in a batch, drifted from NeMo's encoder output.

## 0.2.0 (2026-10-17)

A redesign of the public API around `from_pretrained`, `Transcript` results
and one `StreamingSession`. Almost every entry point changed; the migration
guide below maps the 0.1 calls to their replacements.

### Added

- `from_pretrained(source)` and `from_pretrained_with(source, LoadOptions)` on
  every model. A source is a `.nemo` file, a directory of safetensors weights,
  config and tokenizer, or (with the new `hub` feature) a Hugging Face Hub
  repository id. `HubRepo` sets a revision or token.
- `LoadOptions` and `LoadReport`: loads are strict by default, and
  `LoadOptions::default().strict(false)` keeps missing tensors at their
  initial values and lists them.
- Offline results as `Transcript { text, tokens, words, language }`, with
  token and word times in seconds.
- `TranscribeOptions` (decoding strategy, decoder type, language, language-tag
  stripping, batch size) for `transcribe_with`.
- `AudioInput`: `transcribe` takes WAV paths or `(samples, sample_rate)`, and
  resamples to the model's rate.
- `StreamingSession`, from `model.stream(options)` (borrows the model) or
  `StreamingSession::new(model, options)` (owns it):
  - `push_audio(samples, rate)` resamples statefully
  - `next_update()` returns a `StreamUpdate` with new tokens, the appended
    text and `is_final`
  - `finish()` returns a `Transcript`
  - `buffered_samples()` reports the decoding backlog
- `StreamingOptions` holds the streaming mode and the punctuation, blank and
  silence settings. `model.transcribe_streaming(input, options)` streams a
  whole file.
- `StreamingMode` picks a checkpoint's streaming mode by latency or index:
  - `Default`: the checkpoint's first entry, as in NeMo
  - `LowestLatency`: the least lookahead
  - `MaxLatency(Duration)`: the most lookahead within a budget
  - `Index(n)`: an entry by index

  A `usize` converts to `Index`, so `.mode(3)` still works.
- Hybrid RNN-T/CTC models (e.g. `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`)
  load as `EncDecRnntModel`. `DecoderType::Ctc` decodes with their CTC head,
  offline and streaming.
- `asr::Tokenizer`, exposed through `tokenizer()` on the models and pipelines.
- `audio::StreamResampler` for resampling live audio chunk by chunk.
- The `internals` feature exposes encoder layers, decoders, streaming caches
  and checkpoint parsing for research and debugging. Their API may change in
  any release.
- Models and owned sessions are `Send`. The `StreamingSession` docs show the
  audio-callback-to-worker-thread pattern.

### Changed

- Missing weights or a missing tokenizer are now errors. Before, a load could
  return a partly uninitialized model.
- Types use Rust casing. NeMo's names stay searchable as doc aliases.
  - `EncDecRNNTModel` is now `EncDecRnntModel`
  - `EncDecCTCModel` is now `EncDecCtcModel`
  - `RnntDecodingStrategy` is now `DecodingStrategy`
  - `blank_idx` is now `blank_id`
- Config values NeMo writes as strings are enums that keep unknown values in
  `Other(String)`:
  - `WindowType`, `Normalize`, `Subsampling`, `SelfAttentionModel`
  - `AttContextStyle`, `ConvNormType`, `Activation`, `DecodingAlgorithm`
- Other config changes:
  - Boolean flags are plain `bool`s with NeMo's defaults.
  - `num_classes` and `feat_out` are `Option<usize>`.
  - `AttContextSize::get_default`/`get_mode` are now `default_mode`/`mode`.
- An unknown activation or normalization is now a load error, instead of
  silently falling back to ReLU or no normalization.
- `Error`, config structs, options and output types are `#[non_exhaustive]`.
  Build them with `Default` and the builder methods, and match `Error` with a
  wildcard arm.
- Some items are crate-private without the `internals` feature:
  - encoder layers, decoders, decoding functions and streaming caches
  - checkpoint parsing: `NemoCheckpoint`, `NemoCheckpointWriter` and
    `SafetensorsLoader`, which is no longer re-exported at the crate root
  - `WeightFormat`, `QuantizationConfig`, `ArrayExt` and the `core::config`
    YAML helpers
  - the models' low-level methods: `new(config)`, `load_weights`,
    `load_weights_with`, `load_tokenizer` and `forward`

  Models load through `from_pretrained`. The documented API went from about
  900 items to about 55.

### Removed

- `restore`, `restore_from_directory`, `restore_safetensors` and their `_with`
  variants. Use `from_pretrained`.
- `CacheAwareRNNTPipeline` and `TranscribeOutput`. Use `StreamingSession` and
  `StreamUpdate`.
- `init_streaming`, `process_chunk` and `finalize_streaming` on the RNN-T and
  CTC models, along with:
  - `RnntStreamingSession` and `CtcStreamingSession`
  - `StreamingOutput` and `StreamingMelBuffer`
  - the LCS merge helpers
- `transcribe_samples`, `set_decoding_strategy` and the models'
  `set_strip_lang_tags`. Use `AudioInput` and `TranscribeOptions`.
- The `MelSpectrogram` aliases, the stateless mel API, `reset_encoder_counter`,
  `DebugOutput`, `FromConfig` and the `asr::config` glob re-export.

### Fixed

- `RnntDecodingStrategy::BeamSearch` was ignored. Beam search now runs as a
  port of NeMo's default beam search, for LSTM and stateless decoders.
- The Sortformer diarizer applied an untrained final LayerNorm, which NeMo
  only creates for pre-LN encoders. Its output is now about 8x closer to
  NeMo's.
- Offline transcription used the encoder's current streaming mode for its
  attention context, so streaming in another mode changed later offline
  results. Changing modes also didn't clear cached streaming masks.
- The conformer depthwise convolution ignored `use_bias`.

### Migrating from 0.1

**Loading**

```rust
// 0.1
let mut model = EncDecRNNTModel::restore("model.nemo")?;
let mut model = EncDecRNNTModel::restore_from_directory("model_dir")?;

// 0.2: a .nemo file, a directory, or a Hub id with the `hub` feature
let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
let mut model = EncDecRnntModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;
// Load a checkpoint with missing tensors
let mut model = EncDecRnntModel::from_pretrained_with(
    "model.nemo",
    LoadOptions::default().strict(false),
)?;
```

`restore_safetensors(weights, config, tokenizer, format)` has no direct
replacement. Put `model.safetensors`, `config.json` (or `config.yaml`) and
`tokenizer.model` in one directory and call `from_pretrained` on it. The
weight layout is detected automatically.

`EncDecClassificationModel`, `SortformerEncLabelModel` and
`MultiTalkerPipeline` change the same way: `restore` becomes
`from_pretrained`.

**Offline transcription**

```rust
// 0.1
model.set_decoding_strategy(RnntDecodingStrategy::BeamSearch { beam_size: 4 });
let texts: Vec<String> = model.transcribe(&["a.wav", "b.wav"])?;
let text = model.transcribe_samples(&samples, 48_000)?;

// 0.2
let options = TranscribeOptions::default().beam_search(4);
let transcripts = model.transcribe_with(["a.wav", "b.wav"], &options)?;
let texts: Vec<String> = transcripts.into_iter().map(|t| t.text).collect();
let transcript = model.transcribe_one((&samples[..], 48_000))?;
for word in &transcript.words {
    println!("{:.2}-{:.2} {}", word.start, word.end, word.text);
}
```

`set_strip_lang_tags` moves into `TranscribeOptions::strip_lang_tags` and
`StreamingOptions::strip_lang_tags`. `set_language` stays on the model, and
`TranscribeOptions::language` sets it for one call.

**Streaming**

```rust
// 0.1
let mut pipeline = CacheAwareRNNTPipeline::restore("model.nemo", 0)?;
pipeline.set_blank_threshold(0.5);
pipeline.add_audio(&chunk); // at the model's sample rate
while let Some(output) = pipeline.transcribe_step()? {
    print!("{}", output.text);
}

// 0.2
let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
let mut stream = model.stream(StreamingOptions::default().mode(0).blank_threshold(0.5))?;
stream.push_audio(&chunk, 48_000)?; // any rate; resampled
while let Some(update) = stream.next_update()? {
    print!("{}", update.delta);
}
let transcript = stream.finish()?;
```

Other streaming replacements:

| 0.1 | 0.2 |
|---|---|
| `set_audio_file(path)` | `model.transcribe_streaming(path, options)` |
| `set_suppress_punctuation`, `set_blank_threshold`, `set_silence_threshold` | the matching `StreamingOptions` builders |
| `init_streaming` / `process_chunk` / `finalize_streaming` | a `StreamingSession` |
| a `mode_idx` argument | `StreamingOptions::mode`: an index, or `StreamingMode::LowestLatency` / `MaxLatency(..)` |

There is no pure-CTC streaming model from NVIDIA. For CTC streaming, use a
hybrid RNN-T/CTC model with `StreamingOptions::default().decoder_type(DecoderType::Ctc)`.

**Multi-talker**

```rust
// 0.1
let mut pipeline = MultiTalkerPipeline::restore(asr_path, diar_path, mode)?;
pipeline.add_audio(&samples);
while let Some(output) = pipeline.transcribe_step()? { /* ... */ }

// 0.2
let mut config = MultiTalkerConfig::default();
config.streaming = StreamingOptions::default().mode(mode);
let mut pipeline = MultiTalkerPipeline::from_pretrained_with_config(asr, diar, config)?;
pipeline.push_audio(&samples, sample_rate)?;
while let Some(output) = pipeline.next_update()? { /* ... */ }
let segments = pipeline.finish()?;
```

`transcribe(input)` runs a whole file.

**Internals**

Some code needs the `internals` feature:
- code that used encoder layers, decoders, `asr::decoding` or the
  `asr::streaming` caches
- code that used checkpoint parsing (`NemoCheckpoint`, `SafetensorsLoader`)
- code that built a model by hand with `new(config)` and `load_weights`, or
  ran `forward`

For the diarizer, `SortformerEncLabelModel::diarize` stays public.

```toml
mlx-voice = { version = "0.2", features = ["internals"] }
```

## 0.1.0 and 0.1.1

Initial releases.
