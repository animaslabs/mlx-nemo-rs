# mlx-voice

A Rust implementation of NVIDIA NeMo speech models for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) via [`mlx-rs`](https://github.com/oxideai/mlx-rs).

mlx-voice loads NeMo checkpoints (`.nemo` archives) or converted MLX safetensors directly and runs them on the GPU through Metal, with no Python at inference time.

## Features

- **CTC ASR**: `EncDecCtcModel` (e.g. `nvidia/parakeet-ctc-0.6b`)
- **RNN-T / Transducer ASR**: `EncDecRnntModel` with greedy or beam search decoding (e.g. `nvidia/parakeet-rnnt-0.6b`)
- **TDT (token-and-duration transducer) ASR**: greedy offline decoding with predicted token durations, including the multilingual `nvidia/parakeet-tdt-0.6b-v3` (25 European languages, detected automatically)
- **Cache-aware streaming**: `StreamingSession` for low-latency incremental transcription with selectable latency modes (e.g. `nvidia/nemotron-speech-streaming-en-0.6b`)
- **Hybrid RNN-T/CTC models**: decode with either head, offline or streaming (e.g. `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`)
- **Multilingual streaming**: language-ID prompt conditioning for `nvidia/nemotron-3.5-asr-streaming-0.6b` (40 language-locales), with a chosen target language or automatic language detection
- **Voice activity detection**: `EncDecClassificationModel` with NeMo-style post-processing (e.g. `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`)
- **Speaker diarization**: `SortformerEncLabelModel::diarize` gives per-frame speaker activity for up to four speakers (e.g. `nvidia/diar_streaming_sortformer_4spk-v2.1`)
- **Multi-talker transcription**: `MultiTalkerPipeline` combines streaming ASR with streaming Sortformer diarization (e.g. `nvidia/diar_streaming_sortformer_4spk-v2.1`). With a multitalker model (`nvidia/multitalker-parakeet-streaming-0.6b-v1`) it runs one speaker-conditioned ASR instance per speaker and transcribes overlapping speech; with an ordinary streaming model it attributes a single transcript to speakers
- **FastConformer encoder**, mel-spectrogram preprocessing, and SentencePiece tokenization matching NeMo
- **Weight formats**: `.nemo` archives (PyTorch checkpoints), MLX safetensors, and 8-bit quantized MLX weights

Text-to-speech support is planned.

## Requirements

- macOS on Apple Silicon (M1 or later)
- A recent stable Rust toolchain
- Xcode Command Line Tools (needed to build MLX)

## Installation

Add the crate from crates.io:

```toml
[dependencies]
mlx-voice = "0.2"
```

ASR support is behind the `asr` feature, which is on by default. The `hub` feature lets `from_pretrained` download models from the Hugging Face Hub. The `internals` feature exposes the building blocks (encoder layers, decoders, streaming caches, checkpoint parsing) for research and debugging; their API may change in any release.

## Getting models

Download models from Hugging Face with the bundled `xtask`. Files go to `models/<org>/<name>/` by default.

```bash
cargo xtask download nvidia/parakeet-ctc-0.6b
cargo xtask download nvidia/parakeet-rnnt-0.6b
cargo xtask download nvidia/parakeet-tdt-0.6b-v3 --file parakeet-tdt-0.6b-v3.nemo

# Gated models need a Hugging Face token (or set HF_TOKEN)
cargo xtask download nvidia/nemotron-speech-streaming-en-0.6b --token $HF_TOKEN
cargo xtask download nvidia/nemotron-3.5-asr-streaming-0.6b --file nemotron-3.5-asr-streaming-0.6b.nemo

# Download only specific files
cargo xtask download nvidia/parakeet-ctc-0.6b --file parakeet-ctc-0.6b.nemo
```

`from_pretrained` loads a model from:

- a `.nemo` file (a TAR archive, plain or gzip-compressed, with `model_config.yaml`, `model_weights.ckpt` and the tokenizer)
- a directory containing `model.safetensors`, a config, and `tokenizer.model`
- a Hugging Face Hub repository id such as `nvidia/parakeet-rnnt-0.6b` (with the `hub` feature). The repository's `.nemo` archive, or its safetensors weights, config and tokenizer, are downloaded to the Hugging Face cache. Gated repositories use `HF_TOKEN` or the token saved by `hf auth login`, or `HubRepo::new(id).token(..)`.

An existing path always wins over a repository id.

## Usage

### Offline transcription (CTC)

```rust
use mlx_voice::asr::EncDecCtcModel;

fn main() -> mlx_voice::Result<()> {
    let mut model = EncDecCtcModel::from_pretrained("models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo")?;
    let transcriptions = model.transcribe(&["audio.wav"])?;
    println!("{}", transcriptions[0]);
    Ok(())
}
```

`EncDecRnntModel` has the same `from_pretrained` / `transcribe` API. `transcribe` takes WAV paths or in-memory audio and returns a `Transcript` per input, with the text plus per-token and per-word timestamps in seconds. `TranscribeOptions` selects beam search, a language and the batch size:

```rust
use mlx_voice::asr::{EncDecRnntModel, TranscribeOptions};

let mut model = EncDecRnntModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?; // `hub` feature
let transcript = model.transcribe_one((&samples[..], 48_000))?; // resampled to 16 kHz
for word in &transcript.words {
    println!("{:6.2}-{:6.2} {}", word.start, word.end, word.text);
}

let options = TranscribeOptions::default().beam_search(4).batch_size(8);
let transcripts = model.transcribe_with(["a.wav", "b.wav"], &options)?;
```

`model.tokenizer()` turns token ids back into text.

### Streaming transcription

`model.stream(options)` starts a `StreamingSession`: push audio as it arrives, at any sample rate, and pull updates. Each update carries the text appended since the last one (`delta`) and the new tokens with their times. `StreamingOptions::mode` picks one of the checkpoint's streaming modes (its `att_context_size` list). The default is the checkpoint's own default, which NeMo lists first and which has the most lookahead, so it is the most accurate and has the highest latency. `StreamingMode::LowestLatency` and `StreamingMode::MaxLatency(duration)` choose by lookahead, and a number such as `.mode(3)` picks an entry by index. `model.streaming_modes()` gives each mode's latency, and `stream.streaming_config().mode_idx` the mode a session uses.

```rust
use mlx_voice::asr::{EncDecRnntModel, StreamingOptions};

fn main() -> mlx_voice::Result<()> {
    let mut model = EncDecRnntModel::from_pretrained(
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
    )?;
    let mut stream = model.stream(StreamingOptions::default())?;

    // `chunks` yields Vec<f32> from a 48 kHz microphone
    for chunk in chunks {
        stream.push_audio(&chunk, 48_000)?; // resampled to 16 kHz
        while let Some(update) = stream.next_update()? {
            print!("{}", update.delta);
        }
    }

    let transcript = stream.finish()?; // decodes the rest
    println!("\n{transcript}");
    Ok(())
}
```

For a whole file, `model.transcribe_streaming("audio.wav", options)?` returns the transcript the stream would produce. A session borrows its model; `StreamingSession::new(model, options)` gives one that owns it. Offline `transcribe` on the same model is unaffected by the session's mode.

NVIDIA's streaming CTC models are hybrid RNN-T/CTC models, such as `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`. They load as an `EncDecRnntModel` and decode with the RNN-T decoder by default. `DecoderType::Ctc` switches to their CTC decoder, which is faster and keeps no decoder state:

```rust
use mlx_voice::asr::{DecoderType, StreamingOptions, TranscribeOptions};

let ctc = TranscribeOptions::default().decoder_type(DecoderType::Ctc);
let transcripts = model.transcribe_with(["audio.wav"], &ctc)?;
let stream = model.stream(StreamingOptions::default().decoder_type(DecoderType::Ctc))?;
```

The `transcribe_rnnt`, `streaming` and `eval` examples take `--ctc`.

### Multilingual transcription

`nvidia/nemotron-3.5-asr-streaming-0.6b` conditions its output on a target language. It starts in `"auto"` mode, where the model detects the language and emits a tag such as `<de-DE>` after each sentence's terminal punctuation. Tags are stripped from the text by default but stay in the token ids:

```rust
let mut model = EncDecRnntModel::from_pretrained(
    "models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
)?;
model.set_language("de-DE")?; // or "de", "auto"; see model.languages()
let options = StreamingOptions::default().mode(3); // [56, 13]: 1.12 s chunks
let transcript = model.transcribe_streaming("audio.wav", options)?;
println!("{} ({:?})", transcript, transcript.language);
```

`StreamingOptions::strip_lang_tags(false)` keeps the tags in the text, and `Transcript::language` holds the last detected language. For offline transcription, `TranscribeOptions::language` and `strip_lang_tags` set the same per call. The `streaming`, `transcribe_rnnt` and `eval` examples take `--language` (`--target-lang` for `eval`).

### Multi-talker transcription

`MultiTalkerPipeline` pairs a streaming ASR model with a streaming Sortformer diarizer. The diarizer runs on the same audio chunks as the ASR model, so its speaker activity lines up frame by frame with the encoder output.

With `nvidia/multitalker-parakeet-streaming-0.6b-v1`, each of up to four speakers gets its own ASR instance on the shared audio. The diarizer output conditions each instance's encoder through learned speaker kernels. The instances run as one batched encoder call, so each speaker is transcribed separately even while people talk over each other.

```rust
use mlx_voice::asr::MultiTalkerPipeline;

fn main() -> mlx_voice::Result<()> {
    let mut pipeline = MultiTalkerPipeline::from_pretrained(
        "models/nvidia/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo",
        "models/nvidia/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo",
    )?;
    for segment in pipeline.transcribe("meeting.wav")? {
        println!("[{:.2}s] speaker_{}: {}", segment.start_time, segment.speaker, segment.text);
    }
    Ok(())
}
```

For live audio, `push_audio` / `next_update` / `finish` work as on a `StreamingSession`, and each update lists the words completed in it. `MultiTalkerConfig::streaming` sets the ASR streaming options. With an ordinary streaming model the pipeline runs in serial mode instead. Each word of a single transcript goes to the most active speaker, and speech that overlaps another speaker is lost.

## Examples

| Example | Description |
| --- | --- |
| `transcribe` | Offline transcription with a CTC model |
| `transcribe_rnnt` | Offline transcription with an RNN-T or TDT model (`--beam <size>` for beam search, RNN-T only) |
| `streaming` | Streaming transcription of a file, fed in small chunks |
| `live` | Live microphone transcription |
| `vad` | Voice activity detection on a file |
| `multi_talker` | Streaming transcription with speaker labels |
| `eval` | Transcribe the shared eval set to a hypotheses JSONL for WER scoring |

```bash
cargo run --release --example transcribe -- \
  --model models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo samples/2086-149220-0033.wav

cargo run --release --example transcribe_rnnt -- \
  --model models/nvidia/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b.nemo samples/2086-149220-0033.wav

cargo run --release --example streaming -- \
  --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
  samples/2086-149220-0033.wav

# List the streaming latency modes a model supports
cargo run --release --example streaming -- --model <model.nemo> --list-modes

cargo run --release --example live -- \
  --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo

# Speaker-attributed transcription (models auto-detected under models/)
cargo run --release --example multi_talker -- --output segments.json meeting.wav
```

Each example documents its options at the top of its source file in [`examples/`](examples/). Set `RUST_LOG=mlx_voice=debug` for debug logging.

## Architecture

```
audio (f32) → MelSpectrogram → ConformerEncoder → CtcDecoder | RnntJoint + PredictionNetwork
            → greedy / beam search → token IDs → SentencePiece tokenizer → text
```

- `src/core/`: `.nemo` archive loading, PyTorch pickle and safetensors weights, config parsing, quantization
- `src/asr/models/`: CTC, RNN-T, classification (VAD) and Sortformer models
- `src/asr/modules/`: mel preprocessing, FastConformer encoder, decoders and joint network
- `src/asr/streaming/`: encoder attention/convolution caches, mel and feature buffers, streaming config
- `src/asr/pipeline/`: the cache-aware RNN-T and multi-talker pipelines
- `src/asr/decoding/`: greedy and beam search decoding, TDT greedy decoding, streaming token merging

More detail is in [`docs/`](docs/), including [`lib-architecture.md`](docs/lib-architecture.md) and [`streaming-data-flow.md`](docs/streaming-data-flow.md).

## Development

```bash
cargo build
cargo test
cargo clippy --workspace --all-targets --all-features
```

Tests run one at a time: running several Metal workloads in parallel can crash the GPU command buffer, so `.cargo/config.toml` sets `RUST_TEST_THREADS=1`.

Most tests are `#[ignore]` by default because they need downloaded models in `models/`. Run them with:

```bash
cargo test -- --ignored
```

### Word error rate

The eval set (13 clips: 7 languages from Multilingual LibriSpeech, long-form TED talks, multi-speaker earnings calls and podcasts; about 88 minutes) lives in a private Hugging Face dataset shared with the mlx-models repo. Some clips can't be redistributed, so the audio is never committed here. Scoring is done by mlx-models' `samples/score.py`, so WER is normalized identically across ports:

```bash
cargo xtask download-eval          # to samples/eval/, at the pinned revision
cargo run --release --example eval -- \
  --model models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
(cd ../mlx-models && uv run samples/score.py ../mlx-voice-rs/hypotheses.jsonl)
```

The Python scripts in [`scripts/`](scripts/) compare intermediate outputs against the reference NeMo implementation.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attributions, including NVIDIA NeMo and the LibriSpeech sample in [`samples/`](samples/README.md).

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this project is licensed under Apache-2.0, without any additional terms or conditions.

Model weights are not part of this repository and are covered by their own licenses. Check each model card on Hugging Face.
