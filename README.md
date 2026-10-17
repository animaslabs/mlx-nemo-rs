# mlx-voice

A Rust implementation of NVIDIA NeMo speech models for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) via [`mlx-rs`](https://github.com/oxideai/mlx-rs).

mlx-voice loads NeMo checkpoints (`.nemo` archives) or converted MLX safetensors directly and runs them on the GPU through Metal, with no Python at inference time.

## Features

- **CTC ASR**: `EncDecCTCModel` (e.g. `nvidia/parakeet-ctc-0.6b`)
- **RNN-T / Transducer ASR**: `EncDecRNNTModel` with greedy or beam search decoding (e.g. `nvidia/parakeet-rnnt-0.6b`)
- **Cache-aware streaming**: `CacheAwareRNNTPipeline` for low-latency incremental transcription with selectable latency modes (e.g. `nvidia/nemotron-speech-streaming-en-0.6b`)
- **Multilingual streaming**: language-ID prompt conditioning for `nvidia/nemotron-3.5-asr-streaming-0.6b` (40 language-locales), with a chosen target language or automatic language detection
- **Voice activity detection**: `EncDecClassificationModel` with NeMo-style post-processing (e.g. `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`)
- **Multi-talker transcription**: `MultiTalkerPipeline` combines streaming ASR with streaming Sortformer diarization (e.g. `nvidia/diar_streaming_sortformer_4spk-v2.1`). With a multitalker model (`nvidia/multitalker-parakeet-streaming-0.6b-v1`) it runs one speaker-conditioned ASR instance per speaker and transcribes overlapping speech; with an ordinary streaming model it attributes a single transcript to speakers
- **FastConformer encoder**, mel-spectrogram preprocessing, and SentencePiece tokenization matching NeMo
- **Weight formats**: `.nemo` archives (PyTorch checkpoints), MLX safetensors, and 8-bit quantized MLX weights

TTS and LLM support are planned.

## Requirements

- macOS on Apple Silicon (M1 or later)
- A recent stable Rust toolchain
- Xcode Command Line Tools (needed to build MLX)

## Installation

Add the crate from crates.io:

```toml
[dependencies]
mlx-voice = "0.1"
```

ASR support is behind the `asr` feature, which is on by default. The `hub` feature lets `from_pretrained` download models from the Hugging Face Hub. The `internals` feature exposes the building blocks (encoder layers, decoders, streaming caches, checkpoint parsing) for research and debugging; their API may change in any release.

## Getting models

Download models from Hugging Face with the bundled `xtask`. Files go to `models/<org>/<name>/` by default.

```bash
cargo xtask download nvidia/parakeet-ctc-0.6b
cargo xtask download nvidia/parakeet-rnnt-0.6b

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

An existing path always wins over a repository id. `from_safetensors` loads a lone `.safetensors` file with explicit config and tokenizer paths.

## Usage

### Offline transcription (CTC)

```rust
use mlx_voice::asr::EncDecCTCModel;

fn main() -> mlx_voice::Result<()> {
    let mut model = EncDecCTCModel::from_pretrained("models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo")?;
    let transcriptions = model.transcribe(&["audio.wav"])?;
    println!("{}", transcriptions[0]);
    Ok(())
}
```

`EncDecRNNTModel` has the same `from_pretrained` / `transcribe` API. `transcribe` takes WAV paths or in-memory audio and returns a `Transcript` per input, with the text plus per-token and per-word timestamps in seconds. `TranscribeOptions` selects beam search, a language and the batch size:

```rust
use mlx_voice::asr::{EncDecRNNTModel, TranscribeOptions};

let mut model = EncDecRNNTModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?; // `hub` feature
let transcript = model.transcribe_one((&samples[..], 48_000))?; // resampled to 16 kHz
for word in &transcript.words {
    println!("{:6.2}-{:6.2} {}", word.start, word.end, word.text);
}

let options = TranscribeOptions::default().beam_search(4).batch_size(8);
let transcripts = model.transcribe_with(["a.wav", "b.wav"], &options)?;
```

`model.tokenizer()` turns token ids back into text.

### Streaming transcription

`CacheAwareRNNTPipeline` takes audio incrementally and emits partial results as they become available. The second argument to `from_pretrained` picks the streaming mode, an index into the checkpoint's `att_context_size` list. NeMo lists modes from most to least lookahead, so `0`, the model's default, has the highest latency and accuracy; `pipeline.streaming_modes()` gives each mode's latency.

```rust
use mlx_voice::asr::CacheAwareRNNTPipeline;

fn main() -> mlx_voice::Result<()> {
    let mut pipeline = CacheAwareRNNTPipeline::from_pretrained(
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
        0,
    )?;

    // `chunks` yields Vec<f32> at pipeline.sample_rate()
    for chunk in chunks {
        pipeline.add_audio(&chunk);
        while let Some(output) = pipeline.transcribe_step()? {
            println!("{}", output.text);
        }
    }

    pipeline.end_stream();
    while let Some(output) = pipeline.transcribe_step()? {
        if output.is_final {
            println!("Final: {}", output.text);
        }
    }
    Ok(())
}
```

For a whole file, `pipeline.set_audio_file("audio.wav")?` followed by `pipeline.run()?` returns the full transcript.

### Multilingual transcription

`nvidia/nemotron-3.5-asr-streaming-0.6b` conditions its output on a target language. It starts in `"auto"` mode, where the model detects the language and emits a tag such as `<de-DE>` after each sentence's terminal punctuation. Tags are stripped from the text by default but stay in the token ids:

```rust
let mut pipeline = CacheAwareRNNTPipeline::from_pretrained(
    "models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    3, // [56, 13]: 1.12 s chunks
)?;
pipeline.set_language("de-DE")?; // or "de", "auto"; see pipeline.languages()
pipeline.set_audio_file("audio.wav")?;
let text = pipeline.run()?;
```

`set_strip_lang_tags(false)` keeps the tags in the text, and `detected_language(&output.tokens)` returns the last detected language. For offline transcription with `EncDecRNNTModel`, `TranscribeOptions::language` and `strip_lang_tags` set the same per call, and `Transcript::language` holds the detected language. The `streaming`, `transcribe_rnnt` and `eval` examples take `--language` (`--target-lang` for `eval`).

### Multi-talker transcription

`MultiTalkerPipeline` pairs a streaming ASR model with a streaming Sortformer diarizer. The diarizer runs on the same audio chunks as the ASR model, so its speaker activity lines up frame by frame with the encoder output.

With `nvidia/multitalker-parakeet-streaming-0.6b-v1`, each of up to four speakers gets its own ASR instance on the shared audio. The diarizer output conditions each instance's encoder through learned speaker kernels. The instances run as one batched encoder call, so each speaker is transcribed separately even while people talk over each other.

```rust
use mlx_voice::asr::MultiTalkerPipeline;

fn main() -> mlx_voice::Result<()> {
    let mut pipeline = MultiTalkerPipeline::from_pretrained(
        "models/nvidia/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo",
        "models/nvidia/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo",
        0,
    )?;
    pipeline.set_audio_file("meeting.wav")?;
    while let Some(output) = pipeline.transcribe_step()? {
        for word in &output.new_words {
            println!("[{:.2}s] speaker_{}: {}", word.start_time, word.speaker, word.text);
        }
    }
    for segment in pipeline.segments() {
        println!("speaker_{}: {}", segment.speaker, segment.text);
    }
    Ok(())
}
```

With an ordinary streaming model the pipeline runs in serial mode instead. Each word of a single transcript goes to the most active speaker, and speech that overlaps another speaker is lost.

## Examples

| Example | Description |
| --- | --- |
| `transcribe` | Offline transcription with a CTC model |
| `transcribe_rnnt` | Offline transcription with an RNN-T model (`--beam <size>` for beam search) |
| `streaming` | Streaming transcription of a file, fed in small chunks |
| `live` / `live_pipeline` | Live microphone transcription |
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
audio (f32) → MelSpectrogram → ConformerEncoder → CTCDecoder | RNNTJoint + PredictionNetwork
            → greedy / beam search → token IDs → SentencePiece tokenizer → text
```

- `src/core/`: `.nemo` archive loading, PyTorch pickle and safetensors weights, config parsing, quantization
- `src/asr/models/`: CTC, RNN-T, classification (VAD) and Sortformer models
- `src/asr/modules/`: mel preprocessing, FastConformer encoder, decoders and joint network
- `src/asr/streaming/`: encoder attention/convolution caches, mel and feature buffers, streaming config
- `src/asr/pipeline/`: the cache-aware RNN-T and multi-talker pipelines
- `src/asr/decoding/`: greedy and beam search decoding, streaming token merging

More detail is in [`docs/`](docs/), including [`lib-architecture.md`](docs/lib-architecture.md) and [`streaming-data-flow.md`](docs/streaming-data-flow.md).

## Development

```bash
cargo build
cargo test -- --test-threads=1
```

Run tests with `--test-threads=1`. Running several Metal workloads in parallel can crash the GPU command buffer.

Most tests are `#[ignore]` by default because they need downloaded models in `models/`. Run them with:

```bash
cargo test -- --ignored --test-threads=1
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
