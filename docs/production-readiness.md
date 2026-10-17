# Production Readiness Plan

What `mlx-voice` needs to look and behave like a mature Rust library. It
covers releases, CI, coverage, docs, badges, repository hygiene and code
quality. Written against 0.2.0 (`3911936`, 2026-10-17). The "Where things
stand" table describes that baseline; the [checklist](#checklist) at the end
tracks progress.

## Where things stand

The code itself is in good shape. The public API was redesigned in 0.2.0
(`from_pretrained`, `Transcript`, one `StreamingSession`), internals are
behind a feature, `#![warn(missing_docs)]` is on, there is no `unsafe`, 57
types are `#[non_exhaustive]`, and the CHANGELOG is hand-written with a
migration guide. The project infrastructure around the code is what's
missing.

| Area | Today | Gap |
| --- | --- | --- |
| Releases | `cargo publish` by hand; 0.1.0, 0.1.1, 0.2.0 on crates.io | No git tags, no GitHub Releases, no automation, a long-lived API token on a laptop |
| CI | None (no `.github/` directory) | Nothing checks PRs |
| Docs | rustdoc builds locally with `-D warnings` | **docs.rs has failed for every version** (see [§3](#3-documentation)); the crates.io "Documentation" link goes to a failed build |
| Lints | `cargo fmt --check` passes | `cargo clippy --all-targets --all-features -- -D warnings` fails with 86 errors in the lib (mostly mechanical: 30 `unnecessary_cast`, 26 `needless_borrows_for_generic_args`, 16 `collapsible_if`) |
| Tests | 129 unit tests and 26 doctests pass locally (2.3 s); 32 integration tests, 25 of them `#[ignore]` because they need models | Nothing runs them automatically; no coverage numbers; the end-to-end tests need GBs of models |
| MSRV | Not declared | `mlx-rs` and `mlx-sys` need Rust 1.88, so the crate does too |
| Badges | None | |
| Community files | LICENSE, NOTICE, AUTHORS, CHANGELOG | No CONTRIBUTING, SECURITY, issue/PR templates; repo has no topics and a terse description |
| Supply chain | `Cargo.lock` committed | No `cargo-deny`/`cargo-audit` in CI, no Dependabot |

## Priorities

**P0: do before the next release**

1. Add CI and fix the clippy errors so it starts green ([§2](#2-continuous-integration))
2. Automate releases with crates.io Trusted Publishing ([§1](#1-release-automation))
3. Fix the docs story: host rustdoc on GitHub Pages and point `documentation` at it ([§3](#3-documentation))
4. Declare `rust-version` and test it ([§6.1](#61-msrv))
5. Add badges and missing package metadata ([§4](#4-readme-and-badges))

**P1: next few weeks**

6. Code coverage with `cargo-llvm-cov` and Codecov ([§5](#5-testing-and-coverage))
7. Small model-free fixture tests so CI exercises loading and streaming ([§5.2](#52-tests-that-run-without-downloaded-models))
8. A scheduled model test job on Apple Silicon hardware with a GPU ([§5.3](#53-model-tests))
9. `cargo-deny`, Dependabot, `cargo-semver-checks` ([§7](#7-supply-chain-and-semver))
10. CONTRIBUTING, SECURITY, issue templates, repo metadata ([§8](#8-repository-hygiene))

**P2: when there's time**

11. Fuzz the checkpoint parsers ([§6.4](#64-fuzzing-untrusted-model-files))
12. Load `.nemo` archives without unpacking them to a temp dir ([§6.5](#65-nemo-loading-without-extraction))
13. Benchmarks in the repo and a regression check ([§6.6](#66-benchmarks))
14. Upstream a docs.rs fix to `mlx-sys` ([§3.1](#31-docsrs))
15. Clean up `scripts/` and `docs/` ([§8.3](#83-repository-layout))

---

## 1. Release automation

### 1.1 Recommendation: tag-triggered publish with Trusted Publishing

Push a `vX.Y.Z` tag and a GitHub workflow publishes the crate and creates the
GitHub Release. It authenticates with
[crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing)
(OIDC), so there is no `CARGO_REGISTRY_TOKEN` secret to leak or rotate.

I recommend tag-triggered over [release-plz](https://release-plz.dev/) here.
release-plz is the popular fully automated option: it opens a "release PR" on
every merge, bumps the version from commit messages and generates the
changelog. But this repo's commits aren't Conventional Commits, and the
hand-written CHANGELOG (with migration guides) is better than anything
generated. A tag-triggered flow keeps that. Revisit release-plz if you move
to Conventional Commits.

**Release process after this change:**

1. Open a PR that dates the `## X.Y.Z (unreleased)` heading in
   `CHANGELOG.md` (`## X.Y.Z (YYYY-MM-DD)`) and, if the version wasn't bumped
   when that entry was started, bumps `version` in `Cargo.toml`.
   `cargo set-version` (already installed via `cargo-edit`) does the bump.
2. Merge it. CI runs as usual.
3. `git tag -s vX.Y.Z && git push origin vX.Y.Z`.
4. The workflow checks the tag matches `Cargo.toml` and the CHANGELOG entry
   is dated, runs the checks plus `cargo-semver-checks`, waits for approval
   on the `release` environment,
   publishes, and creates the GitHub Release with that version's CHANGELOG
   section as its notes.

`cargo-release` can do steps 1 and 3 in one command if you want that.

`cargo-semver-checks` fails a release that breaks the API without the
matching version bump: before 1.0, a breaking change needs a minor bump.

### 1.2 Setup

1. On crates.io: crate settings → Trusted Publishing → add GitHub repo
   `animaslabs/mlx-voice-rs`, workflow `release.yml`, environment `release`.
2. In GitHub: Settings → Environments → create `release` with yourself as a
   required reviewer and a deployment rule limited to `v*` tags. This is the
   manual "are you sure" gate.
3. Revoke the crates.io API token you use now once the first automated
   release succeeds.
4. Tag the commits that `v0.1.0`, `v0.1.1` and `v0.2.0` were published from,
   so the history is complete.

### 1.3 Workflow sketch

`cargo publish` verifies by building the packaged crate, which compiles MLX,
so the job runs on macOS.

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ["v*.*.*"]

permissions:
  contents: write   # create the GitHub Release

jobs:
  check:
    uses: ./.github/workflows/ci.yml   # make ci.yml `on: workflow_call` too

  publish:
    needs: check
    runs-on: macos-15
    environment: release
    permissions:
      contents: write
      id-token: write   # Trusted Publishing
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Tag matches Cargo.toml version
        run: |
          v=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name=="mlx-voice") | .version')
          test "v$v" = "$GITHUB_REF_NAME" || { echo "tag $GITHUB_REF_NAME != Cargo.toml $v"; exit 1; }
          grep -q "^## $v " CHANGELOG.md || { echo "no CHANGELOG entry for $v"; exit 1; }

      - uses: obi1kenobi/cargo-semver-checks-action@v2
        with: { package: mlx-voice }

      - uses: rust-lang/crates-io-auth-action@v1
        id: auth
      - run: cargo publish -p mlx-voice
        env:
          CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}

      - name: GitHub Release
        run: |
          v=${GITHUB_REF_NAME#v}
          awk -v v="$v" '$0 ~ "^## "v" " {f=1; next} /^## / && f {exit} f' CHANGELOG.md > notes.md
          gh release create "$GITHUB_REF_NAME" --title "$GITHUB_REF_NAME" --notes-file notes.md --verify-tag
        env:
          GH_TOKEN: ${{ github.token }}
```

Pin third-party actions to commit SHAs (see [§7](#7-supply-chain-and-semver)).

---

## 2. Continuous integration

### 2.1 Runners

The crate only builds on macOS (MLX with Metal), so every job except docs
deployment runs on macOS arm64.

- **GitHub-hosted `macos-15` (arm64)** runs build, clippy, docs and the unit
  tests, Metal included. The first CI runs confirmed the Metal tests pass
  there. The MLX build is slow, so caching matters ([§2.3](#23-jobs)).
- **Model tests need more.** They need GBs of models, so a self-hosted Mac
  mini (models kept on disk) or Blacksmith macOS runners (which `mlx-rs` uses)
  suit them better ([§5.3](#53-model-tests)).

### 2.2 Make single-threaded tests the default

Every contributor has to remember `--test-threads=1`, or Metal crashes. Set
it once in `.cargo/config.toml`:

```toml
[env]
RUST_TEST_THREADS = "1"
```

Then plain `cargo test` is safe, in CI and locally. The alternative is
`cargo-nextest` with a test group of `max-threads = 1`. It also gives better
output and retries, but it's another tool for contributors to install.

### 2.3 Jobs

Building MLX (the C++ library and its Metal shaders, in `mlx-sys`) takes 7–9
minutes on a 3-core hosted runner and is nearly all of CI's cost. The first
version of `ci.yml` ran five macOS jobs that each built it, and none of their
caches were usable (see below), so every run cost about 40 macOS-minutes.
[`ci.yml`](../.github/workflows/ci.yml) is now laid out around that build:

- **`fmt`** on Ubuntu, since rustfmt doesn't compile anything.
- **`check`**, one macOS job that builds MLX once and runs, in order:
  - clippy on all targets and features
  - clippy for each feature on its own (`cargo hack --each-feature`), to
    catch cfg mistakes
  - the tests
  - rustdoc

  Everything uses `-D warnings`.
- **`msrv`**: `cargo check` on 1.88. A second toolchain means a second MLX
  build, so it skips PRs and runs on pushes to `main` and in releases.

What makes the cache work:

- **Save from `main` only** (`save-if`), under one shared key. A PR can
  restore `main`'s cache but not another PR's, so caches saved from PRs
  only use up the 10 GB repository quota.
- **Don't cancel runs on `main`.** `cancel-in-progress` applies to PRs only.
  Before, each merge cancelled the previous `main` run, so `main` never
  saved a cache.
- **Cache `~/.mlx` too.** `mlx-sys` writes `mlx.metallib` to
  `~/.mlx/lib/<hash>/`, outside `target/`. A restored build skips the build
  script that would recreate it.
- **No global `RUSTFLAGS`.** It's part of every crate's fingerprint, so
  `pages.yml` and `release.yml` couldn't reuse the cache without the same
  value. `-D warnings` goes to clippy and rustdoc directly.
- **`CARGO_PROFILE_DEV_DEBUG=line-tables-only`**, the same in every workflow,
  for a smaller cache and faster links.

Restoring a cache doesn't rebuild MLX: cargo treats `mlx-sys` as fresh even
after rust-cache re-extracts the crate sources (checked locally with a
separate `CARGO_HOME`).

Still to do: require `fmt` and `check` in a branch ruleset on `main`, and add
a `deny` job once there's a `deny.toml` (P1).

---

## 3. Documentation

### 3.1 docs.rs

docs.rs has failed to build every version (all builds failed for 0.2.0). The
log shows three causes:

1. `mlx-sys`'s build script can't create its metallib directory in the docs.rs
   sandbox (`Permission denied`).
2. docs.rs builds on Linux with GCC. With
   `targets = ["aarch64-apple-darwin"]`, the C builds of `zstd-sys` and
   `sentencepiece-sys` get Apple-only flags (`-arch`,
   `-mmacosx-version-min`) that GCC rejects. (`zstd-sys` came from `zip`'s
   default features and is gone now that only its deflate codec is enabled.)
3. Even with those fixed, Metal can't build on Linux.

`mlx-rs` has the same problem and hosts its docs on GitHub Pages instead.
The usual fix is for each `-sys` crate's build script to skip the native
build when the `DOCS_RS` environment variable is set. That has to happen
upstream (`mlx-sys`, `sentencepiece-sys`), so it's a long-term item. Open an issue or PR on `oxideai/mlx-rs` for `mlx-sys` first,
since it's the hard blocker.

### 3.2 Host rustdoc on GitHub Pages now

- Add a `pages.yml` workflow that runs on pushes to `main` and on release
  tags. It builds `cargo doc --no-deps --all-features` on macOS, adds an
  `index.html` redirect to `mlx_voice/`, and deploys with
  `actions/deploy-pages`. `mlx-rs`'s `publish-docs.yml` is a working template.
- Building with `--all-features` documents the `internals` API too. Use
  `#![cfg_attr(docsrs, feature(doc_cfg))]` with `--cfg docsrs` (nightly) so
  each item shows the feature it needs ("Available on crate feature
  `internals` only"). That makes the "no stability guarantees" boundary
  visible. Add `#[cfg_attr(docsrs, doc(cfg(feature = "...")))]` to the
  `internal_mod!`/`internal_fn!` macros and the `hub` items.
- Set `documentation = "https://animaslabs.github.io/mlx-voice-rs/mlx_voice/"`
  in `Cargo.toml`, so crates.io stops linking to the failed docs.rs page. To
  keep docs for past releases, deploy each tag to `/vX.Y.Z/` with `/latest/`
  pointing at the newest release, and `main` to `/dev/`.

### 3.3 Docs content

- **Turn on more rustdoc lints:** `rustdoc::broken_intra_doc_links`,
  `rustdoc::private_intra_doc_links` and `rustdoc::missing_crate_level_docs`.
  `missing_docs` already exists; CI's `-D warnings` makes it enforced.
- **Crate-level docs** should cover what the README covers: supported models,
  feature flags (a table), threading rules (see [§6.3](#63-thread-safety)),
  and a pointer to each main entry point (offline, streaming, multilingual,
  multi-talker, VAD).
- **Examples on every main public type:** `EncDecRnntModel`,
  `EncDecCtcModel`, `StreamingSession`, `MultiTalkerPipeline`,
  `EncDecClassificationModel`, `TranscribeOptions`, `StreamingOptions`,
  `StreamingMode`, `LoadOptions`. Use `no_run` where a model is needed. That
  still compiles them, so they can't rot. There are 26 doctests today.
- **Error docs:** an `# Errors` section on fallible public functions
  (`clippy::missing_errors_doc` finds them) and a `# Panics` section where
  needed.
- **Separate user docs from design notes.** `docs/` mixes reference material
  (`streaming-data-flow.md`, `lib-architecture.md`) with point-in-time
  analysis (`api-review.md`, written for 0.1.1; `mlx-audio-gap-analysis.md`;
  `py-comparison.md`). Move the analysis into `docs/design/` or delete what's
  done, so readers don't mistake it for current docs.
- **Optional (P2): an mdBook guide** on the same Pages site. Topics: choosing
  a model, streaming latency modes, multi-talker setup, converting checkpoints
  to safetensors, and performance tuning (`--dtype bfloat16`, quantized
  weights).

---

## 4. README and badges

### 4.1 Badges

```markdown
[![crates.io](https://img.shields.io/crates/v/mlx-voice.svg)](https://crates.io/crates/mlx-voice)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://animaslabs.github.io/mlx-voice-rs/mlx_voice/)
[![CI](https://github.com/animaslabs/mlx-voice-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/animaslabs/mlx-voice-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/animaslabs/mlx-voice-rs/graph/badge.svg)](https://codecov.io/gh/animaslabs/mlx-voice-rs)
[![MSRV](https://img.shields.io/crates/msrv/mlx-voice.svg)](https://crates.io/crates/mlx-voice)
[![License](https://img.shields.io/crates/l/mlx-voice.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)
```

Use the docs.rs badge (`https://docs.rs/mlx-voice/badge.svg`) once docs.rs
builds work. Until then it would show "failed". The MSRV badge reads
`rust-version` from crates.io, so it appears after the next release.

### 4.2 README content to add

- **Model support matrix:** model, type, offline/streaming, decoders, and
  tested status, plus the WER and real-time factor from
  `mlx-models/bench/bench.py` and the chip they were measured on. Numbers
  against `parakeet-mlx`/NeMo are the best reason to pick this crate.
- **Platform support:** macOS on Apple Silicon only, the minimum macOS
  version (the build sets `-mmacosx-version-min=11.0`; check what MLX
  actually needs), and MSRV.
- **Feature flags table:** `asr` (default), `hub` and `internals`, each with
  one line.
- **Stability:** what semver covers before 1.0 (minor = breaking), and that
  `internals` has no stability guarantees.
- **Shorter Development section:** the clippy and test commands, then a link
  to CONTRIBUTING.md.

### 4.3 `Cargo.toml` metadata

```toml
[package]
rust-version = "1.88"
documentation = "https://animaslabs.github.io/mlx-voice-rs/mlx_voice/"
homepage = "https://github.com/animaslabs/mlx-voice-rs"

[package.metadata.docs.rs]
targets = ["aarch64-apple-darwin"]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

`include` is already tight: `cargo package --list` shows 60 files, with no
samples, scripts or models.

---

## 5. Testing and coverage

### 5.1 Coverage with `cargo-llvm-cov`

[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) works on macOS
and is the standard choice.

```yaml
  coverage:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: llvm-tools-preview }
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@cargo-llvm-cov
      - run: cargo llvm-cov --package mlx-voice --all-features --lcov --output-path lcov.info
      - uses: codecov/codecov-action@v5
        with:
          files: lcov.info
          token: ${{ secrets.CODECOV_TOKEN }}
          flags: unit
```

Two caveats:

- **The number will understate real coverage.** The tests that cover the
  models end to end (`tests/integration.rs`, `tests/multitalker.rs`) are
  `#[ignore]` and need downloaded models. Upload them from the model job
  ([§5.3](#53-model-tests)) under a second flag (`flags: models`), and Codecov
  merges the two.
- **Don't gate merges on a coverage percentage yet.** Make the Codecov status
  checks informational (`codecov.yml`: `informational: true`) and watch
  patch coverage on PRs. Add a threshold once the number is stable.

### 5.2 Tests that run without downloaded models

This is the most valuable testing work. Today, CI could only run unit tests.
Checkpoint loading, the encoder forward pass, streaming caches and the
decoders would only be checked on your machine.

- **Tiny synthetic checkpoints.** Build a minimal config (2 encoder layers,
  `d_model` 32, a few mel bins, a 16-token vocabulary) and random weights,
  saved as safetensors plus a tiny SentencePiece model, in
  `tests/fixtures/` (a few hundred KB). Then test
  `from_pretrained` → `transcribe` → `stream` → `finish` for the CTC, RNN-T,
  hybrid and multilingual-prompt shapes. The output is nonsense, but the
  shapes, cache bookkeeping, timestamps and error paths get exercised. A
  small Python script in `scripts/` that builds the fixtures with NeMo keeps
  them reproducible.
- **Streaming equals offline.** With the fixture model, feeding audio in
  chunks of different sizes should give the same tokens as one big
  `push_audio`. This catches buffering and resampling bugs.
- **Golden outputs for model-free components.** Commit NeMo's mel features
  and resampler output for `samples/2086-149220-0033.wav` (a small `.npy` or
  JSON) and compare within a tolerance. The mel front end is where silent
  drift shows up first.
- **Error-path tests:** truncated `.nemo`, a missing tokenizer, a wrong
  sample rate, an empty audio buffer, and an unknown streaming mode. All
  should return `Err`, never panic.
- **Property tests (`proptest`)** for `ctc_collapse`, the token-merging code
  and `StreamResampler` (output length vs input length and rate ratio).

### 5.3 Model tests

Run the `#[ignore]` tests on a schedule (nightly plus manual dispatch), not
on every PR. They need GBs of models and a real GPU.

- Use a self-hosted Apple Silicon runner, or Blacksmith, with the models
  kept on disk. Six models don't fit comfortably in the 10 GB Actions cache.
  If you use the cache anyway, pin the HF revisions and key the cache on them.
- Keep the gated models' `HF_TOKEN` as a secret scoped to that environment.
- Add a WER regression check. `cargo xtask download-eval`, run `examples/eval`
  on one or two models, score with `mlx-models/samples/score.py`, and fail if
  WER rises more than a set margin above a committed baseline. That's what
  matters to users and no unit test catches it.

---

## 6. Code quality

### 6.1 MSRV

`mlx-rs` 0.32 and `mlx-sys` 0.6 declare `rust-version = "1.88"`, so 1.88 is
the floor. `zip`'s default `aes-crypto` feature pulled in `aes` 0.9, which
needs 1.89. PyTorch checkpoints are stored uncompressed, so `zip` now enables
only its deflate codec, and the crate builds on 1.88. `rust-version = "1.88"`
is set and the CI `msrv` job checks it. The policy for
the README: "MSRV is N; raising it is a minor-version change."

### 6.2 Lints in `Cargo.toml`

Move lint config from `lib.rs` attributes to a `[lints]` table (stable since
1.74) so it applies to every target, and tighten it:

```toml
[lints.rust]
unsafe_code = "forbid"          # there is none today; keep it that way
missing_docs = "warn"
missing_debug_implementations = "warn"
unreachable_pub = "warn"        # catches accidental pub in pub(crate) modules
unexpected_cfgs = { level = "warn", check-cfg = ["cfg(docsrs)"] }

[lints.clippy]
unwrap_used = "warn"            # about 10 in library code outside tests
expect_used = "warn"
dbg_macro = "warn"
todo = "warn"
print_stdout = "warn"           # libraries log through tracing
missing_errors_doc = "warn"
missing_panics_doc = "warn"
```

Allow `unwrap_used`/`expect_used` in tests with `#[cfg_attr(test,
allow(...))]` or `clippy.toml`'s `allow-unwrap-in-tests = true`.

### 6.3 Thread safety

Tests must run single-threaded because concurrent Metal work crashes. A
server that transcribes several requests at once will hit the same crash. To
prevent that:

- Decide on the contract and state it in the crate docs. For example:
  "models are `Send` but not `Sync`; run one model per thread, and don't run
  models on several threads at once without a lock."
- Add compile-time assertions (`static_assertions::assert_impl_all!` /
  `assert_not_impl_any!`) for `Send`/`Sync` on the public model and session
  types, so that contract can't change by accident.
- If concurrent use from different threads is unsafe even with separate
  models, enforce it in the library (a process-wide lock around evaluation).
  Don't leave it to documentation alone.
- Add an `examples/server.rs`-style example showing the supported pattern: a
  worker thread that owns the model, fed through a channel.

### 6.4 Fuzzing untrusted model files

`from_pretrained("owner/name")` downloads and parses files from anyone on
the Hub: TAR/gzip, a PyTorch pickle parser, safetensors, YAML. The parsers
are safe Rust, but malformed input can still panic, allocate without bound
or loop forever.

- Add `cargo-fuzz` targets for the pickle parser (`core/pytorch.rs`), config
  parsing and `.nemo` opening. Run them for a few minutes in a weekly
  workflow.
- Bound allocations by declared tensor sizes, and check them against the
  bytes actually present.
- Say in SECURITY.md that loading untrusted checkpoints is supported, and
  which guarantees hold (no code execution, unlike Python pickle).

### 6.5 `.nemo` loading without extraction

`NemoCheckpoint::open` unpacks the whole archive into a `tempfile::tempdir()`
before reading it (`src/core/checkpoint.rs:62`). For a 2.5 GB checkpoint
that doubles the disk I/O, needs that much free space in `$TMPDIR`, and
slows every cold start. Reading the TAR entries in place would remove the
temp copy and the runtime `tempfile` dependency: read the config and
tokenizer into memory, and stream `model_weights.ckpt`. Check that there is
a cleanup path on panic in the meantime (`TempDir` handles drop, but not a
`SIGKILL`).

### 6.6 Benchmarks

Speed comparisons live in `mlx-models/bench/bench.py`. Add a small
`criterion` (or `divan`) suite here too for the hot paths that don't need a
full model: mel spectrogram, resampling, CTC collapse, and RNN-T greedy
decoding with a fixture model. Run it nightly on the self-hosted runner and
watch for regressions. Publish real-time factors in the README.

### 6.7 Small cleanups

- Remove `tokio` from `[dev-dependencies]`. No example or test uses it, and
  `features = ["full"]` adds build time.
- `clippy::too_many_arguments` shows up three times. An options or
  parameters struct usually reads better.
- Watch the `internals` API surface. `cargo public-api` (from
  `cargo-public-api`) can diff the public items between releases, with and
  without `internals`, so nothing leaks back into the stable API after the
  work done in #20.

---

## 7. Supply chain and semver

- **`cargo-deny`** with a `deny.toml`: allow-list licenses (Apache-2.0, MIT,
  BSD, Unicode-3.0, Zlib, ...), deny yanked crates and unknown registries or
  git sources, and check RustSec advisories. `sentencepiece-sys` and
  `mlx-sys` vendor C++. Their licenses should appear in NOTICE if you ever
  ship binaries.
- **Dependabot** (`.github/dependabot.yml`): weekly updates for the `cargo`
  and `github-actions` ecosystems, grouped so you get one PR per ecosystem.
  The commit "Update dependencies to fix security advisories" suggests this
  is manual today.
- **Pin actions to commit SHAs** (`uses: actions/checkout@<sha> # v4`), and
  let Dependabot update them. Set `permissions: contents: read` at the top of
  every workflow and widen it per job.
- **`cargo-semver-checks`** in the release workflow (above), and as a
  non-blocking PR job comparing against the last published version. Before
  1.0 it reports breaking changes so you bump the minor version; after 1.0 it
  blocks accidental breaks.
- **Lockfile:** `Cargo.lock` is committed (right for the examples and
  xtask). Downstream users resolve their own, so the CI `msrv` and
  `features` jobs catch problems with the newest compatible versions. A
  weekly job that runs `cargo update && cargo test` warns early.

---

## 8. Repository hygiene

### 8.1 Community files

- **`CONTRIBUTING.md`:** setup (Xcode CLT, Rust ≥ MSRV), getting models via
  `cargo xtask download`, running unit vs model tests, the NeMo comparison
  scripts, the code style (much of it is in AGENTS.md already), and the
  release process from [§1.1](#11-recommendation-tag-triggered-publish-with-trusted-publishing).
- **`SECURITY.md`:** how to report privately (enable GitHub private
  vulnerability reporting), supported versions, and the untrusted-checkpoint
  stance from [§6.4](#64-fuzzing-untrusted-model-files).
- **`CODE_OF_CONDUCT.md`:** Contributor Covenant 2.1.
- **`.github/ISSUE_TEMPLATE/`:** a bug report form asking for the chip,
  macOS version, crate version, model id and a minimal repro, plus a feature
  or model request form. Add `.github/pull_request_template.md` with a
  checklist (CHANGELOG entry, tests, docs).
- **CHANGELOG:** keep a `## X.Y.Z (unreleased)` section at the top that PRs
  add to, so a release only has to date it. The release workflow rejects an
  undated heading.

### 8.2 GitHub repository settings

- **Description:** "NeMo models for MLX in rust" →
  "NVIDIA NeMo speech recognition (Parakeet, Nemotron, Sortformer) on Apple
  Silicon via MLX, in Rust".
- **Topics:** `rust`, `asr`, `speech-recognition`, `mlx`, `apple-silicon`,
  `nemo`, `parakeet`, `streaming`, `diarization`.
- **Website:** the docs URL.
- **Settings:** enable Discussions for usage questions, auto-delete merged
  branches, squash-merge only (matching the current history), and private
  vulnerability reporting.

### 8.3 Repository layout

- **`scripts/`** holds about 85 one-off Python debugging and tracing scripts
  (`debug_*`, `trace_*`, `compare_*`). They're excluded from the crate
  package, but they make the repo look unfinished and are hard to navigate.
  Keep the few still useful for parity checks in `scripts/parity/` with a
  README, and delete the rest (they stay in git history) or move them to
  `mlx-models`.
- **`examples/`** mixes user examples with seven `internals` debugging tools
  (`debug_*`, `compare_encoder`, `mel_test`, `trace_streaming`). Move the
  debugging tools to `xtask` or a `tools/` workspace member, so `examples/`
  only shows how to use the library.
- **`AGENTS.md` and `CLAUDE.md`** overlap. Make one the source and have the
  other point to it, so build instructions don't drift (for example,
  `AGENTS.md` doesn't mention the `internals` feature).

---

## Checklist

**P0**
- [x] Fix the 86 clippy errors (`cargo clippy --fix`, then the
  `too_many_arguments` cases)
- [x] `.cargo/config.toml`: `[env] RUST_TEST_THREADS = "1"`
- [x] `.github/workflows/ci.yml`: fmt, clippy, test, features, msrv, docs
- [ ] `deny` job (needs a `deny.toml`, see P1)
- [ ] Branch ruleset on `main` requiring `fmt` and `check`
- [x] `.github/workflows/release.yml`
- [ ] crates.io Trusted Publishing and the `release` environment (repo
  settings)
- [ ] Tag `v0.1.0`, `v0.1.1`, `v0.2.0` on the commits they were published from
- [ ] Revoke the local crates.io token
- [x] `pages.yml` for rustdoc; set `documentation`, `homepage`,
  `rust-version`
- [ ] Enable Pages (source: GitHub Actions) and allow `v*` tags on the
  `github-pages` environment
- [ ] README badges, feature table, platform and MSRV, stability section

**P1**
- [ ] `cargo-llvm-cov` plus Codecov (informational)
- [ ] Tiny fixture models and model-free tests (loading, streaming equals
  offline, golden mel, error paths)
- [ ] Nightly model test, coverage and WER job on Apple Silicon
- [ ] `deny.toml`, Dependabot, SHA-pinned actions,
  `cargo-semver-checks`
- [ ] `[lints]` table; reduce `unwrap`/`expect` in library code
- [ ] Thread-safety contract documented and asserted
- [ ] `doc_cfg` feature badges; `# Errors` sections; examples on main types
- [ ] CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue and PR templates
- [ ] Repo description, topics, website, settings

**P2**
- [ ] `cargo-fuzz` targets for checkpoint parsing
- [ ] Load `.nemo` without extracting to a temp dir
- [ ] criterion benchmarks plus nightly regression run
- [ ] Upstream `DOCS_RS` build skip in `mlx-sys`; switch to docs.rs when it
  works
- [ ] Prune `scripts/`, move debugging examples, split `docs/design/`
- [ ] Remove the unused `tokio` dev-dependency
- [ ] mdBook guide
