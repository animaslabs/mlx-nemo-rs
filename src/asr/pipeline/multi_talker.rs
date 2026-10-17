//! Multi-talker streaming pipeline.
//!
//! Combines a cache-aware streaming RNN-T model with a streaming SortFormer
//! diarizer to produce speaker-attributed transcripts. The diarizer runs on
//! the same mel chunk as the ASR model, so both share one 80 ms frame grid.
//!
//! Two modes are selected from the ASR checkpoint:
//!
//! - **Parallel** (multitalker models such as
//!   `nvidia/multitalker-parakeet-streaming-0.6b-v1`): one ASR instance per
//!   speaker. The instances run as a single batched encoder call, each
//!   conditioned on its speaker's diarizer activity through the encoder's
//!   speaker kernels, so overlapping speech is transcribed for every speaker.
//! - **Serial** (ordinary streaming ASR models): a single transcript whose
//!   words are attributed to the most active speaker over each word's frames.
//!   Overlapping speech can't be separated in this mode.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # use mlx_voice::asr::MultiTalkerPipeline;
//! # let mic_chunks: Vec<Vec<f32>> = Vec::new();
//! let mut pipeline = MultiTalkerPipeline::from_pretrained(
//!     "multitalker-parakeet-streaming-0.6b-v1.nemo",
//!     "diar_streaming_sortformer_4spk-v2.1.nemo",
//! )?;
//!
//! // A whole file
//! for segment in pipeline.transcribe("meeting.wav")? {
//!     println!("[Speaker {}] {}", segment.speaker, segment.text);
//! }
//!
//! // Live audio
//! pipeline.reset()?;
//! for chunk in &mic_chunks {
//!     pipeline.push_audio(chunk, 16_000)?;
//!     while let Some(output) = pipeline.next_update()? {
//!         for word in &output.new_words {
//!             println!("[{:.2}s speaker {}] {}", word.start_time, word.speaker, word.text);
//!         }
//!     }
//! }
//! let segments = pipeline.finish()?;
//! # Ok(())
//! # }
//! ```

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

use super::session::{StreamingOptions, StreamingSession};
use crate::asr::Tokenizer;
use crate::asr::models::{EncDecRnntModel, SortformerEncLabelModel};
use crate::asr::modules::DecoderState;
use crate::asr::modules::SpeakerTargets;
use crate::asr::streaming::{EncoderCache, FeatureBuffer, StreamingSortformerState};
use crate::asr::transcript::{AudioInput, DecoderType};
use crate::core::ArrayExt;
use crate::core::error::{Error, Result};
use crate::core::loading::LoadOptions;
use crate::core::source::ModelSource;

/// Speaker activity threshold.
const ACTIVITY_THRESHOLD: f32 = 0.5;

/// Multi-talker pipeline settings.
///
/// Defaults follow NeMo's `MultitalkerTranscriptionConfig`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MultiTalkerConfig {
    /// Binarize diarizer outputs at 0.5 before using them as speaker targets
    pub binary_diar_preds: bool,
    /// If at most one speaker is active in a chunk, transcribe it all as speaker 0
    pub single_speaker_mode: bool,
    /// Only decode speakers that were active within this many recent chunks
    /// (0 decodes every speaker on every chunk)
    pub cache_gating_chunks: usize,
    /// Start a new segment when a speaker pauses longer than this (seconds)
    pub sent_break_sec: f32,
    /// Diarizer FIFO length in encoder frames
    pub diar_fifo_len: usize,
    /// Diarizer speaker cache update period in encoder frames
    pub diar_spkcache_update_period: usize,
    /// ASR streaming options (mode, thresholds, punctuation suppression)
    pub streaming: StreamingOptions,
    /// How strictly to load both models' weights
    pub load: LoadOptions,
}

impl Default for MultiTalkerConfig {
    fn default() -> Self {
        Self {
            binary_diar_preds: true,
            single_speaker_mode: false,
            cache_gating_chunks: 2,
            sent_break_sec: 1.0,
            diar_fifo_len: 188,
            diar_spkcache_update_period: 144,
            streaming: StreamingOptions::default(),
            load: LoadOptions::default(),
        }
    }
}

/// A word attributed to a speaker.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeakerWord {
    /// Speaker index (0-based)
    pub speaker: usize,
    /// Word text
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// A run of words from one speaker (a SegLST entry).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeakerSegment {
    /// Speaker index (0-based, up to max_num_spks - 1)
    pub speaker: usize,
    /// Text content for this segment
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// Output from a multi-talker transcription step.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MultiTalkerOutput {
    /// Words completed during this step, ordered by start time
    pub new_words: Vec<SpeakerWord>,
    /// Speaker activity for this chunk's encoder frames `[frames][speakers]`
    pub speaker_activity: Vec<Vec<f32>>,
    /// Whether this is the final output
    pub is_final: bool,
}

/// How the ASR model is shared between speakers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultiTalkerMode {
    /// One speaker-conditioned ASR instance per speaker
    Parallel,
    /// A single ASR instance whose words are attributed afterwards
    Serial,
}

/// Decoding state for one ASR instance.
struct Instance {
    decoder_state: DecoderState,
    last_token: Option<usize>,
    /// Tokens of the word in progress, with their encoder frame
    pending: Vec<(usize, usize)>,
}

/// Multi-talker streaming pipeline.
pub struct MultiTalkerPipeline {
    /// ASR model, audio buffering and mel streaming
    asr: StreamingSession<EncDecRnntModel>,
    /// Diarization model
    diar_model: SortformerEncLabelModel,
    /// Diarization streaming state
    diar_state: StreamingSortformerState,
    config: MultiTalkerConfig,
    mode: MultiTalkerMode,
    num_spks: usize,
    /// Encoder cache with one batch row per instance
    encoder_cache: EncoderCache,
    instances: Vec<Instance>,
    /// Diarizer predictions per encoder frame [frame][speaker]
    diar_preds: Vec<Vec<f32>>,
    /// Encoder frames processed so far
    frame_offset: usize,
    /// Seconds per encoder frame
    frame_sec: f32,
    /// Chunks since each speaker was last active
    chunks_since_active: Vec<usize>,
    /// All completed words
    words: Vec<SpeakerWord>,
}

impl std::fmt::Debug for MultiTalkerPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiTalkerPipeline")
            .field("mode", &self.mode)
            .field("num_spks", &self.num_spks)
            .field("asr", &self.asr)
            .field("diar_model", &self.diar_model)
            .field("words", &self.words.len())
            .finish_non_exhaustive()
    }
}

impl MultiTalkerPipeline {
    /// Load a multi-talker pipeline from an ASR model and a diarization model.
    ///
    /// # Arguments
    /// * `asr` - A streaming ASR model: a `.nemo` archive, a model directory or
    ///   a Hugging Face Hub repository (see [`ModelSource`]). A multitalker
    ///   model (with speaker kernels) runs in parallel mode, any other in
    ///   serial mode.
    /// * `diar` - A streaming Sortformer diarization `.nemo` model, local or on
    ///   the Hub
    ///
    /// The ASR model streams in mode 0; set
    /// [`MultiTalkerConfig::streaming`] to choose another.
    #[doc(alias = "restore")]
    pub fn from_pretrained(
        asr: impl Into<ModelSource>,
        diar: impl Into<ModelSource>,
    ) -> Result<Self> {
        Self::from_pretrained_with_config(asr, diar, MultiTalkerConfig::default())
    }

    /// Like [`Self::from_pretrained`], with explicit settings.
    pub fn from_pretrained_with_config(
        asr: impl Into<ModelSource>,
        diar: impl Into<ModelSource>,
        config: MultiTalkerConfig,
    ) -> Result<Self> {
        let model = EncDecRnntModel::from_pretrained_with(asr, config.load)?;
        let asr = StreamingSession::new(model, config.streaming.clone())?;
        if asr.decoder_type() != DecoderType::Rnnt {
            return Err(Error::not_implemented(
                "multi-talker transcription with a CTC decoder",
            ));
        }
        let mut diar_model = SortformerEncLabelModel::from_pretrained_with(diar, config.load)?;

        // Drive the diarizer with the ASR chunk geometry, as NeMo's
        // multitalker inference does
        let streaming = &asr.streaming_config;
        let params = diar_model.streaming_params_mut();
        params.chunk_len = streaming.valid_out_len + streaming.cache_drop_size;
        params.chunk_left_context = 0;
        params.chunk_right_context = 0;
        params.fifo_len = config.diar_fifo_len;
        params.spkcache_update_period = config.diar_spkcache_update_period;
        let diar_state = diar_model.create_streaming_state()?;

        let mode = if asr.model.encoder.has_speaker_kernels() {
            MultiTalkerMode::Parallel
        } else {
            MultiTalkerMode::Serial
        };
        let num_spks = diar_model.max_num_spks();
        let frame_sec =
            asr.model.encoder.subsampling_factor() as f32 * asr.mel_stream_config.mel.window_stride;

        tracing::info!(
            ?mode,
            num_spks,
            diar_chunk_len = diar_model.config().sortformer_modules.chunk_len,
            "Multi-talker pipeline initialized"
        );

        let mut pipeline = Self {
            encoder_cache: Self::new_encoder_cache(&asr, 1)?,
            asr,
            diar_model,
            diar_state,
            config,
            mode,
            num_spks,
            instances: Vec::new(),
            diar_preds: Vec::new(),
            frame_offset: 0,
            frame_sec,
            chunks_since_active: Vec::new(),
            words: Vec::new(),
        };
        pipeline.reset_state()?;
        Ok(pipeline)
    }

    fn new_encoder_cache(
        asr: &StreamingSession<EncDecRnntModel>,
        batch_size: usize,
    ) -> Result<EncoderCache> {
        EncoderCache::new(
            asr.model.encoder.num_layers(),
            batch_size,
            asr.streaming_config.last_channel_cache_size,
            asr.streaming_config.last_time_cache_size,
            asr.model.encoder.d_model(),
        )
    }

    fn reset_state(&mut self) -> Result<()> {
        let n_instances = match self.mode {
            MultiTalkerMode::Parallel => self.num_spks,
            MultiTalkerMode::Serial => 1,
        };
        self.encoder_cache = Self::new_encoder_cache(&self.asr, n_instances)?;
        self.instances = (0..n_instances)
            .map(|_| {
                Ok(Instance {
                    decoder_state: self.asr.model.decoder.initialize_state(1)?,
                    last_token: None,
                    pending: Vec::new(),
                })
            })
            .collect::<Result<_>>()?;
        self.diar_state = self.diar_model.create_streaming_state()?;
        self.diar_preds.clear();
        self.frame_offset = 0;
        self.chunks_since_active = vec![usize::MAX; self.num_spks];
        self.words.clear();
        self.asr.model.encoder.clear_speaker_targets();
        Ok(())
    }

    /// The mode selected from the ASR checkpoint.
    pub fn mode(&self) -> MultiTalkerMode {
        self.mode
    }

    /// Transcribe a whole WAV file or clip, starting a new stream.
    pub fn transcribe<'a>(
        &mut self,
        input: impl Into<AudioInput<'a>>,
    ) -> Result<Vec<SpeakerSegment>> {
        let sample_rate = self.sample_rate();
        let samples = input.into().load(sample_rate)?;
        self.reset()?;
        self.push_audio(&samples, sample_rate)?;
        self.finish()
    }

    /// Add mono audio in `[-1, 1]` recorded at `sample_rate` (resampled to
    /// the model's rate). Every push must use the same rate.
    pub fn push_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<()> {
        self.asr.push_audio(samples, sample_rate)
    }

    /// Signal that no more audio will be pushed.
    pub fn end_audio(&mut self) -> Result<()> {
        self.asr.end_audio()
    }

    /// Start a new stream.
    pub fn reset(&mut self) -> Result<()> {
        self.asr.reset()?;
        self.reset_state()
    }

    /// Decode the next chunk, returning the words completed in it, or `None`
    /// if not enough audio is buffered yet (or the stream is done).
    pub fn next_update(&mut self) -> Result<Option<MultiTalkerOutput>> {
        let fb = match self.asr.try_extract_next()? {
            Some(fb) => fb,
            None => return Ok(None),
        };
        self.process_chunk(fb).map(Some)
    }

    /// End the audio, decode everything still buffered, and return the
    /// speaker segments. Call [`reset`](Self::reset) to reuse the pipeline.
    pub fn finish(&mut self) -> Result<Vec<SpeakerSegment>> {
        self.end_audio()?;
        while self.next_update()?.is_some() {}
        Ok(self.segments())
    }

    /// All completed words so far, ordered by start time.
    pub fn words(&self) -> &[SpeakerWord] {
        &self.words
    }

    /// Group the completed words into speaker segments (SegLST style).
    ///
    /// A speaker's segment ends when they pause for longer than
    /// `sent_break_sec`. Segments are ordered by start time and can overlap.
    pub fn segments(&self) -> Vec<SpeakerSegment> {
        build_segments(&self.words, self.num_spks, self.config.sent_break_sec)
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.asr.sample_rate()
    }

    /// The ASR model's tokenizer (`None` only if a non-strict load found no
    /// tokenizer).
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.asr.model().tokenizer()
    }

    fn process_chunk(&mut self, fb: FeatureBuffer) -> Result<MultiTalkerOutput> {
        let n_mels = self.asr.n_mels as i32;
        let chunk_frames = fb.features.dim(1);
        let features = fb.features.reshape(&[1, n_mels, chunk_frames])?;
        let drop_extra = self.asr.streaming_config.drop_extra_for_step(fb.is_first);

        // Diarize the same chunk the ASR model sees. Its valid frames include
        // the pre-encode cache at the front of the chunk.
        let diar_valid = (fb.pre_encode_cache + fb.valid_size).min(chunk_frames as usize);
        let chunk_preds = self.diar_model.forward_streaming_step(
            &features,
            diar_valid,
            &mut self.diar_state,
            drop_extra,
            0,
            0,
        )?;
        let n_diar_frames = chunk_preds.dim(1) as usize;
        tracing::debug!(
            mel_frames = chunk_frames,
            pre_encode_cache = fb.pre_encode_cache,
            valid_size = fb.valid_size,
            n_diar_frames,
            "diarized chunk"
        );
        let preds_flat: Vec<f32> = chunk_preds.to_vec_row_major()?;
        let preds: Vec<Vec<f32>> = preds_flat
            .chunks(self.num_spks)
            .take(n_diar_frames)
            .map(|row| row.to_vec())
            .collect();
        self.update_gating(&preds);

        let targets = match self.mode {
            MultiTalkerMode::Parallel => Some(self.speaker_targets(&preds)?),
            MultiTalkerMode::Serial => None,
        };
        if let Some(targets) = &targets {
            self.asr.model.encoder.set_speaker_targets(targets.clone());
        }

        let n_instances = self.instances.len() as i32;
        let features = ops::broadcast_to(&features, &[n_instances, n_mels, chunk_frames])?;
        let lengths = Array::from_slice(&vec![chunk_frames; n_instances as usize], &[n_instances]);
        let (encoded, enc_len) = self.run_encoder(&features, &lengths, drop_extra)?;
        let enc_seq_len = if self.asr.model.encoder.is_chunked_limited() || fb.is_last {
            enc_len
        } else {
            enc_len.min(self.asr.streaming_config.valid_out_len)
        };

        // Record diarizer output on the encoder frame grid (aligned at the end,
        // like the speaker kernel masks)
        let chunk_activity = align_rows(&preds, enc_seq_len, self.num_spks);
        self.diar_preds.extend(chunk_activity.iter().cloned());

        let tokens = self.decode(&encoded, enc_seq_len, self.frame_offset)?;
        self.frame_offset += enc_seq_len;

        let mut new_words = self.collect_words(tokens, fb.is_last);
        new_words.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
        self.words.extend(new_words.iter().cloned());
        if !new_words.is_empty() {
            self.words
                .sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
        }

        Ok(MultiTalkerOutput {
            new_words,
            speaker_activity: chunk_activity,
            is_final: fb.is_last,
        })
    }

    /// Run the encoder on [instances, n_mels, frames] and update the cache.
    ///
    /// Returns the encoder output [instances, frames, d_model] and its length.
    fn run_encoder(
        &mut self,
        features: &Array,
        lengths: &Array,
        drop_extra: usize,
    ) -> Result<(Array, usize)> {
        let (encoded, encoded_lengths, cache_channel, cache_time, cache_len) =
            self.asr.model.encoder.forward_streaming(
                features,
                lengths,
                &self.encoder_cache.last_channel,
                &self.encoder_cache.last_time,
                &self.encoder_cache.last_channel_len,
                drop_extra,
            )?;
        self.encoder_cache.last_channel = cache_channel;
        self.encoder_cache.last_time = cache_time;
        self.encoder_cache.last_channel_len = cache_len;

        let encoded = ops::transpose_axes(&encoded, &[0, 2, 1])?;
        let enc_len = (encoded_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize)
            .min(encoded.dim(1) as usize);
        Ok((encoded, enc_len))
    }

    /// Build per-instance speaker targets from this chunk's diarizer output.
    fn speaker_targets(&self, preds: &[Vec<f32>]) -> Result<SpeakerTargets> {
        let s = self.num_spks;
        let t = preds.len();
        let mut preds: Vec<Vec<f32>> = preds.to_vec();
        if self.config.binary_diar_preds {
            for row in preds.iter_mut() {
                row.iter_mut()
                    .for_each(|p| *p = if *p > ACTIVITY_THRESHOLD { 1.0 } else { 0.0 });
            }
        }
        if self.config.single_speaker_mode {
            let n_active = (0..s)
                .filter(|&spk| preds.iter().any(|row| row[spk] > ACTIVITY_THRESHOLD))
                .count();
            if n_active <= 1 {
                for row in preds.iter_mut() {
                    row.iter_mut()
                        .enumerate()
                        .for_each(|(spk, p)| *p = if spk == 0 { 1.0 } else { 0.0 });
                }
            }
        }

        let mut spk = vec![0.0f32; s * t];
        let mut bg = vec![0.0f32; s * t];
        for target in 0..s {
            for (frame, row) in preds.iter().enumerate() {
                spk[target * t + frame] = row[target];
                let others_active = row
                    .iter()
                    .enumerate()
                    .any(|(other, &p)| other != target && p > ACTIVITY_THRESHOLD);
                bg[target * t + frame] = if others_active { 1.0 } else { 0.0 };
            }
        }
        let shape = [s as i32, t as i32];
        Ok(SpeakerTargets {
            spk: Array::from_slice(&spk, &shape),
            bg: Array::from_slice(&bg, &shape),
        })
    }

    fn update_gating(&mut self, preds: &[Vec<f32>]) {
        for (spk, since) in self.chunks_since_active.iter_mut().enumerate() {
            if preds.iter().any(|row| row[spk] > ACTIVITY_THRESHOLD) {
                *since = 0;
            } else {
                *since = since.saturating_add(1);
            }
        }
    }

    /// Whether instance `idx` should be decoded this chunk.
    fn is_decoded(&self, idx: usize) -> bool {
        if self.mode == MultiTalkerMode::Serial
            || self.config.cache_gating_chunks == 0
            || (self.config.single_speaker_mode && idx == 0)
        {
            return true;
        }
        self.chunks_since_active[idx] < self.config.cache_gating_chunks
    }

    /// Greedy-decode each instance's encoder output.
    ///
    /// Returns `(token, global encoder frame)` pairs per instance.
    fn decode(
        &mut self,
        encoded: &Array,
        valid_frames: usize,
        frame_offset: usize,
    ) -> Result<Vec<Vec<(usize, usize)>>> {
        let masked_ids = self
            .asr
            .options()
            .suppress_punctuation
            .then_some(self.asr.punctuation_token_ids.as_slice());
        let mut tokens = Vec::with_capacity(self.instances.len());
        for idx in 0..self.instances.len() {
            if !self.is_decoded(idx) {
                tokens.push(Vec::new());
                continue;
            }
            let row = encoded.index((idx as i32..idx as i32 + 1, .., ..));
            let instance = &mut self.instances[idx];
            let output = self.asr.streaming_decoder.decode_frames(
                &row,
                valid_frames,
                &self.asr.model.decoder,
                &mut instance.decoder_state,
                &self.asr.model.joint,
                &mut instance.last_token,
                masked_ids,
            )?;
            tokens.push(
                output
                    .new_tokens
                    .into_iter()
                    .zip(output.timesteps.into_iter().map(|t| frame_offset + t))
                    .collect(),
            );
        }
        Ok(tokens)
    }

    /// Group new tokens into words, completing a word when the next one starts.
    fn collect_words(
        &mut self,
        tokens: Vec<Vec<(usize, usize)>>,
        is_final: bool,
    ) -> Vec<SpeakerWord> {
        let mut words = Vec::new();
        for (idx, new_tokens) in tokens.into_iter().enumerate() {
            for (token, frame) in new_tokens {
                let starts_word = self
                    .asr
                    .model
                    .tokenizer
                    .as_ref()
                    .is_some_and(|t| t.is_word_start(token));
                if starts_word && !self.instances[idx].pending.is_empty() {
                    let pending = std::mem::take(&mut self.instances[idx].pending);
                    words.extend(self.make_word(idx, &pending));
                }
                self.instances[idx].pending.push((token, frame));
            }
            if is_final && !self.instances[idx].pending.is_empty() {
                let pending = std::mem::take(&mut self.instances[idx].pending);
                words.extend(self.make_word(idx, &pending));
            }
        }
        words
    }

    fn make_word(&self, instance: usize, tokens: &[(usize, usize)]) -> Option<SpeakerWord> {
        let tokenizer = self.asr.model.tokenizer.as_ref()?;
        let ids: Vec<usize> = tokens.iter().map(|&(t, _)| t).collect();
        let text = tokenizer
            .decode(&ids, self.asr.options().strip_lang_tags)
            .trim()
            .to_string();
        if text.is_empty() {
            return None;
        }
        let first = tokens.first()?.1;
        let last = tokens.last()?.1;
        let speaker = match self.mode {
            MultiTalkerMode::Parallel => instance,
            MultiTalkerMode::Serial => self.dominant_speaker(first, last),
        };
        Some(SpeakerWord {
            speaker,
            text,
            start_time: first as f32 * self.frame_sec,
            end_time: (last + 1) as f32 * self.frame_sec,
        })
    }

    /// The speaker with the highest mean activity over `first..=last`.
    fn dominant_speaker(&self, first: usize, last: usize) -> usize {
        let end = (last + 1).min(self.diar_preds.len());
        let start = first.min(end.saturating_sub(1));
        let frames = &self.diar_preds[start..end];
        (0..self.num_spks)
            .max_by(|&a, &b| {
                let mean_a: f32 = frames.iter().map(|row| row[a]).sum();
                let mean_b: f32 = frames.iter().map(|row| row[b]).sum();
                mean_a.total_cmp(&mean_b)
            })
            .unwrap_or(0)
    }
}

/// Take the last `n` rows of `rows`, left-padding with zero rows if short.
fn align_rows(rows: &[Vec<f32>], n: usize, width: usize) -> Vec<Vec<f32>> {
    if rows.len() >= n {
        rows[rows.len() - n..].to_vec()
    } else {
        let mut out = vec![vec![0.0; width]; n - rows.len()];
        out.extend(rows.iter().cloned());
        out
    }
}

/// Group time-ordered words into per-speaker segments split at long pauses.
fn build_segments(
    words: &[SpeakerWord],
    num_spks: usize,
    sent_break_sec: f32,
) -> Vec<SpeakerSegment> {
    let mut open: Vec<Option<SpeakerSegment>> = vec![None; num_spks.max(1)];
    let mut segments = Vec::new();
    for word in words {
        if word.speaker >= open.len() {
            open.resize(word.speaker + 1, None);
        }
        let slot = &mut open[word.speaker];
        match slot {
            Some(seg) if word.start_time <= seg.end_time + sent_break_sec => {
                if !is_punctuation(&word.text) {
                    seg.text.push(' ');
                }
                seg.text.push_str(&word.text);
                seg.end_time = seg.end_time.max(word.end_time);
            }
            _ => {
                if let Some(seg) = slot.take() {
                    segments.push(seg);
                }
                *slot = Some(SpeakerSegment {
                    speaker: word.speaker,
                    text: word.text.clone(),
                    start_time: word.start_time,
                    end_time: word.end_time,
                });
            }
        }
    }
    segments.extend(open.into_iter().flatten());
    segments.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    segments
}

fn is_punctuation(text: &str) -> bool {
    text.chars().all(|c| c.is_ascii_punctuation())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(speaker: usize, text: &str, start: f32, end: f32) -> SpeakerWord {
        SpeakerWord {
            speaker,
            text: text.to_string(),
            start_time: start,
            end_time: end,
        }
    }

    #[test]
    fn test_build_segments_overlapping_speakers() {
        let words = vec![
            word(0, "hello", 0.0, 0.4),
            word(1, "hi", 0.2, 0.5),
            word(0, "there", 0.5, 0.9),
            word(0, "again", 3.0, 3.4),
        ];
        let segments = build_segments(&words, 2, 1.0);
        let summary: Vec<(usize, &str)> = segments
            .iter()
            .map(|s| (s.speaker, s.text.as_str()))
            .collect();
        assert_eq!(summary, vec![(0, "hello there"), (1, "hi"), (0, "again")]);
    }

    #[test]
    fn test_build_segments_attaches_punctuation() {
        let words = vec![word(0, "done", 0.0, 0.4), word(0, ".", 0.4, 0.5)];
        let segments = build_segments(&words, 1, 1.0);
        assert_eq!(segments[0].text, "done.");
    }

    #[test]
    fn test_align_rows() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(align_rows(&rows, 2, 1), vec![vec![2.0], vec![3.0]]);
        assert_eq!(
            align_rows(&rows, 4, 1),
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]
        );
    }
}
