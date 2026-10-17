//! Cache-aware streaming transcription with RNN-T models, mirroring NeMo's
//! `CacheAwareStreamingAudioBuffer` and cache-aware streaming inference.

use std::borrow::BorrowMut;
use std::sync::Arc;

use mlx_rs::Array;

use crate::asr::Tokenizer;
use crate::asr::decoding::{IncrementalDetokenizer, RnntStreamingDecoder, TimedToken};
use crate::asr::models::EncDecRnntModel;
use crate::asr::modules::{DecoderState, SpectrumTransform, StreamingMel, StreamingMelConfig};
use crate::asr::streaming::{
    EncoderCache, FeatureBuffer, StreamingConfig, compute_streaming_config,
};
use crate::asr::transcript::{Token, Transcript};
use crate::audio::StreamResampler;
use crate::core::ArrayExt;
use crate::core::error::{Error, Result};

/// Options for a [`StreamingSession`].
///
/// ```no_run
/// # use mlx_voice::asr::StreamingOptions;
/// let options = StreamingOptions::default().mode(3).blank_threshold(0.5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StreamingOptions {
    /// Streaming mode: an index into the checkpoint's `att_context_size` list
    /// (see [`EncDecRnntModel::streaming_modes`] for each mode's latency).
    /// NeMo checkpoints list modes from most to least lookahead, so mode 0,
    /// the default, has the highest latency and accuracy.
    pub mode: usize,
    /// Leave language tags such as `<en-US>` out of text (the default).
    pub strip_lang_tags: bool,
    /// Never emit punctuation tokens.
    pub suppress_punctuation: bool,
    /// Emit blank only when it beats the best other token by more than this
    /// many nats (0 = plain argmax). Around 0.5 to 2.0 helps a model that
    /// drops words by over-predicting blank.
    pub blank_threshold: f32,
    /// Skip audio chunks whose RMS energy is below this (0 = off). Around
    /// 0.005 to 0.05 depending on background noise.
    pub silence_threshold: f32,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            mode: 0,
            strip_lang_tags: true,
            suppress_punctuation: false,
            blank_threshold: 0.0,
            silence_threshold: 0.0,
        }
    }
}

impl StreamingOptions {
    /// Use streaming mode `mode`.
    pub fn mode(mut self, mode: usize) -> Self {
        self.mode = mode;
        self
    }

    /// Keep or strip language tags in text.
    pub fn strip_lang_tags(mut self, strip: bool) -> Self {
        self.strip_lang_tags = strip;
        self
    }

    /// Never emit punctuation tokens.
    pub fn suppress_punctuation(mut self, suppress: bool) -> Self {
        self.suppress_punctuation = suppress;
        self
    }

    /// Set the blank threshold (see [`StreamingOptions::blank_threshold`]).
    pub fn blank_threshold(mut self, threshold: f32) -> Self {
        self.blank_threshold = threshold;
        self
    }

    /// Set the silence threshold (see [`StreamingOptions::silence_threshold`]).
    pub fn silence_threshold(mut self, threshold: f32) -> Self {
        self.silence_threshold = threshold;
        self
    }
}

/// What one streaming step decoded.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct StreamUpdate {
    /// Tokens emitted in this step, with their times from the start of the
    /// stream
    pub tokens: Vec<Token>,
    /// Text appended since the previous update. If decoding revised earlier
    /// text (rare), this is the text after the last unchanged character;
    /// [`StreamingSession::text`] is always the full text.
    pub delta: String,
    /// Whether the stream has ended and this is the last update
    pub is_final: bool,
}

/// Decoding state carried between chunks.
#[derive(Debug)]
struct DecodeState {
    decoder_state: DecoderState,
    last_token: Option<usize>,
    /// Every token so far, with absolute encoder frames
    tokens: Vec<TimedToken>,
    /// Encoder frames decoded so far
    frame_offset: usize,
    detokenizer: IncrementalDetokenizer,
    /// Text of `tokens`
    text: String,
    step_count: usize,
}

/// Check if audio chunk is silent based on RMS energy.
fn is_silent(audio: &[f32], threshold: f32) -> bool {
    if threshold <= 0.0 || audio.is_empty() {
        return false;
    }
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    rms < threshold
}

/// Token ids whose text is only punctuation and whitespace.
fn find_punctuation_tokens(tokenizer: &Tokenizer) -> Vec<usize> {
    (0..tokenizer.vocab_size())
        .filter(|&id| {
            let piece = tokenizer.decode(&[id], false);
            piece
                .chars()
                .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
                && piece.chars().any(|c| c.is_ascii_punctuation())
        })
        .collect()
}

/// The text after the longest common prefix of `old` and `new`.
fn appended_text(old: &str, new: &str) -> String {
    let common = old
        .char_indices()
        .zip(new.chars())
        .find(|((_, a), b)| a != b)
        .map_or(old.len().min(new.len()), |((i, _), _)| i);
    new.get(common..).unwrap_or_default().to_string()
}

/// A cache-aware streaming transcription of one audio stream.
///
/// Push audio as it arrives and pull updates, which emit text with roughly
/// the latency of the chosen [`StreamingOptions::mode`]. The session borrows
/// the model ([`EncDecRnntModel::stream`]) or owns it
/// ([`StreamingSession::new`]).
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// # let mic_chunks: Vec<Vec<f32>> = Vec::new();
/// use mlx_voice::asr::{EncDecRnntModel, StreamingOptions};
///
/// let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
/// let mut stream = model.stream(StreamingOptions::default().mode(3))?;
/// for chunk in &mic_chunks {
///     stream.push_audio(chunk, 48_000)?; // resampled to the model's rate
///     while let Some(update) = stream.next_update()? {
///         print!("{}", update.delta);
///     }
/// }
/// let transcript = stream.finish()?;
/// println!("\n{transcript}");
/// # Ok(())
/// # }
/// ```
///
/// # Threads
///
/// A session is driven from one thread at a time, and an owned session (like
/// the model) is `Send`. With a real-time audio callback, keep the callback to
/// a non-blocking write into a queue (e.g. a lock-free ring buffer), and run
/// the session on a worker thread that drains the queue: `push_audio`
/// allocates and resamples, so it shouldn't run in the callback. Watch
/// [`buffered_samples`](Self::buffered_samples) to notice when decoding falls
/// behind real time.
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// use std::sync::mpsc;
/// use mlx_voice::asr::{EncDecRnntModel, StreamingOptions, StreamingSession};
///
/// // Stands in for a lock-free queue the audio callback writes to
/// let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
/// let (text_tx, text_rx) = mpsc::channel::<String>();
///
/// let model = EncDecRnntModel::from_pretrained("model.nemo")?;
/// let worker = std::thread::spawn(move || -> mlx_voice::Result<()> {
///     let mut session = StreamingSession::new(model, StreamingOptions::default())?;
///     for chunk in audio_rx {
///         session.push_audio(&chunk, 48_000)?;
///         while let Some(update) = session.next_update()? {
///             let _ = text_tx.send(update.delta);
///         }
///         let behind = session.buffered_samples() as f32 / session.sample_rate() as f32;
///         if behind > 2.0 {
///             eprintln!("transcription is {behind:.1}s behind");
///         }
///     }
///     session.finish()?; // the audio sender was dropped: decode the rest
///     Ok(())
/// });
/// # drop((audio_tx, text_rx, worker));
/// # Ok(())
/// # }
/// ```
pub struct StreamingSession<M: BorrowMut<EncDecRnntModel> = EncDecRnntModel> {
    pub(crate) model: M,
    options: StreamingOptions,

    pub(crate) streaming_decoder: RnntStreamingDecoder,
    pub(crate) streaming_config: StreamingConfig,
    encoder_cache: EncoderCache,

    pub(crate) mel_stream_config: StreamingMelConfig,
    streaming_mel: StreamingMel,
    /// Whether the next mel chunk is the first
    is_first_chunk: bool,
    pub(crate) n_mels: usize,

    /// Audio at the model's rate not yet turned into mel frames
    live_buffer: Vec<f32>,
    /// Converts pushed audio to the model's rate, with its input rate
    resampler: Option<(u32, StreamResampler)>,
    audio_ended: bool,
    is_done: bool,
    mel_flushed: bool,

    pub(crate) punctuation_token_ids: Vec<usize>,
    state: DecodeState,
}

impl<M: BorrowMut<EncDecRnntModel>> StreamingSession<M> {
    /// Start a session on `model`, which can be the model itself (the
    /// session owns it) or a mutable reference to it.
    pub fn new(mut model: M, options: StreamingOptions) -> Result<Self> {
        let m = model.borrow_mut();
        m.encoder.set_streaming_mode(options.mode);
        let streaming_config = compute_streaming_config(m.encoder.config(), options.mode, 10000)?;
        m.encoder.setup_streaming(streaming_config.cache_drop_size);

        let encoder_cache = EncoderCache::new(
            m.encoder.num_layers(),
            1,
            streaming_config.last_channel_cache_size,
            streaming_config.last_time_cache_size,
            m.encoder.d_model(),
        )?;
        let mel_config = m.preprocessor.config().clone();
        let n_mels = mel_config.features;
        let mel_stream_config = StreamingMelConfig::new(
            mel_config.clone(),
            &streaming_config,
            m.encoder.subsampling_factor(),
        );
        let streaming_mel = StreamingMel::new(mel_config)?;
        let streaming_decoder =
            RnntStreamingDecoder::new(m.blank_id).with_blank_threshold(options.blank_threshold);
        let punctuation_token_ids = match &m.tokenizer {
            Some(tokenizer) if options.suppress_punctuation => find_punctuation_tokens(tokenizer),
            _ => Vec::new(),
        };
        // Zeros, like NeMo's greedy decoder with state=None
        let decoder_state = m.decoder.initialize_state(1)?;

        Ok(Self {
            model,
            options,
            streaming_decoder,
            streaming_config,
            encoder_cache,
            mel_stream_config,
            streaming_mel,
            is_first_chunk: true,
            n_mels,
            live_buffer: Vec::new(),
            resampler: None,
            audio_ended: false,
            is_done: false,
            mel_flushed: false,
            punctuation_token_ids,
            state: DecodeState {
                decoder_state,
                last_token: None,
                tokens: Vec::new(),
                frame_offset: 0,
                detokenizer: IncrementalDetokenizer::default(),
                text: String::new(),
                step_count: 0,
            },
        })
    }

    /// The model.
    pub fn model(&self) -> &EncDecRnntModel {
        self.model.borrow()
    }

    /// The model, e.g. to change its language between chunks.
    pub fn model_mut(&mut self) -> &mut EncDecRnntModel {
        self.model.borrow_mut()
    }

    /// End the session and hand back the model.
    pub fn into_model(self) -> M {
        self.model
    }

    /// The session's options.
    pub fn options(&self) -> &StreamingOptions {
        &self.options
    }

    /// The sample rate the model expects; [`push_audio`](Self::push_audio)
    /// resamples other rates to it.
    pub fn sample_rate(&self) -> u32 {
        self.model().sample_rate()
    }

    /// The chunk and cache sizes of the session's streaming mode.
    pub fn streaming_config(&self) -> &StreamingConfig {
        &self.streaming_config
    }

    /// Add mono audio in `[-1, 1]` recorded at `sample_rate`. Every push must
    /// use the same rate.
    pub fn push_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<()> {
        if self.audio_ended {
            return Err(Error::audio(
                "audio pushed after end_audio; call reset() to start a new stream",
            ));
        }
        let model_rate = self.sample_rate();
        match &mut self.resampler {
            Some((rate, _)) if *rate != sample_rate => {
                return Err(Error::audio(format!(
                    "sample rate changed from {rate} to {sample_rate} Hz within a stream"
                )));
            }
            Some((_, resampler)) => {
                let resampled = resampler.process(samples)?;
                self.live_buffer.extend_from_slice(&resampled);
            }
            None if sample_rate == model_rate => self.live_buffer.extend_from_slice(samples),
            None => {
                let mut resampler = StreamResampler::new(sample_rate, model_rate)?;
                let resampled = resampler.process(samples)?;
                self.live_buffer.extend_from_slice(&resampled);
                self.resampler = Some((sample_rate, resampler));
            }
        }
        Ok(())
    }

    /// Signal that no more audio will be pushed. The remaining audio is
    /// decoded by the following [`next_update`](Self::next_update) calls.
    pub fn end_audio(&mut self) -> Result<()> {
        if self.audio_ended {
            return Ok(());
        }
        if let Some((_, resampler)) = &mut self.resampler {
            let tail = resampler.flush()?;
            self.live_buffer.extend_from_slice(&tail);
        }
        self.audio_ended = true;
        Ok(())
    }

    /// Decode the next chunk of buffered audio, or `None` if not enough audio
    /// has arrived yet (or the stream is done).
    pub fn next_update(&mut self) -> Result<Option<StreamUpdate>> {
        let Some(feature_buffer) = self.try_extract_next()? else {
            return Ok(None);
        };
        self.state.step_count += 1;
        tracing::debug!(step = self.state.step_count, "streaming step");
        self.process_feature_buffer(feature_buffer).map(Some)
    }

    /// End the audio, decode everything still buffered, and return the
    /// whole transcript. Call [`reset`](Self::reset) to reuse the session.
    pub fn finish(&mut self) -> Result<Transcript> {
        self.end_audio()?;
        while self.next_update()?.is_some() {}
        Ok(self.transcript())
    }

    /// Audio pushed but not yet decoded, in samples at the model's rate
    /// (divide by [`sample_rate`](Self::sample_rate) for seconds).
    ///
    /// In a live stream this hovers around one chunk plus lookahead. If it
    /// keeps growing, decoding is slower than real time and the caller may
    /// want to drop audio or warn.
    pub fn buffered_samples(&self) -> usize {
        let model_rate = self.sample_rate() as usize;
        let resampling = self.resampler.as_ref().map_or(0, |(rate, resampler)| {
            resampler.pending() * model_rate / (*rate as usize).max(1)
        });
        // Mel frames not yet encoded: all of them before the first chunk, then
        // those past the pre-encode cache kept from the last chunk
        let mel_frames = self.streaming_mel.total_frames();
        let undecoded_frames = if self.is_first_chunk {
            mel_frames
        } else {
            mel_frames.saturating_sub(self.mel_stream_config.pre_encode_cache)
        };
        let hop = self.model().preprocessor.hop_samples();
        self.live_buffer.len() + resampling + undecoded_frames * hop
    }

    /// Whether the stream has ended and all of it has been decoded.
    pub fn is_done(&self) -> bool {
        self.is_done
    }

    /// The text so far.
    pub fn text(&self) -> &str {
        &self.state.text
    }

    /// The transcript so far, with token and word timings.
    pub fn transcript(&self) -> Transcript {
        let model = self.model();
        match &model.tokenizer {
            Some(tokenizer) => Transcript::from_tokens(
                tokenizer,
                &self.state.tokens,
                self.frame_sec(),
                self.options.strip_lang_tags,
            ),
            None => Transcript::default(),
        }
    }

    /// Start a new stream with the same model and options.
    pub fn reset(&mut self) -> Result<()> {
        self.live_buffer.clear();
        self.resampler = None;
        self.audio_ended = false;
        self.is_done = false;
        self.mel_flushed = false;
        self.streaming_mel.reset();
        self.is_first_chunk = true;

        let model = self.model.borrow();
        self.encoder_cache = EncoderCache::new(
            model.encoder.num_layers(),
            1,
            self.streaming_config.last_channel_cache_size,
            self.streaming_config.last_time_cache_size,
            model.encoder.d_model(),
        )?;
        self.state = DecodeState {
            decoder_state: model.decoder.initialize_state(1)?,
            last_token: None,
            tokens: Vec::new(),
            frame_offset: 0,
            detokenizer: IncrementalDetokenizer::default(),
            text: String::new(),
            step_count: 0,
        };
        Ok(())
    }

    /// The streaming mel processor, e.g. to draw a live spectrogram from
    /// [`StreamingMel::latest_frame_bins`].
    pub fn streaming_mel(&self) -> &StreamingMel {
        &self.streaming_mel
    }

    /// Modify the power spectrum before the mel filterbank (noise reduction,
    /// gain control).
    pub fn set_spectrum_transform(&mut self, transform: Arc<dyn SpectrumTransform>) {
        self.streaming_mel.set_spectrum_transform(transform);
    }

    /// Turn all buffered audio into mel frames now, even less than a chunk,
    /// so [`streaming_mel`](Self::streaming_mel) reflects the latest audio.
    pub fn flush_audio_to_mel(&mut self) -> Result<()> {
        if self.live_buffer.is_empty() {
            return Ok(());
        }
        let audio = std::mem::take(&mut self.live_buffer);
        self.streaming_mel.add_audio(&audio)?;
        Ok(())
    }

    /// Seconds per encoder frame.
    fn frame_sec(&self) -> f32 {
        self.mel_stream_config.mel.window_stride * self.mel_stream_config.subsampling_factor as f32
    }

    /// Samples per mel chunk.
    fn audio_chunk_size(&self) -> usize {
        self.mel_stream_config
            .audio_chunk_size(self.model().preprocessor.hop_samples())
    }

    /// Turn buffered audio into mel frames, one chunk at a time.
    fn process_pending_audio(&mut self) -> Result<()> {
        let chunk_audio_size = self.audio_chunk_size();
        while self.live_buffer.len() >= chunk_audio_size {
            let audio_chunk: Vec<f32> = self.live_buffer.drain(..chunk_audio_size).collect();
            if is_silent(&audio_chunk, self.options.silence_threshold) {
                tracing::trace!("skipping silent audio chunk");
                continue;
            }
            self.streaming_mel.add_audio(&audio_chunk)?;
            // One chunk at a time, so extraction interleaves with it
            break;
        }

        if self.audio_ended && !self.live_buffer.is_empty() {
            let remaining = std::mem::take(&mut self.live_buffer);
            self.streaming_mel.add_audio(&remaining)?;
        }

        // Flush the final mel frames from the audio lookback buffer
        if self.audio_ended && self.live_buffer.is_empty() && !self.mel_flushed {
            self.mel_flushed = true;
            self.streaming_mel.flush()?;
        }
        Ok(())
    }

    /// Extract a mel chunk from the streaming mel buffer.
    ///
    /// Mirrors NeMo's `CacheAwareStreamingAudioBuffer`: the first chunk is
    /// `chunk_size_first` frames with no pre-encode cache; every later chunk is
    /// the `pre_encode_cache` frames before it plus `chunk_size` new frames.
    /// The last chunk is shorter rather than padded.
    fn try_extract_chunk(&mut self) -> Result<Option<FeatureBuffer>> {
        let is_first = self.is_first_chunk;
        let config = &self.mel_stream_config;
        let chunk_size = config.chunk_size_for_chunk(is_first);
        let pre_encode_cache = config.pre_encode_cache_for_chunk(is_first);
        let shift_size = config.shift_size_for_chunk(is_first);
        let next_pre_encode_cache = config.pre_encode_cache;
        let min_frames = config.min_frames_for_chunk(is_first);
        let min_frames_next = config.min_frames_for_chunk(false);

        // The buffer starts with this chunk's pre-encode cache
        let available = self.streaming_mel.total_frames();
        let required = pre_encode_cache + chunk_size;
        if available < required {
            if !self.audio_ended {
                return Ok(None);
            }
            // NeMo stops once a chunk has too few new frames for one encoder frame
            if available.saturating_sub(pre_encode_cache) < min_frames {
                self.is_done = true;
                return Ok(None);
            }
        }

        let total = available.min(required);
        let mel_chunk = self.streaming_mel.get_frames(0, total)?;
        let new_frames = total - pre_encode_cache;

        // Advance to the next chunk, keeping its pre-encode cache
        self.streaming_mel
            .trim(pre_encode_cache + shift_size - next_pre_encode_cache);
        self.is_first_chunk = false;

        let next_new_frames = self
            .streaming_mel
            .total_frames()
            .saturating_sub(next_pre_encode_cache);
        let is_last = self.audio_ended && next_new_frames < min_frames_next;
        if is_last {
            self.is_done = true;
        }

        Ok(Some(
            FeatureBuffer::new(mel_chunk, new_frames)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        ))
    }

    /// The next mel chunk to encode, if enough audio has arrived.
    pub(crate) fn try_extract_next(&mut self) -> Result<Option<FeatureBuffer>> {
        if self.is_done {
            return Ok(None);
        }
        self.process_pending_audio()?;
        if let Some(buffer) = self.try_extract_chunk()? {
            return Ok(Some(buffer));
        }
        if self.audio_ended && self.streaming_mel.total_frames() == 0 {
            self.is_done = true;
        }
        Ok(None)
    }

    /// Encode and decode one mel chunk.
    fn process_feature_buffer(&mut self, fb: FeatureBuffer) -> Result<StreamUpdate> {
        let is_last = fb.is_last;
        // NeMo drops the extra pre-encoded frames on every chunk but the first
        let drop_extra = self.streaming_config.drop_extra_for_step(fb.is_first);

        // [n_mels, time] -> [1, n_mels, time]; like NeMo, the length covers the
        // pre-encode cache plus the new frames
        let chunk_len = fb.features.dim(1);
        let features = fb.features.reshape(&[1, self.n_mels as i32, chunk_len])?;
        let lengths = Array::from_slice(&[chunk_len], &[1]);

        let model = self.model.borrow_mut();
        let (encoded, encoded_lengths, cache_channel_next, cache_time_next, cache_len_next) =
            model.encoder.forward_streaming(
                &features,
                &lengths,
                &self.encoder_cache.last_channel,
                &self.encoder_cache.last_time,
                &self.encoder_cache.last_channel_len,
                drop_extra,
            )?;
        self.encoder_cache.last_channel = cache_channel_next;
        self.encoder_cache.last_time = cache_time_next;
        self.encoder_cache.last_channel_len = cache_len_next;

        // [B, D, T] -> [B, T, D]
        let encoded = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;

        // chunked_limited keeps every output (NeMo's keep_all_outputs); regular
        // style clips to valid_out_len except on the last chunk
        let actual_frames = encoded.shape()[1] as usize;
        // A host array, so this doesn't wait on the GPU
        let encoded_lengths = encoded_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize;
        let enc_seq_len = if model.encoder.is_chunked_limited() || is_last {
            encoded_lengths.min(actual_frames)
        } else {
            encoded_lengths
                .min(self.streaming_config.valid_out_len)
                .min(actual_frames)
        };

        let masked_ids = self
            .options
            .suppress_punctuation
            .then_some(self.punctuation_token_ids.as_slice());
        let decoded = self.streaming_decoder.decode_frames(
            &encoded,
            enc_seq_len,
            &model.decoder,
            &mut self.state.decoder_state,
            &model.joint,
            &mut self.state.last_token,
            masked_ids,
        )?;

        let frame_offset = self.state.frame_offset;
        let new_tokens: Vec<TimedToken> = decoded
            .new_tokens
            .iter()
            .zip(&decoded.timesteps)
            .map(|(&id, &t)| TimedToken {
                id,
                start_frame: frame_offset + t,
                end_frame: frame_offset + t + 1,
            })
            .collect();
        self.state.frame_offset += enc_seq_len;
        self.state.tokens.extend(&new_tokens);

        let model = self.model.borrow();
        let frame_sec = self.frame_sec();
        let (tokens, delta) = match &model.tokenizer {
            Some(tokenizer) => {
                let ids: Vec<usize> = self.state.tokens.iter().map(|t| t.id).collect();
                let text = self.state.detokenizer.update(
                    tokenizer.processor(),
                    tokenizer.word_start_table(),
                    self.options.strip_lang_tags,
                    &ids,
                );
                let delta = appended_text(&self.state.text, &text);
                self.state.text = text;
                let tokens = new_tokens
                    .iter()
                    .map(|t| Token {
                        id: t.id,
                        text: tokenizer.decode(&[t.id], false),
                        start: t.start_frame as f32 * frame_sec,
                        end: t.end_frame as f32 * frame_sec,
                    })
                    .collect();
                (tokens, delta)
            }
            None => (Vec::new(), String::new()),
        };

        Ok(StreamUpdate {
            tokens,
            delta,
            is_final: is_last,
        })
    }
}

impl<M: BorrowMut<EncDecRnntModel>> std::fmt::Debug for StreamingSession<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingSession")
            .field("model", self.model())
            .field("options", &self.options)
            .field("is_done", &self.is_done)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_appended_text() {
        assert_eq!(appended_text("hello", "hello world"), " world");
        assert_eq!(appended_text("", "hi"), "hi");
        assert_eq!(appended_text("observ", "observed"), "ed");
        // A revision returns the text after the unchanged prefix
        assert_eq!(appended_text("cat", "car park"), "r park");
        assert_eq!(appended_text("héllo", "héllo!"), "!");
    }

    #[test]
    fn test_options_builder() {
        let options = StreamingOptions::default()
            .mode(2)
            .blank_threshold(0.5)
            .suppress_punctuation(true);
        assert_eq!(options.mode, 2);
        assert_eq!(options.blank_threshold, 0.5);
        assert!(options.suppress_punctuation);
        assert!(options.strip_lang_tags);
    }
}
