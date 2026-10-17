//! Audio I/O utilities.
//!
//! This module provides functions for loading and resampling audio files.
//! For mel spectrogram computation, use `asr::modules::MelProcessor`.

use std::path::Path;

use crate::core::error::{Error, Result};

// ============================================================================
// Audio I/O
// ============================================================================

/// Load audio from a WAV file.
///
/// Returns (samples, sample_rate). Stereo files are converted to mono by averaging.
#[cfg(feature = "asr")]
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    use hound::WavReader;

    let path = path.as_ref();
    let reader = WavReader::open(path).map_err(|e| {
        Error::audio(format!(
            "Failed to open WAV file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::audio(format!("Failed to read float samples: {}", e)))?;

            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect()
            } else {
                samples
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;

            let samples: Vec<i32> = reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::audio(format!("Failed to read int samples: {}", e)))?;

            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| {
                        chunk.iter().map(|&s| s as f32).sum::<f32>() / (channels as f32 * max_val)
                    })
                    .collect()
            } else {
                samples.iter().map(|&s| s as f32 / max_val).collect()
            }
        }
    };

    Ok((samples, sample_rate))
}

/// Load audio from a WAV file. Needs the `asr` feature; without it this
/// always returns an error.
#[cfg(not(feature = "asr"))]
pub fn load_wav(_path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    Err(Error::not_implemented(
        "Audio loading requires the 'asr' feature",
    ))
}

/// Resample audio to a target sample rate.
#[cfg(feature = "asr")]
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction, audioadapter_buffers::direct::InterleavedSlice,
    };

    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, 1024, 1, FixedAsync::Input)
        .map_err(|e| Error::audio(format!("Failed to create resampler: {}", e)))?;

    let input = InterleavedSlice::new(samples, 1, samples.len())
        .map_err(|e| Error::audio(format!("Invalid resampler input: {}", e)))?;
    let output = resampler
        .process_all(&input, samples.len(), None)
        .map_err(|e| Error::audio(format!("Resampling failed: {}", e)))?;

    // Mono, so the interleaved buffer is the channel itself.
    Ok(output.take_data())
}

/// Resample audio to a target sample rate. Needs the `asr` feature; without
/// it this always returns an error.
#[cfg(not(feature = "asr"))]
pub fn resample(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
    Err(Error::not_implemented(
        "Audio resampling requires the 'asr' feature",
    ))
}

/// Resamples mono audio that arrives in pieces, keeping the filter state
/// between pieces. Resampling each piece on its own would distort the joins.
///
/// The resampler's delay is trimmed, so output sample `n` lines up with input
/// time `n / to_rate`.
#[cfg(feature = "asr")]
pub struct StreamResampler {
    resampler: rubato::Async<f32>,
    /// Input not yet resampled (less than one resampler chunk)
    pending: Vec<f32>,
    /// Output frames still to drop to undo the resampler delay
    delay: usize,
    /// Input frames seen
    input_frames: usize,
    /// Output frames returned
    output_frames: usize,
    ratio: f64,
}

#[cfg(feature = "asr")]
impl StreamResampler {
    /// Input frames per resampler chunk.
    const CHUNK: usize = 1024;

    /// A resampler from `from_rate` to `to_rate` Hz.
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Self> {
        use rubato::{
            Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
            WindowFunction,
        };

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: Some(0.95),
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = to_rate as f64 / from_rate as f64;
        let resampler =
            Async::<f32>::new_sinc(ratio, 2.0, &params, Self::CHUNK, 1, FixedAsync::Input)
                .map_err(|e| Error::audio(format!("Failed to create resampler: {}", e)))?;
        Ok(Self {
            delay: resampler.output_delay(),
            resampler,
            pending: Vec::new(),
            input_frames: 0,
            output_frames: 0,
            ratio,
        })
    }

    /// Resample the next piece of input. Returns whatever output is ready,
    /// which may be empty until a full resampler chunk has arrived.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        use rubato::Resampler;

        self.input_frames += input.len();
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();
        while self.pending.len() >= self.resampler.input_frames_next() {
            let chunk: Vec<f32> = self
                .pending
                .drain(..self.resampler.input_frames_next())
                .collect();
            out.extend(self.run(&chunk, None)?);
        }
        Ok(self.trim(out))
    }

    /// Input frames received but not yet resampled (less than one internal
    /// chunk of 1024 frames until [`flush`](Self::flush)).
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Resample the remaining input, ending the stream. Afterwards the output
    /// totals the input length times the rate ratio.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let expected = (self.input_frames as f64 * self.ratio).round() as usize;
        let mut out = Vec::new();
        let tail = std::mem::take(&mut self.pending);
        let mut partial = Some(tail);
        while self.output_frames + out.len().saturating_sub(self.delay) < expected {
            let chunk = partial.take().unwrap_or_default();
            let len = chunk.len();
            out.extend(self.run(&chunk, Some(len))?);
        }
        // The last chunk overshoots: drop what lies past the end of the input
        let mut out = self.trim(out);
        let excess = self.output_frames.saturating_sub(expected).min(out.len());
        out.truncate(out.len() - excess);
        self.output_frames -= excess;
        Ok(out)
    }

    /// Resample one chunk; `partial` marks a final chunk shorter than a full one.
    fn run(&mut self, chunk: &[f32], partial: Option<usize>) -> Result<Vec<f32>> {
        use rubato::Indexing;
        use rubato::Resampler;
        use rubato::audioadapter_buffers::direct::InterleavedSlice;

        let needed = self.resampler.input_frames_next();
        let mut padded;
        let input = if chunk.len() < needed {
            padded = chunk.to_vec();
            padded.resize(needed, 0.0);
            &padded[..]
        } else {
            chunk
        };
        let adapter = InterleavedSlice::new(input, 1, input.len())
            .map_err(|e| Error::audio(format!("Invalid resampler input: {}", e)))?;
        let indexing = partial.map(|len| Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(len),
            active_channels_mask: None,
        });
        let output = self
            .resampler
            .process(&adapter, indexing.as_ref())
            .map_err(|e| Error::audio(format!("Resampling failed: {}", e)))?;
        Ok(output.take_data())
    }

    /// Drop the remaining resampler delay from the front of `out`.
    fn trim(&mut self, mut out: Vec<f32>) -> Vec<f32> {
        let drop = self.delay.min(out.len());
        out.drain(..drop);
        self.delay -= drop;
        self.output_frames += out.len();
        out
    }
}

#[cfg(feature = "asr")]
impl std::fmt::Debug for StreamResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamResampler")
            .field("ratio", &self.ratio)
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "asr"))]
mod tests {
    use super::*;

    fn sine(rate: u32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_stream_resampler_matches_whole_clip() {
        let input = sine(48_000, 1.3);
        let whole = resample(&input, 48_000, 16_000).unwrap();

        let mut stream = StreamResampler::new(48_000, 16_000).unwrap();
        let mut pieces = Vec::new();
        for piece in input.chunks(777) {
            pieces.extend(stream.process(piece).unwrap());
            assert!(stream.pending() < 1024);
        }
        pieces.extend(stream.flush().unwrap());

        assert_eq!(stream.pending(), 0);
        assert_eq!(pieces.len(), whole.len());
        let max_diff = pieces
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max diff {max_diff}");
    }
}
