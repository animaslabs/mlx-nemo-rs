//! Audio processing primitives for NeMo models.
//!
//! This module provides composable signal processing functions for converting
//! raw audio waveforms to mel-frequency spectrogram features suitable for
//! speech recognition models.
//!
//! # Architecture
//!
//! The module is organized into layers:
//!
//! 1. **Pure functions** - Core signal processing primitives that can be composed
//! 2. **`MelProcessor`** - Caches filterbank/window for repeated use with same config  
//! 3. **`MelStream`** - Adds streaming state (audio buffer, preemphasis state)
//!
//! # Quick Start (Batch Processing)
//!
//! ```rust,ignore
//! use mlx_nemo::audio::{mel_spectrogram, MelProcessor};
//! use mlx_nemo::asr::config::PreprocessorConfig;
//!
//! // Option 1: Pure function (recomputes filterbank each call)
//! let config = PreprocessorConfig::default();
//! let mel = mel_spectrogram(&audio, &config)?;
//!
//! // Option 2: Cached processor (amortizes filterbank computation)
//! let processor = MelProcessor::new(config)?;
//! let mel1 = processor.forward(&audio1)?;
//! let mel2 = processor.forward(&audio2)?;
//! ```
//!
//! # Streaming
//!
//! ```rust,ignore
//! use mlx_nemo::audio::MelStream;
//!
//! let mut stream = MelStream::new(config)?;
//! loop {
//!     let chunk = get_audio_chunk();
//!     if let Some(mel) = stream.push(&chunk)? {
//!         // Process mel frames [n_mels, new_frames]
//!     }
//! }
//! let final_mel = stream.flush()?;
//! ```
//!
//! # Output Format
//!
//! All functions output mel spectrograms in NeMo format: `[batch, n_mels, frames]`
//! This matches what the Conformer encoder expects.

use std::f32::consts::PI;
use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fft, ops, Array};

use crate::asr::config::PreprocessorConfig;
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
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1)
        .map_err(|e| Error::audio(format!("Failed to create resampler: {}", e)))?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| Error::audio(format!("Resampling failed: {}", e)))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}

#[cfg(not(feature = "asr"))]
pub fn resample(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
    Err(Error::not_implemented(
        "Audio resampling requires the 'asr' feature",
    ))
}

// ============================================================================
// Signal Processing Primitives
// ============================================================================

/// Pre-emphasis filter: y[n] = x[n] - coef * x[n-1]
///
/// Boosts high frequencies to improve consonant recognition.
/// NeMo default coefficient is 0.97.
///
/// # Arguments
/// * `audio` - Input audio `[batch, time]` or `[time]`
/// * `coef` - Pre-emphasis coefficient (typically 0.97)
///
/// # Returns
/// Pre-emphasized audio, same shape as input
pub fn preemphasis(audio: &Array, coef: f32) -> Result<Array> {
    if coef <= 0.0 {
        return Ok(audio.clone());
    }

    let shape = audio.shape();
    let ndim = shape.len();

    // Handle both 1D [time] and 2D [batch, time]
    let (work_audio, was_1d) = if ndim == 1 {
        (audio.reshape(&[1, -1])?, true)
    } else {
        (audio.clone(), false)
    };

    let time_len = work_audio.shape()[1];
    if time_len <= 1 {
        return if was_1d {
            Ok(work_audio.squeeze_axes(&[0])?)
        } else {
            Ok(work_audio)
        };
    }

    // y[n] = x[n] - coef * x[n-1]
    // First sample unchanged, rest filtered
    let first = work_audio.index((.., ..1));
    let current = work_audio.index((.., 1..));
    let previous = work_audio.index((.., ..(time_len - 1)));

    let coef_arr = Array::from_f32(coef);
    let scaled = ops::multiply(&previous, &coef_arr)?;
    let filtered = ops::subtract(&current, &scaled)?;
    let result = ops::concatenate_axis(&[&first, &filtered], 1)?;

    if was_1d {
        Ok(result.squeeze_axes(&[0])?)
    } else {
        Ok(result)
    }
}

/// Pre-emphasis with state tracking for streaming.
///
/// # Arguments
/// * `audio` - Input audio `[time]` (1D only for streaming)
/// * `coef` - Pre-emphasis coefficient
/// * `prev_sample` - Last sample from previous chunk (x[n-1] for first sample)
///
/// # Returns
/// (filtered_audio, last_sample) where last_sample should be passed as prev_sample next call
pub fn preemphasis_streaming(audio: &Array, coef: f32, prev_sample: f32) -> Result<(Array, f32)> {
    if coef <= 0.0 {
        audio.eval()?;
        let data: &[f32] = audio.as_slice();
        let last = *data.last().unwrap_or(&0.0);
        return Ok((audio.clone(), last));
    }

    let len = audio.shape()[0];
    if len == 0 {
        return Ok((audio.clone(), prev_sample));
    }

    // For streaming, we need CPU access to track state
    audio.eval()?;
    let input: &[f32] = audio.as_slice();

    let mut output = Vec::with_capacity(len as usize);

    // First sample uses prev_sample from previous chunk
    output.push(input[0] - coef * prev_sample);

    // Rest of samples
    for i in 1..len as usize {
        output.push(input[i] - coef * input[i - 1]);
    }

    let last = *input.last().unwrap();
    let result = Array::from_slice(&output, &[len]);

    Ok((result, last))
}

/// Create a window function.
///
/// # Arguments
/// * `window_type` - "hann" or "hamming"
/// * `length` - Window length in samples
///
/// # Returns
/// Window array `[length]`
pub fn window(window_type: &str, length: usize) -> Result<Array> {
    let mut values = Vec::with_capacity(length);
    let n = length as f32;

    match window_type {
        "hann" => {
            for i in 0..length {
                let value = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1.0)).cos());
                values.push(value);
            }
        }
        "hamming" => {
            for i in 0..length {
                let value = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1.0)).cos();
                values.push(value);
            }
        }
        _ => {
            return Err(Error::config(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    }

    Ok(Array::from_slice(&values, &[length as i32]))
}

/// Create a window function with centered zero-padding to n_fft.
///
/// Used by STFT implementations where the window needs to be zero-padded
/// to match the FFT size.
///
/// # Arguments
/// * `window_type` - "hann" or "hamming"
/// * `win_length` - Window length in samples
/// * `n_fft` - FFT size (output length)
///
/// # Returns
/// Window array `[n_fft]` with centered zero-padding
pub fn window_padded(window_type: &str, win_length: usize, n_fft: usize) -> Result<Array> {
    if win_length >= n_fft {
        return window(window_type, win_length);
    }

    // Compute window values directly without creating intermediate array
    let win_values: Vec<f32> = match window_type {
        "hann" => (0..win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (win_length as f32 - 1.0)).cos()))
            .collect(),
        "hamming" => (0..win_length)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (win_length as f32 - 1.0)).cos())
            .collect(),
        _ => {
            return Err(Error::config(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    };

    // Center the window with zero-padding
    let left_pad = (n_fft - win_length) / 2;

    let mut padded = vec![0.0f32; n_fft];
    padded[left_pad..left_pad + win_length].copy_from_slice(&win_values);

    Ok(Array::from_slice(&padded, &[n_fft as i32]))
}

/// Compute Short-Time Fourier Transform power spectrum.
///
/// Uses GPU-side `as_strided` for efficient framing without CPU copies.
///
/// # Arguments
/// * `audio` - Input audio `[batch, time]` or `[time]`
/// * `n_fft` - FFT size
/// * `hop_length` - Hop between frames (samples)
/// * `win_length` - Window length (samples)
/// * `win` - Window function `[win_length]`
/// * `center` - If true, pad audio so frames are centered
///
/// # Returns
/// Power spectrum `[batch, n_fft/2+1, frames]`
pub fn stft_power(
    audio: &Array,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    win: &Array,
    center: bool,
) -> Result<Array> {
    let shape = audio.shape();
    let ndim = shape.len();

    // Handle 1D vs 2D input
    let (work_audio, was_1d) = if ndim == 1 {
        (audio.reshape(&[1, -1])?, true)
    } else {
        (audio.clone(), false)
    };

    let batch_size = work_audio.shape()[0] as usize;
    let _audio_len = work_audio.shape()[1] as usize;

    // Apply center padding if requested
    let padded = if center {
        apply_center_padding(&work_audio, n_fft)?
    } else {
        work_audio
    };

    let padded_len = padded.shape()[1] as usize;

    // Frame size is n_fft when centering, win_length otherwise
    let frame_size = if center { n_fft } else { win_length };

    if padded_len < frame_size {
        return Err(Error::audio(format!(
            "Audio too short: {} samples, need at least {}",
            padded_len, frame_size
        )));
    }

    // Number of frames
    let num_frames = 1 + (padded_len - frame_size) / hop_length;

    // Process each batch item separately (as_strided doesn't batch well)
    let mut batch_results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let audio_slice = padded.index((b as i32, ..));

        // Use as_strided for efficient GPU-side framing
        let frames = ops::as_strided(
            &audio_slice,
            &[num_frames as i32, frame_size as i32],
            &[hop_length as i64, 1],
            0,
        )?;

        // Prepare window (zero-pad if needed)
        let padded_window = if win_length < frame_size {
            let zeros = Array::zeros::<f32>(&[(frame_size - win_length) as i32])?;
            ops::concatenate_axis(&[win, &zeros], 0)?
        } else {
            win.clone()
        };

        // Apply window: [num_frames, frame_size] * [frame_size]
        let windowed = ops::multiply(&frames, &padded_window)?;

        // Compute rfft along last axis
        let fft_result = fft::rfft(&windowed, Some(n_fft as i32), Some(-1))?;

        // Power spectrum: |FFT|^2
        let real = ops::real(&fft_result)?;
        let imag = ops::imag(&fft_result)?;
        let power = ops::add(&ops::square(&real)?, &ops::square(&imag)?)?;

        // Transpose to [freq, frames] for this batch
        let power_t = ops::transpose(&power)?;

        batch_results.push(power_t);
    }

    // Stack batch results: [batch, freq, frames]
    let result = if batch_size == 1 {
        batch_results[0].reshape(&[1, -1, num_frames as i32])?
    } else {
        let refs: Vec<&Array> = batch_results.iter().collect();
        ops::stack_axis(&refs, 0)?
    };

    if was_1d {
        Ok(result.squeeze_axes(&[0])?)
    } else {
        Ok(result)
    }
}

/// Apply center padding with reflection.
fn apply_center_padding(audio: &Array, n_fft: usize) -> Result<Array> {
    let batch_size = audio.shape()[0] as usize;
    let audio_len = audio.shape()[1] as usize;
    let pad_size = n_fft / 2;

    if audio_len < pad_size + 1 {
        return Err(Error::audio(format!(
            "Audio too short for center padding: {} samples, need at least {}",
            audio_len,
            pad_size + 1
        )));
    }

    // Process each batch item
    let mut batch_results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let audio_slice = audio.index((b as i32, ..));
        let len = audio_slice.shape()[0];

        // Reflect padding on GPU using as_strided for reversal
        // prefix = x[1:pad+1] reversed
        let prefix_slice = audio_slice.index(1..(pad_size as i32 + 1));
        let prefix = ops::as_strided(&prefix_slice, &[pad_size as i32], &[-1_i64], pad_size - 1)?;

        // suffix = x[-(pad+1):-1] reversed
        let suffix_start = len - (pad_size as i32 + 1);
        let suffix_end = len - 1;
        let suffix_slice = audio_slice.index(suffix_start..suffix_end);
        let suffix = ops::as_strided(&suffix_slice, &[pad_size as i32], &[-1_i64], pad_size - 1)?;

        // Concatenate: [prefix, audio, suffix]
        let padded = ops::concatenate_axis(&[&prefix, &audio_slice, &suffix], 0)?;
        batch_results.push(padded);
    }

    // Stack batch
    if batch_size == 1 {
        Ok(batch_results[0].reshape(&[1, -1])?)
    } else {
        let refs: Vec<&Array> = batch_results.iter().collect();
        ops::stack_axis(&refs, 0)
            .map_err(|e| Error::audio(format!("Failed to stack padded audio: {}", e)))
    }
}

/// Create Slaney-normalized mel filterbank.
///
/// Uses Slaney normalization (area normalization) to ensure consistent
/// energy across mel bands, matching NeMo's torchaudio mel filterbank.
///
/// # Arguments
/// * `n_mels` - Number of mel bins
/// * `n_fft` - FFT size
/// * `sample_rate` - Sample rate in Hz
/// * `f_min` - Minimum frequency
/// * `f_max` - Maximum frequency
///
/// # Returns
/// Filterbank matrix `[n_mels, n_fft/2+1]`
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Result<Array> {
    let n_freqs = n_fft / 2 + 1;

    // Slaney mel scale constants
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.06875177742094912; // ln(6.4) / 27.0

    let hz_to_mel = |hz: f32| -> f32 {
        if hz >= MIN_LOG_HZ {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
        } else {
            hz / F_SP
        }
    };

    let mel_to_hz = |mel: f32| -> f32 {
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
        } else {
            mel * F_SP
        }
    };

    // FFT frequencies: i * sample_rate / n_fft
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // Mel points
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Frequency differences for normalization
    let fdiff: Vec<f32> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

    // Build filterbank with Slaney normalization
    let mut filterbank = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_right = hz_points[m + 2];

        // Slaney normalization factor: 2 / (f_right - f_left)
        let enorm = 2.0 / (f_right - f_left);

        for (k, &freq) in fft_freqs.iter().enumerate() {
            // Triangular filter with proper slopes
            let lower = (freq - f_left) / fdiff[m];
            let upper = (f_right - freq) / fdiff[m + 1];
            let value = lower.min(upper).max(0.0) * enorm;
            filterbank[m * n_freqs + k] = value;
        }
    }

    Ok(Array::from_slice(
        &filterbank,
        &[n_mels as i32, n_freqs as i32],
    ))
}

/// Apply mel filterbank to power spectrum.
///
/// # Arguments
/// * `power` - Power spectrum `[batch, freq_bins, frames]` or `[freq_bins, frames]`
/// * `filterbank` - Mel filterbank `[n_mels, freq_bins]`
///
/// # Returns
/// Mel spectrogram `[batch, n_mels, frames]` or `[n_mels, frames]`
pub fn apply_mel_filterbank(power: &Array, filterbank: &Array) -> Result<Array> {
    // filterbank: [n_mels, freq]
    // power: [batch, freq, frames] or [freq, frames]
    // output: [batch, n_mels, frames] or [n_mels, frames]
    //
    // Use einsum: "mf,bft->bmt" for batched, "mf,ft->mt" for unbatched

    let ndim = power.ndim();
    if ndim == 2 {
        // [freq, frames] -> [n_mels, frames]
        ops::einsum("mf,ft->mt", &[filterbank.clone(), power.clone()])
            .map_err(|e| Error::audio(format!("Mel filterbank application failed: {}", e)))
    } else {
        // [batch, freq, frames] -> [batch, n_mels, frames]
        ops::einsum("mf,bft->bmt", &[filterbank.clone(), power.clone()])
            .map_err(|e| Error::audio(format!("Mel filterbank application failed: {}", e)))
    }
}

/// Log compression with zero guard.
///
/// Computes `log(x + guard)` to avoid log(0).
/// NeMo default guard is 2^-24 ≈ 5.96e-8.
///
/// # Arguments
/// * `mel` - Mel spectrogram
/// * `guard` - Small constant to add before log
///
/// # Returns
/// Log-compressed mel spectrogram
pub fn log_compress(mel: &Array, guard: f32) -> Result<Array> {
    let guard_arr = Array::from_f32(guard);
    let mel_safe = ops::add(mel, &guard_arr)?;
    ops::log(&mel_safe).map_err(|e| Error::audio(format!("Log compression failed: {}", e)))
}

/// Per-feature normalization (zero mean, unit variance per mel bin).
///
/// For input `[batch, n_mels, frames]`, normalizes along the frames axis
/// independently for each mel bin.
///
/// # Arguments
/// * `features` - Input features `[batch, n_mels, frames]`
///
/// # Returns
/// Normalized features, same shape
pub fn normalize_per_feature(features: &Array) -> Result<Array> {
    // Normalize along axis 2 (frames) for [batch, n_mels, frames]
    // mean/std shape: [batch, n_mels, 1]

    let mean = ops::mean_axis(features, 2, true)?;
    let centered = ops::subtract(features, &mean)?;

    let squared = ops::square(&centered)?;
    let variance = ops::mean_axis(&squared, 2, true)?;

    let eps = Array::from_f32(1e-5);
    let var_safe = ops::add(&variance, &eps)?;
    let std = ops::sqrt(&var_safe)?;

    ops::divide(&centered, &std).map_err(|e| Error::audio(format!("Normalization failed: {}", e)))
}

/// Normalize with provided mean and std statistics.
///
/// # Arguments
/// * `features` - Input features `[batch, n_mels, frames]`
/// * `mean` - Per-feature mean `[n_mels]` or `[1, n_mels, 1]`
/// * `std` - Per-feature std `[n_mels]` or `[1, n_mels, 1]`
///
/// # Returns
/// Normalized features
pub fn normalize_with_stats(features: &Array, mean: &Array, std: &Array) -> Result<Array> {
    // Reshape mean/std to [1, n_mels, 1] if needed
    let mean = if mean.ndim() == 1 {
        mean.reshape(&[1, mean.dim(0) as i32, 1])?
    } else {
        mean.clone()
    };

    let std = if std.ndim() == 1 {
        std.reshape(&[1, std.dim(0) as i32, 1])?
    } else {
        std.clone()
    };

    let centered = ops::subtract(features, &mean)?;
    let eps = Array::from_f32(1e-5);
    let std_safe = ops::add(&std, &eps)?;

    ops::divide(&centered, &std_safe)
        .map_err(|e| Error::audio(format!("Normalization with stats failed: {}", e)))
}

// ============================================================================
// High-Level API
// ============================================================================

/// Compute log-mel spectrogram from raw audio.
///
/// This is the main entry point that chains all primitives:
/// preemphasis → STFT → mel filterbank → log → normalize
///
/// # Arguments
/// * `audio` - Raw audio `[batch, time]` or `[time]`
/// * `config` - NeMo preprocessor configuration
///
/// # Returns
/// Log-mel features `[batch, n_mels, frames]`
pub fn mel_spectrogram(audio: &Array, config: &PreprocessorConfig) -> Result<Array> {
    // Derive parameters from config
    let sample_rate = config.sample_rate as f32;
    let win_length = (config.window_size * sample_rate) as usize;
    let hop_length = (config.window_stride * sample_rate) as usize;
    let n_fft = config.n_fft;
    let n_mels = config.features;
    let f_min = config.lowfreq.unwrap_or(0.0);
    let f_max = config.highfreq.unwrap_or(sample_rate / 2.0);

    // Ensure 2D input [batch, time]
    let (audio, was_1d) = if audio.ndim() == 1 {
        (audio.reshape(&[1, -1])?, true)
    } else {
        (audio.clone(), false)
    };

    // 1. Preemphasis
    let audio = if let Some(coef) = config.preemph {
        preemphasis(&audio, coef)?
    } else {
        audio
    };

    // 2. Create window
    let win = window(&config.window, win_length)?;

    // 3. STFT power spectrum
    let power = stft_power(&audio, n_fft, hop_length, win_length, &win, true)?;

    // 4. Mel filterbank
    let fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)?;
    let mel = apply_mel_filterbank(&power, &fb)?;

    // 5. Log compression (NeMo uses 2^-24 guard)
    let log_mel = log_compress(&mel, 2.0_f32.powi(-24))?;

    // 6. Normalization
    let output = match config.normalize.as_str() {
        "per_feature" => normalize_per_feature(&log_mel)?,
        "NA" | "" => log_mel,
        _ => log_mel,
    };

    if was_1d {
        // Keep batch dim for consistency with NeMo
        Ok(output)
    } else {
        Ok(output)
    }
}

// ============================================================================
// MelProcessor - Cached filterbank and window
// ============================================================================

/// Mel spectrogram processor with cached filterbank and window.
///
/// Use this when processing multiple audio files with the same config
/// to avoid recomputing the filterbank each time.
#[derive(Debug)]
pub struct MelProcessor {
    config: PreprocessorConfig,
    filterbank: Array,
    win: Array,
    win_length: usize,
    hop_length: usize,
}

impl MelProcessor {
    /// Create a new processor with the given configuration.
    pub fn new(config: PreprocessorConfig) -> Result<Self> {
        let sample_rate = config.sample_rate as f32;
        let win_length = (config.window_size * sample_rate) as usize;
        let hop_length = (config.window_stride * sample_rate) as usize;
        let n_fft = config.n_fft;
        let n_mels = config.features;
        let f_min = config.lowfreq.unwrap_or(0.0);
        let f_max = config.highfreq.unwrap_or(sample_rate / 2.0);

        let win = window(&config.window, win_length)?;
        let filterbank = mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)?;

        Ok(Self {
            config,
            filterbank,
            win,
            win_length,
            hop_length,
        })
    }

    /// Process audio to mel spectrogram.
    ///
    /// # Arguments
    /// * `audio` - Raw audio `[batch, time]` or `[time]`
    ///
    /// # Returns
    /// Log-mel features `[batch, n_mels, frames]`
    pub fn forward(&self, audio: &Array) -> Result<Array> {
        let (audio, _was_1d) = if audio.ndim() == 1 {
            (audio.reshape(&[1, -1])?, true)
        } else {
            (audio.clone(), false)
        };

        // Preemphasis
        let audio = if let Some(coef) = self.config.preemph {
            preemphasis(&audio, coef)?
        } else {
            audio
        };

        // STFT
        let power = stft_power(
            &audio,
            self.config.n_fft,
            self.hop_length,
            self.win_length,
            &self.win,
            true,
        )?;

        // Mel filterbank
        let mel = apply_mel_filterbank(&power, &self.filterbank)?;

        // Log
        let log_mel = log_compress(&mel, 2.0_f32.powi(-24))?;

        // Normalize
        match self.config.normalize.as_str() {
            "per_feature" => normalize_per_feature(&log_mel),
            _ => Ok(log_mel),
        }
    }

    /// Process without preemphasis (for streaming with external state tracking).
    pub fn forward_no_preemph(&self, audio: &Array) -> Result<Array> {
        let (audio, _was_1d) = if audio.ndim() == 1 {
            (audio.reshape(&[1, -1])?, true)
        } else {
            (audio.clone(), false)
        };

        let power = stft_power(
            &audio,
            self.config.n_fft,
            self.hop_length,
            self.win_length,
            &self.win,
            true,
        )?;

        let mel = apply_mel_filterbank(&power, &self.filterbank)?;
        let log_mel = log_compress(&mel, 2.0_f32.powi(-24))?;

        // Skip normalization for streaming (applied later with running stats)
        Ok(log_mel)
    }

    /// Get the configuration.
    pub fn config(&self) -> &PreprocessorConfig {
        &self.config
    }

    /// Get the cached filterbank.
    pub fn filterbank(&self) -> &Array {
        &self.filterbank
    }

    /// Get hop length in samples.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Get window length in samples.
    pub fn win_length(&self) -> usize {
        self.win_length
    }

    /// Calculate expected number of output frames for given audio length.
    ///
    /// With center padding: frames = audio_len / hop_length + 1
    pub fn num_frames(&self, audio_len: usize) -> usize {
        // With center padding, formula is simpler
        audio_len / self.hop_length + 1
    }
}

// ============================================================================
// MelStream - Streaming processor
// ============================================================================

/// Streaming mel spectrogram processor.
///
/// Handles audio buffering, STFT overlap, and preemphasis state
/// for continuous audio streams.
pub struct MelStream {
    processor: MelProcessor,
    audio_buffer: Vec<f32>,
    preemph_state: f32,
    total_frames: u64,
}

impl MelStream {
    /// Create a new streaming processor.
    pub fn new(config: PreprocessorConfig) -> Result<Self> {
        let processor = MelProcessor::new(config)?;
        Ok(Self {
            processor,
            audio_buffer: Vec::new(),
            preemph_state: 0.0,
            total_frames: 0,
        })
    }

    /// Push audio samples and compute mel frames.
    ///
    /// # Returns
    /// Mel features `[1, n_mels, new_frames]` or None if not enough audio
    pub fn push(&mut self, samples: &[f32]) -> Result<Option<Array>> {
        self.audio_buffer.extend_from_slice(samples);

        // Need at least enough for one frame with center padding
        let min_samples = self.processor.hop_length;
        if self.audio_buffer.len() < min_samples {
            return Ok(None);
        }

        // Apply preemphasis with state tracking
        let audio = Array::from_slice(&self.audio_buffer, &[self.audio_buffer.len() as i32]);
        let preemph_coef = self.processor.config.preemph.unwrap_or(0.0);

        let (preemph_audio, new_state) =
            preemphasis_streaming(&audio, preemph_coef, self.preemph_state)?;

        // Compute mel spectrogram
        let mel = self.processor.forward_no_preemph(&preemph_audio)?;

        // Update state
        self.preemph_state = new_state;

        // Calculate frames produced and samples consumed
        let frames_produced = mel.shape()[2] as usize;
        self.total_frames += frames_produced as u64;

        // Keep overlap for next chunk (enough for STFT window)
        let samples_consumed = frames_produced * self.processor.hop_length;
        let keep = self
            .audio_buffer
            .len()
            .saturating_sub(samples_consumed)
            .max(self.processor.win_length);
        let drain_count = self.audio_buffer.len().saturating_sub(keep);
        self.audio_buffer.drain(..drain_count);

        Ok(Some(mel))
    }

    /// Flush remaining audio at end of stream.
    pub fn flush(&mut self) -> Result<Option<Array>> {
        if self.audio_buffer.is_empty() {
            return Ok(None);
        }

        // Process remaining audio
        let audio = Array::from_slice(&self.audio_buffer, &[self.audio_buffer.len() as i32]);
        let preemph_coef = self.processor.config.preemph.unwrap_or(0.0);
        let (preemph_audio, _) = preemphasis_streaming(&audio, preemph_coef, self.preemph_state)?;

        let mel = self.processor.forward_no_preemph(&preemph_audio)?;

        self.audio_buffer.clear();
        self.preemph_state = 0.0;

        let frames = mel.shape()[2] as u64;
        self.total_frames += frames;

        Ok(Some(mel))
    }

    /// Total frames computed since creation.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Reset for new audio stream.
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.preemph_state = 0.0;
        self.total_frames = 0;
    }

    /// Get the underlying processor.
    pub fn processor(&self) -> &MelProcessor {
        &self.processor
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PreprocessorConfig {
        PreprocessorConfig {
            sample_rate: 16000,
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            features: 80,
            n_fft: 512,
            log: true,
            normalize: "per_feature".to_string(),
            dither: 0.0,
            pad_to: 0,
            preemph: Some(0.97),
            lowfreq: Some(0.0),
            highfreq: None,
            target: None,
        }
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ======================== Window Tests ========================

    #[test]
    fn test_hann_window_shape() {
        let win = window("hann", 400).unwrap();
        assert_eq!(win.shape(), &[400]);
    }

    #[test]
    fn test_hann_window_endpoints() {
        let win = window("hann", 400).unwrap();
        win.eval().unwrap();
        let data: &[f32] = win.as_slice();

        assert!(approx_eq(data[0], 0.0, 1e-6), "First should be ~0");
        assert!(approx_eq(data[399], 0.0, 1e-6), "Last should be ~0");
    }

    #[test]
    fn test_hann_window_peak() {
        let win = window("hann", 401).unwrap();
        win.eval().unwrap();
        let data: &[f32] = win.as_slice();

        assert!(approx_eq(data[200], 1.0, 1e-6), "Center should be ~1");
    }

    #[test]
    fn test_hann_window_symmetry() {
        let win = window("hann", 400).unwrap();
        win.eval().unwrap();
        let data: &[f32] = win.as_slice();

        for i in 0..200 {
            assert!(
                approx_eq(data[i], data[399 - i], 1e-6),
                "Window should be symmetric"
            );
        }
    }

    #[test]
    fn test_hamming_window() {
        let win = window("hamming", 400).unwrap();
        win.eval().unwrap();
        let data: &[f32] = win.as_slice();

        // Hamming starts at ~0.08, not 0
        assert!(approx_eq(data[0], 0.08, 1e-4));
        assert!(approx_eq(data[399], 0.08, 1e-4));
    }

    #[test]
    fn test_unknown_window_error() {
        let result = window("blackman", 400);
        assert!(result.is_err());
    }

    // ======================== Preemphasis Tests ========================

    #[test]
    fn test_preemphasis_formula() {
        let audio = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
        let result = preemphasis(&audio, 0.97).unwrap();
        result.eval().unwrap();
        let data: &[f32] = result.as_slice();

        // y[0] = x[0] = 1.0
        // y[1] = 2.0 - 0.97*1.0 = 1.03
        // y[2] = 3.0 - 0.97*2.0 = 1.06
        assert!(approx_eq(data[0], 1.0, 1e-5));
        assert!(approx_eq(data[1], 1.03, 1e-5));
        assert!(approx_eq(data[2], 1.06, 1e-5));
    }

    #[test]
    fn test_preemphasis_disabled() {
        let audio = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let result = preemphasis(&audio, 0.0).unwrap();
        result.eval().unwrap();

        let input: &[f32] = audio.as_slice();
        let output: &[f32] = result.as_slice();

        for i in 0..3 {
            assert!(approx_eq(input[i], output[i], 1e-6));
        }
    }

    #[test]
    fn test_preemphasis_streaming() {
        // Test that streaming preemphasis maintains continuity
        let chunk1 = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let chunk2 = Array::from_slice(&[4.0f32, 5.0, 6.0], &[3]);

        let (out1, state1) = preemphasis_streaming(&chunk1, 0.97, 0.0).unwrap();
        let (out2, _state2) = preemphasis_streaming(&chunk2, 0.97, state1).unwrap();

        // Process full sequence for comparison
        let full = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let full_out = preemphasis(&full, 0.97).unwrap();

        out1.eval().unwrap();
        out2.eval().unwrap();
        full_out.eval().unwrap();

        let d1: &[f32] = out1.as_slice();
        let d2: &[f32] = out2.as_slice();
        let df: &[f32] = full_out.as_slice();

        // Check first chunk matches
        for i in 0..3 {
            assert!(approx_eq(d1[i], df[i], 1e-5), "Chunk1 mismatch at {}", i);
        }

        // Check second chunk matches (with state continuity)
        for i in 0..3 {
            assert!(
                approx_eq(d2[i], df[i + 3], 1e-5),
                "Chunk2 mismatch at {}: {} vs {}",
                i,
                d2[i],
                df[i + 3]
            );
        }
    }

    // ======================== Filterbank Tests ========================

    #[test]
    fn test_filterbank_shape() {
        let fb = mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        assert_eq!(fb.shape(), &[80, 257]); // n_mels x (n_fft/2 + 1)
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb = mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: &[f32] = fb.as_slice();

        for &val in data {
            assert!(val >= 0.0, "Filterbank should be non-negative");
        }
    }

    #[test]
    fn test_filterbank_triangular() {
        let fb = mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: &[f32] = fb.as_slice();

        // Each filter should have max <= 1.0
        let n_freqs = 257;
        for m in 0..80 {
            let filter = &data[m * n_freqs..(m + 1) * n_freqs];
            let max_val = filter.iter().cloned().fold(0.0f32, f32::max);
            assert!(max_val <= 1.0 + 1e-6, "Filter {} max {} > 1.0", m, max_val);
        }
    }

    // ======================== STFT Tests ========================

    #[test]
    fn test_stft_output_shape() {
        let _config = default_config();
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[1, 16000]);
        let win = window("hann", 400).unwrap();

        let power = stft_power(&audio, 512, 160, 400, &win, true).unwrap();

        // [batch, freq, frames] = [1, 257, 101]
        // frames = 16000 / 160 + 1 = 101
        assert_eq!(power.shape()[0], 1);
        assert_eq!(power.shape()[1], 257);
        assert_eq!(power.shape()[2], 101);
    }

    #[test]
    fn test_stft_non_negative() {
        let audio = Array::from_slice(&vec![0.1f32; 4000], &[4000]);
        let win = window("hann", 400).unwrap();

        let power = stft_power(&audio, 512, 160, 400, &win, true).unwrap();
        power.eval().unwrap();
        let data: &[f32] = power.as_slice();

        for &val in data {
            assert!(val >= 0.0, "Power spectrum should be non-negative");
        }
    }

    // ======================== Log Compression Tests ========================

    #[test]
    fn test_log_compress_finite() {
        let mel = Array::from_slice(&[0.0f32, 0.001, 0.01, 0.1, 1.0], &[1, 5]);
        let log_mel = log_compress(&mel, 2.0_f32.powi(-24)).unwrap();
        log_mel.eval().unwrap();
        let data: &[f32] = log_mel.as_slice();

        for &val in data {
            assert!(val.is_finite(), "Log should be finite");
        }
    }

    // ======================== Normalization Tests ========================

    #[test]
    fn test_normalize_zero_mean() {
        // Create data with known statistics
        let mut data = Vec::with_capacity(80 * 100);
        for i in 0..80 {
            for j in 0..100 {
                data.push(i as f32 + j as f32 * 0.1);
            }
        }
        let features = Array::from_slice(&data, &[1, 80, 100]);

        let normalized = normalize_per_feature(&features).unwrap();
        normalized.eval().unwrap();

        // Check mean is ~0 for each mel bin
        let mean = ops::mean_axis(&normalized, 2, true).unwrap();
        mean.eval().unwrap();
        let mean_data: &[f32] = mean.as_slice();

        for (i, &m) in mean_data.iter().enumerate() {
            assert!(
                approx_eq(m, 0.0, 1e-4),
                "Mean of feature {} should be ~0, got {}",
                i,
                m
            );
        }
    }

    // ======================== Full Pipeline Tests ========================

    #[test]
    fn test_mel_spectrogram_output_shape() {
        let config = default_config();
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[1, 16000]);

        let mel = mel_spectrogram(&audio, &config).unwrap();

        // [batch, n_mels, frames] = [1, 80, 101]
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 80);
        assert_eq!(mel.shape()[2], 101);
    }

    #[test]
    fn test_mel_spectrogram_1d_input() {
        let config = default_config();
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[16000]);

        let mel = mel_spectrogram(&audio, &config).unwrap();

        // Should still get batch dim
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 80);
    }

    #[test]
    fn test_mel_spectrogram_deterministic() {
        let config = default_config();
        let samples: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 8000]);

        let mel1 = mel_spectrogram(&audio, &config).unwrap();
        let mel2 = mel_spectrogram(&audio, &config).unwrap();

        mel1.eval().unwrap();
        mel2.eval().unwrap();

        let d1: &[f32] = mel1.as_slice();
        let d2: &[f32] = mel2.as_slice();

        for i in 0..d1.len() {
            assert!(approx_eq(d1[i], d2[i], 1e-6), "Should be deterministic");
        }
    }

    #[test]
    fn test_mel_spectrogram_finite() {
        let config = default_config();
        // Include silence
        let audio = Array::from_slice(&vec![0.0f32; 16000], &[16000]);

        let mel = mel_spectrogram(&audio, &config).unwrap();
        mel.eval().unwrap();
        let data: &[f32] = mel.as_slice();

        for &val in data {
            assert!(val.is_finite(), "All values should be finite");
        }
    }

    // ======================== MelProcessor Tests ========================

    #[test]
    fn test_processor_creation() {
        let config = default_config();
        let processor = MelProcessor::new(config).unwrap();

        assert_eq!(processor.hop_length(), 160);
        assert_eq!(processor.win_length(), 400);
    }

    #[test]
    fn test_processor_forward() {
        let config = default_config();
        let processor = MelProcessor::new(config).unwrap();

        let audio = Array::from_slice(&vec![0.1f32; 16000], &[1, 16000]);
        let mel = processor.forward(&audio).unwrap();

        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 80);
        assert_eq!(mel.shape()[2], 101);
    }

    #[test]
    fn test_processor_num_frames() {
        let config = default_config();
        let processor = MelProcessor::new(config).unwrap();

        // 16000 samples -> 16000/160 + 1 = 101 frames
        assert_eq!(processor.num_frames(16000), 101);

        // 8000 samples -> 8000/160 + 1 = 51 frames
        assert_eq!(processor.num_frames(8000), 51);
    }

    // ======================== MelStream Tests ========================

    #[test]
    fn test_stream_creation() {
        let config = default_config();
        let stream = MelStream::new(config).unwrap();

        assert_eq!(stream.total_frames(), 0);
    }

    #[test]
    fn test_stream_push() {
        let config = default_config();
        let mut stream = MelStream::new(config).unwrap();

        // Push 1 second of audio
        let samples = vec![0.1f32; 16000];
        let result = stream.push(&samples).unwrap();

        assert!(result.is_some());
        let mel = result.unwrap();
        assert_eq!(mel.shape()[1], 80); // n_mels
        assert!(stream.total_frames() > 0);
    }

    #[test]
    fn test_stream_reset() {
        let config = default_config();
        let mut stream = MelStream::new(config).unwrap();

        stream.push(&vec![0.1f32; 16000]).unwrap();
        assert!(stream.total_frames() > 0);

        stream.reset();
        assert_eq!(stream.total_frames(), 0);
    }
}
