//! Common DSP primitives for audio processing.
//!
//! MLX-based functions for signal processing used throughout the ASR pipeline.
//! Data stays on GPU to avoid CPU/GPU copies.

use std::f32::consts::PI;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, ops};

use crate::core::error::{Error, Result};

// ============================================================================
// Window Functions
// ============================================================================

/// Create Hann window of given length as MLX array.
///
/// The Hann window is defined as:
/// `w[n] = 0.5 * (1 - cos(2πn / (N-1)))`
///
/// # Arguments
/// * `length` - Window length in samples
///
/// # Returns
/// MLX Array of shape `[length]` with window coefficients
pub fn hann_window(length: usize) -> Result<Array> {
    if length == 0 {
        return Ok(Array::zeros::<f32>(&[0])?);
    }
    if length == 1 {
        return Ok(Array::from_f32(1.0).reshape(&[1])?);
    }

    let n = length as f32;
    // Create indices [0, 1, 2, ..., length-1]
    let indices = ops::arange::<_, f32>(0, length as i32, None)?;
    // 2 * PI * i / (n - 1)
    let scale = Array::from_f32(2.0 * PI / (n - 1.0));
    let angles = ops::multiply(&indices, &scale)?;
    // 0.5 * (1 - cos(angles))
    let cos_vals = ops::cos(&angles)?;
    let one = Array::from_f32(1.0);
    let half = Array::from_f32(0.5);
    let one_minus_cos = ops::subtract(&one, &cos_vals)?;
    ops::multiply(&half, &one_minus_cos).map_err(|e| Error::model(e.to_string()))
}

/// Create Hamming window of given length as MLX array.
///
/// The Hamming window is defined as:
/// `w[n] = 0.54 - 0.46 * cos(2πn / (N-1))`
///
/// # Arguments
/// * `length` - Window length in samples
///
/// # Returns
/// MLX Array of shape `[length]` with window coefficients
pub fn hamming_window(length: usize) -> Result<Array> {
    if length == 0 {
        return Ok(Array::zeros::<f32>(&[0])?);
    }
    if length == 1 {
        return Ok(Array::from_f32(1.0).reshape(&[1])?);
    }

    let n = length as f32;
    // Create indices [0, 1, 2, ..., length-1]
    let indices = ops::arange::<_, f32>(0, length as i32, None)?;
    // 2 * PI * i / (n - 1)
    let scale = Array::from_f32(2.0 * PI / (n - 1.0));
    let angles = ops::multiply(&indices, &scale)?;
    // 0.54 - 0.46 * cos(angles)
    let cos_vals = ops::cos(&angles)?;
    let a = Array::from_f32(0.54);
    let b = Array::from_f32(0.46);
    let b_cos = ops::multiply(&b, &cos_vals)?;
    ops::subtract(&a, &b_cos).map_err(|e| Error::model(e.to_string()))
}

/// Create window by type name as MLX array.
///
/// # Arguments
/// * `window_type` - Window type: "hann" or "hamming"
/// * `length` - Window length in samples
///
/// # Returns
/// MLX Array with window coefficients, or error for unknown window type
pub fn create_window(window_type: &str, length: usize) -> Result<Array> {
    match window_type {
        "hann" => hann_window(length),
        "hamming" => hamming_window(length),
        _ => Err(Error::config(format!(
            "Unknown window type: {}",
            window_type
        ))),
    }
}

/// Create a window function with zero-padding to n_fft as MLX array.
///
/// The window is always center-padded: zeros are added symmetrically on both sides.
/// This ensures proper STFT alignment since audio frames are also center-padded.
///
/// # Arguments
/// * `window_type` - Window type: "hann" or "hamming"
/// * `win_length` - Window length in samples
/// * `n_fft` - FFT size (window will be zero-padded to this length)
///
/// # Returns
/// MLX Array of shape `[n_fft]` with center-padded window
pub fn create_window_padded(window_type: &str, win_length: usize, n_fft: usize) -> Result<Array> {
    let window = create_window(window_type, win_length)?;

    if win_length >= n_fft {
        return Ok(window);
    }

    // Always center-pad the window
    let left_pad = (n_fft - win_length) / 2;
    let right_pad = n_fft - win_length - left_pad;

    let left_zeros = Array::zeros::<f32>(&[left_pad as i32])?;
    let right_zeros = Array::zeros::<f32>(&[right_pad as i32])?;

    ops::concatenate(&[&left_zeros, &window, &right_zeros], 0)
        .map_err(|e| Error::model(e.to_string()))
}

// ============================================================================
// Mel Filterbank
// ============================================================================

// Slaney mel scale constants
const F_SP: f32 = 200.0 / 3.0;
const MIN_LOG_HZ: f32 = 1000.0;
const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
const LOGSTEP: f32 = 0.06875177742094912; // ln(6.4) / 27.0

/// Convert Hz to mel (Slaney scale).
///
/// Uses the Slaney mel scale which is linear below 1000 Hz
/// and logarithmic above.
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    } else {
        hz / F_SP
    }
}

/// Convert mel to Hz (Slaney scale).
///
/// Inverse of `hz_to_mel`.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        mel * F_SP
    }
}

/// Create Slaney-normalized mel filterbank as MLX array.
///
/// Returns a matrix of shape [n_mels, n_freqs] where n_freqs = n_fft / 2 + 1.
///
/// # Arguments
/// * `n_mels` - Number of mel filterbank channels
/// * `n_fft` - FFT size
/// * `sample_rate` - Audio sample rate in Hz
/// * `f_min` - Minimum frequency in Hz
/// * `f_max` - Maximum frequency in Hz
///
/// # Returns
/// MLX Array of shape [n_mels, n_freqs]
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Result<Array> {
    let n_freqs = n_fft / 2 + 1;

    // Build filterbank on CPU then convert to MLX
    // (Filterbank is computed once at init, so CPU is fine)
    let mut filterbank = vec![0.0f32; n_mels * n_freqs];

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

// ============================================================================
// Audio Processing Filters
// ============================================================================

/// Apply preemphasis filter to audio: `y[n] = x[n] - coef * x[n-1]`
///
/// # Arguments
/// * `audio` - Input audio array `[batch, time]` or `[time]`
/// * `coef` - Preemphasis coefficient (typically 0.97)
///
/// # Returns
/// Filtered audio with same shape as input
pub fn preemphasis(audio: &Array, coef: f32) -> Result<Array> {
    let ndim = audio.ndim();

    if ndim == 1 {
        // [time] case
        let audio_len = audio.shape()[0];
        if audio_len <= 1 {
            return Ok(audio.clone());
        }

        // Shift audio by 1: [0, x[0], x[1], ..., x[n-2]]
        let zero = Array::zeros::<f32>(&[1])?;
        let shifted = ops::concatenate(&[&zero, &audio.index(..(audio_len - 1))], 0)?;

        // y = x - coef * shifted
        let coef_arr = Array::from_f32(coef);
        let scaled = ops::multiply(&shifted, &coef_arr)?;
        ops::subtract(audio, &scaled).map_err(|e| Error::model(e.to_string()))
    } else if ndim == 2 {
        // [batch, time] case
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        if audio_len <= 1 {
            return Ok(audio.clone());
        }

        // Shift audio by 1: [zeros, x[:, :-1]]
        let zeros = Array::zeros::<f32>(&[batch_size, 1])?;
        let shifted = ops::concatenate(&[&zeros, &audio.index((.., ..(audio_len - 1)))], 1)?;

        // y = x - coef * shifted
        let coef_arr = Array::from_f32(coef);
        let scaled = ops::multiply(&shifted, &coef_arr)?;
        ops::subtract(audio, &scaled).map_err(|e| Error::model(e.to_string()))
    } else {
        Err(Error::config(format!(
            "preemphasis expects 1D or 2D input, got {}D",
            ndim
        )))
    }
}

/// Apply dithering (random noise) to audio.
///
/// # Arguments
/// * `audio` - Input audio array
/// * `amount` - Dither amount (noise amplitude)
///
/// # Returns
/// Audio with added dither noise
pub fn dither(audio: &Array, amount: f32) -> Result<Array> {
    if amount <= 0.0 {
        return Ok(audio.clone());
    }

    let noise = mlx_rs::random::uniform::<_, f32>(-amount, amount, audio.shape(), None)?;
    ops::add(audio, &noise).map_err(|e| Error::model(e.to_string()))
}

// ============================================================================
// STFT Helpers
// ============================================================================

/// Frame audio into overlapping windows as MLX array.
///
/// Returns array of shape [n_frames, n_fft] with center-padded frames.
/// Each frame has the audio centered within the FFT frame size.
///
/// Uses MLX `as_strided` for efficient overlapping window extraction,
/// avoiding GPU→CPU→GPU data transfer.
///
/// # Arguments
/// * `audio` - Input audio `[time]`
/// * `window_len` - Window length in samples
/// * `hop_len` - Hop length in samples
/// * `n_fft` - FFT size (frames will be zero-padded to this length)
///
/// # Returns
/// MLX Array of shape [n_frames, n_fft]
pub fn frame_audio(
    audio: &Array,
    window_len: usize,
    hop_len: usize,
    n_fft: usize,
) -> Result<Array> {
    let audio_len = audio.shape()[0] as usize;

    let n_frames = num_frames(audio_len, window_len, hop_len);
    if n_frames == 0 {
        return Ok(Array::zeros::<f32>(&[0, n_fft as i32])?);
    }

    // Use as_strided to create overlapping windows without copying
    // Shape: [n_frames, window_len], Strides: [hop_len, 1]
    let frames = ops::as_strided(
        audio,
        &[n_frames as i32, window_len as i32],
        &[hop_len as i64, 1],
        0,
    )?;

    // Center-pad each frame from window_len to n_fft
    if window_len < n_fft {
        let left_pad = ((n_fft - window_len) / 2) as i32;
        let right_pad = (n_fft - window_len) as i32 - left_pad;
        // Pad along axis 1 (the frame dimension)
        ops::pad(&frames, &[(0, 0), (left_pad, right_pad)], None, None)
            .map_err(|e| Error::model(e.to_string()))
    } else {
        Ok(frames)
    }
}

/// Frame batched audio into overlapping windows.
///
/// Returns array of shape [batch, n_frames, n_fft] with center-padded frames.
/// Each frame has the audio centered within the FFT frame size.
///
/// Uses MLX `as_strided` for efficient overlapping window extraction,
/// avoiding GPU→CPU→GPU data transfer.
///
/// # Arguments
/// * `audio` - Input audio [batch, time]
/// * `window_len` - Window length in samples
/// * `hop_len` - Hop length in samples
/// * `n_fft` - FFT size (frames will be zero-padded to this length)
///
/// # Returns
/// MLX Array of shape [batch, n_frames, n_fft]
pub fn frame_audio_batch(
    audio: &Array,
    window_len: usize,
    hop_len: usize,
    n_fft: usize,
) -> Result<Array> {
    let batch_size = audio.shape()[0] as usize;
    let audio_len = audio.shape()[1] as usize;

    let n_frames = num_frames(audio_len, window_len, hop_len);
    if n_frames == 0 {
        return Ok(Array::zeros::<f32>(&[batch_size as i32, 0, n_fft as i32])?);
    }

    // Use as_strided to create overlapping windows without copying
    // Input: [batch, time] with strides [time, 1]
    // Output: [batch, n_frames, window_len] with strides [time, hop_len, 1]
    let frames = ops::as_strided(
        audio,
        &[batch_size as i32, n_frames as i32, window_len as i32],
        &[audio_len as i64, hop_len as i64, 1],
        0,
    )?;

    // Center-pad each frame from window_len to n_fft
    if window_len < n_fft {
        let left_pad = ((n_fft - window_len) / 2) as i32;
        let right_pad = (n_fft - window_len) as i32 - left_pad;
        // Pad along axis 2 (the frame dimension)
        ops::pad(
            &frames,
            &[(0, 0), (0, 0), (left_pad, right_pad)],
            None,
            None,
        )
        .map_err(|e| Error::model(e.to_string()))
    } else {
        Ok(frames)
    }
}

/// Calculate number of frames from audio length.
///
/// # Arguments
/// * `audio_len` - Length of audio in samples
/// * `window_len` - Window length in samples
/// * `hop_len` - Hop length in samples
///
/// # Returns
/// Number of complete frames
#[inline]
pub fn num_frames(audio_len: usize, window_len: usize, hop_len: usize) -> usize {
    if audio_len >= window_len {
        (audio_len - window_len) / hop_len + 1
    } else {
        0
    }
}

/// Calculate the overlap size needed for STFT window continuity.
///
/// # Arguments
/// * `window_len` - Window length in samples
/// * `hop_len` - Hop length in samples
///
/// # Returns
/// Number of samples to keep for overlap
#[inline]
pub fn overlap_size(window_len: usize, hop_len: usize) -> usize {
    window_len.saturating_sub(hop_len)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ArrayExt;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ==================== Window Tests ====================

    #[test]
    fn test_hann_window_endpoints() {
        let window = hann_window(400).unwrap();
        window.eval().unwrap();
        let data: Vec<f32> = window.to_vec_row_major().unwrap();

        assert!(data[0].abs() < 1e-6, "First hann value should be ~0");
        assert!(data[399].abs() < 1e-6, "Last hann value should be ~0");
    }

    #[test]
    fn test_hann_window_center() {
        let window = hann_window(400).unwrap();
        window.eval().unwrap();
        let data: Vec<f32> = window.to_vec_row_major().unwrap();

        // Center value should be close to 1.0
        assert!(
            approx_eq(data[199], 1.0, 0.01),
            "Center hann value should be ~1.0"
        );
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let window = hamming_window(400).unwrap();
        window.eval().unwrap();
        let data: Vec<f32> = window.to_vec_row_major().unwrap();

        assert!(approx_eq(data[0], 0.08, 1e-4));
        assert!(approx_eq(data[399], 0.08, 1e-4));
    }

    #[test]
    fn test_create_window() {
        assert!(create_window("hann", 100).is_ok());
        assert!(create_window("hamming", 100).is_ok());
        assert!(create_window("unknown", 100).is_err());
    }

    #[test]
    fn test_create_window_padded() {
        let padded = create_window_padded("hann", 400, 512).unwrap();
        padded.eval().unwrap();
        let data: Vec<f32> = padded.to_vec_row_major().unwrap();

        assert_eq!(data.len(), 512);

        // Window should be center-padded
        let left_pad = (512 - 400) / 2; // 56
        let right_pad = 512 - 400 - left_pad; // 56

        // Check left padding is zeros
        for i in 0..left_pad {
            assert_eq!(data[i], 0.0, "Left padding at index {} should be 0", i);
        }

        // Check window values (Hann endpoints are ~0)
        assert!(data[left_pad].abs() < 1e-6, "First hann value should be ~0");
        assert!(
            data[left_pad + 399].abs() < 1e-6,
            "Last hann value should be ~0"
        );

        // Check right padding is zeros
        for i in (512 - right_pad)..512 {
            assert_eq!(data[i], 0.0, "Right padding at index {} should be 0", i);
        }
    }

    #[test]
    fn test_window_edge_cases() {
        let empty = hann_window(0).unwrap();
        empty.eval().unwrap();
        assert_eq!(empty.shape(), &[0]);

        let single = hann_window(1).unwrap();
        single.eval().unwrap();
        let data: Vec<f32> = single.to_vec_row_major().unwrap();
        assert_eq!(data, &[1.0]);
    }

    // ==================== Mel Scale Tests ====================

    #[test]
    fn test_hz_to_mel_roundtrip() {
        let test_freqs = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        for &hz in &test_freqs {
            let mel = hz_to_mel(hz);
            let hz_back = mel_to_hz(mel);
            // Allow small floating point error (0.01 Hz tolerance)
            assert!(
                approx_eq(hz, hz_back, 0.01),
                "Hz {} -> mel {} -> Hz {}",
                hz,
                mel,
                hz_back
            );
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        assert_eq!(fb.shape(), &[128, 257]); // n_mels, n_fft/2+1
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: Vec<f32> = fb.to_vec_row_major().unwrap();

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min >= 0.0, "Filterbank should be non-negative, got {}", min);
    }

    #[test]
    fn test_mel_filterbank_rows_sum() {
        // Each mel filter row should have some non-zero values
        let n_mels = 128;
        let n_freqs = 257;
        let fb = mel_filterbank(n_mels, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: Vec<f32> = fb.to_vec_row_major().unwrap();

        for m in 0..n_mels {
            let row_start = m * n_freqs;
            let row_sum: f32 = data[row_start..row_start + n_freqs].iter().sum();
            assert!(row_sum > 0.0, "Mel filter {} has zero sum", m);
        }
    }

    // ==================== Audio Processing Tests ====================

    #[test]
    fn test_preemphasis_1d() {
        let audio = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
        let result = preemphasis(&audio, 0.97).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.to_vec_row_major().unwrap();

        // y[0] = 1.0 - 0.97 * 0.0 = 1.0
        assert!(approx_eq(data[0], 1.0, 1e-6));
        // y[1] = 2.0 - 0.97 * 1.0 = 1.03
        assert!(approx_eq(data[1], 1.03, 1e-6));
        // y[2] = 3.0 - 0.97 * 2.0 = 1.06
        assert!(approx_eq(data[2], 1.06, 1e-6));
    }

    #[test]
    fn test_preemphasis_2d() {
        let audio = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = preemphasis(&audio, 0.97).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.to_vec_row_major().unwrap();

        // First batch: [1, 2, 3] -> [1, 1.03, 1.06]
        assert!(approx_eq(data[0], 1.0, 1e-6));
        assert!(approx_eq(data[1], 1.03, 1e-6));
        assert!(approx_eq(data[2], 1.06, 1e-6));

        // Second batch: [4, 5, 6] -> [4, 1.12, 1.15]
        assert!(approx_eq(data[3], 4.0, 1e-6));
        assert!(approx_eq(data[4], 5.0 - 0.97 * 4.0, 1e-6)); // 1.12
        assert!(approx_eq(data[5], 6.0 - 0.97 * 5.0, 1e-6)); // 1.15
    }

    #[test]
    fn test_dither() {
        let audio = Array::zeros::<f32>(&[100]).unwrap();
        let result = dither(&audio, 1e-5).unwrap();
        result.eval().unwrap();
        let data: Vec<f32> = result.to_vec_row_major().unwrap();

        // Result should have same length
        assert_eq!(data.len(), 100);

        // Values should be slightly different from input
        let has_diff = data.iter().any(|&v| v != 0.0);
        assert!(has_diff, "Dither should add noise");

        // Values should be within dither range
        for &v in &data {
            assert!(v.abs() <= 1e-5, "Dithered value {} exceeds range", v);
        }
    }

    #[test]
    fn test_dither_disabled() {
        let audio = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let result = dither(&audio, 0.0).unwrap();
        result.eval().unwrap();

        // Should be identical when amount is 0
        let original: Vec<f32> = audio.to_vec_row_major().unwrap();
        let processed: Vec<f32> = result.to_vec_row_major().unwrap();
        for (a, b) in original.iter().zip(processed.iter()) {
            assert_eq!(a, b);
        }
    }

    // ==================== Framing Tests ====================

    #[test]
    fn test_num_frames() {
        // Standard case: 1600 samples, window=400, hop=160
        // (1600 - 400) / 160 + 1 = 8 frames
        assert_eq!(num_frames(1600, 400, 160), 8);

        // Not enough for one frame
        assert_eq!(num_frames(399, 400, 160), 0);

        // Exactly one frame
        assert_eq!(num_frames(400, 400, 160), 1);
    }

    #[test]
    fn test_overlap_size() {
        assert_eq!(overlap_size(400, 160), 240);
        assert_eq!(overlap_size(512, 256), 256);
        assert_eq!(overlap_size(100, 100), 0);
        assert_eq!(overlap_size(100, 200), 0); // Handles case where hop > window
    }

    #[test]
    fn test_frame_audio() {
        let audio: Vec<f32> = (0..1600).map(|i| i as f32).collect();
        let audio_arr = Array::from_slice(&audio, &[1600]);

        let frames = frame_audio(&audio_arr, 400, 160, 512).unwrap();
        frames.eval().unwrap();

        let n_frames = num_frames(1600, 400, 160);
        assert_eq!(frames.shape(), &[n_frames as i32, 512]);

        // Frames should be center-padded
        let data: Vec<f32> = frames.to_vec_row_major().unwrap();
        let left_pad = (512 - 400) / 2; // 56

        // Left padding should be zeros
        for i in 0..left_pad {
            assert_eq!(data[i], 0.0, "Left padding at index {} should be 0", i);
        }

        // First audio sample (which is 0.0 from the sequence)
        assert_eq!(data[left_pad], 0.0);
        // Second audio sample should be 1.0
        assert_eq!(data[left_pad + 1], 1.0);

        // Right padding should be zeros (indices 56+400=456 to 511)
        let right_start = left_pad + 400;
        for i in right_start..512 {
            assert_eq!(data[i], 0.0, "Right padding at index {} should be 0", i);
        }
    }

    #[test]
    fn test_frame_audio_empty() {
        let audio = Array::from_slice(&[0.0f32; 100], &[100]);
        let frames = frame_audio(&audio, 400, 160, 512).unwrap();
        frames.eval().unwrap();
        assert_eq!(frames.shape(), &[0, 512]);
    }

    #[test]
    fn test_frame_audio_batch() {
        let audio: Vec<f32> = (0..3200).map(|i| i as f32).collect();
        let audio_arr = Array::from_slice(&audio, &[2, 1600]);

        let frames = frame_audio_batch(&audio_arr, 400, 160, 512).unwrap();
        frames.eval().unwrap();

        let n_frames = num_frames(1600, 400, 160);
        assert_eq!(frames.shape(), &[2, n_frames as i32, 512]);

        // Frames should be center-padded
        let data: Vec<f32> = frames.to_vec_row_major().unwrap();
        let left_pad = (512 - 400) / 2; // 56

        // Left padding should be zeros
        for i in 0..left_pad {
            assert_eq!(data[i], 0.0, "Left padding at index {} should be 0", i);
        }
        // First audio sample
        assert_eq!(data[left_pad], 0.0);
        assert_eq!(data[left_pad + 1], 1.0);
    }
}
