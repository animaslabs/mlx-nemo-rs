//! Common DSP primitives for audio processing.
//!
//! Pure functions for signal processing - no MLX dependencies.
//! These functions are used by the mel spectrogram processor and can be
//! reused for other audio processing tasks.

use std::f32::consts::PI;

// ============================================================================
// Window Functions
// ============================================================================

/// Create Hann window of given length.
///
/// The Hann window is defined as:
/// w[n] = 0.5 * (1 - cos(2πn / (N-1)))
///
/// # Arguments
/// * `length` - Window length in samples
///
/// # Returns
/// Vector of window coefficients
pub fn hann_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let n = length as f32;
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1.0)).cos()))
        .collect()
}

/// Create Hamming window of given length.
///
/// The Hamming window is defined as:
/// w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
///
/// # Arguments
/// * `length` - Window length in samples
///
/// # Returns
/// Vector of window coefficients
pub fn hamming_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let n = length as f32;
    (0..length)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1.0)).cos())
        .collect()
}

/// Create window by type name.
///
/// # Arguments
/// * `window_type` - Window type: "hann" or "hamming"
/// * `length` - Window length in samples
///
/// # Returns
/// `Some(Vec<f32>)` with window coefficients, or `None` for unknown window type
pub fn create_window(window_type: &str, length: usize) -> Option<Vec<f32>> {
    match window_type {
        "hann" => Some(hann_window(length)),
        "hamming" => Some(hamming_window(length)),
        _ => None,
    }
}

/// Create a window function with centered zero-padding to n_fft.
///
/// # Arguments
/// * `window_type` - Window type: "hann" or "hamming"
/// * `win_length` - Window length in samples
/// * `n_fft` - FFT size (window will be zero-padded to this length)
///
/// # Returns
/// `Some(Vec<f32>)` with zero-padded window, or `None` for unknown window type
pub fn create_window_padded(window_type: &str, win_length: usize, n_fft: usize) -> Option<Vec<f32>> {
    let win_values = create_window(window_type, win_length)?;

    if win_length >= n_fft {
        return Some(win_values);
    }

    // Center the window with zero-padding
    let left_pad = (n_fft - win_length) / 2;
    let mut padded = vec![0.0f32; n_fft];
    padded[left_pad..left_pad + win_length].copy_from_slice(&win_values);

    Some(padded)
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

/// Create Slaney-normalized mel filterbank.
///
/// Returns a flattened [n_mels × n_freqs] matrix in row-major order,
/// where n_freqs = n_fft / 2 + 1.
///
/// # Arguments
/// * `n_mels` - Number of mel filterbank channels
/// * `n_fft` - FFT size
/// * `sample_rate` - Audio sample rate in Hz
/// * `f_min` - Minimum frequency in Hz
/// * `f_max` - Maximum frequency in Hz
///
/// # Returns
/// Flattened filterbank matrix of size [n_mels * n_freqs]
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

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

    filterbank
}

// ============================================================================
// Audio Processing Filters
// ============================================================================

/// Apply preemphasis filter: y[n] = x[n] - coef * x[n-1]
///
/// # Arguments
/// * `audio` - Input audio samples
/// * `coef` - Preemphasis coefficient (typically 0.97)
/// * `last_sample` - Last sample from previous chunk (for continuity)
///
/// # Returns
/// Filtered audio samples
pub fn preemphasis(audio: &[f32], coef: f32, last_sample: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(audio.len());
    let mut prev = last_sample;
    for &sample in audio {
        out.push(sample - coef * prev);
        prev = sample;
    }
    out
}

/// Apply dithering (deterministic noise for reproducibility).
///
/// Uses a deterministic hash-based noise generator for reproducibility
/// in tests. In production, you might want true random noise.
///
/// # Arguments
/// * `audio` - Input audio samples
/// * `amount` - Dither amount (noise amplitude)
///
/// # Returns
/// Audio samples with added dither noise
pub fn dither(audio: &[f32], amount: f32) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    audio
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let noise = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * amount;
            sample + noise
        })
        .collect()
}

// ============================================================================
// STFT Helpers
// ============================================================================

/// Frame audio into overlapping windows, zero-padded to n_fft.
///
/// Returns flattened [n_frames × n_fft] data in row-major order.
///
/// # Arguments
/// * `audio` - Input audio samples
/// * `window_len` - Window length in samples
/// * `hop_len` - Hop length in samples
/// * `n_fft` - FFT size (frames will be zero-padded to this length)
///
/// # Returns
/// Flattened framed data of size [n_frames * n_fft]
pub fn frame_audio(audio: &[f32], window_len: usize, hop_len: usize, n_fft: usize) -> Vec<f32> {
    let n_frames = num_frames(audio.len(), window_len, hop_len);
    if n_frames == 0 {
        return Vec::new();
    }

    let left_pad = (n_fft - window_len) / 2;
    let mut frames = Vec::with_capacity(n_frames * n_fft);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_len;

        // Left zero-padding
        frames.extend(std::iter::repeat(0.0f32).take(left_pad));

        // Audio samples
        frames.extend_from_slice(&audio[start..start + window_len]);

        // Right zero-padding
        frames.extend(std::iter::repeat(0.0f32).take(n_fft - window_len - left_pad));
    }

    frames
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

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ==================== Window Tests ====================

    #[test]
    fn test_hann_window_endpoints() {
        let window = hann_window(400);
        assert!(window[0].abs() < 1e-6, "First hann value should be ~0");
        assert!(window[399].abs() < 1e-6, "Last hann value should be ~0");
    }

    #[test]
    fn test_hann_window_center() {
        let window = hann_window(400);
        // Center value should be close to 1.0
        assert!(approx_eq(window[199], 1.0, 0.01), "Center hann value should be ~1.0");
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let window = hamming_window(400);
        assert!(approx_eq(window[0], 0.08, 1e-4));
        assert!(approx_eq(window[399], 0.08, 1e-4));
    }

    #[test]
    fn test_create_window() {
        assert!(create_window("hann", 100).is_some());
        assert!(create_window("hamming", 100).is_some());
        assert!(create_window("unknown", 100).is_none());
    }

    #[test]
    fn test_create_window_padded() {
        let padded = create_window_padded("hann", 400, 512).unwrap();
        assert_eq!(padded.len(), 512);

        let left_pad = (512 - 400) / 2;
        // Check padding is zeros
        for i in 0..left_pad {
            assert_eq!(padded[i], 0.0);
        }
        // Check window values
        assert!(padded[left_pad].abs() < 1e-6);
    }

    #[test]
    fn test_window_edge_cases() {
        assert!(hann_window(0).is_empty());
        assert_eq!(hann_window(1), vec![1.0]);
        assert!(hamming_window(0).is_empty());
        assert_eq!(hamming_window(1), vec![1.0]);
    }

    // ==================== Mel Scale Tests ====================

    #[test]
    fn test_hz_to_mel_roundtrip() {
        let test_freqs = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        for &hz in &test_freqs {
            let mel = hz_to_mel(hz);
            let hz_back = mel_to_hz(mel);
            // Allow small floating point error (0.01 Hz tolerance)
            assert!(approx_eq(hz, hz_back, 0.01), "Hz {} -> mel {} -> Hz {}", hz, mel, hz_back);
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(128, 512, 16000.0, 0.0, 8000.0);
        assert_eq!(fb.len(), 128 * 257); // n_mels * (n_fft/2 + 1)
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank(128, 512, 16000.0, 0.0, 8000.0);
        let min = fb.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min >= 0.0, "Filterbank should be non-negative, got {}", min);
    }

    #[test]
    fn test_mel_filterbank_rows_sum() {
        // Each mel filter row should have some non-zero values
        let n_mels = 128;
        let n_freqs = 257;
        let fb = mel_filterbank(n_mels, 512, 16000.0, 0.0, 8000.0);

        for m in 0..n_mels {
            let row_start = m * n_freqs;
            let row_sum: f32 = fb[row_start..row_start + n_freqs].iter().sum();
            assert!(row_sum > 0.0, "Mel filter {} has zero sum", m);
        }
    }

    // ==================== Audio Processing Tests ====================

    #[test]
    fn test_preemphasis() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = preemphasis(&audio, 0.97, 0.0);

        // y[0] = 1.0 - 0.97 * 0.0 = 1.0
        assert!(approx_eq(result[0], 1.0, 1e-6));
        // y[1] = 2.0 - 0.97 * 1.0 = 1.03
        assert!(approx_eq(result[1], 1.03, 1e-6));
        // y[2] = 3.0 - 0.97 * 2.0 = 1.06
        assert!(approx_eq(result[2], 1.06, 1e-6));
    }

    #[test]
    fn test_preemphasis_continuity() {
        let audio1 = vec![1.0, 2.0, 3.0];
        let audio2 = vec![4.0, 5.0, 6.0];

        let result1 = preemphasis(&audio1, 0.97, 0.0);
        let last_sample = audio1[audio1.len() - 1];
        let result2 = preemphasis(&audio2, 0.97, last_sample);

        // Result should be same as processing all at once
        let full_audio = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let full_result = preemphasis(&full_audio, 0.97, 0.0);

        for i in 0..3 {
            assert!(approx_eq(result1[i], full_result[i], 1e-6));
            assert!(approx_eq(result2[i], full_result[3 + i], 1e-6));
        }
    }

    #[test]
    fn test_dither() {
        let audio = vec![0.0; 100];
        let result = dither(&audio, 1e-5);

        // Result should have same length
        assert_eq!(result.len(), 100);

        // Values should be slightly different from input
        let has_diff = result.iter().any(|&v| v != 0.0);
        assert!(has_diff, "Dither should add noise");

        // Values should be within dither range
        for &v in &result {
            assert!(v.abs() <= 1e-5, "Dithered value {} exceeds range", v);
        }
    }

    #[test]
    fn test_dither_deterministic() {
        let audio = vec![0.0; 100];
        let result1 = dither(&audio, 1e-5);
        let result2 = dither(&audio, 1e-5);

        // Same input should give same output (deterministic)
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert_eq!(a, b, "Dither should be deterministic");
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
        let frames = frame_audio(&audio, 400, 160, 512);

        let n_frames = num_frames(1600, 400, 160);
        assert_eq!(frames.len(), n_frames * 512);

        // Check first frame
        let left_pad = (512 - 400) / 2;
        // Padding should be zeros
        for i in 0..left_pad {
            assert_eq!(frames[i], 0.0);
        }
        // First audio sample
        assert_eq!(frames[left_pad], 0.0);
    }

    #[test]
    fn test_frame_audio_empty() {
        let audio = vec![0.0f32; 100];
        let frames = frame_audio(&audio, 400, 160, 512);
        assert!(frames.is_empty());
    }
}
