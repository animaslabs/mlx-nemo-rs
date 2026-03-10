//! Audio preprocessing modules.
//!
//! These modules convert raw audio waveforms to features suitable for
//! ASR model input (typically Mel spectrograms).

mod mel_spectrogram;

pub use mel_spectrogram::MelSpectrogram;
