//! Audio preprocessing modules.
//!
//! These modules convert raw audio waveforms to features suitable for
//! ASR model input (typically Mel spectrograms).

mod mel_spectrogram;

pub use mel_spectrogram::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};
pub(crate) use mel_spectrogram::StreamingMelConfig;
