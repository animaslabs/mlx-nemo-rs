//! Audio preprocessing modules.
//!
//! These modules convert raw audio waveforms to features suitable for
//! ASR model input (typically Mel spectrograms).

mod mel_spectrogram;

// Primary exports (new names)
pub use mel_spectrogram::{MelConfig, MelProcessor};

// Streaming exports
pub use mel_spectrogram::{StreamingMelConfig, StreamingMelProcessor};

// Backwards-compatible aliases
pub use mel_spectrogram::{MelSpectrogram, MelSpectrogramConfig};
