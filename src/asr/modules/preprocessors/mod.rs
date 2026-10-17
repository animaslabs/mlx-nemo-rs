//! Audio preprocessing modules.
//!
//! These modules convert raw audio waveforms to features suitable for
//! ASR model input (typically Mel spectrograms).

mod mel_spectrogram;

// Primary exports (new names)
pub use mel_spectrogram::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};

// Stateless streaming exports (legacy API - use StreamingMel for new code)
pub use mel_spectrogram::{
    MelStreamState, StreamingMelConfig, frames_available, process_audio_streaming,
};

// Backwards-compatible aliases
pub use mel_spectrogram::{MelSpectrogram, MelSpectrogramConfig};
