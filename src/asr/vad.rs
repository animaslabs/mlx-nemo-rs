//! Voice Activity Detection (VAD) post-processing.
//!
//! Converts per-frame speech probabilities from a classification model
//! into speech segments with onset/offset hysteresis, filtering, and padding.

/// Configuration for VAD post-processing.
#[derive(Debug, Clone, PartialEq)]
pub struct VadPostProcessConfig {
    /// Speech probability threshold for onset (start of speech).
    pub onset: f32,
    /// Speech probability threshold for offset (end of speech).
    pub offset: f32,
    /// Padding to add before speech onset (seconds).
    pub pad_onset: f32,
    /// Padding to add after speech offset (seconds).
    pub pad_offset: f32,
    /// Minimum speech segment duration (seconds). Shorter segments are discarded.
    pub min_duration_on: f32,
    /// Minimum silence duration (seconds). Shorter silences merge adjacent segments.
    pub min_duration_off: f32,
}

impl Default for VadPostProcessConfig {
    fn default() -> Self {
        Self {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
        }
    }
}

/// A detected speech segment with start and end timestamps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSegment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

impl SpeechSegment {
    /// Duration of the segment in seconds.
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

/// Post-processor that converts frame-level speech probabilities to speech segments.
#[derive(Debug)]
pub struct VadPostProcessor {
    config: VadPostProcessConfig,
}

impl VadPostProcessor {
    /// Create a new VadPostProcessor with the given configuration.
    pub fn new(config: VadPostProcessConfig) -> Self {
        Self { config }
    }

    /// Process frame probabilities into speech segments.
    ///
    /// # Arguments
    /// * `probs` - Per-frame speech probabilities (0.0 to 1.0)
    /// * `frame_duration` - Duration of each frame in seconds
    /// * `audio_duration` - Total audio duration in seconds (for clipping)
    ///
    /// # Returns
    /// * Vector of speech segments with start/end timestamps
    pub fn process(
        &self,
        probs: &[f32],
        frame_duration: f32,
        audio_duration: f32,
    ) -> Vec<SpeechSegment> {
        if probs.is_empty() {
            return Vec::new();
        }

        // Step 1: Binarize with onset/offset hysteresis
        let binary = self.binarize(probs);

        // Step 2: Extract raw segments
        let mut segments = self.extract_segments(&binary, frame_duration);

        // Step 3: Filter short speech segments
        if self.config.min_duration_on > 0.0 {
            segments.retain(|s| s.duration() >= self.config.min_duration_on);
        }

        // Step 4: Merge segments separated by short silence
        if self.config.min_duration_off > 0.0 {
            segments = self.merge_close_segments(segments);
        }

        // Step 5: Apply onset/offset padding
        if self.config.pad_onset > 0.0 || self.config.pad_offset > 0.0 {
            for seg in &mut segments {
                seg.start = (seg.start - self.config.pad_onset).max(0.0);
                seg.end = (seg.end + self.config.pad_offset).min(audio_duration);
            }
        }

        // Step 6: Clip to audio boundaries
        for seg in &mut segments {
            seg.start = seg.start.max(0.0);
            seg.end = seg.end.min(audio_duration);
        }

        // Remove any zero-duration segments that may have resulted
        segments.retain(|s| s.end > s.start);

        segments
    }

    /// Binarize probabilities using onset/offset hysteresis.
    ///
    /// State machine:
    /// - In silence: transition to speech when prob >= onset
    /// - In speech: transition to silence when prob < offset
    fn binarize(&self, probs: &[f32]) -> Vec<bool> {
        let mut binary = Vec::with_capacity(probs.len());
        let mut is_speech = false;

        for &p in probs {
            if is_speech {
                // Currently in speech: stay in speech unless below offset
                if p < self.config.offset {
                    is_speech = false;
                }
            } else {
                // Currently in silence: switch to speech if above onset
                if p >= self.config.onset {
                    is_speech = true;
                }
            }
            binary.push(is_speech);
        }

        binary
    }

    /// Extract contiguous speech segments from binary labels.
    fn extract_segments(&self, binary: &[bool], frame_duration: f32) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        let mut start: Option<usize> = None;

        for (i, &is_speech) in binary.iter().enumerate() {
            match (is_speech, start) {
                (true, None) => {
                    // Speech onset
                    start = Some(i);
                }
                (false, Some(s)) => {
                    // Speech offset
                    segments.push(SpeechSegment {
                        start: s as f32 * frame_duration,
                        end: i as f32 * frame_duration,
                    });
                    start = None;
                }
                _ => {}
            }
        }

        // Handle segment that extends to end
        if let Some(s) = start {
            segments.push(SpeechSegment {
                start: s as f32 * frame_duration,
                end: binary.len() as f32 * frame_duration,
            });
        }

        segments
    }

    /// Merge segments that are separated by silence shorter than min_duration_off.
    fn merge_close_segments(&self, segments: Vec<SpeechSegment>) -> Vec<SpeechSegment> {
        if segments.len() <= 1 {
            return segments;
        }

        let mut merged = Vec::with_capacity(segments.len());
        let mut current = segments[0].clone();

        for next in segments.iter().skip(1) {
            let gap = next.start - current.end;
            if gap < self.config.min_duration_off {
                // Merge: extend current segment to cover the next one
                current.end = next.end;
            } else {
                merged.push(current);
                current = next.clone();
            }
        }
        merged.push(current);

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binarize_simple() {
        let config = VadPostProcessConfig {
            onset: 0.5,
            offset: 0.3,
            ..Default::default()
        };
        let processor = VadPostProcessor::new(config);

        let probs = vec![0.1, 0.2, 0.6, 0.8, 0.4, 0.2, 0.1];
        let binary = processor.binarize(&probs);
        // Onset at 0.6 (idx 2), stays speech at 0.4 (> offset 0.3),
        // offset at 0.2 (idx 5)
        assert_eq!(binary, vec![false, false, true, true, true, false, false]);
    }

    #[test]
    fn test_extract_segments() {
        let config = VadPostProcessConfig::default();
        let processor = VadPostProcessor::new(config);

        let binary = vec![false, false, true, true, true, false, false, true, false];
        let segments = processor.extract_segments(&binary, 0.02);

        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 0.04).abs() < 1e-6);
        assert!((segments[0].end - 0.10).abs() < 1e-6);
        assert!((segments[1].start - 0.14).abs() < 1e-6);
        assert!((segments[1].end - 0.16).abs() < 1e-6);
    }

    #[test]
    fn test_merge_close_segments() {
        let config = VadPostProcessConfig {
            min_duration_off: 0.1,
            ..Default::default()
        };
        let processor = VadPostProcessor::new(config);

        let segments = vec![
            SpeechSegment {
                start: 0.0,
                end: 0.5,
            },
            SpeechSegment {
                start: 0.55,
                end: 1.0,
            }, // gap = 0.05 < 0.1, merge
            SpeechSegment {
                start: 1.5,
                end: 2.0,
            }, // gap = 0.5 > 0.1, keep separate
        ];
        let merged = processor.merge_close_segments(segments);

        assert_eq!(merged.len(), 2);
        assert!((merged[0].start - 0.0).abs() < 1e-6);
        assert!((merged[0].end - 1.0).abs() < 1e-6);
        assert!((merged[1].start - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_process_full() {
        let config = VadPostProcessConfig {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.02,
            pad_offset: 0.02,
            min_duration_on: 0.04,
            min_duration_off: 0.0,
        };
        let processor = VadPostProcessor::new(config);

        // 10 frames at 20ms each = 200ms total
        let probs = vec![0.1, 0.2, 0.7, 0.8, 0.9, 0.6, 0.3, 0.1, 0.0, 0.0];
        let segments = processor.process(&probs, 0.02, 0.20);

        assert_eq!(segments.len(), 1);
        // Speech: frames 2-5, so 0.04s to 0.12s without padding
        // With padding: 0.02s to 0.14s
        assert!((segments[0].start - 0.02).abs() < 1e-6);
        assert!((segments[0].end - 0.14).abs() < 1e-6);
    }

    #[test]
    fn test_empty_probs() {
        let processor = VadPostProcessor::new(VadPostProcessConfig::default());
        let segments = processor.process(&[], 0.02, 1.0);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_all_speech() {
        let config = VadPostProcessConfig {
            onset: 0.5,
            offset: 0.5,
            ..Default::default()
        };
        let processor = VadPostProcessor::new(config);

        let probs = vec![0.9, 0.8, 0.7, 0.6, 0.8];
        let segments = processor.process(&probs, 0.02, 0.10);

        assert_eq!(segments.len(), 1);
        assert!((segments[0].start - 0.0).abs() < 1e-6);
        assert!((segments[0].end - 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_filter_short_segments() {
        let config = VadPostProcessConfig {
            onset: 0.5,
            offset: 0.5,
            min_duration_on: 0.05,
            ..Default::default()
        };
        let processor = VadPostProcessor::new(config);

        // Single frame of speech at 20ms < min_duration_on 50ms
        let probs = vec![0.1, 0.8, 0.1, 0.1];
        let segments = processor.process(&probs, 0.02, 0.08);

        assert!(segments.is_empty());
    }
}
