//! Greedy TDT (token-and-duration transducer) decoding.
//!
//! A TDT joint scores a duration besides each token: after the token
//! logits (blank last) come one logit per entry of `durations`, the number
//! of encoder frames to advance. Mirrors NeMo's label-looping
//! `GreedyBatchedTDTInfer` (the `greedy_batch` default):
//! - token = argmax of the token logits, skip = `durations[argmax]` of the
//!   duration logits
//! - a token is emitted at the current frame and advances the prediction
//!   network; a blank leaves it unchanged and advances at least one frame
//! - the frame advances by `skip`, and by one after `max_symbols` tokens on
//!   one frame

use mlx_rs::Array;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::stack;
use tracing::debug;

use super::TimedToken;
use crate::asr::modules::{DecoderState, PredictionNetwork, RnntJoint};
use crate::core::ArrayExt;
use crate::core::error::{Error, Result};

/// Encoder frames scored per GPU round trip while the prediction is unchanged.
const DECODE_WINDOW: usize = 32;

/// Maximum number of symbols emitted per encoder frame (NeMo's default).
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

/// Greedy TDT decoder.
#[derive(Debug, Clone)]
pub struct TdtGreedyDecoder {
    /// Blank token ID (the last token logit)
    blank_id: usize,
    /// Frames advanced for each duration logit
    durations: Vec<usize>,
    /// Maximum symbols per encoder frame
    max_symbols_per_step: usize,
}

/// Where greedy decoding goes within frames scored with one prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Walk {
    /// A token at this offset, with the frames it advances
    Token { offset: usize, id: usize, skip: usize },
    /// Only blanks; the next frame to score is at this offset (possibly past
    /// the scored frames)
    Blanks(usize),
}

/// Follow blanks through per-frame `(label, skip)` choices from offset 0
/// until the first token.
fn walk(labels: &[usize], skips: &[usize], blank: usize) -> Walk {
    let mut i = 0;
    while i < labels.len() {
        if labels[i] != blank {
            return Walk::Token {
                offset: i,
                id: labels[i],
                skip: skips[i],
            };
        }
        i += skips[i].max(1);
    }
    Walk::Blanks(i)
}

impl TdtGreedyDecoder {
    /// Create a decoder for a joint with `durations.len()` duration logits.
    pub fn new(blank_id: usize, durations: Vec<usize>) -> Result<Self> {
        if durations.is_empty() {
            return Err(Error::config("a TDT decoder needs at least one duration"));
        }
        Ok(Self {
            blank_id,
            durations,
            max_symbols_per_step: DEFAULT_MAX_SYMBOLS_PER_STEP,
        })
    }

    /// Set the maximum symbols per encoder frame.
    pub fn with_max_symbols(mut self, max_symbols: usize) -> Self {
        self.max_symbols_per_step = max_symbols.max(1);
        self
    }

    /// Greedy-decode one sequence of encoder frames already projected by
    /// [`RnntJoint::project_encoder`] ([time, joint_hidden]), from a fresh
    /// decoder state.
    ///
    /// While the prediction is unchanged (across blanks) the joint scores up
    /// to `DECODE_WINDOW` frames at once and the skips are followed on the
    /// host, so there is one GPU round trip per emitted token or window of
    /// blanks. Each token spans the frames its duration advances (possibly
    /// none, like NeMo's `token_duration`).
    pub fn decode_projected(
        &self,
        enc_proj: &Array,
        valid_frames: usize,
        decoder: &PredictionNetwork,
        joint: &RnntJoint,
    ) -> Result<Vec<TimedToken>> {
        let vocab = joint.vocab_size() as i32;
        if joint.num_extra_outputs() != self.durations.len() {
            return Err(Error::config(format!(
                "the joint has {} duration logits but there are {} TDT durations",
                joint.num_extra_outputs(),
                self.durations.len()
            )));
        }

        let predict = |token: Option<usize>, state: &DecoderState| -> Result<(Array, DecoderState)> {
            let token = token.map(|t| Array::from_slice(&[t as i32], &[1, 1]));
            let (pred, next_state) = decoder.predict_one(token.as_ref(), state)?;
            Ok((
                joint.project_prediction(&pred.reshape(&[1, -1])?)?,
                next_state,
            ))
        };
        // Prediction for the current history, and the state after consuming
        // the last token (committed only once a new token is emitted)
        let (mut pred_proj, mut next_state) = predict(None, &decoder.initialize_state(1)?)?;

        let mut tokens = Vec::new();
        let mut t = 0;
        let mut symbols = 0;
        while t < valid_frames {
            let window = DECODE_WINDOW.min(valid_frames - t);
            let enc = enc_proj.index((t as i32..(t + window) as i32, ..));
            let logits = joint.logits_from_projections(&enc, &pred_proj)?;
            let labels = argmax_axis(logits.index((.., ..vocab)), -1, false)?;
            let duration_ids = argmax_axis(logits.index((.., vocab..)), -1, false)?;
            let choices: Vec<u32> = stack(&[labels, duration_ids], 0)?
                .as_dtype(mlx_rs::Dtype::Uint32)?
                .to_vec_row_major()?;
            let (labels, duration_ids) = choices.split_at(window);
            let labels: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
            let skips: Vec<usize> = duration_ids
                .iter()
                .map(|&d| self.durations[d as usize])
                .collect();

            let (offset, id, skip) = match walk(&labels, &skips, self.blank_id) {
                Walk::Blanks(offset) => {
                    t += offset;
                    symbols = 0;
                    continue;
                }
                Walk::Token { offset, id, skip } => (offset, id, skip),
            };
            if offset > 0 {
                t += offset;
                symbols = 0;
            }

            tokens.push(TimedToken {
                id,
                start_frame: t,
                end_frame: t + skip,
            });
            let (proj, state) = predict(Some(id), &next_state)?;
            next_state = state;
            pred_proj = proj;

            symbols += 1;
            if skip > 0 {
                t += skip;
                symbols = 0;
            } else if symbols >= self.max_symbols_per_step {
                t += 1;
                symbols = 0;
            }
        }

        debug!(
            frames = valid_frames,
            tokens = tokens.len(),
            "TDT greedy decode complete"
        );
        Ok(tokens)
    }
}

/// Move each punctuation token after the first to the end of the token
/// before it, as NeMo does for TDT timestamps (`_refine_timestamps_tdt`):
/// punctuation is often predicted long after the word it follows.
pub fn refine_punctuation_timestamps(
    tokens: &mut [TimedToken],
    is_punctuation: impl Fn(usize) -> bool,
) {
    for i in 1..tokens.len() {
        if is_punctuation(tokens[i].id) {
            let end = tokens[i - 1].end_frame;
            tokens[i].start_frame = end;
            tokens[i].end_frame = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLANK: usize = 9;

    #[test]
    fn test_walk_stops_at_first_token() {
        assert_eq!(
            walk(&[BLANK, 4, BLANK], &[1, 2, 1], BLANK),
            Walk::Token {
                offset: 1,
                id: 4,
                skip: 2
            }
        );
        // A token with duration 0 is still emitted
        assert_eq!(
            walk(&[3], &[0], BLANK),
            Walk::Token {
                offset: 0,
                id: 3,
                skip: 0
            }
        );
    }

    #[test]
    fn test_walk_follows_blank_skips() {
        // Blanks skip frames 1-2 and land on the token at frame 3
        assert_eq!(
            walk(&[BLANK, 5, 5, 7], &[3, 1, 1, 1], BLANK),
            Walk::Token {
                offset: 3,
                id: 7,
                skip: 1
            }
        );
        // A blank with duration 0 advances one frame
        assert_eq!(walk(&[BLANK, BLANK], &[0, 0], BLANK), Walk::Blanks(2));
        // A skip can land past the scored frames
        assert_eq!(walk(&[BLANK, BLANK], &[1, 4], BLANK), Walk::Blanks(5));
        assert_eq!(walk(&[], &[], BLANK), Walk::Blanks(0));
    }

    /// Decode `frames` frames with an all-zero joint, whose argmax is always
    /// token 0 with the first duration.
    fn decode_zero_joint(durations: Vec<usize>, frames: usize) -> Vec<TimedToken> {
        use crate::asr::config::{DecoderConfig, JointConfig, PredNetConfig};
        let config = DecoderConfig {
            prednet: Some(PredNetConfig {
                pred_hidden: 4,
                pred_rnn_layers: None,
                t_max: None,
                dropout: None,
            }),
            context_size: Some(2),
            ..Default::default()
        };
        let decoder = PredictionNetwork::Stateless(
            crate::asr::modules::StatelessTransducerDecoder::new(&config, BLANK).unwrap(),
        );
        let joint_config = JointConfig {
            joint_hidden: 4,
            num_extra_outputs: Some(durations.len()),
            ..Default::default()
        };
        let joint = RnntJoint::new(&joint_config, 4, 4, BLANK + 1).unwrap();
        let enc = Array::zeros::<f32>(&[frames as i32, 4]).unwrap();
        TdtGreedyDecoder::new(BLANK, durations)
            .unwrap()
            .with_max_symbols(3)
            .decode_projected(&joint.project_encoder(&enc).unwrap(), frames, &decoder, &joint)
            .unwrap()
    }

    #[test]
    fn test_decode_max_symbols_per_frame() {
        // Duration 0: max_symbols tokens on each frame, then the next frame
        let tokens = decode_zero_joint(vec![0, 1], 2);
        let frames: Vec<usize> = tokens.iter().map(|t| t.start_frame).collect();
        assert_eq!(frames, vec![0, 0, 0, 1, 1, 1]);
        assert!(tokens.iter().all(|t| t.id == 0 && t.end_frame == t.start_frame));
    }

    #[test]
    fn test_decode_skips_frames() {
        // Duration 3: one token every three frames, spanning them, across
        // window boundaries
        let tokens = decode_zero_joint(vec![3, 1], 70);
        let starts: Vec<usize> = tokens.iter().map(|t| t.start_frame).collect();
        assert_eq!(starts, (0..70).step_by(3).collect::<Vec<_>>());
        assert!(tokens.iter().all(|t| t.end_frame == t.start_frame + 3));
    }

    #[test]
    fn test_refine_punctuation_timestamps() {
        let timed = |id, start_frame, end_frame| TimedToken {
            id,
            start_frame,
            end_frame,
        };
        // Punctuation (id 1) follows the end of the token before it, even if
        // that token is punctuation; a leading one keeps its frames
        let mut tokens = vec![
            timed(1, 0, 2),
            timed(5, 2, 4),
            timed(1, 6, 8),
            timed(1, 9, 9),
            timed(6, 10, 11),
        ];
        refine_punctuation_timestamps(&mut tokens, |id| id == 1);
        assert_eq!(
            tokens,
            vec![
                timed(1, 0, 2),
                timed(5, 2, 4),
                timed(1, 4, 4),
                timed(1, 4, 4),
                timed(6, 10, 11),
            ]
        );
    }

    #[test]
    fn test_decoder_needs_durations() {
        assert!(TdtGreedyDecoder::new(BLANK, vec![]).is_err());
        let decoder = TdtGreedyDecoder::new(BLANK, vec![0, 1, 2])
            .unwrap()
            .with_max_symbols(3);
        assert_eq!(decoder.max_symbols_per_step, 3);
    }
}
