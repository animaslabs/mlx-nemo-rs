//! Speaker kernels for multitalker ASR.
//!
//! Port of NeMo's `SpeakerKernelMixin` (`spk_kernel_type: ff`), used by
//! `nvidia/multitalker-parakeet-streaming-0.6b-v1`. Before selected conformer
//! layers, the encoder input is conditioned on the target speaker's activity
//! (from a diarizer) and on the activity of everyone else:
//!
//! ```text
//! x = x + spk_kernel(x * spk_mask)
//! x = x + bg_spk_kernel(x * bg_mask)
//! ```
//!
//! Each kernel is `Linear(d_model, hidden) -> ReLU -> Linear(hidden, d_model)`,
//! stored as `spk_kernels.{layer}.0` / `spk_kernels.{layer}.3` (and likewise
//! under `bg_spk_kernels`).

use std::collections::{BTreeMap, HashMap};

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

use crate::core::error::Result;
use crate::core::quantization::{LinearLayer, QuantizationConfig, require_linear_layer};

/// A feed-forward speaker kernel.
struct FfKernel {
    up: LinearLayer,
    down: LinearLayer,
}

impl FfKernel {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<Self> {
        Ok(Self {
            up: require_linear_layer(weights, &format!("{}.0", prefix), quant)?,
            down: require_linear_layer(weights, &format!("{}.3", prefix), quant)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.up.forward(x)?;
        let h = ops::maximum(&h, Array::from_f32(0.0))?;
        self.down.forward(&h)
    }
}

/// Speaker and background-speaker kernels, keyed by the conformer layer they
/// are applied before.
pub struct SpeakerKernels {
    spk: BTreeMap<usize, FfKernel>,
    bg: BTreeMap<usize, FfKernel>,
}

impl SpeakerKernels {
    /// Load speaker kernels from a checkpoint.
    ///
    /// Returns `None` if the checkpoint has no `spk_kernels.*` weights (an
    /// ordinary ASR model), and an error if a kernel is incomplete.
    pub fn load(
        weights: &HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
    ) -> Result<Option<Self>> {
        let spk = load_kernels(weights, "spk_kernels", quant)?;
        if spk.is_empty() {
            return Ok(None);
        }
        let bg = load_kernels(weights, "bg_spk_kernels", quant)?;
        tracing::info!(
            layers = ?spk.keys().collect::<Vec<_>>(),
            background = !bg.is_empty(),
            "Loaded speaker kernels"
        );
        Ok(Some(Self { spk, bg }))
    }

    /// Whether a kernel is applied before `layer_idx`.
    pub fn applies_to(&self, layer_idx: usize) -> bool {
        self.spk.contains_key(&layer_idx) || self.bg.contains_key(&layer_idx)
    }

    /// Apply the kernels for `layer_idx` to the layer input.
    ///
    /// # Arguments
    /// * `x` - Layer input [batch, time, d_model]
    /// * `targets` - Speaker targets for this chunk
    pub fn apply(&self, layer_idx: usize, x: &Array, targets: &SpeakerTargets) -> Result<Array> {
        let time = x.dim(1);
        let mut x = x.clone();
        if let Some(kernel) = self.spk.get(&layer_idx) {
            let mask = align_mask(&targets.spk, time, 1.0)?;
            x = ops::add(&x, &kernel.forward(&ops::multiply(&x, &mask)?)?)?;
        }
        if let Some(kernel) = self.bg.get(&layer_idx) {
            let mask = align_mask(&targets.bg, time, 0.0)?;
            x = ops::add(&x, &kernel.forward(&ops::multiply(&x, &mask)?)?)?;
        }
        Ok(x)
    }
}

/// Per-frame speaker activity masks for one encoder call.
#[derive(Clone)]
pub struct SpeakerTargets {
    /// Target speaker activity [batch, time]
    pub spk: Array,
    /// Activity of any other speaker [batch, time]
    pub bg: Array,
}

fn load_kernels(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<BTreeMap<usize, FfKernel>> {
    let layer_prefix = format!("{}.", prefix);
    let mut layers: Vec<usize> = weights
        .keys()
        .filter_map(|k| {
            k.strip_prefix(&layer_prefix)?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    layers.sort_unstable();
    layers.dedup();

    layers
        .into_iter()
        .map(|layer| {
            let kernel = FfKernel::load(weights, &format!("{}.{}", prefix, layer), quant)?;
            Ok((layer, kernel))
        })
        .collect()
}

/// Match a [batch, mask_time] mask to `time` frames and add a feature axis.
///
/// Like NeMo's `solve_length_mismatch`: a short mask is left-padded with
/// `pad_value`, and a long mask keeps its last `time` frames.
fn align_mask(mask: &Array, time: i32, pad_value: f32) -> Result<Array> {
    let mask_time = mask.dim(1);
    let mask = if mask_time < time {
        let pad = ops::full::<f32>(&[mask.dim(0), time - mask_time], Array::from_f32(pad_value))?;
        ops::concatenate(&[&pad, mask], 1)?
    } else if mask_time > time {
        mask.index((.., mask_time - time..))
    } else {
        mask.clone()
    };
    Ok(ops::expand_dims(&mask, -1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ArrayExt;

    #[test]
    fn test_align_mask_pads_left_and_keeps_last() -> Result<()> {
        let mask = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let padded = align_mask(&mask, 4, 0.5)?;
        assert_eq!(padded.shape(), &[1, 4, 1]);
        assert_eq!(padded.to_vec_row_major::<f32>()?, vec![0.5, 0.5, 1.0, 0.0]);

        let long = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let trimmed = align_mask(&long, 2, 0.0)?;
        assert_eq!(trimmed.to_vec_row_major::<f32>()?, vec![2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_load_returns_none_without_kernels() {
        assert!(
            SpeakerKernels::load(&HashMap::new(), None)
                .unwrap()
                .is_none()
        );
    }
}
