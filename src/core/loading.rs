//! Loading checkpoint tensors into model layers.
//!
//! Loads are strict by default: a tensor the model has but the checkpoint
//! lacks is an error. With [`LoadOptions::strict`] set to `false`, such
//! tensors keep their initial values and are listed in a [`LoadReport`]
//! instead, which is useful for partial or experimental checkpoints.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::nn;

use crate::core::error::{Error, Result};
use crate::core::quantization::{
    EmbeddingLayer, LinearLayer, QuantizationConfig, load_embedding_layer, load_linear_layer,
};

/// How to load checkpoint weights into a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadOptions {
    /// Fail if the checkpoint lacks a tensor the model has (the default).
    ///
    /// When `false`, missing tensors keep their initial values (usually zeros,
    /// so the model's output is not meaningful) and are listed in the
    /// [`LoadReport`], and `from_pretrained_with` also accepts a checkpoint
    /// without a tokenizer.
    pub strict: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { strict: true }
    }
}

impl LoadOptions {
    /// Set whether missing tensors are an error.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Handle a checkpoint at `path` without a tokenizer: an error in a
    /// strict load, otherwise a warning.
    pub(crate) fn missing_tokenizer(&self, path: &Path) -> Result<()> {
        if self.strict {
            return Err(Error::missing_tokenizer(path));
        }
        tracing::warn!(
            path = %path.display(),
            "no SentencePiece tokenizer found; transcription will fail until one is loaded"
        );
        Ok(())
    }
}

/// What a weight load left out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadReport {
    /// Tensors the model has that the checkpoint lacked, which kept their
    /// initial values. Always empty after a strict load.
    pub missing: Vec<String>,
}

impl LoadReport {
    /// Whether every tensor the model has was loaded.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
}

/// A checkpoint's tensors, together with how strictly to load them.
///
/// Dereferences to the underlying map for optional lookups.
#[derive(Debug)]
pub struct Weights<'a> {
    map: &'a HashMap<String, Array>,
    strict: bool,
    missing: RefCell<Vec<String>>,
}

impl<'a> Weights<'a> {
    /// Wrap a checkpoint's tensors for loading with `options`.
    pub fn new(map: &'a HashMap<String, Array>, options: LoadOptions) -> Self {
        Self {
            map,
            strict: options.strict,
            missing: RefCell::new(Vec::new()),
        }
    }

    /// The tensors that were missing so far.
    pub fn report(&self) -> LoadReport {
        LoadReport {
            missing: self.missing.borrow().clone(),
        }
    }

    /// Handle a tensor the model needs but the checkpoint lacks, named by
    /// `keys` (the alternatives the loader tried, canonical first): an error
    /// in a strict load, otherwise recorded.
    pub(crate) fn report_missing(&self, keys: &[String]) -> Result<()> {
        if self.strict {
            return Err(Error::missing_weight(&keys.join("` or `")));
        }
        if let Some(key) = keys.first() {
            tracing::warn!(key, "weight missing from checkpoint, keeping initial value");
            self.missing.borrow_mut().push(key.clone());
        }
        Ok(())
    }

    /// The tensor at `key`, or `None` after reporting it missing.
    pub(crate) fn fetch(&self, key: &str) -> Result<Option<&'a Array>> {
        self.fetch_any(&[key.to_string()])
    }

    /// The tensor at the first of `keys` the checkpoint has, or `None` after
    /// reporting it missing.
    pub(crate) fn fetch_any(&self, keys: &[String]) -> Result<Option<&'a Array>> {
        let found = keys.iter().find_map(|key| self.map.get(key));
        if found.is_none() {
            self.report_missing(keys)?;
        }
        Ok(found)
    }

    /// The linear layer at the first of `prefixes` the checkpoint has, or
    /// `None` after reporting it missing.
    pub(crate) fn fetch_linear_any(
        &self,
        prefixes: &[String],
        quant: Option<&QuantizationConfig>,
    ) -> Result<Option<LinearLayer>> {
        let found = prefixes
            .iter()
            .find_map(|p| load_linear_layer(self.map, p, quant));
        if found.is_none() {
            let keys: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
            self.report_missing(&keys)?;
        }
        Ok(found)
    }

    /// The linear layer at `prefix`, or `None` after reporting it missing.
    pub(crate) fn fetch_linear(
        &self,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<Option<LinearLayer>> {
        self.fetch_linear_any(&[prefix.to_string()], quant)
    }

    /// The embedding layer at the first of `prefixes` the checkpoint has, or
    /// `None` after reporting it missing.
    pub(crate) fn fetch_embedding_any(
        &self,
        prefixes: &[String],
        quant: Option<&QuantizationConfig>,
    ) -> Result<Option<EmbeddingLayer>> {
        let found = prefixes
            .iter()
            .find_map(|p| load_embedding_layer(self.map, p, quant));
        if found.is_none() {
            let keys: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
            self.report_missing(&keys)?;
        }
        Ok(found)
    }
}

impl<'a> From<&'a HashMap<String, Array>> for Weights<'a> {
    /// Strict loading of `map`.
    fn from(map: &'a HashMap<String, Array>) -> Self {
        Self::new(map, LoadOptions::default())
    }
}

impl Deref for Weights<'_> {
    type Target = HashMap<String, Array>;

    fn deref(&self) -> &Self::Target {
        self.map
    }
}

/// Set `param` from `key` if the layer has that parameter.
fn load_optional_param(param: &mut Option<Array>, weights: &Weights, key: &str) -> Result<()> {
    if param.is_some()
        && let Some(value) = weights.fetch(key)?
    {
        *param = Some(value.clone());
    }
    Ok(())
}

/// Load a LayerNorm's affine parameters, each needed if the layer has it.
pub(crate) fn load_layer_norm(
    layer: &mut nn::LayerNorm,
    weights: &Weights,
    prefix: &str,
) -> Result<()> {
    load_optional_param(
        &mut layer.weight.value,
        weights,
        &format!("{prefix}.weight"),
    )?;
    load_optional_param(&mut layer.bias.value, weights, &format!("{prefix}.bias"))
}

/// Load Conv1d weights (already in MLX `[O, K, I]` layout); the bias is
/// needed if the layer has one.
pub(crate) fn load_conv1d(layer: &mut nn::Conv1d, weights: &Weights, prefix: &str) -> Result<()> {
    if let Some(weight) = weights.fetch(&format!("{prefix}.weight"))? {
        layer.weight.value = weight.clone();
    }
    load_optional_param(&mut layer.bias.value, weights, &format!("{prefix}.bias"))
}

/// Load Conv2d weights (already in MLX `[O, H, W, I]` layout); the bias is
/// needed if the layer has one.
pub(crate) fn load_conv2d(layer: &mut nn::Conv2d, weights: &Weights, prefix: &str) -> Result<()> {
    if let Some(weight) = weights.fetch(&format!("{prefix}.weight"))? {
        layer.weight.value = weight.clone();
    }
    load_optional_param(&mut layer.bias.value, weights, &format!("{prefix}.bias"))
}

/// Load BatchNorm parameters and running statistics, each needed if the layer
/// has it, and switch the layer to inference mode.
pub(crate) fn load_batch_norm(
    layer: &mut nn::BatchNorm,
    weights: &Weights,
    prefix: &str,
) -> Result<()> {
    for (name, param) in [
        ("weight", &mut layer.weight.value),
        ("bias", &mut layer.bias.value),
        ("running_mean", &mut layer.running_mean.value),
        ("running_var", &mut layer.running_var.value),
    ] {
        load_optional_param(param, weights, &format!("{prefix}.{name}"))?;
    }
    // Use the loaded running stats, not batch stats
    layer.training = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> HashMap<String, Array> {
        HashMap::from([("ln.weight".to_string(), Array::ones::<f32>(&[4]).unwrap())])
    }

    #[test]
    fn test_strict_load_fails_on_missing_tensor() {
        let map = checkpoint();
        let weights = Weights::from(&map);
        let mut ln = nn::LayerNorm::new(4).unwrap();
        let err = load_layer_norm(&mut ln, &weights, "ln").unwrap_err();
        assert!(err.to_string().contains("ln.bias"), "{err}");
    }

    #[test]
    fn test_lenient_load_reports_missing_tensor() {
        let map = checkpoint();
        let weights = Weights::new(&map, LoadOptions::default().strict(false));
        let mut ln = nn::LayerNorm::new(4).unwrap();
        load_layer_norm(&mut ln, &weights, "ln").unwrap();
        assert_eq!(weights.report().missing, vec!["ln.bias".to_string()]);
        assert!(!weights.report().is_complete());
    }

    #[test]
    fn test_fetch_any_reports_canonical_key() {
        let map = checkpoint();
        let weights = Weights::new(&map, LoadOptions::default().strict(false));
        let keys = ["a.weight".to_string(), "b.weight".to_string()];
        assert!(weights.fetch_any(&keys).unwrap().is_none());
        assert_eq!(weights.report().missing, vec!["a.weight".to_string()]);

        let strict = Weights::from(&map);
        let err = strict.fetch_any(&keys).unwrap_err().to_string();
        assert!(err.contains("a.weight` or `b.weight"), "{err}");
    }
}
