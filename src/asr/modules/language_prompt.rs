//! Language-ID prompt conditioning (NeMo `PromptStreamingMixin`).
//!
//! Prompt-conditioned models such as nemotron-3.5-asr-streaming-0.6b
//! concatenate a one-hot language vector to every encoder frame and project
//! back to the encoder dimension before the RNN-T joint:
//!
//! ```text
//! prompt_kernel(concat(h, one_hot(lang))) =
//!     Linear(d + K -> 2d) -> ReLU -> Linear(2d -> d)
//! ```

use std::collections::BTreeMap;

use mlx_rs::{Array, ops};

use crate::asr::config::AsrModelConfig;
use crate::core::error::{Error, Result};
use crate::core::loading::Weights;
use crate::core::quantization::{LinearLayer, QuantizationConfig};

/// Prompt name that lets the model detect the spoken language itself.
pub const AUTO_LANGUAGE: &str = "auto";

/// The `prompt_kernel` projection and the selected language.
pub struct LanguagePrompt {
    proj_in: LinearLayer,
    proj_out: LinearLayer,
    /// Language key -> prompt index (`model_defaults.prompt_dictionary`)
    dictionary: BTreeMap<String, usize>,
    num_prompts: usize,
    language: String,
    /// One-hot row for `language`, [num_prompts]
    one_hot: Array,
}

impl LanguagePrompt {
    /// The model's language prompt, or `None` for models without one (or, in
    /// a non-strict load, whose checkpoint lacks the prompt weights).
    pub fn from_config(
        config: &AsrModelConfig,
        weights: &Weights,
        quant: Option<&QuantizationConfig>,
    ) -> Result<Option<Self>> {
        match config.prompt_config() {
            Some(prompt) => Self::load(
                weights,
                "prompt_kernel",
                quant,
                prompt.dictionary,
                prompt.num_prompts,
            ),
            None => Ok(None),
        }
    }

    /// Load `prompt_kernel.{0,2}` weights.
    ///
    /// Starts with [`AUTO_LANGUAGE`] selected when the dictionary has it,
    /// matching NeMo's `transcribe()` default. Returns `None` if a non-strict
    /// load finds the weights missing.
    pub fn load(
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
        dictionary: BTreeMap<String, usize>,
        num_prompts: usize,
    ) -> Result<Option<Self>> {
        let proj_in = weights.fetch_linear(&format!("{prefix}.0"), quant)?;
        let proj_out = weights.fetch_linear(&format!("{prefix}.2"), quant)?;
        let (Some(proj_in), Some(proj_out)) = (proj_in, proj_out) else {
            return Ok(None);
        };
        if dictionary.is_empty() {
            return Err(Error::config("prompt model has an empty prompt_dictionary"));
        }
        let language = if dictionary.contains_key(AUTO_LANGUAGE) {
            AUTO_LANGUAGE.to_string()
        } else {
            dictionary.keys().next().cloned().unwrap_or_default()
        };
        let mut prompt = Self {
            proj_in,
            proj_out,
            dictionary,
            num_prompts,
            language: String::new(),
            one_hot: Array::zeros::<f32>(&[num_prompts as i32])?,
        };
        prompt.set_language(&language)?;
        Ok(Some(prompt))
    }

    /// Select the target language, e.g. `"en-US"`, `"de"` or [`AUTO_LANGUAGE`].
    pub fn set_language(&mut self, language: &str) -> Result<()> {
        let &index = self.dictionary.get(language).ok_or_else(|| {
            let known: Vec<&str> = self.dictionary.keys().map(String::as_str).collect();
            Error::config(format!(
                "unknown language '{language}'; supported: {}",
                known.join(", ")
            ))
        })?;
        if index >= self.num_prompts {
            return Err(Error::config(format!(
                "prompt index {index} for '{language}' exceeds num_prompts {}",
                self.num_prompts
            )));
        }
        let mut one_hot = vec![0.0f32; self.num_prompts];
        one_hot[index] = 1.0;
        self.one_hot = Array::from_slice(&one_hot, &[self.num_prompts as i32]);
        self.language = language.to_string();
        Ok(())
    }

    /// The selected language key.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Every accepted language key, including aliases such as `"en"`.
    pub fn languages(&self) -> impl Iterator<Item = &str> {
        self.dictionary.keys().map(String::as_str)
    }

    /// Condition encoder frames on the language: [..., d] -> [..., d].
    pub fn forward(&self, encoded: &Array) -> Result<Array> {
        let mut shape = encoded.shape().to_vec();
        *shape.last_mut().unwrap() = self.num_prompts as i32;
        let prompt = ops::broadcast_to(&self.one_hot.as_dtype(encoded.dtype())?, &shape)?;
        let x = ops::concatenate(&[encoded, &prompt], -1)?;
        let x = ops::maximum(&self.proj_in.forward(&x)?, &Array::from_f32(0.0))?;
        self.proj_out.forward(&x)
    }
}
