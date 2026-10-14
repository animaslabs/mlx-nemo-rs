//! ASR model configuration types.
//!
//! These types mirror NeMo's configuration structure for ASR models,
//! allowing loading of NeMo checkpoint configurations.

use serde::{Deserialize, Serialize};

use crate::core::error::{Error, Result};

/// Attention context size configuration.
///
/// NeMo supports two formats:
/// - Single: `[left, right]` - fixed context window (Parakeet style)
/// - Multiple: `[[70,13], [70,6], [70,1], [70,0]]` - multiple streaming modes (Nemotron style)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttContextSize {
    /// Single context window: [left_context, right_context]
    Single(Vec<i32>),
    /// Multiple streaming modes, each with [left_context, right_context]
    Multiple(Vec<Vec<i32>>),
}

impl AttContextSize {
    /// Get the default (first) context size as (left, right).
    ///
    /// For Single, returns the values directly.
    /// For Multiple, returns the first mode (typically the largest context / lowest latency).
    pub fn get_default(&self) -> Option<(i32, i32)> {
        match self {
            AttContextSize::Single(v) if v.len() >= 2 => Some((v[0], v[1])),
            AttContextSize::Multiple(modes) if !modes.is_empty() => {
                let first = &modes[0];
                if first.len() >= 2 {
                    Some((first[0], first[1]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get context size for a specific streaming mode index.
    ///
    /// For Single, always returns the same context regardless of index.
    /// For Multiple, returns the context for the given mode index.
    pub fn get_mode(&self, mode_idx: usize) -> Option<(i32, i32)> {
        match self {
            AttContextSize::Single(v) if v.len() >= 2 => Some((v[0], v[1])),
            AttContextSize::Multiple(modes) if mode_idx < modes.len() => {
                let mode = &modes[mode_idx];
                if mode.len() >= 2 {
                    Some((mode[0], mode[1]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get number of available streaming modes.
    pub fn num_modes(&self) -> usize {
        match self {
            AttContextSize::Single(_) => 1,
            AttContextSize::Multiple(modes) => modes.len(),
        }
    }
}

/// Top-level configuration for ASR models (CTC and RNNT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrModelConfig {
    /// Target class (e.g., "nemo.collections.asr.models.EncDecCTCModelBPE" or "...EncDecRNNTBPEModel")
    #[serde(default)]
    pub target: Option<String>,

    /// Sample rate in Hz
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Preprocessor configuration
    pub preprocessor: PreprocessorConfig,

    /// Encoder configuration
    pub encoder: EncoderConfig,

    /// Decoder configuration (CTC decoder or RNNT prediction network)
    pub decoder: DecoderConfig,

    /// Joint network configuration (RNNT models only)
    #[serde(default)]
    pub joint: Option<JointConfig>,

    /// Tokenizer configuration
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>,

    /// Decoding strategy configuration
    #[serde(default)]
    pub decoding: Option<DecodingConfig>,

    /// Spec augmentation (training only)
    #[serde(default)]
    pub spec_augment: Option<serde_yaml::Value>,

    /// Additional fields we don't need to parse
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

impl AsrModelConfig {
    /// Check if this is an RNNT model based on the target class or presence of joint config.
    pub fn is_rnnt(&self) -> bool {
        if let Some(ref target) = self.target {
            target.contains("RNNT") || target.contains("Transducer")
        } else {
            // Fall back to checking if joint config is present
            self.joint.is_some()
        }
    }

    /// Check if this is a CTC model.
    pub fn is_ctc(&self) -> bool {
        !self.is_rnnt()
    }

    /// Reject model variants that would load without error but decode incorrectly.
    ///
    /// Encoder settings are checked separately by [`EncoderConfig::validate`].
    pub fn validate(&self) -> Result<()> {
        if let Some(target) = self.target.as_deref() {
            if target.contains("WithPrompt") {
                return Err(Error::not_implemented(format!(
                    "prompt-conditioned model `{}` (prompt_kernel and target language are not supported)",
                    target
                )));
            }
        }

        let has_tdt_durations = self
            .extra
            .get("model_defaults")
            .and_then(|d| d.get("tdt_durations"))
            .and_then(|d| d.as_sequence())
            .is_some_and(|d| !d.is_empty());
        if has_tdt_durations {
            return Err(Error::not_implemented(
                "TDT (token-and-duration transducer) models",
            ));
        }

        Ok(())
    }
}

fn default_sample_rate() -> u32 {
    16000
}

/// Audio preprocessor configuration (Mel spectrogram).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Sample rate in Hz
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Window size in seconds
    #[serde(default = "default_window_size")]
    pub window_size: f32,

    /// Window stride (hop) in seconds
    #[serde(default = "default_window_stride")]
    pub window_stride: f32,

    /// Window type (e.g., "hann", "hamming")
    #[serde(default = "default_window")]
    pub window: String,

    /// Number of Mel filterbank features
    #[serde(default = "default_features")]
    pub features: usize,

    /// FFT size
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,

    /// Whether to apply log to Mel spectrogram
    #[serde(default = "default_true")]
    pub log: bool,

    /// Whether to zero-mean normalize per feature
    #[serde(default = "default_normalize")]
    pub normalize: String,

    /// Dither amount for numerical stability
    #[serde(default = "default_dither")]
    pub dither: f32,

    /// Padding to ensure consistent frame count
    #[serde(default)]
    pub pad_to: usize,

    /// Pre-emphasis coefficient (default 0.97 to match NeMo)
    #[serde(default = "default_preemph")]
    pub preemph: Option<f32>,

    /// Mel frequency floor
    #[serde(default)]
    pub lowfreq: Option<f32>,

    /// Mel frequency ceiling
    #[serde(default)]
    pub highfreq: Option<f32>,
}

fn default_window_size() -> f32 {
    0.025
}
fn default_window_stride() -> f32 {
    0.01
}
fn default_window() -> String {
    "hann".to_string()
}
fn default_features() -> usize {
    80
}
fn default_n_fft() -> usize {
    512
}
fn default_true() -> bool {
    true
}
fn default_normalize() -> String {
    "per_feature".to_string()
}
fn default_dither() -> f32 {
    1e-5
}
fn default_preemph() -> Option<f32> {
    // NeMo's FilterbankFeatures has a hardcoded default of 0.97 when preemph is not in config
    Some(0.97)
}

/// Encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (must match preprocessor output)
    #[serde(default = "default_features")]
    pub feat_in: usize,

    /// Output feature dimension (-1 means same as d_model)
    #[serde(default = "default_feat_out")]
    pub feat_out: i32,

    /// Model dimension
    #[serde(default = "default_d_model")]
    pub d_model: usize,

    /// Number of encoder layers
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,

    /// Number of attention heads
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,

    /// Feed-forward dimension multiplier
    #[serde(default = "default_ff_expansion")]
    pub ff_expansion_factor: f32,

    /// Convolution kernel size
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel_size: usize,

    /// Subsampling type
    #[serde(default = "default_subsampling")]
    pub subsampling: String,

    /// Subsampling factor
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,

    /// Subsampling convolution channels
    #[serde(default)]
    pub subsampling_conv_channels: Option<usize>,

    /// Self-attention model type
    #[serde(default = "default_self_attention_model")]
    pub self_attention_model: String,

    /// Attention context size [left, right] (-1 for unlimited)
    /// Can be a single [left, right] or multiple modes [[left1, right1], [left2, right2], ...]
    #[serde(default)]
    pub att_context_size: Option<AttContextSize>,

    /// Attention context style: "regular" (default) or "chunked_limited" (streaming)
    #[serde(default)]
    pub att_context_style: Option<String>,

    /// Whether to use x-scaling for attention
    #[serde(default)]
    pub xscaling: Option<bool>,

    /// Whether to untie position biases across layers (each layer has its own bias)
    #[serde(default)]
    pub untie_biases: Option<bool>,

    /// Positional embedding max length
    #[serde(default = "default_pos_emb_max_len")]
    pub pos_emb_max_len: usize,

    /// Dropout rate
    #[serde(default = "default_dropout")]
    pub dropout: f32,

    /// Dropout rate for attention
    #[serde(default)]
    pub dropout_att: Option<f32>,

    /// Dropout rate before encoder
    #[serde(default)]
    pub dropout_pre_encoder: Option<f32>,

    /// Convolution normalization type: "batch_norm" (default) or "layer_norm"
    #[serde(default = "default_conv_norm")]
    pub conv_norm_type: String,

    /// Convolution context size: null (symmetric, default) or "causal" (left-only padding)
    #[serde(default)]
    pub conv_context_size: Option<String>,

    /// Whether to use causal (left-only) padding in subsampling convolutions
    #[serde(default)]
    pub causal_downsampling: Option<bool>,

    /// Whether to use bias in linear layers (default true for Parakeet, false for Nemotron)
    #[serde(default)]
    pub use_bias: Option<bool>,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_feat_out() -> i32 {
    -1
}
fn default_d_model() -> usize {
    512
}
fn default_n_layers() -> usize {
    17
}
fn default_n_heads() -> usize {
    8
}
fn default_ff_expansion() -> f32 {
    4.0
}
fn default_conv_kernel() -> usize {
    31
}
fn default_subsampling() -> String {
    "dw_striding".to_string()
}
fn default_subsampling_factor() -> usize {
    8
}
fn default_self_attention_model() -> String {
    "rel_pos".to_string()
}
fn default_pos_emb_max_len() -> usize {
    5000
}
fn default_dropout() -> f32 {
    0.1
}
fn default_conv_norm() -> String {
    "batch_norm".to_string()
}

/// Decoder configuration (CTC output projection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (from encoder)
    #[serde(default)]
    pub feat_in: Option<usize>,

    /// Number of output classes (vocabulary size)
    #[serde(default)]
    pub num_classes: i32,

    /// Vocabulary (list of tokens)
    #[serde(default)]
    pub vocabulary: Vec<String>,

    // --- RNNT-specific fields (for StatelessTransducerDecoder) ---
    /// Prediction network hidden dimension
    #[serde(default)]
    pub prednet: Option<PredNetConfig>,

    /// Context size for stateless decoder (number of previous tokens to consider)
    #[serde(default)]
    pub context_size: Option<usize>,

    /// Normalization mode for prediction network
    #[serde(default)]
    pub normalization_mode: Option<String>,

    /// Whether blank token is used as padding
    #[serde(default)]
    pub blank_as_pad: Option<bool>,
}

/// Prediction network configuration (for RNNT models).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredNetConfig {
    /// Hidden dimension of the prediction network
    #[serde(default = "default_pred_hidden")]
    pub pred_hidden: usize,

    /// Number of RNN layers (for LSTM-based decoder, 0 for stateless)
    #[serde(default)]
    pub pred_rnn_layers: Option<usize>,

    /// Maximum time for chrono initialization
    #[serde(default)]
    pub t_max: Option<usize>,

    /// Dropout rate
    #[serde(default)]
    pub dropout: Option<f32>,
}

fn default_pred_hidden() -> usize {
    640
}

impl Default for PredNetConfig {
    fn default() -> Self {
        Self {
            pred_hidden: 640,
            pred_rnn_layers: None,
            t_max: None,
            dropout: None,
        }
    }
}

impl PredNetConfig {
    /// Check if this is an LSTM-based decoder (has pred_rnn_layers > 0).
    pub fn is_lstm_based(&self) -> bool {
        self.pred_rnn_layers.map(|n| n > 0).unwrap_or(false)
    }

    /// Check if this is a stateless decoder.
    pub fn is_stateless(&self) -> bool {
        !self.is_lstm_based()
    }
}

/// Joint network configuration (for RNNT models).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Joint network hidden dimension (direct or from jointnet)
    #[serde(default = "default_joint_hidden")]
    pub joint_hidden: usize,

    /// Encoder hidden dimension (input from encoder)
    #[serde(default)]
    pub encoder_hidden: Option<usize>,

    /// Prediction network hidden dimension (input from prediction network)
    #[serde(default)]
    pub pred_hidden: Option<usize>,

    /// Number of output classes (vocabulary size including blank)
    #[serde(default)]
    pub num_classes: Option<usize>,

    /// Vocabulary (list of tokens)
    #[serde(default)]
    pub vocabulary: Option<Vec<String>>,

    /// Activation function (relu, tanh, sigmoid)
    #[serde(default = "default_joint_activation")]
    pub activation: String,

    /// Whether to use fused batch computation in joint
    #[serde(default)]
    pub fuse_loss_wer: Option<bool>,

    /// Whether to use fused log softmax
    #[serde(default)]
    pub fused_batch_size: Option<usize>,

    /// Experimental: log softmax on CPU
    #[serde(default)]
    pub experimental_fuse_loss_wer: Option<bool>,

    /// Nested jointnet config (NeMo style)
    #[serde(default)]
    pub jointnet: Option<JointNetConfig>,
}

/// Nested joint network config (inside joint.jointnet in NeMo config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointNetConfig {
    /// Joint network hidden dimension
    #[serde(default = "default_joint_hidden")]
    pub joint_hidden: usize,

    /// Activation function
    #[serde(default = "default_joint_activation")]
    pub activation: String,

    /// Dropout rate
    #[serde(default)]
    pub dropout: Option<f32>,

    /// Encoder hidden dimension
    #[serde(default)]
    pub encoder_hidden: Option<usize>,

    /// Prediction network hidden dimension
    #[serde(default)]
    pub pred_hidden: Option<usize>,
}

impl Default for JointNetConfig {
    fn default() -> Self {
        Self {
            joint_hidden: 640,
            activation: "relu".to_string(),
            dropout: None,
            encoder_hidden: None,
            pred_hidden: None,
        }
    }
}

impl JointConfig {
    /// Get the effective joint hidden dimension (from jointnet if present).
    pub fn effective_joint_hidden(&self) -> usize {
        self.jointnet
            .as_ref()
            .map(|jn| jn.joint_hidden)
            .unwrap_or(self.joint_hidden)
    }

    /// Get the effective activation function.
    pub fn effective_activation(&self) -> &str {
        self.jointnet
            .as_ref()
            .map(|jn| jn.activation.as_str())
            .unwrap_or(&self.activation)
    }

    /// Get the effective encoder hidden dimension.
    pub fn effective_encoder_hidden(&self) -> Option<usize> {
        self.jointnet
            .as_ref()
            .and_then(|jn| jn.encoder_hidden)
            .or(self.encoder_hidden)
    }

    /// Get the effective pred hidden dimension.
    pub fn effective_pred_hidden(&self) -> Option<usize> {
        self.jointnet
            .as_ref()
            .and_then(|jn| jn.pred_hidden)
            .or(self.pred_hidden)
    }
}

fn default_joint_hidden() -> usize {
    640
}

fn default_joint_activation() -> String {
    "relu".to_string()
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            target: None,
            joint_hidden: 640,
            encoder_hidden: None,
            pred_hidden: None,
            num_classes: None,
            vocabulary: None,
            activation: "relu".to_string(),
            fuse_loss_wer: None,
            fused_batch_size: None,
            experimental_fuse_loss_wer: None,
            jointnet: None,
        }
    }
}

/// Tokenizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Directory or path to tokenizer model
    #[serde(default)]
    pub dir: Option<String>,

    /// Tokenizer type (bpe, wpe, etc.)
    #[serde(rename = "type", default = "default_tokenizer_type")]
    pub tokenizer_type: String,

    /// Model file path (alternative to dir)
    #[serde(default)]
    pub model: Option<String>,
}

fn default_tokenizer_type() -> String {
    "bpe".to_string()
}

/// Decoding strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingConfig {
    /// Decoding strategy (greedy, beam, etc.)
    #[serde(default = "default_strategy")]
    pub strategy: String,

    /// Beam width for beam search
    #[serde(default)]
    pub beam_size: Option<usize>,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_strategy() -> String {
    "greedy".to_string()
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            target: None,
            sample_rate: 16000,
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            features: 80,
            n_fft: 512,
            log: true,
            normalize: "per_feature".to_string(),
            dither: 1e-5,
            pad_to: 0,
            preemph: Some(0.97), // NeMo default
            lowfreq: None,
            highfreq: None,
        }
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            target: None,
            feat_in: 80,
            feat_out: -1,
            d_model: 512,
            n_layers: 17,
            n_heads: 8,
            ff_expansion_factor: 4.0,
            conv_kernel_size: 31,
            subsampling: "dw_striding".to_string(),
            subsampling_factor: 8,
            subsampling_conv_channels: None,
            self_attention_model: "rel_pos".to_string(),
            att_context_size: None,
            att_context_style: None,
            xscaling: None,
            untie_biases: None,
            pos_emb_max_len: 5000,
            dropout: 0.1,
            dropout_att: None,
            dropout_pre_encoder: None,
            conv_norm_type: "batch_norm".to_string(),
            conv_context_size: None,
            causal_downsampling: None,
            use_bias: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

impl EncoderConfig {
    /// Check if causal convolution is enabled.
    pub fn is_causal_conv(&self) -> bool {
        self.conv_context_size
            .as_ref()
            .map(|s| s == "causal")
            .unwrap_or(false)
    }

    /// Check if causal downsampling is enabled.
    pub fn is_causal_downsampling(&self) -> bool {
        self.causal_downsampling.unwrap_or(false)
    }

    /// Check if this is a streaming-capable encoder (chunked_limited style).
    pub fn is_streaming(&self) -> bool {
        self.att_context_style
            .as_ref()
            .map(|s| s == "chunked_limited")
            .unwrap_or(false)
    }

    /// Get effective use_bias setting (default true for backward compatibility).
    pub fn effective_use_bias(&self) -> bool {
        self.use_bias.unwrap_or(true)
    }

    /// Reject encoder settings this port does not implement.
    ///
    /// Weight loading is lenient, so an unsupported architecture would otherwise
    /// load into the wrong modules and silently produce garbage output.
    pub fn validate(&self) -> Result<()> {
        if let Some(target) = self.target.as_deref() {
            if !target.ends_with("ConformerEncoder") {
                return Err(Error::not_implemented(format!("encoder `{}`", target)));
            }
        }

        if self.self_attention_model != "rel_pos" {
            return Err(Error::not_implemented(format!(
                "self_attention_model `{}` (only rel_pos is supported)",
                self.self_attention_model
            )));
        }

        if let Some(style) = self.att_context_style.as_deref() {
            if style != "regular" && style != "chunked_limited" {
                return Err(Error::not_implemented(format!(
                    "att_context_style `{}` (only regular and chunked_limited are supported)",
                    style
                )));
            }
        }

        if let Some(context) = &self.att_context_size {
            let bad_len = match context {
                AttContextSize::Single(v) => v.len() != 2,
                AttContextSize::Multiple(modes) => modes.iter().any(|m| m.len() != 2),
            };
            if bad_len {
                return Err(Error::config(format!(
                    "att_context_size {:?} must contain [left, right] pairs",
                    context
                )));
            }
        }

        if self.subsampling != "dw_striding" {
            return Err(Error::not_implemented(format!(
                "subsampling `{}` (only dw_striding is supported)",
                self.subsampling
            )));
        }

        if ![2, 4, 8].contains(&self.subsampling_factor) {
            return Err(Error::not_implemented(format!(
                "subsampling_factor {} (only 2, 4 and 8 are supported)",
                self.subsampling_factor
            )));
        }

        Ok(())
    }
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            target: None,
            feat_in: None,
            num_classes: -1,
            vocabulary: Vec::new(),
            prednet: None,
            context_size: None,
            normalization_mode: None,
            blank_as_pad: None,
        }
    }
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            dir: None,
            tokenizer_type: "bpe".to_string(),
            model: None,
        }
    }
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            strategy: "greedy".to_string(),
            beam_size: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

// ==================== Jasper / Classification Config ====================

/// Top-level configuration for classification models (e.g., VAD with MarbleNet).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationModelConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Sample rate in Hz
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Preprocessor configuration
    pub preprocessor: PreprocessorConfig,

    /// Encoder configuration (Jasper blocks)
    pub encoder: JasperEncoderConfig,

    /// Decoder configuration
    pub decoder: ClassificationDecoderConfig,

    /// Class labels (e.g., ["0", "1"] for VAD)
    #[serde(default)]
    pub labels: Vec<String>,

    /// Additional fields we don't need to parse
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

/// Jasper/ConvASR encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JasperEncoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (must match preprocessor output)
    #[serde(default = "default_features")]
    pub feat_in: usize,

    /// Activation function (e.g., "relu")
    #[serde(default = "default_jasper_activation")]
    pub activation: String,

    /// Jasper block configurations
    pub jasper: Vec<JasperBlockConfig>,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_jasper_activation() -> String {
    "relu".to_string()
}

/// Configuration for a single Jasper block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JasperBlockConfig {
    /// Output channels (filters)
    pub filters: usize,

    /// Number of sub-block repetitions
    #[serde(default = "default_repeat")]
    pub repeat: usize,

    /// Kernel sizes
    pub kernel: Vec<usize>,

    /// Strides
    pub stride: Vec<usize>,

    /// Dilations
    pub dilation: Vec<usize>,

    /// Dropout rate
    #[serde(default)]
    pub dropout: f32,

    /// Whether to use residual connections
    #[serde(default)]
    pub residual: bool,

    /// Whether to use depthwise separable convolutions
    #[serde(default)]
    pub separable: bool,

    /// Number of residual connection groups
    #[serde(default = "default_residual_dense")]
    pub residual_dense: bool,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_repeat() -> usize {
    1
}

fn default_residual_dense() -> bool {
    false
}

/// Decoder configuration for classification models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationDecoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (from encoder)
    #[serde(default)]
    pub feat_in: usize,

    /// Number of output classes
    #[serde(default = "default_num_classes_2")]
    pub num_classes: usize,

    /// Whether to apply log softmax
    #[serde(default)]
    pub log_softmax: Option<bool>,

    /// Whether this returns logits per frame or per sequence
    #[serde(default = "default_true")]
    pub return_logits: bool,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_num_classes_2() -> usize {
    2
}

// ==================== SortFormer / Diarization Config ====================

/// Configuration for the SortFormer transformer encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEncoderConfig {
    /// Number of transformer layers
    #[serde(default = "default_tf_num_layers")]
    pub num_layers: usize,

    /// Hidden size (d_model for transformer)
    #[serde(default = "default_tf_hidden_size")]
    pub hidden_size: usize,

    /// Inner (feed-forward) size
    #[serde(default = "default_tf_inner_size")]
    pub inner_size: usize,

    /// Number of attention heads
    #[serde(default = "default_tf_num_heads")]
    pub num_attention_heads: usize,

    /// Activation function for FFN ("relu", "gelu", etc.)
    #[serde(default = "default_tf_hidden_act")]
    pub hidden_act: String,

    /// Whether to use pre-layer-norm (True) or post-layer-norm (False)
    #[serde(default)]
    pub pre_ln: bool,

    /// Whether to add a final layer norm (only used when pre_ln=False)
    #[serde(default)]
    pub pre_ln_final_layer_norm: bool,

    /// Dropout rate for attention (ignored at inference)
    #[serde(default)]
    pub attn_score_dropout: Option<f32>,

    /// Dropout rate for FFN (ignored at inference)
    #[serde(default)]
    pub ffn_dropout: Option<f32>,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_tf_num_layers() -> usize {
    18
}
fn default_tf_hidden_size() -> usize {
    192
}
fn default_tf_inner_size() -> usize {
    768
}
fn default_tf_num_heads() -> usize {
    8
}
fn default_tf_hidden_act() -> String {
    "relu".to_string()
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 18,
            hidden_size: 192,
            inner_size: 768,
            num_attention_heads: 8,
            hidden_act: "relu".to_string(),
            pre_ln: false,
            pre_ln_final_layer_norm: true,
            attn_score_dropout: None,
            ffn_dropout: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Configuration for the SortFormer speaker head modules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortformerModulesConfig {
    /// Number of speakers
    #[serde(default = "default_num_spks")]
    pub num_spks: usize,

    /// FastConformer output dimension
    #[serde(default = "default_fc_d_model")]
    pub fc_d_model: usize,

    /// Transformer hidden dimension
    #[serde(default = "default_tf_d_model")]
    pub tf_d_model: usize,

    /// Dropout rate (ignored at inference)
    #[serde(default)]
    pub dropout_rate: Option<f32>,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_num_spks() -> usize {
    4
}
fn default_fc_d_model() -> usize {
    512
}
fn default_tf_d_model() -> usize {
    192
}

impl Default for SortformerModulesConfig {
    fn default() -> Self {
        Self {
            num_spks: 4,
            fc_d_model: 512,
            tf_d_model: 192,
            dropout_rate: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Top-level configuration for SortFormer diarization models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortformerConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Maximum number of speakers
    #[serde(default = "default_num_spks")]
    pub max_num_of_spks: usize,

    /// Sample rate
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Preprocessor configuration
    pub preprocessor: PreprocessorConfig,

    /// FastConformer encoder configuration
    pub encoder: EncoderConfig,

    /// Transformer encoder configuration
    pub transformer_encoder: TransformerEncoderConfig,

    /// SortFormer modules (speaker head) configuration
    pub sortformer_modules: SortformerModulesConfig,

    /// Streaming chunk length in encoder frames
    #[serde(default = "default_chunk_len")]
    pub chunk_len: usize,

    /// Speaker cache length in encoder frames
    #[serde(default = "default_spkcache_len")]
    pub spkcache_len: usize,

    /// FIFO buffer length in encoder frames
    #[serde(default = "default_fifo_len")]
    pub fifo_len: usize,

    /// Additional fields
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_chunk_len() -> usize {
    50
}
fn default_spkcache_len() -> usize {
    100
}
fn default_fifo_len() -> usize {
    50
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asr_config(target: &str) -> AsrModelConfig {
        AsrModelConfig {
            target: Some(target.to_string()),
            sample_rate: 16000,
            preprocessor: PreprocessorConfig::default(),
            encoder: EncoderConfig::default(),
            decoder: DecoderConfig::default(),
            joint: None,
            tokenizer: None,
            decoding: None,
            spec_augment: None,
            extra: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_encoder_validate_accepts_supported() {
        let mut config = EncoderConfig::default();
        config.target = Some("nemo.collections.asr.modules.ConformerEncoder".to_string());
        assert!(config.validate().is_ok());

        config.att_context_style = Some("chunked_limited".to_string());
        config.att_context_size = Some(AttContextSize::Multiple(vec![vec![70, 13], vec![70, 0]]));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_validate_rejects_unsupported() {
        let cases: Vec<fn(&mut EncoderConfig)> = vec![
            |c| c.target = Some("nemo.collections.asr.modules.TransformerEncoder".to_string()),
            |c| c.self_attention_model = "rope".to_string(),
            |c| c.att_context_style = Some("chunked_limited_with_rc".to_string()),
            |c| c.att_context_size = Some(AttContextSize::Multiple(vec![vec![70, 1, 13]])),
            |c| c.subsampling = "feature_stacking".to_string(),
            |c| c.subsampling_factor = 16,
        ];
        for mutate in cases {
            let mut config = EncoderConfig::default();
            mutate(&mut config);
            assert!(config.validate().is_err(), "accepted {:?}", config);
        }
    }

    #[test]
    fn test_model_validate() {
        assert!(asr_config("nemo.collections.asr.models.EncDecRNNTBPEModel").validate().is_ok());
        assert!(asr_config(
            "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
        )
        .validate()
        .is_err());

        let mut tdt = asr_config("nemo.collections.asr.models.EncDecRNNTBPEModel");
        tdt.extra.insert(
            "model_defaults".to_string(),
            serde_yaml::from_str("tdt_durations: [0, 1, 2, 3, 4]").unwrap(),
        );
        assert!(tdt.validate().is_err());
    }
}
