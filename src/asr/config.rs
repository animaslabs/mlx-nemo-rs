//! ASR model configuration types.
//!
//! These types mirror NeMo's configuration structure for ASR models,
//! allowing loading of NeMo checkpoint configurations.

use serde::{Deserialize, Serialize};

/// Top-level configuration for ASR CTC models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrModelConfig {
    /// Target class (e.g., "nemo.collections.asr.models.EncDecCTCModelBPE")
    #[serde(default)]
    pub target: Option<String>,

    /// Sample rate in Hz
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Preprocessor configuration
    pub preprocessor: PreprocessorConfig,

    /// Encoder configuration
    pub encoder: EncoderConfig,

    /// Decoder configuration
    pub decoder: DecoderConfig,

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

    /// Pre-emphasis coefficient
    #[serde(default)]
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
    #[serde(default)]
    pub att_context_size: Option<Vec<i32>>,

    /// Whether to use x-scaling for attention
    #[serde(default)]
    pub xscaling: Option<bool>,

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

    /// Convolution normalization type
    #[serde(default = "default_conv_norm")]
    pub conv_norm_type: String,

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
            preemph: None,
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
            xscaling: None,
            pos_emb_max_len: 5000,
            dropout: 0.1,
            dropout_att: None,
            dropout_pre_encoder: None,
            conv_norm_type: "batch_norm".to_string(),
            extra: std::collections::HashMap::new(),
        }
    }
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            target: None,
            feat_in: None,
            num_classes: -1,
            vocabulary: Vec::new(),
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
