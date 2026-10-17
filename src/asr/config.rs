//! ASR model configuration types.
//!
//! These types mirror NeMo's configuration structure for ASR models,
//! allowing loading of NeMo checkpoint configurations.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

use crate::core::error::{Error, Result};

/// Declare an enum that (de)serializes as NeMo's string values. A value this
/// crate doesn't know parses as `Other`, so configs from newer NeMo versions
/// still load; the parts of the model that can't run it reject it at load
/// time instead.
macro_rules! config_enum {
    (
        $(#[$meta:meta])*
        $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident = $value:literal $(| $alias:literal)* ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[non_exhaustive]
        pub enum $name {
            $(
                $(#[$vmeta])*
                #[serde(rename = $value $(, alias = $alias)*)]
                $variant,
            )+
            /// A value this crate doesn't recognize
            #[serde(untagged)]
            Other(String),
        }

        impl $name {
            /// The value as written in NeMo configs.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $value,)+
                    Self::Other(value) => value,
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

config_enum!(
    /// STFT analysis window (`preprocessor.window`).
    WindowType {
        /// Hann window
        #[default]
        Hann = "hann",
        /// Hamming window
        Hamming = "hamming",
    }
);

config_enum!(
    /// Mel feature normalization (`preprocessor.normalize`).
    Normalize {
        /// Zero mean, unit variance per mel bin over time
        #[default]
        PerFeature = "per_feature",
        /// Zero mean, unit variance over all features and time
        AllFeatures = "all_features",
        /// No normalization (NeMo writes `NA` or `None`)
        Disabled = "NA" | "None" | "none",
    }
);

config_enum!(
    /// Encoder subsampling module (`encoder.subsampling`).
    Subsampling {
        /// Depthwise-separable striding convolutions (FastConformer)
        #[default]
        DwStriding = "dw_striding",
        /// Striding convolutions
        Striding = "striding",
        /// VGGNet-style subsampling
        Vggnet = "vggnet",
        /// Frame stacking
        Stacking = "stacking",
    }
);

config_enum!(
    /// Encoder self-attention (`encoder.self_attention_model`).
    SelfAttentionModel {
        /// Relative positional attention (Transformer-XL style)
        #[default]
        RelPos = "rel_pos",
        /// Absolute positional attention
        AbsPos = "abs_pos",
        /// Relative positional attention over a local window
        RelPosLocalAttn = "rel_pos_local_attn",
    }
);

config_enum!(
    /// How attention context is limited (`encoder.att_context_style`).
    AttContextStyle {
        /// Each frame sees a fixed window around it
        #[default]
        Regular = "regular",
        /// Frames attend within chunks, as in cache-aware streaming
        ChunkedLimited = "chunked_limited",
    }
);

config_enum!(
    /// Normalization in the conformer convolution module
    /// (`encoder.conv_norm_type`).
    ConvNormType {
        /// Batch normalization
        #[default]
        BatchNorm = "batch_norm",
        /// Layer normalization (cache-aware streaming models)
        LayerNorm = "layer_norm",
    }
);

config_enum!(
    /// An activation function.
    Activation {
        /// `max(x, 0)`
        #[default]
        Relu = "relu",
        /// `tanh(x)`
        Tanh = "tanh",
        /// `1 / (1 + e^-x)`
        Sigmoid = "sigmoid",
        /// Gaussian error linear unit
        Gelu = "gelu",
        /// `x * sigmoid(x)`
        Swish = "swish" | "silu",
    }
);

config_enum!(
    /// Decoding algorithm named in a model's config (`decoding.strategy`).
    DecodingAlgorithm {
        /// Greedy decoding
        #[default]
        Greedy = "greedy",
        /// Batched greedy decoding
        GreedyBatch = "greedy_batch",
        /// Beam search
        Beam = "beam" | "beam_search",
        /// Time-synchronous beam search
        Tsl = "tsl",
        /// Alignment-length synchronous beam search
        Alsd = "alsd",
        /// Modified adaptive expansion search
        Maes = "maes",
    }
);

/// A `null` value as the type's default.
fn null_as_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// A `null` flag as `true`.
fn null_as_true<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<bool, D::Error> {
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(true))
}

/// A count where NeMo writes `-1` (or `null`) for "not set".
fn negative_as_none<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error> {
    Ok(Option::<i64>::deserialize(deserializer)?.and_then(|n| usize::try_from(n).ok()))
}

/// Attention context size configuration.
///
/// NeMo supports two formats:
/// - Single: `[left, right]` - fixed context window (Parakeet style)
/// - Multiple: `[[70,13], [70,6], [70,1], [70,0]]` - multiple streaming modes (Nemotron style)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
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
    /// For Multiple, returns the first mode, which NeMo uses by default. NeMo
    /// checkpoints list modes from most to least lookahead (right context), so
    /// this is the highest-latency, most accurate mode.
    pub fn default_mode(&self) -> Option<(i32, i32)> {
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
    pub fn mode(&self, mode_idx: usize) -> Option<(i32, i32)> {
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
#[non_exhaustive]
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
    spec_augment: Option<yaml_serde::Value>,

    /// Additional fields we don't need to parse
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
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

    /// Whether the model conditions its encoder output on a language prompt
    /// (NeMo `EncDecRNNTBPEModelWithPrompt`, e.g. nemotron-3.5-asr-streaming).
    pub fn is_prompt_conditioned(&self) -> bool {
        let initialize = self
            .extra
            .get("model_defaults")
            .and_then(|d| d.get("initialize_prompt_feature"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        initialize
            || self
                .target
                .as_deref()
                .is_some_and(|t| t.contains("WithPrompt"))
    }

    /// Language prompt settings from `model_defaults`, if the model has a prompt dictionary.
    pub fn prompt_config(&self) -> Option<PromptConfig> {
        let defaults = self.extra.get("model_defaults")?;
        let dictionary: BTreeMap<String, usize> = defaults
            .get("prompt_dictionary")?
            .as_mapping()?
            .iter()
            .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.as_u64()? as usize)))
            .collect();
        if dictionary.is_empty() {
            return None;
        }
        let num_prompts = defaults
            .get("num_prompts")
            .and_then(|v| v.as_u64())
            .map_or(128, |n| n as usize);
        Some(PromptConfig {
            dictionary,
            num_prompts,
        })
    }

    /// Reject model variants that would load without error but decode incorrectly.
    ///
    /// Encoder settings are checked separately by [`EncoderConfig::validate`].
    pub fn validate(&self) -> Result<()> {
        if self.is_prompt_conditioned() && self.prompt_config().is_none() {
            return Err(Error::config(
                "prompt-conditioned model has no model_defaults.prompt_dictionary",
            ));
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

/// Language-ID prompt settings of a prompt-conditioned model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PromptConfig {
    /// Language key (e.g. `en-US`, `de`, `auto`) -> one-hot index
    pub dictionary: BTreeMap<String, usize>,
    /// Length of the one-hot prompt vector
    pub num_prompts: usize,
}

fn default_sample_rate() -> u32 {
    16000
}

/// Audio preprocessor configuration (Mel spectrogram).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

    /// STFT window
    #[serde(default, deserialize_with = "null_as_default")]
    pub window: WindowType,

    /// Number of Mel filterbank features
    #[serde(default = "default_features")]
    pub features: usize,

    /// FFT size
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,

    /// Whether to apply log to Mel spectrogram
    #[serde(default = "default_true")]
    pub log: bool,

    /// Feature normalization
    #[serde(default, deserialize_with = "null_as_default")]
    pub normalize: Normalize,

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
fn default_features() -> usize {
    80
}
fn default_n_fft() -> usize {
    512
}
fn default_true() -> bool {
    true
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
#[non_exhaustive]
pub struct EncoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (must match preprocessor output)
    #[serde(default = "default_features")]
    pub feat_in: usize,

    /// Output feature dimension (`None`, written `-1`, means `d_model`)
    #[serde(default, deserialize_with = "negative_as_none")]
    pub feat_out: Option<usize>,

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

    /// Subsampling module
    #[serde(default, deserialize_with = "null_as_default")]
    pub subsampling: Subsampling,

    /// Subsampling factor
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,

    /// Subsampling convolution channels
    #[serde(default)]
    pub subsampling_conv_channels: Option<usize>,

    /// Self-attention model
    #[serde(default, deserialize_with = "null_as_default")]
    pub self_attention_model: SelfAttentionModel,

    /// Attention context size [left, right] (-1 for unlimited)
    /// Can be a single [left, right] or multiple modes [[left1, right1], [left2, right2], ...]
    #[serde(default)]
    pub att_context_size: Option<AttContextSize>,

    /// How attention context is limited
    #[serde(default, deserialize_with = "null_as_default")]
    pub att_context_style: AttContextStyle,

    /// Scale the subsampled input by `sqrt(d_model)` (default true)
    #[serde(default = "default_true", deserialize_with = "null_as_true")]
    pub xscaling: bool,

    /// Give each layer its own position biases (default true)
    #[serde(default = "default_true", deserialize_with = "null_as_true")]
    pub untie_biases: bool,

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

    /// Normalization in the convolution module
    #[serde(default, deserialize_with = "null_as_default")]
    pub conv_norm_type: ConvNormType,

    /// Convolution context size: null (symmetric, default) or "causal" (left-only padding)
    #[serde(default)]
    pub conv_context_size: Option<String>,

    /// Pad subsampling convolutions on the left only (causal)
    #[serde(default, deserialize_with = "null_as_default")]
    pub causal_downsampling: bool,

    /// Give linear and convolution layers a bias (default true; false for
    /// Nemotron)
    #[serde(default = "default_true", deserialize_with = "null_as_true")]
    pub use_bias: bool,

    /// Additional fields
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
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
fn default_subsampling_factor() -> usize {
    8
}
fn default_pos_emb_max_len() -> usize {
    5000
}
fn default_dropout() -> f32 {
    0.1
}
/// Decoder configuration (CTC output projection).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (from encoder)
    #[serde(default)]
    pub feat_in: Option<usize>,

    /// Number of output classes (vocabulary size), if set (NeMo writes `-1`
    /// to take it from the tokenizer)
    #[serde(default, deserialize_with = "negative_as_none")]
    pub num_classes: Option<usize>,

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

    /// Whether the blank token doubles as padding (default true)
    #[serde(default = "default_true", deserialize_with = "null_as_true")]
    pub blank_as_pad: bool,
}

/// Prediction network configuration (for RNNT models).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// Activation function (relu, tanh or sigmoid)
    #[serde(default, deserialize_with = "null_as_default")]
    pub activation: Activation,

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
#[non_exhaustive]
pub struct JointNetConfig {
    /// Joint network hidden dimension
    #[serde(default = "default_joint_hidden")]
    pub joint_hidden: usize,

    /// Activation function
    #[serde(default, deserialize_with = "null_as_default")]
    pub activation: Activation,

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
            activation: Activation::Relu,
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
    pub fn effective_activation(&self) -> &Activation {
        self.jointnet
            .as_ref()
            .map_or(&self.activation, |jn| &jn.activation)
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

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            target: None,
            joint_hidden: 640,
            encoder_hidden: None,
            pred_hidden: None,
            num_classes: None,
            vocabulary: None,
            activation: Activation::Relu,
            fuse_loss_wer: None,
            fused_batch_size: None,
            experimental_fuse_loss_wer: None,
            jointnet: None,
        }
    }
}

/// Tokenizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct DecodingConfig {
    /// Decoding algorithm
    #[serde(default, deserialize_with = "null_as_default")]
    pub strategy: DecodingAlgorithm,

    /// Beam width for beam search
    #[serde(default)]
    pub beam_size: Option<usize>,

    /// Additional fields
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            target: None,
            sample_rate: 16000,
            window_size: 0.025,
            window_stride: 0.01,
            window: WindowType::Hann,
            features: 80,
            n_fft: 512,
            log: true,
            normalize: Normalize::PerFeature,
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
            feat_out: None,
            d_model: 512,
            n_layers: 17,
            n_heads: 8,
            ff_expansion_factor: 4.0,
            conv_kernel_size: 31,
            subsampling: Subsampling::DwStriding,
            subsampling_factor: 8,
            subsampling_conv_channels: None,
            self_attention_model: SelfAttentionModel::RelPos,
            att_context_size: None,
            att_context_style: AttContextStyle::Regular,
            xscaling: true,
            untie_biases: true,
            pos_emb_max_len: 5000,
            dropout: 0.1,
            dropout_att: None,
            dropout_pre_encoder: None,
            conv_norm_type: ConvNormType::BatchNorm,
            conv_context_size: None,
            causal_downsampling: false,
            use_bias: true,
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

    /// Check if this is a streaming-capable encoder (chunked_limited style).
    pub fn is_streaming(&self) -> bool {
        self.att_context_style == AttContextStyle::ChunkedLimited
    }

    /// Reject encoder settings this port does not implement.
    ///
    /// An unsupported architecture could otherwise share enough weight names
    /// with a supported one to load, and silently produce garbage output.
    pub fn validate(&self) -> Result<()> {
        if let Some(target) = self.target.as_deref() {
            if !target.ends_with("ConformerEncoder") {
                return Err(Error::not_implemented(format!("encoder `{}`", target)));
            }
        }

        if self.self_attention_model != SelfAttentionModel::RelPos {
            return Err(Error::not_implemented(format!(
                "self_attention_model `{}` (only rel_pos is supported)",
                self.self_attention_model
            )));
        }

        if let AttContextStyle::Other(style) = &self.att_context_style {
            return Err(Error::not_implemented(format!(
                "att_context_style `{}` (only regular and chunked_limited are supported)",
                style
            )));
        }

        if let ConvNormType::Other(norm) = &self.conv_norm_type {
            return Err(Error::not_implemented(format!(
                "conv_norm_type `{}` (only batch_norm and layer_norm are supported)",
                norm
            )));
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

        if self.subsampling != Subsampling::DwStriding {
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
            num_classes: None,
            vocabulary: Vec::new(),
            prednet: None,
            context_size: None,
            normalization_mode: None,
            blank_as_pad: true,
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
            strategy: DecodingAlgorithm::Greedy,
            beam_size: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

// ==================== Jasper / Classification Config ====================

/// Top-level configuration for classification models (e.g., VAD with MarbleNet).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

/// Jasper/ConvASR encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JasperEncoderConfig {
    /// Target class
    #[serde(rename = "_target_", default)]
    pub target: Option<String>,

    /// Input feature dimension (must match preprocessor output)
    #[serde(default = "default_features")]
    pub feat_in: usize,

    /// Activation function
    #[serde(default, deserialize_with = "null_as_default")]
    pub activation: Activation,

    /// Jasper block configurations
    pub jasper: Vec<JasperBlockConfig>,

    /// Additional fields
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

/// Configuration for a single Jasper block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

fn default_repeat() -> usize {
    1
}

fn default_residual_dense() -> bool {
    false
}

/// Decoder configuration for classification models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

fn default_num_classes_2() -> usize {
    2
}

// ==================== SortFormer / Diarization Config ====================

/// Configuration for the SortFormer transformer encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

    /// Activation function of the feed-forward layers (relu or gelu)
    #[serde(default, deserialize_with = "null_as_default")]
    pub hidden_act: Activation,

    /// Whether to use pre-layer-norm (True) or post-layer-norm (False)
    #[serde(default)]
    pub pre_ln: bool,

    /// Whether to add a final layer norm (only used when pre_ln=True)
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
    extra: std::collections::HashMap<String, yaml_serde::Value>,
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

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 18,
            hidden_size: 192,
            inner_size: 768,
            num_attention_heads: 8,
            hidden_act: Activation::Relu,
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
#[non_exhaustive]
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

    /// Speaker cache length in encoder frames
    #[serde(default = "default_spkcache_len")]
    pub spkcache_len: usize,

    /// FIFO buffer length in encoder frames
    #[serde(default)]
    pub fifo_len: usize,

    /// Streaming chunk length in encoder frames
    #[serde(default = "default_chunk_len")]
    pub chunk_len: usize,

    /// Number of frames popped from the FIFO into the speaker cache per update
    #[serde(default = "default_spkcache_update_period")]
    pub spkcache_update_period: usize,

    /// Left context of each chunk in encoder frames
    #[serde(default = "default_chunk_context")]
    pub chunk_left_context: usize,

    /// Right context of each chunk in encoder frames
    #[serde(default = "default_chunk_context")]
    pub chunk_right_context: usize,

    /// Silence frames reserved per speaker in the compressed speaker cache
    #[serde(default = "default_spkcache_sil_frames_per_spk")]
    pub spkcache_sil_frames_per_spk: usize,

    /// Probability floor used when scoring frames for the speaker cache
    #[serde(default = "default_pred_score_threshold")]
    pub pred_score_threshold: f32,

    /// Score bonus for frames newly added to the speaker cache
    #[serde(default = "default_scores_boost_latest")]
    pub scores_boost_latest: f32,

    /// Frames whose summed speaker probability is below this count as silence
    #[serde(default = "default_sil_threshold")]
    pub sil_threshold: f32,

    /// Fraction of per-speaker cache slots that get a strong score boost
    #[serde(default = "default_strong_boost_rate")]
    pub strong_boost_rate: f32,

    /// Fraction of per-speaker cache slots that get a weak score boost
    #[serde(default = "default_weak_boost_rate")]
    pub weak_boost_rate: f32,

    /// Fraction of per-speaker cache slots that must score positively before
    /// overlapped frames are dropped
    #[serde(default = "default_min_pos_scores_rate")]
    pub min_pos_scores_rate: f32,

    /// Additional fields
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

fn default_spkcache_len() -> usize {
    188
}
fn default_chunk_len() -> usize {
    188
}
fn default_spkcache_update_period() -> usize {
    188
}
fn default_chunk_context() -> usize {
    1
}
fn default_spkcache_sil_frames_per_spk() -> usize {
    3
}
fn default_pred_score_threshold() -> f32 {
    0.25
}
fn default_scores_boost_latest() -> f32 {
    0.05
}
fn default_sil_threshold() -> f32 {
    0.2
}
fn default_strong_boost_rate() -> f32 {
    0.75
}
fn default_weak_boost_rate() -> f32 {
    1.5
}
fn default_min_pos_scores_rate() -> f32 {
    0.5
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
            spkcache_len: default_spkcache_len(),
            fifo_len: 0,
            chunk_len: default_chunk_len(),
            spkcache_update_period: default_spkcache_update_period(),
            chunk_left_context: default_chunk_context(),
            chunk_right_context: default_chunk_context(),
            spkcache_sil_frames_per_spk: default_spkcache_sil_frames_per_spk(),
            pred_score_threshold: default_pred_score_threshold(),
            scores_boost_latest: default_scores_boost_latest(),
            sil_threshold: default_sil_threshold(),
            strong_boost_rate: default_strong_boost_rate(),
            weak_boost_rate: default_weak_boost_rate(),
            min_pos_scores_rate: default_min_pos_scores_rate(),
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Top-level configuration for SortFormer diarization models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

    /// SortFormer modules (speaker head and streaming) configuration
    pub sortformer_modules: SortformerModulesConfig,

    /// Additional fields
    #[serde(flatten)]
    extra: std::collections::HashMap<String, yaml_serde::Value>,
}

/// Adds typed access to the config fields this crate does not model explicitly.
macro_rules! impl_extra_accessor {
    ($($ty:ty),* $(,)?) => {$(
        impl $ty {
            /// Deserialize an unmodeled config field by key.
            ///
            /// Returns `Ok(None)` if the key is absent, or an error if it cannot be
            /// deserialized as `T`.
            pub fn extra<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
                self.extra
                    .get(key)
                    .map(|v| {
                        yaml_serde::from_value(v.clone()).map_err(|e| {
                            Error::config_with_source(format!("Invalid config field `{}`", key), e)
                        })
                    })
                    .transpose()
            }
        }
    )*};
}

impl_extra_accessor!(
    AsrModelConfig,
    EncoderConfig,
    DecodingConfig,
    ClassificationModelConfig,
    JasperEncoderConfig,
    JasperBlockConfig,
    ClassificationDecoderConfig,
    TransformerEncoderConfig,
    SortformerModulesConfig,
    SortformerConfig,
);

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
    fn test_config_enums_parse_nemo_values() {
        let parse = |yaml: &str| -> PreprocessorConfig { yaml_serde::from_str(yaml).unwrap() };
        assert_eq!(parse("normalize: NA").normalize, Normalize::Disabled);
        assert_eq!(parse("normalize: None").normalize, Normalize::Disabled);
        assert_eq!(
            parse("normalize: per_feature").normalize,
            Normalize::PerFeature
        );
        assert_eq!(parse("{}").normalize, Normalize::PerFeature);
        assert_eq!(parse("window: hamming").window, WindowType::Hamming);
        // Unknown values load, and keep their text
        let other = parse("window: blackman").window;
        assert_eq!(other, WindowType::Other("blackman".to_string()));
        assert_eq!(other.as_str(), "blackman");

        let joint: JointConfig = yaml_serde::from_str("activation: silu").unwrap();
        assert_eq!(joint.activation, Activation::Swish);
        let decoding: DecodingConfig = yaml_serde::from_str("strategy: beam_search").unwrap();
        assert_eq!(decoding.strategy, DecodingAlgorithm::Beam);
    }

    #[test]
    fn test_config_counts_and_flags() {
        let decoder: DecoderConfig = yaml_serde::from_str("num_classes: -1").unwrap();
        assert_eq!(decoder.num_classes, None);
        let decoder: DecoderConfig = yaml_serde::from_str("num_classes: 1024").unwrap();
        assert_eq!(decoder.num_classes, Some(1024));

        let encoder: EncoderConfig =
            yaml_serde::from_str("feat_out: -1\nxscaling: null\nuse_bias: false").unwrap();
        assert_eq!(encoder.feat_out, None);
        assert!(encoder.xscaling);
        assert!(!encoder.use_bias);
        assert!(!encoder.causal_downsampling);
        assert_eq!(encoder.att_context_style, AttContextStyle::Regular);

        // Round trip keeps NeMo's spelling
        let yaml = yaml_serde::to_string(&encoder).unwrap();
        assert!(yaml.contains("att_context_style: regular"), "{yaml}");
    }

    #[test]
    fn test_encoder_validate_accepts_supported() {
        let mut config = EncoderConfig::default();
        config.target = Some("nemo.collections.asr.modules.ConformerEncoder".to_string());
        assert!(config.validate().is_ok());

        config.att_context_style = AttContextStyle::ChunkedLimited;
        config.att_context_size = Some(AttContextSize::Multiple(vec![vec![70, 13], vec![70, 0]]));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_validate_rejects_unsupported() {
        let cases: Vec<fn(&mut EncoderConfig)> = vec![
            |c| c.target = Some("nemo.collections.asr.modules.TransformerEncoder".to_string()),
            |c| c.self_attention_model = SelfAttentionModel::Other("rope".to_string()),
            |c| c.att_context_style = AttContextStyle::Other("chunked_limited_with_rc".to_string()),
            |c| c.att_context_size = Some(AttContextSize::Multiple(vec![vec![70, 1, 13]])),
            |c| c.subsampling = Subsampling::Other("feature_stacking".to_string()),
            |c| c.conv_norm_type = ConvNormType::Other("group_norm".to_string()),
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
        assert!(
            asr_config("nemo.collections.asr.models.EncDecRNNTBPEModel")
                .validate()
                .is_ok()
        );
        assert!(
            asr_config(
                "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
            )
            .validate()
            .is_err()
        );

        let mut prompt = asr_config(
            "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt",
        );
        prompt.extra.insert(
            "model_defaults".to_string(),
            yaml_serde::from_str(
                "num_prompts: 128\nprompt_dictionary: {en-US: 0, en: 0, 'no': 27, auto: 101}",
            )
            .unwrap(),
        );
        assert!(prompt.validate().is_ok());
        let prompt = prompt.prompt_config().unwrap();
        assert_eq!(prompt.num_prompts, 128);
        assert_eq!(prompt.dictionary["no"], 27);
        assert_eq!(prompt.dictionary["auto"], 101);

        let mut tdt = asr_config("nemo.collections.asr.models.EncDecRNNTBPEModel");
        tdt.extra.insert(
            "model_defaults".to_string(),
            yaml_serde::from_str("tdt_durations: [0, 1, 2, 3, 4]").unwrap(),
        );
        assert!(tdt.validate().is_err());
    }

    #[test]
    fn test_extra_accessor() {
        let config: AsrModelConfig = crate::core::config::from_yaml_str(
            "sample_rate: 16000\npreprocessor: {}\nencoder: {}\ndecoder: {}\n\
             model_defaults:\n  tdt_durations: [0, 1, 2]\n",
        )
        .unwrap();

        #[derive(Deserialize)]
        struct ModelDefaults {
            tdt_durations: Vec<u32>,
        }
        let defaults: ModelDefaults = config.extra("model_defaults").unwrap().unwrap();
        assert_eq!(defaults.tdt_durations, vec![0, 1, 2]);

        assert!(config.extra::<String>("missing").unwrap().is_none());
        assert!(config.extra::<String>("model_defaults").is_err());
    }
}
