//! Portable Operator contract.
//!
//! Operators describe inference semantics. They are intentionally separate from
//! Provider kernels: Providers may advertise and execute compatible operators,
//! but operator identity, attributes, tensor contracts, and validation stay
//! Runtime-owned and platform-neutral.

use crate::{
    CapabilityVersion, ComputeDType, DTypeDescriptor, LayoutDescriptor, ResourceAffinity,
    TensorDescriptor, TensorMutabilityKind,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const OPERATOR_CATALOG_VERSION: CapabilityVersion = CapabilityVersion::new(1, 0, 0);

/// Optional on every Operator (`enforce-local-multigpu-and-expert-placement`
/// tensor parallelism): how the Model Component declares the node may be
/// split across Devices. `column`: a matmul whose output features may be
/// split; `row`: a matmul whose input features are split, each Device
/// producing a partial sum; `heads`: a node operating per attention head;
/// `local`: an elementwise node applied to each Device's share;
/// `experts`: a `moe-experts` node whose experts may be split, whole, each
/// Device producing a partial sum over the experts it holds. A Provider
/// that does not split ignores it.
pub const TENSOR_PARALLEL_ATTRIBUTE: &str = "tensor_parallel";

/// The values [`TENSOR_PARALLEL_ATTRIBUTE`] takes.
pub const TENSOR_PARALLEL_KINDS: [&str; 5] = ["column", "row", "heads", "local", "experts"];
pub const OPERATOR_NAMESPACE: &str = "magnetar:operator";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorFamily {
    Tensor,
    LinearAlgebra,
    Normalization,
    PositionEncoding,
    Attention,
    Activation,
    Quantization,
    Layout,
    SamplingSupport,
    Control,
}

impl OperatorFamily {
    pub const ALL: [Self; 10] = [
        Self::Tensor,
        Self::LinearAlgebra,
        Self::Normalization,
        Self::PositionEncoding,
        Self::Attention,
        Self::Activation,
        Self::Quantization,
        Self::Layout,
        Self::SamplingSupport,
        Self::Control,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::LinearAlgebra => "linear-algebra",
            Self::Normalization => "normalization",
            Self::PositionEncoding => "position-encoding",
            Self::Attention => "attention",
            Self::Activation => "activation",
            Self::Quantization => "quantization",
            Self::Layout => "layout",
            Self::SamplingSupport => "sampling-support",
            Self::Control => "control",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperatorId {
    namespace: String,
    name: String,
    version: u32,
    family: OperatorFamily,
}

impl OperatorId {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        version: u32,
        family: OperatorFamily,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version,
            family,
        }
    }

    pub fn magnetar(name: impl Into<String>, version: u32, family: OperatorFamily) -> Self {
        Self::new(OPERATOR_NAMESPACE, name, version, family)
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub const fn family(&self) -> OperatorFamily {
        self.family
    }
}

impl fmt::Display for OperatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}@{}", self.namespace, self.name, self.version)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorRole {
    Input,
    Output,
    Storage,
    Compute,
    Accumulation,
    Index,
    Mask,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorLayoutKind {
    Contiguous,
    Strided,
    Blocked,
    Paged,
    ProviderOpaque,
    QuantizedPacked,
    AttentionSpecific,
    BrowserCompatible,
}

impl TensorLayoutKind {
    pub const fn component_visible(self) -> bool {
        !matches!(self, Self::ProviderOpaque)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorDTypeContract {
    pub role: TensorRole,
    pub supported: BTreeSet<ComputeDType>,
}

impl OperatorDTypeContract {
    pub fn new(role: TensorRole, supported: impl IntoIterator<Item = ComputeDType>) -> Self {
        Self {
            role,
            supported: supported.into_iter().collect(),
        }
    }

    pub fn accepts(&self, dtype: ComputeDType) -> bool {
        self.supported.is_empty() || self.supported.contains(&dtype)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorLayoutContract {
    pub supported: BTreeSet<TensorLayoutKind>,
    pub explicit_conversion_required: bool,
}

impl OperatorLayoutContract {
    pub fn new(supported: impl IntoIterator<Item = TensorLayoutKind>) -> Self {
        Self {
            supported: supported.into_iter().collect(),
            explicit_conversion_required: true,
        }
    }

    pub fn accepts(&self, layout: TensorLayoutKind) -> bool {
        self.supported.is_empty() || self.supported.contains(&layout)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeRule {
    Any,
    SameRank,
    SameShape,
    /// Every tensor has the first input's shape, except that the second
    /// input may instead have shape `[1]`: one value broadcast to every
    /// element (`extend-portable-operator-attributes`).
    SameShapeOrScalar,
    /// Two equally shaped nonempty vectors/row batches, followed by two
    /// scalar parameters; the output has the first input's shape.
    GeodesicResidual,
    Matmul,
    RmsNorm,
    RowBroadcastAdd,
    RowConcat,
    Rank(u64),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperatorMemoryBehavior {
    pub reads_input: bool,
    pub writes_output: bool,
    pub mutates_input: bool,
    pub aliases_output: bool,
    pub requires_workspace: bool,
    pub supports_in_place: bool,
    pub requires_host_visible: bool,
    pub requires_device_resident: bool,
    pub requires_pinned_memory: bool,
    pub supports_streaming_output: bool,
    pub supports_paged_kv_cache: bool,
}

impl OperatorMemoryBehavior {
    pub const fn pure() -> Self {
        Self {
            reads_input: true,
            writes_output: true,
            mutates_input: false,
            aliases_output: false,
            requires_workspace: false,
            supports_in_place: false,
            requires_host_visible: false,
            requires_device_resident: false,
            requires_pinned_memory: false,
            supports_streaming_output: false,
            supports_paged_kv_cache: false,
        }
    }

    pub const fn with_workspace(mut self) -> Self {
        self.requires_workspace = true;
        self
    }

    pub const fn with_paged_kv_cache(mut self) -> Self {
        self.supports_paged_kv_cache = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorDeterminism {
    pub deterministic: bool,
    pub influenced_by_dtype: bool,
    pub influenced_by_provider: bool,
    pub influenced_by_device: bool,
    pub influenced_by_kernel: bool,
    pub influenced_by_reduction: bool,
    pub influenced_by_layout: bool,
}

impl Default for OperatorDeterminism {
    fn default() -> Self {
        Self {
            deterministic: true,
            influenced_by_dtype: true,
            influenced_by_provider: true,
            influenced_by_device: true,
            influenced_by_kernel: true,
            influenced_by_reduction: false,
            influenced_by_layout: true,
        }
    }
}

/// Kernel feature a `rope` Kernel advertises when it honours the banded
/// frequency-scaling attributes ([`RopeBandedFrequencyScaling`]). The Runtime sends
/// those attributes only to a Provider whose every active `rope` Kernel
/// advertises it, and otherwise refuses the model: a Kernel that ignored them
/// would rotate with unscaled frequencies and silently degrade every token.
pub const ROPE_BANDED_FREQUENCY_SCALING_FEATURE: &str = "banded-frequency-scaling";

/// Kernel feature an `attention` Kernel advertises when it honours the
/// `scale` attribute (the factor on `QK^T`, replacing `1/sqrt(head_dimension)`).
/// The Runtime sends `scale` only to a Provider whose every active
/// `attention` Kernel advertises it: a Kernel that ignored it would compute
/// every score with the wrong temperature.
pub const ATTENTION_SCALE_FEATURE: &str = "attention-scale";

/// Kernel feature an `embedding` Kernel advertises when it honours `scale`.
pub const EMBEDDING_SCALE_FEATURE: &str = "embedding-scale";

/// Kernel feature an `rmsnorm` Kernel advertises when it honours
/// `head_count` and `head_dimension` (per-head normalization with a
/// `[head_dimension]` weight).
pub const RMSNORM_PER_HEAD_FEATURE: &str = "rmsnorm-per-head";

/// Kernel feature an `rmsnorm` Kernel advertises when it accepts a single
/// input (no weight, scale 1).
pub const RMSNORM_WITHOUT_WEIGHT_FEATURE: &str = "rmsnorm-without-weight";

/// Kernel feature a `rope` Kernel advertises when it honours
/// `partial_rotary_factor`.
pub const ROPE_PARTIAL_ROTARY_FEATURE: &str = "partial-rotary";

/// Kernel feature a `moe-experts` Kernel advertises when it honours
/// `activation`.
pub const MOE_EXPERTS_ACTIVATION_FEATURE: &str = "moe-experts-activation";

/// Kernel feature a `moe-experts` Kernel advertises when it accepts the
/// sixth input, `per_expert_scale`.
pub const MOE_PER_EXPERT_SCALE_FEATURE: &str = "per-expert-scale";

/// Kernel support for selection-biased sigmoid routing, optional grouping,
/// selected-score normalization and routed scaling.
pub const MOE_SIGMOID_ROUTING_FEATURE: &str = "sigmoid-expert-routing";

/// Ungated `relu(x)²` expert networks with compact two-weight operands.
pub const MOE_RELU_SQUARED_FEATURE: &str = "moe-relu-squared";

/// Elementwise `activation(kind = "relu_squared")`.
pub const ACTIVATION_RELU_SQUARED_FEATURE: &str = "activation-relu-squared";

/// Kernel feature an `attention` Kernel advertises when it honours
/// `value_head_dimension`: V rows and the output are that wide per head
/// while Q and K rows stay `head_dimension` wide (multi-head latent
/// attention, `express-latent-attention-forms`).
pub const ATTENTION_VALUE_HEAD_DIMENSION_FEATURE: &str = "attention-value-head-dimension";

/// Kernel feature an `attention` Kernel advertises when it honours
/// `value_head_stride`: V rows hold `kv_head_count` heads of that many
/// values, each head's value being its leading `value_head_dimension`
/// values, so V may be the key history itself
/// (`cache-latent-attention-history`).
pub const ATTENTION_VALUE_HEAD_STRIDE_FEATURE: &str = "attention-value-head-stride";

/// Kernel feature an `attention` Kernel advertises when it honours
/// `latent_kv_width`: K and V are latent rows `[c | k_rope]` and two more
/// inputs give each head's key `[c W_k_h | k_rope]` and value `c W_v_h`
/// (`chunk-latent-prefills`).
pub const ATTENTION_LATENT_KV_FEATURE: &str = "attention-latent-kv";

/// Kernel feature a `concat` Kernel advertises when it honours `axis =
/// "heads"`: the inputs joined head by head instead of row after row.
pub const CONCAT_HEADS_FEATURE: &str = "concat-heads";

/// Kernel feature a `rope` Kernel advertises when it honours YaRN frequency
/// scaling ([`RopeYarnFrequencyScaling`]).
pub const ROPE_YARN_FREQUENCY_SCALING_FEATURE: &str = "yarn-frequency-scaling";

/// The `kv-write` Kernel feature a Provider advertises when it creates and
/// writes TurboQuant KV stores (`turboquant-kv-history`).
pub const KV_WRITE_TURBOQUANT_FEATURE: &str = "turboquant-kv-store";

/// The `attention` Kernel feature a Provider advertises when its attention
/// reads K and V from TurboQuant KV stores (`turboquant-kv-history`).
pub const ATTENTION_TURBOQUANT_KV_FEATURE: &str = "turboquant-kv-read";

/// The `kv-write` Kernel feature a Provider advertises when it creates
/// TurboQuant KV stores whose first rows stay exact
/// (`keep-turboquant-kv-head-exact`).
pub const KV_WRITE_TURBOQUANT_EXACT_HEAD_FEATURE: &str = "turboquant-kv-exact-head";

/// The Kernel features an `operator` node with these `attributes` and
/// `input_count` inputs needs beyond the Operator's baseline form: the
/// optional attributes and operand forms a Kernel could otherwise ignore
/// and compute something else. A Runtime sends such a node only to a
/// Provider whose every active Kernel for that Operator advertises each of
/// them. Attributes whose absence or default value keeps the baseline
/// behaviour need nothing.
pub fn required_kernel_features(
    operator: &str,
    attributes: &BTreeMap<String, OperatorAttributeValue>,
    input_count: usize,
) -> Vec<&'static str> {
    let present = |name: &str| attributes.contains_key(name);
    let mut features = Vec::new();
    match operator {
        "attention" => {
            if present("scale") {
                features.push(ATTENTION_SCALE_FEATURE);
            }
            if present("value_head_dimension") {
                features.push(ATTENTION_VALUE_HEAD_DIMENSION_FEATURE);
            }
            if present("value_head_stride") {
                features.push(ATTENTION_VALUE_HEAD_STRIDE_FEATURE);
            }
            if present("latent_kv_width") || input_count > 3 {
                features.push(ATTENTION_LATENT_KV_FEATURE);
            }
        }
        "concat" => {
            if concat_axis(attributes) == Some(ConcatAxis::Heads) {
                features.push(CONCAT_HEADS_FEATURE);
            }
        }
        "embedding" if present("scale") => features.push(EMBEDDING_SCALE_FEATURE),
        "activation" => {
            if matches!(attributes.get("kind"), Some(OperatorAttributeValue::String(kind)) if kind == "relu_squared")
            {
                features.push(ACTIVATION_RELU_SQUARED_FEATURE);
            }
        }
        "rmsnorm" => {
            if present("head_count") || present("head_dimension") {
                features.push(RMSNORM_PER_HEAD_FEATURE);
            }
            if input_count < 2
                || matches!(
                    attributes.get("with_scale"),
                    Some(OperatorAttributeValue::Boolean(false))
                )
            {
                features.push(RMSNORM_WITHOUT_WEIGHT_FEATURE);
            }
        }
        "rope" => {
            let full_rotation = match attributes.get("partial_rotary_factor") {
                None => true,
                Some(OperatorAttributeValue::Float(factor)) => *factor == 1.0,
                Some(_) => false,
            };
            if !full_rotation {
                features.push(ROPE_PARTIAL_ROTARY_FEATURE);
            }
            if matches!(
                attributes.get("frequency_scaling"),
                Some(OperatorAttributeValue::String(value))
                    if value == RopeYarnFrequencyScaling::ATTRIBUTE_VALUE
            ) {
                features.push(ROPE_YARN_FREQUENCY_SCALING_FEATURE);
            }
        }
        "moe-experts" => {
            let silu = match attributes.get("activation") {
                None => true,
                Some(OperatorAttributeValue::String(name)) => name == "silu",
                Some(_) => false,
            };
            if !silu {
                features.push(MOE_EXPERTS_ACTIVATION_FEATURE);
            }
            let relu_squared = matches!(
                attributes.get("activation"),
                Some(OperatorAttributeValue::String(name)) if name == "relu_squared"
            );
            if relu_squared {
                features.push(MOE_RELU_SQUARED_FEATURE);
            }
            // Any routing attribute requires the feature, even malformed or
            // partial declarations: an older Kernel must never silently
            // execute such a node with its baseline softmax routing.
            let sigmoid_form = [
                "routing_scoring",
                "routing_normalization",
                "routing_scale",
                "routing_group_count",
                "routing_group_selected",
            ]
            .iter()
            .any(|name| present(name));
            if sigmoid_form {
                features.push(MOE_SIGMOID_ROUTING_FEATURE);
            }
            let base_inputs = if sigmoid_form && relu_squared {
                5
            } else if sigmoid_form {
                6
            } else {
                5
            };
            if input_count > base_inputs {
                features.push(MOE_PER_EXPERT_SCALE_FEATURE);
            }
        }
        _ => {}
    }
    features
}

/// Wavelength-banded RoPE frequency scaling: a model-family-neutral scheme
/// (a source format may name it after the family that introduced it, e.g.
/// a Hugging Face `rope_type`; the loader maps that name onto this one).
/// Unlike linear position scaling it rescales each pair's
/// inverse frequency by its wavelength: short wavelengths (high frequencies)
/// are kept, wavelengths longer than `original_context_length /
/// low_freq_factor` are divided by `factor`, and the band in between is
/// interpolated smoothly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeBandedFrequencyScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_context_length: u64,
}

impl RopeBandedFrequencyScaling {
    /// The `frequency_scaling` attribute value naming this scheme.
    pub const ATTRIBUTE_VALUE: &'static str = "banded";

    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("banded RoPE {name} must be finite and positive"))
            }
        };
        positive("factor", self.factor)?;
        positive("low_freq_factor", self.low_freq_factor)?;
        positive("high_freq_factor", self.high_freq_factor)?;
        if self.high_freq_factor <= self.low_freq_factor {
            return Err("banded RoPE high_freq_factor must exceed low_freq_factor".into());
        }
        if self.original_context_length == 0 {
            return Err("banded RoPE original_max_position_embeddings must be positive".into());
        }
        Ok(())
    }

    /// The scaled inverse frequency of one rotated pair, in `f32` exactly as
    /// every Kernel computes it so Providers agree bit for bit on the
    /// frequency table.
    pub fn scale_frequency(&self, frequency: f32) -> f32 {
        let factor = self.factor as f32;
        let low = self.low_freq_factor as f32;
        let high = self.high_freq_factor as f32;
        let original = self.original_context_length as f32;
        let wavelength = std::f32::consts::TAU / frequency;
        if wavelength < original / high {
            frequency
        } else if wavelength > original / low {
            frequency / factor
        } else {
            let smooth = (original / wavelength - low) / (high - low);
            (1.0 - smooth) * frequency / factor + smooth * frequency
        }
    }

    /// Writes this scaling as `rope` attributes.
    pub fn insert_attributes(&self, attributes: &mut BTreeMap<String, OperatorAttributeValue>) {
        attributes.insert(
            "frequency_scaling".into(),
            OperatorAttributeValue::String(Self::ATTRIBUTE_VALUE.into()),
        );
        attributes.insert(
            "frequency_scaling_factor".into(),
            OperatorAttributeValue::Float(self.factor),
        );
        attributes.insert(
            "low_frequency_factor".into(),
            OperatorAttributeValue::Float(self.low_freq_factor),
        );
        attributes.insert(
            "high_frequency_factor".into(),
            OperatorAttributeValue::Float(self.high_freq_factor),
        );
        attributes.insert(
            "original_context_length".into(),
            OperatorAttributeValue::Integer(self.original_context_length as i64),
        );
    }

    /// Reads the scaling a `rope` invocation carries: `None` when it has no
    /// `frequency_scaling` attribute, an error for an unknown scheme or a
    /// missing or invalid parameter (never a silent fallback to unscaled
    /// frequencies).
    pub fn from_attributes(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<Option<Self>, String> {
        match attributes.get("frequency_scaling") {
            None => return Ok(None),
            Some(OperatorAttributeValue::String(value)) if value == Self::ATTRIBUTE_VALUE => {}
            Some(OperatorAttributeValue::String(value)) => {
                return Err(format!("frequency_scaling '{value}' is not implemented"));
            }
            Some(_) => return Err("frequency_scaling must be a string".into()),
        }
        let float = |name: &str| match attributes.get(name) {
            Some(OperatorAttributeValue::Float(value)) => Ok(*value),
            _ => Err(format!(
                "banded frequency scaling requires a float '{name}'"
            )),
        };
        let original_context_length = match attributes.get("original_context_length") {
            Some(OperatorAttributeValue::Integer(value)) if *value > 0 => *value as u64,
            _ => {
                return Err(
                    "banded frequency scaling requires a positive integer 'original_context_length'"
                        .into(),
                );
            }
        };
        let scaling = Self {
            factor: float("frequency_scaling_factor")?,
            low_freq_factor: float("low_frequency_factor")?,
            high_freq_factor: float("high_frequency_factor")?,
            original_context_length,
        };
        scaling.validate()?;
        Ok(Some(scaling))
    }
}

/// YaRN RoPE frequency scaling (`express-latent-attention-forms`): a
/// model-family-neutral scheme the loaders map a checkpoint's
/// `rope_scaling.type = "yarn"` onto. Each rotated pair `i` of a
/// `dimension`-wide rotation blends its frequency `f` with `f / factor`:
/// pairs below the correction range keep `f`, pairs above it take
/// `f / factor`, linearly in between. The correction range comes from the
/// numbers of rotations `beta_fast` and `beta_slow` complete over
/// `original_context_length` positions. Both `cos` and `sin` are multiplied
/// by `attention_factor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeYarnFrequencyScaling {
    pub factor: f64,
    pub original_context_length: u64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: f64,
    /// Whether the correction range is widened to whole pairs.
    pub truncate: bool,
}

impl RopeYarnFrequencyScaling {
    /// The `frequency_scaling` attribute value naming this scheme.
    pub const ATTRIBUTE_VALUE: &'static str = "yarn";

    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("YaRN RoPE {name} must be finite and positive"))
            }
        };
        positive("factor", self.factor)?;
        positive("beta_fast", self.beta_fast)?;
        positive("beta_slow", self.beta_slow)?;
        positive("attention_factor", self.attention_factor)?;
        if self.original_context_length == 0 {
            return Err("YaRN RoPE original_context_length must be positive".into());
        }
        Ok(())
    }

    /// The pairs `[low, high]` over which a `dimension`-wide rotation with
    /// RoPE base `base` blends the two frequencies, computed in `f64` and
    /// returned as the `f32` bounds every Kernel uses.
    pub fn correction_range(&self, dimension: u64, base: f32) -> (f32, f32) {
        let dimension = dimension as f64;
        let base = base as f64;
        let original = self.original_context_length as f64;
        let pair_for = |rotations: f64| {
            dimension * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let (mut low, mut high) = (pair_for(self.beta_fast), pair_for(self.beta_slow));
        if self.truncate {
            low = low.floor();
            high = high.ceil();
        }
        (low.max(0.0) as f32, high.min(dimension - 1.0) as f32)
    }

    /// The scaled inverse frequency of rotated pair `pair`, whose unscaled
    /// frequency is `frequency`, given [`Self::correction_range`].
    pub fn scale_frequency(&self, frequency: f32, pair: usize, (low, high): (f32, f32)) -> f32 {
        let high = if low == high { high + 0.001 } else { high };
        let ramp = ((pair as f32 - low) / (high - low)).clamp(0.0, 1.0);
        let interpolated = frequency / self.factor as f32;
        interpolated * ramp + frequency * (1.0 - ramp)
    }

    /// Writes this scaling as `rope` attributes.
    pub fn insert_attributes(&self, attributes: &mut BTreeMap<String, OperatorAttributeValue>) {
        attributes.insert(
            "frequency_scaling".into(),
            OperatorAttributeValue::String(Self::ATTRIBUTE_VALUE.into()),
        );
        attributes.insert(
            "frequency_scaling_factor".into(),
            OperatorAttributeValue::Float(self.factor),
        );
        attributes.insert(
            "original_context_length".into(),
            OperatorAttributeValue::Integer(self.original_context_length as i64),
        );
        attributes.insert(
            "beta_fast".into(),
            OperatorAttributeValue::Float(self.beta_fast),
        );
        attributes.insert(
            "beta_slow".into(),
            OperatorAttributeValue::Float(self.beta_slow),
        );
        attributes.insert(
            "attention_factor".into(),
            OperatorAttributeValue::Float(self.attention_factor),
        );
        attributes.insert(
            "truncate".into(),
            OperatorAttributeValue::Boolean(self.truncate),
        );
    }

    /// Reads the YaRN scaling a `rope` invocation carries: `None` when its
    /// `frequency_scaling` is absent or names another scheme, an error for a
    /// missing or invalid parameter. Every parameter is required: a Kernel
    /// never assumes one.
    pub fn from_attributes(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<Option<Self>, String> {
        match attributes.get("frequency_scaling") {
            Some(OperatorAttributeValue::String(value)) if value == Self::ATTRIBUTE_VALUE => {}
            _ => return Ok(None),
        }
        let float = |name: &str| match attributes.get(name) {
            Some(OperatorAttributeValue::Float(value)) => Ok(*value),
            _ => Err(format!("YaRN frequency scaling requires a float '{name}'")),
        };
        let original_context_length = match attributes.get("original_context_length") {
            Some(OperatorAttributeValue::Integer(value)) if *value > 0 => *value as u64,
            _ => {
                return Err(
                    "YaRN frequency scaling requires a positive integer 'original_context_length'"
                        .into(),
                );
            }
        };
        let truncate = match attributes.get("truncate") {
            Some(OperatorAttributeValue::Boolean(value)) => *value,
            _ => return Err("YaRN frequency scaling requires a boolean 'truncate'".into()),
        };
        let scaling = Self {
            factor: float("frequency_scaling_factor")?,
            original_context_length,
            beta_fast: float("beta_fast")?,
            beta_slow: float("beta_slow")?,
            attention_factor: float("attention_factor")?,
            truncate,
        };
        scaling.validate()?;
        Ok(Some(scaling))
    }
}

/// How `concat` joins its inputs (`axis`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConcatAxis {
    /// Row after row (absent `axis`): the KV-history concatenation.
    Rows,
    /// Head by head (`axis = "heads"`, `express-latent-attention-forms`):
    /// `[T, H * a]` and `[T, H * b]` (or `[T, b]`, one head given to every
    /// head) give `[T, H * (a + b)]`, each output head being the first
    /// input's head followed by the second's.
    Heads,
}

/// A `concat` node's axis; `None` for an unknown `axis` value.
pub fn concat_axis(attributes: &BTreeMap<String, OperatorAttributeValue>) -> Option<ConcatAxis> {
    match attributes.get("axis") {
        None => Some(ConcatAxis::Rows),
        Some(OperatorAttributeValue::String(value)) if value == "rows" => Some(ConcatAxis::Rows),
        Some(OperatorAttributeValue::String(value)) if value == "heads" => Some(ConcatAxis::Heads),
        Some(_) => None,
    }
}

/// The per-head widths `(a, b)` of a head-wise `concat` of `[T, first]` and
/// `[T, second]` rows into `head_count` heads, the second input holding
/// `second_head_count` heads (`1`: broadcast to every head, or
/// `head_count`).
pub fn concat_heads_widths(
    first: u64,
    second: u64,
    head_count: u64,
    second_head_count: u64,
) -> Result<(u64, u64), String> {
    if head_count == 0 || !first.is_multiple_of(head_count) {
        return Err(format!(
            "concat across {head_count} heads needs a first input width divisible by them, got {first}"
        ));
    }
    if second_head_count != 1 && second_head_count != head_count {
        return Err(format!(
            "concat's second input holds 1 or {head_count} heads, not {second_head_count}"
        ));
    }
    if !second.is_multiple_of(second_head_count) {
        return Err(format!(
            "concat's second input width {second} is not divisible by its {second_head_count} heads"
        ));
    }
    Ok((first / head_count, second / second_head_count))
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorAttributeKind {
    Boolean,
    Integer,
    Float,
    String,
    DType,
    Layout,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OperatorAttributeValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    DType(ComputeDType),
    Layout(TensorLayoutKind),
}

impl OperatorAttributeValue {
    pub const fn kind(&self) -> OperatorAttributeKind {
        match self {
            Self::Boolean(_) => OperatorAttributeKind::Boolean,
            Self::Integer(_) => OperatorAttributeKind::Integer,
            Self::Float(_) => OperatorAttributeKind::Float,
            Self::String(_) => OperatorAttributeKind::String,
            Self::DType(_) => OperatorAttributeKind::DType,
            Self::Layout(_) => OperatorAttributeKind::Layout,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorAttributeRule {
    pub kind: OperatorAttributeKind,
    pub required: bool,
}

impl OperatorAttributeRule {
    pub const fn required(kind: OperatorAttributeKind) -> Self {
        Self {
            kind,
            required: true,
        }
    }

    pub const fn optional(kind: OperatorAttributeKind) -> Self {
        Self {
            kind,
            required: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OperatorAttributeSchema {
    pub rules: BTreeMap<String, OperatorAttributeRule>,
}

impl OperatorAttributeSchema {
    pub fn with_rule(mut self, name: impl Into<String>, rule: OperatorAttributeRule) -> Self {
        self.rules.insert(name.into(), rule);
        self
    }

    pub fn validate(
        &self,
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), OperatorError> {
        for forbidden in ["provider", "provider_id", "device", "device_id", "kernel"] {
            if attributes.contains_key(forbidden) {
                return Err(OperatorError::OperatorAttributeInvalid {
                    attribute: forbidden.into(),
                    reason: "operator attributes must not select Provider, Device, or kernel"
                        .into(),
                });
            }
        }
        for (name, rule) in &self.rules {
            match attributes.get(name) {
                Some(value) if value.kind() == rule.kind => {}
                Some(_) => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: name.clone(),
                        reason: "attribute kind mismatch".into(),
                    });
                }
                None if rule.required => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: name.clone(),
                        reason: "required attribute missing".into(),
                    });
                }
                None => {}
            }
        }
        if let Some(value) = attributes.get(TENSOR_PARALLEL_ATTRIBUTE) {
            match value {
                OperatorAttributeValue::String(kind)
                    if TENSOR_PARALLEL_KINDS.contains(&kind.as_str()) => {}
                _ => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: TENSOR_PARALLEL_ATTRIBUTE.into(),
                        reason: format!("must be one of {TENSOR_PARALLEL_KINDS:?}"),
                    });
                }
            }
        }
        for name in attributes.keys() {
            if name == TENSOR_PARALLEL_ATTRIBUTE {
                continue;
            }
            if !self.rules.contains_key(name) {
                return Err(OperatorError::OperatorAttributeInvalid {
                    attribute: name.clone(),
                    reason: "attribute is not defined by schema".into(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorSpec {
    pub id: OperatorId,
    /// The most inputs an invocation takes.
    pub inputs: usize,
    /// The fewest inputs an invocation takes: the inputs past it are
    /// optional and trailing (`extend-portable-operator-attributes`).
    pub min_inputs: usize,
    pub outputs: usize,
    pub attributes: OperatorAttributeSchema,
    pub shape_rule: ShapeRule,
    pub dtype_contracts: Vec<OperatorDTypeContract>,
    pub layout_contract: OperatorLayoutContract,
    pub memory: OperatorMemoryBehavior,
    pub determinism: OperatorDeterminism,
}

impl OperatorSpec {
    pub fn new(id: OperatorId, inputs: usize, outputs: usize) -> Self {
        Self {
            id,
            inputs,
            min_inputs: inputs,
            outputs,
            attributes: OperatorAttributeSchema::default(),
            shape_rule: ShapeRule::Any,
            dtype_contracts: Vec::new(),
            layout_contract: OperatorLayoutContract::new([TensorLayoutKind::Contiguous]),
            memory: OperatorMemoryBehavior::pure(),
            determinism: OperatorDeterminism::default(),
        }
    }

    /// Accepts as few as `min_inputs` inputs, the rest trailing and optional.
    pub fn with_min_inputs(mut self, min_inputs: usize) -> Self {
        self.min_inputs = min_inputs.min(self.inputs);
        self
    }

    pub fn with_attributes(mut self, attributes: OperatorAttributeSchema) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn with_shape_rule(mut self, shape_rule: ShapeRule) -> Self {
        self.shape_rule = shape_rule;
        self
    }

    pub fn with_dtype_contract(mut self, contract: OperatorDTypeContract) -> Self {
        self.dtype_contracts.push(contract);
        self
    }

    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = TensorLayoutKind>) -> Self {
        self.layout_contract = OperatorLayoutContract::new(layouts);
        self
    }

    pub const fn with_memory(mut self, memory: OperatorMemoryBehavior) -> Self {
        self.memory = memory;
        self
    }

    pub fn validate_invocation(
        &self,
        inputs: &[TensorDescriptor],
        outputs: &[TensorDescriptor],
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), OperatorError> {
        self.attributes.validate(attributes)?;
        if inputs.len() < self.min_inputs || inputs.len() > self.inputs {
            return Err(OperatorError::InputArityInvalid {
                expected: if inputs.len() < self.min_inputs {
                    self.min_inputs
                } else {
                    self.inputs
                },
                actual: inputs.len(),
            });
        }
        if outputs.len() != self.outputs {
            return Err(OperatorError::OutputArityInvalid {
                expected: self.outputs,
                actual: outputs.len(),
            });
        }
        validate_shape_rule(&self.shape_rule, inputs, outputs, attributes)?;
        for tensor in inputs.iter().chain(outputs) {
            validate_dtype_contracts(&self.dtype_contracts, tensor)?;
            let layout = layout_kind(&tensor.layout);
            if !layout.component_visible() || !self.layout_contract.accepts(layout) {
                return Err(OperatorError::LayoutUnsupported { layout });
            }
        }
        validate_memory_behavior(&self.memory, inputs, outputs)?;
        validate_aliasing_intent(&self.memory, inputs, outputs)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorCatalog {
    pub version: CapabilityVersion,
    pub operators: BTreeMap<OperatorId, OperatorSpec>,
}

impl OperatorCatalog {
    pub fn new(version: CapabilityVersion) -> Self {
        Self {
            version,
            operators: BTreeMap::new(),
        }
    }

    pub fn with_operator(mut self, spec: OperatorSpec) -> Self {
        self.operators.insert(spec.id.clone(), spec);
        self
    }

    pub fn get(&self, id: &OperatorId) -> Result<&OperatorSpec, OperatorError> {
        self.operators.get(id).ok_or_else(|| {
            if self
                .operators
                .keys()
                .any(|known| known.namespace == id.namespace && known.name == id.name)
            {
                OperatorError::OperatorVersionUnsupported {
                    operator: id.clone(),
                }
            } else {
                OperatorError::OperatorNotFound {
                    operator: id.clone(),
                }
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorObservationKind {
    OperatorPlanned,
    DataMovementInserted,
    DTypeConversionInserted,
    LayoutConversionInserted,
    WorkspaceRequested,
    OperatorExecutionStarted,
    OperatorExecutionCompleted,
    OperatorExecutionFailed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorObservation {
    pub kind: OperatorObservationKind,
    pub operator: OperatorId,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl OperatorObservation {
    pub fn new(kind: OperatorObservationKind, operator: OperatorId) -> Self {
        Self {
            kind,
            operator,
            redacted_metadata: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperatorError {
    OperatorNotFound {
        operator: OperatorId,
    },
    OperatorVersionUnsupported {
        operator: OperatorId,
    },
    OperatorAttributeInvalid {
        attribute: String,
        reason: String,
    },
    InputArityInvalid {
        expected: usize,
        actual: usize,
    },
    OutputArityInvalid {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch {
        reason: String,
    },
    ShapeUnsupported {
        reason: String,
    },
    DTypeUnsupported {
        dtype: ComputeDType,
    },
    DTypeConversionRequired {
        from: ComputeDType,
        to: ComputeDType,
    },
    DTypeConversionUnsupported {
        from: ComputeDType,
        to: ComputeDType,
    },
    LayoutUnsupported {
        layout: TensorLayoutKind,
    },
    LayoutConversionRequired {
        from: TensorLayoutKind,
        to: TensorLayoutKind,
    },
    LayoutConversionUnsupported {
        from: TensorLayoutKind,
        to: TensorLayoutKind,
    },
    MemoryBehaviorUnsupported {
        reason: String,
    },
    AliasingUnsupported {
        reason: String,
    },
    WorkspaceUnavailable {
        bytes: u64,
    },
    ResourceAffinityConflict {
        reason: String,
    },
    ProviderCapabilityUnavailable {
        capability: String,
    },
    KernelUnavailable {
        operator: OperatorId,
    },
    GraphValidationFailed {
        reason: String,
    },
    GraphPlanningFailed {
        reason: String,
    },
    GraphExecutionFailed {
        reason: String,
    },
    BrowserFeatureUnsupported {
        feature: String,
    },
    InternalOperator {
        reason: String,
    },
}

impl fmt::Display for OperatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperatorNotFound { operator } => write!(f, "operator not found: {operator}"),
            Self::OperatorVersionUnsupported { operator } => {
                write!(f, "operator version unsupported: {operator}")
            }
            Self::OperatorAttributeInvalid { attribute, reason } => {
                write!(f, "operator attribute invalid: {attribute}: {reason}")
            }
            Self::InputArityInvalid { expected, actual } => {
                write!(f, "input arity invalid: expected {expected}, got {actual}")
            }
            Self::OutputArityInvalid { expected, actual } => {
                write!(f, "output arity invalid: expected {expected}, got {actual}")
            }
            Self::ShapeMismatch { reason } => write!(f, "shape mismatch: {reason}"),
            Self::ShapeUnsupported { reason } => write!(f, "shape unsupported: {reason}"),
            Self::DTypeUnsupported { dtype } => write!(f, "dtype unsupported: {dtype:?}"),
            Self::DTypeConversionRequired { from, to } => {
                write!(f, "dtype conversion required: {from:?} to {to:?}")
            }
            Self::DTypeConversionUnsupported { from, to } => {
                write!(f, "dtype conversion unsupported: {from:?} to {to:?}")
            }
            Self::LayoutUnsupported { layout } => write!(f, "layout unsupported: {layout:?}"),
            Self::LayoutConversionRequired { from, to } => {
                write!(f, "layout conversion required: {from:?} to {to:?}")
            }
            Self::LayoutConversionUnsupported { from, to } => {
                write!(f, "layout conversion unsupported: {from:?} to {to:?}")
            }
            Self::MemoryBehaviorUnsupported { reason } => {
                write!(f, "memory behavior unsupported: {reason}")
            }
            Self::AliasingUnsupported { reason } => write!(f, "aliasing unsupported: {reason}"),
            Self::WorkspaceUnavailable { bytes } => write!(f, "workspace unavailable: {bytes}"),
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "resource affinity conflict: {reason}")
            }
            Self::ProviderCapabilityUnavailable { capability } => {
                write!(f, "provider capability unavailable: {capability}")
            }
            Self::KernelUnavailable { operator } => write!(f, "kernel unavailable: {operator}"),
            Self::GraphValidationFailed { reason } => {
                write!(f, "graph validation failed: {reason}")
            }
            Self::GraphPlanningFailed { reason } => write!(f, "graph planning failed: {reason}"),
            Self::GraphExecutionFailed { reason } => write!(f, "graph execution failed: {reason}"),
            Self::BrowserFeatureUnsupported { feature } => {
                write!(f, "browser feature unsupported: {feature}")
            }
            Self::InternalOperator { reason } => write!(f, "internal operator error: {reason}"),
        }
    }
}

impl Error for OperatorError {}

pub fn layout_kind(layout: &LayoutDescriptor) -> TensorLayoutKind {
    match layout {
        LayoutDescriptor::Contiguous => TensorLayoutKind::Contiguous,
        LayoutDescriptor::Strided { .. } => TensorLayoutKind::Strided,
        LayoutDescriptor::Blocked { .. } => TensorLayoutKind::Blocked,
        LayoutDescriptor::Paged { .. } => TensorLayoutKind::Paged,
        LayoutDescriptor::PackedQuantized { .. } => TensorLayoutKind::QuantizedPacked,
        LayoutDescriptor::AttentionSpecific { .. } => TensorLayoutKind::AttentionSpecific,
        LayoutDescriptor::BrowserCompatible { .. } => TensorLayoutKind::BrowserCompatible,
        LayoutDescriptor::ProviderOpaque { .. } => TensorLayoutKind::ProviderOpaque,
    }
}

pub fn validate_affinity_compatibility(
    a: &ResourceAffinity,
    b: &ResourceAffinity,
) -> Result<(), OperatorError> {
    a.validate_with(b)
        .map_err(|error| OperatorError::ResourceAffinityConflict {
            reason: error.to_string(),
        })
}

/// [`initial_operator_catalog`], built once per process. The catalog is a
/// constant, so hot paths look Operators up here instead of rebuilding it
/// for every dispatch.
pub fn shared_initial_operator_catalog() -> &'static OperatorCatalog {
    static CATALOG: std::sync::OnceLock<OperatorCatalog> = std::sync::OnceLock::new();
    CATALOG.get_or_init(initial_operator_catalog)
}

pub fn initial_operator_catalog() -> OperatorCatalog {
    let floats = [
        ComputeDType::Float16,
        ComputeDType::BrainFloat16,
        ComputeDType::Float32,
        ComputeDType::Float64,
    ];
    let numeric = [
        ComputeDType::UInt8,
        ComputeDType::SInt8,
        ComputeDType::UInt16,
        ComputeDType::SInt16,
        ComputeDType::UInt32,
        ComputeDType::SInt32,
        ComputeDType::UInt64,
        ComputeDType::SInt64,
        ComputeDType::Float16,
        ComputeDType::BrainFloat16,
        ComputeDType::Float32,
        ComputeDType::Float64,
    ];
    let mut catalog = OperatorCatalog::new(OPERATOR_CATALOG_VERSION);
    for (name, family, inputs, outputs, shape) in [
        (
            "matmul",
            OperatorFamily::LinearAlgebra,
            2,
            1,
            ShapeRule::Matmul,
        ),
        (
            "batched-matmul",
            OperatorFamily::LinearAlgebra,
            2,
            1,
            ShapeRule::Any,
        ),
        // `cache-and-optimize-prepared-graphs` C03: `matmul(a, b) + bias`
        // with `bias` one row broadcast to every output row, produced only
        // by `graph_fusion::fuse_matmul_bias` from an existing matmul/add
        // pair, never by a Component.
        (
            "matmul-bias",
            OperatorFamily::LinearAlgebra,
            3,
            1,
            ShapeRule::Any,
        ),
        // C03: `residual + matmul(a, b)` and `silu(gate) * up`, likewise
        // produced only by `graph_fusion`.
        (
            "matmul-residual",
            OperatorFamily::LinearAlgebra,
            3,
            1,
            ShapeRule::Any,
        ),
        ("silu-mul", OperatorFamily::Activation, 2, 1, ShapeRule::Any),
        // `silu(matmul(a, gate)) * matmul(a, up)`, produced only by
        // `graph_fusion::fuse_gated_matmul`.
        (
            "gated-matmul",
            OperatorFamily::LinearAlgebra,
            3,
            1,
            ShapeRule::Any,
        ),
        // `execute-and-cache-moe-experts` F02: routed mixture of experts.
        // Inputs `h [T, H]`, router logits `[T, E]`, stacked expert
        // weights `gate [H, E*I]`, `up [H, E*I]`, `down [I, E*H]`; output
        // `[T, H]`. The sigmoid form adds a `[E]` selection-only bias, then
        // an optional `[E]` output scale. Its `routing_scoring` is
        // `sigmoid`, `routing_normalization` is `never`, `multiple`, or
        // `always`, and `routing_scale` is positive. Optional paired group
        // attributes rank groups by the sum of the top two biased scores.
        // A tie across a group or expert selection boundary is refused.
        // `activation = "relu_squared"` selects an ungated sigmoid form:
        // `h, logits, in, down, bias`, optionally followed by expert scale.
        (
            "moe-experts",
            OperatorFamily::LinearAlgebra,
            7,
            1,
            ShapeRule::Any,
        ),
        // One sequence's Mamba-2 activation and explicit next conv/SSM states.
        (
            "mamba2-mixer",
            OperatorFamily::LinearAlgebra,
            13,
            3,
            ShapeRule::Any,
        ),
        // `out[t, :] = x[t, :] * s[t]` for `s` of shape `[T, 1]`.
        ("row-scale", OperatorFamily::Tensor, 2, 1, ShapeRule::Any),
        ("embedding", OperatorFamily::Tensor, 2, 1, ShapeRule::Any),
        (
            "rmsnorm",
            OperatorFamily::Normalization,
            2,
            1,
            ShapeRule::RmsNorm,
        ),
        (
            "layernorm",
            OperatorFamily::Normalization,
            2,
            1,
            ShapeRule::SameShape,
        ),
        (
            "rope",
            OperatorFamily::PositionEncoding,
            1,
            1,
            ShapeRule::SameShape,
        ),
        // Q, K and V; with `latent_kv_width`, the latent K and V's key and
        // value projections (`chunk-latent-prefills`).
        ("attention", OperatorFamily::Attention, 5, 1, ShapeRule::Any),
        // Single-sequence bidirectional attention with a binary key mask and
        // an optional inclusive symmetric local radius.
        (
            "encoder-attention",
            OperatorFamily::Attention,
            4,
            1,
            ShapeRule::Any,
        ),
        (
            "paged-attention",
            OperatorFamily::Attention,
            3,
            1,
            ShapeRule::Any,
        ),
        (
            "softmax",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "activation",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "gelu",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "silu",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "add",
            OperatorFamily::Tensor,
            2,
            1,
            ShapeRule::RowBroadcastAdd,
        ),
        (
            "mul",
            OperatorFamily::Tensor,
            2,
            1,
            ShapeRule::SameShapeOrScalar,
        ),
        ("concat", OperatorFamily::Tensor, 2, 1, ShapeRule::RowConcat),
        // `append-kv-history-in-place`: writes `rows` into `store` at
        // `row_offset`, in place; the output resource is the store itself.
        ("kv-write", OperatorFamily::Tensor, 2, 1, ShapeRule::Any),
        (
            "residual-add",
            OperatorFamily::Tensor,
            2,
            1,
            ShapeRule::SameShape,
        ),
        // A row-wise, learned geodesic residual update. Scale and bias are
        // tensor inputs so each mixer/MLP keeps its own checkpoint weights.
        (
            "geodesic-residual",
            OperatorFamily::Tensor,
            4,
            1,
            ShapeRule::GeodesicResidual,
        ),
        (
            "dtype-conversion",
            OperatorFamily::Tensor,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "layout-conversion",
            OperatorFamily::Layout,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "quantize",
            OperatorFamily::Quantization,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "dequantize",
            OperatorFamily::Quantization,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "sampling-helper",
            OperatorFamily::SamplingSupport,
            1,
            1,
            ShapeRule::Any,
        ),
        // `reach-architecture-freeze-1` task 5.4/5.5: the first portable
        // Operator with more than one declared output, proving the
        // Resource-based generic execution path's already-existing
        // multi-output support (`KernelInvocation.outputs: Vec<..>`,
        // `store_output(invocation, index, ..)`) end to end, not just at
        // the type level. Splits a rank-2 input's last dimension in half
        // into two equally-shaped outputs; `ShapeRule::Any` because no
        // existing rule expresses "half the last dimension of the input,"
        // and inventing one for a single deliberately-minimal proof
        // operator is not warranted.
        ("split", OperatorFamily::Tensor, 1, 2, ShapeRule::Any),
    ] {
        let mut spec = OperatorSpec::new(OperatorId::magnetar(name, 1, family), inputs, outputs)
            .with_shape_rule(shape)
            .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Input, numeric))
            .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Output, numeric))
            .with_layouts([
                TensorLayoutKind::Contiguous,
                TensorLayoutKind::Strided,
                TensorLayoutKind::Blocked,
                TensorLayoutKind::Paged,
                TensorLayoutKind::QuantizedPacked,
                TensorLayoutKind::AttentionSpecific,
                TensorLayoutKind::BrowserCompatible,
            ]);
        spec.attributes = default_attribute_schema(name);
        // Optional trailing inputs: a scale-less RMSNorm, the experts'
        // optional selection bias and per-expert output scale, and latent K
        // and V's projections.
        spec.min_inputs = match name {
            "rmsnorm" => 1,
            "moe-experts" => 5,
            "mamba2-mixer" => 10,
            "attention" => 3,
            _ => inputs,
        };
        if name == "kv-write" {
            spec.memory = OperatorMemoryBehavior {
                mutates_input: true,
                aliases_output: true,
                supports_in_place: true,
                ..OperatorMemoryBehavior::pure()
            };
        }
        if matches!(family, OperatorFamily::Attention) {
            spec.memory = OperatorMemoryBehavior::pure()
                .with_workspace()
                .with_paged_kv_cache();
        }
        if name == "encoder-attention" {
            spec.memory = OperatorMemoryBehavior::pure().with_workspace();
        }
        if matches!(name, "encoder-attention" | "mamba2-mixer") {
            spec.dtype_contracts = vec![
                OperatorDTypeContract::new(TensorRole::Input, [ComputeDType::Float32]),
                OperatorDTypeContract::new(TensorRole::Output, [ComputeDType::Float32]),
                OperatorDTypeContract::new(TensorRole::Accumulation, [ComputeDType::Float32]),
            ];
        }
        if matches!(
            family,
            OperatorFamily::Normalization | OperatorFamily::Activation
        ) {
            spec.dtype_contracts = vec![
                OperatorDTypeContract::new(TensorRole::Input, floats),
                OperatorDTypeContract::new(TensorRole::Output, floats),
                OperatorDTypeContract::new(TensorRole::Accumulation, floats),
            ];
        }
        if name == "geodesic-residual" {
            // Only the f32 profile is qualified; other output dtypes need
            // independently validated rounding and conversion semantics.
            spec.dtype_contracts = vec![
                OperatorDTypeContract::new(TensorRole::Input, [ComputeDType::Float32]),
                OperatorDTypeContract::new(TensorRole::Output, [ComputeDType::Float32]),
                OperatorDTypeContract::new(TensorRole::Accumulation, [ComputeDType::Float32]),
            ];
        }
        catalog = catalog.with_operator(spec);
    }
    catalog
}

fn default_attribute_schema(name: &str) -> OperatorAttributeSchema {
    match name {
        "mamba2-mixer" => OperatorAttributeSchema::default()
            .with_rule(
                "heads",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_dim",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "groups",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "state_dim",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "conv_kernel",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "epsilon",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "dt_min",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "dt_max",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "conv_bias",
                OperatorAttributeRule::required(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "linear_bias",
                OperatorAttributeRule::required(OperatorAttributeKind::Boolean),
            ),
        "encoder-attention" => OperatorAttributeSchema::default()
            .with_rule(
                "head_count",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_dimension",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "local_radius",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            ),
        "geodesic-residual" => OperatorAttributeSchema::default().with_rule(
            "layer_index",
            OperatorAttributeRule::required(OperatorAttributeKind::Integer),
        ),
        // `cache-latent-attention-history`: `a [T, G*K]` times `b` group by
        // group (`group_count = G`), each group's block `[K, N]` or, with
        // `transpose_b`, `[N, K]` used transposed, giving `[T, G*N]`
        // (`magnetar_provider_api::batched_matmul`).
        "batched-matmul" => OperatorAttributeSchema::default()
            .with_rule(
                "group_count",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "transpose_b",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            ),
        "matmul" | "matmul-bias" | "matmul-residual" | "gated-matmul" => {
            OperatorAttributeSchema::default()
                .with_rule(
                    "transpose_a",
                    OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
                )
                .with_rule(
                    "transpose_b",
                    OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
                )
                .with_rule(
                    "accumulation_dtype",
                    OperatorAttributeRule::optional(OperatorAttributeKind::DType),
                )
        }
        "attention" | "paged-attention" => OperatorAttributeSchema::default()
            .with_rule(
                "causal",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "window_size",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_count",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "kv_head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_dimension",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "attention_mask_kind",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            // `append-kv-history-in-place`: attend to the first `key_count`
            // rows of K and V only (absent: every row). Set by the Runtime
            // only for Kernels advertising the `key-count` feature.
            .with_rule(
                "key_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // The factor applied to `QK^T` instead of `1/sqrt(head_dimension)`.
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            // Width of each head's V rows and output, when it differs from
            // the Q/K `head_dimension` (`express-latent-attention-forms`).
            .with_rule(
                "value_head_dimension",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // Width of each KV head in V's rows, of which the value is the
            // leading `value_head_dimension` (`cache-latent-attention-history`).
            .with_rule(
                "value_head_stride",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // K and V are latent rows of this many values and one shared
            // rope key, expanded per head by the fourth and fifth inputs
            // (`chunk-latent-prefills`).
            .with_rule(
                "latent_kv_width",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            ),
        // `axis = "heads"` joins the inputs head by head (see [`ConcatAxis`]):
        // `head_count` heads, the second input holding `second_head_count`
        // of them (absent: `head_count`; `1`: broadcast).
        "concat" => OperatorAttributeSchema::default()
            .with_rule(
                "axis",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            .with_rule(
                "head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "second_head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            ),
        "kv-write" => OperatorAttributeSchema::default().with_rule(
            "row_offset",
            OperatorAttributeRule::required(OperatorAttributeKind::Integer),
        ),
        "rope" => OperatorAttributeSchema::default()
            .with_rule(
                "base",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "dimension",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "position_mode",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            // Which elements of a head block form each rotated pair
            // (`fix-rope-rotation-pairing`): `interleaved` (2i, 2i+1, the
            // default when absent) or `half_split` (i, i + dimension/2,
            // Hugging Face `rotate_half`). Set by the Model Component;
            // Providers reject any other value.
            .with_rule(
                "rotation_pairing",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            // Absolute position of the first row. Absent means the sequence
            // starts at zero, which is only true for a prefill; a decode step
            // against a populated cache carries the number of tokens already
            // there, without which every generated token would be rotated as
            // if it were the first.
            .with_rule(
                "position_offset",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // Number of equal-width head blocks this row's columns divide
            // into, each rotated independently (`make-first-native-cuda-
            // hot-path-device-resident`). Absent or `1` reproduces the
            // original single-block behavior. Set explicitly by whichever
            // component constructs the graph (the number of heads is
            // already known there -- Runtime must not infer it from a node
            // identifier or other naming convention).
            .with_rule(
                "head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // banded frequency scaling ([`RopeBandedFrequencyScaling`]); absent means
            // unscaled frequencies. Sent only to Kernels advertising
            // [`ROPE_BANDED_FREQUENCY_SCALING_FEATURE`].
            .with_rule(
                "frequency_scaling",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            .with_rule(
                "frequency_scaling_factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "low_frequency_factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "high_frequency_factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "original_context_length",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // Only `floor(factor * dimension / 2)` pairs rotate; the rest
            // keep frequency 0. In `(0, 1]`; absent: every pair rotates.
            .with_rule(
                "partial_rotary_factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            // `frequency_scaling = "yarn"` ([`RopeYarnFrequencyScaling`]).
            .with_rule(
                "beta_fast",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "beta_slow",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "attention_factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "truncate",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            ),
        "rmsnorm" | "layernorm" => OperatorAttributeSchema::default()
            .with_rule(
                "epsilon",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "normalized_dimension",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "accumulation_dtype",
                OperatorAttributeRule::optional(OperatorAttributeKind::DType),
            )
            // Per-head normalization: `head_count` blocks of
            // `head_dimension`, each normalized alone, one shared weight.
            .with_rule(
                "head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_dimension",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // Whether the node has a weight input: `false` with one input
            // (scale 1), `true` with two. The input count is authoritative;
            // a Kernel refuses a node whose `with_scale` contradicts it.
            .with_rule(
                "with_scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            ),
        "moe-experts" => OperatorAttributeSchema::default()
            .with_rule(
                "experts",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "top_k",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "normalize_top_k",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "routing_scoring",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            .with_rule(
                "routing_normalization",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            )
            .with_rule(
                "routing_scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "routing_group_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "routing_group_selected",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "expert_first",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "expert_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            // Gate activation `silu`/`gelu_tanh`, or ungated `relu_squared`.
            .with_rule(
                "activation",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            ),
        "activation" => OperatorAttributeSchema::default()
            .with_rule(
                "kind",
                OperatorAttributeRule::required(OperatorAttributeKind::String),
            )
            // `kind = "softcap"`: `cap * tanh(x / cap)`.
            .with_rule(
                "cap",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            // `kind = "scale"`: `x * factor`.
            .with_rule(
                "factor",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            ),
        // The portable `gelu` is the tanh approximation; `approximate`
        // states it (`"tanh"`), any other value is refused.
        "gelu" => OperatorAttributeSchema::default().with_rule(
            "approximate",
            OperatorAttributeRule::optional(OperatorAttributeKind::String),
        ),
        // The gathered rows multiplied by `scale`.
        "embedding" => OperatorAttributeSchema::default().with_rule(
            "scale",
            OperatorAttributeRule::optional(OperatorAttributeKind::Float),
        ),
        "quantize" | "dequantize" => OperatorAttributeSchema::default()
            .with_rule(
                "storage_dtype",
                OperatorAttributeRule::required(OperatorAttributeKind::DType),
            )
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            ),
        "dtype-conversion" => OperatorAttributeSchema::default().with_rule(
            "dtype",
            OperatorAttributeRule::required(OperatorAttributeKind::DType),
        ),
        "layout-conversion" => OperatorAttributeSchema::default().with_rule(
            "layout",
            OperatorAttributeRule::required(OperatorAttributeKind::Layout),
        ),
        _ => OperatorAttributeSchema::default(),
    }
}

/// A head-wise `concat` ([`ConcatAxis::Heads`]): `[T, H * a]` and `[T, b']`
/// rows (`b'` = `b` or `H * b`) giving `[T, H * (a + b)]`.
fn validate_concat_heads(
    a: &[u64],
    b: &[u64],
    output: &[u64],
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<(), OperatorError> {
    let integer = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => Ok(Some(*value as u64)),
        None => Ok(None),
        Some(_) => Err(OperatorError::OperatorAttributeInvalid {
            attribute: name.into(),
            reason: "must be a positive integer".into(),
        }),
    };
    let head_count =
        integer("head_count")?.ok_or_else(|| OperatorError::OperatorAttributeInvalid {
            attribute: "head_count".into(),
            reason: "concat across heads requires head_count".into(),
        })?;
    let second_head_count = integer("second_head_count")?.unwrap_or(head_count);
    let (&[a_rows, a_width], &[b_rows, b_width]) = (a, b) else {
        return Err(OperatorError::ShapeUnsupported {
            reason: "concat across heads takes two-dimensional inputs".into(),
        });
    };
    if a_rows != b_rows {
        return Err(OperatorError::ShapeMismatch {
            reason: "concat across heads needs inputs with the same rows".into(),
        });
    }
    let (first, second) = concat_heads_widths(a_width, b_width, head_count, second_head_count)
        .map_err(|reason| OperatorError::ShapeMismatch { reason })?;
    if output != [a_rows, head_count * (first + second)] {
        return Err(OperatorError::ShapeMismatch {
            reason: format!(
                "concat across heads expects output [{a_rows}, {}], got {output:?}",
                head_count * (first + second)
            ),
        });
    }
    Ok(())
}

fn validate_shape_rule(
    rule: &ShapeRule,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<(), OperatorError> {
    match rule {
        ShapeRule::Any => Ok(()),
        ShapeRule::Rank(rank) => inputs
            .iter()
            .chain(outputs)
            .all(|tensor| tensor.shape.rank() == *rank)
            .then_some(())
            .ok_or_else(|| OperatorError::ShapeUnsupported {
                reason: format!("expected rank {rank}"),
            }),
        ShapeRule::SameRank => {
            let Some(first) = inputs.first().or_else(|| outputs.first()) else {
                return Ok(());
            };
            inputs
                .iter()
                .chain(outputs)
                .all(|tensor| tensor.shape.rank() == first.shape.rank())
                .then_some(())
                .ok_or_else(|| OperatorError::ShapeMismatch {
                    reason: "tensor ranks differ".into(),
                })
        }
        ShapeRule::SameShape => {
            let Some(first) = inputs.first().or_else(|| outputs.first()) else {
                return Ok(());
            };
            inputs
                .iter()
                .chain(outputs)
                .all(|tensor| tensor.shape.dimensions == first.shape.dimensions)
                .then_some(())
                .ok_or_else(|| OperatorError::ShapeMismatch {
                    reason: "tensor shapes differ".into(),
                })
        }
        ShapeRule::SameShapeOrScalar => {
            let Some(first) = inputs.first().or_else(|| outputs.first()) else {
                return Ok(());
            };
            inputs
                .iter()
                .chain(outputs)
                .enumerate()
                .all(|(index, tensor)| {
                    tensor.shape.dimensions == first.shape.dimensions
                        || (index == 1 && tensor.shape.dimensions.as_slice() == [1])
                })
                .then_some(())
                .ok_or_else(|| OperatorError::ShapeMismatch {
                    reason: "tensor shapes differ and the second input is not a [1] scalar".into(),
                })
        }
        ShapeRule::GeodesicResidual => {
            let input = &inputs[0].shape.dimensions;
            if input.is_empty()
                || input.contains(&0)
                || inputs[1].shape.dimensions != *input
                || outputs[0].shape.dimensions != *input
                || !inputs[2..].iter().all(|tensor| {
                    tensor.shape.dimensions.is_empty() || tensor.shape.dimensions == [1]
                })
            {
                return Err(OperatorError::ShapeMismatch {
                    reason: "geodesic residual expects equal nonempty input/output shapes and two scalar parameters".into(),
                });
            }
            match attributes.get("layer_index") {
                Some(OperatorAttributeValue::Integer(index))
                    if (0..u32::MAX as i64).contains(index) =>
                {
                    Ok(())
                }
                _ => Err(OperatorError::OperatorAttributeInvalid {
                    attribute: "layer_index".into(),
                    reason: "must be a nonnegative original layer index below u32::MAX".into(),
                }),
            }
        }
        ShapeRule::Matmul => {
            if inputs.len() < 2 {
                return Err(OperatorError::InputArityInvalid {
                    expected: 2,
                    actual: inputs.len(),
                });
            }
            let a = &inputs[0].shape.dimensions;
            let b = &inputs[1].shape.dimensions;
            if a.len() < 2 || b.len() < 2 || a[a.len() - 1] != b[b.len() - 2] {
                return Err(OperatorError::ShapeMismatch {
                    reason: "matmul inner dimensions differ".into(),
                });
            }
            Ok(())
        }
        ShapeRule::RmsNorm => {
            if inputs.is_empty() {
                return Err(OperatorError::InputArityInvalid {
                    expected: 1,
                    actual: inputs.len(),
                });
            }
            if outputs.is_empty() {
                return Err(OperatorError::OutputArityInvalid {
                    expected: 1,
                    actual: outputs.len(),
                });
            }
            let input = &inputs[0].shape.dimensions;
            let output = &outputs[0].shape.dimensions;
            if output != input {
                return Err(OperatorError::ShapeMismatch {
                    reason: "RMSNorm output shape must match input".into(),
                });
            }
            // A scale-less RMSNorm has no weight.
            let Some(weight) = inputs.get(1).map(|weight| &weight.shape.dimensions) else {
                return Ok(());
            };
            let Some(cols) = input.last() else {
                return Err(OperatorError::ShapeUnsupported {
                    reason: "RMSNorm input must have at least one dimension".into(),
                });
            };
            let row_broadcast_weight = weight.len() == 2 && weight[0] == 1 && weight[1] == *cols;
            // A per-head RMSNorm's `[head_dimension]` weight divides the
            // row (`head_count * head_dimension`); its kernel checks the
            // attributes against it.
            let head_weight =
                matches!(weight.as_slice(), [width] if *width > 0 && cols.is_multiple_of(*width));
            if weight.as_slice() != [*cols]
                && weight != input
                && !row_broadcast_weight
                && !head_weight
            {
                return Err(OperatorError::ShapeMismatch {
                    reason: "RMSNorm weight shape must be [cols], [1, cols], [head_dimension] \
                             dividing cols, or input-shaped"
                        .into(),
                });
            }
            Ok(())
        }
        // `add`'s second input may either match the first exactly (a
        // regular elementwise add, e.g. `residual-add`'s reuse of `add`'s
        // Kernel), or -- mirroring `ShapeRule::RmsNorm`'s already-
        // permissive weight-row-broadcast convention -- be a single row
        // (`[cols]`/`[1, cols]`, `cols` the first input's last dimension)
        // broadcast across every row (real Qwen2/2.5 QKV projection
        // bias). Additive over strict same-shape: every previously-valid
        // graph still validates identically.
        ShapeRule::RowBroadcastAdd => {
            if inputs.len() < 2 {
                return Err(OperatorError::InputArityInvalid {
                    expected: 2,
                    actual: inputs.len(),
                });
            }
            if outputs.is_empty() {
                return Err(OperatorError::OutputArityInvalid {
                    expected: 1,
                    actual: outputs.len(),
                });
            }
            let a = &inputs[0].shape.dimensions;
            let b = &inputs[1].shape.dimensions;
            let output = &outputs[0].shape.dimensions;
            if output != a {
                return Err(OperatorError::ShapeMismatch {
                    reason: "add output shape must match its first input".into(),
                });
            }
            if a == b {
                return Ok(());
            }
            let Some(cols) = a.last() else {
                return Err(OperatorError::ShapeUnsupported {
                    reason: "add input must have at least one dimension".into(),
                });
            };
            let row_broadcast = b.len() == 2 && b[0] == 1 && b[1] == *cols;
            if b.as_slice() != [*cols] && !row_broadcast {
                return Err(OperatorError::ShapeMismatch {
                    reason: "add's second input must match the first input's shape, or \
                             broadcast as [cols] or [1, cols]"
                        .into(),
                });
            }
            Ok(())
        }
        // Row-wise concatenation: both inputs must share the same column
        // count (last dimension), and the output's row count must equal
        // the sum of both inputs' row counts -- KV-history concatenation's
        // exact contract (`implement-device-resident-multi-step-cuda-
        // decode`). Deliberately does not validate against a graph edge's
        // own *declared* shape (a decode KV edge legitimately declares
        // `[1, kv_dim]` while the real dispatched shape grows every step,
        // the same permissive posture `ShapeRule::RowBroadcastAdd` already
        // takes) -- only the invocation's own real inputs/output are
        // checked against each other here.
        ShapeRule::RowConcat => {
            if inputs.len() < 2 {
                return Err(OperatorError::InputArityInvalid {
                    expected: 2,
                    actual: inputs.len(),
                });
            }
            if outputs.is_empty() {
                return Err(OperatorError::OutputArityInvalid {
                    expected: 1,
                    actual: outputs.len(),
                });
            }
            let a = &inputs[0].shape.dimensions;
            let b = &inputs[1].shape.dimensions;
            let output = &outputs[0].shape.dimensions;
            match concat_axis(attributes) {
                Some(ConcatAxis::Rows) => {}
                Some(ConcatAxis::Heads) => return validate_concat_heads(a, b, output, attributes),
                None => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: "axis".into(),
                        reason: "concat axis must be 'rows' or 'heads'".into(),
                    });
                }
            }
            let (Some((&a_rows, a_rest)), Some((&b_rows, b_rest))) =
                (a.split_first(), b.split_first())
            else {
                return Err(OperatorError::ShapeUnsupported {
                    reason: "concat inputs must have at least one dimension".into(),
                });
            };
            if a_rest != b_rest {
                return Err(OperatorError::ShapeMismatch {
                    reason: "concat inputs must share the same trailing dimensions".into(),
                });
            }
            let expected_output: Vec<u64> = std::iter::once(a_rows + b_rows)
                .chain(a_rest.iter().copied())
                .collect();
            if output.as_slice() != expected_output.as_slice() {
                return Err(OperatorError::ShapeMismatch {
                    reason: "concat output row count must equal the sum of both inputs' row \
                             counts, with matching trailing dimensions"
                        .into(),
                });
            }
            Ok(())
        }
    }
}

/// Reject an invocation whose tensors declare memory-class or mutability
/// intent that conflicts with the Operator's declared memory behavior.
fn validate_memory_behavior(
    memory: &OperatorMemoryBehavior,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), OperatorError> {
    for tensor in inputs.iter().chain(outputs) {
        if let Some(class) = tensor.memory_class_intent {
            if memory.requires_host_visible && !class.is_host_accessible() {
                return Err(OperatorError::MemoryBehaviorUnsupported {
                    reason: format!(
                        "operator requires host-visible memory but tensor declares {class:?}"
                    ),
                });
            }
            if memory.requires_device_resident && class.is_host_accessible() {
                return Err(OperatorError::MemoryBehaviorUnsupported {
                    reason: format!(
                        "operator requires device-resident memory but tensor declares {class:?}"
                    ),
                });
            }
        }
    }
    for tensor in inputs {
        if memory.mutates_input
            && matches!(
                tensor.mutability_intent,
                Some(TensorMutabilityKind::Immutable)
            )
        {
            return Err(OperatorError::MemoryBehaviorUnsupported {
                reason:
                    "operator mutates its input but tensor declares immutable mutability intent"
                        .into(),
            });
        }
    }
    Ok(())
}

/// Reject an invocation whose tensors declare an aliasing intent that
/// requires in-place support the Operator does not advertise.
fn validate_aliasing_intent(
    memory: &OperatorMemoryBehavior,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), OperatorError> {
    for tensor in inputs.iter().chain(outputs) {
        if let Some(aliasing) = tensor.aliasing_intent
            && aliasing.requires_in_place_support()
            && !memory.supports_in_place
        {
            return Err(OperatorError::AliasingUnsupported {
                reason: format!(
                    "tensor declares {aliasing:?} but operator does not support in-place mutation"
                ),
            });
        }
    }
    Ok(())
}

fn validate_dtype_contracts(
    contracts: &[OperatorDTypeContract],
    tensor: &TensorDescriptor,
) -> Result<(), OperatorError> {
    let DTypeDescriptor::Portable(dtype) = &tensor.dtype else {
        return Err(OperatorError::DTypeUnsupported {
            dtype: ComputeDType::UInt8,
        });
    };
    if contracts.iter().any(|contract| contract.accepts(*dtype)) {
        Ok(())
    } else {
        Err(OperatorError::DTypeUnsupported { dtype: *dtype })
    }
}

#[cfg(test)]
mod tests;
