//! Qwen Model Component baseline.
//!
//! This module defines the first concrete architecture baseline on top of the
//! generic Model Component contract: a Qwen-like decoder-only transformer. It
//! composes existing Runtime-owned contracts (Model Component, Execution
//! Graph, Operator, first operator scope, KV cache, tokenizer, adapter) into a
//! configuration and graph-production surface for Qwen-compatible models.
//!
//! The Qwen Model Component is not a Provider: it never selects a Provider or
//! Device, never executes a Kernel directly, and never receives a raw native
//! handle. It only produces portable configuration metadata and Execution
//! Graphs built from required-now Operators.

use crate::{
    ActivationKind, AdapterArchitectureCompatibility, AdapterLayerSelector, AdapterMethod,
    AdapterTargetModule, AdapterTargetModuleRole, ComputeDType, DTypeDescriptor, ExecutionGraph,
    ExecutionGraphId, ExecutionGraphPhase, ExecutionGraphProducer, ExecutionNode, ExecutionNodeId,
    FirstScopeError, FirstScopeErrorCode, GraphError, GraphKvCacheBehavior, GraphKvCacheMetadata,
    GraphModelCompatibility, GraphProductionResult, LayoutDescriptor,
    MODEL_ARTIFACT_SCHEMA_VERSION, ModelComponentArchitectureMetadata, ModelComponentAuthority,
    ModelComponentDescriptor, ModelComponentError, ModelComponentId, ModelComponentIdentity,
    ModelComponentImplementationKind, ModelComponentKvCacheMetadata, ModelComponentModelType,
    ModelComponentObservation, ModelComponentObservationKind,
    ModelComponentQuantizationCompatibility, ModelComponentTokenizerCompatibility,
    ModelComponentVersion, ModelDType, ModelGenerationDefaults, ModelManifest, ModelTensorMetadata,
    NormalizationKind, OperatorAttributeValue, OperatorFamily, OperatorId, OperatorRequirement,
    PositionEncodingKind, ShapeDescriptor, TargetModuleMetadata, TargetModuleRole, TensorAliasing,
    TensorDescriptor, TensorEdge, TensorEdgeId, TensorLayoutKind, browser_feature_supported,
    default_graph_catalog, reference_cpu_kernel_advertisements, validate_first_scope_graph,
    validate_model_component_first_scope_requirements,
    validate_reference_cpu_required_kernel_coverage,
};
use std::{collections::BTreeSet, error::Error, fmt};

pub const QWEN_ARCHITECTURE_FAMILY: &str = "qwen";
pub const QWEN_BASELINE_CONTRACT_VERSION: crate::CapabilityVersion =
    crate::CapabilityVersion::new(1, 0, 0);
/// Tensor Resource and Layout contract version this Qwen baseline targets.
pub const QWEN_TENSOR_CONTRACT_VERSION: crate::CapabilityVersion =
    crate::CapabilityVersion::new(1, 0, 0);
/// Tokenizer Contract version this Qwen baseline targets.
pub const QWEN_TOKENIZER_CONTRACT_VERSION: crate::CapabilityVersion =
    crate::CapabilityVersion::new(1, 0, 0);
/// KV Cache contract version this Qwen baseline targets.
pub const QWEN_KV_CACHE_CONTRACT_VERSION: crate::CapabilityVersion =
    crate::CapabilityVersion::new(1, 0, 0);
/// Adapter Loading contract version this Qwen baseline targets.
pub const QWEN_ADAPTER_CONTRACT_VERSION: crate::CapabilityVersion =
    crate::CapabilityVersion::new(1, 0, 0);

/// Validate that the Tensor/Tokenizer/KV cache/Adapter contract versions a
/// Runtime advertises are compatible with what this Qwen baseline supports.
pub fn qwen_validate_contract_versions(
    tensor: crate::CapabilityVersion,
    tokenizer: crate::CapabilityVersion,
    kv_cache: crate::CapabilityVersion,
    adapter: crate::CapabilityVersion,
) -> Result<(), QwenComponentError> {
    for supported_and_required in [
        (QWEN_TENSOR_CONTRACT_VERSION, tensor),
        (QWEN_TOKENIZER_CONTRACT_VERSION, tokenizer),
        (QWEN_KV_CACHE_CONTRACT_VERSION, kv_cache),
        (QWEN_ADAPTER_CONTRACT_VERSION, adapter),
    ] {
        let (supported, required) = supported_and_required;
        if !supported.is_compatible_with(required) {
            return Err(QwenComponentError::ComponentUnsupportedVersion);
        }
    }
    Ok(())
}

const QWEN_TARGET_MODULE_ROLES: [TargetModuleRole; 9] = [
    TargetModuleRole::QProj,
    TargetModuleRole::KProj,
    TargetModuleRole::VProj,
    TargetModuleRole::OProj,
    TargetModuleRole::GateProj,
    TargetModuleRole::UpProj,
    TargetModuleRole::DownProj,
    TargetModuleRole::LmHead,
    TargetModuleRole::Embedding,
];

const QWEN_REQUIRED_NOW_OPERATORS: [(&str, OperatorFamily); 12] = [
    ("embedding", OperatorFamily::Tensor),
    ("rmsnorm", OperatorFamily::Normalization),
    ("matmul", OperatorFamily::LinearAlgebra),
    ("rope", OperatorFamily::PositionEncoding),
    ("attention", OperatorFamily::Attention),
    ("softmax", OperatorFamily::Activation),
    ("silu", OperatorFamily::Activation),
    ("add", OperatorFamily::Tensor),
    ("mul", OperatorFamily::Tensor),
    ("residual-add", OperatorFamily::Tensor),
    ("dtype-conversion", OperatorFamily::Tensor),
    ("layout-conversion", OperatorFamily::Layout),
];

/// Structured Qwen Model Component error categories.
#[derive(Debug, Eq, PartialEq)]
pub enum QwenComponentError {
    ComponentNotFound,
    ComponentInvalid { reason: String },
    ComponentUntrusted,
    ComponentUnsupportedVersion,
    ArchitectureUnsupported,
    ConfigInvalid { field: &'static str, reason: String },
    TensorInventoryMissing { tensor: String },
    TensorShapeMismatch { tensor: String, reason: String },
    TokenizerIncompatible,
    GenerationMetadataInvalid { reason: String },
    OperatorUnsupported { operator: String },
    GraphProductionFailed { reason: String },
    GraphValidationFailed(GraphError),
    TargetModuleUnavailable { module: String },
    AdapterUnsupported,
    KvCacheMetadataInvalid { reason: String },
    RopeUnsupported { reason: String },
    AttentionVariantUnsupported,
    QuantizationUnsupported,
    DTypeUnsupported,
    LayoutUnsupported,
    ReferenceCpuCoverageMissing,
    CapabilityUnavailable { capability: String },
    AuthorityDenied { authority: String },
    BrowserFeatureUnsupported,
    Internal { reason: String },
}

impl fmt::Display for QwenComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComponentNotFound => write!(f, "qwen component not found"),
            Self::ComponentInvalid { reason } => write!(f, "qwen component invalid: {reason}"),
            Self::ComponentUntrusted => write!(f, "qwen component untrusted"),
            Self::ComponentUnsupportedVersion => write!(f, "qwen component unsupported version"),
            Self::ArchitectureUnsupported => write!(f, "qwen architecture unsupported"),
            Self::ConfigInvalid { field, reason } => {
                write!(f, "qwen config invalid for {field}: {reason}")
            }
            Self::TensorInventoryMissing { tensor } => {
                write!(f, "qwen tensor inventory missing: {tensor}")
            }
            Self::TensorShapeMismatch { tensor, reason } => {
                write!(f, "qwen tensor shape mismatch for {tensor}: {reason}")
            }
            Self::TokenizerIncompatible => write!(f, "qwen tokenizer incompatible"),
            Self::GenerationMetadataInvalid { reason } => {
                write!(f, "qwen generation metadata invalid: {reason}")
            }
            Self::OperatorUnsupported { operator } => {
                write!(f, "qwen operator unsupported: {operator}")
            }
            Self::GraphProductionFailed { reason } => {
                write!(f, "qwen graph production failed: {reason}")
            }
            Self::GraphValidationFailed(error) => {
                write!(f, "qwen graph validation failed: {error}")
            }
            Self::TargetModuleUnavailable { module } => {
                write!(f, "qwen target module unavailable: {module}")
            }
            Self::AdapterUnsupported => write!(f, "qwen adapter unsupported"),
            Self::KvCacheMetadataInvalid { reason } => {
                write!(f, "qwen KV cache metadata invalid: {reason}")
            }
            Self::RopeUnsupported { reason } => write!(f, "qwen RoPE unsupported: {reason}"),
            Self::AttentionVariantUnsupported => write!(f, "qwen attention variant unsupported"),
            Self::QuantizationUnsupported => write!(f, "qwen quantization unsupported"),
            Self::DTypeUnsupported => write!(f, "qwen dtype unsupported"),
            Self::LayoutUnsupported => write!(f, "qwen layout unsupported"),
            Self::ReferenceCpuCoverageMissing => {
                write!(f, "qwen Reference CPU coverage missing")
            }
            Self::CapabilityUnavailable { capability } => {
                write!(f, "qwen capability unavailable: {capability}")
            }
            Self::AuthorityDenied { authority } => write!(f, "qwen authority denied: {authority}"),
            Self::BrowserFeatureUnsupported => write!(f, "qwen browser feature unsupported"),
            Self::Internal { reason } => write!(f, "internal qwen component error: {reason}"),
        }
    }
}

impl Error for QwenComponentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::GraphValidationFailed(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelComponentError> for QwenComponentError {
    fn from(value: ModelComponentError) -> Self {
        match value {
            ModelComponentError::ModelComponentNotFound => Self::ComponentNotFound,
            ModelComponentError::ModelComponentInvalid { reason } => {
                Self::ComponentInvalid { reason }
            }
            ModelComponentError::ModelComponentUntrusted => Self::ComponentUntrusted,
            ModelComponentError::ModelComponentUnsupportedVersion => {
                Self::ComponentUnsupportedVersion
            }
            ModelComponentError::ArchitectureUnsupported => Self::ArchitectureUnsupported,
            ModelComponentError::ArchitectureMetadataInvalid { field, reason } => {
                Self::ConfigInvalid { field, reason }
            }
            ModelComponentError::ModelConfigInvalid { reason } => Self::ConfigInvalid {
                field: "config",
                reason,
            },
            ModelComponentError::ModelArtifactIncompatible => Self::ComponentInvalid {
                reason: "model artifact incompatible".into(),
            },
            ModelComponentError::TokenizerIncompatible => Self::TokenizerIncompatible,
            ModelComponentError::OperatorCatalogIncompatible => Self::OperatorUnsupported {
                operator: "unknown".into(),
            },
            ModelComponentError::GraphContractIncompatible => Self::GraphProductionFailed {
                reason: "graph contract incompatible".into(),
            },
            ModelComponentError::GraphProductionFailed { reason } => {
                Self::GraphProductionFailed { reason }
            }
            ModelComponentError::GraphValidationFailed(error) => Self::GraphValidationFailed(error),
            ModelComponentError::TargetModuleUnavailable { module } => {
                Self::TargetModuleUnavailable { module }
            }
            ModelComponentError::AdapterIncompatible => Self::AdapterUnsupported,
            ModelComponentError::KvCacheMetadataInvalid { reason } => {
                Self::KvCacheMetadataInvalid { reason }
            }
            ModelComponentError::QuantizationUnsupported => Self::QuantizationUnsupported,
            ModelComponentError::CapabilityUnavailable { capability } => {
                Self::CapabilityUnavailable { capability }
            }
            ModelComponentError::AuthorityDenied { authority } => {
                Self::AuthorityDenied { authority }
            }
            ModelComponentError::ProviderAccessDenied => Self::AuthorityDenied {
                authority: "provider".into(),
            },
            ModelComponentError::DeviceAccessDenied => Self::AuthorityDenied {
                authority: "device".into(),
            },
            ModelComponentError::KernelAccessDenied => Self::AuthorityDenied {
                authority: "kernel".into(),
            },
            ModelComponentError::ProviderOwnedResourceAccessDenied => Self::AuthorityDenied {
                authority: "provider-owned-resource".into(),
            },
            ModelComponentError::MemoryPointerAccessDenied => Self::AuthorityDenied {
                authority: "memory-pointer".into(),
            },
            ModelComponentError::BrowserFeatureUnsupported => Self::BrowserFeatureUnsupported,
            ModelComponentError::InternalModelComponent { reason } => Self::Internal { reason },
        }
    }
}

impl From<FirstScopeError> for QwenComponentError {
    fn from(value: FirstScopeError) -> Self {
        match value.code {
            FirstScopeErrorCode::KernelMissing => Self::ReferenceCpuCoverageMissing,
            FirstScopeErrorCode::DTypeUnsupported => Self::DTypeUnsupported,
            FirstScopeErrorCode::LayoutUnsupported | FirstScopeErrorCode::ShapeUnsupported => {
                Self::LayoutUnsupported
            }
            _ => Self::OperatorUnsupported {
                operator: value
                    .operator
                    .as_ref()
                    .map(|operator| operator.name().to_string())
                    .unwrap_or_default(),
            },
        }
    }
}

impl From<GraphError> for QwenComponentError {
    fn from(value: GraphError) -> Self {
        Self::GraphValidationFailed(value)
    }
}

/// Only RoPE position indexing mode supported by the first Qwen baseline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenRopePositionMode {
    Sequential,
}

impl QwenRopePositionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
        }
    }
}

/// Explicit Qwen RoPE metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenRopeConfig {
    pub base: f64,
    pub scale: Option<f64>,
    pub dimension: u64,
    pub position_mode: QwenRopePositionMode,
    pub dynamic_scaling_supported: bool,
}

impl QwenRopeConfig {
    pub fn standard(head_dimension: u64) -> Self {
        Self {
            base: 10_000.0,
            scale: None,
            dimension: head_dimension,
            position_mode: QwenRopePositionMode::Sequential,
            dynamic_scaling_supported: false,
        }
    }

    pub fn validate(&self, head_dimension: u64) -> Result<(), QwenComponentError> {
        if self.base <= 0.0 {
            return Err(QwenComponentError::RopeUnsupported {
                reason: "RoPE base must be positive".into(),
            });
        }
        if self.dimension == 0 || self.dimension > head_dimension {
            return Err(QwenComponentError::RopeUnsupported {
                reason: "RoPE dimension must be within head dimension".into(),
            });
        }
        if let Some(scale) = self.scale
            && scale != 1.0
            && !self.dynamic_scaling_supported
        {
            return Err(QwenComponentError::RopeUnsupported {
                reason: "dynamic RoPE scaling is not supported by this baseline".into(),
            });
        }
        Ok(())
    }
}

/// Qwen-like decoder-only architecture configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenConfig {
    pub architecture: ModelComponentArchitectureMetadata,
    pub rope: QwenRopeConfig,
    pub rmsnorm_epsilon: f32,
    pub tied_embeddings: bool,
    /// Whether tokenizer compatibility SHALL require a BOS special token.
    pub require_bos: bool,
    /// Whether tokenizer compatibility SHALL require a pad special token.
    pub require_pad: bool,
    /// Expected added-token count, when the baseline declares one.
    pub expected_added_tokens: Option<u32>,
    /// Whether Model Artifact compatibility SHALL require chat template
    /// metadata to be present.
    pub chat_template_required: bool,
}

impl QwenConfig {
    pub fn new(architecture: ModelComponentArchitectureMetadata, rope: QwenRopeConfig) -> Self {
        Self {
            architecture,
            rope,
            rmsnorm_epsilon: 1e-6,
            tied_embeddings: false,
            require_bos: false,
            require_pad: false,
            expected_added_tokens: None,
            chat_template_required: false,
        }
    }

    pub fn validate(&self, identity: &ModelComponentIdentity) -> Result<(), QwenComponentError> {
        if self.architecture.family != QWEN_ARCHITECTURE_FAMILY {
            return Err(QwenComponentError::ArchitectureUnsupported);
        }
        self.architecture.validate(identity)?;
        if self.architecture.model_type != ModelComponentModelType::CausalLanguageModel {
            return Err(QwenComponentError::ConfigInvalid {
                field: "model_type",
                reason: "Qwen baseline requires a decoder-only causal language model".into(),
            });
        }
        if self.architecture.normalization != NormalizationKind::RmsNorm {
            return Err(QwenComponentError::ConfigInvalid {
                field: "normalization",
                reason: "Qwen baseline requires RMSNorm".into(),
            });
        }
        if self.architecture.activation != ActivationKind::Silu {
            return Err(QwenComponentError::ConfigInvalid {
                field: "activation",
                reason: "Qwen baseline requires SiLU".into(),
            });
        }
        if self.architecture.position_encoding != PositionEncodingKind::Rotary {
            return Err(QwenComponentError::ConfigInvalid {
                field: "position_encoding",
                reason: "Qwen baseline requires RoPE".into(),
            });
        }
        if self.rmsnorm_epsilon <= 0.0 {
            return Err(QwenComponentError::ConfigInvalid {
                field: "rmsnorm_epsilon",
                reason: "must be positive".into(),
            });
        }
        self.rope.validate(self.architecture.head_dimension)?;
        Ok(())
    }
}

/// Build the decoder-only architecture metadata for a Qwen-like model,
/// setting the fields the baseline requires (family, model type,
/// normalization, activation, position encoding) and deriving the attention
/// variant from the attention/KV head counts.
#[allow(clippy::too_many_arguments)]
pub fn qwen_architecture_metadata(
    hidden_size: u64,
    layer_count: u64,
    attention_head_count: u64,
    kv_head_count: u64,
    head_dimension: u64,
    intermediate_size: u64,
    vocabulary_size: u64,
    context_length: u64,
) -> ModelComponentArchitectureMetadata {
    ModelComponentArchitectureMetadata {
        family: QWEN_ARCHITECTURE_FAMILY.into(),
        model_type: ModelComponentModelType::CausalLanguageModel,
        hidden_size,
        layer_count,
        attention_head_count,
        kv_head_count,
        head_dimension,
        intermediate_size,
        vocabulary_size,
        context_length,
        position_encoding: PositionEncodingKind::Rotary,
        normalization: NormalizationKind::RmsNorm,
        activation: ActivationKind::Silu,
        attention: if kv_head_count < attention_head_count {
            crate::AttentionVariant::GroupedQuery
        } else {
            crate::AttentionVariant::MultiHead
        },
        quantization: None,
        tokenizer_family: None,
        adapter_target_modules: QWEN_TARGET_MODULE_ROLES.into_iter().collect(),
    }
}

/// Build a trusted Qwen Model Component identity for the given id/version.
pub fn qwen_component_identity(
    id: ModelComponentId,
    version: ModelComponentVersion,
    implementation: ModelComponentImplementationKind,
) -> ModelComponentIdentity {
    ModelComponentIdentity::new(id, version, implementation)
        .trusted()
        .with_architecture_family(QWEN_ARCHITECTURE_FAMILY)
        .with_model_artifact_schema_version(MODEL_ARTIFACT_SCHEMA_VERSION)
}

/// Canonical Qwen target modules exposed for Adapter Loading.
pub fn qwen_target_modules() -> Vec<TargetModuleMetadata> {
    QWEN_TARGET_MODULE_ROLES
        .into_iter()
        .map(TargetModuleMetadata::canonical)
        .collect()
}

/// Required-now Operator requirements for the Qwen baseline first executable
/// path. See [`crate::first_operator_scope`].
pub fn qwen_operator_requirements() -> Vec<OperatorRequirement> {
    QWEN_REQUIRED_NOW_OPERATORS
        .into_iter()
        .map(|(name, family)| OperatorRequirement::new(OperatorId::magnetar(name, 1, family)))
        .collect()
}

/// Authority the Qwen Model Component may hold. Deliberately excludes
/// filesystem, network, process, shell, secrets, Git, workspace, Provider,
/// Device, and Kernel authority.
pub fn qwen_authority() -> BTreeSet<ModelComponentAuthority> {
    BTreeSet::from([
        ModelComponentAuthority::ModelArtifactRead,
        ModelComponentAuthority::TokenizerArtifactRead,
        ModelComponentAuthority::AdapterArtifactRead,
        ModelComponentAuthority::QuantizationArtifactRead,
        ModelComponentAuthority::KvCacheAccess,
        ModelComponentAuthority::PrefixCacheAccess,
        ModelComponentAuthority::ComputeCapability,
        ModelComponentAuthority::GraphProduction,
        ModelComponentAuthority::OperatorCatalogRead,
        ModelComponentAuthority::ObservabilityEmit,
        ModelComponentAuthority::RuntimeDiagnostics,
    ])
}

pub fn qwen_kv_cache_metadata(config: &QwenConfig) -> ModelComponentKvCacheMetadata {
    let a = &config.architecture;
    ModelComponentKvCacheMetadata {
        layer_count: a.layer_count,
        head_count: a.attention_head_count,
        kv_head_count: a.kv_head_count,
        head_dimension: a.head_dimension,
        cache_dtype: "f32".into(),
        layout_preference: "contiguous".into(),
        paged: false,
        append_semantics: "append".into(),
        position_behavior: "sequential".into(),
    }
}

pub fn qwen_tokenizer_compatibility(config: &QwenConfig) -> ModelComponentTokenizerCompatibility {
    let mut special_tokens = BTreeSet::from(["eos".to_string()]);
    if config.require_bos {
        special_tokens.insert("bos".to_string());
    }
    if config.require_pad {
        special_tokens.insert("pad".to_string());
    }
    ModelComponentTokenizerCompatibility {
        vocabulary_size: config.architecture.vocabulary_size,
        special_tokens,
        family: config.architecture.tokenizer_family.clone(),
        chat_template_required: config.chat_template_required,
        added_token_behavior: config
            .expected_added_tokens
            .map(|count| format!("expects {count} added tokens")),
    }
}

/// Validate a resolved [`crate::TokenizerMetadata`] against Qwen architecture
/// metadata: vocabulary size compatibility, EOS token availability (always
/// required), and BOS/pad token policy and added-token behavior where the
/// baseline config declares them relevant. Tokenizer execution itself remains
/// owned by the Tokenizer Contract.
pub fn qwen_validate_tokenizer_compatibility(
    config: &QwenConfig,
    tokenizer: &crate::TokenizerMetadata,
) -> Result<(), QwenComponentError> {
    let expected_vocabulary_size =
        u32::try_from(config.architecture.vocabulary_size).map_err(|_| {
            QwenComponentError::ConfigInvalid {
                field: "vocabulary_size",
                reason: "vocabulary size exceeds tokenizer representable range".into(),
            }
        })?;
    let mut expected_special_tokens = vec![crate::SpecialTokenKind::Eos];
    if config.require_bos {
        expected_special_tokens.push(crate::SpecialTokenKind::Bos);
    }
    if config.require_pad {
        expected_special_tokens.push(crate::SpecialTokenKind::Pad);
    }
    let compatibility = crate::TokenizerCompatibility {
        expected_digest: None,
        expected_vocabulary_size: Some(expected_vocabulary_size),
        expected_family: config.architecture.tokenizer_family.clone(),
        expected_model_max_length: None,
        expected_added_tokens: config.expected_added_tokens,
        expected_special_tokens,
        expected_normalization: None,
    };
    tokenizer
        .validate_compatibility(&compatibility)
        .map_err(|_| QwenComponentError::TokenizerIncompatible)
}

/// Validate generation default metadata against Qwen architecture metadata,
/// without owning Generation semantics: checks that declared defaults are
/// consistent with the model's own context length, that stop token metadata
/// is well-formed, and — when a resolved tokenizer is supplied — that EOS
/// (and BOS, where required) special tokens are resolvable. Sampling
/// defaults (temperature/top-p/top-k) are intentionally never inspected here:
/// they remain non-authoritative hints owned by the Generation/Sampling
/// contracts.
pub fn qwen_validate_generation_defaults(
    config: &QwenConfig,
    defaults: &ModelGenerationDefaults,
    tokenizer: Option<&crate::TokenizerMetadata>,
) -> Result<(), QwenComponentError> {
    if let Some(max_tokens) = defaults.max_tokens
        && u64::from(max_tokens) > config.architecture.context_length
    {
        return Err(QwenComponentError::GenerationMetadataInvalid {
            reason: format!(
                "max_tokens {max_tokens} exceeds context length {}",
                config.architecture.context_length
            ),
        });
    }
    if defaults.stop_tokens.iter().any(|token| token.is_empty()) {
        return Err(QwenComponentError::GenerationMetadataInvalid {
            reason: "stop token metadata must not contain empty entries".into(),
        });
    }
    if let Some(tokenizer) = tokenizer {
        qwen_validate_tokenizer_compatibility(config, tokenizer)?;
    }
    Ok(())
}

/// The first Qwen baseline rejects quantized artifacts by declaring no
/// supported quantization methods. A later baseline may add explicit
/// dequantization support here.
pub fn qwen_quantization_compatibility() -> ModelComponentQuantizationCompatibility {
    ModelComponentQuantizationCompatibility {
        supported_methods: BTreeSet::new(),
        tensor_grouping: None,
        scale_metadata_required: false,
        zero_point_metadata_required: false,
        packed_layout: None,
        dequantization_operators: BTreeSet::new(),
        quantized_operators: BTreeSet::new(),
    }
}

/// Assemble and validate a Qwen [`ModelComponentDescriptor`], including
/// first-scope operator requirement validation.
pub fn qwen_component_descriptor(
    identity: ModelComponentIdentity,
    config: &QwenConfig,
) -> Result<ModelComponentDescriptor, QwenComponentError> {
    config.validate(&identity)?;
    let descriptor = ModelComponentDescriptor {
        identity,
        architecture: config.architecture.clone(),
        target_modules: qwen_target_modules(),
        graph_phases: BTreeSet::from([
            ExecutionGraphPhase::Warmup,
            ExecutionGraphPhase::Prefill,
            ExecutionGraphPhase::Decode,
        ]),
        operator_requirements: qwen_operator_requirements(),
        capability_requirements: Vec::new(),
        authority: qwen_authority(),
        kv_cache: Some(qwen_kv_cache_metadata(config)),
        tokenizer: Some(qwen_tokenizer_compatibility(config)),
        quantization: Some(qwen_quantization_compatibility()),
    };
    descriptor.validate()?;
    validate_model_component_first_scope_requirements(&descriptor.operator_requirements)?;
    Ok(descriptor)
}

/// Expected logical tensor names for the Qwen baseline tensor inventory.
pub fn qwen_expected_tensor_names(layer_count: u64, tied_embeddings: bool) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("token_embedding".to_string());
    for layer in 0..layer_count {
        for suffix in [
            "input_norm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attn_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            names.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    names.insert("final_norm".to_string());
    if !tied_embeddings {
        names.insert("lm_head".to_string());
    }
    names
}

/// Validate that every expected logical tensor is present in `tensors`.
pub fn qwen_validate_tensor_inventory(
    config: &QwenConfig,
    tensors: &[ModelTensorMetadata],
) -> Result<(), QwenComponentError> {
    let present: BTreeSet<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    for expected in
        qwen_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        if !present.contains(expected.as_str()) {
            return Err(QwenComponentError::TensorInventoryMissing { tensor: expected });
        }
    }
    Ok(())
}

/// Expected shape for a Qwen logical tensor name, if the baseline declares
/// one.
pub fn qwen_expected_tensor_shape(name: &str, config: &QwenConfig) -> Option<Vec<u64>> {
    let a = &config.architecture;
    let q_dim = a.attention_head_count * a.head_dimension;
    let kv_dim = a.kv_head_count * a.head_dimension;
    match name {
        "token_embedding" => return Some(vec![a.vocabulary_size, a.hidden_size]),
        "final_norm" => return Some(vec![a.hidden_size]),
        "lm_head" => return Some(vec![a.hidden_size, a.vocabulary_size]),
        _ => {}
    }
    let rest = name.strip_prefix("layers.")?.split_once('.')?.1;
    match rest {
        "input_norm" | "post_attn_norm" => Some(vec![a.hidden_size]),
        "self_attn.q_proj" => Some(vec![a.hidden_size, q_dim]),
        "self_attn.k_proj" | "self_attn.v_proj" => Some(vec![a.hidden_size, kv_dim]),
        "self_attn.o_proj" => Some(vec![q_dim, a.hidden_size]),
        "mlp.gate_proj" | "mlp.up_proj" => Some(vec![a.hidden_size, a.intermediate_size]),
        "mlp.down_proj" => Some(vec![a.intermediate_size, a.hidden_size]),
        _ => None,
    }
}

/// Validate declared tensor shapes for every recognized Qwen logical tensor.
pub fn qwen_validate_tensor_shapes(
    config: &QwenConfig,
    tensors: &[ModelTensorMetadata],
) -> Result<(), QwenComponentError> {
    for tensor in tensors {
        if let Some(expected) = qwen_expected_tensor_shape(&tensor.name, config)
            && tensor.shape != expected
        {
            return Err(QwenComponentError::TensorShapeMismatch {
                tensor: tensor.name.clone(),
                reason: format!("expected shape {expected:?}, got {:?}", tensor.shape),
            });
        }
    }
    Ok(())
}

/// Validate the shared embedding tensor's shape when the baseline is
/// configured for tied embeddings. Untied configurations have nothing to
/// check here: `lm_head` shape is covered by [`qwen_validate_tensor_shapes`].
pub fn qwen_validate_tied_embedding_shape(
    config: &QwenConfig,
    tensors: &[ModelTensorMetadata],
) -> Result<(), QwenComponentError> {
    if !config.tied_embeddings {
        return Ok(());
    }
    let Some(embedding) = tensors
        .iter()
        .find(|tensor| tensor.name == "token_embedding")
    else {
        return Ok(());
    };
    let expected = vec![
        config.architecture.vocabulary_size,
        config.architecture.hidden_size,
    ];
    if embedding.shape != expected {
        return Err(QwenComponentError::TensorShapeMismatch {
            tensor: "token_embedding".into(),
            reason: format!(
                "tied embedding shape must be {expected:?}, got {:?}",
                embedding.shape
            ),
        });
    }
    Ok(())
}

/// Validate Model Artifact compatibility: architecture/schema/quantization
/// compatibility through the generic descriptor, Qwen tensor inventory and
/// shape validation (including the tied-embedding case), and chat template
/// presence when the baseline requires it. Preserves Runtime artifact trust:
/// this function never bypasses trust validation performed elsewhere by
/// Model Loading.
pub fn qwen_validate_model_artifact(
    descriptor: &ModelComponentDescriptor,
    config: &QwenConfig,
    manifest: &ModelManifest,
) -> Result<(), QwenComponentError> {
    descriptor.validate_model_artifact(manifest)?;
    qwen_validate_tensor_inventory(config, &manifest.tensors)?;
    qwen_validate_tensor_shapes(config, &manifest.tensors)?;
    qwen_validate_tied_embedding_shape(config, &manifest.tensors)?;
    if config.chat_template_required && manifest.chat_template.is_none() {
        return Err(QwenComponentError::ComponentInvalid {
            reason: "chat template metadata required but not present in Model Artifact".into(),
        });
    }
    Ok(())
}

/// Adapter (e.g. LoRA) architecture compatibility metadata for a Qwen model.
pub fn qwen_adapter_architecture_compatibility(
    config: &QwenConfig,
    implementation: impl Into<String>,
) -> AdapterArchitectureCompatibility {
    let a = &config.architecture;
    AdapterArchitectureCompatibility {
        family: QWEN_ARCHITECTURE_FAMILY.into(),
        implementation: implementation.into(),
        hidden_size: Some(a.hidden_size),
        layer_count: Some(a.layer_count as u32),
        position_encoding: Some("rotary".into()),
        target_modules: QWEN_TARGET_MODULE_ROLES
            .iter()
            .map(|role| role.canonical_name().to_string())
            .collect(),
        supported_storage_dtypes: BTreeSet::from([
            ModelDType::F32,
            ModelDType::F16,
            ModelDType::Bf16,
        ]),
        supported_compute_dtypes: BTreeSet::from([ComputeDType::Float32]),
        supported_quantization_formats: BTreeSet::new(),
    }
}

/// Validate that Reference CPU advertises every required-now Kernel the Qwen
/// baseline needs.
pub fn qwen_validate_reference_cpu_coverage() -> Result<(), QwenComponentError> {
    validate_reference_cpu_required_kernel_coverage(&reference_cpu_kernel_advertisements())
        .map_err(QwenComponentError::from)
}

/// Validate that the given implementation kind is permitted on the current
/// target (in particular, browser/wasm32 targets require a browser-compatible
/// implementation).
pub fn qwen_browser_supported(
    implementation: ModelComponentImplementationKind,
) -> Result<(), QwenComponentError> {
    browser_feature_supported(implementation).map_err(QwenComponentError::from)
}

/// A stable compatibility key combining Qwen Component identity and
/// architecture, suitable for use as KV Cache / Prefix Cache compatibility
/// metadata (e.g. `KvCacheCompatibility::model_architecture`) and as a
/// portable `ModelArchitecture` identifier (dot-separated: identifiers must
/// not contain path, URI, or selector characters such as `:`, `/`, `\`).
pub fn qwen_component_compatibility_key(identity: &ModelComponentIdentity) -> String {
    format!(
        "{QWEN_ARCHITECTURE_FAMILY}.{}.{}",
        identity.id, identity.version.0
    )
}

/// Conformance fixture names the Qwen baseline SHALL cover.
pub fn qwen_conformance_fixture_names() -> BTreeSet<&'static str> {
    BTreeSet::from([
        "valid-minimal-config",
        "invalid-architecture-family",
        "invalid-hidden-head-configuration",
        "missing-tensor-inventory",
        "invalid-tensor-shape",
        "target-module-exposure",
        "prefill-graph-production",
        "decode-graph-production",
        "required-operator-scope-validation",
        "tokenizer-compatibility-validation",
        "kv-cache-metadata-validation",
        "adapter-target-validation",
        "unsupported-quantization-rejection",
        "authority-denial",
        "no-provider-device-kernel-handle-exposure",
    ])
}

fn f32_edge(id: impl Into<String>, dims: Vec<u64>) -> TensorEdge {
    let id = TensorEdgeId::new(id);
    TensorEdge::new(
        id,
        TensorDescriptor::new(
            ShapeDescriptor::new(dims),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    )
}

fn token_id_edge(id: impl Into<String>, dims: Vec<u64>) -> TensorEdge {
    let id = TensorEdgeId::new(id);
    TensorEdge::new(
        id,
        TensorDescriptor::new(
            ShapeDescriptor::new(dims),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    )
}

/// Build the `weight.lm_head` edge. When the baseline uses tied embeddings,
/// this logical tensor is declared with [`TensorAliasing::MayAlias`] pointing
/// at `weight.token_embedding`, recording that it shares storage rather than
/// silently duplicating it; when untied it is an independent tensor.
fn qwen_lm_head_weight_edge(config: &QwenConfig) -> TensorEdge {
    let a = &config.architecture;
    let mut edge = f32_edge("weight.lm_head", vec![a.hidden_size, a.vocabulary_size]);
    if config.tied_embeddings {
        edge.aliasing = TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"));
    }
    edge
}

fn op_node(id: impl Into<String>, name: &str, family: OperatorFamily) -> ExecutionNode {
    ExecutionNode::new(
        ExecutionNodeId::new(id.into()),
        OperatorId::magnetar(name, 1, family),
    )
}

/// Build a Qwen prefill or decode Execution Graph: embedding, `layer_count`
/// repeated pre-norm decoder layers (RMSNorm, QKV matmul, RoPE, attention,
/// output projection, residual-add, RMSNorm, gated MLP, residual-add), a
/// final RMSNorm, and an `lm_head` logits projection. Every node uses a
/// required-now Operator.
pub fn qwen_build_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    phase: ExecutionGraphPhase,
    sequence_length: u64,
    kv_cache_enabled: bool,
    position_offset: u64,
) -> Result<ExecutionGraph, QwenComponentError> {
    if sequence_length == 0 {
        return Err(QwenComponentError::GraphProductionFailed {
            reason: "sequence length must be positive".into(),
        });
    }
    let a = &config.architecture;
    let q_dim = a.attention_head_count * a.head_dimension;
    let kv_dim = a.kv_head_count * a.head_dimension;

    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new(format!(
            "qwen-{phase:?}-{}-{}",
            identity.id, identity.version.0
        )),
        phase,
    )
    .with_producer(ExecutionGraphProducer::ModelComponent {
        component_id: identity.id.as_str().into(),
    });
    graph.model = GraphModelCompatibility {
        model_instance_id: None,
        architecture: Some(QWEN_ARCHITECTURE_FAMILY.into()),
        tokenizer_dependency: None,
    };
    graph.fingerprint = Some(qwen_component_compatibility_key(identity));

    graph = graph
        .with_edge(token_id_edge("input.token_ids", vec![sequence_length]))
        .with_edge(f32_edge(
            "weight.token_embedding",
            vec![a.vocabulary_size, a.hidden_size],
        ))
        .with_edge(f32_edge("hidden.0", vec![sequence_length, a.hidden_size]))
        .with_node(
            op_node("embedding", "embedding", OperatorFamily::Tensor)
                .with_input(TensorEdgeId::new("input.token_ids"))
                .with_input(TensorEdgeId::new("weight.token_embedding"))
                .with_output(TensorEdgeId::new("hidden.0")),
        );

    let mut hidden_edge = "hidden.0".to_string();
    let kv_behavior = match phase {
        ExecutionGraphPhase::Decode => GraphKvCacheBehavior::Append,
        _ => GraphKvCacheBehavior::Output,
    };

    for layer in 0..a.layer_count {
        let prefix = format!("layer{layer}");
        let residual_in = hidden_edge.clone();

        let normed = format!("{prefix}.normed");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.{prefix}.input_norm"),
                vec![a.hidden_size],
            ))
            .with_edge(f32_edge(
                normed.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.input_norm"),
                    "rmsnorm",
                    OperatorFamily::Normalization,
                )
                .with_input(TensorEdgeId::new(residual_in.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.input_norm")))
                .with_output(TensorEdgeId::new(normed.clone()))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
            );

        let q = format!("{prefix}.q");
        let k = format!("{prefix}.k");
        let v = format!("{prefix}.v");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.{prefix}.q_proj"),
                vec![a.hidden_size, q_dim],
            ))
            .with_edge(f32_edge(q.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.q_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.q_proj")))
                .with_output(TensorEdgeId::new(q.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.{prefix}.k_proj"),
                vec![a.hidden_size, kv_dim],
            ))
            .with_edge(f32_edge(k.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.k_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.k_proj")))
                .with_output(TensorEdgeId::new(k.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.{prefix}.v_proj"),
                vec![a.hidden_size, kv_dim],
            ))
            .with_edge(f32_edge(v.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.v_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.v_proj")))
                .with_output(TensorEdgeId::new(v.clone())),
            );

        let q_rope = format!("{prefix}.q_rope");
        let k_rope = format!("{prefix}.k_rope");
        graph = graph
            .with_edge(f32_edge(q_rope.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.rope_q"),
                    "rope",
                    OperatorFamily::PositionEncoding,
                )
                .with_input(TensorEdgeId::new(q.clone()))
                .with_output(TensorEdgeId::new(q_rope.clone()))
                .with_attribute("base", OperatorAttributeValue::Float(config.rope.base))
                .with_attribute(
                    "dimension",
                    OperatorAttributeValue::Integer(config.rope.dimension as i64),
                )
                .with_attribute(
                    "position_mode",
                    OperatorAttributeValue::String(config.rope.position_mode.as_str().into()),
                )
                .with_attribute(
                    "position_offset",
                    OperatorAttributeValue::Integer(position_offset as i64),
                ),
            )
            .with_edge(f32_edge(k_rope.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.rope_k"),
                    "rope",
                    OperatorFamily::PositionEncoding,
                )
                .with_input(TensorEdgeId::new(k.clone()))
                .with_output(TensorEdgeId::new(k_rope.clone()))
                .with_attribute("base", OperatorAttributeValue::Float(config.rope.base))
                .with_attribute(
                    "dimension",
                    OperatorAttributeValue::Integer(config.rope.dimension as i64),
                )
                .with_attribute(
                    "position_mode",
                    OperatorAttributeValue::String(config.rope.position_mode.as_str().into()),
                )
                .with_attribute(
                    "position_offset",
                    OperatorAttributeValue::Integer(position_offset as i64),
                ),
            );

        if kv_cache_enabled {
            let cache_metadata = |cache_role: &str| GraphKvCacheMetadata {
                cache_id: format!("qwen.{prefix}.{cache_role}"),
                behavior: kv_behavior.clone(),
                paged: false,
                compatibility_key: qwen_component_compatibility_key(identity),
            };
            if let Some(edge) = graph.edges.get_mut(&TensorEdgeId::new(k_rope.clone())) {
                edge.kv_cache = Some(cache_metadata("k"));
            }
            if let Some(edge) = graph.edges.get_mut(&TensorEdgeId::new(v.clone())) {
                edge.kv_cache = Some(cache_metadata("v"));
            }
        }

        let attn_out = format!("{prefix}.attn_out");
        graph = graph
            .with_edge(f32_edge(attn_out.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.attention"),
                    "attention",
                    OperatorFamily::Attention,
                )
                .with_input(TensorEdgeId::new(q_rope.clone()))
                .with_input(TensorEdgeId::new(k_rope.clone()))
                .with_input(TensorEdgeId::new(v.clone()))
                .with_output(TensorEdgeId::new(attn_out.clone()))
                .with_attribute("causal", OperatorAttributeValue::Boolean(true))
                .with_attribute(
                    "head_count",
                    OperatorAttributeValue::Integer(a.attention_head_count as i64),
                )
                .with_attribute(
                    "kv_head_count",
                    OperatorAttributeValue::Integer(a.kv_head_count as i64),
                )
                .with_attribute(
                    "head_dimension",
                    OperatorAttributeValue::Integer(a.head_dimension as i64),
                )
                .with_attribute(
                    "attention_mask_kind",
                    OperatorAttributeValue::String("causal".into()),
                ),
            );

        let attn_proj = format!("{prefix}.attn_proj");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.{prefix}.o_proj"),
                vec![q_dim, a.hidden_size],
            ))
            .with_edge(f32_edge(
                attn_proj.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.o_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(attn_out.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.o_proj")))
                .with_output(TensorEdgeId::new(attn_proj.clone())),
            );

        let post_attn = format!("{prefix}.post_attn");
        graph = graph
            .with_edge(f32_edge(
                post_attn.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.residual1"),
                    "residual-add",
                    OperatorFamily::Tensor,
                )
                .with_input(TensorEdgeId::new(residual_in.clone()))
                .with_input(TensorEdgeId::new(attn_proj.clone()))
                .with_output(TensorEdgeId::new(post_attn.clone())),
            );

        let mlp_normed = format!("{prefix}.mlp_normed");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.{prefix}.post_attn_norm"),
                vec![a.hidden_size],
            ))
            .with_edge(f32_edge(
                mlp_normed.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.post_attn_norm"),
                    "rmsnorm",
                    OperatorFamily::Normalization,
                )
                .with_input(TensorEdgeId::new(post_attn.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.post_attn_norm")))
                .with_output(TensorEdgeId::new(mlp_normed.clone()))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
            );

        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        let activated = format!("{prefix}.activated");
        let mlp_hidden = format!("{prefix}.mlp_hidden");
        let mlp_out = format!("{prefix}.mlp_out");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.{prefix}.gate_proj"),
                vec![a.hidden_size, a.intermediate_size],
            ))
            .with_edge(f32_edge(
                gate.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.gate_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(mlp_normed.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.gate_proj")))
                .with_output(TensorEdgeId::new(gate.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.{prefix}.up_proj"),
                vec![a.hidden_size, a.intermediate_size],
            ))
            .with_edge(f32_edge(
                up.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.up_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(mlp_normed.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.up_proj")))
                .with_output(TensorEdgeId::new(up.clone())),
            )
            .with_edge(f32_edge(
                activated.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.silu"), "silu", OperatorFamily::Activation)
                    .with_input(TensorEdgeId::new(gate.clone()))
                    .with_output(TensorEdgeId::new(activated.clone())),
            )
            .with_edge(f32_edge(
                mlp_hidden.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.mul"), "mul", OperatorFamily::Tensor)
                    .with_input(TensorEdgeId::new(activated.clone()))
                    .with_input(TensorEdgeId::new(up.clone()))
                    .with_output(TensorEdgeId::new(mlp_hidden.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.{prefix}.down_proj"),
                vec![a.intermediate_size, a.hidden_size],
            ))
            .with_edge(f32_edge(
                mlp_out.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.down_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(mlp_hidden.clone()))
                .with_input(TensorEdgeId::new(format!("weight.{prefix}.down_proj")))
                .with_output(TensorEdgeId::new(mlp_out.clone())),
            );

        let layer_out = format!("{prefix}.out");
        graph = graph
            .with_edge(f32_edge(
                layer_out.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.residual2"),
                    "residual-add",
                    OperatorFamily::Tensor,
                )
                .with_input(TensorEdgeId::new(post_attn.clone()))
                .with_input(TensorEdgeId::new(mlp_out.clone()))
                .with_output(TensorEdgeId::new(layer_out.clone())),
            );

        hidden_edge = layer_out;
    }

    graph = graph
        .with_edge(f32_edge("weight.final_norm", vec![a.hidden_size]))
        .with_edge(f32_edge(
            "hidden.final",
            vec![sequence_length, a.hidden_size],
        ))
        .with_node(
            op_node("final_norm", "rmsnorm", OperatorFamily::Normalization)
                .with_input(TensorEdgeId::new(hidden_edge.clone()))
                .with_input(TensorEdgeId::new("weight.final_norm"))
                .with_output(TensorEdgeId::new("hidden.final"))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
        )
        .with_edge(qwen_lm_head_weight_edge(config))
        .with_edge(f32_edge("logits", vec![sequence_length, a.vocabulary_size]))
        .with_node(
            op_node("lm_head", "matmul", OperatorFamily::LinearAlgebra)
                .with_input(TensorEdgeId::new("hidden.final"))
                .with_input(TensorEdgeId::new("weight.lm_head"))
                .with_output(TensorEdgeId::new("logits")),
        );

    Ok(graph)
}

/// Produce a validated Qwen prefill Execution Graph for `prompt_length`
/// tokens.
pub fn qwen_prefill_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    prompt_length: u64,
    kv_cache_enabled: bool,
) -> Result<GraphProductionResult, QwenComponentError> {
    let graph = qwen_build_graph(
        config,
        identity,
        ExecutionGraphPhase::Prefill,
        prompt_length,
        kv_cache_enabled,
        // A prefill starts the sequence, so its first row is position zero.
        0,
    )?;
    validate_first_scope_graph(&graph)?;
    GraphProductionResult::validated(graph, &identity.id, &default_graph_catalog())
        .map_err(QwenComponentError::from)
}

/// Produce a validated Qwen decode Execution Graph for a single new token,
/// consuming prior KV cache.
///
/// `cached_token_count` is how many tokens the KV cache already holds, which
/// is the absolute position of the token being generated. It is required
/// rather than defaulted: a decode graph built as if the new token were at
/// position zero produces wrong rotations for every token after the first, and
/// does so silently.
pub fn qwen_decode_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    cached_token_count: u64,
) -> Result<GraphProductionResult, QwenComponentError> {
    let graph = qwen_build_graph(
        config,
        identity,
        ExecutionGraphPhase::Decode,
        1,
        true,
        cached_token_count,
    )?;
    validate_first_scope_graph(&graph)?;
    GraphProductionResult::validated(graph, &identity.id, &default_graph_catalog())
        .map_err(QwenComponentError::from)
}

// ---------------------------------------------------------------------------
// Target module detail (layer selector, adapter methods, graph insertion
// point)
// ---------------------------------------------------------------------------

/// Where in the Qwen decoder layer graph a target module's Operator lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGraphInsertionPoint {
    Embedding,
    AttentionProjection,
    AttentionOutput,
    MlpProjection,
    LogitsProjection,
}

/// A Qwen target module together with the adapter-specific metadata the
/// canonical [`TargetModuleMetadata`] does not itself carry: which layers it
/// applies to, which adapter methods it supports, and where its Operator
/// sits in the decoder layer graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenTargetModuleDetail {
    pub module: TargetModuleMetadata,
    pub layer_selector: AdapterLayerSelector,
    pub supported_adapter_methods: BTreeSet<AdapterMethod>,
    pub graph_insertion_point: QwenGraphInsertionPoint,
}

/// Target module details for every canonical Qwen target module, given the
/// baseline's layer count.
pub fn qwen_target_module_details(layer_count: u64) -> Vec<QwenTargetModuleDetail> {
    let per_layer_selector = AdapterLayerSelector::RangeInclusive {
        start: 0,
        end: layer_count.saturating_sub(1) as u32,
    };
    let methods = BTreeSet::from([AdapterMethod::Lora, AdapterMethod::Qlora]);
    QWEN_TARGET_MODULE_ROLES
        .into_iter()
        .map(|role| {
            let (layer_selector, graph_insertion_point) = match role {
                TargetModuleRole::Embedding => (
                    AdapterLayerSelector::All,
                    QwenGraphInsertionPoint::Embedding,
                ),
                TargetModuleRole::LmHead => (
                    AdapterLayerSelector::All,
                    QwenGraphInsertionPoint::LogitsProjection,
                ),
                TargetModuleRole::QProj | TargetModuleRole::KProj | TargetModuleRole::VProj => (
                    per_layer_selector.clone(),
                    QwenGraphInsertionPoint::AttentionProjection,
                ),
                TargetModuleRole::OProj => (
                    per_layer_selector.clone(),
                    QwenGraphInsertionPoint::AttentionOutput,
                ),
                _ => (
                    per_layer_selector.clone(),
                    QwenGraphInsertionPoint::MlpProjection,
                ),
            };
            QwenTargetModuleDetail {
                module: TargetModuleMetadata::canonical(role),
                layer_selector,
                supported_adapter_methods: methods.clone(),
                graph_insertion_point,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Adapter graph support and tensor shape validation
// ---------------------------------------------------------------------------

/// Whether the Qwen baseline can apply adapter graph modifications. The
/// first baseline supports neither overlay nor merge graphs, so Runtime
/// SHALL reject adapter activation rather than silently running an
/// adapter-free graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenAdapterGraphSupport {
    pub overlay_supported: bool,
    pub merge_supported: bool,
}

impl QwenAdapterGraphSupport {
    pub const fn baseline() -> Self {
        Self {
            overlay_supported: false,
            merge_supported: false,
        }
    }
}

/// Reject adapter activation when neither overlay nor merge graph support is
/// declared.
pub fn qwen_validate_adapter_activation_supported(
    support: QwenAdapterGraphSupport,
) -> Result<(), QwenComponentError> {
    if !support.overlay_supported && !support.merge_supported {
        return Err(QwenComponentError::AdapterUnsupported);
    }
    Ok(())
}

/// Validate adapter target tensor shapes against Qwen architecture metadata.
pub fn qwen_validate_adapter_target_shapes(
    config: &QwenConfig,
    targets: &[AdapterTargetModule],
) -> Result<(), QwenComponentError> {
    let a = &config.architecture;
    let q_dim = a.attention_head_count * a.head_dimension;
    let kv_dim = a.kv_head_count * a.head_dimension;
    for target in targets {
        let expected = match target.role {
            AdapterTargetModuleRole::QueryProjection => Some(vec![a.hidden_size, q_dim]),
            AdapterTargetModuleRole::KeyProjection | AdapterTargetModuleRole::ValueProjection => {
                Some(vec![a.hidden_size, kv_dim])
            }
            AdapterTargetModuleRole::OutputProjection => Some(vec![q_dim, a.hidden_size]),
            AdapterTargetModuleRole::GateProjection | AdapterTargetModuleRole::UpProjection => {
                Some(vec![a.hidden_size, a.intermediate_size])
            }
            AdapterTargetModuleRole::DownProjection => {
                Some(vec![a.intermediate_size, a.hidden_size])
            }
            AdapterTargetModuleRole::Embedding => Some(vec![a.vocabulary_size, a.hidden_size]),
            AdapterTargetModuleRole::LanguageModelHead => {
                Some(vec![a.hidden_size, a.vocabulary_size])
            }
            AdapterTargetModuleRole::Other => None,
        };
        if let Some(expected) = expected
            && target.expected_shape != expected
        {
            return Err(QwenComponentError::TensorShapeMismatch {
                tensor: target.name.clone(),
                reason: format!(
                    "expected adapter target shape {expected:?}, got {:?}",
                    target.expected_shape
                ),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor layout/dtype scope and explicit conversion
// ---------------------------------------------------------------------------

/// Validate that a [`TensorDescriptor`] uses a required-now dtype and layout
/// (portable f32/i32-class dtype, contiguous layout). Unsupported dtypes or
/// layouts SHALL fail explicitly rather than execute silently.
pub fn qwen_validate_tensor_scope(descriptor: &TensorDescriptor) -> Result<(), QwenComponentError> {
    let DTypeDescriptor::Portable(dtype) = &descriptor.dtype else {
        return Err(QwenComponentError::DTypeUnsupported);
    };
    crate::validate_first_scope_dtype(*dtype).map_err(|_| QwenComponentError::DTypeUnsupported)?;
    let layout = crate::layout_kind(&descriptor.layout);
    crate::validate_first_scope_layout(layout)
        .map_err(|_| QwenComponentError::LayoutUnsupported)?;
    Ok(())
}

/// Append an explicit `dtype-conversion` node converting `source` into a new
/// f32 edge, rather than silently reinterpreting an unsupported dtype.
pub fn qwen_insert_dtype_conversion(
    graph: ExecutionGraph,
    node_id: impl Into<String>,
    source: TensorEdgeId,
    target_edge: TensorEdge,
) -> ExecutionGraph {
    let target_id = target_edge.id.clone();
    graph.with_edge(target_edge).with_node(
        op_node(node_id, "dtype-conversion", OperatorFamily::Tensor)
            .with_input(source)
            .with_output(target_id)
            .with_attribute(
                "dtype",
                OperatorAttributeValue::DType(ComputeDType::Float32),
            ),
    )
}

/// Append an explicit `layout-conversion` node converting `source` into a new
/// contiguous edge, rather than silently reinterpreting an unsupported
/// layout.
pub fn qwen_insert_layout_conversion(
    graph: ExecutionGraph,
    node_id: impl Into<String>,
    source: TensorEdgeId,
    target_edge: TensorEdge,
) -> ExecutionGraph {
    let target_id = target_edge.id.clone();
    graph.with_edge(target_edge).with_node(
        op_node(node_id, "layout-conversion", OperatorFamily::Layout)
            .with_input(source)
            .with_output(target_id)
            .with_attribute(
                "layout",
                OperatorAttributeValue::Layout(TensorLayoutKind::Contiguous),
            ),
    )
}

// ---------------------------------------------------------------------------
// Model Loading / Model Instance / KV Cache / Prefix Cache integration
// ---------------------------------------------------------------------------

/// Build the [`crate::ModelArchitectureImplementation`] Model Loading needs
/// to resolve and register a Qwen Component: an opaque architecture selector
/// derived from the Qwen Component's identity and version (family `qwen`,
/// identifier carrying component id and version), leaving Runtime trust,
/// Memory Manager admission, and residency planning untouched.
pub fn qwen_architecture_implementation(
    identity: &ModelComponentIdentity,
    kind: crate::ModelArchitectureImplementationKind,
) -> crate::ModelArchitectureImplementation {
    crate::ModelArchitectureImplementation {
        architecture: crate::ModelArchitecture::new(
            QWEN_ARCHITECTURE_FAMILY,
            qwen_component_compatibility_key(identity),
        ),
        kind,
        required_capabilities: Vec::new(),
    }
}

/// A deterministic fingerprint of Qwen architecture/config fields, distinct
/// from Component identity/version, suitable for Prefix Cache and KV Cache
/// compatibility metadata.
pub fn qwen_config_fingerprint(config: &QwenConfig) -> String {
    let a = &config.architecture;
    format!(
        "h{}-l{}-a{}-kv{}-d{}-i{}-v{}-c{}-rope{}",
        a.hidden_size,
        a.layer_count,
        a.attention_head_count,
        a.kv_head_count,
        a.head_dimension,
        a.intermediate_size,
        a.vocabulary_size,
        a.context_length,
        config.rope.dimension
    )
}

/// Build [`crate::KvCacheCompatibility`] carrying Qwen Component identity and
/// version metadata (via [`qwen_component_compatibility_key`]) in its
/// `model_architecture` field, so that Qwen Component changes are visible in
/// KV Cache compatibility metadata even though the base contract does not yet
/// enforce it in `validate_reuse`.
pub fn qwen_kv_cache_compatibility(
    identity: &ModelComponentIdentity,
    model: crate::GenerationModelReference,
    tokenizer: crate::TokenizerId,
) -> crate::KvCacheCompatibility {
    let mut compatibility = crate::KvCacheCompatibility::new(model, tokenizer);
    compatibility.model_architecture = Some(qwen_component_compatibility_key(identity));
    compatibility
}

/// Build [`crate::PrefixCacheCompatibility`] whose `model_revision` folds in
/// Qwen Component identity/version and the architecture config fingerprint
/// plus active adapter set, whose `position_encoding` carries RoPE metadata,
/// and whose `attention_implementation` carries the attention variant — so
/// `validate_reuse` rejects cross-version, cross-config, cross-adapter-set,
/// or cross-RoPE prefix reuse.
pub fn qwen_prefix_cache_compatibility(
    identity: &ModelComponentIdentity,
    config: &QwenConfig,
    model: crate::GenerationModelReference,
    tokenizer: crate::TokenizerId,
    tokenizer_revision: Option<String>,
    adapter_set: Option<&crate::AdapterSetId>,
) -> crate::PrefixCacheCompatibility {
    let mut compatibility = crate::PrefixCacheCompatibility::new(model, tokenizer);
    let adapter_tag = adapter_set
        .map(|set| set.as_str().to_string())
        .unwrap_or_else(|| "none".into());
    compatibility.model_revision = Some(format!(
        "{}|cfg={}|adapter={adapter_tag}",
        qwen_component_compatibility_key(identity),
        qwen_config_fingerprint(config)
    ));
    compatibility.tokenizer_revision = tokenizer_revision;
    compatibility.position_encoding = Some(format!(
        "rope:base={}:dim={}",
        config.rope.base, config.rope.dimension
    ));
    compatibility.attention_implementation = Some(qwen_attention_implementation_tag(config));
    compatibility
}

fn qwen_attention_implementation_tag(config: &QwenConfig) -> String {
    match config.architecture.attention {
        crate::AttentionVariant::MultiHead => "multi-head".into(),
        crate::AttentionVariant::MultiQuery => "multi-query".into(),
        crate::AttentionVariant::GroupedQuery => "grouped-query".into(),
    }
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

fn qwen_tagged_observation(
    kind: ModelComponentObservationKind,
    component: &ModelComponentId,
    tag: &'static str,
) -> ModelComponentObservation {
    let mut observation = ModelComponentObservation::new(kind, Some(component.clone()));
    observation
        .redacted_metadata
        .insert("qwen-event".into(), tag.into());
    observation
}

pub fn qwen_observation_component_resolved(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::Registered,
        component,
        "component-resolved",
    )
}

pub fn qwen_observation_component_validated(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::Validated,
        component,
        "component-validated",
    )
}

pub fn qwen_observation_component_rejected(
    component: &ModelComponentId,
    reason: impl Into<String>,
) -> ModelComponentObservation {
    let mut observation = qwen_tagged_observation(
        ModelComponentObservationKind::Rejected,
        component,
        "component-rejected",
    );
    observation
        .redacted_metadata
        .insert("reason".into(), reason.into());
    observation
}

pub fn qwen_observation_config_validated(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::ArchitectureCompatibilityChecked,
        component,
        "config-validated",
    )
}

pub fn qwen_observation_tensor_inventory_checked(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::ArchitectureCompatibilityChecked,
        component,
        "tensor-inventory-checked",
    )
}

pub fn qwen_observation_target_modules_exposed(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::TargetModulesExposed,
        component,
        "target-modules-exposed",
    )
}

pub fn qwen_observation_tokenizer_compatibility_checked(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::ArchitectureCompatibilityChecked,
        component,
        "tokenizer-compatibility-checked",
    )
}

pub fn qwen_observation_kv_metadata_produced(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::KvCacheMetadataExposed,
        component,
        "kv-metadata-produced",
    )
}

pub fn qwen_observation_prefill_graph_produced(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::GraphProduced,
        component,
        "prefill-graph-produced",
    )
}

pub fn qwen_observation_decode_graph_produced(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::GraphProduced,
        component,
        "decode-graph-produced",
    )
}

pub fn qwen_observation_graph_validation_failed(
    component: &ModelComponentId,
    reason: impl Into<String>,
) -> ModelComponentObservation {
    let mut observation = qwen_tagged_observation(
        ModelComponentObservationKind::GraphProductionFailed,
        component,
        "graph-validation-failed",
    );
    observation
        .redacted_metadata
        .insert("reason".into(), reason.into());
    observation
}

pub fn qwen_observation_required_operator_missing(
    component: &ModelComponentId,
    operator: &str,
) -> ModelComponentObservation {
    let mut observation = qwen_tagged_observation(
        ModelComponentObservationKind::GraphProductionFailed,
        component,
        "required-operator-missing",
    );
    observation
        .redacted_metadata
        .insert("operator".into(), operator.into());
    observation
}

pub fn qwen_observation_reference_cpu_coverage_missing(
    component: &ModelComponentId,
) -> ModelComponentObservation {
    qwen_tagged_observation(
        ModelComponentObservationKind::Rejected,
        component,
        "reference-cpu-coverage-missing",
    )
}

pub fn qwen_observation_authority_denied(
    component: &ModelComponentId,
    authority: &str,
) -> ModelComponentObservation {
    let mut observation = qwen_tagged_observation(
        ModelComponentObservationKind::AuthorityDenied,
        component,
        "authority-denied",
    );
    observation
        .redacted_metadata
        .insert("authority".into(), authority.into());
    observation
}

pub fn qwen_observation_conformance_result(
    component: &ModelComponentId,
    passed: bool,
) -> ModelComponentObservation {
    let mut observation = qwen_tagged_observation(
        ModelComponentObservationKind::ConformanceResult,
        component,
        "conformance-result",
    );
    observation
        .redacted_metadata
        .insert("passed".into(), passed.to_string());
    observation
}

// ---------------------------------------------------------------------------
// Conformance report
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenConformanceCheck {
    pub name: &'static str,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenConformanceReport {
    pub checks: Vec<QwenConformanceCheck>,
}

impl QwenConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}

/// Run the Qwen baseline's runnable conformance fixtures against `config`
/// and `identity`, producing a report. See [`qwen_conformance_fixture_names`]
/// for the full named fixture set this baseline SHALL cover; some fixtures
/// (e.g. authority denial) are covered by dedicated unit tests rather than
/// this data-driven report.
pub fn qwen_conformance_report(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
) -> QwenConformanceReport {
    let mut checks = Vec::new();

    let valid = config.validate(identity);
    checks.push(QwenConformanceCheck {
        name: "valid-minimal-config",
        passed: valid.is_ok(),
        detail: valid.err().map(|error| error.to_string()),
    });

    let mut invalid_family_architecture = config.architecture.clone();
    invalid_family_architecture.family = "not-qwen".into();
    let invalid_family_result =
        QwenConfig::new(invalid_family_architecture, config.rope.clone()).validate(identity);
    checks.push(QwenConformanceCheck {
        name: "invalid-architecture-family",
        passed: matches!(
            invalid_family_result,
            Err(QwenComponentError::ArchitectureUnsupported)
        ),
        detail: invalid_family_result.err().map(|error| error.to_string()),
    });

    let scope_result =
        validate_model_component_first_scope_requirements(&qwen_operator_requirements())
            .map_err(QwenComponentError::from);
    checks.push(QwenConformanceCheck {
        name: "required-operator-scope-validation",
        passed: scope_result.is_ok(),
        detail: scope_result.err().map(|error| error.to_string()),
    });

    let prefill_result = qwen_prefill_graph(config, identity, 4, true);
    checks.push(QwenConformanceCheck {
        name: "prefill-graph-production",
        passed: prefill_result.is_ok(),
        detail: prefill_result.err().map(|error| error.to_string()),
    });

    // Decode the 5th token, i.e. against the 4 tokens the prefill above cached,
    // so the check exercises a non-zero position rather than the degenerate
    // first-token case.
    let decode_result = qwen_decode_graph(config, identity, 4);
    checks.push(QwenConformanceCheck {
        name: "decode-graph-production",
        passed: decode_result.is_ok(),
        detail: decode_result.err().map(|error| error.to_string()),
    });

    checks.push(QwenConformanceCheck {
        name: "unsupported-quantization-rejection",
        passed: qwen_quantization_compatibility()
            .supported_methods
            .is_empty(),
        detail: None,
    });

    checks.push(QwenConformanceCheck {
        name: "authority-denial",
        passed: crate::validate_model_component_authority(["network"]).is_err()
            && !qwen_authority().is_empty(),
        detail: None,
    });

    checks.push(QwenConformanceCheck {
        name: "target-module-exposure",
        passed: qwen_target_modules().len() == QWEN_TARGET_MODULE_ROLES.len(),
        detail: None,
    });

    QwenConformanceReport { checks }
}

#[cfg(test)]
mod tests;
