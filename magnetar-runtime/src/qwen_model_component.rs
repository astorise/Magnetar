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
            DTypeDescriptor::portable(ComputeDType::SInt32),
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
) -> Result<ExecutionGraph, QwenComponentError> {
    if sequence_length == 0 {
        return Err(QwenComponentError::GraphProductionFailed {
            reason: "sequence length must be positive".into(),
        });
    }
    let a = &config.architecture;
    let batch = 1u64;
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
        .with_edge(token_id_edge(
            "input.token_ids",
            vec![batch, sequence_length],
        ))
        .with_edge(f32_edge(
            "weight.token_embedding",
            vec![a.vocabulary_size, a.hidden_size],
        ))
        .with_edge(f32_edge(
            "hidden.0",
            vec![batch, sequence_length, a.hidden_size],
        ))
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
                vec![batch, sequence_length, a.hidden_size],
            ))
            .with_edge(f32_edge(
                normed.clone(),
                vec![batch, sequence_length, a.hidden_size],
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
            .with_edge(f32_edge(q.clone(), vec![batch, sequence_length, q_dim]))
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
            .with_edge(f32_edge(k.clone(), vec![batch, sequence_length, kv_dim]))
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
            .with_edge(f32_edge(v.clone(), vec![batch, sequence_length, kv_dim]))
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
            .with_edge(f32_edge(
                q_rope.clone(),
                vec![batch, sequence_length, q_dim],
            ))
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
                ),
            )
            .with_edge(f32_edge(
                k_rope.clone(),
                vec![batch, sequence_length, kv_dim],
            ))
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
            .with_edge(f32_edge(
                attn_out.clone(),
                vec![batch, sequence_length, q_dim],
            ))
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
                vec![batch, sequence_length, a.hidden_size],
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
                vec![batch, sequence_length, a.hidden_size],
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
                vec![batch, sequence_length, a.hidden_size],
            ))
            .with_edge(f32_edge(
                mlp_normed.clone(),
                vec![batch, sequence_length, a.hidden_size],
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
                vec![batch, sequence_length, a.intermediate_size],
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
                vec![batch, sequence_length, a.intermediate_size],
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
                vec![batch, sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.silu"), "silu", OperatorFamily::Activation)
                    .with_input(TensorEdgeId::new(gate.clone()))
                    .with_output(TensorEdgeId::new(activated.clone())),
            )
            .with_edge(f32_edge(
                mlp_hidden.clone(),
                vec![batch, sequence_length, a.intermediate_size],
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
                vec![batch, sequence_length, a.hidden_size],
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
                vec![batch, sequence_length, a.hidden_size],
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
        .with_edge(f32_edge(
            "weight.final_norm",
            vec![batch, sequence_length, a.hidden_size],
        ))
        .with_edge(f32_edge(
            "hidden.final",
            vec![batch, sequence_length, a.hidden_size],
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
        .with_edge(f32_edge(
            "logits",
            vec![batch, sequence_length, a.vocabulary_size],
        ))
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
    )?;
    validate_first_scope_graph(&graph)?;
    GraphProductionResult::validated(graph, &identity.id, &default_graph_catalog())
        .map_err(QwenComponentError::from)
}

/// Produce a validated Qwen decode Execution Graph for a single new token,
/// consuming prior KV cache.
pub fn qwen_decode_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
) -> Result<GraphProductionResult, QwenComponentError> {
    let graph = qwen_build_graph(config, identity, ExecutionGraphPhase::Decode, 1, true)?;
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

    let decode_result = qwen_decode_graph(config, identity);
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
mod tests {
    use super::*;
    use crate::{
        AdapterSetId, FallbackClass, GenerationModelReference, MemoryManager, ModelArchitecture,
        ModelArtifactId, ModelArtifactKind, ModelArtifactSource, ModelDigest,
        ModelInstanceDefinition, ModelLoadingCoordinator, ModelLoadingRequest,
        ModelLoadingRequestId, ModelName, ModelQuantizationPolicy, ModelRevision,
        ModelTrustDecision, ModelTrustStatus, ResourceAffinity, TokenizerId,
    };
    use std::collections::BTreeMap;

    fn small_architecture() -> ModelComponentArchitectureMetadata {
        qwen_architecture_metadata(8, 2, 2, 2, 4, 16, 32, 64)
    }

    fn small_config() -> QwenConfig {
        QwenConfig::new(small_architecture(), QwenRopeConfig::standard(4))
    }

    fn small_identity() -> ModelComponentIdentity {
        qwen_component_identity(
            ModelComponentId::new("qwen-baseline").unwrap(),
            ModelComponentVersion::new(1, 0, 0),
            ModelComponentImplementationKind::RuntimeNative,
        )
    }

    #[test]
    fn valid_minimal_config_builds_descriptor() {
        let descriptor = qwen_component_descriptor(small_identity(), &small_config()).unwrap();
        assert_eq!(descriptor.architecture.family, QWEN_ARCHITECTURE_FAMILY);
        assert_eq!(
            descriptor.target_modules.len(),
            QWEN_TARGET_MODULE_ROLES.len()
        );
    }

    #[test]
    fn invalid_architecture_family_is_rejected() {
        let mut architecture = small_architecture();
        architecture.family = "llama".into();
        let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
        assert_eq!(
            config.validate(&small_identity()),
            Err(QwenComponentError::ArchitectureUnsupported)
        );
    }

    #[test]
    fn invalid_hidden_head_configuration_is_rejected() {
        let mut architecture = small_architecture();
        architecture.hidden_size = 9; // not attention_head_count * head_dimension
        let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
        assert!(matches!(
            config.validate(&small_identity()),
            Err(QwenComponentError::ConfigInvalid {
                field: "head dimension",
                ..
            })
        ));
    }

    #[test]
    fn missing_tensor_inventory_is_detected() {
        let config = small_config();
        let tensors = vec![ModelTensorMetadata {
            name: "token_embedding".into(),
            shape: vec![32, 8],
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
        }];
        assert!(matches!(
            qwen_validate_tensor_inventory(&config, &tensors),
            Err(QwenComponentError::TensorInventoryMissing { .. })
        ));
    }

    #[test]
    fn invalid_tensor_shape_is_detected() {
        let config = small_config();
        let tensor = ModelTensorMetadata {
            name: "lm_head".into(),
            shape: vec![1, 1],
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
        };
        assert!(matches!(
            qwen_validate_tensor_shapes(&config, std::slice::from_ref(&tensor)),
            Err(QwenComponentError::TensorShapeMismatch { .. })
        ));
    }

    #[test]
    fn target_modules_are_exposed() {
        let modules = qwen_target_modules();
        assert!(
            modules
                .iter()
                .any(|module| module.role == TargetModuleRole::QProj)
        );
        assert!(
            modules
                .iter()
                .any(|module| module.role == TargetModuleRole::LmHead)
        );
    }

    #[test]
    fn prefill_graph_production_succeeds() {
        let config = small_config();
        let identity = small_identity();
        let result = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
        assert_eq!(result.graph.phase, ExecutionGraphPhase::Prefill);
        assert!(result.validation.is_some());
    }

    #[test]
    fn decode_graph_production_includes_kv_cache_append() {
        let config = small_config();
        let identity = small_identity();
        let result = qwen_decode_graph(&config, &identity).unwrap();
        assert_eq!(result.graph.phase, ExecutionGraphPhase::Decode);
        let appended = result
            .graph
            .edges
            .values()
            .any(|edge| matches!(&edge.kv_cache, Some(metadata) if metadata.behavior == GraphKvCacheBehavior::Append));
        assert!(appended);
    }

    #[test]
    fn required_operator_scope_validation_passes_for_baseline_requirements() {
        validate_model_component_first_scope_requirements(&qwen_operator_requirements()).unwrap();
    }

    #[test]
    fn out_of_scope_operator_is_rejected_by_first_scope() {
        let out_of_scope = OperatorRequirement::new(OperatorId::magnetar(
            "flash-attention",
            1,
            OperatorFamily::Attention,
        ));
        assert!(validate_model_component_first_scope_requirements(&[out_of_scope]).is_err());
    }

    #[test]
    fn tokenizer_compatibility_declares_vocabulary_size() {
        let config = small_config();
        let compatibility = qwen_tokenizer_compatibility(&config);
        assert_eq!(
            compatibility.vocabulary_size,
            config.architecture.vocabulary_size
        );
    }

    #[test]
    fn tokenizer_vocabulary_mismatch_is_rejected() {
        let config = small_config();
        let tokenizer = crate::TokenizerMetadata {
            id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
            artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
            digest: ModelDigest::sha256(b"tokenizer"),
            family: crate::TokenizerFamily::new("bpe").unwrap(),
            revision: crate::TokenizerRevision::new("rev1").unwrap(),
            vocabulary_size: config.architecture.vocabulary_size as u32 + 1,
            added_token_count: 0,
            token_id_range: crate::TokenIdRange::new(0, config.architecture.vocabulary_size as u32),
            model_max_length: None,
            special_tokens: Vec::new(),
            additional_special_tokens: Vec::new(),
            byte_fallback: false,
            normalization: None,
            pre_tokenizer: None,
            supports_offsets: false,
            supports_token_type_ids: false,
            supports_browser: true,
        };
        assert_eq!(
            qwen_validate_tokenizer_compatibility(&config, &tokenizer),
            Err(QwenComponentError::TokenizerIncompatible)
        );
    }

    #[test]
    fn generation_defaults_exceeding_context_length_are_rejected() {
        let config = small_config();
        let defaults = crate::ModelGenerationDefaults {
            max_tokens: Some(1_000),
            ..crate::ModelGenerationDefaults::default()
        };
        assert!(matches!(
            qwen_validate_generation_defaults(&config, &defaults, None),
            Err(QwenComponentError::GenerationMetadataInvalid { .. })
        ));
    }

    #[test]
    fn kv_cache_metadata_matches_architecture() {
        let config = small_config();
        let metadata = qwen_kv_cache_metadata(&config);
        assert_eq!(metadata.layer_count, config.architecture.layer_count);
        assert_eq!(metadata.head_dimension, config.architecture.head_dimension);
        assert!(!metadata.paged);
    }

    #[test]
    fn adapter_target_validation_exposes_expected_modules() {
        let config = small_config();
        let compatibility = qwen_adapter_architecture_compatibility(&config, "qwen-baseline");
        assert!(compatibility.target_modules.contains("q_proj"));
        assert_eq!(
            compatibility.hidden_size,
            Some(config.architecture.hidden_size)
        );
    }

    #[test]
    fn unsupported_quantization_is_rejected() {
        let config = small_config();
        let identity = small_identity();
        let descriptor = qwen_component_descriptor(identity, &config).unwrap();
        let mut manifest_architecture =
            ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
        manifest_architecture.required_component_role = None;
        let manifest = ModelManifest {
            schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
            id: ModelArtifactId::new(
                ModelArtifactKind::ModelBundle,
                ModelName::new("qwen-test").unwrap(),
                ModelRevision::new("rev1").unwrap(),
                ModelDigest::sha256(b"qwen-test"),
            ),
            architecture: manifest_architecture,
            parts: BTreeMap::new(),
            storage_dtype: Some(ModelDType::F32),
            compute_dtype: Some(ModelDType::F32),
            supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
            tensors: Vec::new(),
            tokenizer: None,
            tokenizer_config: None,
            chat_template: None,
            prompt_template: None,
            generation: None,
            quantization: Some(crate::ModelQuantization {
                format: crate::ModelQuantizationFormat::GgufQ4K,
                group_size: None,
                block_size: None,
                scale_dtype: None,
                zero_point_dtype: None,
                per_channel: false,
                workspace_bytes: None,
                required_capabilities: Vec::new(),
            }),
            shards: Vec::new(),
            runtime_features: BTreeSet::new(),
            memory_features: BTreeSet::new(),
            provider_capabilities: Vec::new(),
            component: None,
            license: None,
            provenance: None,
            signatures: Vec::new(),
            source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
        };
        assert_eq!(
            qwen_validate_model_artifact(&descriptor, &config, &manifest),
            Err(QwenComponentError::QuantizationUnsupported)
        );
    }

    #[test]
    fn authority_denies_forbidden_authorities() {
        let authority = qwen_authority();
        assert!(authority.contains(&ModelComponentAuthority::ModelArtifactRead));
        assert!(crate::validate_model_component_authority(["network"]).is_err());
        assert!(crate::validate_model_component_authority(["filesystem"]).is_err());
    }

    #[test]
    fn no_provider_device_kernel_handle_exposure() {
        assert_eq!(
            crate::provider_handle_access_error(),
            ModelComponentError::ProviderAccessDenied
        );
        assert_eq!(
            crate::device_handle_access_error(),
            ModelComponentError::DeviceAccessDenied
        );
        assert_eq!(
            crate::kernel_handle_access_error(),
            ModelComponentError::KernelAccessDenied
        );
    }

    #[test]
    fn rope_dynamic_scaling_unsupported_is_rejected() {
        let mut rope = QwenRopeConfig::standard(4);
        rope.scale = Some(2.0);
        assert!(matches!(
            rope.validate(4),
            Err(QwenComponentError::RopeUnsupported { .. })
        ));
    }

    #[test]
    fn reference_cpu_covers_required_now_operators() {
        qwen_validate_reference_cpu_coverage().unwrap();
    }

    #[test]
    fn browser_support_matches_implementation_kind() {
        assert!(qwen_browser_supported(ModelComponentImplementationKind::RuntimeNative).is_ok());
    }

    #[test]
    fn contract_versions_accept_self_and_reject_newer_major() {
        qwen_validate_contract_versions(
            QWEN_TENSOR_CONTRACT_VERSION,
            QWEN_TOKENIZER_CONTRACT_VERSION,
            QWEN_KV_CACHE_CONTRACT_VERSION,
            QWEN_ADAPTER_CONTRACT_VERSION,
        )
        .unwrap();
        let newer_tensor =
            crate::CapabilityVersion::new(QWEN_TENSOR_CONTRACT_VERSION.major + 1, 0, 0);
        assert_eq!(
            qwen_validate_contract_versions(
                newer_tensor,
                QWEN_TOKENIZER_CONTRACT_VERSION,
                QWEN_KV_CACHE_CONTRACT_VERSION,
                QWEN_ADAPTER_CONTRACT_VERSION,
            ),
            Err(QwenComponentError::ComponentUnsupportedVersion)
        );
    }

    #[test]
    fn tied_embedding_shape_must_match_vocabulary_and_hidden_size() {
        let mut config = small_config();
        config.tied_embeddings = true;
        let mut tensors = Vec::new();
        for name in qwen_expected_tensor_names(config.architecture.layer_count, true) {
            let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
            tensors.push(ModelTensorMetadata {
                name,
                shape,
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: None,
                size_bytes: None,
                quantization: None,
                expected_compute_dtype: None,
            });
        }
        assert!(qwen_validate_tied_embedding_shape(&config, &tensors).is_ok());

        for tensor in &mut tensors {
            if tensor.name == "token_embedding" {
                tensor.shape = vec![1, 1];
            }
        }
        assert!(matches!(
            qwen_validate_tied_embedding_shape(&config, &tensors),
            Err(QwenComponentError::TensorShapeMismatch { .. })
        ));
    }

    #[test]
    fn tied_embeddings_alias_lm_head_weight_in_graph() {
        let mut config = small_config();
        config.tied_embeddings = true;
        let identity = small_identity();
        let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
        let lm_head_edge = result
            .graph
            .edges
            .get(&TensorEdgeId::new("weight.lm_head"))
            .unwrap();
        assert_eq!(
            lm_head_edge.aliasing,
            TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"))
        );
    }

    #[test]
    fn untied_lm_head_weight_does_not_alias() {
        let config = small_config();
        let identity = small_identity();
        let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
        let lm_head_edge = result
            .graph
            .edges
            .get(&TensorEdgeId::new("weight.lm_head"))
            .unwrap();
        assert_eq!(lm_head_edge.aliasing, TensorAliasing::None);
    }

    #[test]
    fn target_module_details_expose_layer_selector_and_insertion_point() {
        let details = qwen_target_module_details(4);
        assert_eq!(details.len(), QWEN_TARGET_MODULE_ROLES.len());
        let q_proj = details
            .iter()
            .find(|detail| detail.module.role == TargetModuleRole::QProj)
            .unwrap();
        assert_eq!(
            q_proj.layer_selector,
            AdapterLayerSelector::RangeInclusive { start: 0, end: 3 }
        );
        assert_eq!(
            q_proj.graph_insertion_point,
            QwenGraphInsertionPoint::AttentionProjection
        );
        assert!(
            q_proj
                .supported_adapter_methods
                .contains(&AdapterMethod::Lora)
        );
        let embedding = details
            .iter()
            .find(|detail| detail.module.role == TargetModuleRole::Embedding)
            .unwrap();
        assert_eq!(embedding.layer_selector, AdapterLayerSelector::All);
        assert_eq!(
            embedding.graph_insertion_point,
            QwenGraphInsertionPoint::Embedding
        );
        let lm_head = details
            .iter()
            .find(|detail| detail.module.role == TargetModuleRole::LmHead)
            .unwrap();
        assert_eq!(
            lm_head.graph_insertion_point,
            QwenGraphInsertionPoint::LogitsProjection
        );
    }

    #[test]
    fn tokenizer_missing_required_bos_is_rejected() {
        let mut config = small_config();
        config.require_bos = true;
        let tokenizer = crate::TokenizerMetadata {
            id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
            artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
            digest: ModelDigest::sha256(b"tokenizer"),
            family: crate::TokenizerFamily::new("bpe").unwrap(),
            revision: crate::TokenizerRevision::new("rev1").unwrap(),
            vocabulary_size: config.architecture.vocabulary_size as u32,
            added_token_count: 0,
            token_id_range: crate::TokenIdRange::new(
                0,
                config.architecture.vocabulary_size as u32 - 1,
            ),
            model_max_length: None,
            special_tokens: vec![crate::SpecialToken::new(
                crate::SpecialTokenKind::Eos,
                "<eos>",
                0,
            )],
            additional_special_tokens: Vec::new(),
            byte_fallback: false,
            normalization: None,
            pre_tokenizer: None,
            supports_offsets: false,
            supports_token_type_ids: false,
            supports_browser: true,
        };
        assert_eq!(
            qwen_validate_tokenizer_compatibility(&config, &tokenizer),
            Err(QwenComponentError::TokenizerIncompatible)
        );
    }

    #[test]
    fn chat_template_required_but_missing_is_rejected() {
        let mut config = small_config();
        config.chat_template_required = true;
        let identity = small_identity();
        let descriptor = qwen_component_descriptor(identity, &config).unwrap();
        let manifest_architecture = ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
        let mut tensors = Vec::new();
        for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
            let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
            tensors.push(ModelTensorMetadata {
                name,
                shape,
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: None,
                size_bytes: None,
                quantization: None,
                expected_compute_dtype: None,
            });
        }
        let manifest = ModelManifest {
            schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
            id: ModelArtifactId::new(
                ModelArtifactKind::ModelBundle,
                ModelName::new("qwen-test").unwrap(),
                ModelRevision::new("rev1").unwrap(),
                ModelDigest::sha256(b"qwen-test"),
            ),
            architecture: manifest_architecture,
            parts: BTreeMap::new(),
            storage_dtype: Some(ModelDType::F32),
            compute_dtype: Some(ModelDType::F32),
            supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
            tensors,
            tokenizer: None,
            tokenizer_config: None,
            chat_template: None,
            prompt_template: None,
            generation: None,
            quantization: None,
            shards: Vec::new(),
            runtime_features: BTreeSet::new(),
            memory_features: BTreeSet::new(),
            provider_capabilities: Vec::new(),
            component: None,
            license: None,
            provenance: None,
            signatures: Vec::new(),
            source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
        };
        assert!(matches!(
            qwen_validate_model_artifact(&descriptor, &config, &manifest),
            Err(QwenComponentError::ComponentInvalid { .. })
        ));
    }

    #[test]
    fn generation_defaults_non_authoritative_sampling_values_are_never_rejected() {
        let config = small_config();
        for temperature in [-5.0, 0.0, 1.0, 1_000.0] {
            let defaults = crate::ModelGenerationDefaults {
                temperature: Some(temperature),
                top_p: Some(-1.0),
                top_k: Some(u32::MAX),
                ..crate::ModelGenerationDefaults::default()
            };
            assert!(qwen_validate_generation_defaults(&config, &defaults, None).is_ok());
        }
    }

    #[test]
    fn adapter_activation_rejected_when_baseline_lacks_graph_support() {
        assert_eq!(
            qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport::baseline()),
            Err(QwenComponentError::AdapterUnsupported)
        );
        assert!(
            qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport {
                overlay_supported: true,
                merge_supported: false,
            })
            .is_ok()
        );
    }

    #[test]
    fn adapter_target_shape_mismatch_is_rejected() {
        let config = small_config();
        let target = AdapterTargetModule {
            name: "q_proj".into(),
            role: AdapterTargetModuleRole::QueryProjection,
            layer_selector: None,
            expected_shape: vec![1, 1],
        };
        assert!(matches!(
            qwen_validate_adapter_target_shapes(&config, std::slice::from_ref(&target)),
            Err(QwenComponentError::TensorShapeMismatch { .. })
        ));
        let a = &config.architecture;
        let matching = AdapterTargetModule {
            expected_shape: vec![a.hidden_size, a.attention_head_count * a.head_dimension],
            ..target
        };
        assert!(qwen_validate_adapter_target_shapes(&config, &[matching]).is_ok());
    }

    #[test]
    fn tensor_scope_rejects_placeholder_dtype_and_non_contiguous_layout() {
        let f32_contiguous = TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 1]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        );
        assert!(qwen_validate_tensor_scope(&f32_contiguous).is_ok());

        let float16 = TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 1]),
            DTypeDescriptor::portable(ComputeDType::Float16),
            LayoutDescriptor::Contiguous,
        );
        assert_eq!(
            qwen_validate_tensor_scope(&float16),
            Err(QwenComponentError::DTypeUnsupported)
        );

        let strided = TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 1]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Strided {
                strides_elements: vec![1, 1],
                offset_elements: 0,
            },
        );
        assert_eq!(
            qwen_validate_tensor_scope(&strided),
            Err(QwenComponentError::LayoutUnsupported)
        );
    }

    #[test]
    fn explicit_dtype_and_layout_conversion_nodes_validate() {
        let raw_input = TensorEdge::new(
            TensorEdgeId::new("raw.input"),
            TensorDescriptor::new(
                ShapeDescriptor::new(vec![1, 2]),
                DTypeDescriptor::portable(ComputeDType::Float16),
                LayoutDescriptor::Strided {
                    strides_elements: vec![1, 1],
                    offset_elements: 0,
                },
            ),
        );
        let converted_dtype = f32_edge("converted.dtype", vec![1, 2]);
        let converted_layout = f32_edge("converted.layout", vec![1, 2]);
        let mut graph = ExecutionGraph::new(
            ExecutionGraphId::new("conversion-test"),
            ExecutionGraphPhase::Test,
        )
        .with_edge(raw_input);
        graph = qwen_insert_dtype_conversion(
            graph,
            "dtype-fix",
            TensorEdgeId::new("raw.input"),
            converted_dtype,
        );
        graph = qwen_insert_layout_conversion(
            graph,
            "layout-fix",
            TensorEdgeId::new("converted.dtype"),
            converted_layout,
        );
        graph.validate(&default_graph_catalog()).unwrap();
    }

    #[test]
    fn reference_cpu_rejects_incompatible_kv_head_count() {
        let q = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
        let k = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
        let v = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
        // head_count=2 is not a multiple of kv_head_count=3: unsupported variant.
        assert!(crate::attention(&q, &k, &v, 2, 2, Some(3), None, true).is_err());
        // head_count=2 dividing kv_head_count=2 (standard multi-head) succeeds.
        assert!(crate::attention(&q, &k, &v, 2, 2, Some(2), None, true).is_ok());
    }

    #[test]
    fn kv_cache_compatibility_embeds_component_version() {
        let identity_v1 = small_identity();
        let identity_v2 = qwen_component_identity(
            ModelComponentId::new("qwen-baseline").unwrap(),
            ModelComponentVersion::new(2, 0, 0),
            ModelComponentImplementationKind::RuntimeNative,
        );
        let model = GenerationModelReference::LoadedModelContext("ctx".into());
        let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();
        let compatibility_v1 =
            qwen_kv_cache_compatibility(&identity_v1, model.clone(), tokenizer.clone());
        let compatibility_v2 = qwen_kv_cache_compatibility(&identity_v2, model, tokenizer);
        assert_ne!(
            compatibility_v1.model_architecture,
            compatibility_v2.model_architecture
        );
    }

    #[test]
    fn prefix_cache_compatibility_rejects_cross_version_and_cross_adapter_reuse() {
        let identity_v1 = small_identity();
        let identity_v2 = qwen_component_identity(
            ModelComponentId::new("qwen-baseline").unwrap(),
            ModelComponentVersion::new(2, 0, 0),
            ModelComponentImplementationKind::RuntimeNative,
        );
        let config = small_config();
        let model = GenerationModelReference::LoadedModelContext("ctx".into());
        let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();

        let base = qwen_prefix_cache_compatibility(
            &identity_v1,
            &config,
            model.clone(),
            tokenizer.clone(),
            None,
            None,
        );
        let same = qwen_prefix_cache_compatibility(
            &identity_v1,
            &config,
            model.clone(),
            tokenizer.clone(),
            None,
            None,
        );
        assert!(base.validate_reuse(&same).is_ok());

        let different_version = qwen_prefix_cache_compatibility(
            &identity_v2,
            &config,
            model.clone(),
            tokenizer.clone(),
            None,
            None,
        );
        assert!(base.validate_reuse(&different_version).is_err());

        let adapter_set = AdapterSetId::empty();
        let with_adapter = qwen_prefix_cache_compatibility(
            &identity_v1,
            &config,
            model,
            tokenizer,
            None,
            Some(&adapter_set),
        );
        assert!(base.validate_reuse(&with_adapter).is_err());
    }

    #[test]
    fn observability_functions_preserve_component_identity_and_tag_events() {
        let component = ModelComponentId::new("qwen-baseline").unwrap();
        let observations = vec![
            qwen_observation_component_resolved(&component),
            qwen_observation_component_validated(&component),
            qwen_observation_component_rejected(&component, "unsupported"),
            qwen_observation_config_validated(&component),
            qwen_observation_tensor_inventory_checked(&component),
            qwen_observation_target_modules_exposed(&component),
            qwen_observation_tokenizer_compatibility_checked(&component),
            qwen_observation_kv_metadata_produced(&component),
            qwen_observation_prefill_graph_produced(&component),
            qwen_observation_decode_graph_produced(&component),
            qwen_observation_graph_validation_failed(&component, "bad graph"),
            qwen_observation_required_operator_missing(&component, "flash-attention"),
            qwen_observation_reference_cpu_coverage_missing(&component),
            qwen_observation_authority_denied(&component, "network"),
            qwen_observation_conformance_result(&component, true),
        ];
        assert_eq!(observations.len(), 15);
        for observation in &observations {
            assert_eq!(observation.component, Some(component.clone()));
            assert!(observation.redacted_metadata.contains_key("qwen-event"));
        }
    }

    #[test]
    fn conformance_report_is_conformant_for_valid_config() {
        let report = qwen_conformance_report(&small_config(), &small_identity());
        assert!(report.is_conformant(), "failing checks: {report:?}");
        let names: BTreeSet<&str> = report.checks.iter().map(|check| check.name).collect();
        assert!(names.contains("valid-minimal-config"));
        assert!(names.contains("prefill-graph-production"));
        assert!(names.contains("decode-graph-production"));
    }

    fn integration_config() -> QwenConfig {
        QwenConfig::new(
            qwen_architecture_metadata(4, 1, 2, 2, 2, 8, 16, 32),
            QwenRopeConfig::standard(2),
        )
    }

    fn integration_manifest(architecture: ModelArchitecture) -> ModelManifest {
        let config = integration_config();
        let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001";
        let mut tensor_yaml = String::new();
        for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
            let shape = qwen_expected_tensor_shape(&name, &config).unwrap();
            let shape_text = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            tensor_yaml.push_str(&format!(
                "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n"
            ));
        }
        ModelManifest::from_yaml_str(&format!(
            r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {digest}
model:
  name: qwen-integration-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
artifacts:
  weights:
    kind: model-weights
    digest: {digest}
    size_bytes: 128
  config:
    kind: model-config
    digest: {digest}
    size_bytes: 16
tensors:
{tensor_yaml}"#,
            family = architecture.family,
            identifier = architecture.identifier,
        ))
        .unwrap()
    }

    /// End-to-end integration: a Qwen Component identity/config drives real
    /// Model Loading (trust, precondition validation, memory admission,
    /// architecture resolution, residency planning), Qwen artifact/tensor
    /// validation against the exact loaded manifest, Model Instance creation
    /// referencing the Qwen architecture implementation, and finally Runtime
    /// graph planning/execution of the produced prefill graph. This is the
    /// concrete mechanism by which Model Loading, Model Instance, and (via
    /// the Execution Graph boundary) Generation use the Qwen Component today;
    /// `generation::prefill`/`decode_step` remain pre-existing
    /// architecture-agnostic placeholders pending a future Runtime inference
    /// API change.
    #[test]
    fn qwen_component_integrates_with_model_loading_and_instance_and_graph_execution() {
        let config = integration_config();
        let identity = qwen_component_identity(
            ModelComponentId::new("qwen-integration").unwrap(),
            ModelComponentVersion::new(1, 0, 0),
            ModelComponentImplementationKind::RuntimeNative,
        );
        let architecture_implementation = qwen_architecture_implementation(
            &identity,
            crate::ModelArchitectureImplementationKind::ComponentBased,
        );
        let manifest = integration_manifest(architecture_implementation.architecture.clone());

        let descriptor = qwen_component_descriptor(identity.clone(), &config).unwrap();
        qwen_validate_model_artifact(&descriptor, &config, &manifest).unwrap();
        assert_eq!(qwen_target_modules().len(), QWEN_TARGET_MODULE_ROLES.len());

        let mut coordinator = ModelLoadingCoordinator::new();
        coordinator.register_architecture(architecture_implementation.clone());
        let mut memory = MemoryManager::default();
        let mut request = ModelLoadingRequest::new(
            ModelLoadingRequestId::new("qwen-integration-load"),
            manifest.id.clone(),
        );
        request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
        let loaded = coordinator
            .load(
                request,
                &manifest,
                &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture"),
                &mut memory,
            )
            .unwrap();

        let instance = ModelInstanceDefinition::from_loaded_context(
            &loaded,
            architecture_implementation.clone(),
            ResourceAffinity::new(FallbackClass::Transparent),
        );
        assert_eq!(
            instance.architecture.architecture,
            architecture_implementation.architecture
        );
        assert_eq!(
            instance.architecture.architecture.identifier,
            qwen_component_compatibility_key(&identity)
        );

        let prefill = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
        assert!(prefill.validation.is_some());
        let policy = crate::GraphPlanningPolicy::default();
        let plan =
            crate::plan_execution_graph(&prefill.graph, &default_graph_catalog(), &policy, None)
                .unwrap();
        assert!(!plan.execution_order.is_empty());
        crate::execute_graph_boundary(&prefill.graph, &default_graph_catalog(), &policy).unwrap();

        let kv_cache_compatibility = qwen_kv_cache_compatibility(
            &identity,
            GenerationModelReference::ModelInstance(
                crate::ModelInstanceId::new("qwen-instance").unwrap(),
            ),
            TokenizerId::new("qwen-tokenizer").unwrap(),
        );
        assert_eq!(
            kv_cache_compatibility.model_architecture,
            Some(qwen_component_compatibility_key(&identity))
        );
    }
}
