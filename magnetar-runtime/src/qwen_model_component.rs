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
//!
//! Tachyon integration audit MAG-01 (#82): [`FirstNativeModelConfig`] and the
//! validation/descriptor functions `first_native_runtime.rs`'s own generic
//! production path (`production_model_fixture`) calls -- previously named
//! `QwenConfig`, `qwen_component_descriptor`, `qwen_validate_model_artifact`,
//! etc. -- were renamed off Qwen branding once investigation confirmed they
//! encode this baseline's actual decoder-block shape (pre-norm, RoPE,
//! grouped-query attention, SwiGLU MLP), not anything unique to real Qwen
//! checkpoints specifically (a real, independently-compiled Llama Component
//! exercises this exact same shape -- see `build_first_native_graphs_from_
//! named_component_serves_a_real_second_architecture_family`). The rest of
//! this module (adapter/LoRA compatibility, conformance fixture identity,
//! observation tagging, browser-support/prefix-cache/KV-cache compatibility
//! helpers) stays Qwen-named: none of it sits on the generic production
//! path, and it remains this baseline's own legitimate territory.

use crate::{
    ActivationKind, AdapterArchitectureCompatibility, AdapterLayerSelector, AdapterMethod,
    AdapterTargetModule, AdapterTargetModuleRole, ComputeDType, DTypeDescriptor, ExecutionGraph,
    ExecutionGraphPhase, ExecutionNode, ExecutionNodeId, FirstScopeError, FirstScopeErrorCode,
    GraphError, MODEL_ARTIFACT_SCHEMA_VERSION, ModelComponentArchitectureMetadata,
    ModelComponentAuthority, ModelComponentDescriptor, ModelComponentError, ModelComponentId,
    ModelComponentIdentity, ModelComponentImplementationKind, ModelComponentKvCacheMetadata,
    ModelComponentModelType, ModelComponentObservation, ModelComponentObservationKind,
    ModelComponentQuantizationCompatibility, ModelComponentTokenizerCompatibility,
    ModelComponentVersion, ModelDType, ModelGenerationDefaults, ModelManifest, ModelTensorMetadata,
    NormalizationKind, OperatorAttributeValue, OperatorFamily, OperatorId, OperatorRequirement,
    PositionEncodingKind, TargetModuleMetadata, TargetModuleRole, TensorDescriptor, TensorEdge,
    TensorEdgeId, TensorLayoutKind, browser_feature_supported, reference_cpu_kernel_advertisements,
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
            ModelComponentError::ArtifactFormatUnsupported => Self::ComponentInvalid {
                reason: "artifact format unsupported".into(),
            },
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
pub enum FirstNativeRopePositionMode {
    Sequential,
}

impl FirstNativeRopePositionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
        }
    }
}

/// Explicit Qwen RoPE metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct FirstNativeRopeConfig {
    pub base: f64,
    pub scale: Option<f64>,
    pub dimension: u64,
    pub position_mode: FirstNativeRopePositionMode,
    pub dynamic_scaling_supported: bool,
}

impl FirstNativeRopeConfig {
    pub fn standard(head_dimension: u64) -> Self {
        Self {
            base: 10_000.0,
            scale: None,
            dimension: head_dimension,
            position_mode: FirstNativeRopePositionMode::Sequential,
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
pub struct FirstNativeModelConfig {
    pub architecture: ModelComponentArchitectureMetadata,
    pub rope: FirstNativeRopeConfig,
    pub rmsnorm_epsilon: f32,
    pub tied_embeddings: bool,
    /// Whether `self_attn.{q,k,v}_proj` carry an additive bias term -- see
    /// [`crate::ModelArchitectureConfig::attention_bias`]'s doc comment.
    pub attention_bias: bool,
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

impl FirstNativeModelConfig {
    pub fn new(
        architecture: ModelComponentArchitectureMetadata,
        rope: FirstNativeRopeConfig,
    ) -> Self {
        Self {
            architecture,
            rope,
            rmsnorm_epsilon: 1e-6,
            tied_embeddings: false,
            attention_bias: false,
            require_bos: false,
            require_pad: false,
            expected_added_tokens: None,
            chat_template_required: false,
        }
    }

    pub fn validate(&self, identity: &ModelComponentIdentity) -> Result<(), QwenComponentError> {
        // astorise/Magnetar#83: this used to hardcode a family check against
        // QWEN_ARCHITECTURE_FAMILY here, ignoring `identity` entirely --
        // strictly more restrictive than (and redundant with)
        // `self.architecture.validate(identity)` below, which already
        // performs the correct, identity-driven check
        // (`ModelComponentIdentity::supported_architecture_families`, empty
        // meaning no restriction). The hardcoded check made every
        // family-mismatch rejection tautological for callers whose
        // `identity` already restricts to "qwen" (redundant, same outcome)
        // and wrongly rejected every other real family for callers whose
        // `identity` does not (the generic production path's own identity).
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
pub fn first_native_architecture_metadata(
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
pub fn first_native_component_identity(
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
pub fn first_native_target_modules() -> Vec<TargetModuleMetadata> {
    QWEN_TARGET_MODULE_ROLES
        .into_iter()
        .map(TargetModuleMetadata::canonical)
        .collect()
}

/// Required-now Operator requirements for the Qwen baseline first executable
/// path. See [`crate::first_operator_scope`].
pub fn first_native_operator_requirements() -> Vec<OperatorRequirement> {
    QWEN_REQUIRED_NOW_OPERATORS
        .into_iter()
        .map(|(name, family)| OperatorRequirement::new(OperatorId::magnetar(name, 1, family)))
        .collect()
}

/// Authority the Qwen Model Component may hold. Deliberately excludes
/// filesystem, network, process, shell, secrets, Git, workspace, Provider,
/// Device, and Kernel authority.
pub fn first_native_authority() -> BTreeSet<ModelComponentAuthority> {
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

pub fn first_native_kv_cache_metadata(
    config: &FirstNativeModelConfig,
) -> ModelComponentKvCacheMetadata {
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

pub fn first_native_tokenizer_compatibility(
    config: &FirstNativeModelConfig,
) -> ModelComponentTokenizerCompatibility {
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
pub fn validate_first_native_tokenizer_compatibility(
    config: &FirstNativeModelConfig,
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
pub fn validate_first_native_generation_defaults(
    config: &FirstNativeModelConfig,
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
        validate_first_native_tokenizer_compatibility(config, tokenizer)?;
    }
    Ok(())
}

/// The first Qwen baseline rejects quantized artifacts by declaring no
/// supported quantization methods. A later baseline may add explicit
/// dequantization support here.
pub fn first_native_quantization_compatibility() -> ModelComponentQuantizationCompatibility {
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
pub fn first_native_component_descriptor(
    identity: ModelComponentIdentity,
    config: &FirstNativeModelConfig,
) -> Result<ModelComponentDescriptor, QwenComponentError> {
    config.validate(&identity)?;
    let descriptor = ModelComponentDescriptor {
        identity,
        architecture: config.architecture.clone(),
        target_modules: first_native_target_modules(),
        graph_phases: BTreeSet::from([
            ExecutionGraphPhase::Warmup,
            ExecutionGraphPhase::Prefill,
            ExecutionGraphPhase::Decode,
        ]),
        operator_requirements: first_native_operator_requirements(),
        capability_requirements: Vec::new(),
        authority: first_native_authority(),
        kv_cache: Some(first_native_kv_cache_metadata(config)),
        tokenizer: Some(first_native_tokenizer_compatibility(config)),
        quantization: Some(first_native_quantization_compatibility()),
    };
    descriptor.validate()?;
    validate_model_component_first_scope_requirements(&descriptor.operator_requirements)?;
    Ok(descriptor)
}

/// Expected logical tensor names for the Qwen baseline tensor inventory.
pub fn first_native_expected_tensor_names(
    layer_count: u64,
    tied_embeddings: bool,
) -> BTreeSet<String> {
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
        // Fused gate/up projection (`define-provider-prepared-kernel-
        // execution-contract` task group 3), additional to (not a
        // replacement for) the standalone `gate_proj`/`up_proj` weights
        // above: the checked-in real Qwen Component's own graph (the
        // strict, default production path) still references those two
        // standalone weights unchanged, while this crate's own Rust
        // test-oracle graph (`qwen_build_graph`, exercised only when no
        // strict Component engine is available) uses this fused tensor
        // instead, halved at dispatch time by a genuinely two-output
        // "split" node. Both consumers resolve against this same fixture
        // tensor inventory, so it is a superset, not a per-path fork.
        names.insert(format!("layers.{layer}.mlp.gate_up_proj"));
    }
    names.insert("final_norm".to_string());
    if !tied_embeddings {
        names.insert("lm_head".to_string());
    }
    names
}

/// Validate that every expected logical tensor is present in `tensors`.
pub fn validate_first_native_tensor_inventory(
    config: &FirstNativeModelConfig,
    tensors: &[ModelTensorMetadata],
) -> Result<(), QwenComponentError> {
    let present: BTreeSet<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    for expected in
        first_native_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        // `mlp.gate_up_proj` is `first_native_expected_tensor_names`'s fixture-
        // generation superset (its own doc comment: "additional to, not a
        // replacement for" standalone gate_proj/up_proj) -- the real
        // compiled Qwen Component's graph never references it, only this
        // crate's own Rust test-oracle graph does. Requiring it here would
        // reject every real production checkpoint, which never carries a
        // redundant fused tensor a real Hugging Face export never
        // produces (`implement-production-qwen-model-loading` task group
        // 10, found wiring real ingested data through this validation for
        // the first time).
        if expected.ends_with("mlp.gate_up_proj") {
            continue;
        }
        if !present.contains(expected.as_str()) {
            return Err(QwenComponentError::TensorInventoryMissing { tensor: expected });
        }
    }
    Ok(())
}

/// Expected shape for a Qwen logical tensor name, if the baseline declares
/// one.
pub fn first_native_expected_tensor_shape(
    name: &str,
    config: &FirstNativeModelConfig,
) -> Option<Vec<u64>> {
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
        // Optional: real Qwen2/2.5 checkpoints declare these (an
        // architectural default, not config-driven -- see
        // `ModelArchitectureConfig::attention_bias`'s doc comment); never
        // required by `first_native_expected_tensor_names`, so an untied/no-bias
        // configuration is unaffected. Checked here only when present.
        "self_attn.q_bias" => Some(vec![q_dim]),
        "self_attn.k_bias" | "self_attn.v_bias" => Some(vec![kv_dim]),
        "self_attn.o_proj" => Some(vec![q_dim, a.hidden_size]),
        "mlp.gate_proj" | "mlp.up_proj" => Some(vec![a.hidden_size, a.intermediate_size]),
        // Fused gate/up projection (`define-provider-prepared-kernel-
        // execution-contract` task group 3): additional to (not a
        // replacement for) the standalone shapes above -- see
        // `first_native_expected_tensor_names`'s doc comment for why both exist.
        // Twice as wide as either standalone projection, halved by the
        // "split" node the Rust test-oracle graph inserts below rather
        // than two separate matmuls -- a genuine real-world LLM-serving
        // fusion, and the one graph shape that recipe exercises a
        // two-output Kernel with.
        "mlp.gate_up_proj" => Some(vec![a.hidden_size, 2 * a.intermediate_size]),
        "mlp.down_proj" => Some(vec![a.intermediate_size, a.hidden_size]),
        _ => None,
    }
}

/// Validate declared tensor shapes for every recognized Qwen logical tensor.
pub fn validate_first_native_tensor_shapes(
    config: &FirstNativeModelConfig,
    tensors: &[ModelTensorMetadata],
) -> Result<(), QwenComponentError> {
    for tensor in tensors {
        if let Some(expected) = first_native_expected_tensor_shape(&tensor.name, config)
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
/// check here: `lm_head` shape is covered by [`validate_first_native_tensor_shapes`].
pub fn validate_first_native_tied_embedding_shape(
    config: &FirstNativeModelConfig,
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
pub fn validate_first_native_model_artifact(
    descriptor: &ModelComponentDescriptor,
    config: &FirstNativeModelConfig,
    manifest: &ModelManifest,
) -> Result<(), QwenComponentError> {
    descriptor.validate_model_artifact(manifest)?;
    validate_first_native_tensor_inventory(config, &manifest.tensors)?;
    validate_first_native_tensor_shapes(config, &manifest.tensors)?;
    validate_first_native_tied_embedding_shape(config, &manifest.tensors)?;
    if config.chat_template_required && manifest.chat_template.is_none() {
        return Err(QwenComponentError::ComponentInvalid {
            reason: "chat template metadata required but not present in Model Artifact".into(),
        });
    }
    Ok(())
}

/// Adapter (e.g. LoRA) architecture compatibility metadata for a Qwen model.
pub fn qwen_adapter_architecture_compatibility(
    config: &FirstNativeModelConfig,
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
    config: &FirstNativeModelConfig,
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

/// Shared by both the production `qwen_insert_dtype_conversion`/
/// `qwen_insert_layout_conversion` below and this module's own `#[cfg(test)]`
/// graph-building test oracles (`qwen_model_component::tests`), so it stays
/// in the parent module rather than moving into the test-only sibling file.
fn op_node(id: impl Into<String>, name: &str, family: OperatorFamily) -> ExecutionNode {
    ExecutionNode::new(
        ExecutionNodeId::new(id.into()),
        OperatorId::magnetar(name, 1, family),
    )
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
pub fn qwen_config_fingerprint(config: &FirstNativeModelConfig) -> String {
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
    config: &FirstNativeModelConfig,
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

fn qwen_attention_implementation_tag(config: &FirstNativeModelConfig) -> String {
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

#[cfg(test)]
mod tests;
// Only qwen_prefill_graph/qwen_decode_graph are called from outside this
// module (first_native_runtime.rs's own #[cfg(test)] oracle path, via
// `use crate::qwen_model_component::*;`) -- qwen_build_graph,
// QwenConformanceCheck/QwenConformanceReport, and qwen_conformance_report
// are used only within this module and its own tests submodule, so they
// stay un-re-exported here.
#[cfg(test)]
pub(crate) use tests::{qwen_decode_graph, qwen_prefill_graph};
