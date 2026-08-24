//! Model Component contract.
//!
//! A Model Component is the portable architecture layer between Model Artifact
//! data and Runtime-validated Execution Graphs. It may describe architecture
//! metadata, target modules, operator requirements, and graph production
//! behavior, but it never selects Providers or Devices and never receives raw
//! native handles.

use crate::{
    CapabilityId, CapabilityVersion, ComponentManifest, ComponentTrustStatus, ExecutionGraph,
    ExecutionGraphPhase, ExecutionGraphProducer, ExecutionGraphVersion, GraphError,
    GraphValidationReport, ModelArchitecture, ModelManifest, ModelQuantizationFormat,
    OPERATOR_CATALOG_VERSION, OperatorCatalog, OperatorFamily, OperatorId, TokenizerFamily,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const MODEL_COMPONENT_ROLE: &str = "model-component";
pub const MODEL_COMPONENT_CONTRACT_VERSION: CapabilityVersion = CapabilityVersion::new(1, 0, 0);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelComponentId(String);

impl ModelComponentId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelComponentError> {
        let value = value.into();
        validate_portable_identity(&value, "model component id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelComponentVersion(pub CapabilityVersion);

impl ModelComponentVersion {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self(CapabilityVersion::new(major, minor, patch))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentImplementationKind {
    WebAssemblyComponent,
    RuntimeNative,
    TestFixture,
    BrowserCompatible,
    JavaScriptMediated,
}

impl ModelComponentImplementationKind {
    pub const fn browser_compatible(self) -> bool {
        matches!(
            self,
            Self::BrowserCompatible | Self::JavaScriptMediated | Self::TestFixture
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentTrustStatus {
    Unknown,
    Trusted,
    Rejected,
    Quarantined,
    Revoked,
}

impl From<ComponentTrustStatus> for ModelComponentTrustStatus {
    fn from(value: ComponentTrustStatus) -> Self {
        match value {
            ComponentTrustStatus::Unknown => Self::Unknown,
            ComponentTrustStatus::Trusted => Self::Trusted,
            ComponentTrustStatus::Rejected => Self::Rejected,
            ComponentTrustStatus::Quarantined => Self::Quarantined,
            ComponentTrustStatus::Revoked => Self::Revoked,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentSignatureState {
    NotApplicable,
    Unsigned,
    Present,
    Verified,
    Invalid,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelComponentProvenance {
    pub source: Option<String>,
    pub publisher: Option<String>,
    pub build_commit: Option<String>,
    pub build_tool: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentIdentity {
    pub id: ModelComponentId,
    pub version: ModelComponentVersion,
    pub implementation: ModelComponentImplementationKind,
    pub supported_architecture_families: BTreeSet<String>,
    pub supported_architecture_revisions: BTreeSet<String>,
    pub supported_model_artifact_schema_versions: BTreeSet<u64>,
    pub supported_runtime_capability_versions: BTreeMap<CapabilityId, CapabilityVersion>,
    pub operator_catalog_version: CapabilityVersion,
    pub execution_graph_contract_version: ExecutionGraphVersion,
    pub trust: ModelComponentTrustStatus,
    pub provenance: ModelComponentProvenance,
    pub signature: ModelComponentSignatureState,
}

impl ModelComponentIdentity {
    pub fn new(
        id: ModelComponentId,
        version: ModelComponentVersion,
        implementation: ModelComponentImplementationKind,
    ) -> Self {
        Self {
            id,
            version,
            implementation,
            supported_architecture_families: BTreeSet::new(),
            supported_architecture_revisions: BTreeSet::new(),
            supported_model_artifact_schema_versions: BTreeSet::new(),
            supported_runtime_capability_versions: BTreeMap::new(),
            operator_catalog_version: OPERATOR_CATALOG_VERSION,
            execution_graph_contract_version: ExecutionGraphVersion(1),
            trust: ModelComponentTrustStatus::Unknown,
            provenance: ModelComponentProvenance::default(),
            signature: ModelComponentSignatureState::NotApplicable,
        }
    }

    pub fn trusted(mut self) -> Self {
        self.trust = ModelComponentTrustStatus::Trusted;
        self
    }

    pub fn with_architecture_family(mut self, family: impl Into<String>) -> Self {
        self.supported_architecture_families.insert(family.into());
        self
    }

    pub fn with_architecture_revision(mut self, revision: impl Into<String>) -> Self {
        self.supported_architecture_revisions
            .insert(revision.into());
        self
    }

    pub fn with_model_artifact_schema_version(mut self, version: u64) -> Self {
        self.supported_model_artifact_schema_versions
            .insert(version);
        self
    }

    pub fn validate(&self) -> Result<(), ModelComponentError> {
        validate_portable_identity(self.id.as_str(), "model component id")?;
        if self.trust != ModelComponentTrustStatus::Trusted {
            return Err(ModelComponentError::ModelComponentUntrusted);
        }
        if self.version.0.major != MODEL_COMPONENT_CONTRACT_VERSION.major {
            return Err(ModelComponentError::ModelComponentUnsupportedVersion);
        }
        if self.operator_catalog_version.major != OPERATOR_CATALOG_VERSION.major {
            return Err(ModelComponentError::OperatorCatalogIncompatible);
        }
        if self.execution_graph_contract_version.0 == 0 {
            return Err(ModelComponentError::GraphContractIncompatible);
        }
        Ok(())
    }

    pub fn supports_architecture(&self, architecture: &ModelArchitecture) -> bool {
        self.supported_architecture_families.is_empty()
            || self
                .supported_architecture_families
                .contains(&architecture.family)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentModelType {
    CausalLanguageModel,
    EmbeddingModel,
    EncoderDecoder,
    Test,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PositionEncodingKind {
    Learned,
    Rotary,
    Alibi,
    Relative,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NormalizationKind {
    LayerNorm,
    RmsNorm,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ActivationKind {
    Gelu,
    Silu,
    Relu,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AttentionVariant {
    MultiHead,
    MultiQuery,
    GroupedQuery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentArchitectureMetadata {
    pub family: String,
    pub model_type: ModelComponentModelType,
    pub hidden_size: u64,
    pub layer_count: u64,
    pub attention_head_count: u64,
    pub kv_head_count: u64,
    pub head_dimension: u64,
    pub intermediate_size: u64,
    pub vocabulary_size: u64,
    pub context_length: u64,
    pub position_encoding: PositionEncodingKind,
    pub normalization: NormalizationKind,
    pub activation: ActivationKind,
    pub attention: AttentionVariant,
    pub quantization: Option<ModelQuantizationFormat>,
    pub tokenizer_family: Option<TokenizerFamily>,
    pub adapter_target_modules: BTreeSet<TargetModuleRole>,
}

impl ModelComponentArchitectureMetadata {
    pub fn validate(&self, identity: &ModelComponentIdentity) -> Result<(), ModelComponentError> {
        validate_portable_identity(&self.family, "architecture family")?;
        if !identity.supported_architecture_families.is_empty()
            && !identity
                .supported_architecture_families
                .contains(&self.family)
        {
            return Err(ModelComponentError::ArchitectureUnsupported);
        }
        for (value, field) in [
            (self.hidden_size, "hidden size"),
            (self.layer_count, "layer count"),
            (self.attention_head_count, "attention head count"),
            (self.kv_head_count, "KV head count"),
            (self.head_dimension, "head dimension"),
            (self.intermediate_size, "intermediate size"),
            (self.vocabulary_size, "vocabulary size"),
            (self.context_length, "context length"),
        ] {
            if value == 0 {
                return Err(ModelComponentError::ArchitectureMetadataInvalid {
                    field,
                    reason: "must be non-zero".into(),
                });
            }
        }
        if self.attention_head_count < self.kv_head_count
            || !self.attention_head_count.is_multiple_of(self.kv_head_count)
        {
            return Err(ModelComponentError::ArchitectureMetadataInvalid {
                field: "KV head count",
                reason: "must divide attention head count".into(),
            });
        }
        if self.hidden_size != self.attention_head_count * self.head_dimension {
            return Err(ModelComponentError::ArchitectureMetadataInvalid {
                field: "head dimension",
                reason: "hidden size must equal attention heads times head dimension".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TargetModuleRole {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
    LmHead,
    Embedding,
    Norm,
    Attention,
    Mlp,
}

impl TargetModuleRole {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
            Self::LmHead => "lm_head",
            Self::Embedding => "embedding",
            Self::Norm => "norm",
            Self::Attention => "attention",
            Self::Mlp => "mlp",
        }
    }

    pub const fn all() -> [Self; 12] {
        [
            Self::QProj,
            Self::KProj,
            Self::VProj,
            Self::OProj,
            Self::GateProj,
            Self::UpProj,
            Self::DownProj,
            Self::LmHead,
            Self::Embedding,
            Self::Norm,
            Self::Attention,
            Self::Mlp,
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetModuleMetadata {
    pub role: TargetModuleRole,
    pub architecture_name: String,
    pub adapter_compatible: bool,
}

impl TargetModuleMetadata {
    pub fn canonical(role: TargetModuleRole) -> Self {
        Self {
            role,
            architecture_name: role.canonical_name().into(),
            adapter_compatible: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphProductionRequest {
    pub component: ModelComponentId,
    pub phase: ExecutionGraphPhase,
    pub model_artifact_id: String,
    pub adapter_set: Option<String>,
    pub validate_with_runtime: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProductionResult {
    pub graph: ExecutionGraph,
    pub validation: Option<GraphValidationReport>,
}

impl GraphProductionResult {
    pub fn validated(
        mut graph: ExecutionGraph,
        component: &ModelComponentId,
        catalog: &OperatorCatalog,
    ) -> Result<Self, ModelComponentError> {
        graph.producer = ExecutionGraphProducer::ModelComponent {
            component_id: component.as_str().into(),
        };
        let validation = graph
            .validate(catalog)
            .map_err(ModelComponentError::GraphValidationFailed)?;
        Ok(Self {
            graph,
            validation: Some(validation),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorRequirement {
    pub operator: OperatorId,
    pub alternatives: Vec<OperatorId>,
    pub shape_constraints: Vec<String>,
    pub dtype_constraints: Vec<String>,
    pub layout_constraints: Vec<String>,
}

impl OperatorRequirement {
    pub fn new(operator: OperatorId) -> Self {
        Self {
            operator,
            alternatives: Vec::new(),
            shape_constraints: Vec::new(),
            dtype_constraints: Vec::new(),
            layout_constraints: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ModelComponentError> {
        validate_portable_operator(&self.operator)?;
        for alternative in &self.alternatives {
            validate_portable_operator(alternative)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentCapabilityKind {
    ModelMetadataValidation,
    GraphProduction,
    OperatorCatalogRead,
    TensorDescriptorCreation,
    KvCacheMetadata,
    AdapterMetadata,
    TokenizerMetadata,
    GenerationDefaultsValidation,
    Diagnostics,
    ObservabilityEmit,
}

impl ModelComponentCapabilityKind {
    pub const fn authority(self) -> ModelComponentAuthority {
        match self {
            Self::ModelMetadataValidation => ModelComponentAuthority::ModelArtifactRead,
            Self::GraphProduction => ModelComponentAuthority::GraphProduction,
            Self::OperatorCatalogRead => ModelComponentAuthority::OperatorCatalogRead,
            Self::TensorDescriptorCreation => ModelComponentAuthority::ComputeCapability,
            Self::KvCacheMetadata => ModelComponentAuthority::KvCacheAccess,
            Self::AdapterMetadata => ModelComponentAuthority::AdapterArtifactRead,
            Self::TokenizerMetadata => ModelComponentAuthority::TokenizerArtifactRead,
            Self::GenerationDefaultsValidation => ModelComponentAuthority::GenerationCapability,
            Self::Diagnostics => ModelComponentAuthority::RuntimeDiagnostics,
            Self::ObservabilityEmit => ModelComponentAuthority::ObservabilityEmit,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentCapabilityRequirement {
    pub kind: ModelComponentCapabilityKind,
    pub id: CapabilityId,
    pub min_version: CapabilityVersion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentAuthority {
    ModelArtifactRead,
    TokenizerArtifactRead,
    PromptTemplateRead,
    AdapterArtifactRead,
    QuantizationArtifactRead,
    InferenceSessionState,
    GenerationSessionState,
    KvCacheAccess,
    PrefixCacheAccess,
    ComputeCapability,
    GenerationCapability,
    SamplingCapability,
    ObservabilityEmit,
    RuntimeDiagnostics,
    GraphProduction,
    OperatorCatalogRead,
}

impl ModelComponentAuthority {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelArtifactRead => "model-artifact-read",
            Self::TokenizerArtifactRead => "tokenizer-artifact-read",
            Self::PromptTemplateRead => "prompt-template-read",
            Self::AdapterArtifactRead => "adapter-artifact-read",
            Self::QuantizationArtifactRead => "quantization-artifact-read",
            Self::InferenceSessionState => "inference-session-state",
            Self::GenerationSessionState => "generation-session-state",
            Self::KvCacheAccess => "kv-cache-access",
            Self::PrefixCacheAccess => "prefix-cache-access",
            Self::ComputeCapability => "compute-capability",
            Self::GenerationCapability => "generation-capability",
            Self::SamplingCapability => "sampling-capability",
            Self::ObservabilityEmit => "observability-emit",
            Self::RuntimeDiagnostics => "runtime-diagnostics",
            Self::GraphProduction => "graph-production",
            Self::OperatorCatalogRead => "operator-catalog-read",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ModelComponentError> {
        match value {
            "model-artifact-read" => Ok(Self::ModelArtifactRead),
            "tokenizer-artifact-read" => Ok(Self::TokenizerArtifactRead),
            "prompt-template-read" => Ok(Self::PromptTemplateRead),
            "adapter-artifact-read" => Ok(Self::AdapterArtifactRead),
            "quantization-artifact-read" => Ok(Self::QuantizationArtifactRead),
            "inference-session-state" => Ok(Self::InferenceSessionState),
            "generation-session-state" => Ok(Self::GenerationSessionState),
            "kv-cache-access" => Ok(Self::KvCacheAccess),
            "prefix-cache-access" => Ok(Self::PrefixCacheAccess),
            "compute-capability" => Ok(Self::ComputeCapability),
            "generation-capability" => Ok(Self::GenerationCapability),
            "sampling-capability" => Ok(Self::SamplingCapability),
            "observability-emit" => Ok(Self::ObservabilityEmit),
            "runtime-diagnostics" => Ok(Self::RuntimeDiagnostics),
            "graph-production" => Ok(Self::GraphProduction),
            "operator-catalog-read" => Ok(Self::OperatorCatalogRead),
            _ if is_forbidden_authority(value) => Err(ModelComponentError::AuthorityDenied {
                authority: value.into(),
            }),
            _ => Err(ModelComponentError::CapabilityUnavailable {
                capability: value.into(),
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentDescriptor {
    pub identity: ModelComponentIdentity,
    pub architecture: ModelComponentArchitectureMetadata,
    pub target_modules: Vec<TargetModuleMetadata>,
    pub graph_phases: BTreeSet<ExecutionGraphPhase>,
    pub operator_requirements: Vec<OperatorRequirement>,
    pub capability_requirements: Vec<ModelComponentCapabilityRequirement>,
    pub authority: BTreeSet<ModelComponentAuthority>,
    pub kv_cache: Option<ModelComponentKvCacheMetadata>,
    pub tokenizer: Option<ModelComponentTokenizerCompatibility>,
    pub quantization: Option<ModelComponentQuantizationCompatibility>,
}

impl ModelComponentDescriptor {
    pub fn validate(&self) -> Result<(), ModelComponentError> {
        self.identity.validate()?;
        self.architecture.validate(&self.identity)?;
        for module in &self.target_modules {
            if module.architecture_name.trim().is_empty() {
                return Err(ModelComponentError::TargetModuleUnavailable {
                    module: module.role.canonical_name().into(),
                });
            }
        }
        for requirement in &self.operator_requirements {
            requirement.validate()?;
        }
        for capability in &self.capability_requirements {
            if !self.authority.contains(&capability.kind.authority()) {
                return Err(ModelComponentError::CapabilityUnavailable {
                    capability: capability.id.as_str().into(),
                });
            }
        }
        Ok(())
    }

    pub fn validate_model_artifact(
        &self,
        manifest: &ModelManifest,
    ) -> Result<(), ModelComponentError> {
        if !self.identity.supports_architecture(&manifest.architecture) {
            return Err(ModelComponentError::ArchitectureUnsupported);
        }
        if !self
            .identity
            .supported_model_artifact_schema_versions
            .is_empty()
            && !self
                .identity
                .supported_model_artifact_schema_versions
                .contains(&manifest.schema_version)
        {
            return Err(ModelComponentError::ModelArtifactIncompatible);
        }
        if let Some(required) = &manifest.architecture.required_component_role
            && required != MODEL_COMPONENT_ROLE
        {
            return Err(ModelComponentError::ModelComponentInvalid {
                reason: "model artifact requires a different component role".into(),
            });
        }
        if let Some(format) = manifest.quantization.as_ref().map(|q| q.format)
            && self
                .quantization
                .as_ref()
                .is_some_and(|q| !q.supported_methods.contains(&format))
        {
            return Err(ModelComponentError::QuantizationUnsupported);
        }
        Ok(())
    }

    pub fn supports_graph_phase(&self, phase: ExecutionGraphPhase) -> bool {
        self.graph_phases.contains(&phase)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentKvCacheMetadata {
    pub layer_count: u64,
    pub head_count: u64,
    pub kv_head_count: u64,
    pub head_dimension: u64,
    pub cache_dtype: String,
    pub layout_preference: String,
    pub paged: bool,
    pub append_semantics: String,
    pub position_behavior: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentTokenizerCompatibility {
    pub vocabulary_size: u64,
    pub special_tokens: BTreeSet<String>,
    pub family: Option<TokenizerFamily>,
    pub chat_template_required: bool,
    pub added_token_behavior: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentQuantizationCompatibility {
    pub supported_methods: BTreeSet<ModelQuantizationFormat>,
    pub tensor_grouping: Option<String>,
    pub scale_metadata_required: bool,
    pub zero_point_metadata_required: bool,
    pub packed_layout: Option<String>,
    pub dequantization_operators: BTreeSet<OperatorId>,
    pub quantized_operators: BTreeSet<OperatorId>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelComponentObservationKind {
    Registered,
    Validated,
    Rejected,
    ArchitectureCompatibilityChecked,
    ModelConfigValidationFailed,
    GraphProductionRequested,
    GraphProduced,
    GraphProductionFailed,
    TargetModulesExposed,
    AdapterMetadataExposed,
    KvCacheMetadataExposed,
    OperatorRequirementsDeclared,
    AuthorityDenied,
    ComponentToProviderAccessDenied,
    ConformanceResult,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentObservation {
    pub kind: ModelComponentObservationKind,
    pub component: Option<ModelComponentId>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ModelComponentObservation {
    pub fn new(kind: ModelComponentObservationKind, component: Option<ModelComponentId>) -> Self {
        Self {
            kind,
            component,
            redacted_metadata: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentConformanceProfile {
    pub id: String,
    pub requires_architecture_validation: bool,
    pub requires_graph_production: bool,
    pub requires_operator_requirements: bool,
    pub requires_target_modules: bool,
    pub requires_authority_tests: bool,
    pub requires_provider_boundary_tests: bool,
    pub requires_browser_behavior: bool,
}

impl Default for ModelComponentConformanceProfile {
    fn default() -> Self {
        Self {
            id: "model-component-core".into(),
            requires_architecture_validation: true,
            requires_graph_production: true,
            requires_operator_requirements: true,
            requires_target_modules: true,
            requires_authority_tests: true,
            requires_provider_boundary_tests: true,
            requires_browser_behavior: cfg!(target_arch = "wasm32"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ModelComponentError {
    ModelComponentNotFound,
    ModelComponentInvalid { reason: String },
    ModelComponentUntrusted,
    ModelComponentUnsupportedVersion,
    ArchitectureUnsupported,
    ArchitectureMetadataInvalid { field: &'static str, reason: String },
    ModelConfigInvalid { reason: String },
    ModelArtifactIncompatible,
    TokenizerIncompatible,
    OperatorCatalogIncompatible,
    GraphContractIncompatible,
    GraphProductionFailed { reason: String },
    GraphValidationFailed(GraphError),
    TargetModuleUnavailable { module: String },
    AdapterIncompatible,
    KvCacheMetadataInvalid { reason: String },
    QuantizationUnsupported,
    CapabilityUnavailable { capability: String },
    AuthorityDenied { authority: String },
    ProviderAccessDenied,
    DeviceAccessDenied,
    KernelAccessDenied,
    MemoryPointerAccessDenied,
    ProviderOwnedResourceAccessDenied,
    BrowserFeatureUnsupported,
    InternalModelComponent { reason: String },
}

impl fmt::Display for ModelComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelComponentNotFound => write!(f, "model component not found"),
            Self::ModelComponentInvalid { reason } => {
                write!(f, "model component invalid: {reason}")
            }
            Self::ModelComponentUntrusted => write!(f, "model component untrusted"),
            Self::ModelComponentUnsupportedVersion => {
                write!(f, "model component unsupported version")
            }
            Self::ArchitectureUnsupported => write!(f, "architecture unsupported"),
            Self::ArchitectureMetadataInvalid { field, reason } => {
                write!(f, "architecture metadata invalid for {field}: {reason}")
            }
            Self::ModelConfigInvalid { reason } => write!(f, "model config invalid: {reason}"),
            Self::ModelArtifactIncompatible => write!(f, "model artifact incompatible"),
            Self::TokenizerIncompatible => write!(f, "tokenizer incompatible"),
            Self::OperatorCatalogIncompatible => write!(f, "operator catalog incompatible"),
            Self::GraphContractIncompatible => write!(f, "graph contract incompatible"),
            Self::GraphProductionFailed { reason } => {
                write!(f, "graph production failed: {reason}")
            }
            Self::GraphValidationFailed(error) => write!(f, "graph validation failed: {error}"),
            Self::TargetModuleUnavailable { module } => {
                write!(f, "target module unavailable: {module}")
            }
            Self::AdapterIncompatible => write!(f, "adapter incompatible"),
            Self::KvCacheMetadataInvalid { reason } => {
                write!(f, "KV cache metadata invalid: {reason}")
            }
            Self::QuantizationUnsupported => write!(f, "quantization unsupported"),
            Self::CapabilityUnavailable { capability } => {
                write!(f, "capability unavailable: {capability}")
            }
            Self::AuthorityDenied { authority } => write!(f, "authority denied: {authority}"),
            Self::ProviderAccessDenied => write!(f, "Provider access denied"),
            Self::DeviceAccessDenied => write!(f, "Device access denied"),
            Self::KernelAccessDenied => write!(f, "Kernel access denied"),
            Self::MemoryPointerAccessDenied => write!(f, "memory pointer access denied"),
            Self::ProviderOwnedResourceAccessDenied => {
                write!(f, "Provider-owned resource access denied")
            }
            Self::BrowserFeatureUnsupported => write!(f, "browser feature unsupported"),
            Self::InternalModelComponent { reason } => {
                write!(f, "internal model component error: {reason}")
            }
        }
    }
}

impl Error for ModelComponentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::GraphValidationFailed(error) => Some(error),
            _ => None,
        }
    }
}

pub fn validate_model_component_role(
    manifest: &ComponentManifest,
) -> Result<(), ModelComponentError> {
    if manifest.role == MODEL_COMPONENT_ROLE {
        Ok(())
    } else {
        Err(ModelComponentError::ModelComponentInvalid {
            reason: format!("component role must be {MODEL_COMPONENT_ROLE}"),
        })
    }
}

pub fn validate_model_component_authority(
    authorities: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<BTreeSet<ModelComponentAuthority>, ModelComponentError> {
    authorities
        .into_iter()
        .map(|authority| ModelComponentAuthority::parse(authority.as_ref()))
        .collect()
}

pub fn validate_model_component_config_data(value: &str) -> Result<(), ModelComponentError> {
    if value.trim().is_empty() {
        return Err(ModelComponentError::ModelConfigInvalid {
            reason: "config data must not be empty".into(),
        });
    }
    if value.contains("file://") || value.contains("../") || value.contains("..\\") {
        return Err(ModelComponentError::AuthorityDenied {
            authority: "filesystem".into(),
        });
    }
    Ok(())
}

pub fn provider_handle_access_error() -> ModelComponentError {
    ModelComponentError::ProviderAccessDenied
}

pub fn device_handle_access_error() -> ModelComponentError {
    ModelComponentError::DeviceAccessDenied
}

pub fn kernel_handle_access_error() -> ModelComponentError {
    ModelComponentError::KernelAccessDenied
}

pub fn memory_pointer_access_error() -> ModelComponentError {
    ModelComponentError::MemoryPointerAccessDenied
}

pub fn provider_owned_resource_access_error() -> ModelComponentError {
    ModelComponentError::ProviderOwnedResourceAccessDenied
}

pub fn browser_feature_supported(
    implementation: ModelComponentImplementationKind,
) -> Result<(), ModelComponentError> {
    if cfg!(target_arch = "wasm32") && !implementation.browser_compatible() {
        Err(ModelComponentError::BrowserFeatureUnsupported)
    } else {
        Ok(())
    }
}

fn validate_portable_operator(operator: &OperatorId) -> Result<(), ModelComponentError> {
    let namespace = operator.namespace();
    let name = operator.name();
    if namespace != crate::OPERATOR_NAMESPACE
        || name.contains('.')
        || name.starts_with("cuda")
        || name.starts_with("metal")
        || name.starts_with("openvino")
        || name.starts_with("qnn")
    {
        return Err(ModelComponentError::OperatorCatalogIncompatible);
    }
    if operator.version() == 0 || !OperatorFamily::ALL.contains(&operator.family()) {
        return Err(ModelComponentError::OperatorCatalogIncompatible);
    }
    Ok(())
}

fn validate_portable_identity(value: &str, label: &'static str) -> Result<(), ModelComponentError> {
    if value.trim().is_empty() {
        return Err(ModelComponentError::ModelComponentInvalid {
            reason: format!("{label} must not be empty"),
        });
    }
    if value.contains('/') || value.contains('\\') || value.contains(':') {
        return Err(ModelComponentError::ModelComponentInvalid {
            reason: format!("{label} must not be a path, URI, Provider, or Device selector"),
        });
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelComponentError::ModelComponentInvalid {
            reason: format!("{label} must use portable ASCII characters"),
        });
    }
    Ok(())
}

fn is_forbidden_authority(value: &str) -> bool {
    matches!(
        value,
        "filesystem"
            | "network"
            | "env"
            | "environment"
            | "process"
            | "shell"
            | "secret"
            | "secrets"
            | "workspace"
            | "git"
            | "source-control"
            | "tool"
            | "tool-execution"
            | "external-service"
    )
}
