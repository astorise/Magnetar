//! Runtime-owned adapter loading contract.
//!
//! Adapters are inference data such as LoRA overlays. They are distinct from
//! base model artifacts, Providers, Devices, kernels, and raw tensor handles.
//! Runtime validates adapter artifacts, plans memory through Memory Manager,
//! and activates loaded adapter residency only through explicit policy.

use crate::{
    CapabilityBinding, ComputeDType, CorrelationId, DeviceBinding, GenerationModelReference,
    InferenceSessionId, MemoryAdmissionDecision, MemoryAdmissionRequest, MemoryAllocationClass,
    MemoryAllocationId, MemoryAllocationLifetime, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryDTypeRelation, MemoryManager, MemoryPlacement, ModelArtifactId, ModelDType, ModelName,
    ModelQuantization, ModelRevision, ModelTensorMetadata, ProviderBinding, ResourceAffinity,
    TokenizerId,
};
use std::{collections::BTreeSet, error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterArtifactId {
    pub name: AdapterName,
    pub revision: AdapterRevision,
    pub digest: AdapterDigest,
}

impl AdapterArtifactId {
    pub fn new(name: AdapterName, revision: AdapterRevision, digest: AdapterDigest) -> Self {
        Self {
            name,
            revision,
            digest,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterResidencyId(String);

impl AdapterResidencyId {
    pub fn new(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into();
        validate_portable_identity(&value, "adapter residency id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AdapterResidencyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterLoadingRequestId(String);

impl AdapterLoadingRequestId {
    pub fn new(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into();
        validate_portable_identity(&value, "adapter loading request id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AdapterLoadingRequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterName(String);

impl AdapterName {
    pub fn new(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into();
        validate_portable_identity(&value, "adapter name")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterRevision(String);

impl AdapterRevision {
    pub fn new(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into();
        validate_portable_identity(&value, "adapter revision")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterDigest {
    pub algorithm: String,
    pub value: String,
}

impl AdapterDigest {
    pub fn parse(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into().to_ascii_lowercase();
        let Some(hex) = value.strip_prefix("sha256:") else {
            return Err(AdapterError::AdapterArtifactInvalid {
                reason: "adapter digest must use sha256".into(),
            });
        };
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AdapterError::AdapterArtifactInvalid {
                reason: "adapter digest must be sha256:<64 lowercase hex chars>".into(),
            });
        }
        Ok(Self {
            algorithm: "sha256".into(),
            value,
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdapterMethod {
    Lora,
    Qlora,
    Ia3,
    PromptTuning,
    PrefixTuning,
    Custom(String),
}

impl AdapterMethod {
    pub fn parse(value: &str) -> Result<Self, AdapterError> {
        match value.to_ascii_lowercase().as_str() {
            "lora" => Ok(Self::Lora),
            "qlora" => Ok(Self::Qlora),
            "ia3" => Ok(Self::Ia3),
            "prompt-tuning" => Ok(Self::PromptTuning),
            "prefix-tuning" => Ok(Self::PrefixTuning),
            value if value.starts_with("custom.") => Ok(Self::Custom(value.into())),
            value => Err(AdapterError::AdapterMethodUnsupported {
                method: value.into(),
            }),
        }
    }

    pub const fn requires_target_modules(&self) -> bool {
        matches!(self, Self::Lora | Self::Qlora | Self::Ia3)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdapterTargetModuleRole {
    QueryProjection,
    KeyProjection,
    ValueProjection,
    OutputProjection,
    GateProjection,
    UpProjection,
    DownProjection,
    Embedding,
    LanguageModelHead,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterTargetModule {
    pub name: String,
    pub role: AdapterTargetModuleRole,
    pub layer_selector: Option<AdapterLayerSelector>,
    pub expected_shape: Vec<u64>,
}

impl AdapterTargetModule {
    pub fn validate(&self) -> Result<(), AdapterError> {
        validate_target_module_name(&self.name)?;
        if self.expected_shape.contains(&0) {
            return Err(AdapterError::TargetTensorMismatch {
                target: self.name.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterLayerSelector {
    All,
    Index(u32),
    RangeInclusive { start: u32, end: u32 },
    Explicit(BTreeSet<u32>),
}

impl AdapterLayerSelector {
    pub fn contains(&self, layer: u32) -> bool {
        match self {
            Self::All => true,
            Self::Index(index) => *index == layer,
            Self::RangeInclusive { start, end } => (*start..=*end).contains(&layer),
            Self::Explicit(layers) => layers.contains(&layer),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterArchitectureCompatibility {
    pub family: String,
    pub implementation: String,
    pub hidden_size: Option<u64>,
    pub layer_count: Option<u32>,
    pub position_encoding: Option<String>,
    pub target_modules: BTreeSet<String>,
    pub supported_storage_dtypes: BTreeSet<ModelDType>,
    pub supported_compute_dtypes: BTreeSet<ComputeDType>,
    pub supported_quantization_formats: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterBaseModelCompatibility {
    pub model_name: ModelName,
    pub model_revision: ModelRevision,
    pub model_artifact: Option<ModelArtifactId>,
    pub tokenizer: Option<TokenizerId>,
    pub architecture: AdapterArchitectureCompatibility,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterArtifact {
    pub id: AdapterArtifactId,
    pub method: AdapterMethod,
    pub base_model: AdapterBaseModelCompatibility,
    pub targets: Vec<AdapterTargetModule>,
    pub storage_dtype: ModelDType,
    pub compute_dtype: Option<ComputeDType>,
    pub rank: Option<u32>,
    pub alpha: Option<u32>,
    pub tensors: Vec<ModelTensorMetadata>,
    pub quantization: Option<ModelQuantization>,
    pub required_capabilities: Vec<CapabilityBinding>,
    pub license: Option<crate::ModelLicenseMetadata>,
    pub provenance: Option<crate::ModelProvenance>,
    pub trust: AdapterTrustStatus,
}

impl AdapterArtifact {
    pub fn validate(&self) -> Result<(), AdapterError> {
        if matches!(
            self.trust,
            AdapterTrustStatus::Untrusted | AdapterTrustStatus::Denied
        ) {
            return Err(AdapterError::AdapterArtifactUntrusted);
        }
        if self.trust == AdapterTrustStatus::Revoked {
            return Err(AdapterError::AdapterArtifactRevoked);
        }
        if self.method.requires_target_modules() && self.targets.is_empty() {
            return Err(AdapterError::TargetModuleMissing {
                target: "adapter target modules".into(),
            });
        }
        if matches!(self.method, AdapterMethod::Lora | AdapterMethod::Qlora)
            && self.rank.unwrap_or(0) == 0
        {
            return Err(AdapterError::AdapterRankUnsupported { rank: 0 });
        }
        for target in &self.targets {
            target.validate()?;
        }
        for tensor in &self.tensors {
            if tensor.shape.contains(&0) {
                return Err(AdapterError::AdapterShapeMismatch {
                    tensor: tensor.name.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn compatibility_key(&self) -> AdapterSetId {
        AdapterSetId::from_adapters([self.id.clone()])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterTrustStatus {
    Unknown,
    Trusted,
    Untrusted,
    Denied,
    Revoked,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterSetId(String);

impl AdapterSetId {
    pub fn empty() -> Self {
        Self("adapter-set:none".into())
    }

    pub fn from_adapters(adapters: impl IntoIterator<Item = AdapterArtifactId>) -> Self {
        let mut values = adapters
            .into_iter()
            .map(|adapter| adapter.digest.value)
            .collect::<Vec<_>>();
        values.sort();
        if values.is_empty() {
            return Self::empty();
        }
        Self(format!("adapter-set:{}", values.join("+")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterLoadingRequest {
    pub request_id: AdapterLoadingRequestId,
    pub artifact: AdapterArtifact,
    pub base_model: GenerationModelReference,
    pub target_usage: AdapterTargetUsage,
    pub requested_compute_dtype: Option<ComputeDType>,
    pub residency_policy: AdapterResidencyPolicy,
    pub activation_policy: AdapterActivationPolicy,
    pub merge_policy: AdapterMergePolicy,
    pub memory_budget_bytes: Option<u64>,
    pub required_capabilities: Vec<CapabilityBinding>,
    pub session: Option<InferenceSessionId>,
    pub priority: u8,
    pub timeout_millis: Option<u64>,
    pub correlation_id: Option<CorrelationId>,
}

impl AdapterLoadingRequest {
    pub fn validate(&self) -> Result<(), AdapterError> {
        self.artifact.validate()?;
        if let Some(limit) = self.memory_budget_bytes
            && limit == 0
        {
            return Err(AdapterError::MemoryFeasibilityFailed {
                reason: "adapter memory budget must be greater than zero".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterTargetUsage {
    Generation,
    Embeddings,
    Classification,
    RuntimePolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterResidencyPolicy {
    PreferHost,
    PreferPinnedHost,
    PreferDevice,
    PreferUnifiedShared,
    ProviderOwnedOpaqueAllowed,
    BrowserLinearMemoryOnly,
    PolicyControlled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterActivationPolicy {
    LoadOnly,
    ActivateOnRequest,
    ActivateForSession,
    ActivateForModelInstance,
    RuntimeDefault,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdapterMergePolicy {
    Overlay,
    MergeOnLoad,
    MergeOnActivation,
    ProviderFused,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterCompositionPolicy {
    SingleAdapterOnly,
    RejectMultipleAdapters,
    MultipleAdaptersOrdered,
    WeightedAdapterComposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterSharingPolicy {
    PrivateToSession,
    ShareWithinRuntime,
    PolicyControlled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterSessionPolicy {
    pub allowed_adapters: Option<BTreeSet<AdapterArtifactId>>,
    pub max_active_adapters: usize,
    pub default_adapter: Option<AdapterArtifactId>,
    pub activation_allowed: bool,
    pub deactivation_allowed: bool,
    pub merge_allowed: bool,
    pub adapter_memory_budget_bytes: Option<u64>,
    pub sharing: AdapterSharingPolicy,
    pub unload_on_session_close: bool,
}

impl Default for AdapterSessionPolicy {
    fn default() -> Self {
        Self {
            allowed_adapters: None,
            max_active_adapters: 1,
            default_adapter: None,
            activation_allowed: false,
            deactivation_allowed: true,
            merge_allowed: false,
            adapter_memory_budget_bytes: None,
            sharing: AdapterSharingPolicy::PrivateToSession,
            unload_on_session_close: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterLifecycleState {
    Requested,
    Validating,
    Planning,
    Allocating,
    Materializing,
    Ready,
    Active,
    Inactive,
    Merging,
    Merged,
    Unmerging,
    Draining,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
}

impl AdapterLifecycleState {
    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Requested, Self::Validating)
                | (Self::Requested, Self::Failed)
                | (Self::Validating, Self::Planning)
                | (Self::Validating, Self::Invalid)
                | (Self::Validating, Self::Failed)
                | (Self::Planning, Self::Allocating)
                | (Self::Planning, Self::Failed)
                | (Self::Allocating, Self::Materializing)
                | (Self::Allocating, Self::Failed)
                | (Self::Materializing, Self::Ready)
                | (Self::Materializing, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Inactive)
                | (Self::Ready, Self::Merging)
                | (Self::Ready, Self::Draining)
                | (Self::Active, Self::Inactive)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Merging)
                | (Self::Inactive, Self::Active)
                | (Self::Inactive, Self::Unloading)
                | (Self::Merging, Self::Merged)
                | (Self::Merging, Self::Failed)
                | (Self::Merged, Self::Unmerging)
                | (Self::Merged, Self::Draining)
                | (Self::Unmerging, Self::Ready)
                | (Self::Unmerging, Self::Failed)
                | (Self::Draining, Self::Unloading)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Unloading, Self::Failed)
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterResidencyLocation {
    Host,
    PinnedHost,
    Device(DeviceBinding),
    UnifiedShared,
    ProviderOwnedOpaque(ProviderBinding),
    BrowserLinearMemory,
    FutureWebGpuBuffer(String),
    Sharded(Vec<AdapterResidencyLocation>),
    Mixed(Vec<AdapterResidencyLocation>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAdapterResource {
    pub provider: ProviderBinding,
    pub handle_kind: String,
    pub release_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterResidency {
    pub id: AdapterResidencyId,
    pub artifact: AdapterArtifactId,
    pub lifecycle: AdapterLifecycleState,
    pub location: AdapterResidencyLocation,
    pub affinity: Option<ResourceAffinity>,
    pub memory_allocation: Option<MemoryAllocationId>,
    pub provider_resource: Option<ProviderAdapterResource>,
}

impl AdapterResidency {
    pub fn transition_to(&mut self, next: AdapterLifecycleState) -> Result<(), AdapterError> {
        if self.lifecycle.allows_transition_to(next) {
            self.lifecycle = next;
            Ok(())
        } else {
            Err(AdapterError::InvalidLifecycleTransition {
                from: self.lifecycle,
                to: next,
            })
        }
    }

    pub fn provider_handle_exposed(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterMemoryEstimate {
    pub tensor_bytes: u64,
    pub quantized_storage_bytes: u64,
    pub compute_materialization_bytes: u64,
    pub transform_workspace_bytes: u64,
    pub merge_workspace_bytes: u64,
    pub unmerge_workspace_bytes: u64,
    pub transfer_staging_bytes: u64,
    pub pinned_memory_bytes: u64,
    pub placement: MemoryPlacement,
    pub affinity: Option<ResourceAffinity>,
    pub queue_allowed: bool,
}

impl AdapterMemoryEstimate {
    pub fn total_bytes(&self) -> Result<u64, AdapterError> {
        [
            self.tensor_bytes,
            self.quantized_storage_bytes,
            self.compute_materialization_bytes,
            self.transform_workspace_bytes,
            self.merge_workspace_bytes,
            self.unmerge_workspace_bytes,
            self.transfer_staging_bytes,
            self.pinned_memory_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, value| {
            total
                .checked_add(value)
                .ok_or(AdapterError::InternalAdapter {
                    reason: "adapter memory estimate overflowed".into(),
                })
        })
    }

    pub fn allocation_request(
        &self,
        request: &AdapterLoadingRequest,
    ) -> Result<MemoryAllocationRequest, AdapterError> {
        let mut allocation = MemoryAllocationRequest::new(
            MemoryAllocationClass::AdapterArtifact,
            self.total_bytes()?,
            self.placement.clone(),
            request
                .session
                .as_ref()
                .map(|session| MemoryAllocationOwner::Session(session.as_str().into()))
                .unwrap_or_else(|| {
                    MemoryAllocationOwner::InferenceArtifact(
                        request.artifact.id.digest.value.clone(),
                    )
                }),
        )
        .with_priority(request.priority);
        allocation.lifetime = if request.session.is_some() {
            MemoryAllocationLifetime::Session
        } else {
            MemoryAllocationLifetime::Runtime
        };
        if let Some(affinity) = &self.affinity {
            allocation = allocation.with_affinity(affinity.clone());
        }
        if let Some(dtype) = request.requested_compute_dtype {
            allocation = allocation.with_dtype_relation(MemoryDTypeRelation::new(
                request.artifact.storage_dtype.descriptor(),
                crate::DTypeDescriptor::portable(dtype),
            ));
        }
        if let Some(deadline) = request.timeout_millis {
            allocation = allocation.with_deadline_millis(deadline);
        }
        Ok(allocation)
    }

    pub fn admission(
        &self,
        request: &AdapterLoadingRequest,
        memory: &MemoryManager,
    ) -> Result<MemoryAdmissionDecision, AdapterError> {
        Ok(memory.admit(MemoryAdmissionRequest {
            allocation: self.allocation_request(request)?,
            pressure: memory.pressure_snapshot(),
            queue_allowed: self.queue_allowed,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterActivationRequest {
    pub residency: AdapterResidencyId,
    pub scope: AdapterActivationScope,
    pub base_model: GenerationModelReference,
    pub adapter_set: AdapterSetId,
    pub policy: AdapterCompositionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterActivationScope {
    Operation(String),
    Session(InferenceSessionId),
    ModelInstance(String),
    Runtime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterDeactivationRequest {
    pub residency: AdapterResidencyId,
    pub scope: AdapterActivationScope,
    pub release_residency: bool,
    pub invalidate_cache_state: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterMergeRecord {
    pub source_adapter: AdapterArtifactId,
    pub affected_base_residency: String,
    pub reversible: bool,
    pub new_residency_state: AdapterLifecycleState,
    pub invalidated_kv_caches: Vec<crate::KvCacheId>,
    pub invalidated_prefix_entries: Vec<crate::PrefixCacheEntryId>,
    pub unload_policy: AdapterUnloadPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterUnloadPolicy {
    UnloadAdapterOnly,
    UnmergeBeforeUnload,
    PreserveMergedBaseResidency,
    RejectWhileActive,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterCacheCompatibility {
    pub base_model: GenerationModelReference,
    pub adapter_set: AdapterSetId,
    pub merge_policy: AdapterMergePolicy,
}

impl AdapterCacheCompatibility {
    pub fn validate_reuse(&self, requested: &Self) -> Result<(), AdapterError> {
        if self.base_model != requested.base_model {
            return Err(AdapterError::BaseModelIncompatible);
        }
        if self.adapter_set != requested.adapter_set {
            return Err(AdapterError::KvCacheIncompatible);
        }
        if self.merge_policy != requested.merge_policy {
            return Err(AdapterError::KvCacheIncompatible);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterBatchCompatibility {
    pub base_model: GenerationModelReference,
    pub adapter_set: AdapterSetId,
    pub execution_strategy: AdapterMergePolicy,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub affinity: Option<ResourceAffinity>,
}

impl AdapterBatchCompatibility {
    pub fn validate_with(&self, other: &Self) -> Result<(), AdapterError> {
        if self.base_model != other.base_model || self.adapter_set != other.adapter_set {
            return Err(AdapterError::ActivationConflict);
        }
        if self.execution_strategy != other.execution_strategy {
            return Err(AdapterError::ProviderAdapterUnsupported);
        }
        if self.provider != other.provider {
            return Err(AdapterError::ProviderCapabilityUnavailable);
        }
        if self.device != other.device {
            return Err(AdapterError::DeviceUnavailable);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterGenerationContext {
    pub base_model: GenerationModelReference,
    pub active_adapter_set: AdapterSetId,
    pub activation: Option<AdapterActivationRequest>,
    pub implicit_loading_allowed: bool,
}

impl AdapterGenerationContext {
    pub fn validate(&self) -> Result<(), AdapterError> {
        let Some(activation) = &self.activation else {
            if self.active_adapter_set == AdapterSetId::empty() {
                return Ok(());
            }
            return Err(AdapterError::ActivationDenied);
        };
        if activation.base_model != self.base_model {
            return Err(AdapterError::BaseModelIncompatible);
        }
        activation_uses_adapter(activation, &self.active_adapter_set)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAdapterCapabilities {
    pub supported_methods: BTreeSet<AdapterMethod>,
    pub max_rank: Option<u32>,
    pub supported_storage_dtypes: BTreeSet<ModelDType>,
    pub supported_compute_dtypes: BTreeSet<ComputeDType>,
    pub merge_strategies: BTreeSet<AdapterMergePolicy>,
    pub fused_kernels: BTreeSet<String>,
    pub target_modules: BTreeSet<String>,
    pub quantized_formats: BTreeSet<String>,
    pub activation_cost_millis: Option<u64>,
}

impl ProviderAdapterCapabilities {
    pub fn supports(&self, artifact: &AdapterArtifact, merge: AdapterMergePolicy) -> bool {
        self.supported_methods.contains(&artifact.method)
            && artifact
                .rank
                .is_none_or(|rank| self.max_rank.is_none_or(|max| rank <= max))
            && self
                .supported_storage_dtypes
                .contains(&artifact.storage_dtype)
            && artifact.compute_dtype.is_none_or(|dtype| {
                self.supported_compute_dtypes.is_empty()
                    || self.supported_compute_dtypes.contains(&dtype)
            })
            && self.merge_strategies.contains(&merge)
            && artifact
                .targets
                .iter()
                .all(|target| self.target_modules.contains(&target.name))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterObservationKind {
    LoadingRequested,
    ArtifactValidated,
    CompatibilityChecked,
    ValidationFailed,
    ResidencyPlanningStarted,
    ResidencyPlanningCompleted,
    MemoryAllocationRequested,
    MemoryAllocationQueued,
    MaterializationStarted,
    MaterializationCompleted,
    Ready,
    Activated,
    Deactivated,
    MergeStarted,
    MergeCompleted,
    UnmergeStarted,
    UnmergeCompleted,
    LoadFailed,
    UnloadStarted,
    Unloaded,
    CacheInvalidation,
    BatchingCompatibilityFailed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterObservation {
    pub kind: AdapterObservationKind,
    pub adapter: Option<AdapterArtifactId>,
    pub residency: Option<AdapterResidencyId>,
    pub message: String,
    pub correlation_id: Option<CorrelationId>,
    pub raw_adapter_tensors_available: bool,
    pub raw_model_weights_available: bool,
    pub raw_provider_handle_available: bool,
    pub raw_prompt_available: bool,
}

impl AdapterObservation {
    pub fn redacted(
        kind: AdapterObservationKind,
        adapter: Option<AdapterArtifactId>,
        residency: Option<AdapterResidencyId>,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) -> Self {
        Self {
            kind,
            adapter,
            residency,
            message: message.into(),
            correlation_id,
            raw_adapter_tensors_available: false,
            raw_model_weights_available: false,
            raw_provider_handle_available: false,
            raw_prompt_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterError {
    AdapterArtifactNotFound,
    AdapterArtifactInvalid {
        reason: String,
    },
    AdapterArtifactUntrusted,
    AdapterArtifactRevoked,
    AdapterMethodUnsupported {
        method: String,
    },
    BaseModelIncompatible,
    ArchitectureIncompatible,
    TargetModuleMissing {
        target: String,
    },
    TargetTensorMismatch {
        target: String,
    },
    TokenizerIncompatible,
    StorageDTypeUnsupported {
        dtype: ModelDType,
    },
    ComputeDTypeUnsupported {
        dtype: ComputeDType,
    },
    QuantizationUnsupported,
    AdapterRankUnsupported {
        rank: u32,
    },
    AdapterShapeMismatch {
        tensor: String,
    },
    MemoryFeasibilityFailed {
        reason: String,
    },
    MemoryAllocationFailed {
        reason: String,
    },
    AdapterLoadingQueued,
    AdapterLoadingTimeout,
    ProviderCapabilityUnavailable,
    ProviderAdapterUnsupported,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceMemoryInsufficient,
    ActivationDenied,
    ActivationConflict,
    MultipleAdaptersUnsupported,
    MergeUnsupported,
    MergeFailed,
    UnmergeUnsupported,
    UnmergeFailed,
    KvCacheIncompatible,
    PrefixCacheInvalidated,
    UnloadFailed,
    BrowserFeatureUnsupported,
    InvalidLifecycleTransition {
        from: AdapterLifecycleState,
        to: AdapterLifecycleState,
    },
    InternalAdapter {
        reason: String,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterArtifactNotFound => f.write_str("adapter artifact not found"),
            Self::AdapterArtifactInvalid { reason } => {
                write!(f, "adapter artifact invalid: {reason}")
            }
            Self::AdapterArtifactUntrusted => f.write_str("adapter artifact untrusted"),
            Self::AdapterArtifactRevoked => f.write_str("adapter artifact revoked"),
            Self::AdapterMethodUnsupported { method } => {
                write!(f, "adapter method unsupported: {method}")
            }
            Self::BaseModelIncompatible => f.write_str("base model incompatible"),
            Self::ArchitectureIncompatible => f.write_str("architecture incompatible"),
            Self::TargetModuleMissing { target } => write!(f, "target module missing: {target}"),
            Self::TargetTensorMismatch { target } => write!(f, "target tensor mismatch: {target}"),
            Self::TokenizerIncompatible => f.write_str("tokenizer incompatible"),
            Self::StorageDTypeUnsupported { dtype } => {
                write!(f, "storage dtype unsupported: {dtype:?}")
            }
            Self::ComputeDTypeUnsupported { dtype } => {
                write!(f, "compute dtype unsupported: {dtype:?}")
            }
            Self::QuantizationUnsupported => f.write_str("adapter quantization unsupported"),
            Self::AdapterRankUnsupported { rank } => {
                write!(f, "adapter rank unsupported: {rank}")
            }
            Self::AdapterShapeMismatch { tensor } => {
                write!(f, "adapter shape mismatch: {tensor}")
            }
            Self::MemoryFeasibilityFailed { reason } => {
                write!(f, "adapter memory feasibility failed: {reason}")
            }
            Self::MemoryAllocationFailed { reason } => {
                write!(f, "adapter memory allocation failed: {reason}")
            }
            Self::AdapterLoadingQueued => f.write_str("adapter loading queued"),
            Self::AdapterLoadingTimeout => f.write_str("adapter loading timeout"),
            Self::ProviderCapabilityUnavailable => f.write_str("Provider capability unavailable"),
            Self::ProviderAdapterUnsupported => f.write_str("Provider adapter unsupported"),
            Self::ProviderNotReady => f.write_str("Provider not ready"),
            Self::ProviderSaturated => f.write_str("Provider saturated"),
            Self::DeviceUnavailable => f.write_str("Device unavailable"),
            Self::DeviceMemoryInsufficient => f.write_str("Device memory insufficient"),
            Self::ActivationDenied => f.write_str("adapter activation denied"),
            Self::ActivationConflict => f.write_str("adapter activation conflict"),
            Self::MultipleAdaptersUnsupported => f.write_str("multiple adapters unsupported"),
            Self::MergeUnsupported => f.write_str("adapter merge unsupported"),
            Self::MergeFailed => f.write_str("adapter merge failed"),
            Self::UnmergeUnsupported => f.write_str("adapter unmerge unsupported"),
            Self::UnmergeFailed => f.write_str("adapter unmerge failed"),
            Self::KvCacheIncompatible => f.write_str("KV cache incompatible"),
            Self::PrefixCacheInvalidated => f.write_str("Prefix Cache invalidated"),
            Self::UnloadFailed => f.write_str("adapter unload failed"),
            Self::BrowserFeatureUnsupported => f.write_str("browser adapter feature unsupported"),
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid adapter lifecycle transition from {from:?} to {to:?}"
                )
            }
            Self::InternalAdapter { reason } => write!(f, "internal adapter error: {reason}"),
        }
    }
}

impl Error for AdapterError {}

pub fn validate_adapter_compatibility(
    adapter: &AdapterArtifact,
    base_model: &AdapterBaseModelCompatibility,
    provider: Option<&ProviderAdapterCapabilities>,
) -> Result<(), AdapterError> {
    adapter.validate()?;
    if adapter.base_model.model_name != base_model.model_name
        || adapter.base_model.model_revision != base_model.model_revision
    {
        return Err(AdapterError::BaseModelIncompatible);
    }
    if adapter.base_model.architecture.family != base_model.architecture.family
        || adapter.base_model.architecture.implementation != base_model.architecture.implementation
    {
        return Err(AdapterError::ArchitectureIncompatible);
    }
    if adapter.base_model.architecture.hidden_size != base_model.architecture.hidden_size
        || adapter.base_model.architecture.layer_count != base_model.architecture.layer_count
    {
        return Err(AdapterError::ArchitectureIncompatible);
    }
    if adapter.base_model.tokenizer != base_model.tokenizer {
        return Err(AdapterError::TokenizerIncompatible);
    }
    for target in &adapter.targets {
        if !base_model
            .architecture
            .target_modules
            .contains(&target.name)
        {
            return Err(AdapterError::TargetModuleMissing {
                target: target.name.clone(),
            });
        }
    }
    if !base_model.architecture.supported_storage_dtypes.is_empty()
        && !base_model
            .architecture
            .supported_storage_dtypes
            .contains(&adapter.storage_dtype)
    {
        return Err(AdapterError::StorageDTypeUnsupported {
            dtype: adapter.storage_dtype,
        });
    }
    if let Some(dtype) = adapter.compute_dtype
        && !base_model.architecture.supported_compute_dtypes.is_empty()
        && !base_model
            .architecture
            .supported_compute_dtypes
            .contains(&dtype)
    {
        return Err(AdapterError::ComputeDTypeUnsupported { dtype });
    }
    if adapter.quantization.is_some()
        && base_model
            .architecture
            .supported_quantization_formats
            .is_empty()
    {
        return Err(AdapterError::QuantizationUnsupported);
    }
    if let Some(capabilities) = provider
        && !capabilities.supports(adapter, AdapterMergePolicy::Overlay)
    {
        return Err(AdapterError::ProviderAdapterUnsupported);
    }
    Ok(())
}

pub fn validate_adapter_activation(
    residency: &AdapterResidency,
    request: &AdapterActivationRequest,
    session_policy: Option<&AdapterSessionPolicy>,
    batch: Option<&AdapterBatchCompatibility>,
) -> Result<(), AdapterError> {
    if residency.id != request.residency {
        return Err(AdapterError::ActivationConflict);
    }
    if !matches!(
        residency.lifecycle,
        AdapterLifecycleState::Ready
            | AdapterLifecycleState::Inactive
            | AdapterLifecycleState::Active
            | AdapterLifecycleState::Merged
    ) {
        return Err(AdapterError::ActivationDenied);
    }
    if let Some(policy) = session_policy {
        if !policy.activation_allowed {
            return Err(AdapterError::ActivationDenied);
        }
        let active_count = request.adapter_set.as_str().split('+').count();
        if active_count > policy.max_active_adapters {
            return Err(AdapterError::MultipleAdaptersUnsupported);
        }
        if let Some(allowed) = &policy.allowed_adapters
            && !allowed.contains(&residency.artifact)
        {
            return Err(AdapterError::ActivationDenied);
        }
    }
    if let Some(batch) = batch
        && batch.adapter_set != request.adapter_set
    {
        return Err(AdapterError::ActivationConflict);
    }
    activation_uses_adapter(request, &request.adapter_set)
}

pub fn apply_adapter_deactivation(
    residency: &mut AdapterResidency,
    request: &AdapterDeactivationRequest,
    session_policy: Option<&AdapterSessionPolicy>,
) -> Result<(), AdapterError> {
    if residency.id != request.residency {
        return Err(AdapterError::ActivationConflict);
    }
    if let Some(policy) = session_policy
        && !policy.deactivation_allowed
    {
        return Err(AdapterError::ActivationDenied);
    }
    if request.release_residency {
        if residency.lifecycle == AdapterLifecycleState::Active {
            residency.transition_to(AdapterLifecycleState::Draining)?;
            residency.transition_to(AdapterLifecycleState::Unloading)?;
        } else if residency.lifecycle == AdapterLifecycleState::Inactive {
            residency.transition_to(AdapterLifecycleState::Unloading)?;
        }
        residency.transition_to(AdapterLifecycleState::Unloaded)?;
    } else if residency.lifecycle == AdapterLifecycleState::Active {
        residency.transition_to(AdapterLifecycleState::Inactive)?;
    }
    Ok(())
}

pub fn adapter_memory_feasibility(
    request: &AdapterLoadingRequest,
    estimate: &AdapterMemoryEstimate,
    memory: &MemoryManager,
) -> Result<MemoryAdmissionDecision, AdapterError> {
    request.validate()?;
    estimate.admission(request, memory)
}

pub fn activation_uses_adapter(
    request: &AdapterActivationRequest,
    active_set: &AdapterSetId,
) -> Result<(), AdapterError> {
    if &request.adapter_set != active_set {
        return Err(AdapterError::ActivationConflict);
    }
    match request.policy {
        AdapterCompositionPolicy::SingleAdapterOnly
        | AdapterCompositionPolicy::RejectMultipleAdapters
            if active_set.as_str().contains('+') =>
        {
            Err(AdapterError::MultipleAdaptersUnsupported)
        }
        _ => Ok(()),
    }
}

fn validate_portable_identity(value: &str, label: &str) -> Result<(), AdapterError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.to_ascii_lowercase().contains("provider")
        || value.to_ascii_lowercase().contains("device")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(AdapterError::AdapterArtifactInvalid {
            reason: format!("{label} must use a portable non-Provider, non-Device identifier"),
        });
    }
    Ok(())
}

fn validate_target_module_name(value: &str) -> Result<(), AdapterError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(AdapterError::TargetModuleMissing {
            target: value.into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
