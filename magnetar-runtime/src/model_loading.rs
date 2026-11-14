//! Runtime-owned model loading contract.
//!
//! Model loading is the boundary between a validated Model Artifact and a
//! Runtime-owned loaded model context. It coordinates artifact preconditions,
//! architecture compatibility, Memory Manager feasibility, residency planning,
//! Provider/Device compatibility policy, lifecycle, unload/reload, structured
//! errors, and observations without exposing raw weights or native handles.

use crate::{
    CapabilityBinding, ComputeDType, DeviceBinding, HostTensor, MemoryAllocation,
    MemoryAllocationClass, MemoryAllocationOwner, MemoryAllocationRequest, MemoryAllocationState,
    MemoryManager, MemoryPlacement, ModelArchitecture, ModelArtifactError, ModelArtifactId,
    ModelDType, ModelDigest, ModelManifest, ModelQuantizationFormat, ModelResidencyPlan,
    ModelTensorMetadata, ModelTrustDecision, ModelTrustStatus, ProviderBinding, ResourceAffinity,
};
use std::{collections::BTreeMap, error::Error, fmt};

// The quantization tables, decoders and half-precision conversions live in
// `quantization`; the ones that were public here keep their paths.
pub(crate) use crate::quantization::*;
pub use crate::quantization::{IQ4_NL_VALUES, f16_to_f32, iq_signs, iq4_xs_scales, q3_k_scales};

pub use crate::model::ModelResidencyPlan as ArtifactResidencyPlan;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelLoadingRequestId(String);

impl ModelLoadingRequestId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelResidencyId(u64);

impl ModelResidencyId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ModelResidencyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model-residency:{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelLoadingTargetUsage {
    FullInference,
    MetadataInspection,
    AdapterOverlay,
    Partial { required_parts: Vec<String> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelStorageHandling {
    PreserveStorageDType,
    MaterializeComputeDType,
    LazyMaterialization,
    ProviderOpaque,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelQuantizationPolicy {
    DirectQuantizedExecution,
    DequantizeAtLoad,
    LazyDequantization,
    ProviderSpecificTransform,
    RejectUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelShardingPolicy {
    Sequential,
    Parallel,
    SingleDevice,
    MultiDevicePlaceholder,
    HostLazy,
    RejectUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingResidencyPolicy {
    Resident,
    Cacheable,
    Ephemeral,
    Lazy,
    PartialAllowed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelPlacementPreference {
    RuntimeDefault,
    Host,
    PinnedHost,
    BrowserLinearMemory,
    Affinity(ResourceAffinity),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingCachePolicy {
    KeepResident,
    AllowEviction,
    NoCache,
    InvalidateKvCacheOnUnload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoadingRequest {
    pub id: ModelLoadingRequestId,
    pub artifact: ModelArtifactId,
    pub target_usage: ModelLoadingTargetUsage,
    pub compute_dtype: Option<ModelDType>,
    pub storage_handling: ModelStorageHandling,
    pub quantization_policy: ModelQuantizationPolicy,
    pub sharding_policy: ModelShardingPolicy,
    pub residency_policy: ModelLoadingResidencyPolicy,
    pub memory_budget_bytes: Option<u64>,
    pub placement_preference: ModelPlacementPreference,
    pub required_capabilities: Vec<CapabilityBinding>,
    pub session: Option<String>,
    pub cache_policy: ModelLoadingCachePolicy,
    pub priority: u8,
    pub timeout_millis: Option<u64>,
    pub correlation_id: Option<String>,
}

impl ModelLoadingRequest {
    pub fn new(id: ModelLoadingRequestId, artifact: ModelArtifactId) -> Self {
        Self {
            id,
            artifact,
            target_usage: ModelLoadingTargetUsage::FullInference,
            compute_dtype: None,
            storage_handling: ModelStorageHandling::MaterializeComputeDType,
            quantization_policy: ModelQuantizationPolicy::RejectUnsupported,
            sharding_policy: ModelShardingPolicy::Sequential,
            residency_policy: ModelLoadingResidencyPolicy::Resident,
            memory_budget_bytes: None,
            placement_preference: ModelPlacementPreference::RuntimeDefault,
            required_capabilities: Vec::new(),
            session: None,
            cache_policy: ModelLoadingCachePolicy::AllowEviction,
            priority: 0,
            timeout_millis: None,
            correlation_id: None,
        }
    }

    pub fn placement_input(&self) -> &ModelPlacementPreference {
        &self.placement_preference
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelArchitectureImplementationKind {
    RuntimeNative,
    ComponentBased,
    ProviderAssisted,
    TestFixture,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArchitectureImplementation {
    pub architecture: ModelArchitecture,
    pub kind: ModelArchitectureImplementationKind,
    pub required_capabilities: Vec<CapabilityBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingState {
    Requested,
    Validating,
    Planning,
    Allocating,
    Materializing,
    Ready,
    Active,
    Draining,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
}

impl ModelLoadingState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Requested, Self::Validating)
                | (Self::Validating, Self::Planning)
                | (Self::Validating, Self::Failed)
                | (Self::Validating, Self::Invalid)
                | (Self::Planning, Self::Allocating)
                | (Self::Planning, Self::Failed)
                | (Self::Allocating, Self::Materializing)
                | (Self::Allocating, Self::Failed)
                | (Self::Materializing, Self::Ready)
                | (Self::Materializing, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Unloading)
                | (Self::Active, Self::Draining)
                | (Self::Draining, Self::Unloading)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Failed, Self::Unloading)
                | (Self::Failed, Self::Invalid)
                | (Self::Invalid, Self::Unloading)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingPhase {
    ReadManifest,
    ValidateParts,
    OpenArtifactBytes,
    ValidateShards,
    PlanMemory,
    AllocateHost,
    AllocateDevice,
    MaterializeWeights,
    DequantizeOrTransform,
    TransferToDevice,
    InitializeProviderState,
    ValidateReady,
    PublishModelContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelResidencyLocation {
    Host,
    PinnedHost,
    Device(DeviceBinding),
    UnifiedShared,
    ProviderOwnedOpaque(ProviderBinding),
    BrowserLinearMemory,
    FutureWebGpuBuffer,
    Sharded(Vec<ModelResidencyLocation>),
    Mixed(Vec<ModelResidencyLocation>),
    Pending,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelQuantizationHandling {
    None,
    Direct(ModelQuantizationFormat),
    DequantizeAtLoad(ModelQuantizationFormat),
    Lazy(ModelQuantizationFormat),
    ProviderSpecificTransform(ModelQuantizationFormat),
}

/// Runtime-issued: producible only as part of a `LoadedModelContext`
/// returned by `ModelLoadingCoordinator::load()`. Fields are `pub(crate)`,
/// not `pub` -- an external caller SHALL NOT be able to construct or
/// mutate one directly and pass it off as evidence Model Loading actually
/// ran (`bind-model-loading-evidence-to-validated-artifact`, closing a gap
/// an external audit of PR #36 found: every field here was previously
/// `pub`, with no crate-internal constructor).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoadingResidencyPlan {
    pub(crate) artifact: ModelArtifactId,
    pub(crate) architecture: ModelArchitecture,
    pub(crate) target_compute_dtype: Option<ModelDType>,
    pub(crate) storage_dtype: Option<ModelDType>,
    pub(crate) quantization_handling: ModelQuantizationHandling,
    pub(crate) shard_placement: ModelShardingPolicy,
    pub(crate) memory_placements: Vec<ModelResidencyLocation>,
    pub(crate) provider_binding: Option<ProviderBinding>,
    pub(crate) device_binding: Option<DeviceBinding>,
    pub(crate) required_data_movement: Vec<String>,
    pub(crate) temporary_workspace_bytes: u64,
    pub(crate) expected_resident_bytes: u64,
    pub(crate) loading_phases: Vec<ModelLoadingPhase>,
    pub(crate) fallback_options: Vec<String>,
    pub(crate) unload_policy: ModelUnloadPolicy,
    pub(crate) diagnostics: Vec<String>,
}

impl ModelLoadingResidencyPlan {
    pub fn has_raw_native_handles(&self) -> bool {
        false
    }

    /// How this plan handles quantization. Read-only: see the struct-level
    /// doc comment for why `quantization_handling` is not a public field.
    pub fn quantization_handling(&self) -> &ModelQuantizationHandling {
        &self.quantization_handling
    }

    /// Where this plan places residency. Read-only: see the struct-level
    /// doc comment for why `memory_placements` is not a public field.
    pub fn memory_placements(&self) -> &[ModelResidencyLocation] {
        &self.memory_placements
    }

    /// Human-readable notes about this plan, including whether an F16/BF16
    /// storage-to-F32-compute conversion is in effect (Decision 8: this
    /// conversion is never silent). Read-only: see the struct-level doc
    /// comment for why `diagnostics` is not a public field.
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelUnloadPolicy {
    DrainActiveUse,
    RejectActiveUse,
    ForceInvalidate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelUnloadRequest {
    pub residency: ModelResidencyId,
    pub policy: ModelUnloadPolicy,
    pub invalidate_kv_caches: bool,
    pub correlation_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelReloadRequest {
    pub previous_residency: ModelResidencyId,
    pub request: ModelLoadingRequest,
    pub allow_context_mutation: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingObservationKind {
    ModelLoadingRequested,
    ArtifactPreconditionsChecked,
    LoadingValidationFailed,
    ResidencyPlanningStarted,
    ResidencyPlanningCompleted,
    MemoryAllocationRequested,
    MemoryAllocationQueued,
    MemoryAllocationFailed,
    ShardLoadingStarted,
    ShardLoadingCompleted,
    MaterializationStarted,
    MaterializationCompleted,
    ProviderStateInitialized,
    ModelReady,
    ModelLoadFailed,
    ModelUnloadingStarted,
    ModelUnloaded,
    ModelReloadRequested,
    ModelReloadCompleted,
    ModelResidencyPressure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoadingObservation {
    pub kind: ModelLoadingObservationKind,
    pub request: Option<ModelLoadingRequestId>,
    pub residency: Option<ModelResidencyId>,
    pub phase: Option<ModelLoadingPhase>,
    pub correlation_id: Option<String>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadingErrorCode {
    ModelArtifactNotFound,
    ModelArtifactInvalid,
    ModelArtifactUntrusted,
    ModelArtifactRevoked,
    ArchitectureUnsupported,
    ArchitectureImplementationMissing,
    TokenizerIncompatible,
    RequiredPartMissing,
    ShardMissing,
    ShardDigestMismatch,
    StorageDTypeUnsupported,
    ComputeDTypeUnsupported,
    DTypeConversionUnsupported,
    QuantizationUnsupported,
    QuantizationTransformFailed,
    MemoryFeasibilityFailed,
    MemoryAllocationFailed,
    LoadingQueued,
    LoadingTimeout,
    ProviderCapabilityUnavailable,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceMemoryInsufficient,
    PlacementUnsupported,
    DataMovementUnsupported,
    MaterializationFailed,
    ProviderInitializationFailed,
    UnloadFailed,
    ReloadFailed,
    BrowserFeatureUnsupported,
    InternalLoadingError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoadingError {
    pub code: ModelLoadingErrorCode,
    pub phase: Option<ModelLoadingPhase>,
    pub message: String,
}

impl ModelLoadingError {
    pub fn new(
        code: ModelLoadingErrorCode,
        phase: Option<ModelLoadingPhase>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            phase,
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelLoadingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl Error for ModelLoadingError {}

impl From<ModelArtifactError> for ModelLoadingError {
    fn from(error: ModelArtifactError) -> Self {
        let code = match error {
            ModelArtifactError::UnsupportedManifestVersion { .. }
            | ModelArtifactError::UnsupportedArtifactKind { .. }
            | ModelArtifactError::UnsupportedArtifactFormat { .. }
            | ModelArtifactError::UnsupportedDigestAlgorithm { .. }
            | ModelArtifactError::InvalidDigest { .. }
            | ModelArtifactError::DigestMismatch { .. }
            | ModelArtifactError::InvalidShard { .. }
            | ModelArtifactError::InvalidManifest { .. }
            | ModelArtifactError::InvalidTensorMetadata { .. } => {
                ModelLoadingErrorCode::ModelArtifactInvalid
            }
            ModelArtifactError::MissingRequiredPart { .. } => {
                ModelLoadingErrorCode::RequiredPartMissing
            }
            ModelArtifactError::MissingShard { .. } => ModelLoadingErrorCode::ShardMissing,
            ModelArtifactError::ShardDigestMismatch { .. } => {
                ModelLoadingErrorCode::ShardDigestMismatch
            }
            ModelArtifactError::UnsupportedStorageDType { .. } => {
                ModelLoadingErrorCode::StorageDTypeUnsupported
            }
            ModelArtifactError::UnsupportedComputeDType { .. } => {
                ModelLoadingErrorCode::ComputeDTypeUnsupported
            }
            ModelArtifactError::UnsupportedQuantizationFormat { .. } => {
                ModelLoadingErrorCode::QuantizationUnsupported
            }
            ModelArtifactError::TokenizerReferenceMissing { .. } => {
                ModelLoadingErrorCode::TokenizerIncompatible
            }
            ModelArtifactError::TemplateReferenceMissing { .. } => {
                ModelLoadingErrorCode::RequiredPartMissing
            }
            ModelArtifactError::ProviderSelectionNotAllowed { .. } => {
                ModelLoadingErrorCode::ProviderCapabilityUnavailable
            }
            ModelArtifactError::DeviceSelectionNotAllowed { .. } => {
                ModelLoadingErrorCode::PlacementUnsupported
            }
            ModelArtifactError::TrustRejected { .. } => {
                ModelLoadingErrorCode::ModelArtifactUntrusted
            }
            ModelArtifactError::RevokedArtifact { .. } => {
                ModelLoadingErrorCode::ModelArtifactRevoked
            }
            ModelArtifactError::LicensePolicyDenied { .. } => {
                ModelLoadingErrorCode::ModelArtifactInvalid
            }
            ModelArtifactError::MemoryFeasibilityFailed { .. } => {
                ModelLoadingErrorCode::MemoryFeasibilityFailed
            }
            ModelArtifactError::SourceUnavailable { .. }
            | ModelArtifactError::ManifestMissing { .. } => {
                ModelLoadingErrorCode::ModelArtifactNotFound
            }
            ModelArtifactError::UnsupportedArchitecture { .. } => {
                ModelLoadingErrorCode::ArchitectureUnsupported
            }
            ModelArtifactError::SizeOverflow => ModelLoadingErrorCode::InternalLoadingError,
        };
        Self::new(
            code,
            Some(ModelLoadingPhase::ReadManifest),
            error.to_string(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Runtime-issued: the only way to obtain one is a successful
/// `ModelLoadingCoordinator::load()` call. Fields are `pub(crate)`, not
/// `pub` -- an external caller SHALL NOT be able to construct one directly
/// and pass it to `Runtime::create_model_instance()` claiming Model Loading
/// (and the trust/digest validation it performs) ran when it did not
/// (`bind-model-loading-evidence-to-validated-artifact`, closing a gap an
/// external audit of PR #36 found: every field here was previously `pub`,
/// with no crate-internal constructor). Public read-only accessors are
/// added only if a real external caller needs to inspect a field after
/// loading -- `can_start_inference()` below already covers the one known
/// need.
pub struct LoadedModelContext {
    pub(crate) residency: ModelResidencyId,
    pub(crate) request: ModelLoadingRequestId,
    pub(crate) artifact: ModelArtifactId,
    pub(crate) state: ModelLoadingState,
    pub(crate) plan: ModelLoadingResidencyPlan,
    pub(crate) allocation: Option<MemoryAllocation>,
    pub(crate) partial: bool,
    /// Tensor names the loaded `ModelManifest` declares. Carried through to
    /// `ModelInstanceDefinition::required_weight_names` so weight-readiness
    /// derivation can check the full mandatory inventory is bound, not only
    /// that whatever happens to be bound has residency (Correctif:
    /// Runtime-owned ModelInstance readiness authority, round 3). Empty for
    /// a manifest that declares no tensors.
    pub(crate) required_weight_names: std::collections::BTreeSet<String>,
    /// Per-tensor content digests the loaded `ModelManifest` declares
    /// (only tensors with a declared digest are present as keys). Carried
    /// through to `ModelInstanceDefinition::required_weight_digests` so
    /// weight-materialization can verify staged content against declared
    /// content, not just that a name was bound
    /// (`bind-materialized-weight-content-to-model-artifact-digests`).
    /// Empty for a manifest that declares no per-tensor digests --
    /// permissive, the same "absent means unknown" precedent
    /// `required_weight_names` already established.
    pub(crate) required_weight_digests: std::collections::BTreeMap<String, ModelDigest>,
    /// Declared `(shape, storage_dtype)` per tensor name the loaded
    /// `ModelManifest` declares. Carried through to
    /// `ModelInstanceDefinition::required_weight_shapes` so weight
    /// materialization can reject content whose shape or dtype disagrees
    /// with the artifact's declared metadata even when no content digest
    /// exists for that tensor -- a digest and a shape/dtype check are
    /// independent guards, not substitutes for each other
    /// (`seal-runtime-model-trust-and-provenance-authority`).
    pub(crate) required_weight_shapes: std::collections::BTreeMap<String, (Vec<u64>, ModelDType)>,
}

impl LoadedModelContext {
    /// Current loading state. Read-only: see the struct-level doc comment
    /// for why `state` is not a public field.
    pub const fn state(&self) -> ModelLoadingState {
        self.state
    }

    /// The residency plan Model Loading produced. Read-only: see the
    /// struct-level doc comment for why `plan` is not a public field.
    pub const fn plan(&self) -> &ModelLoadingResidencyPlan {
        &self.plan
    }

    pub const fn can_start_inference(&self) -> bool {
        matches!(
            self.state,
            ModelLoadingState::Ready | ModelLoadingState::Active
        )
    }

    pub fn transition(&mut self, next: ModelLoadingState) -> Result<(), ModelLoadingError> {
        if self.state.can_transition_to(next) {
            self.state = next;
            Ok(())
        } else {
            Err(ModelLoadingError::new(
                ModelLoadingErrorCode::InternalLoadingError,
                None,
                format!(
                    "invalid model loading transition {:?} -> {:?}",
                    self.state, next
                ),
            ))
        }
    }
}

pub struct ModelLoadingCoordinator {
    next_residency_id: u64,
    architecture_implementations: Vec<ModelArchitectureImplementation>,
    observations: Vec<ModelLoadingObservation>,
}

impl Default for ModelLoadingCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelLoadingCoordinator {
    pub fn new() -> Self {
        Self {
            next_residency_id: 1,
            architecture_implementations: Vec::new(),
            observations: Vec::new(),
        }
    }

    pub fn register_architecture(&mut self, implementation: ModelArchitectureImplementation) {
        self.architecture_implementations.push(implementation);
    }

    pub fn observations(&self) -> &[ModelLoadingObservation] {
        &self.observations
    }

    pub fn validate_preconditions(
        &mut self,
        manifest: &ModelManifest,
        trust: &ModelTrustDecision,
    ) -> Result<ModelResidencyPlan, ModelLoadingError> {
        match trust.status() {
            ModelTrustStatus::Trusted => {}
            ModelTrustStatus::Rejected | ModelTrustStatus::PolicyDenied => {
                self.observe(
                    ModelLoadingObservationKind::LoadingValidationFailed,
                    None,
                    None,
                    Some(ModelLoadingPhase::ReadManifest),
                    "model artifact trust rejected",
                );
                return Err(ModelLoadingError::new(
                    ModelLoadingErrorCode::ModelArtifactUntrusted,
                    Some(ModelLoadingPhase::ReadManifest),
                    trust.reason().to_string(),
                ));
            }
            ModelTrustStatus::Revoked => {
                self.observe(
                    ModelLoadingObservationKind::LoadingValidationFailed,
                    None,
                    None,
                    Some(ModelLoadingPhase::ReadManifest),
                    "model artifact revoked",
                );
                return Err(ModelLoadingError::new(
                    ModelLoadingErrorCode::ModelArtifactRevoked,
                    Some(ModelLoadingPhase::ReadManifest),
                    trust.reason().to_string(),
                ));
            }
            ModelTrustStatus::Unknown => {
                return Err(ModelLoadingError::new(
                    ModelLoadingErrorCode::ModelArtifactUntrusted,
                    Some(ModelLoadingPhase::ReadManifest),
                    "model artifact trust status is unknown",
                ));
            }
        }
        manifest.validate().map_err(ModelLoadingError::from)?;
        self.observe(
            ModelLoadingObservationKind::ArtifactPreconditionsChecked,
            None,
            None,
            Some(ModelLoadingPhase::ValidateParts),
            "model artifact preconditions checked",
        );
        manifest.residency_plan().map_err(ModelLoadingError::from)
    }

    pub fn resolve_architecture(
        &self,
        architecture: &ModelArchitecture,
    ) -> Result<&ModelArchitectureImplementation, ModelLoadingError> {
        self.architecture_implementations
            .iter()
            .find(|implementation| implementation.architecture == *architecture)
            .ok_or_else(|| {
                ModelLoadingError::new(
                    ModelLoadingErrorCode::ArchitectureImplementationMissing,
                    Some(ModelLoadingPhase::ValidateParts),
                    format!(
                        "no compatible implementation for architecture {}:{}",
                        architecture.family, architecture.identifier
                    ),
                )
            })
    }

    pub fn plan(
        &mut self,
        request: &ModelLoadingRequest,
        manifest: &ModelManifest,
        artifact_plan: ModelResidencyPlan,
    ) -> Result<ModelLoadingResidencyPlan, ModelLoadingError> {
        self.observe(
            ModelLoadingObservationKind::ResidencyPlanningStarted,
            Some(request.id.clone()),
            None,
            Some(ModelLoadingPhase::PlanMemory),
            "model residency planning started",
        );
        self.resolve_architecture(&manifest.architecture)?;
        let expected_resident_bytes = artifact_plan
            .compute_ready_bytes
            .checked_add(artifact_plan.quantization_workspace_bytes)
            .ok_or_else(|| {
                ModelLoadingError::new(
                    ModelLoadingErrorCode::InternalLoadingError,
                    Some(ModelLoadingPhase::PlanMemory),
                    "model resident byte estimate overflows u64",
                )
            })?;
        if let Some(budget) = request.memory_budget_bytes
            && expected_resident_bytes > budget
        {
            return Err(ModelLoadingError::new(
                ModelLoadingErrorCode::MemoryFeasibilityFailed,
                Some(ModelLoadingPhase::PlanMemory),
                "model residency plan exceeds memory budget",
            ));
        }
        let memory_placements = match &request.placement_preference {
            ModelPlacementPreference::RuntimeDefault | ModelPlacementPreference::Host => {
                vec![ModelResidencyLocation::Host]
            }
            ModelPlacementPreference::PinnedHost => vec![ModelResidencyLocation::PinnedHost],
            ModelPlacementPreference::BrowserLinearMemory => {
                vec![ModelResidencyLocation::BrowserLinearMemory]
            }
            ModelPlacementPreference::Affinity(affinity) => {
                if let Some(device) = affinity.device() {
                    vec![ModelResidencyLocation::Device(device.clone())]
                } else if let Some(provider) = affinity.provider() {
                    vec![ModelResidencyLocation::ProviderOwnedOpaque(
                        provider.clone(),
                    )]
                } else {
                    vec![ModelResidencyLocation::Host]
                }
            }
        };
        let quantization_handling =
            match (manifest.quantization.as_ref(), request.quantization_policy) {
                (None, _) => ModelQuantizationHandling::None,
                (Some(quantization), ModelQuantizationPolicy::DirectQuantizedExecution) => {
                    ModelQuantizationHandling::Direct(quantization.format)
                }
                (Some(quantization), ModelQuantizationPolicy::DequantizeAtLoad) => {
                    ModelQuantizationHandling::DequantizeAtLoad(quantization.format)
                }
                (Some(quantization), ModelQuantizationPolicy::LazyDequantization) => {
                    ModelQuantizationHandling::Lazy(quantization.format)
                }
                (Some(quantization), ModelQuantizationPolicy::ProviderSpecificTransform) => {
                    ModelQuantizationHandling::ProviderSpecificTransform(quantization.format)
                }
                (Some(_), ModelQuantizationPolicy::RejectUnsupported) => {
                    return Err(ModelLoadingError::new(
                        ModelLoadingErrorCode::QuantizationUnsupported,
                        Some(ModelLoadingPhase::PlanMemory),
                        "quantized model requires an explicit quantization loading policy",
                    ));
                }
            };
        let mut diagnostics = vec!["model loading plan is Runtime-owned".to_string()];
        // Decision 8: an F16/BF16 storage-to-F32-compute conversion is
        // never silent -- it is always visible on the residency plan a
        // caller can inspect, in addition to `host_tensors_from_artifact_
        // bytes` itself performing the conversion explicitly rather than
        // reinterpreting bytes.
        if matches!(
            manifest.storage_dtype,
            Some(ModelDType::F16 | ModelDType::Bf16)
        ) {
            diagnostics.push(format!(
                "storage dtype {:?} is explicitly converted to F32 during weight materialization \
                 (native half-precision compute is not required by this Provider path)",
                manifest.storage_dtype
            ));
        }
        let plan = ModelLoadingResidencyPlan {
            artifact: manifest.id.clone(),
            architecture: manifest.architecture.clone(),
            target_compute_dtype: request.compute_dtype.or(manifest.compute_dtype),
            storage_dtype: manifest.storage_dtype,
            quantization_handling,
            shard_placement: request.sharding_policy,
            memory_placements,
            provider_binding: None,
            device_binding: None,
            required_data_movement: data_movement_for(request),
            temporary_workspace_bytes: artifact_plan.quantization_workspace_bytes,
            expected_resident_bytes,
            loading_phases: default_loading_phases(),
            fallback_options: vec!["queue".into(), "retry".into(), "policy-fallback".into()],
            unload_policy: ModelUnloadPolicy::DrainActiveUse,
            diagnostics,
        };
        self.observe(
            ModelLoadingObservationKind::ResidencyPlanningCompleted,
            Some(request.id.clone()),
            None,
            Some(ModelLoadingPhase::PlanMemory),
            "model residency planning completed",
        );
        Ok(plan)
    }

    /// `pub(crate)`, not `pub`: the only Runtime-sealed path to this is
    /// `inference_api::load_model`/`load_model_observed`, which derive
    /// `trust` from the performing `Runtime`'s own sealed
    /// `ModelTrustStore` rather than accepting one as a parameter. A
    /// caller with only `pub` access could otherwise build their own
    /// `ModelTrustStore`, evaluate it (a real, not forged, decision --
    /// `ModelTrustStore::evaluate` stays legitimately public), and call
    /// this directly with a decision no `Runtime` actually endorsed
    /// (`seal-model-loading-and-instance-creation-primitives`).
    pub(crate) fn load(
        &mut self,
        request: ModelLoadingRequest,
        manifest: &ModelManifest,
        trust: &ModelTrustDecision,
        memory: &mut MemoryManager,
    ) -> Result<LoadedModelContext, ModelLoadingError> {
        self.observe(
            ModelLoadingObservationKind::ModelLoadingRequested,
            Some(request.id.clone()),
            None,
            Some(ModelLoadingPhase::ReadManifest),
            "model loading requested",
        );
        let artifact_plan = self.validate_preconditions(manifest, trust)?;
        let plan = self.plan(&request, manifest, artifact_plan)?;
        let placement = memory_placement_for(&request.placement_preference)?;
        let allocation_request = MemoryAllocationRequest::new(
            MemoryAllocationClass::ModelArtifact,
            plan.expected_resident_bytes,
            placement,
            MemoryAllocationOwner::InferenceArtifact(manifest.id.name.as_str().into()),
        )
        .with_alignment(64)
        .with_priority(request.priority);
        self.observe(
            ModelLoadingObservationKind::MemoryAllocationRequested,
            Some(request.id.clone()),
            None,
            Some(ModelLoadingPhase::AllocateHost),
            "model loading memory allocation requested",
        );
        let allocation = match memory.allocate(allocation_request) {
            Ok(allocation) => allocation,
            Err(error) => {
                let code = if error.to_string().contains("pending") {
                    ModelLoadingErrorCode::LoadingQueued
                } else {
                    ModelLoadingErrorCode::MemoryAllocationFailed
                };
                self.observe(
                    if code == ModelLoadingErrorCode::LoadingQueued {
                        ModelLoadingObservationKind::MemoryAllocationQueued
                    } else {
                        ModelLoadingObservationKind::MemoryAllocationFailed
                    },
                    Some(request.id.clone()),
                    None,
                    Some(ModelLoadingPhase::AllocateHost),
                    error.to_string(),
                );
                return Err(ModelLoadingError::new(
                    code,
                    Some(ModelLoadingPhase::AllocateHost),
                    error.to_string(),
                ));
            }
        };
        let residency = ModelResidencyId::new(self.next_residency_id);
        self.next_residency_id = self.next_residency_id.saturating_add(1);
        self.observe(
            ModelLoadingObservationKind::MaterializationStarted,
            Some(request.id.clone()),
            Some(residency),
            Some(ModelLoadingPhase::MaterializeWeights),
            "model materialization started",
        );
        self.observe(
            ModelLoadingObservationKind::MaterializationCompleted,
            Some(request.id.clone()),
            Some(residency),
            Some(ModelLoadingPhase::MaterializeWeights),
            "model materialization completed",
        );
        self.observe(
            ModelLoadingObservationKind::ModelReady,
            Some(request.id.clone()),
            Some(residency),
            Some(ModelLoadingPhase::PublishModelContext),
            "model ready",
        );
        Ok(LoadedModelContext {
            residency,
            request: request.id,
            artifact: manifest.id.clone(),
            state: ModelLoadingState::Ready,
            plan,
            allocation: Some(allocation),
            partial: matches!(
                request.residency_policy,
                ModelLoadingResidencyPolicy::PartialAllowed
            ),
            required_weight_names: manifest
                .tensors
                .iter()
                .map(|tensor| tensor.name.clone())
                .collect(),
            // An F32 tensor stored transposed is verified against its
            // storage bytes before its transpose (`tensor_from_raw_bytes`),
            // like F16/BF16, never as the transposed content staged.
            required_weight_digests: manifest
                .tensors
                .iter()
                .filter(|tensor| {
                    !(tensor.storage_dtype == ModelDType::F32
                        && tensor.layout.as_deref()
                            == Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT))
                })
                .filter_map(|tensor| {
                    tensor
                        .digest
                        .clone()
                        .map(|digest| (tensor.name.clone(), digest))
                })
                .collect(),
            required_weight_shapes: manifest
                .tensors
                .iter()
                .map(|tensor| {
                    (
                        tensor.name.clone(),
                        (tensor.shape.clone(), tensor.storage_dtype),
                    )
                })
                .collect(),
        })
    }
}

/// Materializes a `BTreeMap<String, HostTensor>` from a generic Model
/// Artifact tensor inventory (`ModelTensorMetadata`, produced by any format
/// parser -- GGUF, Safetensors, or a future format) plus the raw bytes those
/// tensors' `offset_bytes`/`size_bytes` index into, relative to
/// `data_section_start` (the byte offset, within `bytes`, where the file's
/// tensor-data section actually begins -- e.g. `8 + header_length` for
/// Safetensors, or the aligned post-tensor-info-section offset for GGUF).
/// `ModelTensorMetadata.offset_bytes` is deliberately relative to that
/// section, not to the start of the whole file, matching every format
/// parser's own construction (`formats/safetensors::parse`/
/// `formats/gguf::parse` both set it this way); the caller, having already
/// parsed the header, is the one who knows where the data section starts --
/// this generic bridge does not (it never parses a header at all, see
/// below), so it must be told explicitly rather than assuming zero.
///
/// **Real bug found and fixed while implementing this Change's own parity
/// test**, not designed in from the start: an earlier version of this
/// function treated `offset_bytes` as an absolute file offset, silently
/// reading garbage (the tail of the JSON header, reinterpreted as `f32`)
/// for every tensor after the first. Caught by
/// `materialize-weights-from-real-model-artifact`'s own parity test
/// (`e2e_fixture_real_artifact_weights_match_in_memory_weights`) actually
/// failing with denormal/huge float values, not assumed correct because the
/// code compiled and the shapes matched.
///
/// This is the missing "read the actual tensor bytes" step a format
/// parser's own `parse()` deliberately leaves to its caller (see
/// `implement-model-format-parsers`'s design.md): it depends only on
/// `magnetar-runtime`'s own generic types, never a concrete format-parser
/// crate, so it can live in the Core without violating
/// `externalize-runtime-extension-modules`'s "Model Components, Providers,
/// and Formats Are Externalized" requirement.
///
/// `ModelDType::F32`, `ModelDType::F16`, and `ModelDType::Bf16` storage are
/// supported: `F32` bytes are read directly, matching `HostTensor`'s own
/// f32-only representation, while `F16`/`Bf16` bytes are explicitly
/// converted to `F32` (`implement-production-qwen-model-loading` Decision
/// 8 -- storage dtype and compute dtype remain distinct, and any
/// conversion is explicit here, never inferred from a source annotation
/// such as `torch_dtype`). `ModelDType::Q8`/`Q4K`/`Q5K` (GGUF's
/// `block_q8_0`/`block_q4_K`/`block_q5_K` block quantization) are likewise
/// dequantized to `F32` here, ported bit-for-bit from `ggml-org/llama.cpp`'s
/// real `ggml-quants.c` (`support-gguf-quantized-tensor-dequantization` design.md) --
/// this is dequantization-at-load, not quantized compute: every kernel
/// downstream still only ever sees `F32` `HostTensor` content, exactly as
/// for F16/BF16 today, real quantized compute kernels remaining a
/// separate, not-yet-implemented chantier. A tensor declaring any other
/// storage dtype (GPTQ/AWQ/BitsAndBytes, integer types) is rejected with
/// `StorageDTypeUnsupported` rather than reinterpreting its bytes.
///
/// For an `F16`/`Bf16`/`Q8`/`Q4K`/`Q5K` tensor whose metadata declares a
/// content digest, that digest is verified against the *original* storage
/// bytes before conversion, not against the converted `F32` representation
/// (Decision 8's "tensor content integrity is verified against the
/// original storage bytes before conversion") -- a digest computed over
/// the source storage representation could never match the converted `F32`
/// representation's different byte count, so checking anywhere else would
/// be structurally wrong, not merely stricter. `F32` tensors keep their
/// existing digest-check location (`InferenceApiError::WeightContentDigestMismatch`,
/// checked once the `HostTensor` is bound) unchanged, since for `F32`
/// storage the in-memory bytes and the original storage bytes are
/// identical either way.
pub fn host_tensors_from_artifact_bytes(
    tensors: &[ModelTensorMetadata],
    bytes: &[u8],
    data_section_start: u64,
) -> Result<BTreeMap<String, HostTensor>, ModelLoadingError> {
    let mut weights = BTreeMap::new();
    for tensor in tensors {
        let raw = slice_tensor_bytes(tensor, bytes, data_section_start)?;
        let host_tensor = tensor_from_raw_bytes(tensor, raw)?;
        weights.insert(tensor.name.clone(), host_tensor);
    }
    Ok(weights)
}

/// The total byte count this Runtime expects for `element_count` elements
/// of a declared storage dtype, or a structured error for any other dtype
/// (integer types) -- shared by both the whole-buffer
/// ([`host_tensors_from_artifact_bytes`]) and streaming
/// ([`stream_materialize_model_instance_weights`] in `first_native_runtime`)
/// materialization paths. Not a flat "bytes per element" for the
/// block-quantized formats (`Q8_0`/`Q4_K`/`Q5_K`): each ggml block covers a
/// fixed element count at a fixed byte size, so `element_count` must divide
/// evenly by that block's element count.
fn expected_storage_bytes(
    element_count: u64,
    tensor: &ModelTensorMetadata,
) -> Result<u64, ModelLoadingError> {
    let overflow = || {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' byte-size computation overflowed", tensor.name),
        )
    };
    let block_based = |block_elements: u64, block_bytes: u64, format_name: &str| {
        if !element_count.is_multiple_of(block_elements) {
            return Err(ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' element count {element_count} is not a multiple of \
                     {format_name}'s block size {block_elements}",
                    tensor.name
                ),
            ));
        }
        (element_count / block_elements)
            .checked_mul(block_bytes)
            .ok_or_else(overflow)
    };
    match tensor.storage_dtype {
        ModelDType::F32 => element_count.checked_mul(4).ok_or_else(overflow),
        ModelDType::F16 | ModelDType::Bf16 => element_count.checked_mul(2).ok_or_else(overflow),
        ModelDType::Q8 => block_based(Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, "Q8_0"),
        ModelDType::Q4K => block_based(QK_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES, "Q4_K"),
        ModelDType::Q5K => block_based(QK_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, "Q5_K"),
        ModelDType::Q6K => block_based(QK_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, "Q6_K"),
        ModelDType::Q2K => block_based(QK_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES, "Q2_K"),
        ModelDType::Q3K => block_based(QK_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES, "Q3_K"),
        ModelDType::Iq4Nl => block_based(Q8_0_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, "IQ4_NL"),
        ModelDType::Iq4Xs => block_based(QK_BLOCK_ELEMENTS, IQ4_XS_BLOCK_BYTES, "IQ4_XS"),
        ModelDType::Iq2Xxs => block_based(QK_BLOCK_ELEMENTS, IQ2_XXS_BLOCK_BYTES, "IQ2_XXS"),
        ModelDType::Iq2Xs => block_based(QK_BLOCK_ELEMENTS, IQ2_XS_BLOCK_BYTES, "IQ2_XS"),
        ModelDType::Iq2S => block_based(QK_BLOCK_ELEMENTS, IQ2_S_BLOCK_BYTES, "IQ2_S"),
        ModelDType::Iq3Xxs => block_based(QK_BLOCK_ELEMENTS, IQ3_XXS_BLOCK_BYTES, "IQ3_XXS"),
        ModelDType::Iq3S => block_based(QK_BLOCK_ELEMENTS, IQ3_S_BLOCK_BYTES, "IQ3_S"),
        other => Err(ModelLoadingError::new(
            ModelLoadingErrorCode::StorageDTypeUnsupported,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!(
                "tensor '{}' declares unsupported storage dtype {other:?} (only F32, F16, BF16, \
                 Q8_0, the k-quants and the IQ2, IQ3 and IQ4 i-quants are supported)",
                tensor.name
            ),
        )),
    }
}

/// Locates and slices `tensor`'s declared byte range out of `bytes` (a
/// complete, already-in-memory artifact/part buffer), relative to
/// `data_section_start`. This is the whole-buffer materialization path's
/// own concern -- the streaming path ([`stream_materialize_model_instance_
/// weights`]) instead receives exactly one tensor's bytes directly from a
/// bounded [`crate::ProductionArtifactPayloadSource`], with no larger
/// buffer to slice from.
fn slice_tensor_bytes<'a>(
    tensor: &ModelTensorMetadata,
    bytes: &'a [u8],
    data_section_start: u64,
) -> Result<&'a [u8], ModelLoadingError> {
    let element_count = tensor
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| {
            ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' shape element-count computation overflowed",
                    tensor.name
                ),
            )
        })?;
    let expected_size = expected_storage_bytes(element_count, tensor)?;
    let (offset, declared_size) = match (tensor.offset_bytes, tensor.size_bytes) {
        (Some(offset), Some(size)) => (offset, size),
        _ => {
            return Err(ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' has no declared byte offset/size to read from",
                    tensor.name
                ),
            ));
        }
    };
    if declared_size != expected_size {
        return Err(ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!(
                "tensor '{}' declares {declared_size} bytes but its shape implies {expected_size}",
                tensor.name
            ),
        ));
    }
    let offset = data_section_start.checked_add(offset).ok_or_else(|| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!(
                "tensor '{}' data-section-relative offset overflowed",
                tensor.name
            ),
        )
    })?;
    let end = offset.checked_add(declared_size).ok_or_else(|| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' byte range overflowed", tensor.name),
        )
    })?;
    let start = usize::try_from(offset).map_err(|_| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' byte offset is out of range", tensor.name),
        )
    })?;
    let end = usize::try_from(end).map_err(|_| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' byte range is out of range", tensor.name),
        )
    })?;
    bytes.get(start..end).ok_or_else(|| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!(
                "tensor '{}' byte range [{start}, {end}) is out of bounds",
                tensor.name
            ),
        )
    })
}

/// Validates and decodes `raw` -- exactly `tensor`'s declared storage
/// bytes, no more, no less -- into a `HostTensor`, shared by both the
/// whole-buffer and streaming materialization paths (see
/// `slice_tensor_bytes`'s doc comment for which path supplies `raw` how).
/// Verifies `raw.len()` against the shape/dtype-implied byte count, verifies
/// a declared non-F32 (or transposed F32) digest against `raw` before
/// conversion (Decision 8),
/// and converts F16/BF16 to F32 explicitly.
pub fn tensor_from_raw_bytes(
    tensor: &ModelTensorMetadata,
    raw: &[u8],
) -> Result<HostTensor, ModelLoadingError> {
    let element_count = tensor
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| {
            ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' shape element-count computation overflowed",
                    tensor.name
                ),
            )
        })?;
    let expected_size = expected_storage_bytes(element_count, tensor)?;
    let actual_size = raw.len() as u64;
    if actual_size != expected_size {
        return Err(ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!(
                "tensor '{}' supplied {actual_size} bytes but its shape implies {expected_size}",
                tensor.name
            ),
        ));
    }
    let transposed =
        tensor.layout.as_deref() == Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT);
    if (tensor.storage_dtype != ModelDType::F32 || transposed)
        && let Some(expected_digest) = &tensor.digest
    {
        expected_digest.verify_bytes(raw).map_err(|error| {
            ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' storage bytes do not match its declared content digest: {error}",
                    tensor.name
                ),
            )
        })?;
    }
    let data: Vec<f32> = match tensor.storage_dtype {
        ModelDType::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        ModelDType::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
            .collect(),
        ModelDType::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
            .collect(),
        ModelDType::Q8 => dequantize_q8_0(raw),
        ModelDType::Q4K => dequantize_q4_k(raw),
        ModelDType::Q5K => dequantize_q5_k(raw),
        ModelDType::Q6K => dequantize_q6_k(raw),
        ModelDType::Q2K => dequantize_q2_k(raw),
        ModelDType::Q3K => dequantize_q3_k(raw),
        ModelDType::Iq4Nl => dequantize_iq4_nl(raw),
        ModelDType::Iq4Xs => dequantize_iq4_xs(raw),
        ModelDType::Iq2Xxs => dequantize_iq2_xxs(raw),
        ModelDType::Iq2Xs => dequantize_iq2_xs(raw),
        ModelDType::Iq2S => dequantize_iq2_s(raw),
        ModelDType::Iq3Xxs => dequantize_iq3_xxs(raw),
        ModelDType::Iq3S => dequantize_iq3_s(raw),
        other => {
            return Err(ModelLoadingError::new(
                ModelLoadingErrorCode::StorageDTypeUnsupported,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' declares unsupported storage dtype {other:?}",
                    tensor.name
                ),
            ));
        }
    };
    // A tensor stored transposed (`TRANSPOSED_STORAGE_LAYOUT`) holds its
    // declared `[rows, cols]` matrix as `[cols, rows]`: converted in stored
    // order above, then transposed into the declared shape.
    let data = match stored_transposed_dims(tensor)? {
        Some((rows, cols)) => crate::packed_weight::transpose(&data, cols as usize, rows as usize),
        None => data,
    };
    HostTensor::new(tensor.shape.clone(), data).map_err(|error| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' failed to materialize: {error}", tensor.name),
        )
    })
}

/// The declared `[rows, cols]` of a tensor whose layout declares transposed
/// storage, `None` for any other layout. Transposed storage is only
/// meaningful for a 2D tensor; any other layout value is rejected rather
/// than ignored, since reading the bytes as untransposed would silently
/// produce the wrong matrix.
fn stored_transposed_dims(
    tensor: &ModelTensorMetadata,
) -> Result<Option<(u64, u64)>, ModelLoadingError> {
    let reject = |reason: String| {
        Err(ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' {reason}", tensor.name),
        ))
    };
    match tensor.layout.as_deref() {
        None => Ok(None),
        Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT) => match tensor.shape.as_slice() {
            [rows, cols] => Ok(Some((*rows, *cols))),
            shape => reject(format!(
                "declares transposed storage but has the non-2D shape {shape:?}"
            )),
        },
        Some(other) => reject(format!("declares the unsupported storage layout '{other}'")),
    }
}

/// Reads exactly one tensor's declared byte range from `payload_source`,
/// unconverted.
pub fn read_tensor_payload(
    tensor: &ModelTensorMetadata,
    payload_source: &dyn crate::ProductionArtifactPayloadSource,
) -> Result<Vec<u8>, ModelLoadingError> {
    let (offset, length) = match (tensor.offset_bytes, tensor.size_bytes) {
        (Some(offset), Some(size)) => (offset, size),
        _ => {
            return Err(ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' has no declared byte offset/size to read from",
                    tensor.name
                ),
            ));
        }
    };
    let range = crate::ProductionPayloadRange {
        identity: tensor.name.clone(),
        offset,
        length,
        digest: tensor.digest.clone(),
    };
    payload_source.read_payload(&range).map_err(|error| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' payload read failed: {error}", tensor.name),
        )
    })
}

/// The packed format `tensor` keeps when offered packed, `None` when it is
/// only ever dequantized: its storage dtype has no packed form (`Q8_0`, the
/// `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K` and `Q6_K` k-quants, and the IQ2, IQ3 and IQ4
/// i-quants do), it is not a matrix, it is an untransposed k-quant or
/// i-quant (an embedding table, read row by row), or its stored rows are
/// not whole blocks.
pub fn packed_weight_format(
    tensor: &ModelTensorMetadata,
) -> Result<Option<crate::packed_weight::PackedWeightFormat>, ModelLoadingError> {
    let format = match tensor.storage_dtype {
        ModelDType::Q8 => crate::packed_weight::PackedWeightFormat::Q8_0,
        ModelDType::Q4K => crate::packed_weight::PackedWeightFormat::Q4K,
        ModelDType::Q5K => crate::packed_weight::PackedWeightFormat::Q5K,
        ModelDType::Q6K => crate::packed_weight::PackedWeightFormat::Q6K,
        ModelDType::Q2K => crate::packed_weight::PackedWeightFormat::Q2K,
        ModelDType::Q3K => crate::packed_weight::PackedWeightFormat::Q3K,
        ModelDType::Iq4Nl => crate::packed_weight::PackedWeightFormat::Iq4Nl,
        ModelDType::Iq4Xs => crate::packed_weight::PackedWeightFormat::Iq4Xs,
        ModelDType::Iq2Xxs => crate::packed_weight::PackedWeightFormat::Iq2Xxs,
        ModelDType::Iq2Xs => crate::packed_weight::PackedWeightFormat::Iq2Xs,
        ModelDType::Iq2S => crate::packed_weight::PackedWeightFormat::Iq2S,
        ModelDType::Iq3Xxs => crate::packed_weight::PackedWeightFormat::Iq3Xxs,
        ModelDType::Iq3S => crate::packed_weight::PackedWeightFormat::Iq3S,
        _ => return Ok(None),
    };
    let [rows, cols] = tensor.shape.as_slice() else {
        return Ok(None);
    };
    let transposed = stored_transposed_dims(tensor)?.is_some();
    // Packed k-quants and i-quants serve projections (transposed storage, read by the
    // decode matmul); an untransposed one, such as an embedding table, is
    // read row by row, which is only cheap dequantized once at load.
    if format != crate::packed_weight::PackedWeightFormat::Q8_0 && !transposed {
        return Ok(None);
    }
    let stored_cols = if transposed { *rows } else { *cols };
    // Stored rows that are not whole blocks (a block spanning two rows)
    // have no packed matrix form; such a tensor is only ever dequantized.
    if !stored_cols.is_multiple_of(format.block_elements()) {
        return Ok(None);
    }
    Ok(Some(format))
}

/// The bytes `tensor` occupies once materialized: its storage bytes when it
/// stays packed ([`packed_weight_format`]) or is stored as `f32`, else its
/// elements as `f32` (dequantized or converted at load). An upper bound
/// for a Provider that keeps a narrower form, never an underestimate.
pub fn resident_weight_bytes(tensor: &ModelTensorMetadata) -> u64 {
    let stored = tensor.size_bytes.unwrap_or(0);
    if tensor.storage_dtype == ModelDType::F32
        || matches!(packed_weight_format(tensor), Ok(Some(_)))
    {
        return stored;
    }
    let elements = tensor.shape.iter().product::<u64>();
    stored.max(elements.saturating_mul(4))
}

/// The packed form of a block-quantized 2D weight
/// (`execute-direct-quantized-weights`), from the tensor's own validated
/// storage bytes: `Ok(None)` when its storage dtype has no packed form
/// (`Q8_0`, the `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K` and `Q6_K` k-quants, and the
/// IQ2, IQ3 and IQ4 i-quants), it is not a matrix, or its stored rows are
/// not whole blocks. The bytes are checked
/// against the shape and against the declared content digest first, as for
/// any other storage dtype.
pub fn packed_weight_from_raw_bytes(
    tensor: &ModelTensorMetadata,
    raw: &[u8],
) -> Result<Option<crate::packed_weight::PackedWeight>, ModelLoadingError> {
    let Some(format) = packed_weight_format(tensor)? else {
        return Ok(None);
    };
    let [rows, cols] = tensor.shape.as_slice() else {
        return Ok(None);
    };
    let transposed = stored_transposed_dims(tensor)?.is_some();
    let (stored_rows, stored_cols) = if transposed {
        (*cols, *rows)
    } else {
        (*rows, *cols)
    };
    if let Some(expected_digest) = &tensor.digest {
        expected_digest.verify_bytes(raw).map_err(|error| {
            ModelLoadingError::new(
                ModelLoadingErrorCode::MaterializationFailed,
                Some(ModelLoadingPhase::MaterializeWeights),
                format!(
                    "tensor '{}' storage bytes do not match its declared content digest: {error}",
                    tensor.name
                ),
            )
        })?;
    }
    crate::packed_weight::PackedWeight::new(
        format,
        stored_rows,
        stored_cols,
        transposed,
        raw.to_vec(),
    )
    .map(Some)
    .map_err(|reason| {
        ModelLoadingError::new(
            ModelLoadingErrorCode::MaterializationFailed,
            Some(ModelLoadingPhase::MaterializeWeights),
            format!("tensor '{}' {reason}", tensor.name),
        )
    })
}

/// Reads, validates, converts, and returns exactly one tensor's `HostTensor`
/// content from a bounded [`crate::ProductionArtifactPayloadSource`] --
/// the per-tensor building block [`crate::first_native_runtime::
/// stream_materialize_model_instance_weights`] calls in a loop instead of
/// requiring the entire model to first exist as a whole-model
/// `BTreeMap<String, HostTensor>` (`implement-production-qwen-model-loading`
/// task group 6 / "Production Model Loading Supports Bounded Tensor Payload
/// Access"). The returned `HostTensor` is this function's only allocation
/// surviving past its own call -- nothing here accumulates state across
/// tensors, so peak transient host staging stays bounded by whatever the
/// caller does with each `HostTensor` before requesting the next one.
pub fn host_tensor_from_payload_source(
    tensor: &ModelTensorMetadata,
    payload_source: &dyn crate::ProductionArtifactPayloadSource,
) -> Result<HostTensor, ModelLoadingError> {
    let raw = read_tensor_payload(tensor, payload_source)?;
    tensor_from_raw_bytes(tensor, &raw)
}

pub fn invalidates_kv_cache_on_unload(policy: ModelLoadingCachePolicy) -> bool {
    matches!(policy, ModelLoadingCachePolicy::InvalidateKvCacheOnUnload)
}

pub fn reload_is_new_loading_process(request: &ModelReloadRequest) -> bool {
    !request.allow_context_mutation
}

pub fn compute_dtype_supported(manifest: &ModelManifest, requested: ModelDType) -> bool {
    manifest.supported_compute_dtypes.contains(&requested)
        || manifest.compute_dtype == Some(requested)
}

pub fn storage_to_compute_relation(
    manifest: &ModelManifest,
    requested: Option<ModelDType>,
) -> Option<(ComputeDType, ComputeDType)> {
    let storage = manifest.storage_dtype?.descriptor().compute_dtype()?;
    let compute = requested
        .or(manifest.compute_dtype)?
        .descriptor()
        .compute_dtype()?;
    Some((storage, compute))
}

fn memory_placement_for(
    preference: &ModelPlacementPreference,
) -> Result<MemoryPlacement, ModelLoadingError> {
    match preference {
        ModelPlacementPreference::RuntimeDefault | ModelPlacementPreference::Host => {
            Ok(MemoryPlacement::HostOrdinary)
        }
        ModelPlacementPreference::PinnedHost => Ok(MemoryPlacement::HostPinned),
        ModelPlacementPreference::BrowserLinearMemory => Ok(MemoryPlacement::BrowserLinearMemory),
        ModelPlacementPreference::Affinity(affinity) => {
            if let Some(device) = affinity.device() {
                Ok(MemoryPlacement::Device(device.clone()))
            } else if let Some(provider) = affinity.provider() {
                Ok(MemoryPlacement::ProviderOwnedOpaque(provider.clone()))
            } else {
                Ok(MemoryPlacement::HostOrdinary)
            }
        }
    }
}

fn data_movement_for(request: &ModelLoadingRequest) -> Vec<String> {
    match request.placement_preference {
        ModelPlacementPreference::RuntimeDefault | ModelPlacementPreference::Host => Vec::new(),
        ModelPlacementPreference::PinnedHost => vec!["pin-host-memory".into()],
        ModelPlacementPreference::BrowserLinearMemory => vec!["browser-linear-memory".into()],
        ModelPlacementPreference::Affinity(_) => vec!["resolve-affinity-placement".into()],
    }
}

fn default_loading_phases() -> Vec<ModelLoadingPhase> {
    vec![
        ModelLoadingPhase::ReadManifest,
        ModelLoadingPhase::ValidateParts,
        ModelLoadingPhase::OpenArtifactBytes,
        ModelLoadingPhase::ValidateShards,
        ModelLoadingPhase::PlanMemory,
        ModelLoadingPhase::AllocateHost,
        ModelLoadingPhase::AllocateDevice,
        ModelLoadingPhase::MaterializeWeights,
        ModelLoadingPhase::DequantizeOrTransform,
        ModelLoadingPhase::TransferToDevice,
        ModelLoadingPhase::InitializeProviderState,
        ModelLoadingPhase::ValidateReady,
        ModelLoadingPhase::PublishModelContext,
    ]
}

trait DTypeDescriptorExt {
    fn compute_dtype(&self) -> Option<ComputeDType>;
}

impl DTypeDescriptorExt for crate::DTypeDescriptor {
    fn compute_dtype(&self) -> Option<ComputeDType> {
        match self {
            Self::Portable(dtype) => Some(*dtype),
            Self::ProviderSpecific { .. } => None,
        }
    }
}

impl ModelLoadingCoordinator {
    fn observe(
        &mut self,
        kind: ModelLoadingObservationKind,
        request: Option<ModelLoadingRequestId>,
        residency: Option<ModelResidencyId>,
        phase: Option<ModelLoadingPhase>,
        message: impl Into<String>,
    ) {
        self.observations.push(ModelLoadingObservation {
            kind,
            request,
            residency,
            phase,
            correlation_id: None,
            message: message.into(),
        });
    }
}

pub fn allocation_released(allocation: &MemoryAllocation) -> bool {
    matches!(
        allocation.state,
        MemoryAllocationState::Released
            | MemoryAllocationState::Reusable
            | MemoryAllocationState::Failed
            | MemoryAllocationState::Cancelled
    )
}

pub use crate::quantization::iq_grids;

#[cfg(test)]
mod tests;
