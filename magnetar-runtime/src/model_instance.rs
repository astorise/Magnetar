//! Runtime-owned Model Instance lifecycle contracts.
//!
//! A Model Instance is the Runtime-owned loaded inference context created from
//! successful model loading. It is distinct from the immutable Model Artifact,
//! from residency records, from sessions, from Provider resources, and from KV
//! cache state. The public contract exposes stable metadata only; raw model
//! weights, native handles, Device handles, Provider handles, memory pointers,
//! prompts, and cache contents stay behind Runtime-owned boundaries.

use crate::{
    AdapterSetId, CorrelationId, DeviceBinding, GenerationModelReference, InferenceSessionId,
    KernelAutotuningPolicy, KernelId, KernelPerformanceFeedbackMode, KvCacheId, LoadedModelContext,
    MemoryAllocationId, MemoryPressureLevel, ModelArchitectureImplementation, ModelArtifactId,
    ModelDType, ModelDigest, ModelResidencyId, PrefixCacheEntryId, ProviderAdmissionDecision,
    ProviderBinding, ProviderHealthState, ProviderPressureLevel, ProviderReadinessState,
    ResourceAffinity, TensorResourceId, TokenizerId, reproducible_mode_blocks_adaptation,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelInstanceId(String);

impl ModelInstanceId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelInstanceError> {
        let value = value.into();
        validate_instance_identity(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(sequence: u64) -> Self {
        Self(format!("model-instance-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelInstanceLifecycleState {
    Creating,
    Loading,
    Warming,
    Ready,
    Active,
    Idle,
    Draining,
    Suspended,
    Reloading,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
    Removed,
}

impl ModelInstanceLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Unloaded | Self::Failed | Self::Invalid | Self::Removed
        )
    }

    /// Whether this lifecycle state legitimately permits accepting
    /// inference usage. `readiness == Ready` alone SHALL NOT be trusted as
    /// sufficient (Correctif: Runtime-owned ModelInstance readiness
    /// authority) -- a caller-forged or internally inconsistent
    /// `readiness` value must not grant usage while the instance's own
    /// lifecycle has not actually reached one of these states.
    pub const fn supports_inference_use(self) -> bool {
        matches!(self, Self::Ready | Self::Idle | Self::Active)
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Loading)
                | (Self::Creating, Self::Failed)
                | (Self::Creating, Self::Invalid)
                | (Self::Loading, Self::Warming)
                | (Self::Loading, Self::Ready)
                | (Self::Loading, Self::Failed)
                | (Self::Warming, Self::Ready)
                | (Self::Warming, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Idle)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Suspended)
                | (Self::Ready, Self::Reloading)
                | (Self::Ready, Self::Unloading)
                | (Self::Ready, Self::Failed)
                | (Self::Ready, Self::Invalid)
                | (Self::Active, Self::Idle)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Failed)
                | (Self::Idle, Self::Active)
                | (Self::Idle, Self::Draining)
                | (Self::Idle, Self::Suspended)
                | (Self::Idle, Self::Reloading)
                | (Self::Idle, Self::Unloading)
                | (Self::Idle, Self::Failed)
                | (Self::Idle, Self::Invalid)
                | (Self::Draining, Self::Unloading)
                | (Self::Draining, Self::Reloading)
                | (Self::Draining, Self::Failed)
                | (Self::Suspended, Self::Loading)
                | (Self::Suspended, Self::Reloading)
                | (Self::Suspended, Self::Unloading)
                | (Self::Suspended, Self::Failed)
                | (Self::Reloading, Self::Loading)
                | (Self::Reloading, Self::Ready)
                | (Self::Reloading, Self::Failed)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Unloading, Self::Failed)
                | (Self::Unloaded, Self::Removed)
                | (Self::Failed, Self::Unloading)
                | (Self::Failed, Self::Invalid)
                | (Self::Invalid, Self::Unloading)
                | (Self::Invalid, Self::Removed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelInstanceReadiness {
    NotReady,
    Ready,
    ReadOnly,
    Draining,
    Suspended,
    Failed,
}

impl ModelInstanceReadiness {
    pub const fn accepts_generation(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupPolicy {
    Disabled,
    ValidateMetadataOnly,
    ProviderInitialization,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceUnloadPolicy {
    DrainActiveUse,
    RejectActiveUse,
    ForceInvalidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceSharingPolicy {
    Private,
    RuntimeLocal,
    TenantIsolated,
    PolicyControlled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceMutationKind {
    AdapterMerge,
    ProviderPreparation,
    QuantizationTransform,
    ResidencyRelocation,
    Reload,
    Warmup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceSuspensionReason {
    MemoryPressure,
    ProviderPressure,
    DevicePressure,
    AdministrativePolicy,
    BrowserLifecycle,
    TemporaryResourceLoss,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstancePolicy {
    pub warmup: ModelInstanceWarmupPolicy,
    pub unload: ModelInstanceUnloadPolicy,
    pub sharing: ModelInstanceSharingPolicy,
    pub implicit_loading_allowed: bool,
    pub suspension_allowed: bool,
    pub raw_handle_exposure_allowed: bool,
    pub tenant_isolation_required: bool,
    pub browser_linear_memory_limit_bytes: Option<u64>,
    /// Implements "Model Instance May Have Autotuning Policy"
    /// (`define-kernel-runtime-autotuning-and-specialization-contract`):
    /// disabled, optional, required, or pinned Kernel Autotuning behavior.
    pub autotuning: KernelAutotuningPolicy,
    /// Implements "Model Instance Interaction" / "Reproducible Mode"
    /// (`define-kernel-performance-model-and-adaptive-feedback-contract`):
    /// "A Model Instance MAY: consume adaptive performance evidence / use
    /// dynamic selection policy / remain pinned/reproducible and ignore
    /// adaptive changes. Policy SHALL be explicit." The *effective* mode
    /// also depends on whether Kernel selection is pinned -- see
    /// [`effective_performance_feedback_mode`].
    pub performance_feedback: KernelPerformanceFeedbackMode,
}

impl Default for ModelInstancePolicy {
    fn default() -> Self {
        Self {
            warmup: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
            unload: ModelInstanceUnloadPolicy::DrainActiveUse,
            sharing: ModelInstanceSharingPolicy::RuntimeLocal,
            implicit_loading_allowed: false,
            suspension_allowed: true,
            raw_handle_exposure_allowed: false,
            tenant_isolation_required: false,
            browser_linear_memory_limit_bytes: None,
            autotuning: KernelAutotuningPolicy::Disabled,
            performance_feedback: KernelPerformanceFeedbackMode::Disabled,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceCreationChecks {
    pub artifact_identity_valid: bool,
    pub artifact_trusted: bool,
    pub architecture_available: bool,
    pub residency_plan_valid: bool,
    pub memory_admitted: bool,
    pub provider_device_compatible: bool,
    pub tokenizer_compatible: bool,
    pub runtime_policy_allows: bool,
    pub browser_native_supported: bool,
}

impl Default for ModelInstanceCreationChecks {
    fn default() -> Self {
        Self {
            artifact_identity_valid: true,
            artifact_trusted: true,
            architecture_available: true,
            residency_plan_valid: true,
            memory_admitted: true,
            provider_device_compatible: true,
            tokenizer_compatible: true,
            runtime_policy_allows: true,
            browser_native_supported: true,
        }
    }
}

impl ModelInstanceCreationChecks {
    pub fn validate(&self) -> Result<(), ModelInstanceError> {
        if !self.artifact_identity_valid {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.artifact_trusted {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.architecture_available {
            return Err(ModelInstanceError::ModelInstanceNotReady);
        }
        if !self.residency_plan_valid {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        if !self.memory_admitted {
            return Err(ModelInstanceError::ModelInstanceMemoryPressure);
        }
        if !self.provider_device_compatible {
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if !self.tokenizer_compatible {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.runtime_policy_allows {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.browser_native_supported {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceReadinessChecks {
    pub residency_available: bool,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub adapter_ready: bool,
    pub memory_pressure: MemoryPressureLevel,
    pub runtime_policy_allows: bool,
    pub browser_supported: bool,
    /// Whether every mandatory Kernel Artifact preparation for this Model
    /// Instance's execution plan has completed successfully. Implements
    /// "Model Instance Readiness" (`define-kernel-artifact-and-preparation-contract`):
    /// "A Model Instance SHALL NOT silently become ready when required
    /// Kernel preparation has failed." Defaults to `true` so Model Instances
    /// with no artifact-backed Kernels are unaffected.
    pub kernel_preparation_ready: bool,
    /// Whether required Kernel Autotuning has completed, implementing "The
    /// Model Instance SHALL remain in a non-ready or explicitly warming
    /// state until mandatory tuning is complete if deployment policy
    /// requires tuning" (`define-kernel-runtime-autotuning-and-specialization-contract`).
    /// Defaults to `true` so Model Instances with
    /// [`KernelAutotuningPolicy::Required`] unset are unaffected; callers
    /// SHALL set this to the actual completion state when policy requires
    /// tuning (see [`crate::model_instance_autotuning_ready`]).
    pub autotuning_ready: bool,
    /// Whether this Model Instance's declared weights were successfully
    /// materialized into Provider-owned Tensor Resources
    /// (`model-loading-materializes-weight-resources`: "Model Instance
    /// Readiness" SHALL consider weight materialization state, alongside
    /// residency/Provider/Device/adapter/memory-pressure/policy/architecture
    /// readiness). Defaults to `true` so a Model Instance whose caller never
    /// materializes weights through the generic phase (or has none to
    /// materialize) is unaffected; a caller that does materialize weights
    /// SHALL set this from the real success/failure outcome.
    pub weights_materialized: bool,
}

impl Default for ModelInstanceReadinessChecks {
    fn default() -> Self {
        Self {
            residency_available: true,
            provider_ready: true,
            device_ready: true,
            adapter_ready: true,
            memory_pressure: MemoryPressureLevel::Low,
            runtime_policy_allows: true,
            browser_supported: true,
            kernel_preparation_ready: true,
            autotuning_ready: true,
            weights_materialized: true,
        }
    }
}

impl ModelInstanceReadinessChecks {
    pub fn readiness(&self) -> ModelInstanceReadiness {
        if !self.provider_ready
            || !self.device_ready
            || !self.adapter_ready
            || !self.runtime_policy_allows
            || !self.kernel_preparation_ready
            || !self.autotuning_ready
            || !self.weights_materialized
        {
            return ModelInstanceReadiness::Failed;
        }
        if !self.residency_available
            || matches!(
                self.memory_pressure,
                MemoryPressureLevel::High | MemoryPressureLevel::Saturated
            )
        {
            return ModelInstanceReadiness::Suspended;
        }
        if !self.browser_supported {
            return ModelInstanceReadiness::Failed;
        }
        ModelInstanceReadiness::Ready
    }

    pub fn validate(&self) -> Result<(), ModelInstanceError> {
        if !self.residency_available {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        if !self.provider_ready {
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if !self.device_ready {
            return Err(ModelInstanceError::ModelInstanceDeviceUnavailable);
        }
        if !self.adapter_ready {
            return Err(ModelInstanceError::ModelInstanceAdapterIncompatible);
        }
        if !self.kernel_preparation_ready {
            return Err(ModelInstanceError::ModelInstanceKernelPreparationFailed);
        }
        if !self.autotuning_ready {
            return Err(ModelInstanceError::ModelInstanceAutotuningIncomplete);
        }
        if !self.weights_materialized {
            return Err(ModelInstanceError::ModelInstanceWeightsNotMaterialized);
        }
        if matches!(
            self.memory_pressure,
            MemoryPressureLevel::High | MemoryPressureLevel::Saturated
        ) {
            return Err(ModelInstanceError::ModelInstanceMemoryPressure);
        }
        if !self.runtime_policy_allows {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if !self.browser_supported {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupStep {
    ProviderInitialization,
    KernelPreparationPlaceholder,
    OperatorGraphPreparationPlaceholder,
    ShapePlanPreparationPlaceholder,
    TokenizerModelMetadataValidation,
    SmallTestExecutionPlaceholder,
    MemoryResidencyVerification,
    AdapterReadinessVerification,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceWarmupPlan {
    pub policy: ModelInstanceWarmupPolicy,
    pub steps: Vec<ModelInstanceWarmupStep>,
}

impl ModelInstanceWarmupPlan {
    pub fn for_policy(policy: ModelInstanceWarmupPolicy) -> Self {
        let steps = match policy {
            ModelInstanceWarmupPolicy::Disabled => Vec::new(),
            ModelInstanceWarmupPolicy::ValidateMetadataOnly => {
                vec![ModelInstanceWarmupStep::TokenizerModelMetadataValidation]
            }
            ModelInstanceWarmupPolicy::ProviderInitialization => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
            ],
            ModelInstanceWarmupPolicy::Full => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::KernelPreparationPlaceholder,
                ModelInstanceWarmupStep::OperatorGraphPreparationPlaceholder,
                ModelInstanceWarmupStep::ShapePlanPreparationPlaceholder,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::SmallTestExecutionPlaceholder,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
                ModelInstanceWarmupStep::AdapterReadinessVerification,
            ],
        };
        Self { policy, steps }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceUsage {
    pub active_operation_count: usize,
    pub active_session_count: usize,
    pub queued_operation_count: usize,
    pub total_request_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub last_used_millis: Option<u64>,
    pub residency_bytes: u64,
    pub kv_cache_dependencies: BTreeSet<KvCacheId>,
    pub prefix_cache_dependencies: BTreeSet<PrefixCacheEntryId>,
    pub adapter_dependencies: BTreeSet<AdapterSetId>,
    pub failure_count: u64,
}

impl ModelInstanceUsage {
    pub const fn has_active_use(&self) -> bool {
        self.active_operation_count > 0 || self.active_session_count > 0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceAdapterState {
    pub active_adapter_set: Option<AdapterSetId>,
    pub activation_scope: Option<String>,
    pub merged: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderModelResource {
    pub provider: ProviderBinding,
    pub handle_kind: String,
    pub release_required: bool,
}

/// `pub(crate)` fields, not `pub`: the only legitimate way to bind a
/// weight resource is the one authorized weight-materialization
/// transaction (`WeightMaterializationTransaction::commit` in
/// `first_native_runtime.rs`) committing successfully
/// (`bind-model-loading-evidence-to-validated-artifact`, closing a gap an
/// external audit of PR #36 found concretely: this crate's own
/// `contract_tests` helper was directly inserting into `weights` and
/// `memory_allocations` by hand). Public read-only accessors are added
/// only if a real external caller needs to inspect a binding after
/// `warm_model_instance` succeeds; none is known to today.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceResourceBindings {
    /// Canonical model tensor name (matching `ModelManifest`'s declared
    /// tensor names) to the stable `TensorResourceId` Runtime created for it
    /// during model loading, so graph execution can look weights up by name
    /// through this Model Instance rather than a private side-channel
    /// keeping its own copy of tensor bytes.
    pub(crate) weights: BTreeMap<String, TensorResourceId>,
    pub(crate) memory_allocations: BTreeSet<MemoryAllocationId>,
    pub(crate) released_memory_allocations: BTreeSet<MemoryAllocationId>,
    pub(crate) released_provider_resources: BTreeSet<ProviderBinding>,
}

/// Runtime-issued proof that a specific Model Instance's weight resources
/// were bound by the one authorized weight-materialization transaction
/// (`WeightMaterializationTransaction::commit` in `first_native_runtime.rs`),
/// not assembled by hand from otherwise-public Memory Manager/Provider
/// primitives. Not constructible or settable by an external caller --
/// see `bind-model-loading-evidence-to-validated-artifact`'s design.md for
/// why this lives as a separate Runtime-owned record keyed by
/// `ModelInstanceId`, rather than a field on `ModelInstance`/
/// `ModelInstanceDefinition`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializationEvidence {
    artifact: ModelArtifactId,
    resources: BTreeSet<TensorResourceId>,
}

impl MaterializationEvidence {
    pub(crate) fn new(artifact: ModelArtifactId, resources: BTreeSet<TensorResourceId>) -> Self {
        Self {
            artifact,
            resources,
        }
    }

    /// True only if this evidence was minted for `artifact` and the exact
    /// resource set `bound` -- both the instance's own declared artifact
    /// and its exact currently-bound weight resources, so evidence minted
    /// for a different instance or a stale binding set does not match.
    pub(crate) fn matches(
        &self,
        artifact: &ModelArtifactId,
        bound: &BTreeSet<TensorResourceId>,
    ) -> bool {
        &self.artifact == artifact && &self.resources == bound
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstancePlacement {
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub affinity: ResourceAffinity,
    pub provider_resource: Option<ProviderModelResource>,
}

impl ModelInstancePlacement {
    pub fn new(affinity: ResourceAffinity) -> Self {
        Self {
            provider: affinity.provider().cloned(),
            device: affinity.device().cloned(),
            affinity,
            provider_resource: None,
        }
    }

    pub const fn exposes_raw_handles(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceDefinition {
    pub artifact: ModelArtifactId,
    pub architecture: ModelArchitectureImplementation,
    pub residencies: BTreeSet<ModelResidencyId>,
    pub tokenizer: Option<TokenizerId>,
    pub placement: ModelInstancePlacement,
    pub policy: ModelInstancePolicy,
    pub adapter_state: ModelInstanceAdapterState,
    pub associated_sessions: BTreeSet<InferenceSessionId>,
    pub usage: ModelInstanceUsage,
    pub compute_dtype: Option<ModelDType>,
    pub mutation_version: u64,
    pub tenant: Option<String>,
    pub owner: Option<String>,
    /// `pub(crate)`, not `pub`: sealing only `ModelInstanceResourceBindings`'s
    /// own fields is not enough on its own -- an external caller could
    /// still replace this whole field with another instance's (already
    /// legitimately bound) bindings via a plain assignment (`Clone` does
    /// not require field visibility), which is exactly the audit's
    /// "materialization evidence reused for a different instance/artifact"
    /// scenario. See `bind-model-loading-evidence-to-validated-artifact`.
    pub(crate) resource_bindings: ModelInstanceResourceBindings,
    /// Implements "Model Instance Kernel Policy"
    /// (`define-kernel-optimization-and-selection-policy`): "A Model
    /// Instance SHALL own or reference an explicit Kernel selection
    /// policy." References a policy identity rather than embedding the
    /// full `KernelSelectionPolicy`, mirroring how `tokenizer` references a
    /// `TokenizerId`.
    pub kernel_selection_policy: Option<crate::kernel_selection_policy::KernelSelectionPolicyId>,
    /// Tensor names the loaded `ModelManifest` declared as mandatory,
    /// carried from `LoadedModelContext::required_weight_names`.
    /// `pub(crate)`, not `pub`: this is Runtime-recorded evidence of what
    /// the artifact actually requires, not something an external caller
    /// should be able to redeclare after creation to make an incomplete
    /// binding set appear complete. Empty means "unknown" (e.g. a
    /// generically-constructed instance with no loaded manifest behind
    /// it) -- readiness derivation falls back to its prior,
    /// presence-only heuristic in that case rather than treating an
    /// empty requirement as trivially satisfied by anything.
    pub(crate) required_weight_names: BTreeSet<String>,
    /// Per-tensor content digests the loaded `ModelManifest` declared,
    /// carried from `LoadedModelContext::required_weight_digests`.
    /// `pub(crate)` for the same reason as `required_weight_names`: not
    /// something an external caller should be able to redeclare after
    /// creation. Only tensors the artifact declared a digest for are
    /// present as keys; a tensor absent here is unconstrained by content
    /// verification, not treated as having empty/zero content
    /// (`bind-materialized-weight-content-to-model-artifact-digests`).
    pub(crate) required_weight_digests: BTreeMap<String, ModelDigest>,
}

impl ModelInstanceDefinition {
    pub fn from_loaded_context(
        context: &LoadedModelContext,
        architecture: ModelArchitectureImplementation,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            artifact: context.artifact.clone(),
            architecture,
            residencies: BTreeSet::from([context.residency]),
            tokenizer: None,
            placement: ModelInstancePlacement::new(affinity),
            policy: ModelInstancePolicy::default(),
            adapter_state: ModelInstanceAdapterState::default(),
            associated_sessions: BTreeSet::new(),
            usage: ModelInstanceUsage {
                residency_bytes: context.plan.expected_resident_bytes,
                ..ModelInstanceUsage::default()
            },
            compute_dtype: context.plan.target_compute_dtype,
            mutation_version: 0,
            tenant: None,
            owner: None,
            resource_bindings: ModelInstanceResourceBindings::default(),
            kernel_selection_policy: None,
            required_weight_names: context.required_weight_names.clone(),
            required_weight_digests: context.required_weight_digests.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceSharingContext {
    pub owner: Option<String>,
    pub tenant: Option<String>,
    pub adapter_state: ModelInstanceAdapterState,
    pub kv_cache_private: bool,
    pub prefix_cache_private: bool,
    pub affinity: ResourceAffinity,
}

impl ModelInstanceSharingContext {
    pub fn from_definition(definition: &ModelInstanceDefinition) -> Self {
        Self {
            owner: definition.owner.clone(),
            tenant: definition.tenant.clone(),
            adapter_state: definition.adapter_state.clone(),
            kv_cache_private: false,
            prefix_cache_private: false,
            affinity: definition.placement.affinity.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceInvalidationReport {
    pub kv_caches: BTreeSet<KvCacheId>,
    pub prefix_entries: BTreeSet<PrefixCacheEntryId>,
    pub adapters_released: BTreeSet<AdapterSetId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceUnloadReport {
    pub invalidated: ModelInstanceInvalidationReport,
    pub released_memory_allocations: BTreeSet<MemoryAllocationId>,
    /// Weight `TensorResourceId`s this unload SHALL also release from
    /// Provider-owned storage, not merely from Memory Manager accounting
    /// (`transactional-weight-materialization`'s "Unloading A Model
    /// Instance Releases Its Provider-Owned Weight Storage" requirement).
    pub released_weight_resources: BTreeSet<TensorResourceId>,
    pub released_provider_resources: BTreeSet<ProviderBinding>,
    pub dangling_session_references: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceReloadRequest {
    pub replacement: ModelInstanceDefinition,
    pub migrate_sessions: bool,
    pub allow_active_semantic_mutation: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstance {
    pub id: ModelInstanceId,
    // `lifecycle`/`readiness` are intentionally `pub(crate)`, not `pub`: an
    // external caller with `&mut ModelInstance` (obtainable via the
    // already-public `Runtime::model_instances_mut()`/`instance_mut()`)
    // SHALL NOT be able to forge `Ready` by direct field assignment, only
    // through the Runtime-controlled paths that verify real evidence first
    // (Correctif: Runtime-owned ModelInstance readiness authority, closing
    // the public mutable field gap an external audit of PR #36 found still
    // open after `mark_ready`/`transition_to`/`warmup` were sealed).
    // `pub(crate)` (rather than fully private) is what a real embedder's
    // dependency on this crate cannot see or write, while still letting
    // this crate's own test suite construct otherwise-unreachable states
    // to prove defense-in-depth checks (e.g. the `acquire_usage`/
    // `generation_reference` lifecycle+readiness safety net) hold even if
    // some future internal bug were to reintroduce an inconsistency.
    pub(crate) lifecycle: ModelInstanceLifecycleState,
    pub(crate) readiness: ModelInstanceReadiness,
    // `pub(crate)`, not `pub`: an external caller with `&mut ModelInstance`
    // SHALL NOT be able to silently mutate Runtime-owned semantic state
    // (artifact identity, architecture, placement, policy, tokenizer, ...)
    // on an already-`Ready` instance by raw field assignment, nor carry an
    // existing instance's `definition` -- including its already-sealed
    // `resource_bindings`, which `Clone` copies regardless of field
    // visibility -- into a *different* instance via `.clone()` +
    // `ModelInstanceManager::create()`/`reload()` (a further audit of PR
    // #36 found both: direct post-`Ready` mutation went unnoticed by
    // `acquire_usage`/`generation_reference`, which only check
    // lifecycle/readiness, not whether the definition still matches what
    // was evidenced; and a cloned definition's weight bindings could be
    // "adopted" by an empty `materialize_model_instance_weights` call on
    // the new instance, aliasing live Provider resources across two
    // instances). Read-only public accessor below; the real fix for the
    // clone-and-create path is `ModelInstanceManager::create` resetting
    // `resource_bindings` unconditionally, not this field's visibility
    // alone -- see its own doc comment.
    pub(crate) definition: ModelInstanceDefinition,
    pub last_error: Option<ModelInstanceError>,
}

impl ModelInstance {
    /// Current lifecycle state. Read-only: see the struct-level doc comment
    /// for why `lifecycle` is not a public field.
    pub const fn lifecycle(&self) -> ModelInstanceLifecycleState {
        self.lifecycle
    }

    /// This instance's Runtime-owned semantic definition (artifact,
    /// architecture, placement, policy, tokenizer, usage, ...). Read-only:
    /// see the struct-level doc comment for why `definition` is not a
    /// public mutable field.
    pub const fn definition(&self) -> &ModelInstanceDefinition {
        &self.definition
    }

    /// Records a Provider-owned resource handle for cleanup accounting on
    /// unload (`ModelInstanceUnloadReport::released_provider_resources`).
    /// Deliberately narrow: unlike the rest of `placement`, `provider_
    /// resource` drives only this bookkeeping -- never readiness,
    /// generation, or Provider/Device resolution (execution resolves
    /// Providers through `resource_bindings`/`TensorResidency`, not this
    /// field) -- so setting it after `Ready` cannot be used to silently
    /// redirect what artifact, weights, or Provider an instance actually
    /// executes against.
    pub fn set_provider_resource(&mut self, resource: Option<ProviderModelResource>) {
        self.definition.placement.provider_resource = resource;
    }

    /// Tracks an additional Memory Manager allocation against this
    /// instance's resource bindings, for allocations outside the weight-
    /// materialization transaction's scope (e.g. an adapter or workspace
    /// allocation a caller manages directly). Deliberately narrow: this
    /// does not touch `weights` or materialization evidence, so it cannot
    /// be used to forge `weights_materialized` readiness -- see
    /// `bind-model-loading-evidence-to-validated-artifact`'s design.md.
    /// Only reachable post-creation, on this specific instance -- not a way
    /// to pre-populate a definition before `ModelInstanceManager::create`,
    /// which resets `resource_bindings` unconditionally regardless of what
    /// the supplied definition carried (a further audit of PR #36 found a
    /// caller could otherwise clone another instance's already-populated
    /// bindings into a new one).
    pub fn track_memory_allocation(&mut self, allocation: MemoryAllocationId) {
        self.definition
            .resource_bindings
            .memory_allocations
            .insert(allocation);
    }

    /// Current readiness state. Read-only: see the struct-level doc comment
    /// for why `readiness` is not a public field.
    pub const fn readiness(&self) -> ModelInstanceReadiness {
        self.readiness
    }

    pub fn new(id: ModelInstanceId, definition: ModelInstanceDefinition) -> Self {
        Self {
            id,
            lifecycle: ModelInstanceLifecycleState::Creating,
            readiness: ModelInstanceReadiness::NotReady,
            definition,
            last_error: None,
        }
    }

    /// Crate-internal only: a raw lifecycle transition with no readiness
    /// evidence check. Public callers reach lifecycle changes only through
    /// the business methods below (`suspend`, `resume`, `drain`, `unload`,
    /// ...) or, for `Ready` specifically, through
    /// `crate::inference_api::warm_model_instance` -- never through this
    /// primitive directly (Correctif: Runtime-owned ModelInstance readiness
    /// authority).
    pub(crate) fn transition_to(
        &mut self,
        next: ModelInstanceLifecycleState,
    ) -> Result<(), ModelInstanceError> {
        if !self.lifecycle.allows_transition_to(next) {
            return Err(ModelInstanceError::InvalidLifecycleTransition {
                from: self.lifecycle,
                to: next,
            });
        }
        self.lifecycle = next;
        self.readiness = readiness_for_lifecycle(next);
        Ok(())
    }

    /// Crate-internal only. Transitions straight to `Ready` with no
    /// evidence check of its own -- callers of this method are themselves
    /// responsible for having already verified real readiness (the
    /// materialization transaction's `commit`, or `warmup`'s own
    /// Runtime-derived checks). A caller outside this crate cannot reach
    /// this method at all (Correctif: Runtime-owned ModelInstance readiness
    /// authority; this was previously `pub` and directly callable via
    /// `Runtime::model_instances_mut()`, the exact bypass an external audit
    /// of PR #36 demonstrated).
    pub(crate) fn mark_ready(&mut self) -> Result<(), ModelInstanceError> {
        match self.lifecycle {
            ModelInstanceLifecycleState::Creating => {
                self.transition_to(ModelInstanceLifecycleState::Loading)?;
                self.transition_to(ModelInstanceLifecycleState::Ready)
            }
            ModelInstanceLifecycleState::Loading | ModelInstanceLifecycleState::Warming => {
                self.transition_to(ModelInstanceLifecycleState::Ready)
            }
            _ if self.readiness.accepts_generation() => Ok(()),
            _ => Err(ModelInstanceError::ModelInstanceNotReady),
        }
    }

    pub fn validate_creation(
        &self,
        checks: &ModelInstanceCreationChecks,
    ) -> Result<(), ModelInstanceError> {
        checks.validate()
    }

    pub fn validate_readiness(
        &mut self,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        let result = checks.validate();
        let computed = checks.readiness();
        // `Ready` readiness is only meaningful once the lifecycle itself
        // has actually reached (or is transitioning through, e.g.
        // `Warming`) a state that legitimately allows inference use --
        // otherwise this would publish `Ready` readiness on an instance
        // still sitting in `Creating`/`Loading`, the exact
        // lifecycle/readiness inconsistency `WarmupPolicy::Disabled`
        // could previously produce by calling this method without ever
        // transitioning the lifecycle (Correctif: Runtime-owned
        // ModelInstance readiness authority).
        self.readiness = if computed == ModelInstanceReadiness::Ready
            && !self.lifecycle.supports_inference_use()
            && self.lifecycle != ModelInstanceLifecycleState::Warming
        {
            ModelInstanceReadiness::NotReady
        } else {
            computed
        };
        if result.is_err() {
            self.last_error = result.clone().err();
        }
        result
    }

    /// Crate-internal only: `checks` here are trusted as-is, with no
    /// Runtime-side derivation of their own. The public entry point is
    /// `crate::inference_api::warm_model_instance`, which derives the
    /// Runtime-observable facts before calling this (Correctif:
    /// Runtime-owned ModelInstance readiness authority; a direct external
    /// call to this method, bypassing that derivation, was the exact
    /// second bypass an external audit of PR #36 demonstrated alongside
    /// `mark_ready`).
    pub(crate) fn warmup(
        &mut self,
        plan: &ModelInstanceWarmupPlan,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        if plan.policy == ModelInstanceWarmupPolicy::Disabled {
            return self.validate_readiness(checks);
        }
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Creating | ModelInstanceLifecycleState::Loading
        ) {
            if self.lifecycle == ModelInstanceLifecycleState::Creating {
                self.transition_to(ModelInstanceLifecycleState::Loading)?;
            }
            self.transition_to(ModelInstanceLifecycleState::Warming)?;
        }
        match self.validate_readiness(checks) {
            Ok(()) => self.transition_to(ModelInstanceLifecycleState::Ready),
            Err(error) => {
                self.lifecycle = ModelInstanceLifecycleState::Failed;
                self.readiness = ModelInstanceReadiness::Failed;
                self.last_error = Some(error.clone());
                Err(match error {
                    ModelInstanceError::ModelInstanceProviderNotReady => {
                        ModelInstanceError::ModelInstanceProviderNotReady
                    }
                    ModelInstanceError::ModelInstanceAdapterIncompatible => {
                        ModelInstanceError::ModelInstanceAdapterIncompatible
                    }
                    _ => ModelInstanceError::ModelInstanceWarmupFailed,
                })
            }
        }
    }

    pub fn acquire_usage(&mut self, now_millis: u64) -> Result<(), ModelInstanceError> {
        // Both conditions are required, not just readiness: a lifecycle
        // that has not actually reached a usable state must reject usage
        // even if `readiness` was somehow (forged, or left inconsistent by
        // a caller-driven readiness update) reported as `Ready`
        // (Correctif: Runtime-owned ModelInstance readiness authority).
        if !self.lifecycle.supports_inference_use() || !self.readiness.accepts_generation() {
            return Err(readiness_error(self.lifecycle, self.readiness));
        }
        self.definition.usage.active_operation_count = self
            .definition
            .usage
            .active_operation_count
            .saturating_add(1);
        self.definition.usage.total_request_count =
            self.definition.usage.total_request_count.saturating_add(1);
        self.definition.usage.last_used_millis = Some(now_millis);
        if self.lifecycle == ModelInstanceLifecycleState::Ready
            || self.lifecycle == ModelInstanceLifecycleState::Idle
        {
            self.lifecycle = ModelInstanceLifecycleState::Active;
        }
        Ok(())
    }

    pub fn release_usage(&mut self) -> Result<(), ModelInstanceError> {
        if self.definition.usage.active_operation_count == 0 {
            return Err(ModelInstanceError::InternalModelInstance {
                reason: "no active model instance operation to release".into(),
            });
        }
        self.definition.usage.active_operation_count -= 1;
        if self.definition.usage.active_operation_count == 0
            && self.lifecycle == ModelInstanceLifecycleState::Active
        {
            self.lifecycle = ModelInstanceLifecycleState::Idle;
            self.readiness = ModelInstanceReadiness::Ready;
        }
        Ok(())
    }

    pub fn can_unload(&self, policy: ModelInstanceUnloadPolicy) -> bool {
        match policy {
            ModelInstanceUnloadPolicy::ForceInvalidate => true,
            ModelInstanceUnloadPolicy::RejectActiveUse
            | ModelInstanceUnloadPolicy::DrainActiveUse => !self.definition.usage.has_active_use(),
        }
    }

    pub fn record_mutation(&mut self, _kind: ModelInstanceMutationKind) {
        self.definition.mutation_version = self.definition.mutation_version.saturating_add(1);
    }

    pub fn activate_adapters(
        &mut self,
        adapter_set: AdapterSetId,
        scope: impl Into<String>,
        merged: bool,
    ) -> ModelInstanceInvalidationReport {
        self.definition.adapter_state.active_adapter_set = Some(adapter_set.clone());
        self.definition.adapter_state.activation_scope = Some(scope.into());
        self.definition.adapter_state.merged = merged;
        self.definition
            .usage
            .adapter_dependencies
            .insert(adapter_set);
        self.record_mutation(ModelInstanceMutationKind::AdapterMerge);
        self.invalidate_cache_dependencies()
    }

    pub fn invalidate_cache_dependencies(&mut self) -> ModelInstanceInvalidationReport {
        ModelInstanceInvalidationReport {
            kv_caches: std::mem::take(&mut self.definition.usage.kv_cache_dependencies),
            prefix_entries: std::mem::take(&mut self.definition.usage.prefix_cache_dependencies),
            adapters_released: BTreeSet::new(),
        }
    }

    pub fn can_share_with(&self, other: &ModelInstanceSharingContext) -> bool {
        match self.definition.policy.sharing {
            ModelInstanceSharingPolicy::Private => false,
            ModelInstanceSharingPolicy::RuntimeLocal => {
                !other.kv_cache_private
                    && !other.prefix_cache_private
                    && self.definition.adapter_state == other.adapter_state
                    && self
                        .definition
                        .placement
                        .affinity
                        .validate_with(&other.affinity)
                        .is_ok()
            }
            ModelInstanceSharingPolicy::TenantIsolated => {
                self.definition.tenant.is_some()
                    && self.definition.tenant == other.tenant
                    && self.definition.adapter_state == other.adapter_state
            }
            ModelInstanceSharingPolicy::PolicyControlled => {
                self.definition.owner == other.owner
                    && self.definition.adapter_state == other.adapter_state
                    && !other.kv_cache_private
                    && !other.prefix_cache_private
            }
        }
    }

    pub fn suspend(
        &mut self,
        _reason: ModelInstanceSuspensionReason,
    ) -> Result<(), ModelInstanceError> {
        if !self.definition.policy.suspension_allowed {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        if self.definition.usage.has_active_use() {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
        ) {
            self.transition_to(ModelInstanceLifecycleState::Suspended)
        } else {
            Err(readiness_error(self.lifecycle, self.readiness))
        }
    }

    /// Transitions `Suspended -> Loading` only. Reaching `Ready` from
    /// there requires fresh Runtime-derived readiness evidence -- the
    /// state that made this instance eligible for suspension (Provider,
    /// Device, weight materialization) may have changed while it was
    /// suspended, so resuming SHALL NOT jump straight back to `Ready`
    /// without revalidation. `crate::inference_api::resume_model_instance`
    /// performs that revalidation after calling this (Correctif:
    /// Runtime-owned ModelInstance readiness authority, round 3 -- this
    /// method used to transition all the way to `Ready` itself, an
    /// external audit of PR #36 found it was still a direct bypass of
    /// `warm_model_instance`'s derivation even after `mark_ready`/
    /// `transition_to`/`warmup` were sealed).
    pub fn resume(&mut self) -> Result<(), ModelInstanceError> {
        if self.lifecycle == ModelInstanceLifecycleState::Suspended {
            self.transition_to(ModelInstanceLifecycleState::Loading)
        } else {
            Err(ModelInstanceError::ModelInstanceNotReady)
        }
    }

    pub fn drain(&mut self) -> Result<(), ModelInstanceError> {
        if matches!(
            self.lifecycle,
            ModelInstanceLifecycleState::Ready
                | ModelInstanceLifecycleState::Idle
                | ModelInstanceLifecycleState::Active
        ) {
            self.transition_to(ModelInstanceLifecycleState::Draining)
        } else {
            Err(readiness_error(self.lifecycle, self.readiness))
        }
    }

    pub fn fail(&mut self, error: ModelInstanceError) {
        self.lifecycle = ModelInstanceLifecycleState::Failed;
        self.readiness = ModelInstanceReadiness::Failed;
        self.definition.usage.failure_count = self.definition.usage.failure_count.saturating_add(1);
        self.last_error = Some(error);
    }

    pub fn invalidate(&mut self, error: ModelInstanceError) {
        self.lifecycle = ModelInstanceLifecycleState::Invalid;
        self.readiness = ModelInstanceReadiness::Failed;
        self.last_error = Some(error);
    }

    pub fn provider_status_changed(
        &mut self,
        health: ProviderHealthState,
        readiness: ProviderReadinessState,
        pressure: ProviderPressureLevel,
        admission: ProviderAdmissionDecision,
    ) -> Result<(), ModelInstanceError> {
        if matches!(
            health,
            ProviderHealthState::Unhealthy | ProviderHealthState::Failed
        ) {
            self.fail(ModelInstanceError::ModelInstanceProviderFailed);
            return Err(ModelInstanceError::ModelInstanceProviderFailed);
        }
        if matches!(
            readiness,
            ProviderReadinessState::NotReady | ProviderReadinessState::Draining
        ) {
            if readiness == ProviderReadinessState::Draining {
                self.drain()?;
                return Err(ModelInstanceError::ModelInstanceDraining);
            }
            self.readiness = ModelInstanceReadiness::NotReady;
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        if matches!(
            pressure,
            ProviderPressureLevel::High | ProviderPressureLevel::Saturated
        ) || admission == ProviderAdmissionDecision::Reject
        {
            if self.definition.policy.suspension_allowed && !self.definition.usage.has_active_use()
            {
                self.suspend(ModelInstanceSuspensionReason::ProviderPressure)?;
            }
            return Err(ModelInstanceError::ModelInstanceProviderNotReady);
        }
        Ok(())
    }

    pub fn device_unavailable(&mut self, lost: bool) -> Result<(), ModelInstanceError> {
        if lost {
            self.suspend(ModelInstanceSuspensionReason::TemporaryResourceLoss)?;
            return Err(ModelInstanceError::ModelInstanceDeviceLost);
        }
        self.suspend(ModelInstanceSuspensionReason::DevicePressure)?;
        Err(ModelInstanceError::ModelInstanceDeviceUnavailable)
    }

    pub fn browser_supported(&self) -> Result<(), ModelInstanceError> {
        if self
            .definition
            .policy
            .browser_linear_memory_limit_bytes
            .is_some_and(|limit| self.definition.usage.residency_bytes > limit)
        {
            return Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported);
        }
        Ok(())
    }

    pub fn status(&self) -> ModelInstanceStatus {
        ModelInstanceStatus {
            id: self.id.clone(),
            artifact: self.definition.artifact.clone(),
            lifecycle: self.lifecycle,
            readiness: self.readiness,
            active_operation_count: self.definition.usage.active_operation_count,
            active_session_count: self.definition.usage.active_session_count,
            queued_operation_count: self.definition.usage.queued_operation_count,
            total_request_count: self.definition.usage.total_request_count,
            residency_bytes: self.definition.usage.residency_bytes,
            mutation_version: self.definition.mutation_version,
            raw_weights_available: false,
            raw_provider_handle_available: false,
            raw_device_handle_available: false,
            raw_memory_pointer_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceStatus {
    pub id: ModelInstanceId,
    pub artifact: ModelArtifactId,
    pub lifecycle: ModelInstanceLifecycleState,
    pub readiness: ModelInstanceReadiness,
    pub active_operation_count: usize,
    pub active_session_count: usize,
    pub queued_operation_count: usize,
    pub total_request_count: u64,
    pub residency_bytes: u64,
    pub mutation_version: u64,
    pub raw_weights_available: bool,
    pub raw_provider_handle_available: bool,
    pub raw_device_handle_available: bool,
    pub raw_memory_pointer_available: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceObservationKind {
    CreationRequested,
    Created,
    Loading,
    Warming,
    Ready,
    Active,
    Idle,
    Draining,
    Suspended,
    Reloading,
    Unloading,
    Unloaded,
    Failed,
    Invalidated,
    Removed,
    UsageAcquired,
    UsageReleased,
    SharingDenied,
    CacheInvalidation,
    MemoryPressure,
    ProviderPressure,
    DeviceUnavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceObservation {
    pub kind: ModelInstanceObservationKind,
    pub instance: Option<ModelInstanceId>,
    pub message: String,
    pub correlation_id: Option<CorrelationId>,
    pub raw_weights_available: bool,
    pub raw_prompt_available: bool,
    pub raw_cache_available: bool,
    pub raw_provider_handle_available: bool,
    pub raw_device_handle_available: bool,
}

impl ModelInstanceObservation {
    pub fn redacted(
        kind: ModelInstanceObservationKind,
        instance: Option<ModelInstanceId>,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) -> Self {
        Self {
            kind,
            instance,
            message: message.into(),
            correlation_id,
            raw_weights_available: false,
            raw_prompt_available: false,
            raw_cache_available: false,
            raw_provider_handle_available: false,
            raw_device_handle_available: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelInstanceManager {
    next_id: u64,
    instances: BTreeMap<ModelInstanceId, ModelInstance>,
    observations: Vec<ModelInstanceObservation>,
    materialization_evidence: BTreeMap<ModelInstanceId, MaterializationEvidence>,
}

impl ModelInstanceManager {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            instances: BTreeMap::new(),
            observations: Vec::new(),
            materialization_evidence: BTreeMap::new(),
        }
    }

    /// Mints (or replaces) `instance`'s materialization evidence. Called
    /// only by `WeightMaterializationTransaction::commit`
    /// (`first_native_runtime.rs`) once every weight in an attempt has
    /// staged successfully.
    pub(crate) fn record_materialization_evidence(
        &mut self,
        instance: &ModelInstanceId,
        evidence: MaterializationEvidence,
    ) {
        self.materialization_evidence
            .insert(instance.clone(), evidence);
    }

    /// `instance`'s current materialization evidence, if any. Used by
    /// `derive_effective_readiness_checks` (`inference_api.rs`) to derive
    /// `weights_materialized`.
    pub(crate) fn materialization_evidence(
        &self,
        instance: &ModelInstanceId,
    ) -> Option<&MaterializationEvidence> {
        self.materialization_evidence.get(instance)
    }

    /// Clears `instance`'s materialization evidence, if any. Called on
    /// unload and on a failed materialization attempt's rollback, mirroring
    /// `TensorResidency` cleanup (`invalidate-tensor-residency-on-release`).
    pub(crate) fn clear_materialization_evidence(&mut self, instance: &ModelInstanceId) {
        self.materialization_evidence.remove(instance);
    }

    pub fn instances(&self) -> impl Iterator<Item = &ModelInstance> {
        self.instances.values()
    }

    pub fn observations(&self) -> &[ModelInstanceObservation] {
        &self.observations
    }

    pub fn instance(&self, id: &ModelInstanceId) -> Result<&ModelInstance, ModelInstanceError> {
        self.instances
            .get(id)
            .ok_or(ModelInstanceError::ModelInstanceNotFound)
    }

    pub fn instance_mut(
        &mut self,
        id: &ModelInstanceId,
    ) -> Result<&mut ModelInstance, ModelInstanceError> {
        self.instances
            .get_mut(id)
            .ok_or(ModelInstanceError::ModelInstanceNotFound)
    }

    pub fn create(
        &mut self,
        mut definition: ModelInstanceDefinition,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        definition
            .policy
            .browser_linear_memory_limit_bytes
            .map(|limit| definition.usage.residency_bytes <= limit)
            .unwrap_or(true)
            .then_some(())
            .ok_or(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported)?;
        if definition.residencies.is_empty() {
            return Err(ModelInstanceError::ModelInstanceResidencyMissing);
        }
        // A newly created instance SHALL start with no weight resource
        // bindings, regardless of what the caller-supplied `definition`
        // carried -- `ModelInstanceDefinition` is `Clone`, and `Clone`
        // copies `resource_bindings` (and every other field) regardless of
        // their own visibility, so field-sealing `resource_bindings` alone
        // does not stop a caller from cloning an already-`Ready`
        // instance's definition and passing it here (directly, or via
        // `reload`, which also calls `create`) to fabricate a *new*
        // instance that already appears materialized -- for real Provider
        // resources still owned by the original instance. Only this
        // instance's own future `WeightMaterializationTransaction::commit`
        // calls may populate these fields again (a further audit of PR
        // #36 found this exact cross-instance resource-aliasing path).
        definition.resource_bindings = ModelInstanceResourceBindings::default();
        let id = ModelInstanceId::runtime_issued(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.observe(
            ModelInstanceObservationKind::CreationRequested,
            Some(id.clone()),
            "model instance creation requested",
            None,
        );
        let mut instance = ModelInstance::new(id.clone(), definition);
        instance.transition_to(ModelInstanceLifecycleState::Loading)?;
        self.instances.insert(id.clone(), instance);
        self.observe(
            ModelInstanceObservationKind::Created,
            Some(id.clone()),
            "model instance created",
            None,
        );
        Ok(id)
    }

    pub fn create_checked(
        &mut self,
        definition: ModelInstanceDefinition,
        checks: &ModelInstanceCreationChecks,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        checks.validate()?;
        self.create(definition)
    }

    /// Transitions a Model Instance from `Loading` (or `Warming`) to
    /// `Ready`, mirroring [`Self::warmup`]'s exact pattern: call into the
    /// individual instance's own state-machine method, then emit the
    /// matching observation based on the real outcome. `create()` no
    /// longer reaches `Ready` on its own -- a caller with a readiness
    /// condition to satisfy first (weight materialization, warmup, or any
    /// other mandatory readiness check) SHALL call this only after that
    /// condition genuinely holds, per `model-instance`'s "Model Instance
    /// Creation" requirement (creation alone SHALL NOT produce a Ready
    /// instance).
    /// Crate-internal only -- see `ModelInstance::mark_ready` above.
    pub(crate) fn mark_ready(&mut self, id: &ModelInstanceId) -> Result<(), ModelInstanceError> {
        let result = self.instance_mut(id)?.mark_ready();
        self.observe(
            if result.is_ok() {
                ModelInstanceObservationKind::Ready
            } else {
                ModelInstanceObservationKind::Failed
            },
            Some(id.clone()),
            if result.is_ok() {
                "model instance ready"
            } else {
                "model instance mark-ready failed"
            },
            None,
        );
        result
    }

    /// Crate-internal only -- see `ModelInstance::warmup` above.
    pub(crate) fn warmup(
        &mut self,
        id: &ModelInstanceId,
        plan: &ModelInstanceWarmupPlan,
        checks: &ModelInstanceReadinessChecks,
    ) -> Result<(), ModelInstanceError> {
        let result = self.instance_mut(id)?.warmup(plan, checks);
        self.observe(
            if result.is_ok() {
                ModelInstanceObservationKind::Ready
            } else {
                ModelInstanceObservationKind::Failed
            },
            Some(id.clone()),
            if result.is_ok() {
                "model instance warmup completed"
            } else {
                "model instance warmup failed"
            },
            None,
        );
        result
    }

    pub fn acquire_usage(
        &mut self,
        id: &ModelInstanceId,
        now_millis: u64,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.acquire_usage(now_millis)?;
        self.observe(
            ModelInstanceObservationKind::UsageAcquired,
            Some(id.clone()),
            "model instance usage acquired",
            None,
        );
        Ok(())
    }

    pub fn release_usage(&mut self, id: &ModelInstanceId) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.release_usage()?;
        self.observe(
            ModelInstanceObservationKind::UsageReleased,
            Some(id.clone()),
            "model instance usage released",
            None,
        );
        Ok(())
    }

    pub fn generation_reference(
        &self,
        id: &ModelInstanceId,
    ) -> Result<GenerationModelReference, ModelInstanceError> {
        let instance = self.instance(id)?;
        if !instance.lifecycle.supports_inference_use() || !instance.readiness.accepts_generation()
        {
            return Err(readiness_error(instance.lifecycle, instance.readiness));
        }
        Ok(GenerationModelReference::ModelInstance(id.clone()))
    }

    pub fn unload(
        &mut self,
        id: &ModelInstanceId,
        policy: ModelInstanceUnloadPolicy,
    ) -> Result<ModelInstanceUnloadReport, ModelInstanceError> {
        if !self.instance(id)?.can_unload(policy) {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        let report = self.prepare_unload_report(id)?;
        let instance = self.instance_mut(id)?;
        if matches!(
            instance.lifecycle,
            ModelInstanceLifecycleState::Ready
                | ModelInstanceLifecycleState::Idle
                | ModelInstanceLifecycleState::Suspended
                | ModelInstanceLifecycleState::Failed
                | ModelInstanceLifecycleState::Invalid
        ) {
            if matches!(
                instance.lifecycle,
                ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
            ) {
                instance.transition_to(ModelInstanceLifecycleState::Draining)?;
            }
            instance.transition_to(ModelInstanceLifecycleState::Unloading)?;
            instance.transition_to(ModelInstanceLifecycleState::Unloaded)?;
            self.observe(
                ModelInstanceObservationKind::Unloaded,
                Some(id.clone()),
                "model instance unloaded",
                None,
            );
            Ok(report)
        } else {
            Err(readiness_error(instance.lifecycle, instance.readiness))
        }
    }

    pub fn reload(
        &mut self,
        id: &ModelInstanceId,
        request: ModelInstanceReloadRequest,
    ) -> Result<ModelInstanceId, ModelInstanceError> {
        if !request.allow_active_semantic_mutation
            && self.instance(id)?.definition.usage.has_active_use()
        {
            return Err(ModelInstanceError::ModelInstanceActive);
        }
        self.unload(id, ModelInstanceUnloadPolicy::DrainActiveUse)?;
        let replacement = self.create(request.replacement)?;
        self.observe(
            ModelInstanceObservationKind::Reloading,
            Some(id.clone()),
            "model instance reload created replacement",
            None,
        );
        Ok(replacement)
    }

    pub fn activate_adapters(
        &mut self,
        id: &ModelInstanceId,
        adapter_set: AdapterSetId,
        scope: impl Into<String>,
        merged: bool,
    ) -> Result<ModelInstanceInvalidationReport, ModelInstanceError> {
        let report = self
            .instance_mut(id)?
            .activate_adapters(adapter_set, scope, merged);
        self.observe(
            ModelInstanceObservationKind::CacheInvalidation,
            Some(id.clone()),
            "model instance adapter change invalidated dependent caches",
            None,
        );
        Ok(report)
    }

    pub fn invalidate_for_mutation(
        &mut self,
        id: &ModelInstanceId,
        kind: ModelInstanceMutationKind,
    ) -> Result<ModelInstanceInvalidationReport, ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        instance.record_mutation(kind);
        let report = instance.invalidate_cache_dependencies();
        self.observe(
            ModelInstanceObservationKind::CacheInvalidation,
            Some(id.clone()),
            "model instance semantic mutation invalidated dependent caches",
            None,
        );
        Ok(report)
    }

    pub fn fail_instance(
        &mut self,
        id: &ModelInstanceId,
        error: ModelInstanceError,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.fail(error);
        self.observe(
            ModelInstanceObservationKind::Failed,
            Some(id.clone()),
            "model instance failed",
            None,
        );
        Ok(())
    }

    pub fn invalidate_instance(
        &mut self,
        id: &ModelInstanceId,
        error: ModelInstanceError,
    ) -> Result<(), ModelInstanceError> {
        self.instance_mut(id)?.invalidate(error);
        self.observe(
            ModelInstanceObservationKind::Invalidated,
            Some(id.clone()),
            "model instance invalidated",
            None,
        );
        Ok(())
    }

    fn prepare_unload_report(
        &mut self,
        id: &ModelInstanceId,
    ) -> Result<ModelInstanceUnloadReport, ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        let invalidated = ModelInstanceInvalidationReport {
            kv_caches: std::mem::take(&mut instance.definition.usage.kv_cache_dependencies),
            prefix_entries: std::mem::take(
                &mut instance.definition.usage.prefix_cache_dependencies,
            ),
            adapters_released: std::mem::take(&mut instance.definition.usage.adapter_dependencies),
        };
        let released_memory_allocations =
            std::mem::take(&mut instance.definition.resource_bindings.memory_allocations);
        instance
            .definition
            .resource_bindings
            .released_memory_allocations
            .extend(released_memory_allocations.iter().copied());
        // `transactional-weight-materialization`: unload must release the
        // Provider-owned weight Tensor Resources themselves, not only their
        // Memory Manager allocation accounting -- otherwise Provider
        // storage accumulates orphaned weight tensors across every
        // load/unload cycle even though the Memory Manager ledger looks
        // clean.
        let released_weight_resources: BTreeSet<TensorResourceId> = instance
            .definition
            .resource_bindings
            .weights
            .values()
            .cloned()
            .collect();
        instance.definition.resource_bindings.weights.clear();
        let released_provider_resources = instance
            .definition
            .placement
            .provider_resource
            .take()
            .map(|resource| BTreeSet::from([resource.provider]))
            .unwrap_or_default();
        instance
            .definition
            .resource_bindings
            .released_provider_resources
            .extend(released_provider_resources.iter().cloned());
        instance.definition.associated_sessions.clear();
        Ok(ModelInstanceUnloadReport {
            invalidated,
            released_memory_allocations,
            released_weight_resources,
            released_provider_resources,
            dangling_session_references: false,
        })
    }

    pub fn mark_memory_pressure(
        &mut self,
        id: &ModelInstanceId,
        pressure: MemoryPressureLevel,
    ) -> Result<(), ModelInstanceError> {
        let instance = self.instance_mut(id)?;
        if pressure != MemoryPressureLevel::Low && instance.definition.policy.suspension_allowed {
            if matches!(
                instance.lifecycle,
                ModelInstanceLifecycleState::Ready | ModelInstanceLifecycleState::Idle
            ) {
                instance.transition_to(ModelInstanceLifecycleState::Suspended)?;
            }
            self.observe(
                ModelInstanceObservationKind::MemoryPressure,
                Some(id.clone()),
                "model instance memory pressure observed",
                None,
            );
        }
        Ok(())
    }

    fn observe(
        &mut self,
        kind: ModelInstanceObservationKind,
        instance: Option<ModelInstanceId>,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) {
        self.observations.push(ModelInstanceObservation::redacted(
            kind,
            instance,
            message,
            correlation_id,
        ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelInstanceError {
    ModelInstanceNotFound,
    ModelInstanceNotReady,
    ModelInstanceLoading,
    ModelInstanceWarming,
    ModelInstanceDraining,
    ModelInstanceSuspended,
    ModelInstanceUnloading,
    ModelInstanceUnloaded,
    ModelInstanceFailed,
    ModelInstanceInvalid,
    ModelInstanceRemoved,
    ModelInstanceActive,
    ModelInstanceBusy,
    ModelInstanceSharingDenied,
    ModelInstancePolicyDenied,
    ModelInstanceReloadRequired,
    ModelInstanceReloadFailed,
    ModelInstanceUnloadFailed,
    ModelInstanceWarmupFailed,
    ModelInstanceProviderUnavailable,
    ModelInstanceProviderNotReady,
    ModelInstanceProviderFailed,
    ModelInstanceDeviceUnavailable,
    ModelInstanceDeviceLost,
    ModelInstanceMemoryPressure,
    ModelInstanceResidencyMissing,
    ModelInstanceAdapterIncompatible,
    ModelInstanceKernelPreparationFailed,
    ModelInstanceAutotuningIncomplete,
    ModelInstanceWeightsNotMaterialized,
    ModelInstanceKvCacheInvalidated,
    ModelInstancePrefixCacheInvalidated,
    ModelInstanceBrowserFeatureUnsupported,
    InvalidLifecycleTransition {
        from: ModelInstanceLifecycleState,
        to: ModelInstanceLifecycleState,
    },
    InternalModelInstance {
        reason: String,
    },
}

impl fmt::Display for ModelInstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelInstanceNotFound => f.write_str("model instance not found"),
            Self::ModelInstanceNotReady => f.write_str("model instance not ready"),
            Self::ModelInstanceLoading => f.write_str("model instance loading"),
            Self::ModelInstanceWarming => f.write_str("model instance warming"),
            Self::ModelInstanceDraining => f.write_str("model instance draining"),
            Self::ModelInstanceSuspended => f.write_str("model instance suspended"),
            Self::ModelInstanceUnloading => f.write_str("model instance unloading"),
            Self::ModelInstanceUnloaded => f.write_str("model instance unloaded"),
            Self::ModelInstanceFailed => f.write_str("model instance failed"),
            Self::ModelInstanceInvalid => f.write_str("model instance invalid"),
            Self::ModelInstanceRemoved => f.write_str("model instance removed"),
            Self::ModelInstanceActive => f.write_str("model instance active"),
            Self::ModelInstanceBusy => f.write_str("model instance busy"),
            Self::ModelInstanceSharingDenied => f.write_str("model instance sharing denied"),
            Self::ModelInstancePolicyDenied => f.write_str("model instance policy denied"),
            Self::ModelInstanceReloadRequired => f.write_str("model instance reload required"),
            Self::ModelInstanceReloadFailed => f.write_str("model instance reload failed"),
            Self::ModelInstanceUnloadFailed => f.write_str("model instance unload failed"),
            Self::ModelInstanceWarmupFailed => f.write_str("model instance warmup failed"),
            Self::ModelInstanceProviderUnavailable => {
                f.write_str("model instance Provider unavailable")
            }
            Self::ModelInstanceProviderNotReady => f.write_str("model instance Provider not ready"),
            Self::ModelInstanceProviderFailed => f.write_str("model instance Provider failed"),
            Self::ModelInstanceDeviceUnavailable => {
                f.write_str("model instance Device unavailable")
            }
            Self::ModelInstanceDeviceLost => f.write_str("model instance Device lost"),
            Self::ModelInstanceMemoryPressure => f.write_str("model instance memory pressure"),
            Self::ModelInstanceResidencyMissing => f.write_str("model instance residency missing"),
            Self::ModelInstanceAdapterIncompatible => {
                f.write_str("model instance adapter incompatible")
            }
            Self::ModelInstanceKernelPreparationFailed => {
                f.write_str("model instance required Kernel preparation failed")
            }
            Self::ModelInstanceAutotuningIncomplete => {
                f.write_str("model instance required Kernel Autotuning incomplete")
            }
            Self::ModelInstanceWeightsNotMaterialized => {
                f.write_str("model instance weights not materialized")
            }
            Self::ModelInstanceKvCacheInvalidated => {
                f.write_str("model instance KV cache invalidated")
            }
            Self::ModelInstancePrefixCacheInvalidated => {
                f.write_str("model instance Prefix Cache invalidated")
            }
            Self::ModelInstanceBrowserFeatureUnsupported => {
                f.write_str("model instance browser feature unsupported")
            }
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid model instance transition from {from:?} to {to:?}"
                )
            }
            Self::InternalModelInstance { reason } => {
                write!(f, "internal model instance: {reason}")
            }
        }
    }
}

impl Error for ModelInstanceError {}

pub fn readiness_for_lifecycle(lifecycle: ModelInstanceLifecycleState) -> ModelInstanceReadiness {
    match lifecycle {
        ModelInstanceLifecycleState::Ready
        | ModelInstanceLifecycleState::Active
        | ModelInstanceLifecycleState::Idle => ModelInstanceReadiness::Ready,
        ModelInstanceLifecycleState::Draining => ModelInstanceReadiness::Draining,
        ModelInstanceLifecycleState::Suspended => ModelInstanceReadiness::Suspended,
        ModelInstanceLifecycleState::Failed
        | ModelInstanceLifecycleState::Invalid
        | ModelInstanceLifecycleState::Removed => ModelInstanceReadiness::Failed,
        _ => ModelInstanceReadiness::NotReady,
    }
}

pub fn readiness_error(
    lifecycle: ModelInstanceLifecycleState,
    readiness: ModelInstanceReadiness,
) -> ModelInstanceError {
    match lifecycle {
        ModelInstanceLifecycleState::Loading => ModelInstanceError::ModelInstanceLoading,
        ModelInstanceLifecycleState::Warming => ModelInstanceError::ModelInstanceWarming,
        ModelInstanceLifecycleState::Draining => ModelInstanceError::ModelInstanceDraining,
        ModelInstanceLifecycleState::Suspended => ModelInstanceError::ModelInstanceSuspended,
        ModelInstanceLifecycleState::Unloading => ModelInstanceError::ModelInstanceUnloading,
        ModelInstanceLifecycleState::Unloaded => ModelInstanceError::ModelInstanceUnloaded,
        ModelInstanceLifecycleState::Failed => ModelInstanceError::ModelInstanceFailed,
        ModelInstanceLifecycleState::Invalid => ModelInstanceError::ModelInstanceInvalid,
        ModelInstanceLifecycleState::Removed => ModelInstanceError::ModelInstanceRemoved,
        _ if readiness == ModelInstanceReadiness::Failed => ModelInstanceError::ModelInstanceFailed,
        _ => ModelInstanceError::ModelInstanceNotReady,
    }
}

// ---------------------------------------------------------------------
// Generated Kernel Selection Policy
// ---------------------------------------------------------------------

/// Implements "Model Instance Interaction" from
/// `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract`:
/// "A Model Instance MAY use dynamic Kernel selection or may pin a Kernel
/// generation set for reproducibility. The policy SHALL be explicit." The
/// enum itself makes the choice explicit -- there is no implicit third
/// state, and [`PinnedKernelSelection`] carries only stable identity
/// metadata, never a native handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelSelectionPolicy {
    Dynamic,
    Pinned(PinnedKernelSelection),
}

impl KernelSelectionPolicy {
    pub const fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned(_))
    }
}

/// Implements "Session Interaction"
/// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
/// "Inference Sessions SHALL NOT own native Kernel state directly. Sessions
/// MAY inherit Model Instance Kernel selection policy." A Session resolves
/// its effective Kernel policy by borrowing its owning Model Instance's
/// policy through this function rather than storing a competing policy of
/// its own -- `crate::session::InferenceSession` has no
/// [`KernelSelectionPolicy`] field, so there is nothing for a Session to
/// override.
pub fn session_kernel_policy_is_inherited(
    model_instance_policy: &KernelSelectionPolicy,
) -> &KernelSelectionPolicy {
    model_instance_policy
}

/// Implements "Reproducible Mode" / "Reproducible Mode Prevents Adaptation"
/// (`define-kernel-performance-model-and-adaptive-feedback-contract`):
/// "Pinned reproducible execution SHALL not change Kernel from live
/// performance feedback." A pinned [`KernelSelectionPolicy`] always forces
/// [`KernelPerformanceFeedbackMode::Pinned`], overriding
/// [`ModelInstancePolicy::performance_feedback`] regardless of what it
/// requested.
pub fn effective_performance_feedback_mode(
    policy: &ModelInstancePolicy,
    kernel_selection: &KernelSelectionPolicy,
) -> KernelPerformanceFeedbackMode {
    reproducible_mode_blocks_adaptation(kernel_selection.is_pinned(), policy.performance_feedback)
}

/// A pinned Kernel selection, implementing "For strict reproducibility, a
/// Model Instance SHOULD be able to pin: KernelId, artifact digest, Prepared
/// generation, qualification profile" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PinnedKernelSelection {
    pub kernel: KernelId,
    pub artifact_digest: String,
    pub prepared_generation: Option<u64>,
    pub qualification_profile: Option<String>,
    /// Implements "Reproducible Model Instance May Pin Tuning Record"
    /// (`define-kernel-runtime-autotuning-and-specialization-contract`): a
    /// pinned `KernelAutotuningRecord` fingerprint the Model Instance
    /// consumes instead of live tuning. Live re-tuning SHALL NOT change this
    /// value.
    pub autotuning_record_fingerprint: Option<String>,
}

impl PinnedKernelSelection {
    pub fn new(kernel: KernelId, artifact_digest: impl Into<String>) -> Self {
        Self {
            kernel,
            artifact_digest: artifact_digest.into(),
            prepared_generation: None,
            qualification_profile: None,
            autotuning_record_fingerprint: None,
        }
    }

    pub fn with_autotuning_record_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.autotuning_record_fingerprint = Some(fingerprint.into());
        self
    }

    /// Implements "Record artifact digest for reproducibility" and "Record
    /// qualification profile" (tasks): a pinned selection SHALL identify
    /// which artifact and profile it pins, never leaving either implicit.
    pub fn validate(&self) -> Result<(), ModelInstanceError> {
        if self.artifact_digest.trim().is_empty() {
            return Err(ModelInstanceError::ModelInstancePolicyDenied);
        }
        Ok(())
    }
}

fn validate_instance_identity(value: &str) -> Result<(), ModelInstanceError> {
    let lower = value.to_ascii_lowercase();
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("0x")
        || lower.contains("provider")
        || lower.contains("device")
        || lower.contains("ptr")
        || lower.contains("weight")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelInstanceError::ModelInstancePolicyDenied);
    }
    Ok(())
}
