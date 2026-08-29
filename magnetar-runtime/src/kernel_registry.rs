//! Runtime-owned Kernel Registry.
//!
//! The registry stores validated Kernel advertisements and produces metadata
//! candidates for Runtime selection. It never exposes native function pointers,
//! Provider handles, or direct client authority over Provider selection.
//!
//! [`CandidateState`], [`KernelRegistry::promote_generation`],
//! [`KernelRegistry::rollback_generation`], and [`KernelRegistry::revoke_kernel`]
//! implement the Registry-side lifecycle contract from
//! `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract`:
//! atomic promotion, in-flight generation stability, safe retirement (see
//! [`KernelRegistry::retire_prepared_kernel`] /
//! [`KernelRegistry::destroy_prepared_kernel`] in
//! `kernel_artifact.rs`-backed state), rollback, and revocation. See
//! [`run_kernel_registry_lifecycle_conformance`] for the exercised
//! guarantees.

use crate::affinity::*;
use crate::compute::*;
use crate::execution_graph::*;
use crate::kernel::*;
use crate::kernel_artifact::*;
use crate::model_instance::*;
use crate::operator::*;
use std::collections::{BTreeMap, BTreeSet};
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelRegistrationAuthority {
    Provider,
    RuntimeTestFixture,
    Client,
    Component,
}

impl KernelRegistrationAuthority {
    pub const fn may_register_kernel(self) -> bool {
        matches!(self, Self::Provider | Self::RuntimeTestFixture)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelCandidateRejection {
    OperatorMismatch,
    OperatorVersionUnsupported,
    DTypeUnsupported,
    LayoutUnsupported,
    ShapeUnsupported,
    MemoryClassUnsupported,
    ExecutionModeUnsupported,
    ResourceAffinityConflict,
    ProviderUnavailable,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceIncompatible,
    ProviderFeatureMissing,
    DeviceFeatureMissing,
    WorkspaceUnavailable,
    BatchingUnsupported,
    AdapterUnsupported,
    KvCacheUnsupported,
    PrefixCacheUnsupported,
    ConformanceMissing,
    ConformanceFailed,
    PolicyDenied,
    StaleRegistryEntry,
    /// Implements "Revoked Kernel Not Selected"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// a revoked Kernel SHALL NOT receive new work.
    Revoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelRegistryError {
    RegistryUnavailable,
    AdvertisementInvalid(String),
    RegistrationDenied(KernelRegistrationAuthority),
    CandidateNotFound {
        operator: OperatorId,
    },
    CandidateIncompatible {
        reason: KernelCandidateRejection,
    },
    SelectionFailed(String),
    PolicyDenied(String),
    ConformanceRequired,
    ConformanceMissing {
        kernel: String,
    },
    ConformanceFailed {
        kernel: String,
    },
    ProviderUnavailable {
        provider: ProviderBinding,
    },
    ProviderNotReady {
        provider: ProviderBinding,
    },
    ProviderSaturated {
        provider: ProviderBinding,
    },
    DeviceUnavailable {
        device: DeviceBinding,
    },
    DeviceIncompatible {
        device: DeviceBinding,
    },
    MemoryInfeasible(String),
    WorkspaceUnavailable,
    ResourceAffinityConflict(String),
    BrowserFeatureUnsupported(String),
    Internal(String),
    /// Implements "Registry Promotion Is Explicit" and "Atomic Kernel
    /// Promotion"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// promotion requires a `Ready` Prepared Kernel generation for the
    /// target Kernel.
    PromotionNotEligible {
        kernel: String,
    },
    /// Implements "Rollback"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// no retained previous generation is available to roll back to.
    RollbackUnavailable {
        kernel: String,
    },
    /// Implements "Revoked Kernel Not Selected"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// a revoked Kernel SHALL NOT receive new work.
    KernelRevoked {
        kernel: String,
    },
    /// Implements "Add hot-swap errors" (tasks): the `kernel-hot-swap-failed`
    /// error category from the proposal's "Error Model" section.
    HotSwapFailed {
        reason: String,
    },
    /// Implements "Add retirement errors" (tasks): the
    /// `kernel-retirement-in-use` error category -- retirement/destruction
    /// was attempted while the generation is still referenced by active
    /// work.
    RetirementInUse {
        kernel: String,
    },
    /// Implements "Add retirement errors" (tasks): the
    /// `kernel-retirement-failed` error category -- retirement failed for a
    /// reason other than active references.
    RetirementFailed {
        kernel: String,
    },
}

impl KernelRegistryError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::RegistryUnavailable => "kernel-registry-unavailable",
            Self::AdvertisementInvalid(_) => "kernel-advertisement-invalid",
            Self::RegistrationDenied(_) => "kernel-registration-denied",
            Self::CandidateNotFound { .. } => "kernel-candidate-not-found",
            Self::CandidateIncompatible { .. } => "kernel-candidate-incompatible",
            Self::SelectionFailed(_) => "kernel-selection-failed",
            Self::PolicyDenied(_) => "kernel-policy-denied",
            Self::ConformanceRequired => "kernel-conformance-required",
            Self::ConformanceMissing { .. } => "kernel-conformance-missing",
            Self::ConformanceFailed { .. } => "kernel-conformance-failed",
            Self::ProviderUnavailable { .. } => "kernel-provider-unavailable",
            Self::ProviderNotReady { .. } => "kernel-provider-not-ready",
            Self::ProviderSaturated { .. } => "kernel-provider-saturated",
            Self::DeviceUnavailable { .. } => "kernel-device-unavailable",
            Self::DeviceIncompatible { .. } => "kernel-device-incompatible",
            Self::MemoryInfeasible(_) => "kernel-memory-infeasible",
            Self::WorkspaceUnavailable => "kernel-workspace-unavailable",
            Self::ResourceAffinityConflict(_) => "kernel-resource-affinity-conflict",
            Self::BrowserFeatureUnsupported(_) => "kernel-browser-feature-unsupported",
            Self::Internal(_) => "internal-kernel-registry",
            Self::PromotionNotEligible { .. } => "kernel-promotion-not-eligible",
            Self::RollbackUnavailable { .. } => "kernel-rollback-unavailable",
            Self::KernelRevoked { .. } => "kernel-revoked",
            Self::HotSwapFailed { .. } => "kernel-hot-swap-failed",
            Self::RetirementInUse { .. } => "kernel-retirement-in-use",
            Self::RetirementFailed { .. } => "kernel-retirement-failed",
        }
    }
}

impl fmt::Display for KernelRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegistryUnavailable => write!(f, "Kernel Registry is unavailable"),
            Self::AdvertisementInvalid(reason) => {
                write!(f, "Kernel advertisement is invalid: {reason}")
            }
            Self::RegistrationDenied(authority) => {
                write!(f, "Kernel registration denied for {authority:?}")
            }
            Self::CandidateNotFound { operator } => {
                write!(f, "no Kernel candidate found for Operator {operator}")
            }
            Self::CandidateIncompatible { reason } => {
                write!(f, "Kernel candidate is incompatible: {reason:?}")
            }
            Self::SelectionFailed(reason) => write!(f, "Kernel selection failed: {reason}"),
            Self::PolicyDenied(reason) => write!(f, "Kernel policy denied selection: {reason}"),
            Self::ConformanceRequired => write!(f, "Kernel conformance is required"),
            Self::ConformanceMissing { kernel } => {
                write!(f, "Kernel conformance is missing for {kernel}")
            }
            Self::ConformanceFailed { kernel } => {
                write!(f, "Kernel conformance failed for {kernel}")
            }
            Self::ProviderUnavailable { provider } => {
                write!(f, "Kernel Provider unavailable: {provider}")
            }
            Self::ProviderNotReady { provider } => {
                write!(f, "Kernel Provider not ready: {provider}")
            }
            Self::ProviderSaturated { provider } => {
                write!(f, "Kernel Provider saturated: {provider}")
            }
            Self::DeviceUnavailable { device } => {
                write!(f, "Kernel Device unavailable: {device}")
            }
            Self::DeviceIncompatible { device } => {
                write!(f, "Kernel Device incompatible: {device}")
            }
            Self::MemoryInfeasible(reason) => write!(f, "Kernel memory infeasible: {reason}"),
            Self::WorkspaceUnavailable => write!(f, "Kernel workspace unavailable"),
            Self::ResourceAffinityConflict(reason) => {
                write!(f, "Kernel Resource Affinity conflict: {reason}")
            }
            Self::BrowserFeatureUnsupported(feature) => {
                write!(f, "Kernel browser feature unsupported: {feature}")
            }
            Self::Internal(reason) => write!(f, "internal Kernel Registry error: {reason}"),
            Self::PromotionNotEligible { kernel } => {
                write!(
                    f,
                    "Kernel '{kernel}' has no Ready candidate generation to promote"
                )
            }
            Self::RollbackUnavailable { kernel } => {
                write!(f, "no rollback generation retained for Kernel '{kernel}'")
            }
            Self::KernelRevoked { kernel } => write!(f, "Kernel '{kernel}' is revoked"),
            Self::HotSwapFailed { reason } => write!(f, "Kernel hot swap failed: {reason}"),
            Self::RetirementInUse { kernel } => {
                write!(
                    f,
                    "Kernel '{kernel}' retirement is blocked while still in use"
                )
            }
            Self::RetirementFailed { kernel } => {
                write!(f, "Kernel '{kernel}' retirement failed")
            }
        }
    }
}

impl Error for KernelRegistryError {}

/// Logical Registry candidate lifecycle state, implementing "Candidate
/// State"
/// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`).
/// This is distinct from [`crate::kernel_artifact::PreparedKernelState`]
/// (the Provider-owned prepared-object state): this enum describes the
/// Registry's own view of a logical Kernel generation's eligibility for
/// dispatch, independent of the underlying native handle's lifecycle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CandidateState {
    Qualified,
    Candidate,
    Canary,
    Active,
    Retiring,
    Retired,
    Revoked,
}

impl CandidateState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Qualified, Self::Candidate)
                | (Self::Candidate, Self::Canary)
                | (Self::Candidate, Self::Active)
                | (Self::Canary, Self::Active)
                | (Self::Canary, Self::Retiring)
                | (Self::Active, Self::Retiring)
                | (Self::Retiring, Self::Retired)
                | (Self::Qualified, Self::Revoked)
                | (Self::Candidate, Self::Revoked)
                | (Self::Canary, Self::Revoked)
                | (Self::Active, Self::Revoked)
        )
    }

    pub const fn is_dispatchable(self) -> bool {
        matches!(self, Self::Active | Self::Canary)
    }
}

/// Reserved for future automatic rollback triggering, implementing "Reserve
/// automatic rollback" (proposal): "Automatic rollback MAY be supported. If
/// enabled, trigger policy SHALL be explicit." This change does not
/// implement any automatic trigger logic -- `enabled` defaults to `false`,
/// and nothing in this Registry consults this policy yet.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AutomaticRollbackPolicy {
    pub enabled: bool,
}

/// Governs [`KernelRegistry::destroy_prepared_kernel_with_rollback_window`],
/// implementing "Rollback Window" (proposal).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RollbackWindowPolicy {
    pub retain_previous_generation: bool,
}

/// Implements "Define in-flight revocation policy" (proposal): "Existing
/// invocations SHALL follow policy: allow-to-complete, cancel-if-safe,
/// fail-closed, Provider-specific." See
/// [`KernelRegistry::revoke_kernel_with_policy`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RevocationInFlightPolicy {
    AllowToComplete,
    CancelIfSafe,
    FailClosed,
    ProviderSpecific,
}

/// A continuous-batching slot's binding to a specific Prepared Kernel
/// generation, implementing "Continuous Batching" (proposal): "A batch
/// slot/in-flight operation SHALL retain a valid Kernel generation for the
/// duration required by execution semantics. New batch work MAY use the
/// newly promoted generation." See
/// [`KernelRegistry::bind_batch_slot`] and
/// [`KernelRegistry::admit_new_batch_work`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchSlotKernelBinding {
    pub slot: u64,
    pub generation: PreparedKernelId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelRegistryEntry {
    pub advertisement: KernelAdvertisement,
    pub authority: KernelRegistrationAuthority,
    pub active: bool,
    pub invalidation_reason: Option<String>,
}

impl KernelRegistryEntry {
    pub fn new(advertisement: KernelAdvertisement, authority: KernelRegistrationAuthority) -> Self {
        Self {
            advertisement,
            authority,
            active: true,
            invalidation_reason: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelectionRequest {
    pub request_id: String,
    pub operator: OperatorId,
    pub operator_version: u32,
    pub graph_plan: Option<ExecutionGraphId>,
    pub model_instance: Option<ModelInstanceId>,
    pub inputs: Vec<KernelResource>,
    pub outputs: Vec<KernelResource>,
    pub dtype_requirements: BTreeSet<ComputeDType>,
    pub layout_requirements: BTreeSet<TensorLayoutKind>,
    pub memory_class_requirements: BTreeSet<KernelMemoryClass>,
    pub affinity: ResourceAffinity,
    pub deterministic_required: bool,
    pub precision: ComputePrecision,
    pub execution_mode: Option<KernelExecutionMode>,
    pub batching: Option<KernelBatchMetadata>,
    pub kv_cache: Option<KernelKvCacheMetadata>,
    pub prefix_cache: Option<KernelPrefixCacheMetadata>,
    pub adapter_methods: BTreeSet<String>,
    pub deadline_millis: Option<u64>,
    pub policy: BTreeMap<String, String>,
    pub observability_correlation: Option<String>,
    pub require_conformance: bool,
    pub browser_target: bool,
}

impl KernelSelectionRequest {
    pub fn new(
        request_id: impl Into<String>,
        operator: OperatorId,
        affinity: ResourceAffinity,
    ) -> Self {
        let operator_version = operator.version();
        Self {
            request_id: request_id.into(),
            operator,
            operator_version,
            graph_plan: None,
            model_instance: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            dtype_requirements: BTreeSet::new(),
            layout_requirements: BTreeSet::new(),
            memory_class_requirements: BTreeSet::new(),
            affinity,
            deterministic_required: false,
            precision: ComputePrecision::Default,
            execution_mode: None,
            batching: None,
            kv_cache: None,
            prefix_cache: None,
            adapter_methods: BTreeSet::new(),
            deadline_millis: None,
            policy: BTreeMap::new(),
            observability_correlation: None,
            require_conformance: false,
            browser_target: false,
        }
    }

    pub fn with_input(mut self, input: KernelResource) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: KernelResource) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelCandidate {
    pub kernel: KernelId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub operator: OperatorId,
    pub compatible: bool,
    pub dtype_compatible: bool,
    pub layout_compatible: bool,
    pub shape_compatible: bool,
    pub memory_compatible: bool,
    pub workspace_feasible: bool,
    pub affinity_compatible: bool,
    pub deterministic_compatible: bool,
    pub precision_compatible: bool,
    pub provider_ready: bool,
    pub device_ready: bool,
    pub provider_status: Option<ProviderStatusSnapshot>,
    pub device_status: Option<DeviceStatus>,
    pub pressure_score: u32,
    pub conformance_status: Option<String>,
    pub estimated_cost: u64,
    pub fallback_rank: u32,
    pub rejection_reason: Option<KernelCandidateRejection>,
}

impl KernelCandidate {
    fn rejected(advertisement: &KernelAdvertisement, reason: KernelCandidateRejection) -> Self {
        Self {
            kernel: advertisement.id.clone(),
            provider: advertisement.id.provider.clone(),
            device: advertisement.devices.iter().next().cloned(),
            operator: advertisement.implemented_operator.clone(),
            compatible: false,
            dtype_compatible: true,
            layout_compatible: true,
            shape_compatible: true,
            memory_compatible: true,
            workspace_feasible: true,
            affinity_compatible: true,
            deterministic_compatible: true,
            precision_compatible: true,
            provider_ready: true,
            device_ready: true,
            provider_status: None,
            device_status: None,
            pressure_score: 0,
            conformance_status: advertisement.id.conformance_profile.clone(),
            estimated_cost: 0,
            fallback_rank: u32::MAX,
            rejection_reason: Some(reason),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelSelectionResult {
    pub request_id: String,
    pub selected: Option<KernelCandidate>,
    pub candidates: Vec<KernelCandidate>,
    pub fallback_chain: Vec<KernelCandidate>,
    pub observations: Vec<KernelObservation>,
}

#[derive(Clone, Debug, Default)]
pub struct KernelRegistry {
    entries: BTreeMap<String, KernelRegistryEntry>,
    provider_statuses: BTreeMap<ProviderBinding, ProviderStatusSnapshot>,
    device_statuses: BTreeMap<DeviceBinding, DeviceStatus>,
    provider_features: BTreeMap<ProviderBinding, BTreeSet<String>>,
    device_features: BTreeMap<DeviceBinding, BTreeSet<String>>,
    revoked_conformance_profiles: BTreeSet<String>,
    policy_generation: u64,
    observations: Vec<KernelObservation>,
    /// Prepared Kernel state associated with compatible Kernel candidates.
    /// Implements "Registry Tracks Prepared Kernel Readiness" and "Registry
    /// Supports Multiple Prepared Generations"
    /// (`define-kernel-artifact-and-preparation-contract`). Never holds a
    /// native executable pointer -- only opaque `PreparedKernelId`s.
    prepared_kernels: BTreeMap<PreparedKernelId, PreparedKernel>,
    artifact_observations: Vec<KernelArtifactObservation>,
    /// The currently active Prepared Kernel generation per logical Kernel,
    /// implementing "Atomic Registry Promotion"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// a single `BTreeMap` write is the one publication point new dispatches
    /// observe, so a lookup during promotion always sees either the
    /// complete old value or the complete new value, never a partially
    /// updated one.
    active_generations: BTreeMap<String, PreparedKernelId>,
    /// The single previously active generation retained per Kernel,
    /// implementing "Rollback Candidate": "A previously active Kernel
    /// SHOULD be retained long enough to support rollback."
    previous_generations: BTreeMap<String, PreparedKernelId>,
    /// Kernels whose qualification has been revoked, implementing
    /// "Revocation Of Active Kernel": "Runtime SHALL stop new dispatches to
    /// it."
    revoked_kernels: BTreeSet<String>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> impl Iterator<Item = &KernelRegistryEntry> {
        self.entries.values()
    }

    pub fn observations(&self) -> &[KernelObservation] {
        &self.observations
    }

    pub fn artifact_observations(&self) -> &[KernelArtifactObservation] {
        &self.artifact_observations
    }

    /// Associates a compatible Kernel candidate with Prepared Kernel state,
    /// implementing "Registry Tracks Prepared Kernel Readiness". The
    /// Registry stores only `prepared.id`, `prepared.kernel`, and
    /// `prepared.generation` bookkeeping -- it never gains a native
    /// executable pointer through this call, implementing "Registry Does
    /// Not Own Native Handles".
    pub fn register_prepared_kernel(&mut self, prepared: PreparedKernel) {
        self.artifact_observations.push(
            KernelArtifactObservation::new(KernelArtifactObservationKind::PreparedKernelRegistered)
                .with_artifact(prepared.kernel.stable_key())
                .with_redacted_metadata("generation", prepared.generation.value().to_string()),
        );
        // Implements "Observe candidate creation" (tasks): a newly
        // registered Prepared Kernel is a Registry candidate the moment it
        // exists, independent of whether it is ever promoted.
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelCandidateCreated)
                .with_kernel(&prepared.kernel)
                .with_redacted_metadata("generation", prepared.generation.value().to_string()),
        );
        self.prepared_kernels.insert(prepared.id, prepared);
    }

    pub fn prepared_kernel(&self, id: &PreparedKernelId) -> Option<&PreparedKernel> {
        self.prepared_kernels.get(id)
    }

    pub fn prepared_kernels_for<'a>(
        &'a self,
        kernel: &'a KernelId,
    ) -> impl Iterator<Item = &'a PreparedKernel> {
        self.prepared_kernels
            .values()
            .filter(move |prepared| &prepared.kernel == kernel)
    }

    /// Whether `kernel` has at least one dispatchable (`Ready`) Prepared
    /// Kernel. A `KernelId` with no registered Prepared Kernel entries at
    /// all is not artifact-backed and is therefore not gated by this check
    /// -- "Kernel implementation MAY be backed by a Kernel Artifact
    /// lifecycle" (proposal): this is opt-in, not retroactively required for
    /// every existing Kernel.
    pub fn validate_prepared_readiness(
        &self,
        kernel: &KernelId,
    ) -> Result<(), KernelArtifactError> {
        let mut candidates = self.prepared_kernels_for(kernel).peekable();
        if candidates.peek().is_none() {
            return Ok(());
        }
        if candidates.any(|prepared| prepared.state.is_dispatchable()) {
            Ok(())
        } else {
            Err(KernelArtifactError::PreparedNotReady {
                kernel: kernel.stable_key(),
            })
        }
    }

    /// Retires a Prepared Kernel generation ahead of destruction.
    pub fn retire_prepared_kernel(
        &mut self,
        id: &PreparedKernelId,
    ) -> Result<(), KernelArtifactError> {
        let prepared = self.prepared_kernels.get_mut(id).ok_or_else(|| {
            KernelArtifactError::PreparedHandleInvalid {
                reason: format!("unknown Prepared Kernel {id}"),
            }
        })?;
        prepared.retire()?;
        self.artifact_observations.push(
            KernelArtifactObservation::new(KernelArtifactObservationKind::PreparedKernelRetired)
                .with_artifact(prepared.kernel.stable_key()),
        );
        Ok(())
    }

    /// Destroys a Prepared Kernel generation, implementing "Older Prepared
    /// Kernels MAY be destroyed only after no active operation references
    /// them" -- destruction fails while `active_references() > 0`.
    pub fn destroy_prepared_kernel(
        &mut self,
        id: &PreparedKernelId,
    ) -> Result<(), KernelArtifactError> {
        let prepared = self.prepared_kernels.get_mut(id).ok_or_else(|| {
            KernelArtifactError::PreparedHandleInvalid {
                reason: format!("unknown Prepared Kernel {id}"),
            }
        })?;
        prepared.destroy()?;
        let kernel = prepared.kernel.stable_key();
        self.prepared_kernels.remove(id);
        self.artifact_observations.push(
            KernelArtifactObservation::new(KernelArtifactObservationKind::PreparedKernelDestroyed)
                .with_artifact(kernel),
        );
        Ok(())
    }

    /// Guards [`Self::destroy_prepared_kernel`] with an explicit rollback
    /// window, implementing "Rollback Window" (proposal): "Allow retention
    /// period for previous generation" and "Prevent immediate destruction
    /// when rollback required." While `policy.retain_previous_generation` is
    /// set, destroying the generation still recorded as any Kernel's
    /// rollback candidate (see [`Self::rollback_generation`]) is denied.
    pub fn destroy_prepared_kernel_with_rollback_window(
        &mut self,
        id: &PreparedKernelId,
        policy: RollbackWindowPolicy,
    ) -> Result<(), KernelArtifactError> {
        if policy.retain_previous_generation
            && self
                .previous_generations
                .values()
                .any(|previous| previous == id)
        {
            let generation = self
                .prepared_kernels
                .get(id)
                .map(|prepared| prepared.generation.value())
                .unwrap_or_default();
            return Err(KernelArtifactError::PreparedGenerationInUse { generation });
        }
        self.destroy_prepared_kernel(id)
    }

    /// Returns the currently active Prepared Kernel for `kernel`, implementing
    /// "Multiple Prepared Generations": Registry tracks at most one active
    /// generation per logical Kernel at a time.
    pub fn active_prepared_kernel(&self, kernel: &KernelId) -> Option<&PreparedKernel> {
        self.active_generations
            .get(&kernel.stable_key())
            .and_then(|id| self.prepared_kernels.get(id))
    }

    /// Binds a continuous-batching slot to the Kernel generation it
    /// currently acquires, implementing "Continuous Batching" (proposal):
    /// "Preserve Kernel generation for in-flight batch work" (tasks). The
    /// returned [`BatchSlotKernelBinding`] is a plain value snapshot -- it
    /// stays valid for the slot's lifetime even after a later promotion
    /// changes [`Self::active_prepared_kernel`].
    pub fn bind_batch_slot(&self, kernel: &KernelId, slot: u64) -> Option<BatchSlotKernelBinding> {
        self.active_prepared_kernel(kernel)
            .map(|prepared| BatchSlotKernelBinding {
                slot,
                generation: prepared.id,
            })
    }

    /// Implements "Allow new batch work to use new generation" (tasks): new
    /// batch admissions always resolve against the *current* active
    /// generation, independent of any existing [`BatchSlotKernelBinding`].
    pub fn admit_new_batch_work(&self, kernel: &KernelId) -> Option<PreparedKernelId> {
        self.active_prepared_kernel(kernel)
            .map(|prepared| prepared.id)
    }

    /// Promotes `candidate` to be the active Prepared Kernel generation for
    /// `kernel`, implementing "Registry Promotion Is Explicit" and "Atomic
    /// Kernel Promotion"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// promotion never happens implicitly (it is the only path that writes
    /// `active_generations`), requires the candidate to already be `Ready`
    /// (implementing "Registry Promotion Is Explicit": preparation-without-
    /// promotion SHALL NOT become active on its own), retires the previous
    /// generation via [`Self::retire_prepared_kernel`] rather than
    /// destroying it outright (implementing "In-Flight Stability": an
    /// invocation holding the old generation keeps using it), and retains
    /// the previous generation as the rollback candidate (implementing
    /// "Rollback Candidate").
    pub fn promote_generation(
        &mut self,
        kernel: &KernelId,
        candidate: PreparedKernelId,
    ) -> Result<(), KernelRegistryError> {
        if self.revoked_kernels.contains(&kernel.stable_key()) {
            return Err(KernelRegistryError::KernelRevoked {
                kernel: kernel.stable_key(),
            });
        }
        let is_ready = self
            .prepared_kernels
            .get(&candidate)
            .is_some_and(|prepared| prepared.state.is_dispatchable());
        if !is_ready {
            return Err(KernelRegistryError::PromotionNotEligible {
                kernel: kernel.stable_key(),
            });
        }
        let key = kernel.stable_key();
        if let Some(previous) = self.active_generations.get(&key).copied() {
            // Retiring (rather than destroying) preserves in-flight safety:
            // an invocation that already acquired `previous` keeps a valid
            // reference until it releases it (see
            // `crate::kernel_artifact::PreparedKernel::destroy`).
            let _ = self.retire_prepared_kernel(&previous);
            self.previous_generations.insert(key.clone(), previous);
        }
        // The single map write below is the atomic publication point: a
        // concurrent lookup via `active_prepared_kernel` observes either the
        // pre-promotion value or this one, never an intermediate state.
        self.active_generations.insert(key.clone(), candidate);
        let generation = self
            .prepared_kernels
            .get(&candidate)
            .map(|prepared| prepared.generation.value())
            .unwrap_or_default();
        self.artifact_observations.push(
            KernelArtifactObservation::new(
                KernelArtifactObservationKind::ArtifactReplacementOccurred,
            )
            .with_artifact(key.clone())
            // Implements "Record generation" (tasks): the promotion
            // observation identifies which generation became active.
            .with_redacted_metadata("generation", generation.to_string()),
        );
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelGenerationPromoted)
                .with_kernel(kernel)
                .with_redacted_metadata("generation", generation.to_string()),
        );
        Ok(())
    }

    /// Validates trust/qualification eligibility before delegating to
    /// [`Self::promote_generation`], implementing "Validate eligibility
    /// before promotion" (tasks): promotion SHALL NOT proceed for a
    /// candidate that fails
    /// [`crate::kernel_qualification::evaluate_eligibility`], independent of
    /// the Prepared Kernel readiness check `promote_generation` already
    /// performs on its own.
    pub fn promote_generation_with_eligibility(
        &mut self,
        kernel: &KernelId,
        candidate: PreparedKernelId,
        trust: crate::kernel_artifact::KernelArtifactTrust,
        qualification_status: crate::kernel_qualification::QualificationStatus,
        policy: &crate::kernel_qualification::KernelEligibilityPolicy,
    ) -> Result<(), KernelRegistryError> {
        crate::kernel_qualification::evaluate_eligibility(trust, qualification_status, policy)
            .map_err(|error| KernelRegistryError::PromotionNotEligible {
                kernel: format!("{}: {error}", kernel.stable_key()),
            })?;
        self.promote_generation(kernel, candidate)
    }

    /// Implements "Resource Affinity" (proposal) at promotion time:
    /// "Validate candidate Device affinity before promotion" and "Reject
    /// incompatible prepared target." Tensor residency constraints are
    /// preserved because a candidate is never promoted onto a Device other
    /// than the one its Prepared Kernel state actually resides on.
    pub fn promote_generation_with_affinity(
        &mut self,
        kernel: &KernelId,
        candidate: PreparedKernelId,
        required_device: &DeviceBinding,
    ) -> Result<(), KernelRegistryError> {
        let device_matches = self
            .prepared_kernels
            .get(&candidate)
            .is_some_and(|prepared| &prepared.device == required_device);
        if !device_matches {
            return Err(KernelRegistryError::ResourceAffinityConflict(format!(
                "candidate Prepared Kernel Device does not match required Device '{required_device}'"
            )));
        }
        self.promote_generation(kernel, candidate)
    }

    /// Rolls back `kernel` to its previously active generation, implementing
    /// "Rollback"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// available only while a retained previous generation still exists and
    /// remains dispatchable.
    pub fn rollback_generation(&mut self, kernel: &KernelId) -> Result<(), KernelRegistryError> {
        let key = kernel.stable_key();
        let previous = self
            .previous_generations
            .get(&key)
            .copied()
            .ok_or_else(|| KernelRegistryError::RollbackUnavailable {
                kernel: key.clone(),
            })?;
        let still_dispatchable = self
            .prepared_kernels
            .get(&previous)
            .is_some_and(|prepared| {
                matches!(
                    prepared.state,
                    crate::kernel_artifact::PreparedKernelState::Ready
                        | crate::kernel_artifact::PreparedKernelState::Retiring
                )
            });
        if !still_dispatchable {
            return Err(KernelRegistryError::RollbackUnavailable { kernel: key });
        }
        if let Some(prepared) = self.prepared_kernels.get_mut(&previous) {
            // Undo the retirement so the rolled-back generation accepts new
            // dispatches again.
            prepared.state = crate::kernel_artifact::PreparedKernelState::Ready;
        }
        self.active_generations.insert(key.clone(), previous);
        self.previous_generations.remove(&key);
        self.artifact_observations.push(
            KernelArtifactObservation::new(
                KernelArtifactObservationKind::ArtifactReplacementOccurred,
            )
            .with_artifact(key)
            .with_redacted_metadata("event", "rollback"),
        );
        Ok(())
    }

    /// Marks `kernel` revoked, implementing "Revocation Of Active Kernel"
    /// (`define-generated-kernel-qualification-cache-and-hot-swap-contract`):
    /// "Runtime SHALL stop new dispatches to it." Existing in-flight
    /// invocations are unaffected -- this only stops new candidate lookups
    /// (see `candidate_for_entry`) from selecting it.
    pub fn revoke_kernel(&mut self, kernel: &KernelId, reason: impl Into<String>) {
        self.revoke_kernel_with_policy(kernel, reason, RevocationInFlightPolicy::AllowToComplete);
    }

    /// Implements "Define in-flight revocation policy" (proposal): "Existing
    /// invocations SHALL follow policy: allow-to-complete, cancel-if-safe,
    /// fail-closed, Provider-specific." Recorded alongside the revocation
    /// observation; enforcing cancellation/fail-closed behavior against a
    /// live invocation is a Provider/execution-layer concern outside this
    /// Registry's scope -- this Registry only stops *new* dispatches (see
    /// `candidate_for_entry`) and records which policy governs the
    /// in-flight ones.
    pub fn revoke_kernel_with_policy(
        &mut self,
        kernel: &KernelId,
        reason: impl Into<String>,
        in_flight_policy: RevocationInFlightPolicy,
    ) {
        let key = kernel.stable_key();
        self.revoked_kernels.insert(key.clone());
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.active = false;
            entry.invalidation_reason = Some(reason.into());
        }
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelRevoked)
                .with_kernel(kernel)
                .with_redacted_metadata("in-flight-policy", format!("{in_flight_policy:?}")),
        );
    }

    pub fn is_kernel_revoked(&self, kernel: &KernelId) -> bool {
        self.revoked_kernels.contains(&kernel.stable_key())
    }

    pub fn set_provider_status(&mut self, status: ProviderStatusSnapshot) {
        let provider = status.provider.clone();
        let reason = match status.provider_health_compat() {
            HealthState::Draining => Some("provider draining"),
            HealthState::Initializing => Some("provider not ready"),
            HealthState::Saturated => Some("provider saturated"),
            HealthState::Unavailable | HealthState::Interrupted => Some("provider unavailable"),
            HealthState::Unknown => Some("provider status stale"),
            HealthState::Available | HealthState::Degraded => None,
        };
        self.provider_statuses.insert(provider.clone(), status);
        if let Some(reason) = reason {
            self.invalidate_provider(&provider, reason);
        }
    }

    pub fn set_device_status(&mut self, status: DeviceStatus) {
        let device = status.device.clone();
        let reason = match status.availability {
            HealthState::Available | HealthState::Degraded | HealthState::Draining => None,
            HealthState::Saturated => Some("device pressure saturated"),
            HealthState::Unknown | HealthState::Initializing => Some("device not ready"),
            HealthState::Unavailable => Some("device unavailable"),
            HealthState::Interrupted => Some("device lost"),
        };
        self.device_statuses.insert(device.clone(), status);
        if let Some(reason) = reason {
            self.invalidate_device(&device, reason);
        }
    }

    pub fn set_provider_features(
        &mut self,
        provider: ProviderBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) {
        self.provider_features
            .insert(provider, features.into_iter().map(Into::into).collect());
    }

    pub fn set_device_features(
        &mut self,
        device: DeviceBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) {
        self.device_features
            .insert(device, features.into_iter().map(Into::into).collect());
    }

    pub fn advertisement(&self, kernel: &KernelId) -> Option<&KernelAdvertisement> {
        self.entries
            .get(&kernel.stable_key())
            .map(|entry| &entry.advertisement)
    }

    pub fn active_advertisement(&self, kernel: &KernelId) -> Option<&KernelAdvertisement> {
        self.entries
            .get(&kernel.stable_key())
            .filter(|entry| entry.active)
            .map(|entry| &entry.advertisement)
    }

    pub fn register_provider_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
    ) -> Result<(), KernelRegistryError> {
        self.register_advertisement(advertisement, KernelRegistrationAuthority::Provider)
    }

    pub fn register_fixture_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
    ) -> Result<(), KernelRegistryError> {
        self.register_advertisement(
            advertisement,
            KernelRegistrationAuthority::RuntimeTestFixture,
        )
    }

    pub fn register_advertisement(
        &mut self,
        advertisement: KernelAdvertisement,
        authority: KernelRegistrationAuthority,
    ) -> Result<(), KernelRegistryError> {
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelAdvertisementReceived)
                .with_kernel(&advertisement.id),
        );
        if !authority.may_register_kernel() {
            self.observations.push(
                KernelObservation::new(KernelObservationKind::KernelAdvertisementRejected)
                    .with_kernel(&advertisement.id)
                    .with_redacted_metadata("error", "kernel-registration-denied"),
            );
            return Err(KernelRegistryError::RegistrationDenied(authority));
        }
        validate_kernel_advertisement(&advertisement)?;
        let key = advertisement.id.stable_key();
        self.entries.insert(
            key,
            KernelRegistryEntry::new(advertisement.clone(), authority),
        );
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelAdvertisementAccepted)
                .with_kernel(&advertisement.id),
        );
        self.observations.push(
            KernelObservation::new(KernelObservationKind::KernelRegistryUpdated)
                .with_kernel(&advertisement.id),
        );
        Ok(())
    }

    pub fn invalidate_provider(&mut self, provider: &ProviderBinding, reason: impl Into<String>) {
        let reason = reason.into();
        for entry in self.entries.values_mut() {
            if &entry.advertisement.id.provider == provider {
                entry.active = false;
                entry.invalidation_reason = Some(reason.clone());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", reason.clone()),
                );
            }
        }
    }

    pub fn invalidate_device(&mut self, device: &DeviceBinding, reason: impl Into<String>) {
        let reason = reason.into();
        for entry in self.entries.values_mut() {
            if entry.advertisement.devices.contains(device) {
                entry.active = false;
                entry.invalidation_reason = Some(reason.clone());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", reason.clone()),
                );
            }
        }
    }

    pub fn revoke_conformance_profile(&mut self, profile: impl Into<String>) {
        let profile = profile.into();
        self.revoked_conformance_profiles.insert(profile.clone());
        for entry in self.entries.values_mut() {
            if entry.advertisement.id.conformance_profile.as_deref() == Some(profile.as_str()) {
                entry.active = false;
                entry.invalidation_reason = Some("kernel conformance revoked".into());
                self.observations.push(
                    KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                        .with_kernel(&entry.advertisement.id)
                        .with_redacted_metadata("reason", "kernel conformance revoked"),
                );
            }
        }
    }

    pub fn apply_policy_change(&mut self, generation: u64) {
        self.policy_generation = generation;
        for entry in self.entries.values_mut() {
            entry.active = false;
            entry.invalidation_reason = Some("policy changed".into());
            self.observations.push(
                KernelObservation::new(KernelObservationKind::KernelRegistryInvalidated)
                    .with_kernel(&entry.advertisement.id)
                    .with_redacted_metadata("reason", "policy changed"),
            );
        }
    }

    pub fn candidates(&self, request: &KernelSelectionRequest) -> Vec<KernelCandidate> {
        self.observations_for_lookup(request);
        self.entries
            .values()
            .filter(|entry| {
                entry.advertisement.implemented_operator.namespace() == request.operator.namespace()
                    && entry.advertisement.implemented_operator.name() == request.operator.name()
            })
            .map(|entry| self.candidate_for_entry(entry, request))
            .collect()
    }

    pub fn select(
        &self,
        request: &KernelSelectionRequest,
    ) -> Result<KernelSelectionResult, KernelRegistryError> {
        let mut candidates = self.candidates(request);
        if candidates.is_empty() {
            return Err(KernelRegistryError::CandidateNotFound {
                operator: request.operator.clone(),
            });
        }
        candidates.sort_by_key(|candidate| {
            (
                !candidate.compatible,
                candidate.fallback_rank,
                candidate.pressure_score,
                candidate.estimated_cost,
                candidate.kernel.stable_key(),
            )
        });
        let selected = candidates
            .iter()
            .find(|candidate| candidate.compatible)
            .cloned();
        let Some(selected_candidate) = selected.clone() else {
            let reason = candidates
                .first()
                .and_then(|candidate| candidate.rejection_reason.clone())
                .unwrap_or(KernelCandidateRejection::PolicyDenied);
            return Err(KernelRegistryError::CandidateIncompatible { reason });
        };
        let fallback_chain = candidates
            .iter()
            .filter(|candidate| {
                candidate.compatible && candidate.kernel != selected_candidate.kernel
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut observations = vec![
            KernelObservation::new(KernelObservationKind::KernelCandidateLookup)
                .with_redacted_metadata("request_id", request.request_id.clone()),
        ];
        observations.extend(candidates.iter().filter_map(|candidate| {
            candidate.rejection_reason.as_ref().map(|reason| {
                KernelObservation::new(KernelObservationKind::KernelCandidateRejected)
                    .with_kernel(&candidate.kernel)
                    .with_redacted_metadata("reason", format!("{reason:?}"))
            })
        }));
        observations.extend(
            candidates
                .iter()
                .filter(|candidate| candidate.compatible)
                .map(|candidate| {
                    KernelObservation::new(KernelObservationKind::KernelCandidateRanked)
                        .with_kernel(&candidate.kernel)
                        .with_redacted_metadata("rank", candidate.fallback_rank.to_string())
                }),
        );
        observations.push(
            KernelObservation::new(KernelObservationKind::KernelSelected)
                .with_kernel(&selected_candidate.kernel)
                .with_redacted_metadata("request_id", request.request_id.clone()),
        );
        Ok(KernelSelectionResult {
            request_id: request.request_id.clone(),
            selected,
            candidates,
            fallback_chain,
            observations,
        })
    }

    fn observations_for_lookup(&self, _request: &KernelSelectionRequest) {}

    fn candidate_for_entry(
        &self,
        entry: &KernelRegistryEntry,
        request: &KernelSelectionRequest,
    ) -> KernelCandidate {
        let advertisement = &entry.advertisement;
        if self
            .revoked_kernels
            .contains(&advertisement.id.stable_key())
        {
            return KernelCandidate::rejected(advertisement, KernelCandidateRejection::Revoked);
        }
        if !entry.active {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::StaleRegistryEntry,
            );
        }
        if !advertisement
            .id
            .operator_versions
            .contains(request.operator_version)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::OperatorVersionUnsupported,
            );
        }
        if request.browser_target && !advertisement.browser_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ProviderUnavailable,
            );
        }
        if let Some(profile) = advertisement.id.conformance_profile.as_ref()
            && self.revoked_conformance_profiles.contains(profile)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ConformanceFailed,
            );
        }
        if request.require_conformance && advertisement.id.conformance_profile.is_none() {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ConformanceMissing,
            );
        }
        if let Some(mode) = request.execution_mode
            && !advertisement.execution_modes.contains(&mode)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ExecutionModeUnsupported,
            );
        }
        if let Some(status) = self.provider_statuses.get(&advertisement.id.provider) {
            if matches!(status.provider_health_compat(), HealthState::Saturated) {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderSaturated,
                );
            }
            if !status.accepts_new_work_by_default() {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderNotReady,
                );
            }
        }
        for device in &advertisement.devices {
            if let Some(status) = self.device_statuses.get(device)
                && !matches!(
                    status.availability,
                    HealthState::Available | HealthState::Degraded | HealthState::Draining
                )
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::DeviceUnavailable,
                );
            }
        }
        if !advertisement.required_provider_features.is_empty() {
            let features = self
                .provider_features
                .get(&advertisement.id.provider)
                .cloned()
                .unwrap_or_default();
            if !advertisement
                .required_provider_features
                .iter()
                .all(|feature| features.contains(feature))
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ProviderFeatureMissing,
                );
            }
        }
        if !advertisement.required_device_features.is_empty() {
            let device_features_match = advertisement.devices.iter().any(|device| {
                let features = self
                    .device_features
                    .get(device)
                    .cloned()
                    .unwrap_or_default();
                advertisement
                    .required_device_features
                    .iter()
                    .all(|feature| features.contains(feature))
            });
            if !device_features_match {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::DeviceFeatureMissing,
                );
            }
        }
        let dtype_compatible = request.dtype_requirements.iter().all(|required| {
            advertisement
                .supported_dtypes
                .values()
                .any(|supported| supported.is_empty() || supported.contains(required))
        });
        if !dtype_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::DTypeUnsupported,
            );
        }
        let layout_compatible = request.layout_requirements.iter().all(|required| {
            advertisement.supported_layouts.is_empty()
                || advertisement.supported_layouts.contains(required)
        });
        if !layout_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::LayoutUnsupported,
            );
        }
        let memory_compatible = request.memory_class_requirements.iter().all(|required| {
            advertisement.memory_classes.is_empty()
                || advertisement.memory_classes.contains(required)
        });
        if !memory_compatible {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::MemoryClassUnsupported,
            );
        }
        if !shape_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::ShapeUnsupported,
            );
        }
        if advertisement.workspace.required
            && advertisement.workspace.size_bytes_upper_bound == Some(0)
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::WorkspaceUnavailable,
            );
        }
        if !batching_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::BatchingUnsupported,
            );
        }
        if !adapter_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::AdapterUnsupported,
            );
        }
        if !kv_cache_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::KvCacheUnsupported,
            );
        }
        if !prefix_cache_compatible(advertisement, request) {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PrefixCacheUnsupported,
            );
        }
        if request.deterministic_required && !advertisement.determinism.deterministic {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PolicyDenied,
            );
        }
        if request.precision == ComputePrecision::Exact && advertisement.precision.approximate_math
        {
            return KernelCandidate::rejected(
                advertisement,
                KernelCandidateRejection::PolicyDenied,
            );
        }
        for input in &request.inputs {
            if let Err(_error) =
                validate_affinity_compatibility(&request.affinity, &input.resource.affinity)
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ResourceAffinityConflict,
                );
            }
        }
        for output in &request.outputs {
            if let Err(_error) =
                validate_affinity_compatibility(&request.affinity, &output.resource.affinity)
            {
                return KernelCandidate::rejected(
                    advertisement,
                    KernelCandidateRejection::ResourceAffinityConflict,
                );
            }
        }
        let provider_status = self
            .provider_statuses
            .get(&advertisement.id.provider)
            .cloned();
        let device = advertisement.devices.iter().next().cloned();
        let device_status = device
            .as_ref()
            .and_then(|device| self.device_statuses.get(device))
            .cloned();
        let provider_pressure = provider_status
            .as_ref()
            .map(|status| pressure_score(status.pressure))
            .unwrap_or(0);
        let device_pressure = device_status
            .as_ref()
            .map(|status| pressure_score(status.pressure))
            .unwrap_or(0);
        KernelCandidate {
            kernel: advertisement.id.clone(),
            provider: advertisement.id.provider.clone(),
            device,
            operator: advertisement.implemented_operator.clone(),
            compatible: true,
            dtype_compatible: true,
            layout_compatible: true,
            shape_compatible: true,
            memory_compatible: true,
            workspace_feasible: true,
            affinity_compatible: true,
            deterministic_compatible: true,
            precision_compatible: true,
            provider_ready: provider_status
                .as_ref()
                .map(ProviderStatusSnapshot::accepts_new_work_by_default)
                .unwrap_or(true),
            device_ready: device_status
                .as_ref()
                .map(|status| {
                    matches!(
                        status.availability,
                        HealthState::Available | HealthState::Degraded | HealthState::Draining
                    )
                })
                .unwrap_or(true),
            provider_status,
            device_status,
            pressure_score: provider_pressure
                + device_pressure
                + advertisement
                    .performance_hints
                    .get("pressure")
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
            conformance_status: advertisement.id.conformance_profile.clone(),
            estimated_cost: advertisement
                .performance_hints
                .get("estimated-cost")
                .or_else(|| advertisement.performance_hints.get("expected-latency"))
                .or_else(|| {
                    advertisement
                        .performance_hints
                        .get("expected-throughput-cost")
                })
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            fallback_rank: advertisement
                .performance_hints
                .get("fallback-rank")
                .and_then(|value| value.parse().ok())
                .unwrap_or(self.policy_generation as u32),
            rejection_reason: None,
        }
    }
}

pub fn validate_kernel_advertisement(
    advertisement: &KernelAdvertisement,
) -> Result<(), KernelRegistryError> {
    if advertisement.id.provider.as_str().trim().is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Provider identity must not be empty".into(),
        ));
    }
    if advertisement.id.name.trim().is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Kernel identity must not be empty".into(),
        ));
    }
    if advertisement.implemented_operator != advertisement.id.operator {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "implemented Operator must match Kernel identity".into(),
        ));
    }
    if !advertisement
        .id
        .operator_versions
        .contains(advertisement.implemented_operator.version())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "Kernel Operator version range must include implemented Operator".into(),
        ));
    }
    if advertisement.execution_modes.is_empty() {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "execution mode metadata must not be empty".into(),
        ));
    }
    if advertisement
        .required_provider_features
        .iter()
        .any(|feature| feature.trim().is_empty())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "required Provider features must not be empty".into(),
        ));
    }
    if advertisement
        .required_device_features
        .iter()
        .any(|feature| feature.trim().is_empty())
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "required Device features must not be empty".into(),
        ));
    }
    if advertisement.browser_compatible
        && advertisement
            .execution_modes
            .contains(&KernelExecutionMode::GraphCaptured)
    {
        return Err(KernelRegistryError::AdvertisementInvalid(
            "browser-compatible Kernel cannot require graph-captured native execution".into(),
        ));
    }
    Ok(())
}

fn pressure_score(pressure: ProviderPressureLevel) -> u32 {
    match pressure {
        ProviderPressureLevel::Unknown => 10,
        ProviderPressureLevel::Low => 0,
        ProviderPressureLevel::Moderate => 25,
        ProviderPressureLevel::High => 75,
        ProviderPressureLevel::Saturated => 100,
    }
}

fn shape_compatible(advertisement: &KernelAdvertisement, request: &KernelSelectionRequest) -> bool {
    request
        .inputs
        .iter()
        .chain(&request.outputs)
        .all(|resource| {
            let shape = &resource.resource.descriptor.shape;
            if let Some(rank) = advertisement.shape.rank
                && shape.rank() != rank
            {
                return false;
            }
            for (index, expected) in &advertisement.shape.static_dimensions {
                if shape.dimensions.get(*index) != Some(expected) {
                    return false;
                }
            }
            if let Some(alignment) = advertisement.shape.alignment
                && shape
                    .dimensions
                    .iter()
                    .any(|dimension| dimension % alignment != 0)
            {
                return false;
            }
            if let Some(max) = advertisement.shape.max_total_elements {
                return shape.element_count().is_ok_and(|count| count <= max);
            }
            true
        })
}

fn batching_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.batching.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.batching.as_ref() else {
        return false;
    };
    (!required.supports_ragged_batches || supported.supports_ragged_batches)
        && (!required.per_operation_output_mapping || supported.per_operation_output_mapping)
        && required
            .max_batch_size
            .zip(supported.max_batch_size)
            .is_none_or(|(required, supported)| required <= supported)
        && required
            .max_active_sequences
            .zip(supported.max_active_sequences)
            .is_none_or(|(required, supported)| required <= supported)
        && required
            .max_total_tokens
            .zip(supported.max_total_tokens)
            .is_none_or(|(required, supported)| required <= supported)
}

fn adapter_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    if request.adapter_methods.is_empty() {
        return true;
    }
    let Some(adapter) = advertisement.adapter.as_ref() else {
        return false;
    };
    request
        .adapter_methods
        .iter()
        .all(|method| adapter.methods.contains(method))
}

fn kv_cache_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.kv_cache.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.kv_cache.as_ref() else {
        return false;
    };
    (!required.paged_cache || supported.paged_cache)
        && (!required.append || supported.append)
        && (!required.read || supported.read)
        && required
            .layouts
            .iter()
            .all(|layout| supported.layouts.contains(layout))
        && required
            .dtypes
            .iter()
            .all(|dtype| supported.dtypes.contains(dtype))
        && required
            .memory_classes
            .iter()
            .all(|class| supported.memory_classes.contains(class))
}

fn prefix_cache_compatible(
    advertisement: &KernelAdvertisement,
    request: &KernelSelectionRequest,
) -> bool {
    let Some(required) = request.prefix_cache.as_ref() else {
        return true;
    };
    let Some(supported) = advertisement.prefix_cache.as_ref() else {
        return false;
    };
    (!required.supports_adjusted_sequence_length || supported.supports_adjusted_sequence_length)
        && (!required.supports_adjusted_context_length
            || supported.supports_adjusted_context_length)
        && (!required.supports_reused_prefix_boundary || supported.supports_reused_prefix_boundary)
}

// ---------------------------------------------------------------------
// Performance Evidence Identity
// (define-kernel-performance-model-and-adaptive-feedback-contract)
// ---------------------------------------------------------------------

/// Implements "Registry Preserves Performance Evidence Identity" (proposal):
/// "Registry SHALL associate performance evidence with the correct Kernel
/// Artifact, specialization, and generation context." Keying on the opaque
/// [`PreparedKernelId`] as well as the artifact digest means a new Prepared
/// Kernel generation -- which always allocates a fresh, distinct id via
/// [`PreparedKernelIdAllocator`] -- can never collide with a prior
/// generation's evidence key, so "N observations do not silently become N+1
/// observations."
pub fn performance_evidence_key(
    artifact: &CompiledKernelArtifactId,
    generation: PreparedKernelId,
) -> String {
    format!("{artifact}|{generation}")
}

/// Implements "Registry Does Not Generate Performance Evidence" (proposal):
/// "Kernel Registry SHALL not fabricate missing benchmark or online
/// metrics." A missing entry resolves to `None` -- never to another
/// candidate's evidence, and never to a synthesized value.
pub fn lookup_performance_evidence<'a>(
    evidence: &'a BTreeMap<String, crate::kernel_performance_model::KernelPerformanceMetricSummary>,
    artifact: &CompiledKernelArtifactId,
    generation: PreparedKernelId,
) -> Option<&'a crate::kernel_performance_model::KernelPerformanceMetricSummary> {
    evidence.get(&performance_evidence_key(artifact, generation))
}

// ---------------------------------------------------------------------
// Generated Kernel lifecycle conformance
// ---------------------------------------------------------------------

/// A single Registry lifecycle conformance check result, mirroring
/// [`crate::kernel_artifact::KernelArtifactConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelRegistryLifecycleConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelRegistryLifecycleConformanceReport {
    pub results: Vec<KernelRegistryLifecycleConformanceResult>,
}

impl KernelRegistryLifecycleConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn lifecycle_record(
    results: &mut Vec<KernelRegistryLifecycleConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelRegistryLifecycleConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the Registry-side generated-Kernel lifecycle conformance checks
/// required by
/// `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract/specs/kernel-registry/spec.md`
/// and the corresponding requirements of `specs/conformance/spec.md`:
/// atomic promotion, in-flight generation safety, safe retirement, rollback,
/// and revocation.
pub fn run_kernel_registry_lifecycle_conformance() -> KernelRegistryLifecycleConformanceReport {
    let mut results = Vec::new();

    let kernel = KernelId::new(
        ProviderBinding::new("conformance-provider"),
        "conformance-kernel",
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::TestFixture,
    );
    let device = DeviceBinding::new(crate::DeviceId::new("conformance-device"));
    let mut allocator = PreparedKernelIdAllocator::default();
    let mut registry = KernelRegistry::new();
    registry
        .register_fixture_advertisement(KernelAdvertisement::new(kernel.clone()))
        .ok();

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-v1"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().ok();
    let generation_one_id = generation_one.id;
    registry.register_prepared_kernel(generation_one);

    let promote_first = registry.promote_generation(&kernel, generation_one_id);
    lifecycle_record(
        &mut results,
        "promotion requires an explicit call and is not implicit on registration",
        promote_first.is_ok()
            && registry
                .active_prepared_kernel(&kernel)
                .is_some_and(|prepared| prepared.id == generation_one_id),
        format!("unexpected outcome: {promote_first:?}"),
    );

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-v2"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().ok();
    let generation_two_id = generation_two.id;
    registry.register_prepared_kernel(generation_two);

    // Simulate an in-flight invocation holding generation one before
    // promoting generation two.
    registry
        .prepared_kernels
        .get_mut(&generation_one_id)
        .unwrap()
        .add_reference();

    let promote_second = registry.promote_generation(&kernel, generation_two_id);
    lifecycle_record(
        &mut results,
        "atomic promotion: active generation observes complete new value",
        promote_second.is_ok()
            && registry
                .active_prepared_kernel(&kernel)
                .is_some_and(|prepared| prepared.id == generation_two_id),
        format!("unexpected outcome: {promote_second:?}"),
    );

    lifecycle_record(
        &mut results,
        "in-flight invocation retains a valid old generation after promotion",
        registry
            .prepared_kernel(&generation_one_id)
            .is_some_and(|prepared| {
                prepared.state == crate::kernel_artifact::PreparedKernelState::Retiring
                    && prepared.active_references() > 0
            }),
        "expected old generation to be Retiring but still referenced",
    );

    let destroy_while_referenced = registry.destroy_prepared_kernel(&generation_one_id);
    lifecycle_record(
        &mut results,
        "safe retirement: destruction is blocked while old generation is referenced",
        destroy_while_referenced.is_err(),
        format!("unexpected outcome: {destroy_while_referenced:?}"),
    );

    let rollback = registry.rollback_generation(&kernel);
    lifecycle_record(
        &mut results,
        "rollback restores the previously active generation",
        rollback.is_ok()
            && registry
                .active_prepared_kernel(&kernel)
                .is_some_and(|prepared| prepared.id == generation_one_id),
        format!("unexpected outcome: {rollback:?}"),
    );

    // Rollback window: a dedicated scenario where a retained rollback
    // candidate has zero active references (so an ordinary
    // `destroy_prepared_kernel` would otherwise succeed) is still protected
    // while the window policy requires retention, and destroyable once the
    // policy is relaxed.
    {
        let window_kernel = KernelId::new(
            ProviderBinding::new("conformance-provider"),
            "conformance-window-kernel",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
            KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::TestFixture,
        );
        let mut window_registry = KernelRegistry::new();
        let mut window_gen_one = PreparedKernel::new(
            allocator.allocate(),
            window_kernel.clone(),
            CompiledKernelArtifactId::from_digest("digest-window-v1"),
            ProviderBinding::new("conformance-provider"),
            device.clone(),
            PreparedKernelGeneration::new(1),
        );
        window_gen_one.mark_ready().ok();
        let window_gen_one_id = window_gen_one.id;
        window_registry.register_prepared_kernel(window_gen_one);
        window_registry
            .promote_generation(&window_kernel, window_gen_one_id)
            .ok();

        let mut window_gen_two = PreparedKernel::new(
            allocator.allocate(),
            window_kernel.clone(),
            CompiledKernelArtifactId::from_digest("digest-window-v2"),
            ProviderBinding::new("conformance-provider"),
            device.clone(),
            PreparedKernelGeneration::new(2),
        );
        window_gen_two.mark_ready().ok();
        let window_gen_two_id = window_gen_two.id;
        window_registry.register_prepared_kernel(window_gen_two);
        // Promoting generation two retires generation one (zero active
        // references) and records it as the rollback candidate.
        window_registry
            .promote_generation(&window_kernel, window_gen_two_id)
            .ok();

        let retained = window_registry
            .previous_generations
            .get(&window_kernel.stable_key())
            == Some(&window_gen_one_id);
        let blocked_destroy = window_registry.destroy_prepared_kernel_with_rollback_window(
            &window_gen_one_id,
            RollbackWindowPolicy {
                retain_previous_generation: true,
            },
        );
        lifecycle_record(
            &mut results,
            "rollback window blocks destroying the retained rollback candidate",
            retained
                && matches!(
                    blocked_destroy,
                    Err(KernelArtifactError::PreparedGenerationInUse { .. })
                ),
            format!("unexpected outcome: {blocked_destroy:?}"),
        );

        let allowed_destroy = window_registry.destroy_prepared_kernel_with_rollback_window(
            &window_gen_one_id,
            RollbackWindowPolicy {
                retain_previous_generation: false,
            },
        );
        lifecycle_record(
            &mut results,
            "destruction proceeds once the rollback window policy is relaxed",
            allowed_destroy.is_ok(),
            format!("unexpected outcome: {allowed_destroy:?}"),
        );
    }

    // Failure atomicity: promoting a not-yet-ready candidate leaves the
    // active generation untouched.
    let mut unready_candidate = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-v3"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(3),
    );
    let unready_candidate_id = unready_candidate.id;
    // Never marked ready -- simulates a preparation failure.
    let _ = unready_candidate.mark_failed("simulated preparation failure");
    registry.register_prepared_kernel(unready_candidate);
    let active_before_failed_promotion = registry
        .active_prepared_kernel(&kernel)
        .map(|prepared| prepared.id);
    let failed_promotion = registry.promote_generation(&kernel, unready_candidate_id);
    lifecycle_record(
        &mut results,
        "failed promotion leaves the active generation intact",
        matches!(
            failed_promotion,
            Err(KernelRegistryError::PromotionNotEligible { .. })
        ) && registry
            .active_prepared_kernel(&kernel)
            .map(|prepared| prepared.id)
            == active_before_failed_promotion,
        format!("unexpected outcome: {failed_promotion:?}"),
    );

    // Resource Affinity: promotion is rejected when the candidate's Device
    // does not match the required Device, and accepted when it does. Uses a
    // dedicated fresh generation so it does not disturb the generation-one/
    // generation-two bookkeeping exercised above.
    let mut generation_affinity = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-v4"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(4),
    );
    generation_affinity.mark_ready().ok();
    let generation_affinity_id = generation_affinity.id;
    registry.register_prepared_kernel(generation_affinity);

    let other_device = DeviceBinding::new(crate::DeviceId::new("other-conformance-device"));
    let affinity_mismatch =
        registry.promote_generation_with_affinity(&kernel, generation_affinity_id, &other_device);
    lifecycle_record(
        &mut results,
        "promotion is rejected when candidate Device does not match required affinity",
        matches!(
            affinity_mismatch,
            Err(KernelRegistryError::ResourceAffinityConflict(_))
        ),
        format!("unexpected outcome: {affinity_mismatch:?}"),
    );
    let affinity_match =
        registry.promote_generation_with_affinity(&kernel, generation_affinity_id, &device);
    lifecycle_record(
        &mut results,
        "promotion succeeds when candidate Device matches required affinity",
        affinity_match.is_ok(),
        format!("unexpected outcome: {affinity_match:?}"),
    );

    // Eligibility gate: promotion is rejected for an unqualified candidate
    // even though it is Ready, and accepted once eligible. Uses its own
    // fresh generation for the same reason as the affinity check above.
    let mut generation_eligibility = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-v5"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(5),
    );
    generation_eligibility.mark_ready().ok();
    let generation_eligibility_id = generation_eligibility.id;
    registry.register_prepared_kernel(generation_eligibility);

    let ineligible_policy = crate::kernel_qualification::KernelEligibilityPolicy {
        require_trusted: false,
        require_qualified: true,
    };
    let eligibility_rejected = registry.promote_generation_with_eligibility(
        &kernel,
        generation_eligibility_id,
        crate::kernel_artifact::KernelArtifactTrust::Untrusted,
        crate::kernel_qualification::QualificationStatus::Unqualified,
        &ineligible_policy,
    );
    lifecycle_record(
        &mut results,
        "promotion is rejected for an unqualified candidate when policy requires qualification",
        matches!(
            eligibility_rejected,
            Err(KernelRegistryError::PromotionNotEligible { .. })
        ),
        format!("unexpected outcome: {eligibility_rejected:?}"),
    );
    let eligibility_accepted = registry.promote_generation_with_eligibility(
        &kernel,
        generation_eligibility_id,
        crate::kernel_artifact::KernelArtifactTrust::Untrusted,
        crate::kernel_qualification::QualificationStatus::Qualified,
        &ineligible_policy,
    );
    lifecycle_record(
        &mut results,
        "promotion succeeds for a qualified candidate meeting policy",
        eligibility_accepted.is_ok(),
        format!("unexpected outcome: {eligibility_accepted:?}"),
    );

    registry.revoke_kernel_with_policy(
        &kernel,
        "qualification suite defect",
        RevocationInFlightPolicy::CancelIfSafe,
    );
    let request = KernelSelectionRequest::new(
        "conformance-request",
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        crate::ResourceAffinity::new(crate::FallbackClass::Transparent),
    );
    let selection = registry.select(&request);
    lifecycle_record(
        &mut results,
        "revoked Kernel receives no new work",
        matches!(
            selection,
            Err(KernelRegistryError::CandidateIncompatible {
                reason: KernelCandidateRejection::Revoked
            })
        ),
        format!("unexpected outcome: {selection:?}"),
    );

    KernelRegistryLifecycleConformanceReport { results }
}
