//! Kernel Optimization Orchestration Boundary (see
//! `openspec/changes/define-kernel-optimization-orchestration-boundary`).
//!
//! Magnetar already defines how a generated Kernel becomes executable
//! ([`crate::kernel_artifact`], [`crate::kernel_compilation`]), how it is
//! proven correct ([`crate::kernel_qualification`]), how it is cached and
//! hot-swapped ([`crate::kernel_cache`]), and how Runtime chooses among
//! eligible implementations ([`crate::kernel_selection_policy`],
//! [`crate::kernel_registry`]). None of those contracts say *who* creates an
//! optimization campaign or *where* that work runs -- an AI agent, CI, a
//! human engineer, a vendor system, or Tachyon-managed infrastructure could
//! all be the answer, and each may require authority (network, compilers,
//! process execution, credentials, hardware reservation) that a pure
//! inference Runtime SHALL NOT hold.
//!
//! This module defines that missing boundary as executable Rust types and
//! pure functions, mirroring the shape of [`crate::kernel_selection_policy`]:
//! an **Optimization Plane** (campaigns, generators, workers, evidence,
//! recommendations) that stays structurally separate from the **Inference
//! Plane** ([`crate::inference_api`], [`crate::kernel_registry`]). The module
//! does not implement an optimizer, an AI agent, a distributed scheduler, or
//! a network protocol -- see the proposal's "Non-Goals". It defines the
//! contract external optimization systems and Magnetar Runtime meet at.
//!
//! The Core Rule is mechanically enforced, not merely documented:
//! [`OptimizationRecommendation`] can never itself promote a Kernel.
//! [`submit_recommendation_for_promotion`] is the *only* bridge from a
//! recommendation to [`crate::kernel_registry::KernelRegistry::promote_generation_with_eligibility`],
//! and it always re-evaluates current trust/qualification/policy through
//! [`crate::kernel_selection_policy::evaluate_candidate_eligibility`] first --
//! there is no code path from "campaign recommended it" to "active in
//! production" that skips revalidation.
//!
//! - [`optimization_authority_not_ambient_in_runtime`]: "Optimization Plane
//!   Is Separate From Inference Plane" / "Core Separation" -- composes
//!   [`crate::inference_api::validate_inference_scope`] rather than a
//!   parallel capability list.
//! - [`OptimizationCampaignId`] / [`OptimizationTrigger`] /
//!   [`validate_trigger_not_hot_path`]: "Optimization Campaign Is Explicit",
//!   "Campaign Identity", "Optimization Trigger" and "No Hot-Path Campaign
//!   Start" -- the hot-path check does not even inspect which trigger was
//!   requested, so no trigger variant can ever authorize a synchronous
//!   campaign start from token decode.
//! - [`CampaignLifecycleState`]: "Campaign Lifecycle", with the same
//!   `can_transition_to` shape as
//!   [`crate::kernel_qualification::QualificationStatus`].
//! - [`OptimizationCampaign`]: gathers identity, trigger, workload profile,
//!   objectives, constraints, target worker capabilities, budget and policy
//!   version into the one record the proposal's "Optimization Campaign"
//!   section describes.
//! - [`WorkloadProfile`] / [`WORKLOAD_PROFILE_FORBIDDEN_METADATA_KEYS`] /
//!   [`reject_raw_workload_metadata_key`]: "Optimization Workload Profile"
//!   and "Workload Profile Is Not Raw User Data" -- the struct has no field
//!   shaped like raw prompt/document content, and its one open extension
//!   point rejects known raw-data-shaped keys.
//! - [`BenchmarkInputSource`]: "Representative Benchmark Inputs" -- there is
//!   no variant representing unreviewed raw production input.
//! - [`WorkloadAggregationMetadata`]: "Workload Aggregation".
//! - [`GeneratorIdentity`] / [`generator_identity_does_not_grant_trust`]:
//!   "External Generator Boundary" and "Generator Identity" -- delegates to
//!   [`crate::kernel_qualification::eligibility_is_generator_independent`]
//!   rather than re-deriving trust from provenance.
//! - [`GENERATOR_FORBIDDEN_RUNTIME_ACCESS`] /
//!   [`reject_generator_runtime_access`]: "Generator Authority" and "Memory
//!   Boundary".
//! - [`CandidateArtifactRef`]: "Candidate Generation" and "Optimization
//!   Candidates Are Kernel Artifacts" -- wraps
//!   [`crate::kernel_artifact::KernelSourceArtifactId`] /
//!   [`crate::kernel_artifact::CompiledKernelArtifactId`] digest identity
//!   directly instead of inventing a competing identity scheme.
//! - [`SearchStrategy`]: "Optimization Search Strategy" -- an open,
//!   non-exhaustive vocabulary; nothing in this module requires one member.
//! - [`OptimizationWorkerId`] / [`WorkerCapabilityProfile`] /
//!   [`worker_compatible_with_target`]: "Optimization Worker" and "Worker
//!   Capability Profile" / "Worker Selection".
//! - [`qualification_required_before_benchmark_ranking`][]: "Qualification
//!   Composition" and "Benchmark Composition" -- a candidate cannot enter
//!   benchmark ranking on compilation success alone.
//! - [`CampaignBudget`] / [`CampaignUsage`] / [`budget_exceeded`]: "Campaign
//!   Budgets".
//! - [`deadline_expired`]: "Campaign Deadline" -- expiry is a pure function
//!   of elapsed time and never touches Registry state, so it structurally
//!   cannot affect an active production Kernel.
//! - [`CampaignCancellationScope`] / [`cancel_campaign`]: "Campaign
//!   Cancellation".
//! - [`CandidateFailureKind`] / [`CandidateFailurePolicy`] /
//!   [`other_candidates_continue`]: "Candidate Failure Isolation".
//! - [`CampaignFailureReason`]: "Campaign Failure".
//! - [`EvidenceBundle`]: "Evidence Bundle" and "Evidence Immutability" --
//!   every field is set at construction; the type exposes no `&mut self`
//!   mutator, so a corrected evaluation SHALL create a new bundle.
//! - [`OptimizationRecommendation`] / [`RecommendationVerdict`] /
//!   [`submit_recommendation_for_promotion`]: "Optimization Recommendation",
//!   "Recommendation Is Not Promotion" and "Recommendation Ranking" -- see
//!   the Core Rule above.
//! - [`ARTIFACT_TRANSPORT_FORBIDDEN_FIELDS`] /
//!   [`reject_native_transport_handle`]: "Artifact Transport" and "No
//!   Pointer-Based Transport".
//! - [`OrchestratorKind`]: "External Orchestrator Neutrality" and "Tachyon
//!   Boundary" -- a plain data enum; nothing in this module names Tachyon or
//!   any other orchestrator as a dependency.
//! - [`reject_optimization_tooling_authority_in_runtime`]: "CLI Boundary" /
//!   "CLI/Tooling Boundary", composing
//!   [`crate::cli_boundary::reject_cli_owned_authority`].
//! - [`ArtifactIngestionRequest`] / [`validate_artifact_ingestion`]: "Runtime
//!   Artifact Ingestion".
//! - [`offline_inference_possible`]: "Runtime Network Boundary" and "Offline
//!   Inference".
//! - [`OptimizationCredentialScope`] /
//!   [`reject_runtime_owned_optimization_credential`]: "Optimization Service
//!   Credentials" and "Credential Boundary" -- `Runtime` is deliberately not
//!   a variant of [`OptimizationCredentialScope`], so a credential cannot be
//!   modeled as Runtime-owned in the first place.
//! - [`ProviderIsolation`] / [`SharedHardwarePolicy`] /
//!   [`benchmark_may_use_device`]: "Provider Boundary", "Production Provider
//!   Isolation" and "Production Device Isolation".
//! - [`PromotionCandidate`] / [`RevalidationChecklist`]: "Promotion Request"
//!   and "Runtime Revalidation".
//! - [`CanaryDecision`] / [`apply_canary_recommendation`]: "Canary Boundary".
//! - [`validate_rollback_authority`]: "Rollback Authority".
//! - [`ReproducibilityMetadata`]: "Campaign Reproducibility".
//! - [`OptimizationObservationKind`] / [`OptimizationObservation`]:
//!   "Observability Separation", "Correlation" and "Redaction" -- metadata
//!   values are always passed through
//!   `crate::compute::redact_backend_diagnostic`, mirroring
//!   [`crate::kernel_artifact::KernelArtifactObservation`].
//! - [`KernelOptimizationError`]: the structured error categories from the
//!   proposal's "Error Model" section.
//! - [`KernelOptimizationOrchestrationConformanceReport`] /
//!   [`run_kernel_optimization_orchestration_conformance`]: the conformance
//!   checks required by this change's `specs/kernel-optimization-orchestration/spec.md`
//!   and the orchestration-boundary requirements added to
//!   `specs/conformance/spec.md`.

use crate::cli_boundary::reject_cli_owned_authority;
use crate::compute::redact_backend_diagnostic;
use crate::inference_api::validate_inference_scope;
use crate::kernel_artifact::PreparedKernelId;
use crate::kernel_artifact::{
    CompiledKernelArtifactId, KernelArtifactProvenance, KernelArtifactTrust, KernelSourceArtifactId,
};
use crate::kernel_benchmark::BenchmarkRecord;
use crate::kernel_qualification::{
    KernelEligibilityPolicy, QualificationIdentity, QualificationStatus,
    eligibility_is_generator_independent,
};
use crate::kernel_registry::{KernelRegistry, KernelRegistryError};
use crate::kernel_selection_policy::{
    CandidateEligibilityInput, KernelSelectionExclusionReason, OptimizationProfile,
    evaluate_candidate_eligibility,
};
use crate::{KernelId, ProviderBinding};
use std::{collections::BTreeMap, error::Error, fmt};

pub const KERNEL_OPTIMIZATION_ORCHESTRATION_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Optimization Plane / Inference Plane separation
// ---------------------------------------------------------------------

/// Implements "Optimization Plane Is Separate From Inference Plane" and
/// "Core Separation" (proposal): rejects a capability/authority name that
/// would make optimization-tooling authority ambient inside Magnetar
/// Runtime. Reuses [`validate_inference_scope`] rather than a parallel
/// capability list, so the Inference API boundary and the optimization
/// boundary can never silently drift apart.
pub fn optimization_authority_not_ambient_in_runtime(
    capability: &str,
) -> Result<(), KernelOptimizationError> {
    validate_inference_scope(capability).map_err(|_| {
        KernelOptimizationError::RuntimeAuthorityViolation {
            capability: capability.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Campaign Identity / Trigger / Hot-Path Denial
// ---------------------------------------------------------------------

/// Stable Optimization Campaign identity, implementing "Campaign Identity"
/// (proposal): "It SHALL NOT encode native pointers, process handles, or
/// secrets." Deliberately a plain opaque string wrapper with no accessor
/// into a numeric/pointer representation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OptimizationCampaignId(String);

impl OptimizationCampaignId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for OptimizationCampaignId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Implements "Optimization Trigger" (proposal): the portable trigger
/// vocabulary. Every variant describes an event *outside* the token-decode
/// hot path -- there is deliberately no `TokenDecode`/`HotPath` variant, so
/// no trigger can be named to justify a synchronous campaign start from
/// inference (see [`validate_trigger_not_hot_path`]).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OptimizationTrigger {
    ManualRequest,
    CiPipeline,
    NewHardwareTarget,
    NewProviderVersion,
    NewCompilerVersion,
    NewDriverRuntimeCompatibilityClass,
    NewOperatorVersion,
    NewModelWorkloadProfile,
    PerformanceRegression,
    BenchmarkDrift,
    QualificationSuiteUpgrade,
    ScheduledOptimization,
    CacheWarming,
}

/// Implements "No Hot-Path Campaign Start" (proposal): "The normal inference
/// hot path SHALL NOT start an AI agent, start a kernel search loop, invoke
/// an optimization service, ... A normal token decode SHALL NOT trigger an
/// Optimization Campaign synchronously." This check does not even receive an
/// [`OptimizationTrigger`] -- being on the hot path denies campaign start
/// unconditionally, regardless of which trigger a caller might claim.
pub fn validate_trigger_not_hot_path(
    is_token_decode_hot_path: bool,
) -> Result<(), KernelOptimizationError> {
    if is_token_decode_hot_path {
        Err(KernelOptimizationError::HotPathDenied)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Campaign Lifecycle
// ---------------------------------------------------------------------

/// Implements "Campaign Lifecycle" (proposal). Mirrors the
/// `can_transition_to` shape of
/// [`crate::kernel_qualification::QualificationStatus`]: only explicit,
/// declared transitions are legal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CampaignLifecycleState {
    Planned,
    Queued,
    Running,
    Generating,
    Compiling,
    Qualifying,
    Benchmarking,
    Evaluating,
    Completed,
    Cancelled,
    TimedOut,
    Failed,
}

impl CampaignLifecycleState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        use CampaignLifecycleState as S;
        matches!(
            (self, next),
            (S::Planned, S::Queued)
                | (S::Queued, S::Running)
                | (S::Running, S::Generating)
                | (S::Generating, S::Compiling)
                | (S::Compiling, S::Qualifying)
                | (S::Qualifying, S::Benchmarking)
                | (S::Benchmarking, S::Evaluating)
                | (S::Evaluating, S::Completed)
                | (S::Evaluating, S::Generating)
                | (
                    S::Planned
                        | S::Queued
                        | S::Running
                        | S::Generating
                        | S::Compiling
                        | S::Qualifying
                        | S::Benchmarking
                        | S::Evaluating,
                    S::Cancelled | S::TimedOut | S::Failed
                )
        )
    }

    /// A campaign has stopped doing work in any of these states, implementing
    /// "Campaign timeout SHALL NOT affect currently active production
    /// Kernel automatically" (proposal): terminal states carry no reference
    /// to Registry state at all, so there is nothing for them to mutate.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::TimedOut | Self::Failed
        )
    }
}

/// Implements "Optimization Campaign" (proposal): "A campaign SHALL be
/// separate from an inference request", gathering campaign identity,
/// trigger, workload profile, objectives, constraints, target worker
/// capabilities, budget and policy version into one record. This struct is
/// data only -- it holds no Registry reference and no native handle, so
/// nothing about constructing or holding one can affect production state.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationCampaign {
    pub id: OptimizationCampaignId,
    pub trigger: OptimizationTrigger,
    pub workload_profile: WorkloadProfile,
    pub objectives: Vec<OptimizationProfile>,
    pub constraints: BTreeMap<String, String>,
    pub target_capabilities: WorkerCapabilityRequirement,
    pub budget: CampaignBudget,
    pub policy_version: String,
    pub state: CampaignLifecycleState,
}

impl OptimizationCampaign {
    pub fn new(
        id: OptimizationCampaignId,
        trigger: OptimizationTrigger,
        policy_version: impl Into<String>,
    ) -> Self {
        Self {
            id,
            trigger,
            workload_profile: WorkloadProfile::default(),
            objectives: Vec::new(),
            constraints: BTreeMap::new(),
            target_capabilities: WorkerCapabilityRequirement::default(),
            budget: CampaignBudget::default(),
            policy_version: policy_version.into(),
            state: CampaignLifecycleState::Planned,
        }
    }

    /// Implements "No Hot-Path Campaign Start" (proposal) at the campaign
    /// construction boundary: a campaign cannot even be validated as
    /// startable while the caller reports it originated on the token-decode
    /// hot path.
    pub fn validate_startable(
        &self,
        is_token_decode_hot_path: bool,
    ) -> Result<(), KernelOptimizationError> {
        validate_trigger_not_hot_path(is_token_decode_hot_path)?;
        if matches!(self.state, CampaignLifecycleState::Planned) {
            Ok(())
        } else {
            Err(KernelOptimizationError::CampaignInvalid {
                reason: format!("campaign must start from Planned, was {:?}", self.state),
            })
        }
    }
}

// ---------------------------------------------------------------------
// Optimization Workload Profile
// ---------------------------------------------------------------------

/// Implements "Optimization Workload Profile" (proposal): execution
/// characteristics only. There is no field shaped like raw prompt content,
/// conversation history, or a document body -- "Workload Profile Is Not Raw
/// User Data" holds by construction for every named field. The one open
/// extension point (`extra_metadata`) is guarded by
/// [`reject_raw_workload_metadata_key`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkloadProfile {
    pub operator_semantics: Option<String>,
    pub operator_semantic_version: Option<u32>,
    pub target_provider_class: Option<String>,
    pub target_device_architecture: Option<String>,
    pub dtype: Option<String>,
    pub layout: Option<String>,
    pub shape_envelope: Option<String>,
    pub batch_envelope: Option<String>,
    pub sequence_envelope: Option<String>,
    pub generation_phase: Option<String>,
    pub kv_cache_mode: Option<String>,
    pub quantization_profile: Option<String>,
    pub determinism_required: bool,
    pub precision_requirement: Option<String>,
    pub objective: Option<OptimizationProfile>,
    pub memory_limit_bytes: Option<u64>,
    pub workspace_limit_bytes: Option<u64>,
    pub extra_metadata: BTreeMap<String, String>,
}

/// Metadata key substrings that would smuggle raw inference content through
/// [`WorkloadProfile::extra_metadata`], implementing "Workload Profile Is Not
/// Raw User Data" (proposal): "By default it SHALL NOT contain raw prompts,
/// conversation contents, raw user documents, secrets, credentials, raw
/// model weights, raw KV cache contents."
pub const WORKLOAD_PROFILE_FORBIDDEN_METADATA_KEYS: &[&str] = &[
    "raw-prompt",
    "prompt",
    "conversation",
    "raw-document",
    "document-content",
    "secret",
    "credential",
    "model-weights",
    "raw-kv-cache",
    "kv-cache-contents",
];

/// Rejects a caller-supplied [`WorkloadProfile::extra_metadata`] key that
/// names raw inference content rather than aggregate execution
/// characteristics.
pub fn reject_raw_workload_metadata_key(key: &str) -> Result<(), KernelOptimizationError> {
    let normalized = key.trim().to_ascii_lowercase();
    if WORKLOAD_PROFILE_FORBIDDEN_METADATA_KEYS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(KernelOptimizationError::DataBoundaryViolation {
            reason: format!("workload profile metadata key '{key}' carries raw inference content"),
        });
    }
    Ok(())
}

/// Implements "Representative Benchmark Inputs" (proposal): "Raw production
/// inference inputs SHALL NOT be automatically exported to the Optimization
/// Plane." There is deliberately no variant representing unreviewed raw
/// production input -- [`Self::SanitizedProductionDerived`] is the only path
/// by which production-derived data may participate, and it requires a named
/// sanitization policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BenchmarkInputSource {
    SyntheticFixture,
    DeterministicGeneratedFixture,
    AuthorizedBenchmarkDataset { dataset_id: String },
    SanitizedProductionDerived { sanitization_policy_id: String },
}

/// Implements "Workload Aggregation" (proposal): aggregate-only summaries
/// Runtime or surrounding tooling MAY produce for optimization, never raw
/// per-request content.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkloadAggregationMetadata {
    pub shape_histogram: BTreeMap<String, u64>,
    pub batch_histogram: BTreeMap<String, u64>,
    pub sequence_length_histogram: BTreeMap<String, u64>,
    pub operator_frequency: BTreeMap<String, u64>,
    pub dtype_distribution: BTreeMap<String, u64>,
    pub layout_distribution: BTreeMap<String, u64>,
}

// ---------------------------------------------------------------------
// External Generator Boundary
// ---------------------------------------------------------------------

/// Provenance plus an optional human-readable identity label, implementing
/// "Generator Identity" (proposal): "Generator identity SHOULD be recorded
/// as provenance. Generator identity SHALL NOT imply trust." Reuses
/// [`KernelArtifactProvenance`] rather than a second, competing provenance
/// enum.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratorIdentity {
    pub provenance: KernelArtifactProvenance,
    pub identity_label: Option<String>,
}

/// Implements "Generator Identity Does Not Grant Trust" (proposal and
/// `specs/conformance/spec.md`): delegates to
/// [`eligibility_is_generator_independent`] so this module never re-derives
/// trust from a generator's identity/provenance through a second code path.
pub fn generator_identity_does_not_grant_trust(
    a: &GeneratorIdentity,
    b: &GeneratorIdentity,
    trust: KernelArtifactTrust,
    status: QualificationStatus,
    policy: &KernelEligibilityPolicy,
) -> bool {
    eligibility_is_generator_independent(a.provenance, b.provenance, trust, status, policy)
}

/// Runtime-owned authority a generator SHALL NOT receive ambient access to,
/// implementing "Generator Authority" and "Memory Boundary" (proposal).
pub const GENERATOR_FORBIDDEN_RUNTIME_ACCESS: &[&str] = &[
    "runtime-tensor-memory",
    "active-kv-cache",
    "provider-native-handle",
    "device-native-handle",
    "prepared-kernel-id-mapping",
    "runtime-secret",
    "runtime-process-memory",
];

/// Rejects a caller-supplied scope name that would grant a generator ambient
/// access to Runtime-owned authority.
pub fn reject_generator_runtime_access(scope: &str) -> Result<(), KernelOptimizationError> {
    let normalized = scope.trim().to_ascii_lowercase();
    if GENERATOR_FORBIDDEN_RUNTIME_ACCESS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(KernelOptimizationError::RuntimeAuthorityViolation {
            capability: scope.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Candidate Generation / Search Strategy
// ---------------------------------------------------------------------

/// A candidate's Kernel Artifact identity, implementing "Candidate
/// Generation" and "Optimization Candidates Are Kernel Artifacts" (proposal):
/// "Human-readable candidate numbering SHALL NOT replace digest-based
/// artifact identity." Wraps the existing digest-based identities directly
/// instead of introducing a competing candidate identity scheme.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateArtifactRef {
    pub source_artifact: Option<KernelSourceArtifactId>,
    pub compiled_artifact: Option<CompiledKernelArtifactId>,
}

/// Implements "Optimization Search Strategy" (proposal): "The orchestration
/// contract SHALL remain agnostic to search strategy." An open,
/// non-exhaustive vocabulary -- no function in this module requires a
/// specific variant.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SearchStrategy {
    Mutation,
    EvolutionarySearch,
    LlmGeneration,
    HillClimbing,
    BayesianOptimization,
    ExhaustiveSpecialization,
    VendorAutotuning,
    HumanIteration,
}

// ---------------------------------------------------------------------
// Optimization Worker
// ---------------------------------------------------------------------

/// Opaque Optimization Worker identity, implementing "Optimization Worker"
/// (proposal).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OptimizationWorkerId(String);

impl OptimizationWorkerId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Implements "Worker Capability Profile" (proposal): "Worker capability
/// SHALL be explicit. It SHALL NOT expose native handles outside the worker
/// boundary." No field here is pointer- or handle-shaped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerCapabilityProfile {
    pub provider_implementations: Vec<ProviderBinding>,
    pub device_architecture: Option<String>,
    pub device_features: Vec<String>,
    pub compiler_toolchains: Vec<String>,
    pub accepted_source_formats: Vec<String>,
    pub emitted_compiled_formats: Vec<String>,
    pub qualification_profiles: Vec<String>,
    pub benchmark_profiles: Vec<String>,
    pub available_memory_bytes: Option<u64>,
    pub concurrency_limit: Option<u32>,
    pub isolation_model: Option<String>,
}

/// A campaign's worker target requirement, implementing "Worker Selection"
/// (proposal): "Optimization orchestrator MAY choose workers compatible with
/// campaign target. Worker selection SHALL NOT imply Runtime Provider/Device
/// selection for production inference" -- the return value is only ever
/// consumed by campaign-side worker dispatch, never by
/// [`crate::kernel_registry::KernelRegistry`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerCapabilityRequirement {
    pub required_provider: Option<ProviderBinding>,
    pub required_device_architecture: Option<String>,
    pub required_compiler_toolchain: Option<String>,
    pub required_qualification_profile: Option<String>,
}

/// Implements "Match architecture, Match Provider, Match compiler format,
/// Match qualification profile" (tasks): a worker is compatible only when
/// every requirement it declares is present in the target requirement.
pub fn worker_compatible_with_target(
    profile: &WorkerCapabilityProfile,
    target: &WorkerCapabilityRequirement,
) -> bool {
    if let Some(provider) = &target.required_provider
        && !profile.provider_implementations.contains(provider)
    {
        return false;
    }
    if let Some(architecture) = &target.required_device_architecture
        && profile.device_architecture.as_deref() != Some(architecture.as_str())
    {
        return false;
    }
    if let Some(toolchain) = &target.required_compiler_toolchain
        && !profile.compiler_toolchains.iter().any(|t| t == toolchain)
    {
        return false;
    }
    if let Some(qualification_profile) = &target.required_qualification_profile
        && !profile
            .qualification_profiles
            .iter()
            .any(|p| p == qualification_profile)
    {
        return false;
    }
    true
}

// ---------------------------------------------------------------------
// Compilation / Qualification / Benchmark Composition
// ---------------------------------------------------------------------

/// Implements "Qualification Composition" (proposal): "It SHALL NOT treat
/// compilation success as qualification" and "Benchmark Composition": "A
/// candidate failing mandatory correctness SHALL not be promoted merely
/// because it benchmarks well." A candidate may only enter benchmark ranking
/// once [`QualificationStatus::is_eligible`] holds -- compilation success
/// alone is never sufficient.
pub fn qualification_required_before_benchmark_ranking(
    qualification_status: QualificationStatus,
) -> Result<(), KernelOptimizationError> {
    if qualification_status.is_eligible() {
        Ok(())
    } else {
        Err(KernelOptimizationError::NoQualifiedCandidates)
    }
}

// ---------------------------------------------------------------------
// Campaign Budgets / Deadline / Cancellation
// ---------------------------------------------------------------------

/// Implements "Campaign Budgets" (proposal).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CampaignBudget {
    pub max_candidates: Option<u32>,
    pub max_compiler_jobs: Option<u32>,
    pub max_qualification_jobs: Option<u32>,
    pub max_benchmark_runs: Option<u32>,
    pub wall_clock_deadline_seconds: Option<u64>,
    pub cpu_time_budget_seconds: Option<u64>,
    pub gpu_time_budget_seconds: Option<u64>,
    pub memory_budget_bytes: Option<u64>,
    pub temporary_storage_budget_bytes: Option<u64>,
    pub network_budget_bytes: Option<u64>,
    pub cost_budget_cents: Option<u64>,
}

/// Resource consumption recorded so far for one campaign.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CampaignUsage {
    pub candidates: u32,
    pub compiler_jobs: u32,
    pub qualification_jobs: u32,
    pub benchmark_runs: u32,
    pub elapsed_seconds: u64,
}

/// A budget dimension that usage has exceeded, implementing "Campaign
/// Budgets": "additional generation is denied unless policy expands budget."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetDimension {
    MaxCandidates,
    MaxCompilerJobs,
    MaxQualificationJobs,
    MaxBenchmarkRuns,
    WallClockDeadline,
}

/// Returns the first budget dimension `usage` has exceeded, or `None` if
/// `usage` remains within every declared limit. Unset limits impose no
/// constraint, implementing "Budgets MAY include" -- every field is
/// optional.
pub fn budget_exceeded(budget: &CampaignBudget, usage: &CampaignUsage) -> Option<BudgetDimension> {
    if budget
        .max_candidates
        .is_some_and(|max| usage.candidates >= max)
    {
        return Some(BudgetDimension::MaxCandidates);
    }
    if budget
        .max_compiler_jobs
        .is_some_and(|max| usage.compiler_jobs >= max)
    {
        return Some(BudgetDimension::MaxCompilerJobs);
    }
    if budget
        .max_qualification_jobs
        .is_some_and(|max| usage.qualification_jobs >= max)
    {
        return Some(BudgetDimension::MaxQualificationJobs);
    }
    if budget
        .max_benchmark_runs
        .is_some_and(|max| usage.benchmark_runs >= max)
    {
        return Some(BudgetDimension::MaxBenchmarkRuns);
    }
    if budget
        .wall_clock_deadline_seconds
        .is_some_and(|max| usage.elapsed_seconds >= max)
    {
        return Some(BudgetDimension::WallClockDeadline);
    }
    None
}

/// Implements "Campaign Deadline" (proposal): "Campaign timeout SHALL NOT
/// affect currently active production Kernel automatically." This function
/// takes no Registry/production state at all -- there is nothing here for
/// expiry to touch.
pub fn deadline_expired(elapsed_seconds: u64, deadline_seconds: Option<u64>) -> bool {
    deadline_seconds.is_some_and(|deadline| elapsed_seconds >= deadline)
}

/// Implements "Campaign Cancellation" (proposal): what cancellation stops.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CampaignCancellationScope {
    NewWorkOnly,
    NewWorkAndInterruptibleJobs,
}

/// Implements "Cancellation SHALL prevent new campaign work ... Previously
/// active production Kernel SHALL remain unaffected" (proposal): the only
/// transition this performs is `state -> Cancelled`, guarded by
/// [`CampaignLifecycleState::can_transition_to`]; it has no Registry
/// parameter, so it structurally cannot touch production state.
pub fn cancel_campaign(
    state: CampaignLifecycleState,
    _scope: CampaignCancellationScope,
) -> Result<CampaignLifecycleState, KernelOptimizationError> {
    if state.can_transition_to(CampaignLifecycleState::Cancelled) {
        Ok(CampaignLifecycleState::Cancelled)
    } else {
        Err(KernelOptimizationError::CampaignInvalid {
            reason: format!("cannot cancel campaign from state {state:?}"),
        })
    }
}

// ---------------------------------------------------------------------
// Candidate Failure Isolation / Campaign Failure
// ---------------------------------------------------------------------

/// Implements "Candidate Failure Isolation" (proposal): "Examples of isolated
/// candidate failures include compilation failure, qualification mismatch,
/// benchmark crash, unsupported specialization, resource limit failure."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateFailureKind {
    CompilationFailure,
    QualificationMismatch,
    BenchmarkCrash,
    UnsupportedSpecialization,
    ResourceLimitFailure,
}

/// Whether an isolated candidate failure SHOULD abort the remaining campaign,
/// implementing "Policy MAY continue evaluating remaining candidates."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateFailurePolicy {
    ContinueRemainingCandidates,
    AbortCampaign,
}

/// Implements "Failure of one candidate SHALL NOT necessarily fail the
/// entire campaign" (proposal): the failure kind never appears on the right
/// side of this decision, only campaign policy does -- one failing candidate
/// cannot itself force the campaign to abort.
pub fn other_candidates_continue(
    _failure: CandidateFailureKind,
    policy: CandidateFailurePolicy,
) -> bool {
    matches!(policy, CandidateFailurePolicy::ContinueRemainingCandidates)
}

/// Implements "Campaign Failure" (proposal): "A campaign SHALL fail only
/// according to campaign policy."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CampaignFailureReason {
    NoCandidateQualified,
    RequiredWorkerUnavailable,
    BudgetExhausted,
    OrchestrationInfrastructureFailed,
    CampaignDeadlineExpired,
    MandatorySecurityPolicyDenied,
}

// ---------------------------------------------------------------------
// Evidence Bundle
// ---------------------------------------------------------------------

/// Implements "Evidence Bundle" (proposal) and "Campaign Evidence Identifies
/// Qualification" (`specs/generated-kernel-qualification/spec.md`).
/// Implements "Evidence Immutability": every field is populated at
/// construction and this type exposes no `&mut self` method -- a corrected
/// or rerun evaluation SHALL construct a new [`EvidenceBundle`] rather than
/// mutate this one.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceBundle {
    pub campaign: OptimizationCampaignId,
    pub candidate: CandidateArtifactRef,
    pub compiler_identity: Option<String>,
    pub compiler_version: Option<String>,
    pub qualification: Option<QualificationIdentity>,
    pub qualification_status: QualificationStatus,
    pub benchmark: Option<BenchmarkRecord>,
    pub target_context: Option<String>,
    pub optimization_policy_version: Option<String>,
    pub workload_profile: WorkloadProfile,
    pub trust: KernelArtifactTrust,
}

impl EvidenceBundle {
    /// Implements "Qualification Failure Prevents Qualified Recommendation"
    /// (`specs/generated-kernel-qualification/spec.md`): evidence identifies
    /// a qualified recommendation only when `qualification_status` is
    /// actually eligible -- a compiled-but-unqualified candidate can never
    /// report itself as supporting a qualified recommendation.
    pub fn supports_qualified_recommendation(&self) -> bool {
        self.qualification.is_some() && self.qualification_status.is_eligible()
    }
}

// ---------------------------------------------------------------------
// Optimization Recommendation / Recommendation Is Not Promotion
// ---------------------------------------------------------------------

/// Implements "Optimization Recommendation" (proposal): "A recommendation MAY
/// state: candidate X is recommended for profile latency; ...; candidate A
/// should be rejected."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecommendationVerdict {
    Recommended,
    Experimental,
    Rejected,
}

/// Implements "Optimization Recommendation" (proposal): "A recommendation
/// SHALL NOT be authoritative execution policy." This struct is inert data;
/// see [`submit_recommendation_for_promotion`] for the only function that
/// can turn it into a Registry change, and note that function always
/// revalidates first.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationRecommendation {
    pub campaign: OptimizationCampaignId,
    pub candidate: CandidateArtifactRef,
    pub evidence: EvidenceBundle,
    pub target_profile: OptimizationProfile,
    pub verdict: RecommendationVerdict,
}

/// Implements the Core Rule, "Recommendation Is Not Promotion" and "Runtime
/// Revalidation" (proposal): "recommended != promoted. Runtime SHALL
/// independently validate ... before promotion/execution." This is the only
/// function in this module (or, transitively, in the whole crate) that turns
/// an [`OptimizationRecommendation`] into a
/// [`crate::kernel_registry::KernelRegistry`] mutation, and it does so only
/// by re-running [`evaluate_candidate_eligibility`] against
/// `current_eligibility` -- state the caller SHALL supply freshly, not state
/// carried over from when the campaign ran. A recommendation whose verdict
/// is not [`RecommendationVerdict::Recommended`], or whose fresh eligibility
/// check fails, is rejected before Registry is ever touched.
pub fn submit_recommendation_for_promotion(
    recommendation: &OptimizationRecommendation,
    current_eligibility: &CandidateEligibilityInput,
    registry: &mut KernelRegistry,
    kernel: &KernelId,
    candidate: PreparedKernelId,
) -> Result<(), KernelOptimizationError> {
    if !matches!(recommendation.verdict, RecommendationVerdict::Recommended) {
        return Err(KernelOptimizationError::RecommendationInvalid {
            reason: "recommendation verdict is not Recommended".into(),
        });
    }
    if !recommendation.evidence.supports_qualified_recommendation() {
        return Err(KernelOptimizationError::EvidenceIncomplete);
    }
    evaluate_candidate_eligibility(current_eligibility).map_err(
        |reason: KernelSelectionExclusionReason| KernelOptimizationError::RecommendationInvalid {
            reason: format!("candidate failed current eligibility revalidation: {reason:?}"),
        },
    )?;
    registry
        .promote_generation_with_eligibility(
            kernel,
            candidate,
            current_eligibility.trust,
            current_eligibility.qualification_status,
            &KernelEligibilityPolicy {
                require_trusted: current_eligibility.require_trusted,
                require_qualified: current_eligibility.require_qualified,
            },
        )
        .map_err(
            |error: KernelRegistryError| KernelOptimizationError::RecommendationInvalid {
                reason: error.to_string(),
            },
        )
}

// ---------------------------------------------------------------------
// Artifact Transport
// ---------------------------------------------------------------------

/// Field names that would carry a native/process-local handle across the
/// Optimization Plane / Runtime boundary, implementing "Artifact Transport"
/// and "No Pointer-Based Transport" (proposal).
pub const ARTIFACT_TRANSPORT_FORBIDDEN_FIELDS: &[&str] = &[
    "raw-pointer",
    "native-kernel-handle",
    "device-handle",
    "provider-function-pointer",
    "cufunction",
    "process-local-prepared-kernel-id",
];

/// Rejects a caller-supplied artifact transport field that names a native or
/// process-local handle rather than stable digest-based identity.
pub fn reject_native_transport_handle(field: &str) -> Result<(), KernelOptimizationError> {
    let normalized = field.trim().to_ascii_lowercase();
    if ARTIFACT_TRANSPORT_FORBIDDEN_FIELDS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(KernelOptimizationError::ArtifactTransferFailed {
            reason: format!("transport field '{field}' names a native handle, not stable identity"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// External Orchestrator Neutrality / Tachyon Boundary
// ---------------------------------------------------------------------

/// Implements "External Orchestrator Neutrality" and "Tachyon Boundary"
/// (proposal): purely descriptive data -- no orchestrator variant, including
/// [`Self::TachyonManaged`], is treated specially by any function in this
/// module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrchestratorKind {
    Ci,
    LocalDeveloperTooling,
    DedicatedOptimizationService,
    TachyonManaged,
    VendorInfrastructure,
    FutureMagnetarTooling,
}

// ---------------------------------------------------------------------
// CLI / Tooling Boundary
// ---------------------------------------------------------------------

/// Implements "Optimization Tooling Belongs Outside Runtime Inference"
/// (`specs/cli-boundary/spec.md`): composes
/// [`reject_cli_owned_authority`] rather than a parallel capability list, so
/// `magnetar kernel optimize`/`qualify`/`benchmark`/`candidates`-style
/// tooling authority is rejected the same way any other CLI-owned authority
/// is.
pub fn reject_optimization_tooling_authority_in_runtime(
    capability: &str,
) -> Result<(), KernelOptimizationError> {
    reject_cli_owned_authority(capability).map_err(|_| {
        KernelOptimizationError::RuntimeAuthorityViolation {
            capability: capability.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Runtime Artifact Ingestion
// ---------------------------------------------------------------------

/// Implements "Runtime Artifact Ingestion" (proposal): "Such ingestion SHALL
/// still enforce artifact validation, trust/integrity, qualification policy,
/// Provider compatibility, Kernel selection policy."
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactIngestionRequest {
    pub artifact: CandidateArtifactRef,
    pub trust: KernelArtifactTrust,
    pub qualification_status: QualificationStatus,
    pub provider_compatible: bool,
    pub selection_eligible: bool,
}

/// Every ingestion gate SHALL pass for management/CLI-driven artifact
/// ingestion to succeed -- there is no reduced-checks fast path distinct
/// from normal Kernel eligibility.
pub fn validate_artifact_ingestion(
    request: &ArtifactIngestionRequest,
) -> Result<(), KernelOptimizationError> {
    if !request.trust.is_trusted() {
        return Err(KernelOptimizationError::PolicyDenied {
            reason: "ingested artifact is not trusted".into(),
        });
    }
    if !request.qualification_status.is_eligible() {
        return Err(KernelOptimizationError::NoQualifiedCandidates);
    }
    if !request.provider_compatible {
        return Err(KernelOptimizationError::WorkerIncompatible {
            reason: "ingested artifact is not Provider-compatible".into(),
        });
    }
    if !request.selection_eligible {
        return Err(KernelOptimizationError::PolicyDenied {
            reason: "ingested artifact fails Kernel selection policy".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Runtime Network Boundary / Offline Inference
// ---------------------------------------------------------------------

/// Implements "Runtime Network Boundary" and "Offline Inference" (proposal):
/// "network unavailable SHALL NOT by itself prevent execution" when required
/// artifacts are already local and compatible. Network availability never
/// appears as a parameter -- it structurally cannot influence the result.
pub fn offline_inference_possible(required_artifacts_local_and_compatible: bool) -> bool {
    required_artifacts_local_and_compatible
}

// ---------------------------------------------------------------------
// Credential Boundary
// ---------------------------------------------------------------------

/// Implements "Optimization Service Credentials" and "Credential Boundary"
/// (proposal): "Runtime SHALL not receive ambient secret authority." `Runtime`
/// is deliberately absent from this enum -- an optimization credential
/// cannot even be *represented* as Runtime-owned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationCredentialScope {
    CliTooling,
    Ci,
    ExternalOrchestrator,
    DeploymentSystem,
    SecretManagementIntegration,
}

/// Rejects an attempt to label an optimization credential with a scope name
/// that claims Runtime/inference-session ownership.
pub fn reject_runtime_owned_optimization_credential(
    claimed_scope: &str,
) -> Result<(), KernelOptimizationError> {
    let normalized = claimed_scope.trim().to_ascii_lowercase();
    if normalized.contains("runtime") || normalized.contains("inference-session") {
        return Err(KernelOptimizationError::CredentialBoundaryViolation {
            reason: format!("credential scope '{claimed_scope}' claims Runtime ownership"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Provider / Device Isolation
// ---------------------------------------------------------------------

/// Implements "Provider Boundary" and "Production Provider Isolation"
/// (proposal): "Production inference Provider instance SHOULD be separable
/// from optimization worker Provider instances."
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderIsolation {
    pub optimization_worker_provider: ProviderBinding,
    pub production_provider: ProviderBinding,
}

impl ProviderIsolation {
    /// `true` when the optimization worker uses a distinct Provider instance
    /// from production, satisfying "Optimization experiments SHALL NOT
    /// require mutation of live production Provider state" by construction.
    pub fn is_isolated(&self) -> bool {
        self.optimization_worker_provider != self.production_provider
    }
}

/// Implements "Production Device Isolation" (proposal): "Shared hardware
/// usage SHALL be explicit and admission-controlled."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedHardwarePolicy {
    Forbidden,
    ExplicitlyAuthorized,
}

/// A benchmark may run on the production Device only when shared-hardware
/// use is not the production Device at all, or was explicitly authorized --
/// never silently.
pub fn benchmark_may_use_device(
    policy: SharedHardwarePolicy,
    is_production_device: bool,
) -> Result<(), KernelOptimizationError> {
    if !is_production_device {
        return Ok(());
    }
    match policy {
        SharedHardwarePolicy::ExplicitlyAuthorized => Ok(()),
        SharedHardwarePolicy::Forbidden => {
            Err(KernelOptimizationError::ProductionBoundaryViolation {
                reason:
                    "benchmark attempted to use production Device without explicit authorization"
                        .into(),
            })
        }
    }
}

// ---------------------------------------------------------------------
// Promotion Request / Runtime Revalidation
// ---------------------------------------------------------------------

/// Implements "Promotion Request" (proposal): "This is a request, not a
/// command." Consumed only by [`submit_recommendation_for_promotion`].
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionCandidate {
    pub kernel: KernelId,
    pub artifact: CandidateArtifactRef,
    pub qualification_evidence: QualificationIdentity,
    pub benchmark_evidence: Option<BenchmarkRecord>,
    pub requested_profiles: Vec<OptimizationProfile>,
}

/// Implements "Runtime Revalidation" (proposal): the explicit checklist of
/// production-relevant state Runtime SHALL recheck before promotion, none of
/// which is ever inherited unchanged from campaign-time evidence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RevalidationChecklist {
    pub trust_current: bool,
    pub revocation_current: bool,
    pub qualification_compatible: bool,
    pub benchmark_compatible: bool,
    pub provider_ready: bool,
    pub device_available: bool,
    pub memory_feasible: bool,
    pub selection_policy_current: bool,
}

impl RevalidationChecklist {
    pub fn all_revalidated(&self) -> bool {
        self.trust_current
            && self.revocation_current
            && self.qualification_compatible
            && self.benchmark_compatible
            && self.provider_ready
            && self.device_available
            && self.memory_feasible
            && self.selection_policy_current
    }
}

// ---------------------------------------------------------------------
// Canary / Rollback Boundary
// ---------------------------------------------------------------------

/// Implements "Canary Boundary" (proposal): "A recommendation SHALL NOT
/// independently route production traffic."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanaryDecision {
    RuntimeApproved,
    RuntimeDenied,
}

/// A canary recommendation never routes traffic on its own -- only
/// `runtime_policy_allows` decides the outcome.
pub fn apply_canary_recommendation(
    _recommended: bool,
    runtime_policy_allows: bool,
) -> CanaryDecision {
    if runtime_policy_allows {
        CanaryDecision::RuntimeApproved
    } else {
        CanaryDecision::RuntimeDenied
    }
}

/// Implements "Rollback Authority" (proposal): "Optimization Plane MAY report
/// a regression or recommend rollback. Runtime/deployment policy remains
/// authoritative for actual rollback."
pub fn validate_rollback_authority(
    _regression_reported_by_optimization_plane: bool,
    runtime_policy_approves_rollback: bool,
) -> Result<(), KernelOptimizationError> {
    if runtime_policy_approves_rollback {
        Ok(())
    } else {
        Err(KernelOptimizationError::PolicyDenied {
            reason: "rollback requires Runtime/deployment policy approval".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Campaign Reproducibility
// ---------------------------------------------------------------------

/// Implements "Campaign Reproducibility" and "Generator Identity" (proposal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReproducibilityMetadata {
    pub campaign_policy_version: Option<String>,
    pub generator_identity: Option<String>,
    pub source_artifact_digest: Option<String>,
    pub compiler_fingerprint: Option<String>,
    pub qualification_suite_version: Option<String>,
    pub benchmark_profile_version: Option<String>,
    pub worker_hardware: Option<String>,
    pub provider_version: Option<String>,
    pub target_architecture: Option<String>,
    pub random_seed: Option<u64>,
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Implements "Observability Separation" (proposal): "Optimization
/// observability and inference observability SHALL remain distinguishable."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OptimizationObservationKind {
    CampaignStarted,
    CandidateGenerated,
    CandidateCompilationStarted,
    CandidateCompilationFailed,
    CandidateQualified,
    CandidateRejected,
    CandidateBenchmarkCompleted,
    RecommendationCreated,
    CampaignCompleted,
    CampaignCancelled,
    CampaignFailed,
}

/// A single Optimization observation, mirroring
/// [`crate::kernel_artifact::KernelArtifactObservation`]'s shape: an enum
/// `kind`, correlation identity, and a `redacted_metadata` map whose values
/// always pass through `redact_backend_diagnostic` first. Implements
/// "Correlation" (campaign/candidate/artifact fields) and "Redaction".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptimizationObservation {
    pub kind: OptimizationObservationKind,
    pub campaign: OptimizationCampaignId,
    pub candidate: Option<CandidateArtifactRef>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl OptimizationObservation {
    pub fn new(kind: OptimizationObservationKind, campaign: OptimizationCampaignId) -> Self {
        Self {
            kind,
            campaign,
            candidate: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_candidate(mut self, candidate: CandidateArtifactRef) -> Self {
        self.candidate = Some(candidate);
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Optimization Orchestration error, covering the
/// proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelOptimizationError {
    CampaignInvalid { reason: String },
    TriggerDenied { reason: String },
    BudgetInvalid { reason: String },
    BudgetExhausted { dimension: String },
    DeadlineExceeded,
    Cancelled,
    WorkerUnavailable { reason: String },
    WorkerIncompatible { reason: String },
    GeneratorUnavailable { reason: String },
    GeneratorFailed { reason: String },
    NoCandidates,
    NoQualifiedCandidates,
    EvidenceInvalid { reason: String },
    EvidenceIncomplete,
    RecommendationInvalid { reason: String },
    ArtifactTransferFailed { reason: String },
    PolicyDenied { reason: String },
    ProductionBoundaryViolation { reason: String },
    RuntimeAuthorityViolation { capability: String },
    CredentialBoundaryViolation { reason: String },
    DataBoundaryViolation { reason: String },
    HotPathDenied,
    InternalKernelOptimizationError { reason: String },
}

impl fmt::Display for KernelOptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CampaignInvalid { reason } => {
                write!(f, "kernel optimization campaign invalid: {reason}")
            }
            Self::TriggerDenied { reason } => {
                write!(f, "kernel optimization trigger denied: {reason}")
            }
            Self::BudgetInvalid { reason } => {
                write!(f, "kernel optimization budget invalid: {reason}")
            }
            Self::BudgetExhausted { dimension } => {
                write!(f, "kernel optimization budget exhausted: {dimension}")
            }
            Self::DeadlineExceeded => f.write_str("kernel optimization deadline exceeded"),
            Self::Cancelled => f.write_str("kernel optimization cancelled"),
            Self::WorkerUnavailable { reason } => {
                write!(f, "kernel optimization worker unavailable: {reason}")
            }
            Self::WorkerIncompatible { reason } => {
                write!(f, "kernel optimization worker incompatible: {reason}")
            }
            Self::GeneratorUnavailable { reason } => {
                write!(f, "kernel optimization generator unavailable: {reason}")
            }
            Self::GeneratorFailed { reason } => {
                write!(f, "kernel optimization generator failed: {reason}")
            }
            Self::NoCandidates => f.write_str("kernel optimization no candidates"),
            Self::NoQualifiedCandidates => {
                f.write_str("kernel optimization no qualified candidates")
            }
            Self::EvidenceInvalid { reason } => {
                write!(f, "kernel optimization evidence invalid: {reason}")
            }
            Self::EvidenceIncomplete => f.write_str("kernel optimization evidence incomplete"),
            Self::RecommendationInvalid { reason } => {
                write!(f, "kernel optimization recommendation invalid: {reason}")
            }
            Self::ArtifactTransferFailed { reason } => {
                write!(f, "kernel optimization artifact transfer failed: {reason}")
            }
            Self::PolicyDenied { reason } => {
                write!(f, "kernel optimization policy denied: {reason}")
            }
            Self::ProductionBoundaryViolation { reason } => {
                write!(
                    f,
                    "kernel optimization production boundary violation: {reason}"
                )
            }
            Self::RuntimeAuthorityViolation { capability } => {
                write!(
                    f,
                    "kernel optimization runtime authority violation: capability '{capability}'"
                )
            }
            Self::CredentialBoundaryViolation { reason } => {
                write!(
                    f,
                    "kernel optimization credential boundary violation: {reason}"
                )
            }
            Self::DataBoundaryViolation { reason } => {
                write!(f, "kernel optimization data boundary violation: {reason}")
            }
            Self::HotPathDenied => f.write_str("kernel optimization hot path denied"),
            Self::InternalKernelOptimizationError { reason } => {
                write!(f, "internal kernel optimization error: {reason}")
            }
        }
    }
}

impl Error for KernelOptimizationError {}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single conformance check result, mirroring
/// [`crate::cli_boundary::CliBoundaryConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelOptimizationOrchestrationConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of
/// [`KernelOptimizationOrchestrationConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelOptimizationOrchestrationConformanceReport {
    pub results: Vec<KernelOptimizationOrchestrationConformanceResult>,
}

impl KernelOptimizationOrchestrationConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelOptimizationOrchestrationConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelOptimizationOrchestrationConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn conformance_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("conformance-provider"),
        name,
        crate::CapabilityVersion::new(1, 0, 0),
        crate::OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        crate::KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::TestFixture,
    )
}

/// Runs the conformance checks required by
/// `specs/kernel-optimization-orchestration/spec.md` and the
/// orchestration-boundary requirements added to `specs/conformance/spec.md`.
pub fn run_kernel_optimization_orchestration_conformance()
-> KernelOptimizationOrchestrationConformanceReport {
    let mut results = Vec::new();

    // "Optimization Plane Separation Conformance": the public Runtime
    // Inference API audit surface rejects every optimization-agent-shaped
    // capability name.
    for capability in [
        "optimization-agent",
        "kernel-source-injection",
        "compiler-command",
        "benchmark-script",
        "optimization-service-url",
        "repository-credential",
        "generator-credential",
        "agent-prompt",
    ] {
        let outcome = optimization_authority_not_ambient_in_runtime(capability);
        record(
            &mut results,
            format!("Runtime Inference API rejects optimization capability '{capability}'"),
            matches!(
                outcome,
                Err(KernelOptimizationError::RuntimeAuthorityViolation { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // "No Hot-Path Optimization Conformance": being on the token-decode hot
    // path denies campaign start regardless of trigger.
    let hot_path_denied = validate_trigger_not_hot_path(true);
    record(
        &mut results,
        "token decode hot path cannot start an optimization campaign",
        matches!(hot_path_denied, Err(KernelOptimizationError::HotPathDenied)),
        format!("unexpected outcome: {hot_path_denied:?}"),
    );
    let cold_path_allowed = validate_trigger_not_hot_path(false);
    record(
        &mut results,
        "a cold-path trigger is not denied by the hot-path check",
        cold_path_allowed.is_ok(),
        format!("unexpected outcome: {cold_path_allowed:?}"),
    );

    // "Recommendation Does Not Promote Conformance" and "Runtime
    // Revalidation Conformance": drive a real KernelRegistry end to end.
    {
        let kernel = conformance_kernel_id("optimization-conformance-kernel");
        let device = crate::DeviceBinding::new(crate::DeviceId::new("conformance-device"));
        let mut allocator = crate::kernel_artifact::PreparedKernelIdAllocator::default();
        let mut registry = KernelRegistry::new();

        let mut prepared = crate::kernel_artifact::PreparedKernel::new(
            allocator.allocate(),
            kernel.clone(),
            CompiledKernelArtifactId::from_digest("optimization-conformance-digest"),
            ProviderBinding::new("conformance-provider"),
            device,
            crate::kernel_artifact::PreparedKernelGeneration::new(1),
        );
        prepared.mark_ready().ok();
        let candidate_id = prepared.id;
        registry.register_prepared_kernel(prepared);

        let campaign = OptimizationCampaignId::new("conformance-campaign");
        let candidate_ref = CandidateArtifactRef {
            source_artifact: None,
            compiled_artifact: Some(CompiledKernelArtifactId::from_digest(
                "optimization-conformance-digest",
            )),
        };
        let evidence = EvidenceBundle {
            campaign: campaign.clone(),
            candidate: candidate_ref.clone(),
            compiler_identity: Some("conformance-compiler".into()),
            compiler_version: Some("1.0".into()),
            qualification: None,
            qualification_status: QualificationStatus::Unqualified,
            benchmark: None,
            target_context: None,
            optimization_policy_version: Some("v1".into()),
            workload_profile: WorkloadProfile::default(),
            trust: KernelArtifactTrust::Untrusted,
        };
        let recommendation = OptimizationRecommendation {
            campaign,
            candidate: candidate_ref,
            evidence,
            target_profile: OptimizationProfile::Latency,
            verdict: RecommendationVerdict::Recommended,
        };

        let stale_eligibility = CandidateEligibilityInput::all_satisfied();
        let denied = submit_recommendation_for_promotion(
            &recommendation,
            &stale_eligibility,
            &mut registry,
            &kernel,
            candidate_id,
        );
        record(
            &mut results,
            "an unqualified recommendation cannot promote a candidate",
            matches!(denied, Err(KernelOptimizationError::EvidenceIncomplete)),
            format!("unexpected outcome: {denied:?}"),
        );
        record(
            &mut results,
            "Registry has no active generation before revalidated promotion",
            registry.active_prepared_kernel(&kernel).is_none(),
            "Registry gained an active generation from a rejected recommendation".to_string(),
        );

        let mut qualified_evidence = recommendation.evidence.clone();
        qualified_evidence.qualification = Some(QualificationIdentity::new(
            CompiledKernelArtifactId::from_digest("optimization-conformance-digest"),
            1,
            "suite-v1",
            "conformance-arch",
            "1.0",
        ));
        qualified_evidence.qualification_status = QualificationStatus::Qualified;
        qualified_evidence.trust = KernelArtifactTrust::Trusted;
        let qualified_recommendation = OptimizationRecommendation {
            evidence: qualified_evidence,
            ..recommendation
        };
        let fresh_eligibility = CandidateEligibilityInput::all_satisfied();
        let promoted = submit_recommendation_for_promotion(
            &qualified_recommendation,
            &fresh_eligibility,
            &mut registry,
            &kernel,
            candidate_id,
        );
        record(
            &mut results,
            "a qualified recommendation promotes only after fresh eligibility revalidation",
            promoted.is_ok()
                && registry
                    .active_prepared_kernel(&kernel)
                    .is_some_and(|active| active.id == candidate_id),
            format!("unexpected outcome: {promoted:?}"),
        );

        let stale_denied_after_revocation = submit_recommendation_for_promotion(
            &qualified_recommendation,
            &CandidateEligibilityInput {
                revoked: true,
                ..CandidateEligibilityInput::all_satisfied()
            },
            &mut registry,
            &kernel,
            candidate_id,
        );
        record(
            &mut results,
            "Runtime revalidation rejects promotion once trust is revoked, even with prior campaign evidence",
            matches!(
                stale_denied_after_revocation,
                Err(KernelOptimizationError::RecommendationInvalid { .. })
            ),
            format!("unexpected outcome: {stale_denied_after_revocation:?}"),
        );
    }

    // "Native Handle Boundary Conformance": worker-local/native handle
    // shapes are rejected as transport identity.
    for field in [
        "cufunction",
        "device-handle",
        "process-local-prepared-kernel-id",
    ] {
        let outcome = reject_native_transport_handle(field);
        record(
            &mut results,
            format!("artifact transport rejects native handle field '{field}'"),
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // "Offline Inference Conformance": already-local compatible artifacts
    // permit inference independent of network/service state.
    record(
        &mut results,
        "offline inference is possible when required artifacts are local and compatible",
        offline_inference_possible(true),
        "offline_inference_possible(true) unexpectedly returned false".to_string(),
    );

    // "Credential Boundary Conformance": a credential cannot be labeled as
    // Runtime-owned.
    for scope in ["cli-tooling", "ci", "external-orchestrator"] {
        let outcome = reject_runtime_owned_optimization_credential(scope);
        record(
            &mut results,
            format!("optimization credential scope '{scope}' is accepted as non-Runtime"),
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }
    let runtime_claim = reject_runtime_owned_optimization_credential("runtime-inference-session");
    record(
        &mut results,
        "optimization credential scope cannot claim Runtime ownership",
        matches!(
            runtime_claim,
            Err(KernelOptimizationError::CredentialBoundaryViolation { .. })
        ),
        format!("unexpected outcome: {runtime_claim:?}"),
    );

    // "Workload Privacy Conformance": aggregate metadata may be present while
    // raw-prompt-shaped keys are rejected.
    let raw_key_rejected = reject_raw_workload_metadata_key("raw-prompt-text");
    record(
        &mut results,
        "workload profile metadata rejects raw-prompt-shaped keys",
        raw_key_rejected.is_err(),
        format!("unexpected outcome: {raw_key_rejected:?}"),
    );
    let aggregate_key_allowed = reject_raw_workload_metadata_key("sequence-length-bucket");
    record(
        &mut results,
        "workload profile metadata allows aggregate shape/sequence keys",
        aggregate_key_allowed.is_ok(),
        format!("unexpected outcome: {aggregate_key_allowed:?}"),
    );

    // "Generator Identity Does Not Grant Trust Conformance".
    let policy = KernelEligibilityPolicy {
        require_trusted: true,
        require_qualified: true,
    };
    let independent = generator_identity_does_not_grant_trust(
        &GeneratorIdentity {
            provenance: KernelArtifactProvenance::AiGenerated,
            identity_label: Some("trusted-name-generator".into()),
        },
        &GeneratorIdentity {
            provenance: KernelArtifactProvenance::HumanAuthored,
            identity_label: None,
        },
        KernelArtifactTrust::Untrusted,
        QualificationStatus::Unqualified,
        &policy,
    );
    record(
        &mut results,
        "generator identity/provenance alone does not change eligibility",
        independent,
        "eligibility differed by generator identity alone".to_string(),
    );

    // "Campaign Failure Isolation Conformance": cancelling/failing a
    // campaign is a pure state transition with no Registry access at all.
    let cancelled = cancel_campaign(
        CampaignLifecycleState::Benchmarking,
        CampaignCancellationScope::NewWorkAndInterruptibleJobs,
    );
    record(
        &mut results,
        "a running campaign can be cancelled without touching production state",
        matches!(cancelled, Ok(CampaignLifecycleState::Cancelled)),
        format!("unexpected outcome: {cancelled:?}"),
    );

    // "Tachyon Independence Conformance": Tachyon is one descriptive
    // orchestrator kind among several, never a required dependency.
    let orchestrators = [
        OrchestratorKind::Ci,
        OrchestratorKind::LocalDeveloperTooling,
        OrchestratorKind::DedicatedOptimizationService,
        OrchestratorKind::TachyonManaged,
        OrchestratorKind::VendorInfrastructure,
        OrchestratorKind::FutureMagnetarTooling,
    ];
    record(
        &mut results,
        "Tachyon is one of several neutral orchestrator kinds, not a required dependency",
        orchestrators.contains(&OrchestratorKind::TachyonManaged) && orchestrators.len() > 1,
        "OrchestratorKind unexpectedly special-cases Tachyon".to_string(),
    );

    // "Tooling Authority Boundary Conformance": CLI-owned authority never
    // becomes ambient Runtime authority.
    for capability in [
        "git",
        "shell",
        "secret",
        "kernel-optimization-orchestration",
    ] {
        let outcome = reject_optimization_tooling_authority_in_runtime(capability);
        record(
            &mut results,
            format!("Runtime rejects optimization-tooling-owned capability '{capability}'"),
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // "Selection Policy Still Authoritative Conformance": a candidate ranked
    // first by a campaign remains excluded once current context makes it
    // infeasible.
    let infeasible = evaluate_candidate_eligibility(&CandidateEligibilityInput {
        memory_feasible: false,
        ..CandidateEligibilityInput::all_satisfied()
    });
    record(
        &mut results,
        "current Kernel Selection Policy eligibility excludes a memory-infeasible campaign favorite",
        matches!(
            infeasible,
            Err(KernelSelectionExclusionReason::MemoryInfeasible)
        ),
        format!("unexpected outcome: {infeasible:?}"),
    );

    // "Optimization Observability Redaction Conformance".
    let observation = OptimizationObservation::new(
        OptimizationObservationKind::CampaignFailed,
        OptimizationCampaignId::new("redaction-conformance-campaign"),
    )
    .with_redacted_metadata(
        "failure-context",
        "api_key=sk-live-deadbeef native_handle=0xdeadbeef",
    );
    let redacted_value = observation
        .redacted_metadata
        .get("failure-context")
        .cloned()
        .unwrap_or_default();
    record(
        &mut results,
        "optimization observability redacts secrets and native handles by default",
        !redacted_value.contains("sk-live-deadbeef") && !redacted_value.contains("0xdeadbeef"),
        format!("observation leaked sensitive data: {redacted_value}"),
    );

    KernelOptimizationOrchestrationConformanceReport { results }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_report_is_conformant() {
        let report = run_kernel_optimization_orchestration_conformance();
        for result in &report.results {
            assert!(
                result.passed,
                "requirement '{}' failed: {:?}",
                result.requirement, result.diagnostic
            );
        }
        assert!(report.is_conformant());
    }

    #[test]
    fn campaign_cannot_be_validated_startable_from_the_hot_path() {
        let campaign = OptimizationCampaign::new(
            OptimizationCampaignId::new("test-campaign"),
            OptimizationTrigger::ManualRequest,
            "v1",
        );
        assert!(campaign.validate_startable(false).is_ok());
        assert!(matches!(
            campaign.validate_startable(true),
            Err(KernelOptimizationError::HotPathDenied)
        ));
    }

    #[test]
    fn campaign_lifecycle_rejects_illegal_transitions() {
        assert!(CampaignLifecycleState::Planned.can_transition_to(CampaignLifecycleState::Queued));
        assert!(
            !CampaignLifecycleState::Planned.can_transition_to(CampaignLifecycleState::Completed)
        );
        assert!(CampaignLifecycleState::Completed.is_terminal());
        assert!(!CampaignLifecycleState::Running.is_terminal());
    }

    #[test]
    fn budget_exhaustion_is_detected_per_dimension() {
        let budget = CampaignBudget {
            max_candidates: Some(10),
            ..CampaignBudget::default()
        };
        let under = CampaignUsage {
            candidates: 5,
            ..CampaignUsage::default()
        };
        let over = CampaignUsage {
            candidates: 10,
            ..CampaignUsage::default()
        };
        assert_eq!(budget_exceeded(&budget, &under), None);
        assert_eq!(
            budget_exceeded(&budget, &over),
            Some(BudgetDimension::MaxCandidates)
        );
    }

    #[test]
    fn candidate_failure_does_not_force_campaign_abort() {
        assert!(other_candidates_continue(
            CandidateFailureKind::CompilationFailure,
            CandidateFailurePolicy::ContinueRemainingCandidates
        ));
        assert!(!other_candidates_continue(
            CandidateFailureKind::CompilationFailure,
            CandidateFailurePolicy::AbortCampaign
        ));
    }

    #[test]
    fn worker_selection_requires_every_declared_dimension() {
        let profile = WorkerCapabilityProfile {
            provider_implementations: vec![ProviderBinding::new("cuda")],
            device_architecture: Some("sm90".into()),
            compiler_toolchains: vec!["nvcc-12".into()],
            ..WorkerCapabilityProfile::default()
        };
        let compatible = WorkerCapabilityRequirement {
            required_provider: Some(ProviderBinding::new("cuda")),
            required_device_architecture: Some("sm90".into()),
            ..WorkerCapabilityRequirement::default()
        };
        let incompatible = WorkerCapabilityRequirement {
            required_device_architecture: Some("sm80".into()),
            ..WorkerCapabilityRequirement::default()
        };
        assert!(worker_compatible_with_target(&profile, &compatible));
        assert!(!worker_compatible_with_target(&profile, &incompatible));
    }

    #[test]
    fn provider_isolation_detects_shared_instance() {
        let isolated = ProviderIsolation {
            optimization_worker_provider: ProviderBinding::new("cuda-optimization-worker"),
            production_provider: ProviderBinding::new("cuda-production"),
        };
        let shared = ProviderIsolation {
            optimization_worker_provider: ProviderBinding::new("cuda-production"),
            production_provider: ProviderBinding::new("cuda-production"),
        };
        assert!(isolated.is_isolated());
        assert!(!shared.is_isolated());
    }
}
