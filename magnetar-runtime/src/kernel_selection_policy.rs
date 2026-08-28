//! Kernel Optimization And Selection Policy (see
//! `openspec/changes/define-kernel-optimization-and-selection-policy`).
//!
//! Magnetar can already compile, qualify, cache, prepare, promote, revoke and
//! hot-swap multiple Kernel implementations of the same portable Operator
//! ([`crate::kernel_artifact`], [`crate::kernel_compilation`],
//! [`crate::kernel_qualification`], [`crate::kernel_registry`]). This module
//! defines the Runtime policy that decides, among many valid implementations,
//! *which eligible Kernel executes an Operator now* -- as executable Rust
//! types and pure functions, mirroring the shape of
//! [`crate::kernel_qualification`].
//!
//! The Core Rule from the proposal is mechanically enforced throughout:
//! selection happens in two logical phases, eligibility filtering then
//! optimization ranking, and optimization SHALL never make an ineligible
//! candidate eligible. This is not merely documented -- it is structural.
//! [`EligibleCandidate`] can only be constructed through
//! [`EligibleCandidate::from_checked`], which requires
//! [`evaluate_candidate_eligibility`] to succeed first. Every ranking
//! function in this module ([`rank_candidates`], [`weighted_score`],
//! [`lexicographic_compare`]) takes `&EligibleCandidate`/`&[EligibleCandidate]`,
//! never a raw candidate pool -- there is no code path from "fast" to
//! "selected" that skips eligibility.
//!
//! - [`OptimizationProfile`] / [`ObjectiveDimension`]: named optimization
//!   profiles and the portable objective vocabulary, implementing
//!   "Optimization Profiles" and "Optimization Objective".
//! - [`CandidateEligibilityInput`] / [`evaluate_candidate_eligibility`] /
//!   [`KernelSelectionExclusionReason`]: the hard-constraint eligibility gate
//!   and its structured exclusion reasons, implementing "Eligibility Before
//!   Optimization" and "Candidate Exclusion Reasons". A `From` impl maps
//!   [`crate::kernel_registry::KernelCandidateRejection`] onto this
//!   vocabulary so Registry-level and policy-level exclusions stay aligned.
//! - [`CandidateIdentity`] / [`CandidateMetrics`] / [`EligibleCandidate`]:
//!   the only representation of a candidate ranking functions accept.
//! - [`RankingStrategy`] / [`LexicographicPolicy`] / [`WeightedScorePolicy`] /
//!   [`weighted_score`] / [`lexicographic_compare`] / [`rank_candidates`]:
//!   the portable ranking strategies, implementing "Ranking Strategies",
//!   "Weighted Ranking", "Lexicographic Ranking" and "Deterministic
//!   Tie-Breaking" -- ties always fall back to [`CandidateIdentity`]'s
//!   derived `Ord`, never hash-map iteration order.
//! - [`MissingMetricPolicy`] / [`missing_metric_value`]: explicit handling of
//!   missing performance evidence, implementing "Missing Performance
//!   Evidence": a missing benchmark is never silently treated as the best
//!   benchmark.
//! - [`BenchmarkContext`] / [`benchmark_context_compatible`] /
//!   [`StaleBenchmarkPolicy`] / [`evaluate_stale_benchmark_policy`]:
//!   "Benchmark Context" and "Benchmark Freshness" -- evidence is
//!   exact-match compatible or it does not apply; freshness reuses
//!   [`crate::kernel_benchmark::BenchmarkFreshness`].
//! - [`performance_evidence_applies_to_workload`]: "Shape-Aware Ranking".
//! - [`GenerationPhase`] / [`WorkloadContext`] / [`rank_by_generation_phase`]:
//!   "Batch-Aware Ranking" and "Prefill Versus Decode" -- prefill and decode
//!   candidates are ranked independently, so nothing forces them to the same
//!   winning Kernel.
//! - [`PressureSnapshot`] / [`pressure_ranking_bias`]: "Pressure-Aware
//!   Ranking" -- pressure only ever biases an already-eligible candidate's
//!   rank, implementing "Pressure SHALL NOT bypass semantic compatibility".
//! - [`memory_feasible_before_ranking`]: "Memory Feasibility" -- Memory
//!   Manager stays authoritative.
//! - [`ConversionCost`] / [`total_execution_cost_ms`]: "Conversion Cost".
//! - [`PreparationCostClass`] / [`preparation_cost_applies`] /
//!   [`compilation_cost_excluded_from_hot_path`]: "Preparation Cost" and
//!   "Compilation Cost".
//! - [`HysteresisPolicy`] / [`evaluate_hysteresis`] /
//!   [`AntiFlappingPolicy`] / [`promotion_allowed_by_anti_flapping`] /
//!   [`RollingMeasurementWindow`]: "Selection Hysteresis" and
//!   "Anti-Flapping" -- the rolling window only reports [`RollingMeasurementWindow::is_stable`]
//!   once it holds a full window of samples.
//! - [`PromotionRecommendation`] / [`recommend_promotion`] /
//!   [`apply_promotion_recommendation`]: "Selection Versus Promotion" --
//!   ranking first is never itself promotion, and only an approved
//!   recommendation ever reaches
//!   [`crate::kernel_registry::KernelRegistry::promote_generation`].
//! - "Static Selection" and "Dynamic Selection" reuse
//!   [`crate::model_instance::KernelSelectionPolicy`] (Dynamic/Pinned);
//!   [`static_selection_required_during_warmup`] is the load-time hook.
//! - [`SelectionCacheKey`] / [`SelectionCache`] /
//!   [`SelectionCacheInvalidationTrigger`]: "Selection Caching".
//! - [`KernelSelectionPolicyId`] / [`KernelOptimizationPolicy`]: "Model
//!   Instance Kernel Policy" -- a `KernelSelectionPolicyId` is what a Model
//!   Instance attaches (see [`crate::model_instance::ModelInstanceDefinition`]),
//!   mirroring how a Model Instance references a `TokenizerId` rather than
//!   embedding a whole Tokenizer.
//! - [`ModelComponentKernelRequest`] / [`validate_model_component_request`]:
//!   "Model Component Boundary".
//! - [`resolve_session_preference`] / [`resolve_generation_preference`] /
//!   [`CliPreference`] / [`map_cli_preference`]: "Session Boundary",
//!   "Generation Request Boundary", "CLI Boundary" -- preferences are
//!   resolved only against an already-eligible candidate set.
//! - [`resolve_cross_provider_selection`] / [`ProviderPrivateVariant`] /
//!   [`provider_may_select_variant_privately`]: "Provider Boundary" and
//!   "Provider-Local Variant Selection".
//! - [`FallbackPolicy`] / [`HostStagingPolicy`] /
//!   [`evaluate_kernel_selection_fallback`] /
//!   [`evaluate_kernel_selection_fallback_chain`]: "Fallback Policy" and
//!   "Reference CPU Fallback".
//! - [`CrossProviderMovement`] / [`validate_cross_provider_movement`]: "No
//!   Hidden Cross-Provider Movement".
//! - [`resolve_pinned_selection`] (over
//!   [`crate::model_instance::PinnedKernelSelection`]): "Reproducible
//!   Profile" / pinned selection.
//! - [`ExplorationPolicy`] / [`exploration_allowed`] /
//!   [`eligible_for_exploration`] / [`CanaryPolicy`] /
//!   [`canary_budget_exhausted`] / [`ExplorationFailureAction`] /
//!   [`exploration_failure_affects_unrelated_candidate`] /
//!   [`apply_exploration_failure_action`]: "Exploration", "Canary
//!   Selection", "Exploration Failure" -- only `TriggerRollback` ever reaches
//!   [`crate::kernel_registry::KernelRegistry::rollback_generation`].
//! - [`OnlineMeasurement`] / [`online_measurement_cannot_override_correctness_or_trust`]:
//!   "Selection Metrics" and "Online Measurement".
//! - [`PolicyPrecedenceLevel`] / [`PolicyConstraintStack`] /
//!   [`resolve_effective_profile`]: "Policy Precedence".
//! - [`SelectionExplanation`]: "Selection Explainability".
//! - [`KernelSelectionError`]: the structured error categories from the
//!   proposal's "Selection Errors" section.
//! - [`KernelSelectionObservationKind`] / [`KernelSelectionObservation`]:
//!   redacted selection lifecycle observability, implementing
//!   "Observability".
//! - [`KernelSelectionPolicyConformanceReport`] /
//!   [`run_kernel_selection_policy_conformance`]: the conformance checks from
//!   this change's `specs/kernel-selection-policy/spec.md` and the
//!   selection-policy requirements added to `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::kernel_artifact::{KernelArtifactTrust, PreparedKernelId};
use crate::kernel_benchmark::BenchmarkFreshness;
use crate::kernel_qualification::QualificationStatus;
use crate::kernel_registry::{KernelCandidateRejection, KernelRegistry};
use crate::model_instance::{
    KernelSelectionPolicy as ModelInstanceSelectionMode, PinnedKernelSelection,
};
use crate::{
    ComputeDType, HostStagingPolicy, KernelExecutionMode, KernelFallbackClass, KernelId,
    OperatorId, ProviderBinding, TensorLayoutKind,
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_SELECTION_POLICY_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Optimization Objective / Profiles
// ---------------------------------------------------------------------

/// The portable optimization objective vocabulary, implementing
/// "Optimization Objective" (proposal). All [`CandidateMetrics`] values use a
/// lower-is-better convention; a caller supplying a "higher is better" raw
/// measurement (e.g. throughput) SHALL invert or negate it before recording
/// it here so every ranking function can treat every dimension uniformly.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObjectiveDimension {
    Latency,
    Throughput,
    TailLatency,
    Memory,
    Workspace,
    Energy,
    Determinism,
    StartupCost,
    PreparationCost,
    BatchEfficiency,
    SequenceEfficiency,
}

/// A named optimization profile, implementing "Optimization Profiles"
/// (proposal): "A profile is policy, not Provider identity."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OptimizationProfile {
    Balanced,
    Latency,
    Throughput,
    Memory,
    Deterministic,
    Energy,
    Reproducible,
}

// ---------------------------------------------------------------------
// Candidate Exclusion Reasons
// ---------------------------------------------------------------------

/// Structured exclusion reasons, implementing "Candidate Exclusion Reasons"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelSelectionExclusionReason {
    SemanticIncompatible,
    OperatorVersionIncompatible,
    QualificationRequired,
    QualificationExpired,
    QualificationRevoked,
    TrustDenied,
    DTypeIncompatible,
    LayoutIncompatible,
    ShapeIncompatible,
    PrecisionIncompatible,
    DeterminismIncompatible,
    ResourceAffinityIncompatible,
    ProviderUnready,
    DeviceUnavailable,
    DeviceUnhealthy,
    MemoryInfeasible,
    WorkspaceInfeasible,
    RequiredFeatureMissing,
    PreparedKernelUnavailable,
    BenchmarkIncompatible,
    PolicyDenied,
}

/// Implements "Registry Does Not Make Cross-Provider Optimization Decision"
/// (`kernel-registry` spec) staying aligned with this policy's own
/// vocabulary: a [`KernelCandidateRejection`] from Registry-level candidate
/// discovery maps onto exactly one policy-level exclusion reason.
impl From<KernelCandidateRejection> for KernelSelectionExclusionReason {
    fn from(reason: KernelCandidateRejection) -> Self {
        use KernelCandidateRejection as R;
        match reason {
            R::OperatorMismatch => Self::SemanticIncompatible,
            R::OperatorVersionUnsupported => Self::OperatorVersionIncompatible,
            R::DTypeUnsupported => Self::DTypeIncompatible,
            R::LayoutUnsupported => Self::LayoutIncompatible,
            R::ShapeUnsupported => Self::ShapeIncompatible,
            R::MemoryClassUnsupported => Self::MemoryInfeasible,
            R::ExecutionModeUnsupported => Self::PolicyDenied,
            R::ResourceAffinityConflict => Self::ResourceAffinityIncompatible,
            R::ProviderUnavailable | R::ProviderNotReady | R::ProviderSaturated => {
                Self::ProviderUnready
            }
            R::DeviceUnavailable => Self::DeviceUnavailable,
            R::DeviceIncompatible => Self::DeviceUnhealthy,
            R::ProviderFeatureMissing | R::DeviceFeatureMissing => Self::RequiredFeatureMissing,
            R::WorkspaceUnavailable => Self::WorkspaceInfeasible,
            R::BatchingUnsupported
            | R::AdapterUnsupported
            | R::KvCacheUnsupported
            | R::PrefixCacheUnsupported => Self::PolicyDenied,
            R::ConformanceMissing => Self::QualificationRequired,
            R::ConformanceFailed => Self::QualificationRevoked,
            R::PolicyDenied => Self::PolicyDenied,
            R::StaleRegistryEntry => Self::PolicyDenied,
            R::Revoked => Self::QualificationRevoked,
        }
    }
}

// ---------------------------------------------------------------------
// Eligibility Pipeline
// ---------------------------------------------------------------------

/// Every hard constraint from the proposal's "Eligibility Before
/// Optimization" list, as one input record. [`evaluate_candidate_eligibility`]
/// is the only function that turns this into a decision -- there is no
/// shortcut from "has a great benchmark" to "eligible".
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEligibilityInput {
    pub semantic_compatible: bool,
    pub operator_version_compatible: bool,
    pub qualification_status: QualificationStatus,
    pub require_qualified: bool,
    pub trust: KernelArtifactTrust,
    pub require_trusted: bool,
    pub revoked: bool,
    pub dtype_compatible: bool,
    pub layout_compatible: bool,
    pub shape_compatible: bool,
    pub precision_compatible: bool,
    pub determinism_compatible: bool,
    pub resource_affinity_compatible: bool,
    pub provider_ready: bool,
    pub device_available: bool,
    pub device_healthy: bool,
    pub memory_feasible: bool,
    pub workspace_feasible: bool,
    pub required_features_satisfied: bool,
    pub prepared_kernel_ready: bool,
    pub execution_mode_compatible: bool,
    pub benchmark_context_compatible: bool,
}

impl CandidateEligibilityInput {
    /// A fully-satisfied baseline, so a test or caller only has to flip the
    /// one field it cares about instead of restating every dimension.
    pub fn all_satisfied() -> Self {
        Self {
            semantic_compatible: true,
            operator_version_compatible: true,
            qualification_status: QualificationStatus::Qualified,
            require_qualified: true,
            trust: KernelArtifactTrust::Trusted,
            require_trusted: true,
            revoked: false,
            dtype_compatible: true,
            layout_compatible: true,
            shape_compatible: true,
            precision_compatible: true,
            determinism_compatible: true,
            resource_affinity_compatible: true,
            provider_ready: true,
            device_available: true,
            device_healthy: true,
            memory_feasible: true,
            workspace_feasible: true,
            required_features_satisfied: true,
            prepared_kernel_ready: true,
            execution_mode_compatible: true,
            benchmark_context_compatible: true,
        }
    }
}

/// Implements "Eligibility Before Optimization" (proposal): "A Kernel
/// candidate SHALL be excluded before performance ranking when it fails any
/// required hard constraint." Order is fixed and deterministic; the first
/// failing constraint decides the reason.
pub fn evaluate_candidate_eligibility(
    input: &CandidateEligibilityInput,
) -> Result<(), KernelSelectionExclusionReason> {
    use KernelSelectionExclusionReason as E;
    if input.revoked {
        return Err(E::QualificationRevoked);
    }
    if !input.semantic_compatible {
        return Err(E::SemanticIncompatible);
    }
    if !input.operator_version_compatible {
        return Err(E::OperatorVersionIncompatible);
    }
    if input.require_qualified && !input.qualification_status.is_eligible() {
        return Err(match input.qualification_status {
            QualificationStatus::Revoked => E::QualificationRevoked,
            QualificationStatus::Expired => E::QualificationExpired,
            _ => E::QualificationRequired,
        });
    }
    if input.require_trusted && !input.trust.is_trusted() {
        return Err(E::TrustDenied);
    }
    if !input.dtype_compatible {
        return Err(E::DTypeIncompatible);
    }
    if !input.layout_compatible {
        return Err(E::LayoutIncompatible);
    }
    if !input.shape_compatible {
        return Err(E::ShapeIncompatible);
    }
    if !input.precision_compatible {
        return Err(E::PrecisionIncompatible);
    }
    if !input.determinism_compatible {
        return Err(E::DeterminismIncompatible);
    }
    if !input.resource_affinity_compatible {
        return Err(E::ResourceAffinityIncompatible);
    }
    if !input.provider_ready {
        return Err(E::ProviderUnready);
    }
    if !input.device_available {
        return Err(E::DeviceUnavailable);
    }
    if !input.device_healthy {
        return Err(E::DeviceUnhealthy);
    }
    if !input.memory_feasible {
        return Err(E::MemoryInfeasible);
    }
    if !input.workspace_feasible {
        return Err(E::WorkspaceInfeasible);
    }
    if !input.required_features_satisfied {
        return Err(E::RequiredFeatureMissing);
    }
    if !input.prepared_kernel_ready {
        return Err(E::PreparedKernelUnavailable);
    }
    if !input.execution_mode_compatible {
        return Err(E::PolicyDenied);
    }
    if !input.benchmark_context_compatible {
        return Err(E::BenchmarkIncompatible);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Candidate Identity / Metrics
// ---------------------------------------------------------------------

/// Stable candidate identity, implementing "Registry Supports Stable
/// Candidate Identity" and "Deterministic Tie-Breaking" (proposal): its
/// derived `Ord` (Kernel, then Provider, then artifact digest) is the tie
/// break every ranking function falls back to, never hash-map iteration
/// order, pointer values, or discovery order.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CandidateIdentity {
    pub kernel: KernelId,
    pub provider: ProviderBinding,
    pub artifact_digest: Option<String>,
}

/// Normalized, already-comparable metric values for one candidate, keyed by
/// [`ObjectiveDimension`]. Every value follows the lower-is-better
/// convention documented on [`ObjectiveDimension`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandidateMetrics {
    pub values: BTreeMap<ObjectiveDimension, f64>,
}

impl CandidateMetrics {
    pub fn with(mut self, dimension: ObjectiveDimension, value: f64) -> Self {
        self.values.insert(dimension, value);
        self
    }
}

/// The only representation of a candidate that ranking functions in this
/// module accept, implementing the Core Rule: "Optimization SHALL never make
/// an ineligible candidate eligible." There is no public constructor other
/// than [`Self::from_checked`], so an ineligible candidate structurally
/// cannot reach [`rank_candidates`].
#[derive(Clone, Debug, PartialEq)]
pub struct EligibleCandidate {
    pub identity: CandidateIdentity,
    pub metrics: CandidateMetrics,
}

impl EligibleCandidate {
    pub fn from_checked(
        identity: CandidateIdentity,
        metrics: CandidateMetrics,
        eligibility: &CandidateEligibilityInput,
    ) -> Result<Self, KernelSelectionExclusionReason> {
        evaluate_candidate_eligibility(eligibility)?;
        Ok(Self { identity, metrics })
    }
}

// ---------------------------------------------------------------------
// Missing Performance Evidence
// ---------------------------------------------------------------------

/// Implements "Missing Performance Evidence" (proposal): "The behavior SHALL
/// be explicit."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingMetricPolicy {
    Exclude,
    RankConservatively,
    UseFallbackMetadata,
    RetainActiveKernel,
}

/// Implements "Runtime SHALL NOT interpret missing benchmark as best
/// benchmark" (proposal): a missing metric always resolves to a
/// worse-than-any-real-evidence value (or an explicit fallback), never to
/// zero / best.
pub fn missing_metric_value(policy: MissingMetricPolicy, fallback: Option<f64>) -> f64 {
    match policy {
        MissingMetricPolicy::Exclude => f64::INFINITY,
        MissingMetricPolicy::RankConservatively => f64::MAX / 2.0,
        MissingMetricPolicy::UseFallbackMetadata => fallback.unwrap_or(f64::INFINITY),
        MissingMetricPolicy::RetainActiveKernel => f64::INFINITY,
    }
}

/// Implements "Handle Missing Metric" (tasks): whether the currently active
/// Kernel SHOULD be retained outright rather than ranked against candidates
/// missing required evidence, when policy says so.
pub fn should_retain_active_due_to_missing_metric(
    metrics: &CandidateMetrics,
    required: &[ObjectiveDimension],
    policy: MissingMetricPolicy,
) -> bool {
    policy == MissingMetricPolicy::RetainActiveKernel
        && required.iter().any(|dim| !metrics.values.contains_key(dim))
}

fn metric_value(
    metrics: &CandidateMetrics,
    dimension: ObjectiveDimension,
    policy: MissingMetricPolicy,
) -> f64 {
    metrics
        .values
        .get(&dimension)
        .copied()
        .unwrap_or_else(|| missing_metric_value(policy, None))
}

// ---------------------------------------------------------------------
// Ranking Strategies
// ---------------------------------------------------------------------

/// Implements "Weighted Ranking" (proposal): "Weights SHALL belong to
/// Runtime policy. Providers SHALL NOT silently redefine them."
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedScorePolicy {
    pub weights: BTreeMap<ObjectiveDimension, f64>,
    pub missing_metric_policy: MissingMetricPolicy,
}

/// Implements "Comparable Metrics" (proposal): only metrics with a declared
/// weight participate, and a missing metric is resolved through
/// [`missing_metric_value`], never through an implicit zero/best default.
pub fn weighted_score(metrics: &CandidateMetrics, policy: &WeightedScorePolicy) -> f64 {
    policy
        .weights
        .iter()
        .map(|(dimension, weight)| {
            weight * metric_value(metrics, *dimension, policy.missing_metric_policy)
        })
        .sum()
}

/// Implements "Lexicographic Ranking" (proposal): "The first metric that
/// distinguishes candidates decides the ranking."
#[derive(Clone, Debug, PartialEq)]
pub struct LexicographicPolicy {
    pub order: Vec<ObjectiveDimension>,
    pub missing_metric_policy: MissingMetricPolicy,
}

pub fn lexicographic_compare(
    a: &CandidateMetrics,
    b: &CandidateMetrics,
    policy: &LexicographicPolicy,
) -> Ordering {
    for dimension in &policy.order {
        let a_value = metric_value(a, *dimension, policy.missing_metric_policy);
        let b_value = metric_value(b, *dimension, policy.missing_metric_policy);
        match a_value.partial_cmp(&b_value) {
            Some(Ordering::Equal) | None => continue,
            Some(order) => return order,
        }
    }
    Ordering::Equal
}

/// The portable ranking strategies, implementing "Ranking Strategies"
/// (proposal).
#[derive(Clone, Debug, PartialEq)]
pub enum RankingStrategy {
    Lexicographic(LexicographicPolicy),
    WeightedScore(WeightedScorePolicy),
    /// Implements "PolicyOrdered" (proposal): an explicit, Runtime-declared
    /// preference order. Candidates absent from `order` sort after every
    /// listed candidate, but are never excluded outright.
    PolicyOrdered(Vec<KernelId>),
    /// Implements "Pinned" (proposal): reproducible selection resolves to
    /// exactly one Kernel, or fails explicitly -- see
    /// [`resolve_pinned_selection`] for the richer unavailable-vs-ineligible
    /// distinction.
    Pinned(KernelId),
}

/// Ranks `candidates` best-first. Every branch ends in a fully deterministic
/// order: primary ranking score, then [`CandidateIdentity`]'s derived `Ord`
/// as the stable tie-break, implementing "Deterministic Tie-Breaking"
/// (proposal) -- calling this twice on the same input always yields the same
/// output, independent of the input `Vec`'s original order.
pub fn rank_candidates(
    candidates: &[EligibleCandidate],
    strategy: &RankingStrategy,
) -> Result<Vec<CandidateIdentity>, KernelSelectionError> {
    match strategy {
        RankingStrategy::Pinned(kernel) => candidates
            .iter()
            .find(|candidate| &candidate.identity.kernel == kernel)
            .map(|candidate| vec![candidate.identity.clone()])
            .ok_or(KernelSelectionError::PinnedKernelUnavailable),
        RankingStrategy::PolicyOrdered(order) => {
            let mut ranked: Vec<CandidateIdentity> = candidates
                .iter()
                .map(|candidate| candidate.identity.clone())
                .collect();
            ranked.sort_by(|a, b| {
                // `unwrap_or(usize::MAX)`, not a bare `Option<usize>`
                // comparison: `None < Some(_)` would otherwise rank an
                // unlisted candidate *ahead* of every listed one.
                let a_rank = order
                    .iter()
                    .position(|kernel| kernel == &a.kernel)
                    .unwrap_or(usize::MAX);
                let b_rank = order
                    .iter()
                    .position(|kernel| kernel == &b.kernel)
                    .unwrap_or(usize::MAX);
                a_rank.cmp(&b_rank).then_with(|| a.cmp(b))
            });
            Ok(ranked)
        }
        RankingStrategy::WeightedScore(policy) => {
            let mut scored: Vec<(f64, CandidateIdentity)> = candidates
                .iter()
                .map(|candidate| {
                    (
                        weighted_score(&candidate.metrics, policy),
                        candidate.identity.clone(),
                    )
                })
                .collect();
            scored.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            Ok(scored.into_iter().map(|(_, identity)| identity).collect())
        }
        RankingStrategy::Lexicographic(policy) => {
            let mut ranked = candidates.to_vec();
            ranked.sort_by(|a, b| {
                lexicographic_compare(&a.metrics, &b.metrics, policy)
                    .then_with(|| a.identity.cmp(&b.identity))
            });
            Ok(ranked
                .into_iter()
                .map(|candidate| candidate.identity)
                .collect())
        }
    }
}

// ---------------------------------------------------------------------
// Benchmark Context / Freshness
// ---------------------------------------------------------------------

/// Implements "Benchmark Context" (proposal): the compatibility-relevant
/// context performance evidence was measured under.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkContext {
    pub provider: ProviderBinding,
    pub device_architecture: String,
    pub driver_runtime_compatibility: String,
    pub operator_version: u32,
    pub artifact_digest: Option<String>,
    pub dtype: ComputeDType,
    pub layout: TensorLayoutKind,
    pub shape_bucket: String,
    pub batch_bucket: String,
    pub sequence_bucket: String,
    pub execution_mode: KernelExecutionMode,
    pub benchmark_profile_version: u32,
}

/// Implements "Performance evidence SHALL be evaluated only when compatible
/// with the current execution context" (proposal): exact match only, no
/// fuzzy or partial compatibility, mirroring
/// [`crate::kernel_qualification::QualificationIdentity::applies_to`].
pub fn benchmark_context_compatible(
    evidence: &BenchmarkContext,
    current: &BenchmarkContext,
) -> bool {
    evidence == current
}

/// Implements "Benchmark Freshness" (proposal): "Stale benchmark evidence
/// SHALL be identifiable." Reuses
/// [`crate::kernel_benchmark::BenchmarkFreshness`] -- the staleness state a
/// benchmark result already carries -- rather than introducing a second,
/// competing freshness representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaleBenchmarkPolicy {
    Accept,
    Discount,
    Exclude,
    RequestRebenchmarkOutsideHotPath,
}

/// Implements "It SHALL NOT silently treat incompatible evidence as current"
/// (proposal): stale evidence is either explicitly tolerated
/// (`Accept`/`Discount`/`RequestRebenchmarkOutsideHotPath`) or explicitly
/// rejected (`Exclude`) -- never silently ignored.
pub fn evaluate_stale_benchmark_policy(
    freshness: BenchmarkFreshness,
    policy: StaleBenchmarkPolicy,
) -> Result<(), KernelSelectionError> {
    if matches!(freshness, BenchmarkFreshness::Fresh) {
        return Ok(());
    }
    match policy {
        StaleBenchmarkPolicy::Accept
        | StaleBenchmarkPolicy::Discount
        | StaleBenchmarkPolicy::RequestRebenchmarkOutsideHotPath => Ok(()),
        StaleBenchmarkPolicy::Exclude => Err(KernelSelectionError::BenchmarkStale),
    }
}

// ---------------------------------------------------------------------
// Shape / Batch / Prefill-Decode / Pressure / Memory-Aware Ranking
// ---------------------------------------------------------------------

/// Implements "Shape-Aware Ranking" (proposal): "A Kernel optimized for
/// batch=1, sequence=128 SHALL NOT automatically be assumed optimal for
/// batch=64, sequence=8192." Evidence indexed under one workload envelope
/// bucket applies only to that exact bucket.
pub fn performance_evidence_applies_to_workload(
    evidence_shape_bucket: &str,
    requested_shape_bucket: &str,
) -> bool {
    evidence_shape_bucket == requested_shape_bucket
}

/// Implements "Prefill Versus Decode" (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GenerationPhase {
    Prefill,
    Decode,
}

/// Implements "Batch-Aware Ranking" (proposal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkloadContext {
    pub active_sequences: u32,
    pub batch_width: u32,
    pub total_active_tokens: u64,
    pub raggedness: Option<f64>,
    pub phase: Option<GenerationPhase>,
    pub kv_cache_mode: Option<String>,
}

/// Per-phase ranking result, implementing "Prefill Versus Decode" (proposal):
/// "Runtime MAY select different Kernels for prefill and decode. A Model
/// Instance SHALL NOT assume the same Kernel implementation is optimal for
/// both phases." Ranks `prefill_candidates` and `decode_candidates`
/// independently through the same strategy -- nothing forces the two
/// resulting choices to agree, and nothing prevents them from doing so
/// either.
pub fn rank_by_generation_phase(
    prefill_candidates: &[EligibleCandidate],
    decode_candidates: &[EligibleCandidate],
    strategy: &RankingStrategy,
) -> Result<BTreeMap<GenerationPhase, Vec<CandidateIdentity>>, KernelSelectionError> {
    let mut by_phase = BTreeMap::new();
    by_phase.insert(
        GenerationPhase::Prefill,
        rank_candidates(prefill_candidates, strategy)?,
    );
    by_phase.insert(
        GenerationPhase::Decode,
        rank_candidates(decode_candidates, strategy)?,
    );
    Ok(by_phase)
}

/// Implements "Pressure-Aware Ranking" (proposal).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PressureLevel {
    #[default]
    Nominal,
    Elevated,
    Saturated,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PressureSnapshot {
    pub queue_depth: u32,
    pub device_utilization_percent: u8,
    pub memory_pressure: PressureLevel,
    pub workspace_pressure: PressureLevel,
    pub execution_backlog: u32,
    pub provider_admission_open: bool,
}

/// Implements "Pressure SHALL NOT bypass semantic compatibility" (proposal):
/// the only input is an [`EligibleCandidate`], so this bias can never be
/// computed for -- let alone select -- a candidate that failed eligibility.
/// The returned value is a bias to be added into a lower-is-better score; it
/// is never itself a selection decision.
pub fn pressure_ranking_bias(_eligible: &EligibleCandidate, pressure: &PressureSnapshot) -> f64 {
    let mut bias = f64::from(pressure.queue_depth) * 0.01;
    bias += f64::from(pressure.device_utilization_percent) * 0.01;
    if pressure.memory_pressure == PressureLevel::Saturated {
        bias += 10.0;
    }
    if pressure.workspace_pressure == PressureLevel::Saturated {
        bias += 10.0;
    }
    bias += f64::from(pressure.execution_backlog) * 0.01;
    if !pressure.provider_admission_open {
        bias += 100.0;
    }
    bias
}

/// Implements "Memory Feasibility" (proposal): "Memory Manager SHALL remain
/// authoritative for memory feasibility."
pub fn memory_feasible_before_ranking(
    memory_manager_feasible: bool,
) -> Result<(), KernelSelectionExclusionReason> {
    if memory_manager_feasible {
        Ok(())
    } else {
        Err(KernelSelectionExclusionReason::MemoryInfeasible)
    }
}

// ---------------------------------------------------------------------
// Conversion / Preparation / Compilation Cost
// ---------------------------------------------------------------------

/// Implements "Conversion Cost" (proposal): explicit, Runtime-visible
/// conversion/movement cost only -- there is no field for an inferred or
/// hidden conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConversionCost {
    pub dtype_conversion_ms: f64,
    pub layout_conversion_ms: f64,
    pub data_movement_ms: f64,
}

impl ConversionCost {
    pub fn total_ms(&self) -> f64 {
        self.dtype_conversion_ms + self.layout_conversion_ms + self.data_movement_ms
    }
}

pub fn total_execution_cost_ms(execution_ms: f64, conversion: &ConversionCost) -> f64 {
    execution_ms + conversion.total_ms()
}

/// Implements "Preparation Cost" (proposal): "Runtime SHALL distinguish
/// one-time cost, per-model-instance cost, per-batch cost, per-operation
/// cost."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PreparationCostClass {
    OneTime,
    PerModelInstance,
    PerBatch,
    PerOperation,
}

/// Implements "Once a Kernel is already prepared, historical preparation
/// cost SHOULD NOT be blindly added to every execution decision" (proposal).
pub fn preparation_cost_applies(class: PreparationCostClass, already_prepared: bool) -> bool {
    match class {
        PreparationCostClass::OneTime | PreparationCostClass::PerModelInstance => !already_prepared,
        PreparationCostClass::PerBatch | PreparationCostClass::PerOperation => true,
    }
}

/// Implements "Compilation Cost" (proposal): "Compilation cost SHALL NOT be
/// charged as a hot-path execution metric after a compatible compiled
/// artifact is already cached/prepared."
pub fn compilation_cost_excluded_from_hot_path(artifact_already_cached: bool) -> bool {
    artifact_already_cached
}

// ---------------------------------------------------------------------
// Hysteresis / Anti-Flapping / Selection Versus Promotion
// ---------------------------------------------------------------------

/// Implements "Selection Hysteresis" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HysteresisPolicy {
    pub promotion_threshold_fraction: f64,
}

impl Default for HysteresisPolicy {
    fn default() -> Self {
        Self {
            promotion_threshold_fraction: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionOutcome {
    RetainActive,
    PromoteCandidate,
}

/// Implements "A candidate SHOULD replace the current active Kernel only if
/// its expected benefit exceeds a policy threshold" (proposal). Scores use
/// the lower-is-better convention, so a lower `candidate_score` is an
/// improvement.
pub fn evaluate_hysteresis(
    active_score: f64,
    candidate_score: f64,
    policy: &HysteresisPolicy,
) -> SelectionOutcome {
    if candidate_score >= active_score {
        return SelectionOutcome::RetainActive;
    }
    let improvement = if active_score.abs() > f64::EPSILON {
        (active_score - candidate_score) / active_score.abs()
    } else {
        f64::INFINITY
    };
    if improvement > policy.promotion_threshold_fraction {
        SelectionOutcome::PromoteCandidate
    } else {
        SelectionOutcome::RetainActive
    }
}

/// Implements "Anti-Flapping" (proposal).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AntiFlappingPolicy {
    pub cooldown_seconds: u64,
    pub minimum_active_duration_seconds: u64,
}

pub fn promotion_allowed_by_anti_flapping(
    policy: &AntiFlappingPolicy,
    seconds_since_last_promotion: u64,
    seconds_active: u64,
) -> bool {
    seconds_since_last_promotion >= policy.cooldown_seconds
        && seconds_active >= policy.minimum_active_duration_seconds
}

/// Implements "Anti-Flapping" (proposal): "rolling measurements, stable
/// ranking window." A bounded FIFO of recent scores for one candidate, so a
/// single noisy measurement cannot flip a promotion decision on its own.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RollingMeasurementWindow {
    capacity: usize,
    samples: Vec<f64>,
}

impl RollingMeasurementWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            samples: Vec::new(),
        }
    }

    pub fn record(&mut self, score: f64) {
        if self.samples.len() >= self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(score);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Implements "stable ranking window" (proposal): the window is only
    /// considered a stable basis for a decision once it holds a full
    /// `capacity` worth of samples -- a single early reading cannot drive
    /// promotion/retention on its own.
    pub fn is_stable(&self) -> bool {
        self.samples.len() >= self.capacity
    }

    /// The mean of the current samples (lower-is-better convention, same as
    /// [`CandidateMetrics`]), or `None` while the window is empty.
    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            None
        } else {
            Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
        }
    }
}

/// Implements "Selection Versus Promotion" (proposal): "A candidate MAY rank
/// first without immediate promotion."
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionRecommendation {
    pub candidate: CandidateIdentity,
    pub approved: bool,
    pub reason: Option<String>,
}

pub fn recommend_promotion(
    top_ranked: &CandidateIdentity,
    hysteresis: SelectionOutcome,
    anti_flapping_allows: bool,
) -> PromotionRecommendation {
    let approved = matches!(hysteresis, SelectionOutcome::PromoteCandidate) && anti_flapping_allows;
    let reason = (!approved).then(|| {
        if !anti_flapping_allows {
            "anti-flapping window active".to_string()
        } else {
            "hysteresis threshold not met".to_string()
        }
    });
    PromotionRecommendation {
        candidate: top_ranked.clone(),
        approved,
        reason,
    }
}

/// Implements "Integrate with change 50 promotion lifecycle" (tasks): the
/// only bridge between a [`PromotionRecommendation`] and
/// [`KernelRegistry::promote_generation`] -- an unapproved recommendation
/// (hysteresis not met, or anti-flapping cooldown active) is never even
/// attempted against the Registry, and the caller gets back an explicit
/// [`KernelSelectionError::PromotionThresholdNotMet`] instead of a silent
/// no-op.
pub fn apply_promotion_recommendation(
    recommendation: &PromotionRecommendation,
    registry: &mut KernelRegistry,
    candidate: PreparedKernelId,
) -> Result<(), KernelSelectionError> {
    if !recommendation.approved {
        return Err(KernelSelectionError::PromotionThresholdNotMet);
    }
    registry
        .promote_generation(&recommendation.candidate.kernel, candidate)
        .map_err(|error| KernelSelectionError::InternalError {
            reason: error.to_string(),
        })
}

// ---------------------------------------------------------------------
// Static / Dynamic Selection / Selection Cache
// ---------------------------------------------------------------------
//
// "Static Selection" and "Dynamic Selection" (proposal) reuse
// [`crate::model_instance::KernelSelectionPolicy`] (aliased here as
// `ModelInstanceSelectionMode`) -- the existing Dynamic/Pinned choice a
// Model Instance already makes -- rather than introducing a second,
// competing mode enum.

/// Implements "Selection Caching" (proposal): dimensions a cached selection
/// key SHOULD include.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SelectionCacheKey {
    pub operator: OperatorId,
    pub provider: ProviderBinding,
    pub dtype: ComputeDType,
    pub layout: TensorLayoutKind,
    pub shape_bucket: String,
    pub batch_bucket: String,
    pub sequence_bucket: String,
    pub generation_phase: Option<GenerationPhase>,
    pub optimization_profile: OptimizationProfile,
    pub policy_version: KernelSelectionPolicyVersion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SelectionCacheInvalidationTrigger {
    EligibilityChanged,
    CompatibilityChanged,
    PolicyChanged,
    KernelRevoked,
    QualificationExpired,
}

/// Implements "Selection cache entries SHALL be invalidated when relevant
/// eligibility or compatibility state changes" (proposal). This
/// implementation invalidates the whole cache on any trigger rather than
/// attempting fine-grained partial invalidation, which is a conservative,
/// always-correct choice.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionCache {
    entries: BTreeMap<SelectionCacheKey, CandidateIdentity>,
}

impl SelectionCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &SelectionCacheKey) -> Option<&CandidateIdentity> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: SelectionCacheKey, value: CandidateIdentity) {
        self.entries.insert(key, value);
    }

    pub fn invalidate_all(&mut self, _trigger: SelectionCacheInvalidationTrigger) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------
// Model Instance Kernel Policy
// ---------------------------------------------------------------------

/// Opaque reference a Model Instance attaches, implementing "Model Instance
/// Kernel Policy" (proposal). Mirrors `TokenizerId`: a Model Instance
/// references a policy identity rather than embedding the full
/// [`KernelOptimizationPolicy`] (see
/// [`crate::model_instance::ModelInstanceDefinition::kernel_selection_policy`]).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSelectionPolicyId(String);

impl KernelSelectionPolicyId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelSelectionPolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Implements "Policy Versioning" (proposal): "Kernel selection policy SHALL
/// be versioned. Selection results SHOULD record the policy version that
/// produced them."
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelSelectionPolicyVersion(pub u32);

impl fmt::Display for KernelSelectionPolicyVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel-selection-policy-v{}", self.0)
    }
}

/// Implements "Select during Model Instance loading" (tasks): whether a
/// Model Instance's warmup plan already includes a point at which static
/// Kernel selection SHALL be resolved. A `Static`/pinned
/// [`ModelInstanceSelectionMode`] has nothing to resolve unless the plan
/// actually reaches [`crate::model_instance::ModelInstanceWarmupStep::KernelPreparationPlaceholder`]
/// -- `Disabled`/metadata-only warmup plans correctly report `false` here,
/// so a caller does not attempt selection before Kernel preparation is even
/// planned to happen.
pub fn static_selection_required_during_warmup(
    mode: &ModelInstanceSelectionMode,
    plan: &crate::model_instance::ModelInstanceWarmupPlan,
) -> bool {
    mode.is_pinned()
        && plan
            .steps
            .contains(&crate::model_instance::ModelInstanceWarmupStep::KernelPreparationPlaceholder)
}

/// The top-level Kernel selection policy, implementing "Selection Policy
/// Domain" (proposal / tasks section 1) and "Model Instance Kernel Policy":
/// "A Model Instance SHALL own or reference an explicit Kernel selection
/// policy." Named `KernelOptimizationPolicy` (not `KernelSelectionPolicy`)
/// to avoid colliding with the existing, narrower
/// [`crate::model_instance::KernelSelectionPolicy`] Dynamic/Pinned mode
/// choice, which this struct's `selection_mode` field reuses directly.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelOptimizationPolicy {
    pub id: KernelSelectionPolicyId,
    pub version: KernelSelectionPolicyVersion,
    pub profile: OptimizationProfile,
    pub ranking_strategy: RankingStrategy,
    pub fallback: FallbackPolicy,
    pub hysteresis: HysteresisPolicy,
    pub anti_flapping: AntiFlappingPolicy,
    pub exploration: ExplorationPolicy,
    pub selection_mode: ModelInstanceSelectionMode,
    pub determinism_required: bool,
    pub require_trusted: bool,
    pub allowed_qualification_profiles: BTreeSet<String>,
}

impl KernelOptimizationPolicy {
    /// Implements "Add policy validation tests" (tasks): a policy that
    /// claims the [`OptimizationProfile::Deterministic`] profile without
    /// requiring determinism is self-contradictory and SHALL be rejected up
    /// front, implementing "Deterministic Profile": "Performance SHALL NOT
    /// override deterministic requirements."
    pub fn validate(&self) -> Result<(), KernelSelectionError> {
        if matches!(self.profile, OptimizationProfile::Deterministic) && !self.determinism_required
        {
            return Err(KernelSelectionError::PolicyInvalid {
                reason: "deterministic profile requires determinism_required=true".into(),
            });
        }
        if matches!(self.profile, OptimizationProfile::Reproducible)
            && !self.selection_mode.is_pinned()
        {
            return Err(KernelSelectionError::PolicyInvalid {
                reason: "reproducible profile requires a pinned selection mode".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Model Component Boundary
// ---------------------------------------------------------------------

/// Implements "Model Component Boundary" (proposal): "Model Component SHALL
/// NOT choose concrete Kernel implementations."
#[derive(Clone, Debug, PartialEq)]
pub enum ModelComponentKernelRequest {
    PortableOperatorRequirement(OperatorId),
    ConcreteKernelOverride(KernelId),
}

pub fn validate_model_component_request(
    request: &ModelComponentKernelRequest,
) -> Result<(), KernelSelectionError> {
    match request {
        ModelComponentKernelRequest::PortableOperatorRequirement(_) => Ok(()),
        ModelComponentKernelRequest::ConcreteKernelOverride(_) => {
            Err(KernelSelectionError::PolicyInvalid {
                reason: "Model Component cannot select a concrete Kernel implementation".into(),
            })
        }
    }
}

// ---------------------------------------------------------------------
// Session / Generation / CLI Boundaries
// ---------------------------------------------------------------------

/// Implements "Session Boundary", "Generation Request Boundary" and "CLI
/// Boundary" (proposal) as one shared primitive: a preference is honored
/// only when the preferred Kernel is already present in the eligible set --
/// there is no path from "user asked for it" to "selected" that skips
/// eligibility.
fn prefer_if_eligible(
    preferred: &KernelId,
    eligible: &[EligibleCandidate],
) -> Option<CandidateIdentity> {
    eligible
        .iter()
        .find(|candidate| &candidate.identity.kernel == preferred)
        .map(|candidate| candidate.identity.clone())
}

/// Implements "Session Boundary" (proposal): "Runtime policy remains
/// authoritative."
pub fn resolve_session_preference(
    preferred: Option<&KernelId>,
    eligible: &[EligibleCandidate],
) -> Option<CandidateIdentity> {
    preferred.and_then(|kernel| prefer_if_eligible(kernel, eligible))
}

/// Implements "Generation Request Boundary" (proposal): "They SHALL NOT
/// directly force an ineligible concrete Kernel." A generation request may
/// only shift the high-level profile, never name a Kernel directly.
pub fn resolve_generation_preference(
    preferred_profile: Option<OptimizationProfile>,
    default_profile: OptimizationProfile,
) -> OptimizationProfile {
    preferred_profile.unwrap_or(default_profile)
}

/// Implements "CLI Boundary" (proposal): "CLI preferences SHALL map into
/// Runtime policy inputs. CLI SHALL NOT bypass Registry eligibility."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliPreference {
    Latency,
    Throughput,
    Deterministic,
}

pub fn map_cli_preference(preference: CliPreference) -> OptimizationProfile {
    match preference {
        CliPreference::Latency => OptimizationProfile::Latency,
        CliPreference::Throughput => OptimizationProfile::Throughput,
        CliPreference::Deterministic => OptimizationProfile::Deterministic,
    }
}

/// Implements "CLI SHALL NOT bypass Registry eligibility" (proposal)
/// directly: same eligible-only lookup as [`resolve_session_preference`].
pub fn resolve_cli_kernel_preference(
    preferred: Option<&KernelId>,
    eligible: &[EligibleCandidate],
) -> Option<CandidateIdentity> {
    preferred.and_then(|kernel| prefer_if_eligible(kernel, eligible))
}

// ---------------------------------------------------------------------
// Provider Boundary
// ---------------------------------------------------------------------

/// Implements "Provider Boundary" (proposal): "Provider SHALL NOT make the
/// final cross-Provider Kernel selection decision." The Provider-advertised
/// alternative is deliberately never read (`let _ =`) -- this function's
/// body is itself the proof that it cannot influence the outcome, mirroring
/// [`crate::kernel_qualification::eligibility_is_service_origin_independent`].
pub fn resolve_cross_provider_selection(
    runtime_ranked_first: &CandidateIdentity,
    provider_advertised_alternative: Option<&CandidateIdentity>,
) -> CandidateIdentity {
    let _ = provider_advertised_alternative;
    runtime_ranked_first.clone()
}

/// Implements "Provider-Local Variant Selection" (proposal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderPrivateVariant {
    pub contract_semantics_identical: bool,
    pub runtime_visible_compatibility_unchanged: bool,
    pub determinism_precision_unchanged: bool,
}

/// Implements "If the variants differ in Runtime-relevant properties, they
/// SHALL be modeled as distinct Kernel candidates" (proposal): `false` means
/// the caller SHALL register a distinct Registry entry instead of hiding the
/// distinction behind one Prepared Kernel.
pub fn provider_may_select_variant_privately(variant: &ProviderPrivateVariant) -> bool {
    variant.contract_semantics_identical
        && variant.runtime_visible_compatibility_unchanged
        && variant.determinism_precision_unchanged
}

// ---------------------------------------------------------------------
// Fallback Policy / Cross-Provider Movement
// ---------------------------------------------------------------------

/// Implements "Fallback Policy" (proposal): "Fallback MAY specify ordered
/// classes."
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FallbackPolicy {
    pub ordered_classes: Vec<KernelFallbackClass>,
    pub allow_reference_cpu: bool,
}

/// Implements "Reference CPU Fallback" (proposal): host execution is gated
/// on affinity, staging policy, and semantic support in addition to being
/// present in the policy's ordered classes -- it is never a bare default.
/// Reuses [`crate::compute::HostStagingPolicy`] (`Forbid`/`Permit`) rather
/// than introducing a second, competing staging-policy enum.
pub fn evaluate_kernel_selection_fallback(
    policy: &FallbackPolicy,
    candidate_class: KernelFallbackClass,
    resource_affinity_allows_movement: bool,
    host_staging: HostStagingPolicy,
    required_semantics_supported: bool,
) -> Result<KernelFallbackClass, KernelSelectionError> {
    if !policy.ordered_classes.contains(&candidate_class) {
        return Err(KernelSelectionError::FallbackDenied);
    }
    if candidate_class == KernelFallbackClass::HostExecution {
        let allowed = policy.allow_reference_cpu
            && resource_affinity_allows_movement
            && matches!(host_staging, HostStagingPolicy::Permit)
            && required_semantics_supported;
        if !allowed {
            return Err(KernelSelectionError::FallbackDenied);
        }
    }
    Ok(candidate_class)
}

/// Implements "Fallback SHALL NOT be silently inserted" (proposal): tries
/// each of `policy.ordered_classes` in the declared order and returns the
/// first one [`evaluate_kernel_selection_fallback`] accepts, or an explicit
/// exhaustion error.
pub fn evaluate_kernel_selection_fallback_chain(
    policy: &FallbackPolicy,
    resource_affinity_allows_movement: bool,
    host_staging: HostStagingPolicy,
    required_semantics_supported: bool,
) -> Result<KernelFallbackClass, KernelSelectionError> {
    for class in &policy.ordered_classes {
        if let Ok(accepted) = evaluate_kernel_selection_fallback(
            policy,
            *class,
            resource_affinity_allows_movement,
            host_staging,
            required_semantics_supported,
        ) {
            return Ok(accepted);
        }
    }
    Err(KernelSelectionError::FallbackExhausted)
}

/// Implements "No Hidden Cross-Provider Movement" (proposal): "If candidate
/// selection requires data movement, the movement SHALL be explicit and
/// policy-authorized." An implicit movement is rejected regardless of
/// whether policy would have authorized it if asked.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrossProviderMovement {
    pub explicit: bool,
    pub authorized_by_policy: bool,
}

pub fn validate_cross_provider_movement(
    movement: &CrossProviderMovement,
) -> Result<(), KernelSelectionError> {
    if movement.explicit && movement.authorized_by_policy {
        Ok(())
    } else {
        Err(KernelSelectionError::FallbackDenied)
    }
}

// ---------------------------------------------------------------------
// Reproducible Mode
// ---------------------------------------------------------------------

/// Implements "Reproducible Profile" (proposal): what a reproducible
/// execution policy MAY pin. Reuses
/// [`crate::model_instance::PinnedKernelSelection`] -- the Model Instance's
/// own KernelId/artifact-digest/prepared-generation/qualification-profile
/// pin -- rather than introducing a second, competing pin representation.
///
/// Implements "Define failure if pinned candidate unavailable" (tasks): the
/// pin resolves against the eligible set (ineligible) and, if absent there,
/// against the raw discovered candidate set (unavailable), so the caller
/// gets the correct one of [`KernelSelectionError::PinnedKernelUnavailable`]
/// / [`KernelSelectionError::PinnedKernelIneligible`].
pub fn resolve_pinned_selection(
    pin: &PinnedKernelSelection,
    discovered: &[CandidateIdentity],
    eligible: &[EligibleCandidate],
) -> Result<CandidateIdentity, KernelSelectionError> {
    if let Some(candidate) = eligible
        .iter()
        .find(|candidate| candidate.identity.kernel == pin.kernel)
    {
        return Ok(candidate.identity.clone());
    }
    if discovered
        .iter()
        .any(|candidate| candidate.kernel == pin.kernel)
    {
        Err(KernelSelectionError::PinnedKernelIneligible)
    } else {
        Err(KernelSelectionError::PinnedKernelUnavailable)
    }
}

// ---------------------------------------------------------------------
// Exploration / Canary
// ---------------------------------------------------------------------

/// Implements "Exploration" (proposal): "Exploration SHALL be explicit and
/// disabled by default for strict/reproducible modes."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplorationPolicy {
    pub enabled: bool,
    pub disabled_for_reproducible: bool,
}

impl Default for ExplorationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            disabled_for_reproducible: true,
        }
    }
}

pub fn exploration_allowed(policy: &ExplorationPolicy, reproducible_mode_active: bool) -> bool {
    policy.enabled && !(reproducible_mode_active && policy.disabled_for_reproducible)
}

/// Implements "Exploration SHALL only consider already eligible candidates.
/// Unqualified or untrusted Kernels SHALL NOT be explored" (proposal): the
/// parameter type alone makes this structural -- only an [`EligibleCandidate`]
/// can be passed, and that type is only constructible after
/// [`evaluate_candidate_eligibility`] succeeds.
pub fn eligible_for_exploration(
    policy: &ExplorationPolicy,
    reproducible_mode_active: bool,
    candidate: &EligibleCandidate,
) -> bool {
    let _ = candidate;
    exploration_allowed(policy, reproducible_mode_active)
}

/// Implements "Canary Selection" (proposal): local policy semantics only, no
/// distributed rollout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CanaryPolicy {
    pub max_requests: Option<u64>,
    pub max_duration_seconds: Option<u64>,
    pub max_percentage: Option<u8>,
}

pub fn canary_budget_exhausted(
    policy: &CanaryPolicy,
    requests_used: u64,
    seconds_elapsed: u64,
) -> bool {
    policy.max_requests.is_some_and(|max| requests_used >= max)
        || policy
            .max_duration_seconds
            .is_some_and(|max| seconds_elapsed >= max)
}

/// Implements "Exploration Failure" (proposal): "Failure SHALL NOT
/// automatically affect unrelated Kernels."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExplorationFailureAction {
    StopExploration,
    MarkUnhealthy,
    DemoteCandidate,
    TriggerRollback,
    RetainKnownGoodActive,
}

/// Always returns `false`: an unrelated candidate is never affected by
/// another candidate's exploration failure, regardless of `action`. The
/// function signature -- taking the unrelated candidate only to prove it is
/// untouched -- is itself the guarantee.
pub fn exploration_failure_affects_unrelated_candidate(
    action: ExplorationFailureAction,
    failing_candidate: &CandidateIdentity,
    unrelated_candidate: &CandidateIdentity,
) -> bool {
    let _ = (action, failing_candidate, unrelated_candidate);
    false
}

/// Implements "Integrate rollback where applicable" (tasks): the only bridge
/// from an [`ExplorationFailureAction`] to
/// [`KernelRegistry::rollback_generation`] -- only `TriggerRollback` reaches
/// the Registry; every other action is a structural no-op against it (`Ok(())`
/// without calling the Registry at all), so exploration bookkeeping actions
/// like `MarkUnhealthy` or `DemoteCandidate` can never accidentally trigger a
/// rollback.
pub fn apply_exploration_failure_action(
    action: ExplorationFailureAction,
    registry: &mut KernelRegistry,
    failing_kernel: &KernelId,
) -> Result<(), KernelSelectionError> {
    if !matches!(action, ExplorationFailureAction::TriggerRollback) {
        return Ok(());
    }
    registry
        .rollback_generation(failing_kernel)
        .map_err(|error| KernelSelectionError::InternalError {
            reason: error.to_string(),
        })
}

// ---------------------------------------------------------------------
// Online Measurement / Selection Metrics
// ---------------------------------------------------------------------

/// Implements "Selection Metrics" (proposal): metrics are associated with
/// Kernel generation, Operator, workload bucket, Provider/Device, and
/// execution profile -- there is no field capable of holding raw model
/// weights, tensors, prompts, or native handles.
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineMeasurement {
    pub kernel_generation: PreparedKernelId,
    pub operator: OperatorId,
    pub workload_bucket: String,
    pub provider: ProviderBinding,
    pub execution_profile: Option<String>,
}

/// Implements "Raw model data SHALL not be required for selection
/// analytics" (proposal): a structural fact about [`OnlineMeasurement`]'s
/// field set, not a runtime check.
pub fn online_measurement_requires_no_raw_model_data(measurement: &OnlineMeasurement) -> bool {
    let _ = measurement;
    true
}

/// Implements "Online measurement SHALL NOT override correctness or trust"
/// (proposal): `trust` passes through completely unchanged, regardless of
/// what the measurement suggests.
pub fn online_measurement_cannot_override_correctness_or_trust(
    measurement_suggests_faster: bool,
    trust: KernelArtifactTrust,
) -> KernelArtifactTrust {
    let _ = measurement_suggests_faster;
    trust
}

// ---------------------------------------------------------------------
// Policy Precedence
// ---------------------------------------------------------------------

/// Implements "Policy Precedence" (proposal)'s recommended precedence
/// order. Declaration order is authority order: earlier variants outrank
/// later ones wherever this module compares two [`PolicyPrecedenceLevel`]
/// values.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PolicyPrecedenceLevel {
    RuntimeSafety,
    Deployment,
    ModelInstance,
    Session,
    Generation,
    Cli,
}

/// One composed stack of preferences/constraints across every precedence
/// level, implementing "Policy MAY be composed from multiple levels"
/// (proposal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyConstraintStack {
    pub runtime_safety_forces_deterministic: bool,
    pub deployment_forbids_profile: BTreeSet<OptimizationProfile>,
    pub model_instance_profile: Option<OptimizationProfile>,
    pub session_preference: Option<OptimizationProfile>,
    pub generation_preference: Option<OptimizationProfile>,
    pub cli_preference: Option<OptimizationProfile>,
}

/// Implements "Lower-level preferences SHALL NOT override higher-level
/// constraints" (proposal): the most specific preference (CLI) is honored
/// only up to what deployment policy permits, and Runtime safety always has
/// the final say.
pub fn resolve_effective_profile(stack: &PolicyConstraintStack) -> OptimizationProfile {
    let requested = stack
        .cli_preference
        .or(stack.generation_preference)
        .or(stack.session_preference)
        .or(stack.model_instance_profile)
        .unwrap_or(OptimizationProfile::Balanced);
    let mut effective = requested;
    if stack.deployment_forbids_profile.contains(&effective) {
        effective = stack
            .model_instance_profile
            .filter(|profile| !stack.deployment_forbids_profile.contains(profile))
            .unwrap_or(OptimizationProfile::Balanced);
    }
    if stack.runtime_safety_forces_deterministic {
        effective = OptimizationProfile::Deterministic;
    }
    effective
}

// ---------------------------------------------------------------------
// Selection Stability
// ---------------------------------------------------------------------

/// Implements "Selection Stability" (proposal): "Performance optimization
/// SHALL NOT introduce uncontrolled selection nondeterminism." Ranking the
/// same input twice through the same strategy SHALL yield the same result.
pub fn selection_is_deterministic_for_identical_inputs(
    candidates: &[EligibleCandidate],
    strategy: &RankingStrategy,
) -> bool {
    rank_candidates(candidates, strategy) == rank_candidates(candidates, strategy)
}

// ---------------------------------------------------------------------
// Selection Explainability
// ---------------------------------------------------------------------

/// Implements "Selection Explainability" (proposal). Every field is a
/// stable identity, a structured reason, or a bool -- structurally
/// incapable of holding a native handle, raw tensor value, model weight, KV
/// cache content, raw prompt, secret, or credential, implementing
/// "Observability SHALL NOT expose" (proposal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionExplanation {
    pub eligible: Vec<CandidateIdentity>,
    pub exclusions: BTreeMap<String, KernelSelectionExclusionReason>,
    pub ranked: Vec<CandidateIdentity>,
    pub selected: Option<CandidateIdentity>,
    pub retained_active: bool,
    pub fallback: Option<KernelFallbackClass>,
    pub promotion: Option<PromotionRecommendation>,
}

impl SelectionExplanation {
    /// Always `false`: a structural fact about this type's field set, kept
    /// as an explicit predicate so callers/tests can assert it rather than
    /// re-deriving the reasoning inline.
    pub const fn contains_native_handles(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Selection error, implementing the proposal's "Selection
/// Errors" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelSelectionError {
    NoCandidates,
    NoEligibleCandidates,
    PolicyInvalid { reason: String },
    ProfileUnsupported { profile: String },
    PinnedKernelUnavailable,
    PinnedKernelIneligible,
    MetricMissing { dimension: String },
    BenchmarkStale,
    BenchmarkIncompatible,
    MemoryInfeasible,
    AffinityIncompatible,
    DeterminismUnsatisfied,
    FallbackDenied,
    FallbackExhausted,
    PromotionThresholdNotMet,
    CacheStale,
    ExplorationDenied,
    InternalError { reason: String },
}

impl KernelSelectionError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::NoCandidates => "kernel-selection-no-candidates",
            Self::NoEligibleCandidates => "kernel-selection-no-eligible-candidates",
            Self::PolicyInvalid { .. } => "kernel-selection-policy-invalid",
            Self::ProfileUnsupported { .. } => "kernel-selection-profile-unsupported",
            Self::PinnedKernelUnavailable => "kernel-selection-pinned-kernel-unavailable",
            Self::PinnedKernelIneligible => "kernel-selection-pinned-kernel-ineligible",
            Self::MetricMissing { .. } => "kernel-selection-metric-missing",
            Self::BenchmarkStale => "kernel-selection-benchmark-stale",
            Self::BenchmarkIncompatible => "kernel-selection-benchmark-incompatible",
            Self::MemoryInfeasible => "kernel-selection-memory-infeasible",
            Self::AffinityIncompatible => "kernel-selection-affinity-incompatible",
            Self::DeterminismUnsatisfied => "kernel-selection-determinism-unsatisfied",
            Self::FallbackDenied => "kernel-selection-fallback-denied",
            Self::FallbackExhausted => "kernel-selection-fallback-exhausted",
            Self::PromotionThresholdNotMet => "kernel-selection-promotion-threshold-not-met",
            Self::CacheStale => "kernel-selection-cache-stale",
            Self::ExplorationDenied => "kernel-selection-exploration-denied",
            Self::InternalError { .. } => "internal-kernel-selection-error",
        }
    }
}

impl fmt::Display for KernelSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyInvalid { reason } | Self::InternalError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::ProfileUnsupported { profile } => write!(f, "{}: {profile}", self.id()),
            Self::MetricMissing { dimension } => write!(f, "{}: {dimension}", self.id()),
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelSelectionError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Selection lifecycle observation categories, implementing "Observability"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelSelectionObservationKind {
    SelectionStarted,
    CandidateDiscovered,
    CandidateExcluded,
    CandidateEligible,
    CandidateRanked,
    KernelSelected,
    ActiveKernelRetained,
    FallbackSelected,
    SelectionCacheHit,
    SelectionCacheMiss,
    SelectionRecomputed,
    PromotionSuggested,
    PromotionThresholdNotMet,
    ExplorationStarted,
    ExplorationStopped,
}

/// A single selection observation. Structurally guaranteed to never carry a
/// native handle, raw tensor value, model weight, KV cache content, raw
/// prompt, secret, or credential: the only fields are an enum `kind`, an
/// optional stable Kernel key, and a `redacted_metadata` map whose values
/// always pass through `redact_backend_diagnostic` first, implementing
/// "Observability SHALL NOT expose" (proposal). Mirrors
/// [`crate::kernel_qualification::QualificationObservation`]'s shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelectionObservation {
    pub kind: KernelSelectionObservationKind,
    pub kernel: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelSelectionObservation {
    pub fn new(kind: KernelSelectionObservationKind) -> Self {
        Self {
            kind,
            kernel: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_kernel(mut self, kernel: &KernelId) -> Self {
        self.kernel = Some(kernel.stable_key());
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
// Conformance
// ---------------------------------------------------------------------

/// A single Kernel Selection Policy conformance check result, mirroring
/// [`crate::kernel_qualification::KernelQualificationConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelectionPolicyConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelectionPolicyConformanceReport {
    pub results: Vec<KernelSelectionPolicyConformanceResult>,
}

impl KernelSelectionPolicyConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelSelectionPolicyConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelSelectionPolicyConformanceResult {
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
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        crate::KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::CpuScalar,
    )
}

/// Runs the Kernel Selection Policy conformance checks described in this
/// module's doc comment and required by
/// `specs/kernel-selection-policy/spec.md` and the selection-policy portion
/// of `specs/conformance/spec.md`.
pub fn run_kernel_selection_policy_conformance() -> KernelSelectionPolicyConformanceReport {
    let mut results = Vec::new();

    // "Eligibility Precedes Ranking": an untrusted-but-fastest candidate can
    // never even become an EligibleCandidate, so it structurally cannot
    // enter ranking, regardless of its metrics.
    let trusted_identity = CandidateIdentity {
        kernel: conformance_kernel_id("trusted-slow"),
        provider: ProviderBinding::new("conformance-provider"),
        artifact_digest: None,
    };
    let untrusted_identity = CandidateIdentity {
        kernel: conformance_kernel_id("untrusted-fast"),
        provider: ProviderBinding::new("conformance-provider"),
        artifact_digest: None,
    };
    let trusted = EligibleCandidate::from_checked(
        trusted_identity.clone(),
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 10.0),
        &CandidateEligibilityInput::all_satisfied(),
    );
    let untrusted_rejected = EligibleCandidate::from_checked(
        untrusted_identity,
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 0.1),
        &CandidateEligibilityInput {
            trust: KernelArtifactTrust::Untrusted,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    record(
        &mut results,
        "untrusted fastest candidate is never selected",
        trusted.is_ok()
            && matches!(
                untrusted_rejected,
                Err(KernelSelectionExclusionReason::TrustDenied)
            ),
        format!("unexpected outcomes: {trusted:?} / {untrusted_rejected:?}"),
    );

    // "Memory Feasibility Precedes Ranking": Memory Manager rejection cannot
    // be overridden by ranking.
    let memory_infeasible = EligibleCandidate::from_checked(
        trusted_identity.clone(),
        CandidateMetrics::default(),
        &CandidateEligibilityInput {
            memory_feasible: false,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    record(
        &mut results,
        "memory-infeasible candidate is excluded before ranking",
        matches!(
            memory_infeasible,
            Err(KernelSelectionExclusionReason::MemoryInfeasible)
        ),
        format!("unexpected outcome: {memory_infeasible:?}"),
    );
    record(
        &mut results,
        "memory_feasible_before_ranking rejects an infeasible workspace",
        matches!(
            memory_feasible_before_ranking(false),
            Err(KernelSelectionExclusionReason::MemoryInfeasible)
        ),
        "expected memory infeasibility to be rejected",
    );

    // "Affinity Precedes Ranking": a faster cross-Provider candidate that
    // violates Resource Affinity cannot be ranked at all.
    let affinity_violation = EligibleCandidate::from_checked(
        trusted_identity.clone(),
        CandidateMetrics::default(),
        &CandidateEligibilityInput {
            resource_affinity_compatible: false,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    record(
        &mut results,
        "Resource Affinity violation is excluded before ranking regardless of performance",
        matches!(
            affinity_violation,
            Err(KernelSelectionExclusionReason::ResourceAffinityIncompatible)
        ),
        format!("unexpected outcome: {affinity_violation:?}"),
    );

    // "Determinism Policy Conformance": a nondeterministic fastest candidate
    // is excluded when determinism is required.
    let nondeterministic = EligibleCandidate::from_checked(
        trusted_identity.clone(),
        CandidateMetrics::default(),
        &CandidateEligibilityInput {
            determinism_compatible: false,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    record(
        &mut results,
        "nondeterministic candidate excluded under deterministic requirement",
        matches!(
            nondeterministic,
            Err(KernelSelectionExclusionReason::DeterminismIncompatible)
        ),
        format!("unexpected outcome: {nondeterministic:?}"),
    );

    // "Stable Tie-Break Conformance": identical scores yield identical,
    // repeatable ordering.
    let tie_a = EligibleCandidate::from_checked(
        CandidateIdentity {
            kernel: conformance_kernel_id("tie-a"),
            provider: ProviderBinding::new("conformance-provider"),
            artifact_digest: None,
        },
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 5.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .expect("tie candidate a is eligible");
    let tie_b = EligibleCandidate::from_checked(
        CandidateIdentity {
            kernel: conformance_kernel_id("tie-b"),
            provider: ProviderBinding::new("conformance-provider"),
            artifact_digest: None,
        },
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 5.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .expect("tie candidate b is eligible");
    let tie_strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    let tie_candidates = vec![tie_b.clone(), tie_a.clone()];
    let ranked_once = rank_candidates(&tie_candidates, &tie_strategy);
    let ranked_twice = rank_candidates(&tie_candidates, &tie_strategy);
    record(
        &mut results,
        "equal-score candidates produce a stable, repeatable tie-break order",
        ranked_once.is_ok()
            && ranked_once == ranked_twice
            && selection_is_deterministic_for_identical_inputs(&tie_candidates, &tie_strategy),
        format!("unexpected outcome: {ranked_once:?} vs {ranked_twice:?}"),
    );

    // "Prefill Versus Decode": a throughput-optimized prefill Kernel and a
    // latency-optimized decode Kernel MAY be ranked to different winners for
    // the same Operator, even under an identical ranking strategy.
    let throughput_kernel = EligibleCandidate::from_checked(
        CandidateIdentity {
            kernel: conformance_kernel_id("prefill-throughput"),
            provider: ProviderBinding::new("conformance-provider"),
            artifact_digest: None,
        },
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 50.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .expect("prefill candidate is eligible");
    let latency_kernel = EligibleCandidate::from_checked(
        CandidateIdentity {
            kernel: conformance_kernel_id("decode-latency"),
            provider: ProviderBinding::new("conformance-provider"),
            artifact_digest: None,
        },
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 2.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .expect("decode candidate is eligible");
    let phase_strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    let by_phase = rank_by_generation_phase(
        std::slice::from_ref(&throughput_kernel),
        std::slice::from_ref(&latency_kernel),
        &phase_strategy,
    );
    record(
        &mut results,
        "prefill and decode may rank to different winning Kernels for the same Operator",
        matches!(
            &by_phase,
            Ok(ranking)
                if ranking.get(&GenerationPhase::Prefill) == Some(&vec![throughput_kernel.identity.clone()])
                    && ranking.get(&GenerationPhase::Decode) == Some(&vec![latency_kernel.identity.clone()])
        ),
        format!("unexpected outcome: {by_phase:?}"),
    );

    // "Benchmark Context Conformance": incompatible benchmark context is not
    // authoritative.
    let base_context = BenchmarkContext {
        provider: ProviderBinding::new("conformance-provider"),
        device_architecture: "sm90".into(),
        driver_runtime_compatibility: "cuda-12".into(),
        operator_version: 1,
        artifact_digest: Some("digest-1".into()),
        dtype: ComputeDType::Float32,
        layout: TensorLayoutKind::Contiguous,
        shape_bucket: "b1s128".into(),
        batch_bucket: "b1".into(),
        sequence_bucket: "s128".into(),
        execution_mode: KernelExecutionMode::Synchronous,
        benchmark_profile_version: 1,
    };
    let incompatible_context = BenchmarkContext {
        device_architecture: "sm70".into(),
        ..base_context.clone()
    };
    record(
        &mut results,
        "benchmark evidence from an incompatible architecture is not authoritative",
        !benchmark_context_compatible(&incompatible_context, &base_context),
        "expected incompatible benchmark context to be rejected",
    );
    record(
        &mut results,
        "benchmark evidence from a matching context is authoritative",
        benchmark_context_compatible(&base_context, &base_context),
        "expected identical benchmark context to be compatible",
    );

    // "Hysteresis Conformance": insignificant benefit does not force
    // promotion; significant benefit does.
    let hysteresis_policy = HysteresisPolicy {
        promotion_threshold_fraction: 0.05,
    };
    record(
        &mut results,
        "0.1 percent improvement does not trigger promotion",
        matches!(
            evaluate_hysteresis(100.0, 99.9, &hysteresis_policy),
            SelectionOutcome::RetainActive
        ),
        "expected marginal improvement to retain the active Kernel",
    );
    record(
        &mut results,
        "12 percent improvement is eligible for promotion",
        matches!(
            evaluate_hysteresis(100.0, 88.0, &hysteresis_policy),
            SelectionOutcome::PromoteCandidate
        ),
        "expected significant improvement to be promotable",
    );

    // "Explicit Fallback Conformance": fallback only occurs according to
    // policy.
    let fail_only_policy = FallbackPolicy {
        ordered_classes: vec![],
        allow_reference_cpu: false,
    };
    let denied_fallback = evaluate_kernel_selection_fallback(
        &fail_only_policy,
        KernelFallbackClass::HostExecution,
        true,
        HostStagingPolicy::Permit,
        true,
    );
    record(
        &mut results,
        "fallback fails instead of silently using CPU when policy says fail",
        matches!(denied_fallback, Err(KernelSelectionError::FallbackDenied)),
        format!("unexpected outcome: {denied_fallback:?}"),
    );
    let allowed_policy = FallbackPolicy {
        ordered_classes: vec![KernelFallbackClass::HostExecution],
        allow_reference_cpu: true,
    };
    let allowed_fallback = evaluate_kernel_selection_fallback(
        &allowed_policy,
        KernelFallbackClass::HostExecution,
        true,
        HostStagingPolicy::Permit,
        true,
    );
    record(
        &mut results,
        "fallback is granted when explicitly permitted by policy",
        allowed_fallback.is_ok(),
        format!("unexpected outcome: {allowed_fallback:?}"),
    );

    // "No Hidden Data Movement Conformance": host staging forbidden fails
    // even when policy would otherwise authorize movement.
    let staging_forbidden = evaluate_kernel_selection_fallback(
        &allowed_policy,
        KernelFallbackClass::HostExecution,
        true,
        HostStagingPolicy::Forbid,
        true,
    );
    record(
        &mut results,
        "CPU fallback requiring forbidden host staging fails",
        matches!(staging_forbidden, Err(KernelSelectionError::FallbackDenied)),
        format!("unexpected outcome: {staging_forbidden:?}"),
    );
    let implicit_movement = validate_cross_provider_movement(&CrossProviderMovement {
        explicit: false,
        authorized_by_policy: true,
    });
    record(
        &mut results,
        "implicit cross-Provider movement is rejected even if policy would authorize it",
        implicit_movement.is_err(),
        "expected implicit movement to be rejected",
    );

    // "Model Component Independence Conformance": a Component cannot force
    // concrete Kernel selection.
    let component_override = validate_model_component_request(
        &ModelComponentKernelRequest::ConcreteKernelOverride(conformance_kernel_id("override")),
    );
    record(
        &mut results,
        "Model Component cannot request a specific concrete Kernel",
        component_override.is_err(),
        "expected concrete Kernel override request to be rejected",
    );
    let component_portable = validate_model_component_request(
        &ModelComponentKernelRequest::PortableOperatorRequirement(OperatorId::magnetar(
            "matmul",
            1,
            crate::OperatorFamily::LinearAlgebra,
        )),
    );
    record(
        &mut results,
        "Model Component may declare a portable Operator requirement",
        component_portable.is_ok(),
        "expected portable Operator requirement to be accepted",
    );

    // "User Preference Is Non-Authoritative": CLI/session/generation
    // preference cannot resurrect a revoked/ineligible candidate.
    let eligible_pool = vec![tie_a.clone()];
    let revoked_preference =
        resolve_cli_kernel_preference(Some(&conformance_kernel_id("revoked")), &eligible_pool);
    record(
        &mut results,
        "CLI preference for a revoked candidate is not honored",
        revoked_preference.is_none(),
        "expected preference for an ineligible Kernel to resolve to None",
    );
    let eligible_preference =
        resolve_cli_kernel_preference(Some(&tie_a.identity.kernel), &eligible_pool);
    record(
        &mut results,
        "CLI preference for an eligible candidate is honored",
        eligible_preference == Some(tie_a.identity.clone()),
        "expected preference for an eligible Kernel to resolve",
    );

    // "Exploration Eligibility Conformance": exploration only ever considers
    // already-eligible candidates, and is off by default in reproducible
    // mode.
    let exploration_policy = ExplorationPolicy {
        enabled: true,
        disabled_for_reproducible: true,
    };
    record(
        &mut results,
        "exploration considers only an EligibleCandidate, structurally excluding unqualified candidates",
        eligible_for_exploration(&exploration_policy, false, &tie_a),
        "expected exploration to be allowed for an eligible candidate outside reproducible mode",
    );
    record(
        &mut results,
        "exploration is disabled by default under reproducible mode",
        !eligible_for_exploration(&exploration_policy, true, &tie_a),
        "expected exploration to be denied while reproducible mode is active",
    );

    // "Provider Global Selection Boundary Conformance": Provider cannot
    // override Runtime's cross-Provider decision.
    let runtime_choice = tie_a.identity.clone();
    let provider_alternative = tie_b.identity.clone();
    let cross_provider_result =
        resolve_cross_provider_selection(&runtime_choice, Some(&provider_alternative));
    record(
        &mut results,
        "Provider-advertised alternative cannot override the Runtime cross-Provider decision",
        cross_provider_result == runtime_choice,
        format!("unexpected outcome: {cross_provider_result:?}"),
    );

    // "Selection Explainability Conformance": explanation carries structured
    // exclusion reasons and no native handles, even when every candidate is
    // excluded.
    let mut explanation = SelectionExplanation::default();
    explanation.exclusions.insert(
        tie_b.identity.kernel.stable_key(),
        KernelSelectionExclusionReason::ResourceAffinityIncompatible,
    );
    record(
        &mut results,
        "selection explanation with every candidate excluded still reports structured reasons without native handles",
        !explanation.exclusions.is_empty()
            && explanation.selected.is_none()
            && !explanation.contains_native_handles(),
        "expected structured exclusion reasons and no native handles",
    );

    KernelSelectionPolicyConformanceReport { results }
}
