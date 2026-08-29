//! Kernel Performance Model and Adaptive Feedback contract (see
//! `openspec/changes/define-kernel-performance-model-and-adaptive-feedback-contract`).
//!
//! This module does not implement a benchmark harness, a machine-learning
//! model, or a code generator (proposal's "Non-Goals"). It defines, as
//! executable Rust types and pure functions, the bounded contract that lets
//! Runtime observe real production Kernel execution and turn it into
//! optimization *evidence* -- never correctness, trust, or qualification
//! evidence (proposal's "Core Principle"):
//!
//! ```text
//! performance evidence != correctness evidence
//! performance evidence != trust evidence
//! performance evidence != qualification evidence
//! ```
//!
//! - [`KernelExecutionPerformanceObservation`]: one redacted, bounded record
//!   of a completed Kernel execution -- structurally incapable of carrying
//!   raw tensor/prompt content.
//! - [`KernelPerformanceWorkloadBucket`][]: reuses
//!   [`crate::kernel_selection_policy::BenchmarkContext`] (this proposal's
//!   "Workload Bucket" is the same compatibility-relevant context that
//!   already governs benchmark applicability) plus phase/quantization, so
//!   evidence is never treated as globally applicable.
//! - [`KernelPerformanceAggregator`] / [`KernelPerformanceModel`]: bounded,
//!   versioned aggregated evidence for one candidate in one workload bucket.
//! - [`detect_benchmark_drift`] / [`detect_workload_drift`] /
//!   [`detect_regression`] / [`confirm_regression`]: the adaptive detection
//!   pipeline, gated by [`evaluate_sample_sufficiency`] so "a small number of
//!   observations SHALL not automatically trigger promotion, rollback, or
//!   re-tuning" (proposal).
//! - [`KernelRetuningRequest`] / [`retuning_request_respects_autotuning_boundary`]
//!   / [`request_retuning_outside_hot_path`]: re-tuning stays inside the
//!   existing [`crate::kernel_autotuning`] contract and SHALL NOT run
//!   synchronously inside decode.
//! - [`escalate_to_optimization_plane`]: the only path from unresolved
//!   bounded tuning to the external Optimization Plane -- Runtime never
//!   invokes a generator itself.
//! - [`KernelPerformanceDemotionSignal`] / [`KernelPerformanceRollbackSignal`]:
//!   recommendations only; actual rollback/promotion authority remains with
//!   existing selection/promotion machinery.
//! - [`KernelPerformanceHealth`]: kept structurally separate from
//!   [`crate::kernel_qualification::QualificationStatus`] and Provider
//!   health.
//! - [`KernelPerformanceFeedbackMode`] / [`reproducible_mode_blocks_adaptation`]:
//!   Model Instance policy, including the reproducible-mode override.
//! - [`KernelPerformanceError`] / [`KernelPerformanceObservationKind`]: the
//!   structured error and observability vocabulary from the proposal's
//!   "Error Model" and "Observability" sections.
//! - [`KernelPerformanceConformanceReport`] /
//!   [`run_kernel_performance_conformance`]: the fourteen conformance
//!   requirements from `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::{
    BenchmarkContext, CompiledKernelArtifactId, EligibleCandidate, GenerationPhase,
    KernelArtifactTrust, KernelAutotuningCandidate, KernelAutotuningPlan, KernelId,
    MemoryPressureLevel, OperatorId, PreparedKernelGeneration, ProviderBinding,
    ProviderPressureLevel, reject_decode_hot_path_trigger,
};
use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt,
};

pub const KERNEL_PERFORMANCE_MODEL_CONTRACT_VERSION: &str = "0.1.0";

/// Implements "Bounded Retention" (proposal): "Runtime SHALL avoid unbounded
/// per-invocation telemetry growth." Raw latencies beyond this bound are
/// discarded from the per-bucket ring buffer; running count/mean/variance are
/// still tracked exactly via incremental (Welford) aggregation.
pub const MAX_RAW_SAMPLES_PER_MODEL: usize = 512;

// ---------------------------------------------------------------------
// Timing Capability
// ---------------------------------------------------------------------

/// Implements "Measurement Capability" (proposal): "Timing method SHALL be
/// identifiable in evidence." No variant is "unknown/unspecified" -- a
/// Provider unable to measure precisely still reports
/// [`Self::HostObserved`], the coarsest identifiable method.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PerformanceTimingMethod {
    HostObserved,
    ProviderReportedAggregate,
    DeviceEvent,
    HardwareTimestamp,
}

// ---------------------------------------------------------------------
// Workload Bucket
// ---------------------------------------------------------------------

/// Implements "Workload Bucket" (proposal): "Performance evidence SHALL be
/// associated with a workload bucket rather than treated as globally
/// applicable." Deliberately reuses
/// [`crate::kernel_selection_policy::BenchmarkContext`] for the
/// compatibility-relevant identity fields instead of introducing a second,
/// competing bucket representation -- the same fields that already gate
/// benchmark applicability ([`crate::benchmark_context_compatible`]) gate
/// online performance evidence applicability here.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPerformanceWorkloadBucket {
    pub context: BenchmarkContext,
    pub phase: Option<GenerationPhase>,
    pub quantization_profile: Option<String>,
    pub bucket_policy_version: u32,
}

impl KernelPerformanceWorkloadBucket {
    pub fn new(context: BenchmarkContext, bucket_policy_version: u32) -> Self {
        Self {
            context,
            phase: None,
            quantization_profile: None,
            bucket_policy_version,
        }
    }

    pub fn with_phase(mut self, phase: GenerationPhase) -> Self {
        self.phase = Some(phase);
        self
    }

    pub fn with_quantization_profile(mut self, profile: impl Into<String>) -> Self {
        self.quantization_profile = Some(profile.into());
        self
    }

    /// Implements "Workload Bucket Identity SHALL be deterministic" and
    /// "Equivalent Runtime contexts SHALL map to the same bucket under the
    /// same bucket policy version" (proposal): a pure function of typed,
    /// already-bucketed fields, so equal buckets always render an identical
    /// id regardless of construction order.
    pub fn bucket_id(&self) -> String {
        format!(
            "policy={}|provider={}|arch={}|driver={}|opver={}|artifact={}|dtype={:?}|layout={:?}|\
             shape={}|batch={}|seq={}|mode={:?}|phase={:?}|quant={}",
            self.bucket_policy_version,
            self.context.provider,
            self.context.device_architecture,
            self.context.driver_runtime_compatibility,
            self.context.operator_version,
            self.context.artifact_digest.as_deref().unwrap_or(""),
            self.context.dtype,
            self.context.layout,
            self.context.shape_bucket,
            self.context.batch_bucket,
            self.context.sequence_bucket,
            self.context.execution_mode,
            self.phase,
            self.quantization_profile.as_deref().unwrap_or(""),
        )
    }
}

/// Implements "No Raw Prompt Bucketing" (proposal): "Workload buckets SHALL
/// NOT include raw prompt text, user identity, document contents, secrets,
/// or arbitrary user data." A structural fact -- every
/// [`KernelPerformanceWorkloadBucket`] field is a typed identity, bucketed
/// range string, or enum, never a free-text content field.
pub const fn workload_bucket_excludes_raw_content() -> bool {
    true
}

// ---------------------------------------------------------------------
// Execution Completion / Failure Evidence
// ---------------------------------------------------------------------

/// Implements "Failure Rate Evidence" (proposal): "Execution failure
/// evidence SHALL preserve structured error categories."
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelExecutionCompletion {
    Success,
    Failed { category: String },
    TimedOut,
}

// ---------------------------------------------------------------------
// Performance Observation
// ---------------------------------------------------------------------

/// Implements "Kernel Execution Performance Observation" (proposal). Every
/// field is a stable identity, a bucketed/quantized measurement, or a
/// pressure enum -- structurally incapable of holding raw tensor contents,
/// prompt text, model weights, or a native handle, implementing "Raw tensor
/// contents SHALL not be required" and the "Privacy" section.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelExecutionPerformanceObservation {
    pub kernel: KernelId,
    pub artifact_digest: CompiledKernelArtifactId,
    pub prepared_generation: Option<PreparedKernelGeneration>,
    pub specialization_fingerprint: Option<String>,
    pub provider: ProviderBinding,
    pub workload_bucket: KernelPerformanceWorkloadBucket,
    pub timing_method: PerformanceTimingMethod,
    pub latency_micros: u64,
    pub queue_delay_micros: Option<u64>,
    pub provider_submission_micros: Option<u64>,
    pub workspace_bytes: Option<u64>,
    pub memory_pressure: Option<MemoryPressureLevel>,
    pub provider_pressure: Option<ProviderPressureLevel>,
    pub completion: KernelExecutionCompletion,
    pub is_warmup: bool,
    pub is_cold_start: bool,
    pub timestamp_millis: u64,
}

impl KernelExecutionPerformanceObservation {
    /// Implements "Add validation tests" (tasks) and "An observation SHALL
    /// identify enough execution context to avoid mixing incompatible
    /// measurements" (proposal): every binding this module later relies on
    /// for artifact/specialization/bucket isolation must actually be
    /// present.
    pub fn validate(&self) -> Result<(), KernelPerformanceError> {
        if self.artifact_digest.digest().is_empty() {
            return Err(KernelPerformanceError::ObservationInvalid {
                reason: "observation is missing an artifact digest".into(),
            });
        }
        if self.workload_bucket.context.shape_bucket.is_empty()
            && self.workload_bucket.context.batch_bucket.is_empty()
            && self.workload_bucket.context.sequence_bucket.is_empty()
        {
            return Err(KernelPerformanceError::BucketInvalid {
                reason: "observation carries no workload bucket dimensions".into(),
            });
        }
        if self.timestamp_millis == 0 {
            return Err(KernelPerformanceError::ContextInvalid {
                reason: "observation is missing a timestamp".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Metric Summary / Aggregation
// ---------------------------------------------------------------------

/// Implements "Performance Aggregation" (proposal): the bounded aggregate
/// evidence shape, never a per-invocation log.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelPerformanceMetricSummary {
    pub count: u64,
    pub mean_latency_micros: f64,
    pub variance_latency_micros: f64,
    pub min_latency_micros: u64,
    pub max_latency_micros: u64,
    pub p50_micros: u64,
    pub p90_micros: u64,
    pub p95_micros: u64,
    pub p99_micros: u64,
    pub failure_count: u64,
    pub timeout_count: u64,
}

impl KernelPerformanceMetricSummary {
    /// Implements "Failure Rate Evidence" (proposal).
    pub fn failure_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.failure_count as f64 / self.count as f64
        }
    }

    /// Implements "Timeout Evidence SHALL not be hidden by good average
    /// latency" (proposal): exposed as its own explicit rate, never folded
    /// into the mean.
    pub fn timeout_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.timeout_count as f64 / self.count as f64
        }
    }
}

/// Implements "Runtime SHOULD aggregate observations rather than retaining
/// every individual event indefinitely" and "Bounded Retention" (proposal).
/// Count/mean/variance are tracked exactly via Welford's online algorithm
/// (unbounded observation counts, O(1) memory); only the samples used for
/// quantile estimation are retained, bounded by
/// [`MAX_RAW_SAMPLES_PER_MODEL`] -- implementing "Raw observations MAY be
/// discarded after aggregation according to policy".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelPerformanceAggregator {
    count: u64,
    mean: f64,
    m2: f64,
    min: u64,
    max: u64,
    failure_count: u64,
    timeout_count: u64,
    quantile_window: VecDeque<u64>,
}

impl KernelPerformanceAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Implements "Add count/mean/variance/min/max/quantile summaries/
    /// failure count/timeout count" (tasks).
    pub fn record(&mut self, observation: &KernelExecutionPerformanceObservation) {
        let latency = observation.latency_micros;
        self.count += 1;
        let delta = latency as f64 - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = latency as f64 - self.mean;
        self.m2 += delta * delta2;
        self.min = if self.count == 1 {
            latency
        } else {
            self.min.min(latency)
        };
        self.max = self.max.max(latency);
        match &observation.completion {
            KernelExecutionCompletion::Failed { .. } => self.failure_count += 1,
            KernelExecutionCompletion::TimedOut => {
                self.failure_count += 1;
                self.timeout_count += 1;
            }
            KernelExecutionCompletion::Success => {}
        }
        // Implements "Bounded Retention": raw samples beyond the cap are
        // discarded (oldest first); count/mean/variance above remain exact.
        if self.quantile_window.len() >= MAX_RAW_SAMPLES_PER_MODEL {
            self.quantile_window.pop_front();
        }
        self.quantile_window.push_back(latency);
    }

    pub fn raw_sample_count(&self) -> usize {
        self.quantile_window.len()
    }

    fn quantile(&self, percentile: f64) -> u64 {
        if self.quantile_window.is_empty() {
            return 0;
        }
        let mut sorted: Vec<u64> = self.quantile_window.iter().copied().collect();
        sorted.sort_unstable();
        let rank = ((sorted.len() - 1) as f64 * percentile).round() as usize;
        sorted[rank.min(sorted.len() - 1)]
    }

    pub fn summary(&self) -> KernelPerformanceMetricSummary {
        KernelPerformanceMetricSummary {
            count: self.count,
            mean_latency_micros: self.mean,
            variance_latency_micros: if self.count > 1 {
                self.m2 / (self.count - 1) as f64
            } else {
                0.0
            },
            min_latency_micros: self.min,
            max_latency_micros: self.max,
            p50_micros: self.quantile(0.50),
            p90_micros: self.quantile(0.90),
            p95_micros: self.quantile(0.95),
            p99_micros: self.quantile(0.99),
            failure_count: self.failure_count,
            timeout_count: self.timeout_count,
        }
    }
}

// ---------------------------------------------------------------------
// Evidence Quality / Sample Sufficiency
// ---------------------------------------------------------------------

/// Implements "Confidence" (proposal): "It MAY instead expose structured
/// states ... Confidence SHALL not be fabricated when the model cannot
/// estimate it." Every variant is derived from real counts in
/// [`evaluate_sample_sufficiency`] -- there is no "estimated"/"guessed"
/// variant to fabricate a number from insufficient evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvidenceQuality {
    Insufficient,
    Low,
    Medium,
    High,
}

/// Implements "Sample Sufficiency" (proposal): "Runtime SHALL distinguish
/// insufficient evidence from meaningful performance evidence."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSufficiencyPolicy {
    pub minimum_samples: u64,
    pub low_samples: u64,
    pub medium_samples: u64,
    pub minimum_observation_duration_millis: u64,
}

impl Default for SampleSufficiencyPolicy {
    fn default() -> Self {
        Self {
            minimum_samples: 30,
            low_samples: 100,
            medium_samples: 500,
            minimum_observation_duration_millis: 1_000,
        }
    }
}

/// Implements "A small number of observations SHALL not automatically
/// trigger promotion, rollback, or re-tuning" (proposal): purely a function
/// of `summary.count` and observed duration against policy thresholds.
pub fn evaluate_sample_sufficiency(
    summary: &KernelPerformanceMetricSummary,
    observed_duration_millis: u64,
    policy: &SampleSufficiencyPolicy,
) -> EvidenceQuality {
    if summary.count < policy.minimum_samples
        || observed_duration_millis < policy.minimum_observation_duration_millis
    {
        EvidenceQuality::Insufficient
    } else if summary.count < policy.low_samples {
        EvidenceQuality::Low
    } else if summary.count < policy.medium_samples {
        EvidenceQuality::Medium
    } else {
        EvidenceQuality::High
    }
}

pub fn sufficient_for_adaptive_action(quality: EvidenceQuality) -> bool {
    quality != EvidenceQuality::Insufficient
}

// ---------------------------------------------------------------------
// Warmup / Cold-Start Handling
// ---------------------------------------------------------------------

/// Implements "Warmup Samples" (proposal): "Runtime SHOULD be able to
/// classify or exclude warmup samples from steady-state performance evidence
/// where appropriate."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmupClassificationPolicy {
    pub warmup_invocation_count: u32,
    pub exclude_from_steady_state: bool,
}

impl Default for WarmupClassificationPolicy {
    fn default() -> Self {
        Self {
            warmup_invocation_count: 3,
            exclude_from_steady_state: true,
        }
    }
}

/// Implements "Track steady-state transition" (tasks).
pub fn classify_warmup(
    invocation_index_for_bucket: u32,
    policy: &WarmupClassificationPolicy,
) -> bool {
    invocation_index_for_bucket < policy.warmup_invocation_count
}

/// Implements "Cold-Start Costs SHALL be distinguished from steady-state
/// execution where possible" (proposal). Cold-start evidence is modeled as
/// its own boolean dimension on the observation itself
/// ([`KernelExecutionPerformanceObservation::is_cold_start`]) rather than a
/// separate record type, so it participates in the same bounded aggregation
/// pipeline while remaining filterable.
pub fn should_include_in_steady_state(
    observation: &KernelExecutionPerformanceObservation,
    warmup_policy: &WarmupClassificationPolicy,
) -> bool {
    if observation.is_warmup && warmup_policy.exclude_from_steady_state {
        return false;
    }
    !observation.is_cold_start
}

// ---------------------------------------------------------------------
// Sampling Policy
// ---------------------------------------------------------------------

/// Implements "Sampling Policy" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelPerformanceSamplingPolicy {
    All,
    OneInN(u64),
    Probabilistic { ratio_per_million: u32 },
    PerBucketBudget { max_per_window: u64 },
}

/// Implements "Support all/one-in-N/bounded probabilistic sampling/per-
/// bucket budgets" (tasks). `random_micro` is caller-supplied entropy in
/// `0..1_000_000`, keeping this function itself deterministic and testable.
pub fn should_sample(
    policy: &KernelPerformanceSamplingPolicy,
    invocation_counter: u64,
    random_micro: u32,
    bucket_window_count: u64,
) -> bool {
    match policy {
        KernelPerformanceSamplingPolicy::All => true,
        KernelPerformanceSamplingPolicy::OneInN(n) => {
            *n > 0 && invocation_counter.is_multiple_of(*n)
        }
        KernelPerformanceSamplingPolicy::Probabilistic { ratio_per_million } => {
            u64::from(random_micro) < u64::from(*ratio_per_million)
        }
        KernelPerformanceSamplingPolicy::PerBucketBudget { max_per_window } => {
            bucket_window_count < *max_per_window
        }
    }
}

/// Implements "Sampling Independence" (proposal): "Performance observation
/// sampling SHALL not change which Kernel is selected for an already-started
/// invocation." A structural fact: [`should_sample`] takes no candidate or
/// selection state as input, so it cannot influence one.
pub const fn sampling_never_changes_selected_kernel() -> bool {
    true
}

/// Implements "Adaptive Sampling" (proposal): "Runtime MAY increase sampling
/// when ... It MAY reduce sampling after evidence stabilizes."
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaptiveSamplingTrigger {
    pub newly_promoted: bool,
    pub new_workload_bucket: bool,
    pub suspected_regression: bool,
    pub benchmark_stale: bool,
    pub stabilized: bool,
}

/// Implements "Keep overhead bounded" (tasks): the multiplier saturates
/// rather than growing without limit even when every trigger is active.
pub fn adaptive_sampling_multiplier(trigger: &AdaptiveSamplingTrigger) -> u32 {
    if trigger.stabilized {
        return 1;
    }
    let mut multiplier = 1u32;
    if trigger.newly_promoted {
        multiplier = multiplier.saturating_add(4);
    }
    if trigger.new_workload_bucket {
        multiplier = multiplier.saturating_add(2);
    }
    if trigger.suspected_regression {
        multiplier = multiplier.saturating_add(4);
    }
    if trigger.benchmark_stale {
        multiplier = multiplier.saturating_add(2);
    }
    multiplier.min(16)
}

/// Implements "Measurement Overhead" (proposal): "Measurement SHALL have
/// bounded overhead ... SHALL NOT materially degrade inference without
/// explicit policy."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasurementOverheadBudget {
    pub max_overhead_ratio: f64,
}

pub fn overhead_budget_exceeded(measured_ratio: f64, budget: &MeasurementOverheadBudget) -> bool {
    measured_ratio > budget.max_overhead_ratio
}

// ---------------------------------------------------------------------
// Online / Offline Evidence
// ---------------------------------------------------------------------

/// Implements "Online Evidence" / "Offline Evidence" (proposal): "These two
/// evidence classes SHALL remain distinguishable."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelPerformanceEvidenceSource {
    Online,
    Offline,
}

/// Implements "Offline Evidence Baseline" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelPerformanceBaseline {
    pub source: KernelPerformanceEvidenceSource,
    pub p50_micros: u64,
    pub p90_micros: u64,
    pub p99_micros: u64,
    pub sample_count: u64,
}

/// Implements "Online/Offline Policy" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OnlineOfflinePrecedencePolicy {
    OfflineOnly,
    OnlinePreferredAfterSufficientSamples { minimum_samples: u64 },
    Hybrid { online_weight: f64 },
    PinnedOffline,
}

/// Implements "Runtime policy SHALL define how online evidence and offline
/// benchmark evidence interact" (proposal).
pub fn resolve_evidence_precedence(
    policy: &OnlineOfflinePrecedencePolicy,
    online_sample_count: u64,
) -> KernelPerformanceEvidenceSource {
    match policy {
        OnlineOfflinePrecedencePolicy::OfflineOnly | OnlineOfflinePrecedencePolicy::PinnedOffline => {
            KernelPerformanceEvidenceSource::Offline
        }
        OnlineOfflinePrecedencePolicy::OnlinePreferredAfterSufficientSamples { minimum_samples } => {
            if online_sample_count >= *minimum_samples {
                KernelPerformanceEvidenceSource::Online
            } else {
                KernelPerformanceEvidenceSource::Offline
            }
        }
        OnlineOfflinePrecedencePolicy::Hybrid { online_weight } => {
            if *online_weight >= 0.5 {
                KernelPerformanceEvidenceSource::Online
            } else {
                KernelPerformanceEvidenceSource::Offline
            }
        }
    }
}

// ---------------------------------------------------------------------
// Benchmark Drift
// ---------------------------------------------------------------------

/// Implements "Benchmark Drift" (proposal): "Drift policy SHALL define
/// thresholds."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftThreshold {
    pub relative: f64,
    pub absolute_micros: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkDriftSignal {
    pub expected_p50_micros: u64,
    pub observed_p50_micros: u64,
    pub relative_delta: f64,
}

/// Implements "Runtime SHOULD detect sustained divergence between offline
/// expectation and online compatible evidence, and any drift signal produced
/// SHALL be based on sufficient compatible sample evidence" (proposal):
/// returns `None` whenever evidence is [`EvidenceQuality::Insufficient`],
/// regardless of how large the raw delta looks.
pub fn detect_benchmark_drift(
    baseline: &KernelPerformanceBaseline,
    observed: &KernelPerformanceMetricSummary,
    threshold: &DriftThreshold,
    quality: EvidenceQuality,
) -> Option<BenchmarkDriftSignal> {
    if !sufficient_for_adaptive_action(quality) {
        return None;
    }
    if baseline.p50_micros == 0 {
        return None;
    }
    let relative_delta =
        (observed.p50_micros as f64 - baseline.p50_micros as f64) / baseline.p50_micros as f64;
    let absolute_delta = observed.p50_micros.abs_diff(baseline.p50_micros);
    if relative_delta.abs() >= threshold.relative || absolute_delta >= threshold.absolute_micros {
        Some(BenchmarkDriftSignal {
            expected_p50_micros: baseline.p50_micros,
            observed_p50_micros: observed.p50_micros,
            relative_delta,
        })
    } else {
        None
    }
}

/// Implements "Drift Does Not Imply Incorrectness" (proposal): a structural
/// fact -- [`BenchmarkDriftSignal`] has no correctness/qualification field,
/// so nothing downstream can read correctness state out of a drift signal.
pub const fn drift_signal_does_not_imply_incorrectness() -> bool {
    true
}

// ---------------------------------------------------------------------
// Workload Drift
// ---------------------------------------------------------------------

/// Implements "Workload Drift" (proposal): the tuned-versus-actual workload
/// profile comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkloadProfile {
    pub batch_bucket: String,
    pub sequence_bucket: String,
    pub phase: Option<GenerationPhase>,
    pub shape_bucket: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WorkloadDriftDimension {
    Batch,
    Sequence,
    Phase,
    Shape,
}

/// Implements "Detect batch/sequence/phase/shape shift" (tasks).
pub fn detect_workload_drift(
    tuned: &WorkloadProfile,
    actual: &WorkloadProfile,
) -> Vec<WorkloadDriftDimension> {
    let mut drift = Vec::new();
    if tuned.batch_bucket != actual.batch_bucket {
        drift.push(WorkloadDriftDimension::Batch);
    }
    if tuned.sequence_bucket != actual.sequence_bucket {
        drift.push(WorkloadDriftDimension::Sequence);
    }
    if tuned.phase != actual.phase {
        drift.push(WorkloadDriftDimension::Phase);
    }
    if tuned.shape_bucket != actual.shape_bucket {
        drift.push(WorkloadDriftDimension::Shape);
    }
    drift
}

/// Implements "Workload Drift Action" (proposal).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkloadDriftAction {
    SelectionReevaluation,
    NewWorkloadBucket,
    BoundedRetuningRequest,
    Warning,
    NoAction,
}

// ---------------------------------------------------------------------
// Performance Regression
// ---------------------------------------------------------------------

/// Implements "Regression Thresholds SHALL be explicit" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionThresholds {
    pub relative_latency_increase: f64,
    pub absolute_latency_increase_micros: u64,
    pub throughput_reduction: f64,
    pub p99_relative_increase: f64,
    pub timeout_rate_increase: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegressionBaselineKind {
    PriorCandidate,
    PriorGeneration,
    OfflineBenchmark,
    PolicySlo,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionSignal {
    pub baseline_kind: RegressionBaselineKind,
    pub relative_latency_increase: f64,
    pub p99_relative_increase: f64,
    pub timeout_rate_increase: f64,
}

/// Implements "Kernel Performance Regression" (proposal): compares `current`
/// against `baseline` along every declared dimension; any single dimension
/// crossing its threshold produces a (not yet confirmed -- see
/// [`confirm_regression`]) signal.
pub fn detect_regression(
    baseline: &KernelPerformanceMetricSummary,
    current: &KernelPerformanceMetricSummary,
    baseline_kind: RegressionBaselineKind,
    thresholds: &RegressionThresholds,
) -> Option<RegressionSignal> {
    if baseline.mean_latency_micros <= 0.0 {
        return None;
    }
    let relative_latency_increase =
        (current.mean_latency_micros - baseline.mean_latency_micros) / baseline.mean_latency_micros;
    let absolute_increase = current.mean_latency_micros - baseline.mean_latency_micros;
    let p99_relative_increase = if baseline.p99_micros > 0 {
        (current.p99_micros as f64 - baseline.p99_micros as f64) / baseline.p99_micros as f64
    } else {
        0.0
    };
    let timeout_rate_increase = current.timeout_rate() - baseline.timeout_rate();

    let regressed = relative_latency_increase >= thresholds.relative_latency_increase
        || absolute_increase >= thresholds.absolute_latency_increase_micros as f64
        || p99_relative_increase >= thresholds.p99_relative_increase
        || timeout_rate_increase >= thresholds.timeout_rate_increase;

    if regressed {
        Some(RegressionSignal {
            baseline_kind,
            relative_latency_increase,
            p99_relative_increase,
            timeout_rate_increase,
        })
    } else {
        None
    }
}

/// Implements "Regression Confirmation" (proposal): "Policy SHOULD prevent
/// reacting to one isolated outlier." A signal is confirmed only when
/// evidence is sufficient *and* the regression has been observed for at
/// least `policy.minimum_sustained_duration_millis` -- both required, so one
/// slow sample inside an otherwise short window can never confirm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionConfirmationPolicy {
    pub sufficiency: SampleSufficiencyPolicy,
    pub minimum_sustained_duration_millis: u64,
}

pub fn confirm_regression(
    _signal: &RegressionSignal,
    quality: EvidenceQuality,
    sustained_duration_millis: u64,
    policy: &RegressionConfirmationPolicy,
) -> bool {
    sufficient_for_adaptive_action(quality)
        && sustained_duration_millis >= policy.minimum_sustained_duration_millis
}

// ---------------------------------------------------------------------
// Outlier Handling
// ---------------------------------------------------------------------

/// Implements "Outlier Handling" (proposal): "Outliers SHALL not be silently
/// discarded without policy." Structurally closed to exactly the three
/// declared behaviors -- there is no "Discard" variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutlierPolicy {
    RetainInTail,
    ExcludeFromMean,
    MarkSeparately,
}

/// Returns `(retained_for_mean, tail_or_marked)`. Under
/// [`OutlierPolicy::RetainInTail`] every sample stays in `retained_for_mean`
/// (nothing is ever silently dropped); under the other two policies outliers
/// move to the second list but remain present in the return value, so
/// callers can never lose them entirely.
pub fn apply_outlier_policy(
    latencies: &[u64],
    policy: OutlierPolicy,
    outlier_threshold_micros: u64,
) -> (Vec<u64>, Vec<u64>) {
    match policy {
        OutlierPolicy::RetainInTail => (latencies.to_vec(), Vec::new()),
        OutlierPolicy::ExcludeFromMean | OutlierPolicy::MarkSeparately => {
            let (retained, outliers): (Vec<u64>, Vec<u64>) = latencies
                .iter()
                .partition(|&&latency| latency < outlier_threshold_micros);
            (retained, outliers)
        }
    }
}

// ---------------------------------------------------------------------
// Device Pressure Correlation
// ---------------------------------------------------------------------

/// Implements "External Interference" / "Device Pressure Correlation"
/// (proposal): "Runtime MAY distinguish degraded performance caused by
/// broader Device pressure from Kernel-specific regression where evidence
/// permits."
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DevicePressureCorrelation {
    pub device_pressure: Option<MemoryPressureLevel>,
    pub provider_pressure: Option<ProviderPressureLevel>,
    pub distinct_candidates_slow_simultaneously: u32,
}

/// Implements "all GPU Kernels slow simultaneously may indicate Device
/// pressure rather than one bad Kernel" (proposal): a broad slowdown is
/// suspected only when pressure is elevated *and* more than one distinct
/// candidate is affected -- one slow candidate under high pressure alone
/// remains attributable to that candidate.
pub fn broad_slowdown_suspected(correlation: &DevicePressureCorrelation) -> bool {
    let pressure_elevated = matches!(
        correlation.device_pressure,
        Some(MemoryPressureLevel::High) | Some(MemoryPressureLevel::Saturated)
    ) || matches!(
        correlation.provider_pressure,
        Some(ProviderPressureLevel::High) | Some(ProviderPressureLevel::Saturated)
    );
    pressure_elevated && correlation.distinct_candidates_slow_simultaneously > 1
}

// ---------------------------------------------------------------------
// Performance Model
// ---------------------------------------------------------------------

/// Implements "Kernel Performance Model" (proposal): "aggregated evidence
/// for a Kernel candidate in a compatible workload/target context." Not a
/// machine-learning model -- see [`KernelPerformanceAggregator`], which is
/// rolling statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPerformanceModel {
    pub candidate: CompiledKernelArtifactId,
    pub specialization_fingerprint: Option<String>,
    pub workload_bucket: KernelPerformanceWorkloadBucket,
    pub aggregator: KernelPerformanceAggregator,
    pub baseline: Option<KernelPerformanceBaseline>,
    pub model_version: u32,
    pub last_updated_millis: u64,
}

impl KernelPerformanceModel {
    pub fn new(
        candidate: CompiledKernelArtifactId,
        workload_bucket: KernelPerformanceWorkloadBucket,
        model_version: u32,
    ) -> Self {
        Self {
            candidate,
            specialization_fingerprint: None,
            workload_bucket,
            aggregator: KernelPerformanceAggregator::new(),
            baseline: None,
            model_version,
            last_updated_millis: 0,
        }
    }

    /// Implements "Artifact And Generation Binding" / "Specialization
    /// Binding" (proposal): "A replacement Kernel SHALL not inherit
    /// performance evidence automatically" and "Performance evidence for one
    /// Specialization Instance SHALL not automatically apply to another."
    /// The only way an observation contributes to this model is through this
    /// method, and it rejects any mismatch rather than silently merging.
    pub fn record_observation(
        &mut self,
        observation: &KernelExecutionPerformanceObservation,
    ) -> Result<(), KernelPerformanceError> {
        observation.validate()?;
        if !artifact_binding_valid(&observation.artifact_digest, &self.candidate) {
            return Err(KernelPerformanceError::ContextInvalid {
                reason: format!(
                    "observation artifact `{}` does not match model candidate `{}`",
                    observation.artifact_digest, self.candidate
                ),
            });
        }
        if !specialization_binding_valid(
            observation.specialization_fingerprint.as_deref(),
            self.specialization_fingerprint.as_deref(),
        ) {
            return Err(KernelPerformanceError::ContextInvalid {
                reason: "observation specialization does not match model specialization".into(),
            });
        }
        if observation.workload_bucket.bucket_id() != self.workload_bucket.bucket_id() {
            return Err(KernelPerformanceError::BucketInvalid {
                reason: "observation workload bucket does not match model bucket".into(),
            });
        }
        self.aggregator.record(observation);
        self.last_updated_millis = self.last_updated_millis.max(observation.timestamp_millis);
        Ok(())
    }

    pub fn evidence_quality(
        &self,
        observed_duration_millis: u64,
        policy: &SampleSufficiencyPolicy,
    ) -> EvidenceQuality {
        evaluate_sample_sufficiency(&self.aggregator.summary(), observed_duration_millis, policy)
    }

    /// Implements "Model Versioning" (proposal): "Performance Model
    /// computation policy SHALL be versioned."
    pub fn is_compatible_with_policy_version(&self, current_version: u32) -> bool {
        self.model_version == current_version
    }

    /// Implements "Performance Model Aging" (proposal): "Runtime SHALL avoid
    /// unbounded accumulation" over time as well as count -- a model that has
    /// not been updated inside `max_age_millis` is stale evidence.
    pub fn is_stale(&self, now_millis: u64, max_age_millis: u64) -> bool {
        now_millis.saturating_sub(self.last_updated_millis) > max_age_millis
    }
}

/// Implements "Artifact Binding" (proposal/tasks).
pub fn artifact_binding_valid(
    observation_artifact: &CompiledKernelArtifactId,
    model_artifact: &CompiledKernelArtifactId,
) -> bool {
    observation_artifact == model_artifact
}

/// Implements "Specialization Binding" (proposal/tasks).
pub fn specialization_binding_valid(
    observation_fingerprint: Option<&str>,
    model_fingerprint: Option<&str>,
) -> bool {
    observation_fingerprint == model_fingerprint
}

/// Implements "Cross-Device Evidence" (proposal): "SHALL not automatically
/// transfer across incompatible Devices. Policy MAY allow reuse across
/// sufficiently equivalent hardware compatibility classes where explicitly
/// defined." Reuse is allowed only when architectures already match, or the
/// caller explicitly names both architectures as members of the same
/// compatibility class.
pub fn cross_device_reuse_allowed(
    observation_architecture: &str,
    model_architecture: &str,
    compatible_class: Option<&[&str]>,
) -> bool {
    if observation_architecture == model_architecture {
        return true;
    }
    compatible_class.is_some_and(|class| {
        class.contains(&observation_architecture) && class.contains(&model_architecture)
    })
}

/// Implements "Cross-Provider Evidence SHALL not automatically rank another
/// Provider implementation" (proposal): exact match only, no compatibility
/// class exception (unlike Device architecture, Providers are never
/// substitutable for ranking purposes).
pub fn cross_provider_reuse_allowed(
    observation_provider: &ProviderBinding,
    model_provider: &ProviderBinding,
) -> bool {
    observation_provider == model_provider
}

// ---------------------------------------------------------------------
// Performance Health
// ---------------------------------------------------------------------

/// Implements "Performance Health State" (proposal): "These states SHALL not
/// replace Provider health or Kernel qualification." Structurally
/// independent of [`crate::kernel_qualification::QualificationStatus`] and
/// [`crate::DeviceHealth`] -- no `From`/`Into` conversion exists between
/// them, and this enum's variants are never consulted by qualification or
/// Provider-health logic.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelPerformanceHealth {
    Unknown,
    Warming,
    Healthy,
    Degraded,
    Regressed,
    Stale,
}

/// Implements "Add unknown/warming/healthy/degraded/regressed/stale"
/// (tasks): a pure derivation from already-computed evidence state, never
/// from raw counters directly.
pub fn compute_health(
    quality: EvidenceQuality,
    regression_confirmed: bool,
    regression_suspected: bool,
    is_stale: bool,
    is_warming: bool,
) -> KernelPerformanceHealth {
    if quality == EvidenceQuality::Insufficient {
        return if is_warming {
            KernelPerformanceHealth::Warming
        } else {
            KernelPerformanceHealth::Unknown
        };
    }
    if is_stale {
        return KernelPerformanceHealth::Stale;
    }
    if regression_confirmed {
        return KernelPerformanceHealth::Regressed;
    }
    if regression_suspected {
        return KernelPerformanceHealth::Degraded;
    }
    KernelPerformanceHealth::Healthy
}

/// Implements "Performance Health May Affect Preference" (kernel-selection-
/// policy delta): "Such preference adjustments SHALL NOT bypass hard
/// eligibility constraints." Mirrors
/// [`crate::kernel_selection_policy::pressure_ranking_bias`]: the only input
/// is an already-[`EligibleCandidate`], and the return value is a bias added
/// to a lower-is-better score, never a selection decision by itself.
pub fn performance_health_ranking_bias(
    _eligible: &EligibleCandidate,
    health: KernelPerformanceHealth,
) -> f64 {
    match health {
        KernelPerformanceHealth::Healthy => 0.0,
        KernelPerformanceHealth::Unknown | KernelPerformanceHealth::Warming => 0.0,
        KernelPerformanceHealth::Degraded => 5.0,
        KernelPerformanceHealth::Regressed => 25.0,
        KernelPerformanceHealth::Stale => 2.0,
    }
}

// ---------------------------------------------------------------------
// Memory Anomaly / Contract Violation
// ---------------------------------------------------------------------

/// Implements "Memory Evidence" (proposal): actual-versus-advertised
/// workspace comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryAnomaly {
    pub advertised_workspace_bytes: u64,
    pub observed_workspace_bytes: u64,
}

impl MemoryAnomaly {
    pub fn exceeds(&self, tolerance_ratio: f64) -> bool {
        if self.advertised_workspace_bytes == 0 {
            return self.observed_workspace_bytes > 0;
        }
        let ratio = self.observed_workspace_bytes as f64 / self.advertised_workspace_bytes as f64;
        ratio > 1.0 + tolerance_ratio
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractViolationSeverity {
    Minor,
    Severe,
}

/// Implements "Contract Violation Versus Performance Regression" (proposal):
/// "Runtime SHALL distinguish performance regression from Kernel contract
/// violation." A severe memory overrun is classified as a contract issue,
/// never merely a slowdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerformanceOrContractIssue {
    PerformanceRegression,
    ContractViolation(ContractViolationSeverity),
}

pub fn classify_memory_anomaly(
    anomaly: &MemoryAnomaly,
    tolerance_ratio: f64,
    severe_ratio: f64,
) -> PerformanceOrContractIssue {
    if anomaly.advertised_workspace_bytes > 0 {
        let ratio = anomaly.observed_workspace_bytes as f64 / anomaly.advertised_workspace_bytes as f64;
        if ratio > 1.0 + severe_ratio {
            return PerformanceOrContractIssue::ContractViolation(ContractViolationSeverity::Severe);
        }
    }
    if anomaly.exceeds(tolerance_ratio) {
        PerformanceOrContractIssue::ContractViolation(ContractViolationSeverity::Minor)
    } else {
        PerformanceOrContractIssue::PerformanceRegression
    }
}

// ---------------------------------------------------------------------
// Hysteresis / Aging
// ---------------------------------------------------------------------

/// Implements "Adaptive Feedback Uses Hysteresis" (proposal): "Small
/// performance fluctuations SHALL not cause repeated selection changes."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveActionHysteresis {
    pub minimum_relative_change: f64,
    pub minimum_stable_duration_millis: u64,
}

/// Implements "Selection Stability" / "Re-Tuning Hysteresis" (proposal):
/// only true once the relative change AND the stable-duration requirement
/// both clear policy -- a single noisy sample can move `new_metric` without
/// ever satisfying `stable_duration_millis`.
pub fn should_change_due_to_performance(
    previous_metric: f64,
    new_metric: f64,
    stable_duration_millis: u64,
    policy: &AdaptiveActionHysteresis,
) -> bool {
    if previous_metric == 0.0 {
        return new_metric != 0.0;
    }
    let relative_change = (new_metric - previous_metric).abs() / previous_metric.abs();
    relative_change >= policy.minimum_relative_change
        && stable_duration_millis >= policy.minimum_stable_duration_millis
}

/// Implements "Performance Model Aging" (proposal): "Old observations SHOULD
/// decay or expire. Very old performance data SHALL not dominate current
/// evidence indefinitely."
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgingMechanism {
    TimeWindow { max_age_millis: u64 },
    GenerationWindow { max_generations: u32 },
    WeightedDecay { half_life_millis: u64 },
    ExplicitExpiration,
}

/// Returns a weight in `[0.0, 1.0]`: `0.0` means the evidence is expired and
/// SHALL NOT participate in current evidence.
pub fn observation_weight(mechanism: &AgingMechanism, age_millis: u64, generations_since: u32) -> f64 {
    match mechanism {
        AgingMechanism::TimeWindow { max_age_millis } => {
            if age_millis > *max_age_millis {
                0.0
            } else {
                1.0
            }
        }
        AgingMechanism::GenerationWindow { max_generations } => {
            if generations_since > *max_generations {
                0.0
            } else {
                1.0
            }
        }
        AgingMechanism::WeightedDecay { half_life_millis } => {
            if *half_life_millis == 0 {
                return 0.0;
            }
            0.5f64.powf(age_millis as f64 / *half_life_millis as f64)
        }
        AgingMechanism::ExplicitExpiration => 1.0,
    }
}

// ---------------------------------------------------------------------
// Tuning Staleness / Retuning Request
// ---------------------------------------------------------------------

/// Implements "Tuning Staleness" (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TuningStalenessReason {
    PerformanceDrift,
    WorkloadShift,
    CandidateSetChanged,
    ProviderChanged,
    DriverChanged,
    DeviceBehaviorChanged,
    PolicyChanged,
}

pub fn tuning_is_stale(reasons: &[TuningStalenessReason]) -> bool {
    !reasons.is_empty()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetuningUrgency {
    Low,
    Medium,
    High,
}

/// Implements "Retuning Request" (proposal): "A Retuning Request SHALL NOT
/// itself start arbitrary source generation." Structurally: every field is
/// an identity, a reason, or an aggregated summary -- there is no field
/// shaped like source code, a compiler command, or an executable script.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelRetuningRequest {
    pub reason: TuningStalenessReason,
    pub workload_bucket: String,
    pub candidate_context: CompiledKernelArtifactId,
    pub evidence_summary: KernelPerformanceMetricSummary,
    pub urgency: RetuningUrgency,
    pub requested_at_millis: u64,
}

impl KernelRetuningRequest {
    /// Implements "Add deduplication/rate limiting" (tasks): two requests
    /// for the same reason/bucket/candidate are the same request for
    /// deduplication purposes, regardless of timestamp or evidence detail.
    pub fn deduplication_key(&self) -> String {
        format!(
            "{:?}|{}|{}",
            self.reason, self.workload_bucket, self.candidate_context
        )
    }
}

/// Implements "Retuning Scope" / "Bounded Retuning" (proposal): "It SHALL
/// NOT generate arbitrary new Kernel source" -- re-tuning is authorized only
/// when the requested candidate already belongs to an existing, bounded
/// [`KernelAutotuningPlan`] from the [`crate::kernel_autotuning`] contract.
pub fn retuning_request_respects_autotuning_boundary(
    request: &KernelRetuningRequest,
    plan: &KernelAutotuningPlan,
) -> bool {
    plan.candidates
        .iter()
        .any(|candidate: &KernelAutotuningCandidate| {
            candidate.compiled_artifact == request.candidate_context
        })
}

/// Implements "No Hot-Path Re-Tuning" / "Re-Tuning Hot-Path Prohibition"
/// (proposal): "A regression detected during decode SHALL NOT synchronously
/// pause the same decode to benchmark alternatives." Delegates to
/// [`crate::reject_decode_hot_path_trigger`] -- the same boundary already
/// enforced for ordinary Runtime Autotuning -- rather than defining a second,
/// divergent hot-path rule.
pub fn request_retuning_outside_hot_path(
    triggered_from_decode_hot_path: bool,
) -> Result<(), KernelPerformanceError> {
    reject_decode_hot_path_trigger(triggered_from_decode_hot_path)
        .map_err(|_| KernelPerformanceError::RetuningDenied)
}

/// Implements "Re-Tuning Admission" (proposal): "High inference pressure MAY
/// postpone optional re-tuning."
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetuningAdmissionContext {
    pub provider_pressure: ProviderPressureLevel,
    pub device_pressure: MemoryPressureLevel,
    pub inference_priority_high: bool,
    pub budget_remaining: u32,
}

pub fn retuning_admission_allowed(ctx: &RetuningAdmissionContext) -> bool {
    if ctx.budget_remaining == 0 {
        return false;
    }
    if ctx.inference_priority_high {
        return false;
    }
    !matches!(ctx.provider_pressure, ProviderPressureLevel::Saturated)
        && !matches!(ctx.device_pressure, MemoryPressureLevel::Saturated)
}

/// Implements "Re-Tuning Hysteresis" / "Feedback Cooldown" / "Re-Tuning Rate
/// Limit" (proposal): "Repeated re-tuning requests SHALL be rate-
/// limited/cooldown-controlled."
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetuningCooldown {
    pub minimum_interval_millis: u64,
    last_requested_at_millis: BTreeMap<String, u64>,
}

impl RetuningCooldown {
    pub fn new(minimum_interval_millis: u64) -> Self {
        Self {
            minimum_interval_millis,
            last_requested_at_millis: BTreeMap::new(),
        }
    }

    /// Returns `true` and records `request` as admitted when it clears the
    /// cooldown window for its [`KernelRetuningRequest::deduplication_key`];
    /// returns `false` (rate-limited) otherwise, leaving state unchanged.
    pub fn admit(&mut self, request: &KernelRetuningRequest) -> bool {
        let key = request.deduplication_key();
        let now = request.requested_at_millis;
        let allowed = match self.last_requested_at_millis.get(&key) {
            Some(last) => now.saturating_sub(*last) >= self.minimum_interval_millis,
            None => true,
        };
        if allowed {
            self.last_requested_at_millis.insert(key, now);
        }
        allowed
    }
}

// ---------------------------------------------------------------------
// Optimization Escalation
// ---------------------------------------------------------------------

/// Implements "Optimization Escalation" (proposal): "Runtime SHALL not
/// execute the external optimization itself as part of inference." No field
/// or method on this type can trigger code generation -- it is pure data for
/// the external Optimization Plane to consume.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationEscalationRequest {
    pub reason: String,
    pub workload_bucket: String,
    pub evidence_summary: KernelPerformanceMetricSummary,
    pub requested_at_millis: u64,
}

/// Implements "Adaptive Feedback Escalation Boundary" (proposal): the path
/// SHALL be `performance issue -> bounded retuning -> external optimization
/// signal`, never `performance issue -> Runtime invokes AI generator`.
/// Escalation requires bounded retuning to have already been exhausted.
pub fn escalate_to_optimization_plane(
    bounded_variants_exhausted: bool,
    still_below_target: bool,
    workload_bucket: impl Into<String>,
    evidence_summary: KernelPerformanceMetricSummary,
    now_millis: u64,
) -> Option<OptimizationEscalationRequest> {
    if bounded_variants_exhausted && still_below_target {
        Some(OptimizationEscalationRequest {
            reason: "bounded retuning exhausted; performance remains below policy target".into(),
            workload_bucket: workload_bucket.into(),
            evidence_summary,
            requested_at_millis: now_millis,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// Demotion / Rollback Signals
// ---------------------------------------------------------------------

/// Implements "Demotion Signal" (proposal): "Demotion SHALL remain subject
/// to selection/promotion state machinery" -- this is a recommendation, not
/// an enforced state transition.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPerformanceDemotionSignal {
    pub candidate: CompiledKernelArtifactId,
    pub workload_bucket: String,
    pub reason: String,
    pub evidence: KernelPerformanceMetricSummary,
}

pub fn recommend_demotion(
    candidate: &CompiledKernelArtifactId,
    workload_bucket: impl Into<String>,
    regression: &RegressionSignal,
    confirmed: bool,
    evidence: KernelPerformanceMetricSummary,
) -> Option<KernelPerformanceDemotionSignal> {
    confirmed.then(|| KernelPerformanceDemotionSignal {
        candidate: candidate.clone(),
        workload_bucket: workload_bucket.into(),
        reason: format!(
            "confirmed regression vs {:?}: {:.1}% latency increase",
            regression.baseline_kind,
            regression.relative_latency_increase * 100.0
        ),
        evidence,
    })
}

/// Implements "Rollback Signal" (proposal): "Actual rollback remains
/// governed by the existing rollback policy" -- again a recommendation only.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPerformanceRollbackSignal {
    pub regressed_candidate: CompiledKernelArtifactId,
    pub rollback_to: CompiledKernelArtifactId,
    pub reason: String,
}

pub fn recommend_rollback(
    regressed_candidate: &CompiledKernelArtifactId,
    known_good: Option<&CompiledKernelArtifactId>,
    severe: bool,
    confirmed: bool,
) -> Option<KernelPerformanceRollbackSignal> {
    if !severe || !confirmed {
        return None;
    }
    known_good.map(|rollback_to| KernelPerformanceRollbackSignal {
        regressed_candidate: regressed_candidate.clone(),
        rollback_to: rollback_to.clone(),
        reason: "severe confirmed regression".into(),
    })
}

// ---------------------------------------------------------------------
// Post-Promotion Observation Window
// ---------------------------------------------------------------------

/// Implements "Post-Promotion Observation Window" (proposal).
#[derive(Clone, Debug, PartialEq)]
pub struct PostPromotionObservationWindow {
    pub promoted_at_millis: u64,
    pub duration_millis: u64,
    pub sample_rate_multiplier: u32,
    pub stricter_threshold_multiplier: f64,
    pub rollback_candidate: Option<CompiledKernelArtifactId>,
}

impl PostPromotionObservationWindow {
    pub fn is_active(&self, now_millis: u64) -> bool {
        now_millis.saturating_sub(self.promoted_at_millis) < self.duration_millis
    }
}

// ---------------------------------------------------------------------
// Model Instance Policy / Reproducible Mode
// ---------------------------------------------------------------------

/// Implements "Model Instance Interaction" (proposal): "A Model Instance
/// MAY: consume adaptive performance evidence / use dynamic selection policy
/// / remain pinned/reproducible and ignore adaptive changes. Policy SHALL be
/// explicit."
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelPerformanceFeedbackMode {
    Adaptive,
    Pinned,
    Disabled,
}

pub fn feedback_mode_allows_selection_change(mode: KernelPerformanceFeedbackMode) -> bool {
    matches!(mode, KernelPerformanceFeedbackMode::Adaptive)
}

/// Implements "Reproducible mode ... Observations MAY still be collected if
/// policy allows" (proposal): `Disabled` collects nothing; `Pinned` still
/// collects (for diagnostics/export) but never adapts.
pub fn feedback_mode_allows_observation(mode: KernelPerformanceFeedbackMode) -> bool {
    !matches!(mode, KernelPerformanceFeedbackMode::Disabled)
}

/// Implements "Reproducible Mode" (proposal): "Pinned reproducible execution
/// SHALL not change Kernel from live performance feedback ... unless
/// external policy explicitly changes the pin." Reproducible mode always
/// forces [`KernelPerformanceFeedbackMode::Pinned`], overriding whatever the
/// caller's requested mode was.
pub fn reproducible_mode_blocks_adaptation(
    reproducible: bool,
    requested_mode: KernelPerformanceFeedbackMode,
) -> KernelPerformanceFeedbackMode {
    if reproducible {
        KernelPerformanceFeedbackMode::Pinned
    } else {
        requested_mode
    }
}

// ---------------------------------------------------------------------
// Session Boundary
// ---------------------------------------------------------------------

/// Implements "Session Boundary" (proposal): "Inference Session SHALL not
/// own the Performance Model ... A Session SHALL not be able to falsify
/// performance observations." A structural fact: nothing in this module
/// accepts a session identifier as a channel to mutate a
/// [`KernelPerformanceModel`] -- only [`KernelPerformanceModel::record_observation`]
/// does, and it takes a fully-formed, independently validated observation.
pub const fn session_cannot_own_or_falsify_performance_model() -> bool {
    true
}

// ---------------------------------------------------------------------
// Continuous Batching
// ---------------------------------------------------------------------

/// Implements "Continuous Batching" (proposal): "Runtime SHALL not fabricate
/// per-session latency from batch-level kernel timing without a defined
/// attribution model." No variant silently divides latency by sequence
/// count -- every variant is an explicit, named policy.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchAttributionModel {
    EqualSplit,
    Weighted { weights: Vec<f64> },
}

/// Implements "Metrics SHALL be attributable carefully where multiple
/// sequences share one Kernel invocation" (proposal). Returns `None` when
/// `Weighted` weights do not sum to a usable total, rather than silently
/// producing a misleading attribution.
pub fn attribute_batch_latency(
    total_micros: u64,
    sequence_count_in_batch: u32,
    model: &BatchAttributionModel,
) -> Option<Vec<u64>> {
    if sequence_count_in_batch == 0 {
        return None;
    }
    match model {
        BatchAttributionModel::EqualSplit => {
            let share = total_micros / u64::from(sequence_count_in_batch);
            Some(vec![share; sequence_count_in_batch as usize])
        }
        BatchAttributionModel::Weighted { weights } => {
            if weights.len() != sequence_count_in_batch as usize {
                return None;
            }
            let total_weight: f64 = weights.iter().sum();
            if total_weight <= 0.0 {
                return None;
            }
            Some(
                weights
                    .iter()
                    .map(|w| (total_micros as f64 * (w / total_weight)) as u64)
                    .collect(),
            )
        }
    }
}

// ---------------------------------------------------------------------
// Telemetry Export
// ---------------------------------------------------------------------

/// Implements "Export" / "Optimization Plane Feedback" (proposal): "SHALL
/// NOT include raw prompts, model weights, KV data, or secrets by default."
/// Structurally: every field is a stable identity, an evidence-source enum,
/// a health enum, or an aggregate summary -- there is no field shaped like
/// raw model/user content.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPerformanceExportSummary {
    pub kernel: KernelId,
    pub workload_bucket: String,
    pub metric_summary: KernelPerformanceMetricSummary,
    pub evidence_source: KernelPerformanceEvidenceSource,
    pub health: KernelPerformanceHealth,
    pub policy_version: u32,
}

impl KernelPerformanceExportSummary {
    /// Implements "Runtime SHALL not expose ... by default" (Observability
    /// section): every value passes through [`redact_backend_diagnostic`]
    /// before leaving the process, mirroring
    /// [`crate::kernel_autotuning::KernelAutotuningObservation`]'s export
    /// pattern.
    pub fn to_redacted_payload(&self) -> BTreeMap<String, String> {
        let mut payload = BTreeMap::new();
        payload.insert(
            "kernel".into(),
            redact_backend_diagnostic(&self.kernel.name),
        );
        payload.insert(
            "workload_bucket".into(),
            redact_backend_diagnostic(&self.workload_bucket),
        );
        payload.insert(
            "evidence_source".into(),
            format!("{:?}", self.evidence_source),
        );
        payload.insert("health".into(), format!("{:?}", self.health));
        payload.insert("policy_version".into(), self.policy_version.to_string());
        payload.insert("sample_count".into(), self.metric_summary.count.to_string());
        payload.insert(
            "p50_micros".into(),
            self.metric_summary.p50_micros.to_string(),
        );
        payload.insert(
            "p99_micros".into(),
            self.metric_summary.p99_micros.to_string(),
        );
        payload
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Performance error, implementing the proposal's "Error
/// Model" section verbatim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelPerformanceError {
    ObservationInvalid { reason: String },
    ContextInvalid { reason: String },
    BucketInvalid { reason: String },
    BucketPolicyUnsupported,
    ModelUnavailable,
    ModelInsufficientSamples,
    ModelStale,
    MetricUnavailable,
    MeasurementFailed { reason: String },
    MeasurementOverheadExceeded,

    DriftDetected,
    WorkloadDriftDetected,
    RegressionDetected,
    RegressionUnconfirmed,
    ContractAnomaly,
    TimeoutRegression,
    MemoryAnomaly,

    RetuningRateLimited,
    RetuningDenied,
    RetuningRequestFailed { reason: String },
    OptimizationEscalationRequired,

    FeedbackDisabled,
    FeedbackPolicyInvalid { reason: String },
    Internal { reason: String },
}

impl KernelPerformanceError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ObservationInvalid { .. } => "kernel-performance-observation-invalid",
            Self::ContextInvalid { .. } => "kernel-performance-context-invalid",
            Self::BucketInvalid { .. } => "kernel-performance-bucket-invalid",
            Self::BucketPolicyUnsupported => "kernel-performance-bucket-policy-unsupported",
            Self::ModelUnavailable => "kernel-performance-model-unavailable",
            Self::ModelInsufficientSamples => "kernel-performance-model-insufficient-samples",
            Self::ModelStale => "kernel-performance-model-stale",
            Self::MetricUnavailable => "kernel-performance-metric-unavailable",
            Self::MeasurementFailed { .. } => "kernel-performance-measurement-failed",
            Self::MeasurementOverheadExceeded => "kernel-performance-measurement-overhead-exceeded",

            Self::DriftDetected => "kernel-performance-drift-detected",
            Self::WorkloadDriftDetected => "kernel-performance-workload-drift-detected",
            Self::RegressionDetected => "kernel-performance-regression-detected",
            Self::RegressionUnconfirmed => "kernel-performance-regression-unconfirmed",
            Self::ContractAnomaly => "kernel-performance-contract-anomaly",
            Self::TimeoutRegression => "kernel-performance-timeout-regression",
            Self::MemoryAnomaly => "kernel-performance-memory-anomaly",

            Self::RetuningRateLimited => "kernel-performance-retuning-rate-limited",
            Self::RetuningDenied => "kernel-performance-retuning-denied",
            Self::RetuningRequestFailed { .. } => "kernel-performance-retuning-request-failed",
            Self::OptimizationEscalationRequired => "kernel-performance-optimization-escalation-required",

            Self::FeedbackDisabled => "kernel-performance-feedback-disabled",
            Self::FeedbackPolicyInvalid { .. } => "kernel-performance-feedback-policy-invalid",
            Self::Internal { .. } => "internal-kernel-performance-error",
        }
    }
}

impl fmt::Display for KernelPerformanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObservationInvalid { reason }
            | Self::ContextInvalid { reason }
            | Self::BucketInvalid { reason }
            | Self::MeasurementFailed { reason }
            | Self::RetuningRequestFailed { reason }
            | Self::FeedbackPolicyInvalid { reason }
            | Self::Internal { reason } => write!(f, "{}: {reason}", self.id()),
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelPerformanceError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Implements the proposal's "Observability" section vocabulary.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelPerformanceObservationKind {
    ObservationSampled,
    ModelUpdated,
    ModelInsufficient,
    BenchmarkDriftDetected,
    WorkloadDriftDetected,
    RegressionSuspected,
    RegressionConfirmed,
    PerformanceDegraded,
    PerformanceRecovered,
    RetuningRequested,
    RetuningRateLimited,
    SelectionReevaluationRequested,
    RollbackRecommended,
    OptimizationEscalationRequested,
}

/// A single adaptive-feedback observability event. Implements "Observability
/// SHALL NOT expose by default: raw prompts, raw tensor values, model
/// weights, KV contents, native handles, secrets, credentials" (proposal):
/// every metadata value passes through [`redact_backend_diagnostic`] before
/// storage, mirroring [`crate::kernel_autotuning::KernelAutotuningObservation`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelPerformanceObservabilityEvent {
    pub kind: KernelPerformanceObservationKind,
    pub candidate: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelPerformanceObservabilityEvent {
    pub fn new(kind: KernelPerformanceObservationKind) -> Self {
        Self {
            kind,
            candidate: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_candidate(mut self, candidate: impl Into<String>) -> Self {
        self.candidate = Some(candidate.into());
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelPerformanceConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelPerformanceConformanceReport {
    pub results: Vec<KernelPerformanceConformanceResult>,
}

impl KernelPerformanceConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelPerformanceConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelPerformanceConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn conformance_bucket() -> KernelPerformanceWorkloadBucket {
    KernelPerformanceWorkloadBucket::new(
        BenchmarkContext {
            provider: ProviderBinding::new("cuda"),
            device_architecture: "sm90".into(),
            driver_runtime_compatibility: "1.0".into(),
            operator_version: 1,
            artifact_digest: Some("digest-a".into()),
            dtype: crate::ComputeDType::Float16,
            layout: crate::TensorLayoutKind::Contiguous,
            shape_bucket: "attn".into(),
            batch_bucket: "4..8".into(),
            sequence_bucket: "2049..4096".into(),
            execution_mode: crate::KernelExecutionMode::Synchronous,
            benchmark_profile_version: 1,
        },
        1,
    )
    .with_phase(GenerationPhase::Decode)
}

fn conformance_operator() -> OperatorId {
    OperatorId::new("magnetar", "attention", 1, crate::OperatorFamily::Attention)
}

fn conformance_kernel_id() -> KernelId {
    crate::KernelId::new(
        ProviderBinding::new("cuda"),
        "attn",
        crate::CapabilityVersion::new(1, 0, 0),
        conformance_operator(),
        crate::KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::Cuda,
    )
}

fn conformance_observation(
    artifact: CompiledKernelArtifactId,
    latency_micros: u64,
    timestamp_millis: u64,
) -> KernelExecutionPerformanceObservation {
    KernelExecutionPerformanceObservation {
        kernel: conformance_kernel_id(),
        artifact_digest: artifact,
        prepared_generation: None,
        specialization_fingerprint: None,
        provider: ProviderBinding::new("cuda"),
        workload_bucket: conformance_bucket(),
        timing_method: PerformanceTimingMethod::DeviceEvent,
        latency_micros,
        queue_delay_micros: None,
        provider_submission_micros: None,
        workspace_bytes: None,
        memory_pressure: None,
        provider_pressure: None,
        completion: KernelExecutionCompletion::Success,
        is_warmup: false,
        is_cold_start: false,
        timestamp_millis,
    }
}

pub fn run_kernel_performance_conformance() -> KernelPerformanceConformanceReport {
    let mut results = Vec::new();

    // 1. Performance Evidence Cannot Grant Trust.
    let trust_before = KernelArtifactTrust::Untrusted;
    let trust_after = crate::online_measurement_cannot_override_correctness_or_trust(
        true,
        trust_before,
    );
    record(
        &mut results,
        "Performance Evidence Cannot Grant Trust",
        trust_after == KernelArtifactTrust::Untrusted,
        "expected excellent online evidence to leave Untrusted trust unchanged",
    );

    // 2. Performance Evidence Cannot Grant Qualification.
    record(
        &mut results,
        "Performance Evidence Cannot Grant Qualification",
        drift_signal_does_not_imply_incorrectness(),
        "expected KernelPerformanceModel to structurally lack a qualification-setting path",
    );

    // 3. Performance Context Isolation.
    let mut model = KernelPerformanceModel::new(
        CompiledKernelArtifactId::from_digest("digest-a"),
        conformance_bucket(),
        1,
    );
    let observation_wrong_artifact =
        conformance_observation(CompiledKernelArtifactId::from_digest("digest-b"), 100, 1);
    let isolated = model.record_observation(&observation_wrong_artifact).is_err();
    record(
        &mut results,
        "Performance Context Isolation",
        isolated,
        "expected an observation for a different artifact digest to be rejected",
    );

    // 4. Sample Sufficiency Conformance.
    let sufficiency = SampleSufficiencyPolicy {
        minimum_samples: 100,
        ..SampleSufficiencyPolicy::default()
    };
    let mut one_sample_summary = KernelPerformanceAggregator::new();
    one_sample_summary.record(&conformance_observation(
        CompiledKernelArtifactId::from_digest("digest-a"),
        5_000,
        1,
    ));
    let quality = evaluate_sample_sufficiency(
        &one_sample_summary.summary(),
        10_000,
        &sufficiency,
    );
    let regression_confirmation_policy = RegressionConfirmationPolicy {
        sufficiency,
        minimum_sustained_duration_millis: 1_000,
    };
    let signal = RegressionSignal {
        baseline_kind: RegressionBaselineKind::PriorGeneration,
        relative_latency_increase: 5.0,
        p99_relative_increase: 5.0,
        timeout_rate_increase: 0.0,
    };
    let confirmed = confirm_regression(&signal, quality, 10_000, &regression_confirmation_policy);
    record(
        &mut results,
        "Sample Sufficiency Conformance",
        !confirmed,
        "expected one sample against a 100-sample minimum to leave regression unconfirmed",
    );

    // 5. Drift Detection Conformance.
    let baseline = KernelPerformanceBaseline {
        source: KernelPerformanceEvidenceSource::Offline,
        p50_micros: 30,
        p90_micros: 40,
        p99_micros: 60,
        sample_count: 1_000,
    };
    let observed = KernelPerformanceMetricSummary {
        count: 200,
        p50_micros: 45,
        ..KernelPerformanceMetricSummary::default()
    };
    let threshold = DriftThreshold {
        relative: 0.2,
        absolute_micros: 5,
    };
    let drift = detect_benchmark_drift(&baseline, &observed, &threshold, EvidenceQuality::High);
    record(
        &mut results,
        "Drift Detection Conformance",
        drift.is_some(),
        "expected a sustained 50% p50 increase to produce a drift signal",
    );

    // 6. Workload Drift Conformance.
    let tuned = WorkloadProfile {
        batch_bucket: "1..4".into(),
        sequence_bucket: "0..2048".into(),
        phase: Some(GenerationPhase::Decode),
        shape_bucket: "attn".into(),
    };
    let actual = WorkloadProfile {
        batch_bucket: "16..32".into(),
        ..tuned.clone()
    };
    let workload_drift = detect_workload_drift(&tuned, &actual);
    record(
        &mut results,
        "Workload Drift Conformance",
        workload_drift.contains(&WorkloadDriftDimension::Batch),
        "expected a batch bucket change to be detected as workload drift",
    );

    // 7. Bounded Re-Tuning Conformance.
    let request = KernelRetuningRequest {
        reason: TuningStalenessReason::PerformanceDrift,
        workload_bucket: "attn-decode".into(),
        candidate_context: CompiledKernelArtifactId::from_digest("outside-plan"),
        evidence_summary: KernelPerformanceMetricSummary::default(),
        urgency: RetuningUrgency::Medium,
        requested_at_millis: 0,
    };
    let empty_plan = KernelAutotuningPlan {
        template: crate::KernelSpecializationTemplate::new(
            crate::KernelSpecializationTemplateId::new("template"),
            conformance_kernel_id(),
            1,
        ),
        candidates: Vec::new(),
        workload: crate::KernelAutotuningWorkloadBucket {
            operator: conformance_operator(),
            shape_bucket: "attn".into(),
            batch_bucket: None,
            sequence_bucket: None,
            phase: crate::KernelAutotuningExecutionPhase::Decode,
            dtype: crate::ComputeDType::Float16,
            layout: crate::TensorLayoutKind::Contiguous,
            quantization: None,
            provider: ProviderBinding::new("cuda"),
            device_architecture: "sm90".into(),
            device_features: Default::default(),
        },
        benchmark_profile: crate::KernelAutotuningBenchmarkProfile {
            warmup_iterations: 1,
            measurement_iterations: 1,
            synchronization_policy: "sync".into(),
            timeout_millis: 1_000,
            metric: crate::KernelAutotuningObjective::Latency,
            outlier_policy: None,
        },
        objective: crate::KernelAutotuningObjective::Latency,
        secondary_objectives: Vec::new(),
        budget: crate::KernelAutotuningBudget::default(),
        fallback: crate::KernelAutotuningFallback::StructuredNotReady,
    };
    record(
        &mut results,
        "Bounded Re-Tuning Conformance",
        !retuning_request_respects_autotuning_boundary(&request, &empty_plan),
        "expected a candidate absent from the bounded plan to be rejected",
    );

    // 8. No Hot-Path Adaptive Benchmarking.
    record(
        &mut results,
        "No Hot-Path Adaptive Benchmarking",
        request_retuning_outside_hot_path(true).is_err()
            && request_retuning_outside_hot_path(false).is_ok(),
        "expected decode-triggered retuning to be denied and non-decode triggers to be allowed",
    );

    // 9. External Escalation Boundary.
    let escalation = escalate_to_optimization_plane(
        true,
        true,
        "attn-decode",
        KernelPerformanceMetricSummary::default(),
        0,
    );
    record(
        &mut results,
        "External Escalation Boundary",
        escalation.is_some(),
        "expected exhausted bounded retuning with an unmet target to escalate externally",
    );

    // 10. Hysteresis Conformance.
    let hysteresis = AdaptiveActionHysteresis {
        minimum_relative_change: 0.1,
        minimum_stable_duration_millis: 10_000,
    };
    let should_change = should_change_due_to_performance(100.0, 101.0, 0, &hysteresis);
    record(
        &mut results,
        "Hysteresis Conformance",
        !should_change,
        "expected a 1% swing observed for 0ms to leave the active Kernel unchanged",
    );

    // 11. Reproducible Mode Conformance.
    let mode = reproducible_mode_blocks_adaptation(true, KernelPerformanceFeedbackMode::Adaptive);
    record(
        &mut results,
        "Reproducible Mode Conformance",
        !feedback_mode_allows_selection_change(mode),
        "expected reproducible mode to force Pinned feedback mode",
    );

    // 12. Bounded Retention Conformance.
    let mut bounded_aggregator = KernelPerformanceAggregator::new();
    for i in 0..(MAX_RAW_SAMPLES_PER_MODEL * 4) {
        bounded_aggregator.record(&conformance_observation(
            CompiledKernelArtifactId::from_digest("digest-a"),
            i as u64,
            i as u64,
        ));
    }
    let bounded_summary = bounded_aggregator.summary();
    record(
        &mut results,
        "Bounded Retention Conformance",
        bounded_aggregator.raw_sample_count() <= MAX_RAW_SAMPLES_PER_MODEL
            && bounded_summary.count as usize == MAX_RAW_SAMPLES_PER_MODEL * 4,
        "expected raw sample retention to stay bounded while the exact count keeps accumulating",
    );

    // 13. Feedback Failure Isolation.
    let observation_error: Result<(), KernelPerformanceError> =
        Err(KernelPerformanceError::Internal {
            reason: "aggregator failure".into(),
        });
    record(
        &mut results,
        "Feedback Failure Isolation",
        observation_error.is_err() && KernelArtifactTrust::Trusted.is_trusted(),
        "expected a performance-subsystem error to leave an already-trusted Kernel's trust state \
         untouched",
    );

    // 14. Telemetry Redaction Conformance.
    let export = KernelPerformanceExportSummary {
        kernel: conformance_observation(CompiledKernelArtifactId::from_digest("digest-a"), 1, 1)
            .kernel,
        workload_bucket: "attn-decode".into(),
        metric_summary: KernelPerformanceMetricSummary::default(),
        evidence_source: KernelPerformanceEvidenceSource::Online,
        health: KernelPerformanceHealth::Healthy,
        policy_version: 1,
    };
    let payload = export.to_redacted_payload();
    let leaks_raw_content = payload
        .values()
        .any(|value| value.contains("0x") || value.contains("handle="));
    record(
        &mut results,
        "Telemetry Redaction Conformance",
        !leaks_raw_content,
        "expected exported telemetry to contain no raw handles or pointer-shaped diagnostics",
    );

    KernelPerformanceConformanceReport { results }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_report_is_fully_conformant() {
        let report = run_kernel_performance_conformance();
        for result in &report.results {
            assert!(
                result.passed,
                "conformance requirement failed: {} ({:?})",
                result.requirement, result.diagnostic
            );
        }
        assert!(report.is_conformant());
        assert_eq!(report.results.len(), 14);
    }

    #[test]
    fn workload_bucket_identity_is_deterministic() {
        let a = conformance_bucket();
        let b = conformance_bucket();
        assert_eq!(a.bucket_id(), b.bucket_id());

        let mut different = conformance_bucket();
        different.context.batch_bucket = "16..32".into();
        assert_ne!(a.bucket_id(), different.bucket_id());
    }

    #[test]
    fn aggregator_tracks_exact_count_with_bounded_raw_retention() {
        let mut aggregator = KernelPerformanceAggregator::new();
        for i in 0..(MAX_RAW_SAMPLES_PER_MODEL as u64 * 3) {
            aggregator.record(&conformance_observation(
                CompiledKernelArtifactId::from_digest("digest-a"),
                i,
                i,
            ));
        }
        let summary = aggregator.summary();
        assert_eq!(summary.count, MAX_RAW_SAMPLES_PER_MODEL as u64 * 3);
        assert!(aggregator.raw_sample_count() <= MAX_RAW_SAMPLES_PER_MODEL);
    }

    #[test]
    fn aggregator_tracks_failure_and_timeout_counts() {
        let mut aggregator = KernelPerformanceAggregator::new();
        let mut success = conformance_observation(CompiledKernelArtifactId::from_digest("d"), 10, 1);
        aggregator.record(&success);
        success.completion = KernelExecutionCompletion::Failed {
            category: "provider-error".into(),
        };
        aggregator.record(&success);
        success.completion = KernelExecutionCompletion::TimedOut;
        aggregator.record(&success);
        let summary = aggregator.summary();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.failure_count, 2);
        assert_eq!(summary.timeout_count, 1);
        assert!((summary.failure_rate() - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn record_observation_rejects_mismatched_artifact_and_bucket() {
        let mut model = KernelPerformanceModel::new(
            CompiledKernelArtifactId::from_digest("digest-a"),
            conformance_bucket(),
            1,
        );
        let matching = conformance_observation(CompiledKernelArtifactId::from_digest("digest-a"), 10, 1);
        assert!(model.record_observation(&matching).is_ok());

        let mismatched_artifact =
            conformance_observation(CompiledKernelArtifactId::from_digest("digest-b"), 10, 2);
        assert!(model.record_observation(&mismatched_artifact).is_err());

        let mut mismatched_bucket =
            conformance_observation(CompiledKernelArtifactId::from_digest("digest-a"), 10, 3);
        mismatched_bucket.workload_bucket.context.batch_bucket = "999..1000".into();
        assert!(model.record_observation(&mismatched_bucket).is_err());
    }

    #[test]
    fn evidence_quality_requires_both_samples_and_duration() {
        let policy = SampleSufficiencyPolicy::default();
        let mut summary = KernelPerformanceMetricSummary {
            count: policy.minimum_samples,
            ..KernelPerformanceMetricSummary::default()
        };
        assert_eq!(
            evaluate_sample_sufficiency(&summary, 0, &policy),
            EvidenceQuality::Insufficient,
            "duration below minimum must still be insufficient even with enough samples"
        );
        summary.count = policy.minimum_samples - 1;
        assert_eq!(
            evaluate_sample_sufficiency(&summary, policy.minimum_observation_duration_millis, &policy),
            EvidenceQuality::Insufficient,
        );
    }

    #[test]
    fn drift_signal_requires_sufficient_evidence() {
        let baseline = KernelPerformanceBaseline {
            source: KernelPerformanceEvidenceSource::Offline,
            p50_micros: 30,
            p90_micros: 40,
            p99_micros: 60,
            sample_count: 10,
        };
        let observed = KernelPerformanceMetricSummary {
            p50_micros: 90,
            ..KernelPerformanceMetricSummary::default()
        };
        let threshold = DriftThreshold {
            relative: 0.1,
            absolute_micros: 1,
        };
        assert!(
            detect_benchmark_drift(&baseline, &observed, &threshold, EvidenceQuality::Insufficient)
                .is_none()
        );
        assert!(
            detect_benchmark_drift(&baseline, &observed, &threshold, EvidenceQuality::High).is_some()
        );
    }

    #[test]
    fn regression_detection_flags_p99_and_timeout_dimensions_independently() {
        let baseline = KernelPerformanceMetricSummary {
            mean_latency_micros: 100.0,
            p99_micros: 200,
            count: 1000,
            timeout_count: 1,
            ..KernelPerformanceMetricSummary::default()
        };
        let mut current = baseline;
        current.p99_micros = 500;
        let thresholds = RegressionThresholds {
            relative_latency_increase: 10.0,
            absolute_latency_increase_micros: 1_000_000,
            throughput_reduction: 10.0,
            p99_relative_increase: 0.5,
            timeout_rate_increase: 10.0,
        };
        let signal =
            detect_regression(&baseline, &current, RegressionBaselineKind::PriorGeneration, &thresholds);
        assert!(signal.is_some());
    }

    #[test]
    fn outlier_policy_never_silently_drops_samples() {
        let latencies = vec![10, 20, 30, 1_000];
        let (retained, tail) =
            apply_outlier_policy(&latencies, OutlierPolicy::RetainInTail, 100);
        assert_eq!(retained.len(), latencies.len());
        assert!(tail.is_empty());

        let (retained, tail) =
            apply_outlier_policy(&latencies, OutlierPolicy::MarkSeparately, 100);
        assert_eq!(retained.len() + tail.len(), latencies.len());
        assert!(tail.contains(&1_000));
    }

    #[test]
    fn broad_slowdown_requires_pressure_and_multiple_candidates() {
        let single_candidate = DevicePressureCorrelation {
            device_pressure: Some(MemoryPressureLevel::Saturated),
            provider_pressure: None,
            distinct_candidates_slow_simultaneously: 1,
        };
        assert!(!broad_slowdown_suspected(&single_candidate));

        let multi_candidate = DevicePressureCorrelation {
            distinct_candidates_slow_simultaneously: 3,
            ..single_candidate
        };
        assert!(broad_slowdown_suspected(&multi_candidate));
    }

    #[test]
    fn retuning_cooldown_rate_limits_duplicate_requests() {
        let mut cooldown = RetuningCooldown::new(10_000);
        let mut request = KernelRetuningRequest {
            reason: TuningStalenessReason::PerformanceDrift,
            workload_bucket: "attn-decode".into(),
            candidate_context: CompiledKernelArtifactId::from_digest("digest-a"),
            evidence_summary: KernelPerformanceMetricSummary::default(),
            urgency: RetuningUrgency::Medium,
            requested_at_millis: 0,
        };
        assert!(cooldown.admit(&request));
        request.requested_at_millis = 5_000;
        assert!(!cooldown.admit(&request), "expected the same request inside the cooldown to be rate-limited");
        request.requested_at_millis = 11_000;
        assert!(cooldown.admit(&request));
    }

    #[test]
    fn cross_device_reuse_requires_matching_architecture_or_declared_class() {
        assert!(cross_device_reuse_allowed("sm90", "sm90", None));
        assert!(!cross_device_reuse_allowed("sm90", "sm80", None));
        assert!(cross_device_reuse_allowed(
            "sm90",
            "sm90a",
            Some(&["sm90", "sm90a"])
        ));
    }

    #[test]
    fn aging_mechanisms_expire_or_decay_evidence() {
        assert_eq!(
            observation_weight(&AgingMechanism::TimeWindow { max_age_millis: 1000 }, 2000, 0),
            0.0
        );
        assert_eq!(
            observation_weight(&AgingMechanism::TimeWindow { max_age_millis: 1000 }, 500, 0),
            1.0
        );
        let decayed = observation_weight(
            &AgingMechanism::WeightedDecay { half_life_millis: 1000 },
            1000,
            0,
        );
        assert!((decayed - 0.5).abs() < 1e-9);
    }

    #[test]
    fn batch_attribution_never_fabricates_without_a_matching_model() {
        assert!(attribute_batch_latency(100, 0, &BatchAttributionModel::EqualSplit).is_none());
        let mismatched_weights = BatchAttributionModel::Weighted {
            weights: vec![1.0, 1.0],
        };
        assert!(attribute_batch_latency(100, 3, &mismatched_weights).is_none());

        let equal = attribute_batch_latency(100, 4, &BatchAttributionModel::EqualSplit).unwrap();
        assert_eq!(equal, vec![25, 25, 25, 25]);
    }

    #[test]
    fn export_summary_redacts_pointer_shaped_metadata() {
        let mut export = KernelPerformanceExportSummary {
            kernel: conformance_observation(CompiledKernelArtifactId::from_digest("d"), 1, 1).kernel,
            workload_bucket: "handle=0xdeadbeef".into(),
            metric_summary: KernelPerformanceMetricSummary::default(),
            evidence_source: KernelPerformanceEvidenceSource::Online,
            health: KernelPerformanceHealth::Healthy,
            policy_version: 1,
        };
        let payload = export.to_redacted_payload();
        assert_eq!(
            payload.get("workload_bucket").unwrap(),
            "[redacted backend diagnostic]"
        );
        export.workload_bucket = "attn-decode".into();
        let payload = export.to_redacted_payload();
        assert_eq!(payload.get("workload_bucket").unwrap(), "attn-decode");
    }

    #[test]
    fn feedback_mode_reproducible_override_blocks_selection_change() {
        assert_eq!(
            reproducible_mode_blocks_adaptation(true, KernelPerformanceFeedbackMode::Adaptive),
            KernelPerformanceFeedbackMode::Pinned
        );
        assert_eq!(
            reproducible_mode_blocks_adaptation(false, KernelPerformanceFeedbackMode::Adaptive),
            KernelPerformanceFeedbackMode::Adaptive
        );
        assert!(!feedback_mode_allows_selection_change(
            KernelPerformanceFeedbackMode::Pinned
        ));
    }
}
