//! Kernel Benchmarking and Performance Ranking (see
//! `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract`).
//!
//! This module does not implement a benchmark harness or any specific
//! measurement methodology (proposal's "Non-Goals"). It defines, as
//! executable Rust types and validation functions, the contract that
//! performance evidence SHALL follow once correctness qualification has
//! already passed:
//!
//! ```text
//! QualifiedKernelArtifact -> Benchmarking -> Performance Ranking
//! ```
//!
//! - [`BenchmarkProfile`]: workload/environment identity a benchmark result
//!   is bound to, implementing "Benchmark Profiles": "A benchmark result
//!   without workload/context metadata SHALL NOT be used as authoritative
//!   ranking evidence."
//! - [`BenchmarkMetrics`]: the extensible metric set (latency, throughput,
//!   launch overhead, memory bandwidth, workspace usage, energy, compile/
//!   prepare cost, tail latency), implementing "Benchmarking".
//! - [`BenchmarkRecord`] / [`BenchmarkFreshness`]: a benchmark result plus
//!   its staleness state, implementing "Benchmark Stability": "A benchmark
//!   result SHOULD NOT be considered permanently valid."
//! - [`RankingCandidate`] / [`rank_eligible_candidates`]: implements
//!   "Performance Ranking": correctness, trust, compatibility, readiness,
//!   Resource Affinity, and policy SHALL be evaluated before performance
//!   ranking -- an ineligible candidate can never outrank an eligible one
//!   regardless of measured performance.
//! - [`RegressionPolicy`] / [`evaluate_regression_policy`]: "Regression
//!   Policy": a newly generated Kernel MAY be rejected even when correct if
//!   it violates performance policy.
//! - [`KernelBenchmarkError`]: the benchmark subset of the proposal's "Error
//!   Model" section.
//! - [`BenchmarkObservationKind`] / [`BenchmarkObservation`]: redacted
//!   benchmark lifecycle observability.
//! - [`KernelBenchmarkConformanceReport`] / [`run_kernel_benchmark_conformance`]:
//!   the conformance checks from this change's benchmark-related
//!   requirements.

use crate::compute::redact_backend_diagnostic;
use std::{collections::BTreeMap, error::Error, fmt};

pub const KERNEL_BENCHMARK_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Benchmark Profiles
// ---------------------------------------------------------------------

/// Workload and environment identity a benchmark result is bound to,
/// implementing "Benchmark Profiles" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkProfile {
    pub target_device: String,
    pub hardware_architecture: String,
    pub provider_version: String,
    pub driver_runtime_version: Option<String>,
    pub input_shapes: String,
    pub dtype_layout: String,
    pub batch_size: Option<u64>,
    pub sequence_length: Option<u64>,
    pub warmup_count: u32,
    pub measurement_count: u32,
    pub synchronization_policy: String,
    pub benchmark_version: String,
}

impl BenchmarkProfile {
    /// Implements "A benchmark result without workload/context metadata
    /// SHALL NOT be used as authoritative ranking evidence" (proposal).
    pub fn is_authoritative(&self) -> bool {
        !self.target_device.trim().is_empty()
            && !self.hardware_architecture.trim().is_empty()
            && !self.provider_version.trim().is_empty()
            && !self.input_shapes.trim().is_empty()
            && self.measurement_count > 0
    }
}

// ---------------------------------------------------------------------
// Benchmark Metrics
// ---------------------------------------------------------------------

/// Extensible benchmark metric set, implementing "Benchmarking" (proposal).
/// `extra` keeps this open for energy, compile-cost, and other
/// forward-declared metrics without requiring a struct field per metric.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchmarkMetrics {
    pub latency_millis: Option<f64>,
    pub throughput_per_second: Option<f64>,
    pub launch_overhead_millis: Option<f64>,
    pub memory_bandwidth_gb_per_second: Option<f64>,
    pub workspace_bytes: Option<u64>,
    pub tail_latency_p99_millis: Option<f64>,
    pub extra: BTreeMap<String, f64>,
}

// ---------------------------------------------------------------------
// Benchmark Freshness
// ---------------------------------------------------------------------

/// Implements "Benchmark Stability" (proposal): "Performance evidence MAY
/// become stale because of driver/Provider/compiler changes, hardware
/// revisions, firmware changes, thermal/power policy, Runtime changes."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkFreshness {
    Fresh,
    Stale { reason: BenchmarkStalenessReason },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkStalenessReason {
    DriverChanged,
    ProviderChanged,
    CompilerChanged,
    HardwareRevisionChanged,
    FirmwareChanged,
    ThermalPowerPolicyChanged,
    RuntimeChanged,
}

/// A recorded benchmark result, implementing "Benchmark Domain" (proposal).
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRecord {
    pub profile: BenchmarkProfile,
    pub metrics: BenchmarkMetrics,
    pub freshness: BenchmarkFreshness,
}

impl BenchmarkRecord {
    /// Implements "Bind results to Provider/Device architecture/driver
    /// class/workload profile" and "Mark stale results" (tasks): a record is
    /// usable as ranking evidence only when both authoritative and fresh.
    pub fn usable_as_ranking_evidence(&self) -> bool {
        self.profile.is_authoritative() && matches!(self.freshness, BenchmarkFreshness::Fresh)
    }
}

// ---------------------------------------------------------------------
// Performance Ranking
// ---------------------------------------------------------------------

/// A ranking candidate, implementing "Performance Ranking" (proposal)'s
/// eligibility formula:
///
/// ```text
/// eligible =
///     semantics compatible
///     && qualification accepted
///     && trust policy accepted
///     && Provider ready
///     && Device compatible
///     && memory feasible
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RankingCandidate {
    pub id: String,
    pub semantics_compatible: bool,
    pub qualification_accepted: bool,
    pub trust_policy_accepted: bool,
    pub provider_ready: bool,
    pub device_compatible: bool,
    pub memory_feasible: bool,
    pub benchmark: Option<BenchmarkRecord>,
}

impl RankingCandidate {
    pub fn is_eligible(&self) -> bool {
        self.semantics_compatible
            && self.qualification_accepted
            && self.trust_policy_accepted
            && self.provider_ready
            && self.device_compatible
            && self.memory_feasible
    }

    fn latency_key(&self) -> f64 {
        self.benchmark
            .as_ref()
            .filter(|record| record.usable_as_ranking_evidence())
            .and_then(|record| record.metrics.latency_millis)
            .unwrap_or(f64::INFINITY)
    }
}

/// Implements "Correctness, trust, compatibility, readiness, Resource
/// Affinity, and policy SHALL be evaluated before performance ranking"
/// (proposal): ineligible candidates are placed after every eligible
/// candidate regardless of measured performance, so "A faster incorrect
/// Kernel SHALL never outrank a correct Kernel" holds structurally.
pub fn rank_eligible_candidates(candidates: &[RankingCandidate]) -> Vec<&RankingCandidate> {
    let mut ranked: Vec<&RankingCandidate> = candidates.iter().collect();
    ranked.sort_by(|a, b| {
        let eligibility = b.is_eligible().cmp(&a.is_eligible());
        if eligibility != std::cmp::Ordering::Equal {
            return eligibility;
        }
        a.latency_key()
            .partial_cmp(&b.latency_key())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

// ---------------------------------------------------------------------
// Regression Policy
// ---------------------------------------------------------------------

/// Implements "Regression Policy" (proposal): "A newly generated Kernel MAY
/// be rejected even when correct if it violates performance policy."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegressionPolicy {
    CorrectnessOnly,
    MustBeatCurrent,
    WithinPercentOfCurrent { max_regression_percent: u32 },
    MemoryOptimized,
    EnergyOptimized,
}

/// Implements "Prevent incorrect kernel from winning ranking" and "Define
/// must-beat-current policy"/"Define within-regression-threshold policy"
/// (tasks).
pub fn evaluate_regression_policy(
    policy: RegressionPolicy,
    candidate_latency_millis: f64,
    current_active_latency_millis: f64,
) -> Result<(), KernelBenchmarkError> {
    match policy {
        RegressionPolicy::CorrectnessOnly
        | RegressionPolicy::MemoryOptimized
        | RegressionPolicy::EnergyOptimized => Ok(()),
        RegressionPolicy::MustBeatCurrent => {
            if candidate_latency_millis < current_active_latency_millis {
                Ok(())
            } else {
                Err(KernelBenchmarkError::Regression)
            }
        }
        RegressionPolicy::WithinPercentOfCurrent {
            max_regression_percent,
        } => {
            let max_allowed =
                current_active_latency_millis * (1.0 + f64::from(max_regression_percent) / 100.0);
            if candidate_latency_millis <= max_allowed {
                Ok(())
            } else {
                Err(KernelBenchmarkError::Regression)
            }
        }
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Benchmark error, covering the benchmark subset of the
/// proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelBenchmarkError {
    Unavailable,
    Invalid { reason: String },
    Failed { reason: String },
    Regression,
    Stale,
}

impl KernelBenchmarkError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Unavailable => "kernel-benchmark-unavailable",
            Self::Invalid { .. } => "kernel-benchmark-invalid",
            Self::Failed { .. } => "kernel-benchmark-failed",
            Self::Regression => "kernel-benchmark-regression",
            Self::Stale => "kernel-benchmark-stale",
        }
    }
}

impl fmt::Display for KernelBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { reason } | Self::Failed { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelBenchmarkError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BenchmarkObservationKind {
    BenchmarkStarted,
    BenchmarkCompleted,
    BenchmarkRegression,
}

/// A single benchmark observation. Structurally guaranteed to never carry
/// raw test tensors or native handles, implementing "Runtime MAY emit
/// benchmark summaries without exposing sensitive inputs" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkObservation {
    pub kind: BenchmarkObservationKind,
    pub candidate: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl BenchmarkObservation {
    pub fn new(kind: BenchmarkObservationKind) -> Self {
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
pub struct KernelBenchmarkConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelBenchmarkConformanceReport {
    pub results: Vec<KernelBenchmarkConformanceResult>,
}

impl KernelBenchmarkConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelBenchmarkConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelBenchmarkConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn conformance_profile(device: &str) -> BenchmarkProfile {
    BenchmarkProfile {
        target_device: device.into(),
        hardware_architecture: "sm90".into(),
        provider_version: "1.0.0".into(),
        driver_runtime_version: Some("1.0".into()),
        input_shapes: "[1,4096]".into(),
        dtype_layout: "fp16/row-major".into(),
        batch_size: Some(1),
        sequence_length: Some(4096),
        warmup_count: 5,
        measurement_count: 20,
        synchronization_policy: "device-sync".into(),
        benchmark_version: "1".into(),
    }
}

/// Runs the Kernel Benchmark conformance checks described in this module's
/// doc comment: "Performance ranking occurs only among eligible candidates"
/// and "A faster incorrect Kernel SHALL never outrank a correct Kernel"
/// (`specs/conformance/spec.md`).
pub fn run_kernel_benchmark_conformance() -> KernelBenchmarkConformanceReport {
    let mut results = Vec::new();

    let incorrect_but_fast = RankingCandidate {
        id: "incorrect-fast".into(),
        semantics_compatible: true,
        qualification_accepted: false,
        trust_policy_accepted: true,
        provider_ready: true,
        device_compatible: true,
        memory_feasible: true,
        benchmark: Some(BenchmarkRecord {
            profile: conformance_profile("gpu-0"),
            metrics: BenchmarkMetrics {
                latency_millis: Some(1.0),
                ..BenchmarkMetrics::default()
            },
            freshness: BenchmarkFreshness::Fresh,
        }),
    };
    let correct_but_slower = RankingCandidate {
        id: "correct-slow".into(),
        semantics_compatible: true,
        qualification_accepted: true,
        trust_policy_accepted: true,
        provider_ready: true,
        device_compatible: true,
        memory_feasible: true,
        benchmark: Some(BenchmarkRecord {
            profile: conformance_profile("gpu-0"),
            metrics: BenchmarkMetrics {
                latency_millis: Some(10.0),
                ..BenchmarkMetrics::default()
            },
            freshness: BenchmarkFreshness::Fresh,
        }),
    };
    let candidates = [incorrect_but_fast, correct_but_slower];
    let ranked = rank_eligible_candidates(&candidates);
    record(
        &mut results,
        "faster incorrect candidate never outranks a correct candidate",
        ranked.first().map(|candidate| candidate.id.as_str()) == Some("correct-slow"),
        format!("unexpected ranking order: {ranked:?}"),
    );

    let stale_record = BenchmarkRecord {
        profile: conformance_profile("gpu-0"),
        metrics: BenchmarkMetrics::default(),
        freshness: BenchmarkFreshness::Stale {
            reason: BenchmarkStalenessReason::DriverChanged,
        },
    };
    record(
        &mut results,
        "stale benchmark evidence is not usable for ranking",
        !stale_record.usable_as_ranking_evidence(),
        "expected stale record to be excluded from ranking evidence",
    );

    let regression = evaluate_regression_policy(RegressionPolicy::MustBeatCurrent, 12.0, 10.0);
    record(
        &mut results,
        "must-beat-current policy rejects a slower candidate",
        matches!(regression, Err(KernelBenchmarkError::Regression)),
        format!("unexpected outcome: {regression:?}"),
    );

    KernelBenchmarkConformanceReport { results }
}
