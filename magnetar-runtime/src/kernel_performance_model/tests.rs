//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

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
    let matching =
        conformance_observation(CompiledKernelArtifactId::from_digest("digest-a"), 10, 1);
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
        evaluate_sample_sufficiency(
            &summary,
            policy.minimum_observation_duration_millis,
            &policy
        ),
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
        detect_benchmark_drift(
            &baseline,
            &observed,
            &threshold,
            EvidenceQuality::Insufficient
        )
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
    let signal = detect_regression(
        &baseline,
        &current,
        RegressionBaselineKind::PriorGeneration,
        &thresholds,
    );
    assert!(signal.is_some());
}

#[test]
fn outlier_policy_never_silently_drops_samples() {
    let latencies = vec![10, 20, 30, 1_000];
    let (retained, tail) = apply_outlier_policy(&latencies, OutlierPolicy::RetainInTail, 100);
    assert_eq!(retained.len(), latencies.len());
    assert!(tail.is_empty());

    let (retained, tail) = apply_outlier_policy(&latencies, OutlierPolicy::MarkSeparately, 100);
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
    assert!(
        !cooldown.admit(&request),
        "expected the same request inside the cooldown to be rate-limited"
    );
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
        observation_weight(
            &AgingMechanism::TimeWindow {
                max_age_millis: 1000
            },
            2000,
            0
        ),
        0.0
    );
    assert_eq!(
        observation_weight(
            &AgingMechanism::TimeWindow {
                max_age_millis: 1000
            },
            500,
            0
        ),
        1.0
    );
    let decayed = observation_weight(
        &AgingMechanism::WeightedDecay {
            half_life_millis: 1000,
        },
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
