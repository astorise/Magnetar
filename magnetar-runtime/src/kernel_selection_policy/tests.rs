//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::ProviderBinding;
use crate::kernel::{KernelId, KernelOperatorVersionRange};
use crate::operator::{OperatorFamily, OperatorId};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::compute::{ComputeDType, HostStagingPolicy};
use crate::kernel::{KernelExecutionMode, KernelFallbackClass};
use crate::kernel_artifact::KernelArtifactTrust;
use crate::kernel_benchmark::{BenchmarkFreshness, BenchmarkStalenessReason};
use crate::kernel_registry::KernelCandidateRejection;
use crate::model_instance::KernelSelectionPolicy;
use crate::model_instance::{
    ModelInstanceWarmupPlan, ModelInstanceWarmupPolicy, PinnedKernelSelection,
};
use crate::operator::TensorLayoutKind;
fn selection_policy_identity(name: &str) -> CandidateIdentity {
    CandidateIdentity {
        kernel: selection_policy_kernel_id(name),
        provider: ProviderBinding::new("selection-policy-provider"),
        artifact_digest: None,
    }
}

fn selection_policy_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("selection-policy-provider"),
        name,
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::CpuScalar,
    )
}

#[test]
fn eligible_candidate_cannot_be_constructed_without_passing_eligibility() {
    let rejected = EligibleCandidate::from_checked(
        selection_policy_identity("ineligible"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput {
            shape_compatible: false,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    assert_eq!(
        rejected,
        Err(KernelSelectionExclusionReason::ShapeIncompatible)
    );
}

#[test]
fn eligibility_checks_fail_in_deterministic_order() {
    // Both dtype and layout are wrong; dtype is checked first, so it wins.
    let input = CandidateEligibilityInput {
        dtype_compatible: false,
        layout_compatible: false,
        ..CandidateEligibilityInput::all_satisfied()
    };
    assert_eq!(
        evaluate_candidate_eligibility(&input),
        Err(KernelSelectionExclusionReason::DTypeIncompatible)
    );
}

#[test]
fn missing_metric_never_resolves_to_best_possible_value() {
    assert_eq!(
        missing_metric_value(MissingMetricPolicy::Exclude, None),
        f64::INFINITY
    );
    assert!(missing_metric_value(MissingMetricPolicy::RankConservatively, None) > 0.0);
    assert_eq!(
        missing_metric_value(MissingMetricPolicy::UseFallbackMetadata, Some(42.0)),
        42.0
    );
}

#[test]
fn should_retain_active_due_to_missing_metric_only_under_that_policy() {
    let metrics = CandidateMetrics::default();
    assert!(should_retain_active_due_to_missing_metric(
        &metrics,
        &[ObjectiveDimension::Latency],
        MissingMetricPolicy::RetainActiveKernel,
    ));
    assert!(!should_retain_active_due_to_missing_metric(
        &metrics,
        &[ObjectiveDimension::Latency],
        MissingMetricPolicy::Exclude,
    ));
}

#[test]
fn weighted_score_ranks_lower_combined_cost_first() {
    let fast = CandidateMetrics::default()
        .with(ObjectiveDimension::Latency, 1.0)
        .with(ObjectiveDimension::Memory, 1.0);
    let slow = CandidateMetrics::default()
        .with(ObjectiveDimension::Latency, 10.0)
        .with(ObjectiveDimension::Memory, 1.0);
    let policy = WeightedScorePolicy {
        weights: BTreeMap::from([
            (ObjectiveDimension::Latency, 1.0),
            (ObjectiveDimension::Memory, 1.0),
        ]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    };
    assert!(weighted_score(&fast, &policy) < weighted_score(&slow, &policy));
}

#[test]
fn lexicographic_ranking_uses_first_differentiating_objective() {
    let a = CandidateMetrics::default()
        .with(ObjectiveDimension::Determinism, 0.0)
        .with(ObjectiveDimension::Latency, 100.0);
    let b = CandidateMetrics::default()
        .with(ObjectiveDimension::Determinism, 1.0)
        .with(ObjectiveDimension::Latency, 1.0);
    let policy = LexicographicPolicy {
        order: vec![ObjectiveDimension::Determinism, ObjectiveDimension::Latency],
        missing_metric_policy: MissingMetricPolicy::Exclude,
    };
    assert_eq!(
        lexicographic_compare(&a, &b, &policy),
        std::cmp::Ordering::Less
    );
}

#[test]
fn rank_candidates_is_deterministic_across_repeated_calls() {
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("z"),
            CandidateMetrics::default().with(ObjectiveDimension::Latency, 3.0),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
        EligibleCandidate::from_checked(
            selection_policy_identity("a"),
            CandidateMetrics::default().with(ObjectiveDimension::Latency, 3.0),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    assert!(selection_is_deterministic_for_identical_inputs(
        &candidates,
        &strategy
    ));
    // Tied score: stable tie-break falls back to CandidateIdentity's Ord.
    let ranked = rank_candidates(&candidates, &strategy).unwrap();
    assert_eq!(ranked[0].kernel.name, "a");
}

#[test]
fn pinned_ranking_strategy_fails_when_kernel_is_not_eligible() {
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("present"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::Pinned(selection_policy_kernel_id("absent"));
    assert_eq!(
        rank_candidates(&candidates, &strategy),
        Err(KernelSelectionError::PinnedKernelUnavailable)
    );
}

#[test]
fn policy_ordered_ranking_places_unlisted_candidates_last() {
    let listed = selection_policy_kernel_id("listed");
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("unlisted"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
        EligibleCandidate::from_checked(
            CandidateIdentity {
                kernel: listed.clone(),
                provider: ProviderBinding::new("selection-policy-provider"),
                artifact_digest: None,
            },
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::PolicyOrdered(vec![listed.clone()]);
    let ranked = rank_candidates(&candidates, &strategy).unwrap();
    assert_eq!(ranked[0].kernel, listed);
}

#[test]
fn pressure_bias_only_applies_to_an_already_eligible_candidate() {
    let candidate = EligibleCandidate::from_checked(
        selection_policy_identity("pressure"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let saturated = PressureSnapshot {
        memory_pressure: PressureLevel::Saturated,
        provider_admission_open: true,
        ..PressureSnapshot::default()
    };
    let nominal = PressureSnapshot {
        provider_admission_open: true,
        ..PressureSnapshot::default()
    };
    assert!(
        pressure_ranking_bias(&candidate, &saturated) > pressure_ranking_bias(&candidate, &nominal)
    );
}

#[test]
fn conversion_cost_can_flip_the_faster_raw_execution_choice() {
    let kernel_a = ConversionCost {
        layout_conversion_ms: 30.0,
        ..ConversionCost::default()
    };
    let kernel_b = ConversionCost::default();
    let total_a = total_execution_cost_ms(20.0, &kernel_a);
    let total_b = total_execution_cost_ms(35.0, &kernel_b);
    assert!(total_b < total_a);
}

#[test]
fn preparation_cost_is_not_charged_again_once_already_prepared() {
    assert!(!preparation_cost_applies(
        PreparationCostClass::OneTime,
        true
    ));
    assert!(preparation_cost_applies(
        PreparationCostClass::PerOperation,
        true
    ));
}

#[test]
fn promotion_recommendation_is_denied_when_hysteresis_or_anti_flapping_blocks_it() {
    let candidate = selection_policy_identity("candidate");
    let denied = recommend_promotion(&candidate, SelectionOutcome::RetainActive, true);
    assert!(!denied.approved);
    let denied_by_cooldown =
        recommend_promotion(&candidate, SelectionOutcome::PromoteCandidate, false);
    assert!(!denied_by_cooldown.approved);
    let approved = recommend_promotion(&candidate, SelectionOutcome::PromoteCandidate, true);
    assert!(approved.approved);
}

#[test]
fn session_and_cli_preference_only_apply_to_already_eligible_candidates() {
    let eligible = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("eligible"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let ineligible_kernel = selection_policy_kernel_id("ineligible");
    assert!(resolve_session_preference(Some(&ineligible_kernel), &eligible).is_none());
    assert!(resolve_cli_kernel_preference(Some(&ineligible_kernel), &eligible).is_none());
    let eligible_kernel = eligible[0].identity.kernel.clone();
    assert!(resolve_session_preference(Some(&eligible_kernel), &eligible).is_some());
}

#[test]
fn cli_preference_maps_onto_the_matching_optimization_profile() {
    assert_eq!(
        map_cli_preference(CliPreference::Latency),
        OptimizationProfile::Latency
    );
    assert_eq!(
        map_cli_preference(CliPreference::Deterministic),
        OptimizationProfile::Deterministic
    );
}

#[test]
fn provider_private_variant_requires_every_contract_dimension_unchanged() {
    assert!(!provider_may_select_variant_privately(
        &ProviderPrivateVariant::default()
    ));
    assert!(provider_may_select_variant_privately(
        &ProviderPrivateVariant {
            contract_semantics_identical: true,
            runtime_visible_compatibility_unchanged: true,
            determinism_precision_unchanged: true,
        }
    ));
}

#[test]
fn cross_provider_movement_requires_both_explicit_and_authorized() {
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: true,
            authorized_by_policy: true,
        })
        .is_ok()
    );
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: true,
            authorized_by_policy: false,
        })
        .is_err()
    );
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: false,
            authorized_by_policy: true,
        })
        .is_err()
    );
}

#[test]
fn exploration_is_denied_by_default_under_reproducible_mode_but_allowed_when_enabled_outside_it() {
    let candidate = EligibleCandidate::from_checked(
        selection_policy_identity("explore"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let policy = ExplorationPolicy::default();
    assert!(!exploration_allowed(&policy, false));
    let enabled = ExplorationPolicy {
        enabled: true,
        disabled_for_reproducible: true,
    };
    assert!(eligible_for_exploration(&enabled, false, &candidate));
    assert!(!eligible_for_exploration(&enabled, true, &candidate));
}

#[test]
fn policy_precedence_lets_runtime_safety_override_every_lower_preference() {
    let stack = PolicyConstraintStack {
        runtime_safety_forces_deterministic: true,
        cli_preference: Some(OptimizationProfile::Latency),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Deterministic
    );
}

#[test]
fn policy_precedence_lets_deployment_forbid_a_cli_requested_profile() {
    let stack = PolicyConstraintStack {
        cli_preference: Some(OptimizationProfile::Latency),
        deployment_forbids_profile: BTreeSet::from([OptimizationProfile::Latency]),
        model_instance_profile: Some(OptimizationProfile::Balanced),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Balanced
    );
}

#[test]
fn policy_precedence_honors_cli_preference_when_nothing_overrides_it() {
    let stack = PolicyConstraintStack {
        cli_preference: Some(OptimizationProfile::Throughput),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Throughput
    );
}

#[test]
fn rank_by_generation_phase_permits_distinct_prefill_and_decode_winners() {
    let prefill_throughput = EligibleCandidate::from_checked(
        selection_policy_identity("prefill-throughput"),
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 50.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let decode_latency = EligibleCandidate::from_checked(
        selection_policy_identity("decode-latency"),
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 2.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    let by_phase = rank_by_generation_phase(
        std::slice::from_ref(&prefill_throughput),
        std::slice::from_ref(&decode_latency),
        &strategy,
    )
    .unwrap();
    assert_eq!(
        by_phase[&GenerationPhase::Prefill],
        vec![prefill_throughput.identity]
    );
    assert_eq!(
        by_phase[&GenerationPhase::Decode],
        vec![decode_latency.identity]
    );
}

#[test]
fn rolling_measurement_window_only_reports_stable_once_full() {
    let mut window = RollingMeasurementWindow::new(3);
    assert!(!window.is_stable());
    window.record(10.0);
    window.record(12.0);
    assert!(!window.is_stable());
    window.record(11.0);
    assert!(window.is_stable());
    assert_eq!(window.len(), 3);
    window.record(9.0);
    assert_eq!(window.len(), 3);
    assert!(window.mean().is_some());
}

#[test]
fn generation_preference_boundary_has_no_way_to_carry_a_kernel_identity() {
    // Structural: `resolve_generation_preference` only ever takes and
    // returns an `OptimizationProfile`, so a generation request cannot use
    // it to smuggle a concrete `PreparedKernelId` or `KernelId` into
    // selection, implementing "Generation requests MAY provide high-level
    // policy preferences. They SHALL NOT directly force an ineligible
    // concrete Kernel" (proposal).
    assert_eq!(
        resolve_generation_preference(
            Some(OptimizationProfile::Latency),
            OptimizationProfile::Balanced
        ),
        OptimizationProfile::Latency
    );
    assert_eq!(
        resolve_generation_preference(None, OptimizationProfile::Balanced),
        OptimizationProfile::Balanced
    );
}

#[test]
fn kernel_candidate_rejection_maps_onto_selection_exclusion_reason() {
    assert_eq!(
        KernelSelectionExclusionReason::from(KernelCandidateRejection::Revoked),
        KernelSelectionExclusionReason::QualificationRevoked
    );
    assert_eq!(
        KernelSelectionExclusionReason::from(KernelCandidateRejection::WorkspaceUnavailable),
        KernelSelectionExclusionReason::WorkspaceInfeasible
    );
}

#[test]
fn stale_benchmark_policy_explicitly_governs_exclusion() {
    let stale = BenchmarkFreshness::Stale {
        reason: BenchmarkStalenessReason::DriverChanged,
    };
    assert!(evaluate_stale_benchmark_policy(stale, StaleBenchmarkPolicy::Accept).is_ok());
    assert_eq!(
        evaluate_stale_benchmark_policy(stale, StaleBenchmarkPolicy::Exclude),
        Err(KernelSelectionError::BenchmarkStale)
    );
    assert!(
        evaluate_stale_benchmark_policy(BenchmarkFreshness::Fresh, StaleBenchmarkPolicy::Exclude)
            .is_ok()
    );
}

#[test]
fn hysteresis_retains_active_kernel_below_threshold_and_promotes_above_it() {
    let policy = HysteresisPolicy {
        promotion_threshold_fraction: 0.05,
    };
    assert_eq!(
        evaluate_hysteresis(100.0, 99.9, &policy),
        SelectionOutcome::RetainActive
    );
    assert_eq!(
        evaluate_hysteresis(100.0, 80.0, &policy),
        SelectionOutcome::PromoteCandidate
    );
}

#[test]
fn kernel_optimization_policy_rejects_self_contradictory_deterministic_profile() {
    let policy = KernelOptimizationPolicy {
        id: KernelSelectionPolicyId::new("policy-1"),
        version: KernelSelectionPolicyVersion(1),
        profile: OptimizationProfile::Deterministic,
        ranking_strategy: RankingStrategy::WeightedScore(WeightedScorePolicy {
            weights: BTreeMap::new(),
            missing_metric_policy: MissingMetricPolicy::Exclude,
        }),
        fallback: FallbackPolicy::default(),
        hysteresis: HysteresisPolicy::default(),
        anti_flapping: AntiFlappingPolicy::default(),
        exploration: ExplorationPolicy::default(),
        selection_mode: KernelSelectionPolicy::Dynamic,
        determinism_required: false,
        require_trusted: true,
        allowed_qualification_profiles: BTreeSet::new(),
    };
    assert!(policy.validate().is_err());
}

#[test]
fn kernel_selection_error_ids_are_stable_and_match_the_proposal_vocabulary() {
    assert_eq!(
        KernelSelectionError::NoEligibleCandidates.id(),
        "kernel-selection-no-eligible-candidates"
    );
    assert_eq!(
        KernelSelectionError::PinnedKernelIneligible.id(),
        "kernel-selection-pinned-kernel-ineligible"
    );
    assert_eq!(
        KernelSelectionError::InternalError { reason: "x".into() }.id(),
        "internal-kernel-selection-error"
    );
}

#[test]
fn shape_aware_evidence_does_not_apply_outside_its_bucket() {
    assert!(performance_evidence_applies_to_workload("b1s128", "b1s128"));
    assert!(!performance_evidence_applies_to_workload(
        "b1s128", "b64s8192"
    ));
}

#[test]
fn compilation_cost_is_excluded_once_artifact_is_cached() {
    assert!(compilation_cost_excluded_from_hot_path(true));
    assert!(!compilation_cost_excluded_from_hot_path(false));
}

#[test]
fn exploration_failure_never_affects_an_unrelated_candidate() {
    let failing = selection_policy_identity("failing");
    let unrelated = selection_policy_identity("unrelated");
    assert!(!exploration_failure_affects_unrelated_candidate(
        ExplorationFailureAction::TriggerRollback,
        &failing,
        &unrelated,
    ));
}

#[test]
fn online_measurement_never_overrides_trust_or_correctness() {
    assert_eq!(
        online_measurement_cannot_override_correctness_or_trust(
            true,
            KernelArtifactTrust::Untrusted
        ),
        KernelArtifactTrust::Untrusted
    );
}

#[test]
fn selection_explanation_never_contains_native_handles() {
    let mut explanation = SelectionExplanation::default();
    explanation.exclusions.insert(
        "some-kernel".into(),
        KernelSelectionExclusionReason::PolicyDenied,
    );
    assert!(!explanation.contains_native_handles());
}

#[test]
fn kernel_selection_observation_redacts_metadata_and_carries_kernel_identity() {
    let kernel = selection_policy_kernel_id("observed");
    let observation =
        KernelSelectionObservation::new(KernelSelectionObservationKind::KernelSelected)
            .with_kernel(&kernel)
            .with_redacted_metadata("path", "C:\\secret\\path");
    assert_eq!(observation.kernel, Some(kernel.stable_key()));
    assert_eq!(
        observation
            .redacted_metadata
            .get("path")
            .map(String::as_str),
        Some("[redacted backend diagnostic]")
    );
}

#[test]
fn static_selection_required_during_warmup_needs_pinned_mode_and_kernel_step() {
    let kernel = selection_policy_kernel_id("pinned");
    let pinned_mode = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(kernel, "digest"));
    let dynamic_mode = KernelSelectionPolicy::Dynamic;
    let full_plan = ModelInstanceWarmupPlan::for_policy(ModelInstanceWarmupPolicy::Full);
    let metadata_only_plan =
        ModelInstanceWarmupPlan::for_policy(ModelInstanceWarmupPolicy::ValidateMetadataOnly);

    assert!(static_selection_required_during_warmup(
        &pinned_mode,
        &full_plan
    ));
    assert!(!static_selection_required_during_warmup(
        &dynamic_mode,
        &full_plan
    ));
    assert!(!static_selection_required_during_warmup(
        &pinned_mode,
        &metadata_only_plan
    ));
}

#[test]
fn kernel_selection_policy_conformance_report_is_conformant() {
    let report = run_kernel_selection_policy_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn benchmark_context_requires_exact_match() {
    let context = BenchmarkContext {
        provider: ProviderBinding::new("p"),
        device_architecture: "sm90".into(),
        driver_runtime_compatibility: "cuda-12".into(),
        operator_version: 1,
        artifact_digest: None,
        dtype: ComputeDType::Float32,
        layout: TensorLayoutKind::Contiguous,
        shape_bucket: "b1s128".into(),
        batch_bucket: "b1".into(),
        sequence_bucket: "s128".into(),
        execution_mode: KernelExecutionMode::Synchronous,
        benchmark_profile_version: 1,
    };
    let different_shape = BenchmarkContext {
        shape_bucket: "b64s8192".into(),
        ..context.clone()
    };
    assert!(benchmark_context_compatible(&context, &context));
    assert!(!benchmark_context_compatible(&different_shape, &context));
}

#[test]
fn anti_flapping_blocks_promotion_inside_cooldown_or_minimum_duration() {
    let policy = AntiFlappingPolicy {
        cooldown_seconds: 30,
        minimum_active_duration_seconds: 10,
    };
    assert!(!promotion_allowed_by_anti_flapping(&policy, 5, 20));
    assert!(promotion_allowed_by_anti_flapping(&policy, 30, 10));
}

#[test]
fn provider_advertised_alternative_never_overrides_runtime_cross_provider_decision() {
    let runtime_choice = selection_policy_identity("runtime-choice");
    let provider_alternative = selection_policy_identity("provider-alternative");
    assert_eq!(
        resolve_cross_provider_selection(&runtime_choice, Some(&provider_alternative)),
        runtime_choice
    );
}

#[test]
fn fallback_chain_tries_classes_in_order_and_exhausts_explicitly() {
    let policy = FallbackPolicy {
        ordered_classes: vec![KernelFallbackClass::HostExecution],
        allow_reference_cpu: false,
    };
    assert_eq!(
        evaluate_kernel_selection_fallback_chain(&policy, true, HostStagingPolicy::Permit, true),
        Err(KernelSelectionError::FallbackExhausted)
    );
    let permissive = FallbackPolicy {
        ordered_classes: vec![KernelFallbackClass::HostExecution],
        allow_reference_cpu: true,
    };
    assert_eq!(
        evaluate_kernel_selection_fallback_chain(
            &permissive,
            true,
            HostStagingPolicy::Permit,
            true
        ),
        Ok(KernelFallbackClass::HostExecution)
    );
}

#[test]
fn resolve_pinned_selection_distinguishes_unavailable_from_ineligible() {
    let kernel = selection_policy_kernel_id("pinned");
    let pin = PinnedKernelSelection::new(kernel.clone(), "digest-1");
    let unavailable = resolve_pinned_selection(&pin, &[], &[]);
    assert_eq!(
        unavailable,
        Err(KernelSelectionError::PinnedKernelUnavailable)
    );

    let discovered = vec![CandidateIdentity {
        kernel: kernel.clone(),
        provider: ProviderBinding::new("selection-policy-provider"),
        artifact_digest: None,
    }];
    let ineligible = resolve_pinned_selection(&pin, &discovered, &[]);
    assert_eq!(
        ineligible,
        Err(KernelSelectionError::PinnedKernelIneligible)
    );

    let eligible = vec![
        EligibleCandidate::from_checked(
            CandidateIdentity {
                kernel: kernel.clone(),
                provider: ProviderBinding::new("selection-policy-provider"),
                artifact_digest: None,
            },
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    assert!(resolve_pinned_selection(&pin, &discovered, &eligible).is_ok());
}

#[test]
fn canary_budget_exhaustion_is_explicit() {
    let policy = CanaryPolicy {
        max_requests: Some(100),
        max_duration_seconds: None,
        max_percentage: None,
    };
    assert!(!canary_budget_exhausted(&policy, 99, 0));
    assert!(canary_budget_exhausted(&policy, 100, 0));
}

#[test]
fn workload_context_carries_batch_and_phase_metadata_for_ranking() {
    let context = WorkloadContext {
        active_sequences: 32,
        batch_width: 32,
        total_active_tokens: 4096,
        raggedness: Some(0.2),
        phase: Some(GenerationPhase::Decode),
        kv_cache_mode: Some("paged".into()),
    };
    // Batch-aware ranking evidence is only meaningful when a candidate's
    // metrics are indexed under the matching workload bucket -- confirm the
    // context can drive that bucket key deterministically.
    let bucket = format!(
        "batch{}-seq{}",
        context.batch_width, context.total_active_tokens
    );
    assert!(performance_evidence_applies_to_workload(&bucket, &bucket));
    assert_eq!(context.phase, Some(GenerationPhase::Decode));
}
