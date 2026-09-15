//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

fn conformance_axis() -> KernelSpecializationAxis {
    KernelSpecializationAxis::new(
        KernelSpecializationAxisId::new("triton", "num-warps"),
        AxisDomain::FiniteSet(BTreeSet::from([4, 8])),
    )
}

fn conformance_kernel_id() -> KernelId {
    KernelId::new(
        ProviderBinding::new("cuda"),
        "attn",
        crate::CapabilityVersion::new(1, 0, 0),
        conformance_operator(),
        crate::KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::Cuda,
    )
}

fn conformance_template() -> KernelSpecializationTemplate {
    KernelSpecializationTemplate::new(
        KernelSpecializationTemplateId::new("attn-tile"),
        conformance_kernel_id(),
        1,
    )
    .with_axis(conformance_axis())
}

use super::*;

#[test]
fn conformance_report_is_conformant() {
    let report = run_kernel_autotuning_conformance();
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
fn axis_domain_rejects_zero_and_oversized_cardinality() {
    assert!(!AxisDomain::BoundedIntegerRange { min: 5, max: 1 }.is_bounded());
    assert!(
        !AxisDomain::BoundedIntegerRange {
            min: 0,
            max: i64::MAX,
        }
        .is_bounded()
    );
    assert!(AxisDomain::FiniteSet(BTreeSet::from([4, 8])).is_bounded());
}

#[test]
fn plan_validate_rejects_candidate_with_different_operator_semantics() {
    let attention = conformance_operator();
    let layout = OperatorId::magnetar("layout", 1, crate::OperatorFamily::Layout);
    assert!(require_specialization_preserves_operator_semantics(&attention, &attention).is_ok());
    assert!(require_specialization_preserves_operator_semantics(&attention, &layout).is_err());
    assert!(semantic_difference_requires_distinct_kernel_candidate(
        &attention, &layout
    ));
    assert!(!semantic_difference_requires_distinct_kernel_candidate(
        &attention, &attention
    ));

    let template = KernelSpecializationTemplate::new(
        KernelSpecializationTemplateId::new("attn-tile"),
        KernelId::new(
            ProviderBinding::new("cuda"),
            "attn",
            crate::CapabilityVersion::new(1, 0, 0),
            attention.clone(),
            crate::KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::Cuda,
        ),
        1,
    )
    .with_axis(conformance_axis());
    let plan = KernelAutotuningPlan {
        template,
        candidates: vec![KernelAutotuningCandidate {
            compiled_artifact: CompiledKernelArtifactId::from_digest("mismatched"),
            operator_semantics: layout,
            artifact_trust: KernelArtifactTrust::Trusted,
            quarantined: false,
            specialization: None,
            qualification_covered: true,
            memory_feasible: true,
            provider_ready: true,
            device_compatible: true,
        }],
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        objective: KernelAutotuningObjective::Latency,
        secondary_objectives: Vec::new(),
        budget: KernelAutotuningBudget::default(),
        fallback: KernelAutotuningFallback::StructuredNotReady,
    };
    assert!(matches!(
        plan.validate(),
        Err(KernelAutotuningError::SpecializationInvalid { .. })
    ));
}

#[test]
fn template_validate_rejects_undeclared_constraint_axis() {
    let axis = conformance_axis();
    let template = KernelSpecializationTemplate::new(
        KernelSpecializationTemplateId::new("attn-tile"),
        KernelId::new(
            ProviderBinding::new("cuda"),
            "attn",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("attention", 1, crate::OperatorFamily::Attention),
            crate::KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::Cuda,
        ),
        1,
    )
    .with_axis(axis)
    .with_constraint(SpecializationConstraint::Equals {
        axis: KernelSpecializationAxisId::new("triton", "undeclared"),
        value: SpecializationAxisValue::Integer(1),
    });
    assert!(template.validate().is_err());
}

#[test]
fn instantiate_rejects_out_of_domain_and_enforces_constraints() {
    let block_m = KernelSpecializationAxis::new(
        KernelSpecializationAxisId::new("triton", "block-m"),
        AxisDomain::FiniteSet(BTreeSet::from([32, 64])),
    );
    let num_warps = KernelSpecializationAxis::new(
        KernelSpecializationAxisId::new("triton", "num-warps"),
        AxisDomain::FiniteSet(BTreeSet::from([4, 8])),
    );
    let template = KernelSpecializationTemplate::new(
        KernelSpecializationTemplateId::new("attn-tile"),
        KernelId::new(
            ProviderBinding::new("cuda"),
            "attn",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("attention", 1, crate::OperatorFamily::Attention),
            crate::KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::Cuda,
        ),
        1,
    )
    .with_axis(block_m.clone())
    .with_axis(num_warps.clone())
    .with_constraint(SpecializationConstraint::Implies {
        if_axis: num_warps.id.clone(),
        if_value: SpecializationAxisValue::Integer(8),
        then_axis: block_m.id.clone(),
        then_values: BTreeSet::from([SpecializationAxisValue::Integer(64)]),
    });

    let mut invalid_assignment = BTreeMap::new();
    invalid_assignment.insert(block_m.id.clone(), SpecializationAxisValue::Integer(999));
    invalid_assignment.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(4));
    assert!(template.instantiate(invalid_assignment).is_err());

    let mut constraint_violation = BTreeMap::new();
    constraint_violation.insert(block_m.id.clone(), SpecializationAxisValue::Integer(32));
    constraint_violation.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(8));
    assert!(template.instantiate(constraint_violation).is_err());

    let mut valid = BTreeMap::new();
    valid.insert(block_m.id.clone(), SpecializationAxisValue::Integer(64));
    valid.insert(num_warps.id.clone(), SpecializationAxisValue::Integer(8));
    assert!(template.instantiate(valid).is_ok());
}

#[test]
fn instance_fingerprint_is_order_independent() {
    let a_id = KernelSpecializationAxisId::new("triton", "a");
    let b_id = KernelSpecializationAxisId::new("triton", "b");
    let mut first = BTreeMap::new();
    first.insert(a_id.clone(), SpecializationAxisValue::Integer(1));
    first.insert(b_id.clone(), SpecializationAxisValue::Integer(2));
    let mut second = BTreeMap::new();
    second.insert(b_id, SpecializationAxisValue::Integer(2));
    second.insert(a_id, SpecializationAxisValue::Integer(1));

    let template = KernelSpecializationTemplateId::new("t");
    let first_instance = KernelSpecializationInstance {
        template: template.clone(),
        template_version: 1,
        assignments: first,
    };
    let second_instance = KernelSpecializationInstance {
        template,
        template_version: 1,
        assignments: second,
    };
    assert_eq!(
        first_instance.fingerprint("digest"),
        second_instance.fingerprint("digest")
    );
}

#[test]
fn session_state_rejects_skipping_preparing() {
    let mut session = KernelAutotuningSession::new(
        KernelAutotuningSessionIdAllocator::default().allocate(),
        "plan",
        KernelAutotuningTriggerPoint::ModelInstanceWarmup,
    );
    assert!(
        session
            .transition_to(KernelAutotuningSessionState::Benchmarking)
            .is_err()
    );
    assert!(
        session
            .transition_to(KernelAutotuningSessionState::Planning)
            .is_ok()
    );
    assert!(
        session
            .transition_to(KernelAutotuningSessionState::Preparing)
            .is_ok()
    );
}

#[test]
fn budget_exceeded_reports_first_violated_dimension() {
    let budget = KernelAutotuningBudget {
        max_candidates: Some(4),
        ..KernelAutotuningBudget::default()
    };
    let usage = KernelAutotuningBudgetUsage {
        candidates_evaluated: 5,
        ..KernelAutotuningBudgetUsage::default()
    };
    assert_eq!(
        budget_exceeded(&budget, &usage),
        Some(KernelAutotuningBudgetDimension::Candidates)
    );
    assert_eq!(
        budget_exceeded(&KernelAutotuningBudget::default(), &usage),
        None
    );
}

#[test]
fn search_strategy_never_expands_candidate_domain() {
    let candidates = vec![
        KernelAutotuningCandidate {
            compiled_artifact: CompiledKernelArtifactId::from_digest("a"),
            operator_semantics: conformance_operator(),
            artifact_trust: KernelArtifactTrust::Trusted,
            quarantined: false,
            specialization: None,
            qualification_covered: true,
            memory_feasible: true,
            provider_ready: true,
            device_compatible: true,
        },
        KernelAutotuningCandidate {
            compiled_artifact: CompiledKernelArtifactId::from_digest("b"),
            operator_semantics: conformance_operator(),
            artifact_trust: KernelArtifactTrust::Untrusted,
            quarantined: false,
            specialization: None,
            qualification_covered: true,
            memory_feasible: true,
            provider_ready: true,
            device_compatible: true,
        },
    ];
    let indices = apply_search_strategy(
        &KernelAutotuningSearchStrategy::ExhaustiveBounded,
        &candidates,
    );
    assert!(indices.iter().all(|&index| index < candidates.len()));
    assert_eq!(indices, vec![0]);
}

#[test]
fn cache_hit_revalidation_rejects_stale_and_ineligible_records() {
    let mut record = KernelAutotuningRecord {
        plan_fingerprint: "plan".into(),
        target_provider: ProviderBinding::new("cuda"),
        target_provider_version: "1.0.0".into(),
        target_device_architecture: "sm90".into(),
        target_device_features: BTreeSet::new(),
        candidate_artifacts: Vec::new(),
        specialization_fingerprints: Vec::new(),
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        measurements: Vec::new(),
        winner: Some(CompiledKernelArtifactId::from_digest("winner")),
        qualification_references: Vec::new(),
        policy_version: 1,
        created_at_millis: 0,
        freshness: KernelAutotuningFreshness::Fresh,
    };
    let eligible = KernelAutotuningCacheRevalidation {
        not_revoked: true,
        trusted: true,
        qualified: true,
        provider_ready: true,
        device_available: true,
        memory_feasible: true,
        prepared_kernel_ready: true,
    };
    assert!(revalidate_cache_hit(&record, &eligible).is_ok());

    record.freshness = KernelAutotuningFreshness::Stale {
        reason: KernelAutotuningStalenessReason::DriverRuntimeUpdated,
    };
    assert!(revalidate_cache_hit(&record, &eligible).is_err());

    record.freshness = KernelAutotuningFreshness::Fresh;
    let ineligible = KernelAutotuningCacheRevalidation {
        memory_feasible: false,
        ..eligible
    };
    assert!(revalidate_cache_hit(&record, &ineligible).is_err());
}

#[test]
fn compile_specialization_instance_enforces_cold_path() {
    let template = conformance_template();
    let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
    let mut assignment = BTreeMap::new();
    assignment.insert(axis_id, SpecializationAxisValue::Integer(4));
    let instance = template.instantiate(assignment).unwrap();

    let source_artifact = crate::KernelSourceArtifact::new(
        crate::KernelSourceArtifactId::from_digest("source-digest"),
        crate::KernelSourceFormat::new("triton", "source"),
        conformance_operator(),
        crate::KernelArtifactProvenance::HumanAuthored,
    );
    let target = crate::CompilationTarget::new(
        ProviderBinding::new("cuda"),
        DeviceBinding::new(crate::DeviceId::new("gpu-0")),
        "sm90",
    );

    assert!(
        compile_specialization_instance(
            &instance,
            &source_artifact,
            b"source".to_vec(),
            target.clone(),
            crate::KernelArtifactPath::Hot,
        )
        .is_err()
    );
    let request = compile_specialization_instance(
        &instance,
        &source_artifact,
        b"source".to_vec(),
        target,
        crate::KernelArtifactPath::Cold,
    )
    .unwrap();
    assert_eq!(request.source_artifact_id, source_artifact.id);
}

#[test]
fn precompiled_bundle_matches_variant_without_recompilation() {
    let template = conformance_template();
    let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
    let mut assignment = BTreeMap::new();
    assignment.insert(axis_id, SpecializationAxisValue::Integer(8));
    let instance = template.instantiate(assignment).unwrap();

    let mut bundle = PrecompiledSpecializationBundle::new();
    assert!(bundle.match_variant(&instance, "artifact-digest").is_none());
    bundle.insert(
        &instance,
        "artifact-digest",
        CompiledKernelArtifactId::from_digest("compiled"),
    );
    assert_eq!(bundle.len(), 1);
    assert_eq!(
        bundle.match_variant(&instance, "artifact-digest"),
        Some(&CompiledKernelArtifactId::from_digest("compiled"))
    );
}

#[test]
fn preparation_time_specialization_requires_explicit_assignments() {
    let template = conformance_template();
    let axis_id = KernelSpecializationAxisId::new("triton", "num-warps");
    let mut assignment = BTreeMap::new();
    assignment.insert(axis_id, SpecializationAxisValue::Integer(4));
    let instance = template.instantiate(assignment).unwrap();

    let explicit = PreparationTimeSpecialization {
        kind: PreparationSpecializationKind::LaunchMetadataSpecialization,
        instance,
    };
    assert!(explicit.is_explicit());

    let empty_instance = KernelSpecializationInstance {
        template: template.id.clone(),
        template_version: template.version,
        assignments: BTreeMap::new(),
    };
    let implicit = PreparationTimeSpecialization {
        kind: PreparationSpecializationKind::PipelineConfiguration,
        instance: empty_instance,
    };
    assert!(!implicit.is_explicit());
}

#[test]
fn provider_execution_parameter_requires_bounded_and_covered() {
    let bounded_covered = ProviderExecutionParameter {
        name: KernelSpecializationAxisId::new("cuda", "l2-persist"),
        domain: AxisDomain::EnumeratedSymbolic(BTreeSet::from(["on".into(), "off".into()])),
        covered_by_kernel_contract: true,
    };
    assert!(bounded_covered.may_participate_in_autotuning());

    let uncovered = ProviderExecutionParameter {
        covered_by_kernel_contract: false,
        ..bounded_covered.clone()
    };
    assert!(!uncovered.may_participate_in_autotuning());

    let unbounded = ProviderExecutionParameter {
        domain: AxisDomain::BoundedIntegerRange {
            min: 0,
            max: i64::MAX,
        },
        ..bounded_covered
    };
    assert!(!unbounded.may_participate_in_autotuning());
}

#[test]
fn specialized_artifact_trust_is_re_evaluated_independently() {
    let result = require_independent_trust_evaluation(KernelArtifactTrust::Trusted, false);
    assert_eq!(result, KernelArtifactTrust::Untrusted);
    let result = require_independent_trust_evaluation(KernelArtifactTrust::Untrusted, true);
    assert_eq!(result, KernelArtifactTrust::Trusted);
}

#[test]
fn publish_autotuning_record_atomically_replaces_prior_entry() {
    let mut cache = KernelAutotuningCache::new();
    let key = KernelAutotuningCacheKey {
        operator: conformance_operator(),
        candidate_set_fingerprint: "fp".into(),
        template: KernelSpecializationTemplateId::new("attn-tile"),
        template_version: 1,
        provider_version: "1.0.0".into(),
        device_architecture: "sm90".into(),
        device_features: BTreeSet::new(),
        driver_runtime_compatibility: BTreeSet::new(),
        dtype: ComputeDType::Float16,
        layout: TensorLayoutKind::Contiguous,
        workload_fingerprint: "wl".into(),
        objective: KernelAutotuningObjective::Latency,
        policy_version: 1,
    };
    let record = KernelAutotuningRecord {
        plan_fingerprint: "plan".into(),
        target_provider: ProviderBinding::new("cuda"),
        target_provider_version: "1.0.0".into(),
        target_device_architecture: "sm90".into(),
        target_device_features: BTreeSet::new(),
        candidate_artifacts: Vec::new(),
        specialization_fingerprints: Vec::new(),
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        measurements: Vec::new(),
        winner: Some(CompiledKernelArtifactId::from_digest("winner")),
        qualification_references: Vec::new(),
        policy_version: 1,
        created_at_millis: 0,
        freshness: KernelAutotuningFreshness::Fresh,
    };
    assert!(cache.get(&key).is_none());
    publish_autotuning_record_atomically(&mut cache, &key, record.clone());
    assert_eq!(cache.get(&key), Some(&record));
    assert_eq!(cache.len(), 1);
}

#[test]
fn fixture_sources_are_all_authorized() {
    assert!(fixture_source_is_authorized(
        KernelAutotuningFixtureSource::Synthetic
    ));
    assert!(fixture_source_is_authorized(
        KernelAutotuningFixtureSource::DeterministicGenerated
    ));
    assert!(fixture_source_is_authorized(
        KernelAutotuningFixtureSource::AuthorizedBenchmarkDataset
    ));
}

#[test]
fn tuning_never_disturbs_active_continuous_batch() {
    let batch = KernelAutotuningBatchingContext {
        active_sequences: 4,
        total_active_tokens: 1024,
        raggedness_bucket: "low".into(),
        kv_cache_mode: "paged".into(),
    };
    let before = batch.clone();
    assert!(tuning_respects_active_batch(&batch));
    assert_eq!(batch, before);
}

#[test]
fn stale_tuning_result_is_never_silently_current() {
    let stale_record = KernelAutotuningRecord {
        plan_fingerprint: "plan".into(),
        target_provider: ProviderBinding::new("cuda"),
        target_provider_version: "1.0.0".into(),
        target_device_architecture: "sm90".into(),
        target_device_features: BTreeSet::new(),
        candidate_artifacts: Vec::new(),
        specialization_fingerprints: Vec::new(),
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        measurements: Vec::new(),
        winner: Some(CompiledKernelArtifactId::from_digest("winner")),
        qualification_references: Vec::new(),
        policy_version: 1,
        created_at_millis: 0,
        freshness: KernelAutotuningFreshness::Stale {
            reason: KernelAutotuningStalenessReason::PolicyVersionChanged,
        },
    };
    assert!(resolve_stale_record(&stale_record, StaleTuningReusePolicy::Ignore).is_none());
    let conservative =
        resolve_stale_record(&stale_record, StaleTuningReusePolicy::UseConservatively).unwrap();
    assert!(matches!(
        conservative.freshness,
        KernelAutotuningFreshness::Stale { .. }
    ));
    let temporary = resolve_stale_record(
        &stale_record,
        StaleTuningReusePolicy::UseTemporarilyWhileRetuning,
    )
    .unwrap();
    assert!(matches!(
        temporary.freshness,
        KernelAutotuningFreshness::Stale { .. }
    ));
}

#[test]
fn admission_lowers_priority_and_denies_under_pressure() {
    let default_policy = KernelAutotuningResourcePolicy::default();
    assert_eq!(
        evaluate_autotuning_admission(MemoryPressureLevel::Saturated, true, &default_policy),
        KernelAutotuningAdmissionDecision::Deny
    );
    assert_eq!(
        evaluate_autotuning_admission(MemoryPressureLevel::High, true, &default_policy),
        KernelAutotuningAdmissionDecision::Postpone
    );
    assert_eq!(
        evaluate_autotuning_admission(MemoryPressureLevel::Low, false, &default_policy),
        KernelAutotuningAdmissionDecision::Admit
    );

    let lower_priority_policy = KernelAutotuningResourcePolicy {
        lower_priority_under_pressure: true,
        dedicated_device: None,
    };
    assert_eq!(
        evaluate_autotuning_admission(MemoryPressureLevel::Moderate, false, &lower_priority_policy),
        KernelAutotuningAdmissionDecision::AdmitLowerPriority
    );
}

#[test]
fn dedicated_tuning_device_is_preferred_over_inference_device() {
    let inference_device = DeviceBinding::new(crate::DeviceId::new("gpu-0"));
    let dedicated_device = DeviceBinding::new(crate::DeviceId::new("gpu-1"));
    let with_dedicated = KernelAutotuningResourcePolicy {
        lower_priority_under_pressure: false,
        dedicated_device: Some(dedicated_device.clone()),
    };
    assert_eq!(
        effective_tuning_device(&with_dedicated, &inference_device),
        &dedicated_device
    );
    let without_dedicated = KernelAutotuningResourcePolicy::default();
    assert_eq!(
        effective_tuning_device(&without_dedicated, &inference_device),
        &inference_device
    );
}

#[test]
fn temporary_tuning_allocations_must_all_be_released() {
    let mut allocations = vec![
        TemporaryTuningAllocation::new(1),
        TemporaryTuningAllocation::new(2),
    ];
    assert!(!session_leaks_no_tensor_resources(&allocations));
    allocations[0].release();
    assert!(!session_leaks_no_tensor_resources(&allocations));
    allocations[1].release();
    assert!(session_leaks_no_tensor_resources(&allocations));
}

#[test]
fn tuning_only_preparation_retires_through_normal_lifecycle() {
    let mut allocator = crate::PreparedKernelIdAllocator::default();
    let prepared_id = allocator.allocate();
    let mut prepared = crate::PreparedKernel::new(
        prepared_id,
        conformance_kernel_id(),
        CompiledKernelArtifactId::from_digest("compiled"),
        ProviderBinding::new("cuda"),
        DeviceBinding::new(crate::DeviceId::new("gpu-0")),
        crate::PreparedKernelGeneration::new(1),
    );
    prepared.mark_ready().unwrap();

    let mut tuning_prep = TuningOnlyPreparation::new(prepared_id);
    assert!(!tuning_prep.retired);
    tuning_prep.retire(&mut prepared).unwrap();
    assert!(tuning_prep.retired);
    assert_eq!(prepared.state, crate::PreparedKernelState::Retiring);
}

#[test]
fn candidate_failure_policy_controls_session_continuation() {
    assert!(!candidate_failure_fails_session(
        KernelAutotuningCandidateFailurePolicy::IsolateAndContinue
    ));
    assert!(candidate_failure_fails_session(
        KernelAutotuningCandidateFailurePolicy::FailSession
    ));
}

#[test]
fn tuning_winner_promotion_requires_hysteresis_clearance() {
    let policy = crate::HysteresisPolicy::default();
    assert!(!tuning_winner_respects_hysteresis(10.0, 9.99, &policy));
    assert!(tuning_winner_respects_hysteresis(10.0, 5.0, &policy));
}

#[test]
fn provider_hint_ordering_never_adds_candidates() {
    let candidates = vec![
        KernelAutotuningCandidate {
            compiled_artifact: CompiledKernelArtifactId::from_digest("a"),
            operator_semantics: conformance_operator(),
            artifact_trust: KernelArtifactTrust::Trusted,
            quarantined: false,
            specialization: None,
            qualification_covered: true,
            memory_feasible: true,
            provider_ready: true,
            device_compatible: true,
        },
        KernelAutotuningCandidate {
            compiled_artifact: CompiledKernelArtifactId::from_digest("b"),
            operator_semantics: conformance_operator(),
            artifact_trust: KernelArtifactTrust::Trusted,
            quarantined: false,
            specialization: None,
            qualification_covered: true,
            memory_feasible: true,
            provider_ready: true,
            device_compatible: true,
        },
    ];
    let hint = KernelAutotuningProviderHint::PreferredOrder {
        artifact_order: vec![CompiledKernelArtifactId::from_digest("b")],
    };
    let indices = apply_provider_hint_ordering(&hint, &candidates);
    assert_eq!(indices.len(), candidates.len());
    assert_eq!(indices[0], 1);

    let recommended = &candidates[1];
    assert!(provider_recommended_default_is_authoritative(recommended));
    let infeasible_recommendation = KernelAutotuningCandidate {
        memory_feasible: false,
        ..candidates[0].clone()
    };
    assert!(!provider_recommended_default_is_authoritative(
        &infeasible_recommendation
    ));
}

#[test]
fn provider_native_autotuning_requires_full_boundary() {
    let authorized = KernelAutotuningProviderBoundary {
        declares_candidate_domain: true,
        cold_or_warm_path_only: true,
        respects_budget: true,
        satisfies_kernel_contract: true,
        determinism_precision_preserved: true,
    };
    assert!(evaluate_provider_native_autotuning(&authorized).is_ok());

    let unauthorized = KernelAutotuningProviderBoundary {
        cold_or_warm_path_only: false,
        ..authorized
    };
    assert!(evaluate_provider_native_autotuning(&unauthorized).is_err());
}

#[test]
fn specialized_artifact_store_dedups_by_content_digest() {
    let mut store = SpecializedArtifactStore::new();
    assert!(store.insert(CompiledKernelArtifactId::from_digest("digest-1")));
    assert_eq!(store.len(), 1);
    assert!(!store.insert(CompiledKernelArtifactId::from_digest("digest-1")));
    assert_eq!(store.len(), 1);
    assert!(store.insert(CompiledKernelArtifactId::from_digest("digest-2")));
    assert_eq!(store.len(), 2);
    assert!(store.get("digest-1").is_some());
}

#[test]
fn cross_device_reuse_requires_matching_features_and_provider_version() {
    let record = KernelAutotuningRecord {
        plan_fingerprint: "plan".into(),
        target_provider: ProviderBinding::new("cuda"),
        target_provider_version: "1.0.0".into(),
        target_device_architecture: "sm90".into(),
        target_device_features: BTreeSet::from(["fp8".to_string()]),
        candidate_artifacts: Vec::new(),
        specialization_fingerprints: Vec::new(),
        workload: conformance_workload(),
        benchmark_profile: conformance_benchmark_profile(),
        measurements: Vec::new(),
        winner: Some(CompiledKernelArtifactId::from_digest("winner")),
        qualification_references: Vec::new(),
        policy_version: 1,
        created_at_millis: 0,
        freshness: KernelAutotuningFreshness::Fresh,
    };
    assert!(tuning_result_applies_to_target(
        &record,
        "sm90",
        &BTreeSet::from(["fp8".to_string()]),
        "1.0.0",
        true
    ));
    assert!(!tuning_result_applies_to_target(
        &record,
        "sm90",
        &BTreeSet::new(),
        "1.0.0",
        true
    ));
    assert!(!tuning_result_applies_to_target(
        &record,
        "sm90",
        &BTreeSet::from(["fp8".to_string()]),
        "2.0.0",
        true
    ));
}

#[test]
fn offline_deployment_requires_no_live_tuning_when_precomputed() {
    let empty = KernelAutotuningOfflineDeployment {
        precomputed_records: Vec::new(),
        pinned_selection: None,
    };
    assert!(!empty.requires_no_live_tuning());

    let with_pinned = KernelAutotuningOfflineDeployment {
        precomputed_records: Vec::new(),
        pinned_selection: Some(CompiledKernelArtifactId::from_digest("pinned")),
    };
    assert!(with_pinned.requires_no_live_tuning());
}
