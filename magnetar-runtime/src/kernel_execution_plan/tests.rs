//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{
    CapabilityVersion, CompiledKernelArtifactId, ComputeDType, DTypeDescriptor, DeviceId,
    ExecutionGraphId, ExecutionNode, KernelImplementationFamily, KernelOperatorVersionRange,
    LayoutDescriptor, OperatorAttributeValue, OperatorFamily, OperatorId, PreparedKernel,
    PreparedKernelIdAllocator, ShapeDescriptor, TensorDescriptor, TensorEdge, TensorEdgeId,
};

#[test]
fn prepared_execution_plan_lifecycle_enforces_state_order() {
    let mut plan = test_plan();

    assert_eq!(plan.state, PreparedExecutionPlanState::Building);
    assert!(!plan.accepts_new_work());
    assert!(matches!(
        plan.acquire_lease(),
        Err(PreparedExecutionPlanError::PlanNotReady {
            state: PreparedExecutionPlanState::Building
        })
    ));

    plan.transition_to(PreparedExecutionPlanState::Validating)
        .unwrap();
    plan.transition_to(PreparedExecutionPlanState::Preparing)
        .unwrap();
    plan.transition_to(PreparedExecutionPlanState::Ready)
        .unwrap();
    assert!(plan.accepts_new_work());

    let lease = plan.acquire_lease().unwrap();
    assert_eq!(plan.active_leases(), 1);
    plan.transition_to(PreparedExecutionPlanState::Retiring)
        .unwrap();
    assert!(matches!(
        plan.transition_to(PreparedExecutionPlanState::Retired),
        Err(PreparedExecutionPlanError::PlanGenerationInUse { active_leases: 1 })
    ));
    plan.release_lease(lease).unwrap();
    plan.transition_to(PreparedExecutionPlanState::Retired)
        .unwrap();
}

#[test]
fn prepared_execution_plan_id_rejects_pointer_shaped_identity() {
    assert!(PreparedExecutionPlanId::new("decode-plan-1").is_ok());
    assert!(PreparedExecutionPlanId::new("0x7ffee").is_err());
    assert!(PreparedExecutionPlanId::new("native-handle").is_err());
}

#[test]
fn plan_scope_validates_model_instance_revision_adapter_and_policy() {
    let instance = ModelInstanceId::new("model-decode").unwrap();
    let adapter = AdapterRevision::new("adapter-r1").unwrap();
    let scope = PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode)
        .with_model_instance(instance.clone(), 7)
        .with_adapter_revision(adapter.clone())
        .with_execution_policy_revision("policy-r2")
        .with_workload_bucket("decode-short");

    assert!(scope.validate().is_ok());
    assert!(
        scope
            .validate_model_binding(&instance, 7, Some(&adapter), Some("policy-r2"))
            .is_ok()
    );
    assert!(matches!(
        scope.validate_model_binding(&instance, 8, Some(&adapter), Some("policy-r2")),
        Err(PreparedExecutionPlanError::ModelInstanceRevisionMismatch)
    ));
    assert!(matches!(
        scope.validate_model_binding(&instance, 7, None, Some("policy-r2")),
        Err(PreparedExecutionPlanError::AdapterRevisionMismatch)
    ));
    assert!(matches!(
        scope.validate_model_binding(&instance, 7, Some(&adapter), Some("policy-r3")),
        Err(PreparedExecutionPlanError::ExecutionPolicyRevisionMismatch)
    ));
}

#[test]
fn plan_scope_captures_phase_shape_dtype_layout_provider_and_modes() {
    let mut scope = PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Prefill)
        .with_workload_bucket("prefill-small");
    scope.shape_envelopes.push(PlanShapeEnvelope::new([
        ShapeDimensionEnvelope::Range { min: 1, max: 8 },
        ShapeDimensionEnvelope::Exact(128),
    ]));
    scope
        .dtypes
        .insert(DTypeDescriptor::portable(ComputeDType::Float32));
    scope.layouts.insert(TensorLayoutKind::Contiguous);
    scope.batching_mode = Some("continuous".into());
    scope.kv_cache_mode = Some("paged".into());
    scope.provider = Some(ProviderBinding::new("reference-cpu"));
    scope.quantization_mode = Some("none".into());

    assert!(scope.validate().is_ok());
    assert_eq!(scope.phase, PreparedExecutionPhase::Prefill);
    assert!(scope.shape_envelopes[0].contains(&[4, 128]));
    assert!(!scope.shape_envelopes[0].contains(&[9, 128]));
    assert!(scope.provider.is_some());
}

#[test]
fn plan_scope_rejects_incomplete_model_binding_and_bad_shape_envelope() {
    let mut incomplete = PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode);
    incomplete.model_instance_revision = Some(1);
    assert!(matches!(
        incomplete.validate(),
        Err(PreparedExecutionPlanError::ModelBindingIncomplete)
    ));

    let mut bad_shape = PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode);
    bad_shape
        .shape_envelopes
        .push(PlanShapeEnvelope::new([ShapeDimensionEnvelope::Range {
            min: 8,
            max: 1,
        }]));
    assert!(matches!(
        bad_shape.validate(),
        Err(PreparedExecutionPlanError::InvalidShapeEnvelope)
    ));
}

#[test]
fn plan_node_binding_records_exact_kernel_artifact_and_specialization() {
    let kernel = test_kernel("attention", 1);
    let provider = kernel.provider.clone();
    let mut allocator = PreparedKernelIdAllocator::default();
    let prepared = allocator.allocate();
    let prepared_generation = PreparedKernelGeneration::new(3);
    let binding = PlanNodeBinding::new(
        [ExecutionNodeId::new("attention-0")],
        kernel.clone(),
        provider.clone(),
    )
    .unwrap()
    .with_qualification_profile("standard")
    .with_artifact_digest("sha256:kernel-a")
    .with_specialization("block-64")
    .with_device(DeviceBinding::new(DeviceId::new("cpu-0")))
    .with_prepared_kernel(prepared, prepared_generation);

    assert!(binding.validate().is_ok());
    assert_eq!(binding.provider, provider);
    assert_eq!(binding.device.as_ref().unwrap().to_string(), "cpu-0");
    assert!(binding.prepared_kernel.is_some());
    assert!(
        binding
            .validate_exact_binding(
                &kernel,
                Some("standard"),
                Some("sha256:kernel-a"),
                Some("block-64"),
                Some(prepared_generation)
            )
            .is_ok()
    );
}

#[test]
fn plan_node_binding_rejects_implicit_latest_kernel_substitution() {
    let selected = test_kernel("attention", 1);
    let latest = test_kernel("attention", 2);
    let binding = PlanNodeBinding::new(
        [ExecutionNodeId::new("attention-0")],
        selected.clone(),
        selected.provider.clone(),
    )
    .unwrap()
    .with_qualification_profile("standard")
    .with_artifact_digest("sha256:selected")
    .with_specialization("block-64");

    assert!(matches!(
        binding.validate_exact_binding(
            &latest,
            Some("standard"),
            Some("sha256:selected"),
            Some("block-64"),
            None
        ),
        Err(PreparedExecutionPlanError::KernelBindingMismatch)
    ));
    assert!(matches!(
        binding.validate_exact_binding(
            &selected,
            Some("different-profile"),
            Some("sha256:selected"),
            Some("block-64"),
            None
        ),
        Err(PreparedExecutionPlanError::QualificationProfileMismatch)
    ));
    assert!(matches!(
        binding.validate_exact_binding(
            &selected,
            Some("standard"),
            Some("sha256:latest"),
            Some("block-64"),
            None
        ),
        Err(PreparedExecutionPlanError::KernelArtifactDigestMismatch)
    ));
    assert!(matches!(
        binding.validate_exact_binding(
            &selected,
            Some("standard"),
            Some("sha256:selected"),
            Some("block-128"),
            None
        ),
        Err(PreparedExecutionPlanError::SpecializationMismatch)
    ));
}

#[test]
fn plan_fingerprint_changes_when_node_binding_changes() {
    let mut plan = test_plan();
    let original = plan.fingerprint.clone();
    let kernel = test_kernel("matmul", 1);
    let binding = PlanNodeBinding::new(
        [ExecutionNodeId::new("matmul")],
        kernel.clone(),
        kernel.provider.clone(),
    )
    .unwrap()
    .with_qualification_profile("standard")
    .with_artifact_digest("sha256:kernel-a");

    plan.add_node_binding(binding).unwrap();

    assert_ne!(original, plan.fingerprint);
}

#[test]
fn native_handle_boundary_rejects_and_redacts_pointer_shaped_data() {
    assert!(ProviderPreparedSegmentId::new("cuda-graph-1").is_ok());
    assert!(ProviderPreparedSegmentId::new("0xdeadbeef").is_err());
    assert!(PlanResourceSlotId::new("input:hidden-state").is_ok());
    assert_eq!(redact_plan_diagnostic("ptr=0xdeadbeef"), "[redacted]");
    assert_eq!(redact_plan_diagnostic("logical-plan-id"), "logical-plan-id");
}

#[test]
fn prepared_execution_segment_validates_lifecycle_and_semantics() {
    let segment = PreparedExecutionSegment::new(
        PreparedExecutionSegmentId::new("segment-1").unwrap(),
        [ExecutionNodeId::new("attention-0")],
        ProviderBinding::new("reference-cpu"),
    )
    .unwrap()
    .with_device(DeviceBinding::new(DeviceId::new("cpu-0")))
    .with_provider_prepared_segment(ProviderPreparedSegmentId::new("segment-state-1").unwrap());

    assert!(segment.validate().is_ok());
    assert_eq!(segment.provider_state, ProviderPreparedSegmentState::Ready);

    let mut invalid = segment.clone();
    invalid.preserves_graph_semantics = false;
    assert!(matches!(
        invalid.validate(),
        Err(PreparedExecutionPlanError::SegmentSemanticMismatch)
    ));

    let mut lifecycle = segment;
    lifecycle.invalidate().unwrap();
    lifecycle.destroy().unwrap();
    assert_eq!(
        lifecycle.provider_state,
        ProviderPreparedSegmentState::Destroyed
    );
}

#[test]
fn cross_provider_segments_require_explicit_movement_and_affinity_policy() {
    let mut segment = PreparedExecutionSegment::new(
        PreparedExecutionSegmentId::new("segment-cross").unwrap(),
        [ExecutionNodeId::new("matmul")],
        ProviderBinding::new("cuda"),
    )
    .unwrap();
    segment.device = Some(DeviceBinding::new(DeviceId::new("gpu-0")));
    segment.resource_affinity = Some(ResourceAffinity::new(crate::FallbackClass::ProviderPinned));
    segment
        .explicit_data_movement
        .push("cuda-to-cpu-copy".into());
    segment.host_staging_allowed = true;

    assert!(segment.validate().is_ok());
    assert!(!segment.explicit_data_movement.is_empty());
    assert!(segment.host_staging_allowed);
}

#[test]
fn resource_binding_plan_distinguishes_stable_and_dynamic_slots() {
    let mut resources = ResourceBindingPlan::default();
    resources
        .add_slot(PlanResourceSlot::new(
            PlanResourceSlotId::new("model:weights").unwrap(),
            PlanResourceSlotKind::ModelWeight,
            PlanResourceSlotStability::Stable,
            PlanResourceOwner::MemoryManager,
        ))
        .unwrap();
    resources
        .add_slot(PlanResourceSlot::new(
            PlanResourceSlotId::new("workspace:attention").unwrap(),
            PlanResourceSlotKind::Workspace,
            PlanResourceSlotStability::Dynamic,
            PlanResourceOwner::Invocation,
        ))
        .unwrap();
    resources
        .add_slot(PlanResourceSlot::new(
            PlanResourceSlotId::new("session:kv-key").unwrap(),
            PlanResourceSlotKind::KvKey,
            PlanResourceSlotStability::Dynamic,
            PlanResourceOwner::Session,
        ))
        .unwrap();

    assert!(resources.validate().is_ok());

    let captured_session_resource = PlanResourceSlot::new(
        PlanResourceSlotId::new("session:kv-value").unwrap(),
        PlanResourceSlotKind::KvValue,
        PlanResourceSlotStability::Dynamic,
        PlanResourceOwner::Session,
    )
    .with_bound_resource(TensorResourceId::new("session-1-kv"));
    assert!(matches!(
        captured_session_resource.validate(),
        Err(PreparedExecutionPlanError::DynamicResourceCaptured)
    ));
}

#[test]
fn kv_and_memory_requirements_preserve_runtime_ownership() {
    let memory = PlanMemoryRequirements {
        workspace_upper_bound_bytes: Some(4096),
        allocation_lifetime: PlanAllocationLifetime::BatchQuantum,
        reuse: PlanBufferReuse::NonOverlappingIntermediates,
        placement: Some(ResourceAffinity::new(crate::FallbackClass::Transparent)),
        preserves_memory_manager_authority: true,
    };
    assert!(memory.validate().is_ok());

    let kv = PlanKvCacheRequirements {
        layout: "paged-kv-v1".into(),
        affinity: Some(ResourceAffinity::new(crate::FallbackClass::Transparent)),
        append_required: true,
        read_required: true,
        contents_owned_by_session: true,
    };
    assert!(kv.validate().is_ok());

    let captured = PlanKvCacheRequirements {
        contents_owned_by_session: false,
        ..kv
    };
    assert!(matches!(
        captured.validate(),
        Err(PreparedExecutionPlanError::KvContentsCaptured)
    ));
}

#[test]
fn plan_guards_cover_shape_dtype_layout_phase_batch_sequence_kv_and_readiness() {
    let adapter = AdapterRevision::new("adapter-r1").unwrap();
    let mut allowed_dtypes = BTreeSet::new();
    allowed_dtypes.insert(DTypeDescriptor::portable(ComputeDType::Float32));
    let mut allowed_layouts = BTreeSet::new();
    allowed_layouts.insert(TensorLayoutKind::Contiguous);
    let guards = vec![
        PlanGuard::Shape(PlanShapeEnvelope::new([
            ShapeDimensionEnvelope::Range { min: 1, max: 8 },
            ShapeDimensionEnvelope::Exact(128),
        ])),
        PlanGuard::DType(allowed_dtypes),
        PlanGuard::Layout(allowed_layouts),
        PlanGuard::Phase(PreparedExecutionPhase::Decode),
        PlanGuard::BatchRange { min: 1, max: 8 },
        PlanGuard::SequenceRange { min: 1, max: 4096 },
        PlanGuard::ActiveSequences { max: 16 },
        PlanGuard::TotalTokens { max: 8192 },
        PlanGuard::Raggedness { allowed: true },
        PlanGuard::PagedKv { required: true },
        PlanGuard::AdapterRevision(adapter.clone()),
        PlanGuard::KvLayout("paged-kv-v1".into()),
        PlanGuard::AffinityRequired,
        PlanGuard::Readiness,
        PlanGuard::MemoryFeasible,
    ];
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![4, 128];
    context.dtype = Some(DTypeDescriptor::portable(ComputeDType::Float32));
    context.layout = Some(TensorLayoutKind::Contiguous);
    context.batch_size = Some(4);
    context.sequence_length = Some(2048);
    context.active_sequences = Some(8);
    context.total_tokens = Some(4096);
    context.ragged = true;
    context.paged_kv = true;
    context.adapter_revision = Some(adapter);
    context.kv_layout = Some("paged-kv-v1".into());
    context.affinity = Some(ResourceAffinity::new(crate::FallbackClass::Transparent));

    let report = evaluate_plan_guards(&guards, &context).unwrap();
    assert_eq!(report.checked_guards, guards.len());
    assert!(report.is_hot_path_bounded());

    context.sequence_length = Some(8192);
    assert!(matches!(
        evaluate_plan_guards(&guards, &context),
        Err(PreparedExecutionPlanError::PlanWorkloadIncompatible)
    ));
}

#[test]
fn guard_failure_routes_to_alternate_replan_or_explicit_fallback() {
    let request = PlanRebuildRequest {
        reason: PlanRebuildReason::GuardFailed,
        desired_scope: PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode),
        urgency: PlanRebuildUrgency::RequiredBeforeNewWork,
    };

    assert!(matches!(
        handle_guard_failure(
            Some(PreparedExecutionPlanId::new("alternate").unwrap()),
            request.clone(),
            false
        ),
        PlanFailureAction::UseAlternatePlan(_)
    ));
    assert!(matches!(
        handle_guard_failure(None, request.clone(), true),
        PlanFailureAction::ExplicitFallback(_)
    ));
    assert!(matches!(
        handle_guard_failure(None, request, false),
        PlanFailureAction::RequestReplan(_)
    ));
}

#[test]
fn ready_plan_execution_uses_bounded_dispatch_path() {
    let mut plan = ready_plan("decode-ready", "attention", 1);
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![1, 128];

    let report = plan.execute_ready_path(&context).unwrap();

    assert_eq!(report.dispatched_kernels, 1);
    assert!(report.avoids_full_hot_path_rebuild());
    assert_eq!(plan.active_leases(), 0);
}

#[test]
fn prepared_plan_executor_resolves_node_binding_without_registry_selection() {
    let graph = test_graph(1);
    let mut plan = ready_graph_plan(&graph, "matmul", 1);
    let registry = registry_with_prepared_binding(&plan.node_bindings[0]);
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![1, 128];

    let execution = PreparedExecutionPlanExecutor::new()
        .prepare_node_execution(
            &graph,
            &mut plan,
            &registry,
            &context,
            &ExecutionNodeId::new("matmul"),
        )
        .unwrap();

    assert_eq!(execution.graph_node, ExecutionNodeId::new("matmul"));
    assert_eq!(execution.kernel, plan.node_bindings[0].kernel);
    assert_eq!(
        execution.prepared_kernel,
        plan.node_bindings[0].prepared_kernel.unwrap()
    );
    assert_eq!(execution.provider, ProviderBinding::new("reference-cpu"));
    assert_eq!(plan.active_leases(), 0);
}

#[test]
fn prepared_plan_executor_rejects_missing_graph_node_binding() {
    let graph = test_graph(1);
    let mut plan = ready_graph_plan(&graph, "matmul", 1);
    let registry = registry_with_prepared_binding(&plan.node_bindings[0]);
    let context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);

    assert!(matches!(
        PreparedExecutionPlanExecutor::new().prepare_node_execution(
            &graph,
            &mut plan,
            &registry,
            &context,
            &ExecutionNodeId::new("absent-node"),
        ),
        Err(PreparedExecutionPlanError::PlanNodeBindingMissing)
    ));
}

#[test]
fn prepared_plan_executor_rejects_missing_prepared_kernel() {
    let graph = test_graph(1);
    let mut plan = ready_graph_plan(&graph, "matmul", 1);
    let registry = KernelRegistry::new();
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![1, 128];

    assert!(matches!(
        PreparedExecutionPlanExecutor::new().prepare_node_execution(
            &graph,
            &mut plan,
            &registry,
            &context,
            &ExecutionNodeId::new("matmul"),
        ),
        Err(PreparedExecutionPlanError::PlanPreparedKernelMissing)
    ));
}

#[test]
fn stale_plan_remains_distinct_from_invalidated_new_work_policy() {
    let mut plan = ready_plan("stale-plan", "attention", 1);
    plan.mark_stale(
        PlanRebuildReason::KernelPromotion,
        PlanRebuildUrgency::Background,
    )
    .unwrap();
    assert_eq!(plan.state, PreparedExecutionPlanState::Stale);
    assert!(plan.accepts_new_work());

    plan.hard_invalidate(PlanRebuildReason::KernelRevoked)
        .unwrap();
    assert_eq!(plan.state, PreparedExecutionPlanState::Invalidated);
    assert!(!plan.accepts_new_work());
    assert!(plan.state.requires_replacement_for_new_work());
}

#[test]
fn stale_plan_outside_rebuild_policy_refuses_execution_but_still_accepts_new_work() {
    let mut plan = ready_plan("stale-outside-policy-plan", "attention", 1);
    plan.mark_stale(
        PlanRebuildReason::KernelRevoked,
        PlanRebuildUrgency::RequiredBeforeNewWork,
    )
    .unwrap();
    assert_eq!(plan.state, PreparedExecutionPlanState::Stale);
    // `accepts_new_work` is a coarse state check (Ready|Stale) used by
    // plan-cache lookups; it does not by itself guarantee execution is
    // allowed -- `execute_ready_path` enforces the finer-grained rebuild
    // policy.
    assert!(plan.accepts_new_work());
    assert!(plan.is_stale_outside_policy());

    let context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    assert!(matches!(
        plan.execute_ready_path(&context),
        Err(PreparedExecutionPlanError::PlanStaleOutsidePolicy)
    ));
}

#[test]
fn cache_lookup_invalidation_rebuild_dedup_and_restart_revalidation_work() {
    let plan = ready_plan("cache-plan", "attention", 1);
    let family = PlanFamilyKey::from_plan(&plan);
    let context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    let kernel = plan.node_bindings[0].kernel.clone();
    let mut cache = PreparedExecutionPlanCache::default();
    let plan_id = plan.id.clone();
    cache.insert(plan);

    assert!(cache.lookup_ready(&family, &context).is_some());
    assert_eq!(cache.invalidate_kernel(&kernel), vec![plan_id.clone()]);
    assert!(cache.lookup_ready(&family, &context).is_none());

    let request = PlanRebuildRequest {
        reason: PlanRebuildReason::KernelRevoked,
        desired_scope: PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode),
        urgency: PlanRebuildUrgency::RequiredBeforeNewWork,
    };
    assert!(cache.request_rebuild(request.clone()));
    assert!(!cache.request_rebuild(request));

    let mut restart_cache = PreparedExecutionPlanCache::default();
    restart_cache.insert(ready_plan("restart-plan", "attention", 1));
    let hard = PlanHardDependencyStatus {
        revocation_clear: false,
        ..PlanHardDependencyStatus::default()
    };
    let restart_id = PreparedExecutionPlanId::new("restart-plan").unwrap();
    assert!(matches!(
        restart_cache.revalidate_cached_plan(&restart_id, &hard),
        Err(PreparedExecutionPlanError::PlanKernelRevoked)
    ));
}

#[test]
fn registry_preference_change_after_publication_does_not_alter_plan_execution() {
    let graph = test_graph(1);
    let mut plan = ready_graph_plan(&graph, "matmul", 1);
    let bound_kernel = plan.node_bindings[0].kernel.clone();
    let mut registry = registry_with_prepared_binding(&plan.node_bindings[0]);
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![1, 128];
    let executor = PreparedExecutionPlanExecutor::new();
    let node = ExecutionNodeId::new("matmul");

    let before = executor
        .prepare_node_execution(&graph, &mut plan, &registry, &context, &node)
        .unwrap();
    assert_eq!(before.kernel, bound_kernel);

    // A registry preference change after the plan was published: a newer
    // generation of the same logical Kernel becomes the registry's
    // active/preferred one for *new* selection.
    let newer = test_kernel("matmul", 2);
    let mut allocator = PreparedKernelIdAllocator::default();
    // Skip the id `prepared_binding` would have allocated first for its
    // own fresh allocator (both start the same deterministic sequence),
    // so this registration cannot collide with the plan's own
    // `PreparedKernelId`.
    let _ = allocator.allocate();
    let newer_id = allocator.allocate();
    let mut newer_prepared = PreparedKernel::new(
        newer_id,
        newer.clone(),
        CompiledKernelArtifactId::from_digest("test:matmul-newer"),
        newer.provider.clone(),
        DeviceBinding::new(DeviceId::new("cpu-0")),
        PreparedKernelGeneration::new(2),
    );
    newer_prepared.mark_ready().unwrap();
    registry.register_prepared_kernel(newer_prepared);
    registry.promote_generation(&newer, newer_id).unwrap();
    assert_eq!(
        registry.active_prepared_kernel(&newer).unwrap().id,
        newer_id
    );

    // Execution of the already-published plan must still resolve
    // through its own `PlanNodeBinding`/`PreparedKernelId`, not the
    // registry's newly promoted preference.
    let after = executor
        .prepare_node_execution(&graph, &mut plan, &registry, &context, &node)
        .unwrap();
    assert_eq!(after.kernel, bound_kernel);
    assert_ne!(bound_kernel, newer);
}

#[test]
fn kernel_revocation_blocks_new_work_while_in_flight_lease_still_completes() {
    let graph = test_graph(1);
    let mut plan = ready_graph_plan(&graph, "matmul", 1);
    let registry = registry_with_prepared_binding(&plan.node_bindings[0]);
    let mut context = PlanGuardContext::for_phase(PreparedExecutionPhase::Decode);
    context.shape = vec![1, 128];
    let executor = PreparedExecutionPlanExecutor::new();
    let node = ExecutionNodeId::new("matmul");
    let prepared_kernel_id = plan.node_bindings[0].prepared_kernel.unwrap();

    // Work admitted before revocation acquires a plan-level lease --
    // this is the "in-flight" work the revocation policy must not break.
    let in_flight_lease = plan.acquire_lease().unwrap();

    // Kernel revocation: the Prepared Kernel this plan's binding
    // resolves to is retired (no longer dispatchable for new work), the
    // same transition `promote_generation` uses ahead of destroying a
    // superseded generation.
    let mut registry = registry;
    registry
        .retire_prepared_kernel(&prepared_kernel_id)
        .unwrap();
    assert!(
        !registry
            .prepared_kernel(&prepared_kernel_id)
            .unwrap()
            .state
            .is_dispatchable()
    );

    // New work is blocked: preparing this node's execution again now
    // fails because the bound Prepared Kernel is no longer dispatchable.
    assert!(matches!(
        executor.prepare_node_execution(&graph, &mut plan, &registry, &context, &node),
        Err(PreparedExecutionPlanError::PlanKernelRevoked)
    ));

    // The lease acquired before revocation is still valid and completes
    // normally -- releasing it only checks plan/generation identity, not
    // current Kernel dispatchability.
    plan.release_lease(in_flight_lease).unwrap();
}

#[test]
fn persisted_plan_is_recipe_and_strips_prepared_kernel_state() {
    let plan = ready_plan("persisted-plan", "attention", 1);
    assert!(plan.node_bindings[0].prepared_kernel.is_some());

    let persisted = PersistedPreparedExecutionPlan::from_plan(&plan);
    assert!(persisted.node_bindings[0].prepared_kernel.is_none());
    let recipe = persisted.into_recipe().unwrap();
    assert_eq!(recipe.state, PreparedExecutionPlanState::Building);
    assert!(recipe.node_bindings[0].prepared_kernel.is_none());
}

#[test]
fn provider_prepared_segment_capability_is_optional_and_opaque() {
    let mut segment = PreparedExecutionSegment::new(
        PreparedExecutionSegmentId::new("segment-provider").unwrap(),
        [ExecutionNodeId::new("attention")],
        ProviderBinding::new("reference-cpu"),
    )
    .unwrap();
    let unsupported = ProviderPreparedSegmentCapability {
        provider: ProviderBinding::new("reference-cpu"),
        advertised: false,
    };
    assert_eq!(unsupported.prepare_segment(&mut segment).unwrap(), None);
    assert_eq!(
        segment.fallback,
        SegmentCaptureFallback::IndividualKernelDispatch
    );

    let supported = ProviderPreparedSegmentCapability {
        provider: ProviderBinding::new("reference-cpu"),
        advertised: true,
    };
    assert!(supported.prepare_segment(&mut segment).unwrap().is_some());
    assert!(segment.provider_prepared_segment.is_some());
}

#[test]
fn atomic_replacement_preserves_in_flight_generation() {
    let mut old = ready_plan("atomic-old", "attention", 1);
    let lease = old.acquire_lease().unwrap();
    let new = ready_plan("atomic-new", "attention", 2);
    let mut set = PreparedExecutionPlanGenerationSet::default();
    set.publish_ready(old).unwrap();
    set.publish_ready(new).unwrap();

    assert_eq!(set.active().unwrap().id.as_str(), "atomic-new");
    assert_eq!(set.retiring.len(), 1);
    assert_eq!(set.retiring[0].state, PreparedExecutionPlanState::Retiring);
    assert_eq!(set.retiring[0].active_leases(), 1);
    assert!(matches!(
        set.retiring[0].transition_to(PreparedExecutionPlanState::Retired),
        Err(PreparedExecutionPlanError::PlanGenerationInUse { active_leases: 1 })
    ));
    set.retiring[0].release_lease(lease).unwrap();
    set.retiring[0]
        .transition_to(PreparedExecutionPlanState::Retired)
        .unwrap();
}

#[test]
fn plan_build_pipeline_runs_cold_path_and_denies_ai_or_campaign_launch() {
    let graph = test_graph(1);
    let binding = prepared_binding("matmul", 1);
    let (plan, report) = build_prepared_execution_plan_from_decisions(
        &graph,
        PreparedExecutionPlanId::new("pipeline-plan").unwrap(),
        PreparedExecutionPlanGeneration::new(1),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Warmup),
        binding,
        &PlanBuildPolicy::default(),
    )
    .unwrap();

    assert_eq!(plan.state, PreparedExecutionPlanState::Ready);
    assert!(report.graph_validated);
    assert!(report.registry_queried);
    assert!(report.eligibility_applied);
    assert!(report.selection_policy_applied);
    assert!(report.specialization_resolved);
    assert!(report.autotuning_evidence_consumed);
    assert!(report.memory_plan_built);
    assert!(report.kernels_prepared);
    assert!(report.final_plan_validated);
    assert!(report.ready_published);
    assert!(!report.ai_generation_launched);
    assert!(!report.optimization_campaign_launched);

    let denied = PlanBuildPolicy {
        allow_ai_generation: true,
        ..PlanBuildPolicy::default()
    };
    assert!(matches!(
        build_prepared_execution_plan_from_decisions(
            &graph,
            PreparedExecutionPlanId::new("denied-plan").unwrap(),
            PreparedExecutionPlanGeneration::new(1),
            PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Warmup),
            prepared_binding("matmul", 1),
            &denied,
        ),
        Err(PreparedExecutionPlanError::PlanHotPathRebuildDenied)
    ));
}

#[test]
fn plan_error_codes_and_observability_are_structured_and_redacted() {
    assert_eq!(
        PreparedExecutionPlanError::PlanPreparedKernelMissing.id(),
        "kernel-execution-plan-prepared-kernel-missing"
    );
    assert_eq!(
        PreparedExecutionPlanError::PlanWorkloadIncompatible.id(),
        "kernel-execution-plan-workload-incompatible"
    );

    let observation = PreparedExecutionPlanObservation::new(
        PreparedExecutionPlanObservationKind::PlanGuardFailed,
        PreparedExecutionPlanId::new("obs-plan").unwrap(),
        PreparedExecutionPlanGeneration::new(1),
    )
    .with_redacted_metadata("native", "handle=0xdeadbeef")
    .with_redacted_metadata("reason", "sequence-too-long");

    assert_eq!(observation.redacted_metadata["native"], "[redacted]");
    assert_eq!(observation.redacted_metadata["reason"], "sequence-too-long");
}

#[test]
fn conformance_properties_are_exercised_by_contract_surface() {
    let mut plan = ready_plan("conformance-plan", "attention", 1);
    let original_binding = plan.node_bindings[0].clone();
    let persisted = PersistedPreparedExecutionPlan::from_plan(&plan);
    assert!(persisted.node_bindings[0].prepared_kernel.is_none());
    assert!(plan.memory_requirements.preserves_memory_manager_authority);

    plan.mark_stale(
        PlanRebuildReason::PerformanceRegression,
        PlanRebuildUrgency::Background,
    )
    .unwrap();
    assert_eq!(plan.node_bindings[0], original_binding);
    plan.hard_invalidate(PlanRebuildReason::TrustDenied)
        .unwrap();
    assert!(matches!(
        plan.execute_ready_path(&PlanGuardContext::for_phase(PreparedExecutionPhase::Decode)),
        Err(PreparedExecutionPlanError::PlanNotReadyForExecution)
    ));
}

#[test]
fn semantic_graph_fingerprint_is_deterministic_and_semantic() {
    let graph = test_graph(1);
    let rebuilt = test_graph(1);
    let changed_operator = test_graph(2);

    assert_eq!(
        semantic_graph_fingerprint(&graph),
        semantic_graph_fingerprint(&rebuilt)
    );
    assert_ne!(
        semantic_graph_fingerprint(&graph),
        semantic_graph_fingerprint(&changed_operator)
    );
}

#[test]
fn semantic_graph_fingerprint_includes_topology_and_tensor_descriptors() {
    let graph = test_graph(1);
    let extra_edge = TensorEdge::new(
        TensorEdgeId::new("logits"),
        TensorDescriptor::new(
            ShapeDescriptor::new([1, 32000]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    );
    let changed = graph.clone().with_edge(extra_edge);

    assert_ne!(
        semantic_graph_fingerprint(&graph),
        semantic_graph_fingerprint(&changed)
    );
}

fn test_plan() -> PreparedExecutionPlan {
    PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new("decode-plan").unwrap(),
        PreparedExecutionPlanGeneration::new(1),
        ExecutionGraphSemanticFingerprint::new("sha256:test").unwrap(),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode),
    )
    .unwrap()
}

fn ready_plan(id: &str, kernel_name: &str, kernel_patch: u64) -> PreparedExecutionPlan {
    let mut plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new(id).unwrap(),
        PreparedExecutionPlanGeneration::new(kernel_patch),
        ExecutionGraphSemanticFingerprint::new(format!("sha256:{id}")).unwrap(),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode)
            .with_workload_bucket("decode"),
    )
    .unwrap();
    plan.add_node_binding(prepared_binding(kernel_name, kernel_patch))
        .unwrap();
    plan.add_guard(PlanGuard::Phase(PreparedExecutionPhase::Decode));
    plan.add_guard(PlanGuard::Shape(PlanShapeEnvelope::new([
        ShapeDimensionEnvelope::Range { min: 1, max: 8 },
        ShapeDimensionEnvelope::Exact(128),
    ])));
    plan.mark_ready_atomically().unwrap();
    plan
}

fn ready_graph_plan(
    graph: &ExecutionGraph,
    kernel_name: &str,
    kernel_patch: u64,
) -> PreparedExecutionPlan {
    let mut plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new(format!("graph-{kernel_name}-plan")).unwrap(),
        PreparedExecutionPlanGeneration::new(kernel_patch),
        semantic_graph_fingerprint(graph),
        PreparedExecutionPlanScope::for_phase(PreparedExecutionPhase::Decode)
            .with_workload_bucket("decode"),
    )
    .unwrap();
    plan.add_node_binding(prepared_binding(kernel_name, kernel_patch))
        .unwrap();
    plan.add_guard(PlanGuard::Phase(PreparedExecutionPhase::Decode));
    plan.add_guard(PlanGuard::Shape(PlanShapeEnvelope::new([
        ShapeDimensionEnvelope::Range { min: 1, max: 8 },
        ShapeDimensionEnvelope::Exact(128),
    ])));
    plan.mark_ready_atomically().unwrap();
    plan
}

fn registry_with_prepared_binding(binding: &PlanNodeBinding) -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    let mut prepared = PreparedKernel::new(
        binding.prepared_kernel.unwrap(),
        binding.kernel.clone(),
        CompiledKernelArtifactId::from_digest(format!("test:{}", binding.kernel.stable_key())),
        binding.provider.clone(),
        binding
            .device
            .clone()
            .unwrap_or_else(|| DeviceBinding::new(DeviceId::new("cpu-0"))),
        binding.prepared_kernel_generation.unwrap(),
    );
    prepared.mark_ready().unwrap();
    registry.register_prepared_kernel(prepared);
    registry
}

fn prepared_binding(kernel_name: &str, kernel_patch: u64) -> PlanNodeBinding {
    let kernel = test_kernel(kernel_name, kernel_patch);
    let mut allocator = PreparedKernelIdAllocator::default();
    PlanNodeBinding::new(
        [ExecutionNodeId::new(kernel_name)],
        kernel.clone(),
        kernel.provider.clone(),
    )
    .unwrap()
    .with_qualification_profile("standard")
    .with_artifact_digest(format!("sha256:{kernel_name}-{kernel_patch}"))
    .with_specialization("default")
    .with_prepared_kernel(
        allocator.allocate(),
        PreparedKernelGeneration::new(kernel_patch),
    )
}

fn test_graph(operator_version: u32) -> ExecutionGraph {
    let input = TensorEdgeId::new("hidden");
    let output = TensorEdgeId::new("projected");
    let node = ExecutionNode::new(
        ExecutionNodeId::new("matmul"),
        OperatorId::magnetar("matmul", operator_version, OperatorFamily::LinearAlgebra),
    )
    .with_input(input.clone())
    .with_output(output.clone())
    .with_attribute("transpose_b", OperatorAttributeValue::Boolean(false));

    ExecutionGraph::new(ExecutionGraphId::new("decode"), ExecutionGraphPhase::Decode)
        .with_edge(TensorEdge::new(
            input,
            TensorDescriptor::new(
                ShapeDescriptor::new([1, 128]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ))
        .with_edge(TensorEdge::new(
            output,
            TensorDescriptor::new(
                ShapeDescriptor::new([1, 128]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ))
        .with_node(node)
}

fn test_kernel(name: &str, patch: u64) -> KernelId {
    KernelId::new(
        ProviderBinding::new("reference-cpu"),
        name,
        CapabilityVersion::new(1, 0, patch),
        OperatorId::magnetar(name, 1, OperatorFamily::Attention),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::CpuScalar,
    )
}
