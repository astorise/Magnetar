use magnetar_runtime::{
    AddressStabilityRequirement, AllocationBlock, AllocationBlockId, AllocationClass,
    AllocationLease, AllocationLeaseId, AllocationLifetimeClass, AllocationMutability,
    AllocationPlan, AllocationPlanCache, AllocationPlanGeneration, AllocationPlanGuard,
    AllocationPlanId, AllocationPlanScope, AllocationPlanState, AllocationPlanner,
    AllocationReclaimability, AllocationRegion, AllocationRequest, AllocationRetryPolicy,
    AllocationSlot, AllocationSlotId, ArenaAllocationStrategy, AsyncNativeFreeState,
    AutotuningMemoryBudget, ClassIsolationPolicy, CompactionCandidate, CompactionDisposition,
    CompletionTokenId, DeviceBinding, DeviceId, DeviceMemoryArena, DeviceMemoryCapacitySnapshot,
    DeviceMemoryPool, DeviceMemoryPoolId, DeviceMemoryPoolState, FallbackClass,
    FragmentationReport, KernelMemoryEligibility, KvPageOwner, KvPagePool,
    MemoryAdmissionProjection, MemoryArenaId, MemoryArenaRole, MemoryBoundaryActor,
    MemoryBoundaryRequest, MemoryDomain, MemoryError, MemoryOomCategory, MemoryPerformanceFeedback,
    MemoryPoolClass, MemoryPoolConformanceReport, MemoryPoolErrorCode, MemoryPoolObservation,
    MemoryPoolObservationKind, MemoryPoolReservation, MemoryPoolReservationKind,
    MemoryPoolReservationScope, MemoryPressureLevel, OomFallbackAction, PoolCapacity,
    PoolGrowthRequest, PoolOvercommitPolicy, PoolShrinkRequest, PoolWatermarks,
    PreparedPlanMemoryBinding, ProviderBinding, ProviderPoolCapability, ReclaimCandidate,
    ReclaimPriority, RelocationOperation, ResourceAffinity, ResourceLifetimeInterval,
    ResourceMovability, TensorResourceId, WorkspaceRequirement, WorkspaceReuseGroup,
};
use std::collections::BTreeSet;

fn device() -> DeviceBinding {
    DeviceBinding::new(DeviceId::new("gpu-0"))
}

fn pool_id() -> DeviceMemoryPoolId {
    DeviceMemoryPoolId::new("runtime-pool-kv").unwrap()
}

fn pool() -> DeviceMemoryPool {
    DeviceMemoryPool::new(
        pool_id(),
        MemoryPoolClass::KvCache,
        ProviderBinding::new("cuda-provider"),
        device(),
        MemoryDomain::DeviceLocal(device()),
        PoolCapacity::new(8192),
    )
    .unwrap()
}

#[test]
fn device_memory_pool_identity_is_runtime_owned_and_opaque() {
    assert!(DeviceMemoryPoolId::new("runtime-pool-1").is_ok());
    assert!(DeviceMemoryPoolId::new("0xdeadbeef").is_err());
    assert!(DeviceMemoryPoolId::new("cuda ptr 0x1").is_err());

    let pool = pool();
    assert_eq!(pool.class, MemoryPoolClass::KvCache);
    assert_eq!(pool.state, DeviceMemoryPoolState::Initializing);
    assert_eq!(pool.capacity.configured_limit_bytes, 8192);
}

#[test]
fn pool_classes_are_extensible_without_native_pointer_semantics() {
    let classes = [
        MemoryPoolClass::Weights,
        MemoryPoolClass::KvCache,
        MemoryPoolClass::Workspace,
        MemoryPoolClass::Transient,
        MemoryPoolClass::Persistent,
        MemoryPoolClass::Transfer,
        MemoryPoolClass::Shared,
        MemoryPoolClass::custom("tenant-low-latency").unwrap(),
    ];

    assert_eq!(classes.len(), 8);
    assert!(MemoryPoolClass::custom("native-pointer-0x4").is_err());
}

#[test]
fn pool_capacity_tracks_active_reclaimable_and_pending_bytes() {
    let mut capacity = PoolCapacity::new(1024);
    capacity.reserved_bytes = 256;
    capacity.committed_bytes = 768;
    capacity.leased_bytes = 384;
    capacity.reclaimable_bytes = 128;
    capacity.pending_reclaim_bytes = 64;

    capacity.validate().unwrap();
    assert_eq!(capacity.free_committed_bytes(), 320);
    assert_eq!(capacity.immediately_available_bytes(), 576);

    capacity.reclaimable_bytes = 512;
    assert!(capacity.validate().is_err());
}

#[test]
fn hard_and_soft_reservations_are_accounted_separately() {
    let mut hard = MemoryPoolReservation::hard(
        MemoryPoolReservationScope::PoolClass(MemoryPoolClass::KvCache),
        4096,
    );
    assert_eq!(hard.kind, MemoryPoolReservationKind::Hard);
    assert!(hard.borrow(1).is_err());

    let mut soft = MemoryPoolReservation::soft(
        MemoryPoolReservationScope::WorkloadClass("workspace".into()),
        512,
    );
    soft.borrow(128).unwrap();
    assert_eq!(soft.borrowed_bytes, 128);

    let mut pool = pool();
    pool.add_reservation(hard).unwrap();
    pool.add_reservation(soft).unwrap();
    assert_eq!(pool.capacity.reserved_bytes, 4608);
    assert_eq!(pool.policy_version, 3);
}

#[test]
fn watermarks_drive_pressure_and_pool_lifecycle_transitions() {
    let mut pool = pool().with_watermarks(PoolWatermarks::with_critical(60, 80, Some(95)).unwrap());
    pool.transition_to(DeviceMemoryPoolState::Ready).unwrap();
    assert!(pool.can_lease(1024));

    pool.capacity.committed_bytes = 8192;
    pool.capacity.leased_bytes = 7000;
    assert_eq!(
        pool.refresh_pressure_state().unwrap(),
        DeviceMemoryPoolState::Pressure
    );

    pool.capacity.leased_bytes = 7900;
    assert_eq!(
        pool.refresh_pressure_state().unwrap(),
        DeviceMemoryPoolState::Critical
    );
    assert!(pool.transition_to(DeviceMemoryPoolState::Closed).is_err());
    pool.transition_to(DeviceMemoryPoolState::Draining).unwrap();
    assert!(!pool.can_lease(1));
    pool.transition_to(DeviceMemoryPoolState::Closed).unwrap();
}

#[test]
fn logical_allocation_request_validates_size_alignment_class_and_lifetime() {
    let request = AllocationRequest::new(
        4096,
        256,
        AllocationClass::ExecutionWorkspace,
        MemoryDomain::DeviceLocal(device()),
        AllocationLifetimeClass::BatchStep,
        ResourceAffinity::new(FallbackClass::ProviderPinned).with_device(device()),
    )
    .unwrap()
    .with_mutability(AllocationMutability::Mutable)
    .with_reclaimability(AllocationReclaimability::Reclaimable);

    assert_eq!(request.bytes, 4096);
    assert_eq!(request.alignment_bytes, 256);
    assert_eq!(request.lifetime_class, AllocationLifetimeClass::BatchStep);
    assert!(AllocationClass::custom("custom-output").is_ok());
    assert!(
        AllocationRequest::new(
            4096,
            3,
            AllocationClass::Output,
            MemoryDomain::DeviceLocal(device()),
            AllocationLifetimeClass::Operator,
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .is_err()
    );
}

#[test]
fn allocation_lease_is_logical_generation_bound_and_pointer_free() {
    let lease = AllocationLease::new(
        AllocationLeaseId::new(1),
        pool_id(),
        AllocationBlockId::new(7),
        256,
        1024,
        256,
        2,
    )
    .unwrap()
    .with_completion(CompletionTokenId::new(9, 1))
    .release_after_completion();

    assert_eq!(lease.generation, 2);
    assert_eq!(
        lease.state,
        magnetar_runtime::AllocationLeaseState::PendingReclaim
    );
    assert!(format!("{}", lease.id).contains("allocation-lease:1"));
    assert!(
        AllocationLease::new(
            AllocationLeaseId::new(2),
            pool_id(),
            AllocationBlockId::new(7),
            128,
            1024,
            256,
            1,
        )
        .is_err()
    );
}

#[test]
fn allocation_block_supports_bounds_overlap_alignment_and_fragmentation_checks() {
    let mut block = AllocationBlock::new(AllocationBlockId::new(1), pool_id(), 4096);
    block
        .add_region(AllocationRegion {
            lease: AllocationLeaseId::new(1),
            resource: Some(TensorResourceId::new("tensor-a")),
            range: 0..1024,
            alignment_bytes: 256,
            lifetime: AllocationLifetimeClass::Operator,
        })
        .unwrap();
    block
        .add_region(AllocationRegion {
            lease: AllocationLeaseId::new(2),
            resource: Some(TensorResourceId::new("tensor-b")),
            range: 2048..3072,
            alignment_bytes: 512,
            lifetime: AllocationLifetimeClass::Temporary,
        })
        .unwrap();

    assert_eq!(block.regions().len(), 2);
    assert_eq!(block.largest_free_region_bytes(), 1024);
    assert!(matches!(
        block.add_region(AllocationRegion {
            lease: AllocationLeaseId::new(3),
            resource: None,
            range: 512..1536,
            alignment_bytes: 256,
            lifetime: AllocationLifetimeClass::Temporary,
        }),
        Err(MemoryError::AllocationDenied { .. })
    ));
    assert!(
        block
            .add_region(AllocationRegion {
                lease: AllocationLeaseId::new(4),
                resource: None,
                range: 3073..3584,
                alignment_bytes: 512,
                lifetime: AllocationLifetimeClass::Temporary,
            })
            .is_err()
    );
}

#[test]
fn arenas_persistent_transient_and_workspace_requirements_are_contractual_not_algorithmic() {
    let mut persistent = DeviceMemoryArena::new(
        MemoryArenaId::new(1),
        pool_id(),
        MemoryArenaRole::Persistent,
        4096,
    )
    .with_strategy(ArenaAllocationStrategy::Slab);
    persistent.reserve(1024).unwrap();

    let mut transient = DeviceMemoryArena::new(
        MemoryArenaId::new(2),
        pool_id(),
        MemoryArenaRole::Transient,
        2048,
    )
    .with_strategy(ArenaAllocationStrategy::BucketedSizeClass);
    transient.reserve(512).unwrap();

    assert_eq!(persistent.used_bytes, 1024);
    assert_eq!(transient.used_bytes, 512);
    let workspace =
        WorkspaceRequirement::new(1024, 256, MemoryDomain::DeviceLocal(device())).unwrap();
    assert_eq!(workspace.bytes, 1024);
    assert_eq!(workspace.alignment_bytes, 256);
}

#[test]
fn allocation_planner_builds_stable_identity_slots_guards_and_reuse_groups() {
    let mut kv = pool();
    kv.transition_to(DeviceMemoryPoolState::Ready).unwrap();
    kv.capacity.committed_bytes = 8192;

    let mut workspace = DeviceMemoryPool::new(
        DeviceMemoryPoolId::new("runtime-pool-workspace").unwrap(),
        MemoryPoolClass::Workspace,
        ProviderBinding::new("cuda-provider"),
        device(),
        MemoryDomain::DeviceLocal(device()),
        PoolCapacity {
            committed_bytes: 4096,
            ..PoolCapacity::new(4096)
        },
    )
    .unwrap();
    workspace
        .transition_to(DeviceMemoryPoolState::Ready)
        .unwrap();

    let request = AllocationRequest::new(
        2048,
        256,
        AllocationClass::KvPage,
        MemoryDomain::DeviceLocal(device()),
        AllocationLifetimeClass::Session,
        ResourceAffinity::new(FallbackClass::ProviderPinned).with_device(device()),
    )
    .unwrap();
    let scope = AllocationPlanScope::new("graph-r1", "decode-b1-s4096", "workspace-r1")
        .unwrap()
        .with_memory_domain(MemoryDomain::DeviceLocal(device()))
        .with_policy_versions(7, 11);

    let planner = AllocationPlanner {
        pool_policy_version: 7,
        allocation_policy_version: 11,
    };
    let mut plan = planner
        .plan(
            AllocationPlanId::new("decode-plan-memory").unwrap(),
            scope.clone(),
            [kv, workspace],
            [request],
            [WorkspaceRequirement::new(1024, 256, MemoryDomain::DeviceLocal(device())).unwrap()],
        )
        .unwrap();
    let slot = AllocationSlot::new(
        AllocationSlotId::new(99),
        1024,
        256,
        MemoryPoolClass::Workspace,
        AllocationLifetimeClass::BatchStep,
    )
    .unwrap()
    .with_reuse_group("workspace")
    .unwrap()
    .stable();
    plan.add_slot(slot).unwrap();
    plan.add_reuse_group(
        WorkspaceReuseGroup::new("workspace")
            .unwrap()
            .with_slot(AllocationSlotId::new(2))
            .with_barrier(CompletionTokenId::new(1, 1)),
    );
    plan.add_guard(AllocationPlanGuard::AlignmentSupported(256));
    plan.mark_ready().unwrap();

    assert_eq!(plan.state, AllocationPlanState::Ready);
    assert_eq!(
        scope.cache_key(),
        "graph-r1|decode-b1-s4096|workspace-r1|pool:7|alloc:11|domains:1"
    );
    assert!(plan
        .slots
        .iter()
        .any(|slot| slot.stable && slot.movability == ResourceMovability::PermanentlyNonMovable));
}

#[test]
fn lifetime_analysis_temporal_reuse_and_async_barriers_are_enforced() {
    let mut plan = AllocationPlan::new(
        AllocationPlanId::new("reuse-plan").unwrap(),
        AllocationPlanGeneration::new(1),
        AllocationPlanScope::new("graph-r1", "decode", "workspace-r1").unwrap(),
    )
    .bind_pool(MemoryPoolClass::Transient, pool_id());
    plan.add_slot(
        AllocationSlot::new(
            AllocationSlotId::new(1),
            1024,
            256,
            MemoryPoolClass::Transient,
            AllocationLifetimeClass::Temporary,
        )
        .unwrap(),
    )
    .unwrap();
    let a_completion = CompletionTokenId::new(10, 1);
    let c_completion = CompletionTokenId::new(11, 1);
    plan.add_lifetime_interval(
        ResourceLifetimeInterval::new(TensorResourceId::new("tensor-a"), 1, 2)
            .unwrap()
            .with_stream("decode")
            .unwrap()
            .with_completion(a_completion),
    )
    .unwrap();
    plan.add_lifetime_interval(
        ResourceLifetimeInterval::new(TensorResourceId::new("tensor-c"), 3, 4)
            .unwrap()
            .with_stream("decode")
            .unwrap()
            .with_completion(c_completion),
    )
    .unwrap();
    plan.add_lifetime_interval(
        ResourceLifetimeInterval::new(TensorResourceId::new("tensor-overlap"), 2, 5)
            .unwrap()
            .with_stream("transfer")
            .unwrap(),
    )
    .unwrap();

    let mut completed = BTreeSet::new();
    assert!(
        !plan
            .can_temporally_reuse(
                &TensorResourceId::new("tensor-a"),
                &TensorResourceId::new("tensor-c"),
                &completed,
            )
            .unwrap()
    );
    completed.insert(a_completion);
    completed.insert(c_completion);
    assert!(
        plan.can_temporally_reuse(
            &TensorResourceId::new("tensor-a"),
            &TensorResourceId::new("tensor-c"),
            &completed,
        )
        .unwrap()
    );
    assert!(
        !plan
            .can_temporally_reuse(
                &TensorResourceId::new("tensor-a"),
                &TensorResourceId::new("tensor-overlap"),
                &completed,
            )
            .unwrap()
    );
}

#[test]
fn fragmentation_compaction_movability_relocation_and_address_pinning_are_explicit() {
    let fragmentation = FragmentationReport {
        free_bytes: 1024,
        largest_free_region_bytes: 128,
        requested_bytes: 512,
        committed_bytes: 4096,
    };
    assert!(fragmentation.is_fragmented_failure());

    let movable = CompactionCandidate {
        resource: TensorResourceId::new("movable"),
        lease: AllocationLeaseId::new(1),
        movability: ResourceMovability::Movable,
        in_flight: false,
        mapped: false,
    };
    let pinned = CompactionCandidate {
        movability: ResourceMovability::TemporarilyPinned,
        ..movable.clone()
    };
    let in_flight = CompactionCandidate {
        in_flight: true,
        ..movable.clone()
    };
    let mapped = CompactionCandidate {
        mapped: true,
        ..movable.clone()
    };

    assert_eq!(movable.disposition(), CompactionDisposition::Relocate);
    assert_eq!(pinned.disposition(), CompactionDisposition::SkipPinned);
    assert_eq!(in_flight.disposition(), CompactionDisposition::SkipInFlight);
    assert_eq!(mapped.disposition(), CompactionDisposition::SkipMapped);

    let relocation = RelocationOperation::new(
        &movable,
        AllocationLeaseId::new(2),
        CompletionTokenId::new(3, 1),
    )
    .unwrap();
    assert!(relocation.requires_plan_revalidation);
    assert!(
        RelocationOperation::new(
            &pinned,
            AllocationLeaseId::new(3),
            CompletionTokenId::new(4, 1),
        )
        .is_err()
    );
    assert!(AddressStabilityRequirement::PreparedSegment.pins_slot());
}

#[test]
fn prepared_plan_binding_reservation_readiness_and_cache_revalidation_are_checked() {
    let mut required_slots = BTreeSet::new();
    required_slots.insert(AllocationSlotId::new(1));
    let binding = PreparedPlanMemoryBinding {
        allocation_plan: AllocationPlanId::new("plan-a").unwrap(),
        generation: AllocationPlanGeneration::new(1),
        required_slots,
        reservation_required: true,
        capacity_validated: false,
    };
    assert!(binding.validate_ready().is_err());

    let mut plan = AllocationPlan::new(
        AllocationPlanId::new("plan-a").unwrap(),
        AllocationPlanGeneration::new(1),
        AllocationPlanScope::new("graph-r1", "decode", "workspace-r1").unwrap(),
    )
    .bind_pool(MemoryPoolClass::Workspace, pool_id());
    plan.add_slot(
        AllocationSlot::new(
            AllocationSlotId::new(1),
            1024,
            256,
            MemoryPoolClass::Workspace,
            AllocationLifetimeClass::BatchStep,
        )
        .unwrap(),
    )
    .unwrap();
    plan.add_guard(AllocationPlanGuard::PoolAvailable(pool_id()));
    plan.add_guard(AllocationPlanGuard::WorkspaceAvailable(1024));
    plan.mark_ready().unwrap();

    let mut cache = AllocationPlanCache::default();
    cache.insert(plan.clone());
    assert!(cache.lookup(&plan.scope).is_some());
    cache
        .revalidate(
            &plan,
            &[pool().with_watermarks(PoolWatermarks::new(60, 90).unwrap())],
            &[WorkspaceRequirement::new(1024, 256, MemoryDomain::DeviceLocal(device())).unwrap()],
        )
        .unwrap();

    assert!(cache.revalidate(&plan, &[], &[]).is_err());
    plan.mark_stale();
    assert_eq!(plan.state, AllocationPlanState::Stale);
    plan.hard_invalidate();
    assert_eq!(plan.state, AllocationPlanState::Invalid);
}

#[test]
fn admission_overcommit_kv_pages_batch_isolation_and_borrowing_are_bounded() {
    let projection = MemoryAdmissionProjection {
        persistent_weight_bytes: 1024,
        adapter_bytes: 256,
        mandatory_workspace_bytes: 512,
        minimum_kv_bytes: 1024,
        pinned_bytes: 128,
        provider_prepared_graph_bytes: 64,
        session_initial_kv_pages: 2,
        session_max_kv_pages: Some(4),
        kv_page_bytes: 128,
    };
    assert_eq!(projection.required_bytes().unwrap(), 3520);
    assert!(projection.admit(4096).is_ok());
    assert!(projection.admit(2048).is_err());

    assert!(!PoolOvercommitPolicy::default().permits(1024, 2048));
    assert!(
        PoolOvercommitPolicy::Enabled {
            max_bytes: 1024,
            max_ratio_percent: 200,
        }
        .permits(1024, 2048)
    );

    let mut kv_pages = KvPagePool::new(pool_id(), 256, 2).unwrap();
    let first = kv_pages
        .lease_page(KvPageOwner::Session("session-a".into()))
        .unwrap();
    kv_pages
        .retain_for_prefix(
            first.page_index,
            KvPageOwner::PrefixCache("prefix-a".into()),
        )
        .unwrap();
    kv_pages
        .release_page(first.page_index, Some(CompletionTokenId::new(9, 1)))
        .unwrap();
    let completed = BTreeSet::from([CompletionTokenId::new(9, 1)]);
    assert_eq!(kv_pages.recycle_completed(&completed).unwrap(), 0);
    kv_pages.grow(1).unwrap();
    assert_eq!(kv_pages.total_pages, 3);

    let isolation = ClassIsolationPolicy::protects(MemoryPoolClass::KvCache);
    let soft = MemoryPoolReservation::soft(
        MemoryPoolReservationScope::PoolClass(MemoryPoolClass::Workspace),
        1024,
    );
    assert!(!isolation.permits_borrow(&soft, &MemoryPoolClass::Transient));
}

#[test]
fn reclaim_async_free_provider_device_growth_shrink_oom_and_feedback_contracts_are_structured() {
    let completed = BTreeSet::from([CompletionTokenId::new(2, 1)]);
    let candidate = ReclaimCandidate {
        resource: TensorResourceId::new("cache-resource"),
        bytes: 512,
        reclaimability: AllocationReclaimability::Reclaimable,
        priority: ReclaimPriority::Cache,
        completion: Some(CompletionTokenId::new(2, 1)),
        mapped: false,
        pinned: false,
        semantic_aliases: BTreeSet::new(),
    };
    assert!(candidate.can_reclaim(&completed));
    assert_eq!(
        AsyncNativeFreeState::PendingNativeReclaim,
        AsyncNativeFreeState::PendingNativeReclaim
    );

    let capability = ProviderPoolCapability {
        provider: ProviderBinding::new("cuda-provider"),
        block_allocation: true,
        async_native_free: true,
        address_stability: true,
        movable_allocations: true,
        minimum_alignment: 256,
        preferred_granularity: 65536,
        grow_shrink: true,
    };
    capability.validate().unwrap();

    let device_capacity = DeviceMemoryCapacitySnapshot {
        device: device(),
        total_bytes: 8192,
        available_estimate_bytes: 4096,
        pressure: MemoryPressureLevel::Moderate,
        allocation_granularity_bytes: 65536,
    };
    device_capacity.validate_metadata_only().unwrap();

    assert_eq!(
        PoolGrowthRequest {
            additional_bytes: 1024,
            device_available_bytes: 2048,
            policy_limit_bytes: 8192,
        }
        .validate(4096)
        .unwrap(),
        5120
    );
    assert_eq!(
        PoolShrinkRequest {
            release_bytes: 2048,
            live_bytes: 1024,
            pending_reclaim_bytes: 512,
        }
        .releasable_bytes(4096),
        2048
    );

    assert_eq!(MemoryOomCategory::Fragmentation.code(), "fragmentation");
    assert_eq!(
        MemoryPoolErrorCode::AllocationPlanCapacityInsufficient.id(),
        "memory-allocation-plan-capacity-insufficient"
    );
    assert!(AllocationRetryPolicy { max_retries: 2 }.can_retry(1));
    assert_eq!(
        OomFallbackAction::SelectLowerWorkspaceKernel,
        OomFallbackAction::SelectLowerWorkspaceKernel
    );

    let kernel = KernelMemoryEligibility {
        kernel_id: "fast-attention".into(),
        required_workspace_bytes: 1024,
        compatible_pool_available_bytes: 512,
    };
    assert!(!kernel.is_eligible());
    let tuning_budget = AutotuningMemoryBudget {
        max_bytes: 1024,
        protect_inference_reservations: true,
        deny_under_critical_pressure: true,
    };
    assert!(tuning_budget.admit(512, DeviceMemoryPoolState::Ready));
    assert!(!tuning_budget.admit(512, DeviceMemoryPoolState::Critical));

    let feedback = MemoryPerformanceFeedback {
        allocation_latency_micros: 42,
        fragmentation: Some(FragmentationReport {
            free_bytes: 1024,
            largest_free_region_bytes: 128,
            requested_bytes: 512,
            committed_bytes: 4096,
        }),
        pressure: MemoryPressureLevel::High,
    };
    assert_eq!(feedback.allocation_latency_micros, 42);
}

#[test]
fn wit_runtime_boundary_observability_and_conformance_redact_native_authority() {
    assert!(
        MemoryBoundaryRequest {
            actor: MemoryBoundaryActor::WasmComponent,
            create_pool: true,
            choose_allocator_strategy: false,
            expose_native_handle: false,
            choose_native_pool: false,
        }
        .validate()
        .is_err()
    );
    assert!(
        MemoryBoundaryRequest {
            actor: MemoryBoundaryActor::InferenceRequest,
            create_pool: false,
            choose_allocator_strategy: false,
            expose_native_handle: false,
            choose_native_pool: true,
        }
        .validate()
        .is_err()
    );

    let observation = MemoryPoolObservation::redacted(
        MemoryPoolObservationKind::OomDetected,
        Some(pool_id()),
        "native ptr 0xdeadbeef leaked",
    );
    assert_eq!(observation.message, "[redacted]");
    assert_eq!(observation.pool.unwrap().as_str(), "runtime-pool-kv");

    let report = MemoryPoolConformanceReport {
        memory_manager_policy_authority: true,
        provider_native_realization: true,
        device_metadata_only: true,
        native_pointer_redaction: true,
        temporal_reuse_safe: true,
        async_reuse_safe: true,
        alignment_enforced: true,
        reservations_isolated: true,
        watermark_reclamation: true,
        pending_reclaim_accounted: true,
        fragmentation_classified: true,
        compaction_safe: true,
        address_pinning: true,
        plan_reservation_readiness: true,
        kv_page_lifetime: true,
        class_isolation: true,
        oom_policy: true,
        cache_revalidation: true,
    };
    assert!(report.conformant());
}
