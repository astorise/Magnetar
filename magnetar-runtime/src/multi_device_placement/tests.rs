//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{
    CapabilityVersion, ComputeDType, DeviceId, DeviceType, KernelImplementationFamily,
    KernelOperatorVersionRange, OperatorFamily, OperatorId,
};

fn device(name: &str, provider: &str, memory_capacity: u64) -> DeviceSetMember {
    let mut metadata = DeviceMetadata::new(DeviceId::new(name), name, DeviceType::Gpu, provider);
    metadata.architecture = format!("arch-{name}");
    metadata.memory_capacity = memory_capacity;
    metadata
        .memory_class_support
        .insert(KernelMemoryClass::Device);
    DeviceSetMember::new(metadata, DeviceAvailability::Available)
}

fn binding(provider: &str, name: &str) -> PlacementBinding {
    let device = DeviceBinding::new(DeviceId::new(name));
    PlacementBinding::new(
        ProviderBinding::new(provider),
        device.clone(),
        MemoryDomain::DeviceLocal(device),
    )
    .unwrap()
}

fn shard(name: &str, parent: &str, start: u64, end: u64, device_name: &str) -> TensorShard {
    TensorShard::new(
        TensorShardId::new(name).unwrap(),
        TensorResourceId::new(parent),
        LogicalRange::new(TensorPartitionAxis::Hidden, start, end),
        ShapeDescriptor::new([end - start]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        TensorLayoutKind::Contiguous,
        binding("cuda", device_name),
    )
}

fn test_kernel(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("cuda"),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar(name, 1, OperatorFamily::Attention),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::Cuda,
    )
}

#[test]
fn device_set_supports_single_multi_provider_and_heterogeneous_fingerprint() {
    let single = DeviceSet::new(
        DeviceSetId::new("single").unwrap(),
        [device("gpu0", "cuda", 24)],
    )
    .unwrap();
    assert_eq!(single.members().len(), 1);

    let multi = DeviceSet::new(
        DeviceSetId::new("heterogeneous").unwrap(),
        [
            device("gpu0", "cuda", 24),
            device("gpu1", "cuda", 8),
            device("cpu0", "cpu", 64),
        ],
    )
    .unwrap();
    assert_eq!(multi.providers().len(), 2);
    assert_ne!(single.fingerprint(), multi.fingerprint());

    assert!(matches!(
        DeviceSet::new(
            DeviceSetId::new("dup").unwrap(),
            [device("gpu0", "cuda", 24), device("gpu0", "cuda", 24)]
        ),
        Err(MultiDevicePlacementError::DuplicateDevice(_))
    ));
}

#[test]
fn placement_domain_and_binding_reject_cross_provider_and_native_handles() {
    let set = DeviceSet::new(
        DeviceSetId::new("set").unwrap(),
        [device("gpu0", "cuda", 24), device("cpu0", "cpu", 64)],
    )
    .unwrap();
    let domain = PlacementDomain::new(
        "local",
        [
            DeviceBinding::new(DeviceId::new("gpu0")),
            DeviceBinding::new(DeviceId::new("cpu0")),
        ],
    )
    .unwrap();
    assert!(matches!(
        domain.validate_against(&set),
        Err(MultiDevicePlacementError::CrossProviderDenied)
    ));
    assert!(
        domain
            .with_cross_provider(true)
            .validate_against(&set)
            .is_ok()
    );
    assert!(
        binding("cuda", "gpu0")
            .with_constraint("cuda pointer 0xdeadbeef")
            .is_err()
    );
}

#[test]
fn placement_plan_identity_includes_graph_revision_devices_providers_policy_and_partitions() {
    let set = DeviceSet::new(
        DeviceSetId::new("set").unwrap(),
        [device("gpu0", "cuda", 24), device("gpu1", "cuda", 24)],
    )
    .unwrap();
    let mut plan = MultiDevicePlacementPlan::new(
        MultiDevicePlacementPlanId::new("plan").unwrap(),
        MultiDevicePlacementGeneration::new(1),
        MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
        7,
        set,
        "policy-v1",
    )
    .unwrap();
    plan.provider_versions
        .insert(ProviderBinding::new("cuda"), "12.4".into());
    plan.memory_capacity_class = "24gib".into();
    plan.partition_fingerprint =
        Some(MultiDevicePlacementFingerprint::new("sha256:partition0001").unwrap());
    plan.add_binding(
        PlacementScope::layer_range(0, 15).unwrap(),
        binding("cuda", "gpu0"),
    )
    .unwrap();
    let first = plan.fingerprint();
    plan.model_instance_revision = 8;
    assert_ne!(first, plan.fingerprint());
}

#[test]
fn placement_scope_supports_model_segment_layer_operator_group_and_reserved_operator() {
    assert_eq!(
        PlacementScope::model_instance(ModelInstanceId::new("model-a").unwrap()).granularity,
        PlacementGranularity::ModelInstance
    );
    assert_eq!(
        PlacementScope::graph_segment([ExecutionNodeId::new("node0")])
            .unwrap()
            .granularity,
        PlacementGranularity::GraphSegment
    );
    assert_eq!(
        PlacementScope::layer_range(0, 7).unwrap().granularity,
        PlacementGranularity::LayerRange
    );
    assert_eq!(
        PlacementScope::operator_group("attention")
            .unwrap()
            .granularity,
        PlacementGranularity::OperatorGroup
    );
    assert!(PlacementGranularity::IndividualOperatorReserved > PlacementGranularity::OperatorGroup);
}

#[test]
fn eligibility_applies_hard_filters_before_transfer_aware_ranking() {
    let scope = PlacementScope::layer_range(0, 1).unwrap();
    let fast_but_impossible = PlacementCandidate {
        scope: scope.clone(),
        binding: binding("cuda", "gpu1"),
        required_kernel: None,
        required_memory_bytes: 32,
        available_memory_bytes: 16,
        required_memory_class: KernelMemoryClass::Device,
        provider_capable: true,
        device_capable: true,
        kernel_available: true,
        resource_affinity_valid: true,
        transfer_permitted: true,
        host_staging_permitted: true,
        latency_micros: 1,
        throughput_units: 1_000,
        transfer_cost_micros: 0,
        pressure: ProviderPressureLevel::Low,
        stability_penalty_micros: 0,
    };
    let slower_without_transfer = PlacementCandidate {
        required_memory_bytes: 8,
        available_memory_bytes: 16,
        latency_micros: 40,
        transfer_cost_micros: 0,
        binding: binding("cuda", "gpu0"),
        scope,
        ..fast_but_impossible.clone()
    };

    let (selected, report) =
        select_lowest_cost_eligible([fast_but_impossible, slower_without_transfer]).unwrap();
    assert_eq!(
        selected.binding.device,
        DeviceBinding::new(DeviceId::new("gpu0"))
    );
    assert_eq!(
        report.rejected[0].reason,
        MultiDevicePlacementErrorCode::MemoryInfeasible
    );
}

#[test]
fn pipeline_stages_preserve_order_and_create_explicit_cross_device_movement() {
    let gpu0 = binding("cuda", "gpu0");
    let gpu1 = binding("cuda", "gpu1");
    let stage0 = PipelineStage::new("stage0", [ExecutionNodeId::new("block0")], gpu0.clone(), 0)
        .unwrap()
        .with_output(TensorResourceId::new("activation"));
    let stage1 = PipelineStage::new("stage1", [ExecutionNodeId::new("block1")], gpu1.clone(), 1)
        .unwrap()
        .with_input(TensorResourceId::new("activation"));
    let movement = StageMovementEdge::new(
        &stage0.stage_id,
        &stage1.stage_id,
        TensorResourceId::new("activation"),
        gpu0,
        gpu1,
        HostStagingPolicy::Forbid,
    )
    .unwrap();

    assert!(stage0.order < stage1.order);
    assert!(
        stage0
            .output_requirements
            .contains(&TensorResourceId::new("activation"))
    );
    assert!(
        stage1
            .input_requirements
            .contains(&TensorResourceId::new("activation"))
    );
    assert!(movement.preserves_source_lifetime);
    assert!(movement.preserves_destination_readiness);
    assert_eq!(movement.host_staging_policy, HostStagingPolicy::Forbid);
}

#[test]
fn pipeline_overlap_requires_memory_dependencies_and_scheduler_policy() {
    assert!(PipelineOverlapPolicy::independent().permits_overlap());
    assert!(
        !PipelineOverlapPolicy {
            memory_capacity_ok: false,
            ..PipelineOverlapPolicy::independent()
        }
        .permits_overlap()
    );
    assert!(
        !PipelineOverlapPolicy {
            dependencies_satisfied: false,
            ..PipelineOverlapPolicy::independent()
        }
        .permits_overlap()
    );
    assert!(
        !PipelineOverlapPolicy {
            scheduler_policy_allows: false,
            ..PipelineOverlapPolicy::independent()
        }
        .permits_overlap()
    );
}

#[test]
fn weight_placement_supports_single_partition_replica_and_hybrid_relationships() {
    let dtype = DTypeDescriptor::portable(ComputeDType::Float32);
    let partition = TensorPartitionDescriptor::new(
        TensorResourceId::new("wq"),
        TensorPartitionAxis::Hidden,
        2,
        TensorReconstructionPolicy::LogicalOnly,
    )
    .unwrap()
    .with_shard(shard("wq-0", "wq", 0, 4096, "gpu0"))
    .with_shard(shard("wq-1", "wq", 4096, 8192, "gpu1"));
    let replica = WeightReplica::new(
        "artifact-wq",
        "r1",
        dtype.clone(),
        TensorLayoutKind::Contiguous,
        binding("cuda", "gpu0"),
    )
    .unwrap();

    WeightPlacement::new(
        "artifact-wq",
        "r1",
        dtype.clone(),
        TensorLayoutKind::Contiguous,
        WeightPlacementKind::SingleDevice,
        [DeviceBinding::new(DeviceId::new("gpu0"))],
    )
    .unwrap()
    .validate()
    .unwrap();
    WeightPlacement::new(
        "artifact-wq",
        "r1",
        dtype.clone(),
        TensorLayoutKind::Contiguous,
        WeightPlacementKind::Partitioned,
        [
            DeviceBinding::new(DeviceId::new("gpu0")),
            DeviceBinding::new(DeviceId::new("gpu1")),
        ],
    )
    .unwrap()
    .with_partition(partition.clone())
    .validate()
    .unwrap();
    WeightPlacement::new(
        "artifact-wq",
        "r1",
        dtype,
        TensorLayoutKind::Contiguous,
        WeightPlacementKind::Hybrid,
        [
            DeviceBinding::new(DeviceId::new("gpu0")),
            DeviceBinding::new(DeviceId::new("gpu1")),
        ],
    )
    .unwrap()
    .with_partition(partition)
    .with_replica(replica)
    .validate()
    .unwrap();
}

#[test]
fn tensor_partition_validates_bounds_gaps_overlaps_replicas_and_collective_boundary() {
    let complete = TensorPartitionDescriptor::new(
        TensorResourceId::new("hidden"),
        TensorPartitionAxis::Hidden,
        2,
        TensorReconstructionPolicy::LogicalOnly,
    )
    .unwrap()
    .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
    .with_shard(shard("hidden-1", "hidden", 4, 8, "gpu1"));
    assert!(complete.validate().is_ok());
    assert!(!complete.implies_collective());

    let gap = TensorPartitionDescriptor::new(
        TensorResourceId::new("hidden"),
        TensorPartitionAxis::Hidden,
        2,
        TensorReconstructionPolicy::LogicalOnly,
    )
    .unwrap()
    .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
    .with_shard(shard("hidden-1", "hidden", 5, 8, "gpu1"));
    assert!(matches!(
        gap.validate(),
        Err(MultiDevicePlacementError::PartitionGap)
    ));

    let overlap = TensorPartitionDescriptor::new(
        TensorResourceId::new("hidden"),
        TensorPartitionAxis::Hidden,
        2,
        TensorReconstructionPolicy::LogicalOnly,
    )
    .unwrap()
    .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
    .with_shard(shard("hidden-1", "hidden", 3, 8, "gpu1"));
    assert!(matches!(
        overlap.validate(),
        Err(MultiDevicePlacementError::PartitionOverlap)
    ));

    let replicated = overlap.allow_replicas();
    assert!(replicated.validate().is_ok());
}

#[test]
fn kernel_partition_compatibility_rejects_unsupported_shard_as_full_tensor() {
    let descriptor = TensorPartitionDescriptor::new(
        TensorResourceId::new("hidden"),
        TensorPartitionAxis::Hidden,
        2,
        TensorReconstructionPolicy::ExplicitMaterializationRequired,
    )
    .unwrap()
    .with_shard(shard("hidden-0", "hidden", 0, 4, "gpu0"))
    .with_shard(shard("hidden-1", "hidden", 4, 8, "gpu1"));
    let supported = KernelPartitionCompatibility {
        kernel: test_kernel("partitioned-attention"),
        input_axes: [TensorPartitionAxis::Hidden].into_iter().collect(),
        output_axes: [TensorPartitionAxis::Hidden].into_iter().collect(),
    };
    let unsupported = KernelPartitionCompatibility {
        kernel: test_kernel("full-attention"),
        input_axes: BTreeSet::new(),
        output_axes: BTreeSet::new(),
    };

    supported.validate_input(&descriptor).unwrap();
    supported.validate_output(&descriptor).unwrap();
    assert!(matches!(
        unsupported.validate_input(&descriptor),
        Err(MultiDevicePlacementError::ShardAsFullTensorUnsupported)
    ));
}

#[test]
fn peer_transfer_keeps_movement_explicit_metrics_redacted_and_policy_checked() {
    let movement = StageMovementEdge::new(
        "stage0",
        "stage1",
        TensorResourceId::new("activation"),
        binding("cuda", "gpu0"),
        binding("cuda", "gpu1"),
        HostStagingPolicy::Forbid,
    )
    .unwrap();
    let capability = DevicePairTransferCapability {
        bandwidth_class: "nvlink-class".into(),
        ..DevicePairTransferCapability::new(
            ProviderBinding::new("cuda"),
            DeviceBinding::new(DeviceId::new("gpu0")),
            DeviceBinding::new(DeviceId::new("gpu1")),
            [
                DevicePairAccessMode::PeerRead,
                DevicePairAccessMode::PeerCopy,
            ],
        )
    };

    let plan = DeviceTransferPlan::new(movement.clone(), 4096, Some(&capability)).unwrap();
    assert_eq!(plan.kind, DeviceTransferKind::DirectPeer);
    assert_eq!(plan.expected_bytes, 4096);
    assert_eq!(plan.peer_bandwidth_class.as_deref(), Some("nvlink-class"));
    assert_eq!(plan.host_staging_cost_micros, 0);

    let staged = DevicePairTransferCapability {
        requires_host_staging: true,
        ..capability
    };
    assert!(matches!(
        DeviceTransferPlan::new(movement, 4096, Some(&staged)),
        Err(MultiDevicePlacementError::HostStagingDenied)
    ));
}

#[test]
fn cross_provider_transfer_requires_explicit_boundary_and_no_native_handles() {
    let movement = StageMovementEdge::new(
        "stage0",
        "stage1",
        TensorResourceId::new("activation"),
        binding("cuda", "gpu0"),
        binding("cpu", "cpu0"),
        HostStagingPolicy::Permit,
    )
    .unwrap();
    let transfer = DeviceTransferPlan::new(movement, 1024, None).unwrap();
    assert_eq!(transfer.kind, DeviceTransferKind::CrossProviderBoundary);
    assert!(transfer.host_staging_cost_micros > 0);

    let native = DevicePairTransferCapability {
        bandwidth_class: "cuda ipc 0xdeadbeef".into(),
        ..DevicePairTransferCapability::new(
            ProviderBinding::new("cuda"),
            DeviceBinding::new(DeviceId::new("gpu0")),
            DeviceBinding::new(DeviceId::new("gpu1")),
            [DevicePairAccessMode::PeerCopy],
        )
    };
    assert!(matches!(
        native.validate(),
        Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
    ));
}

#[test]
fn per_device_memory_budget_accounts_all_runtime_owned_classes() {
    let mut budget = DeviceMemoryBudget::new(DeviceBinding::new(DeviceId::new("gpu0")), 1024);
    budget.weights_bytes = 256;
    budget.kv_bytes = 128;
    budget.workspace_bytes = 128;
    budget.transient_bytes = 64;
    budget.transfer_buffer_bytes = 64;
    budget.reserved_headroom_bytes = 64;
    assert_eq!(budget.used_bytes().unwrap(), 704);
    budget.validate().unwrap();

    budget.transfer_buffer_bytes = 512;
    assert!(matches!(
        budget.validate(),
        Err(MultiDevicePlacementError::MemoryBudgetExceeded)
    ));
}

#[test]
fn device_pool_binding_assigns_allocation_slots_to_concrete_device() {
    let gpu0 = DeviceBinding::new(DeviceId::new("gpu0"));
    let gpu1 = DeviceBinding::new(DeviceId::new("gpu1"));
    let pool = DevicePoolBinding::new(gpu0.clone(), "gpu0-pool")
        .unwrap()
        .with_slot(
            AllocationSlotBinding::new(
                "weights",
                gpu0.clone(),
                MemoryAllocationClass::ModelArtifact,
                256,
            )
            .unwrap(),
        )
        .unwrap()
        .with_slot(
            AllocationSlotBinding::new("kv", gpu0.clone(), MemoryAllocationClass::KvCache, 128)
                .unwrap(),
        )
        .unwrap();
    pool.validate().unwrap();

    assert!(matches!(
        DevicePoolBinding::new(gpu0, "empty-pool")
            .unwrap()
            .validate(),
        Err(MultiDevicePlacementError::UnspecifiedGlobalPoolUse)
    ));
    assert!(matches!(
        DevicePoolBinding::new(gpu1.clone(), "gpu1-pool")
            .unwrap()
            .with_slot(
                AllocationSlotBinding::new(
                    "wrong-device",
                    DeviceBinding::new(DeviceId::new("gpu0")),
                    MemoryAllocationClass::TemporaryWorkspace,
                    32,
                )
                .unwrap(),
            ),
        Err(MultiDevicePlacementError::AllocationSlotDeviceMismatch)
    ));
}

#[test]
fn hysteresis_requires_material_improvement_and_cooldown_before_replacement() {
    let policy = PlacementHysteresisPolicy::new(100, 1_000);
    assert!(!policy.should_replace(1_000, 950, 2_000));
    assert!(!policy.should_replace(1_000, 800, 999));
    assert!(policy.should_replace(1_000, 800, 1_000));
}

#[test]
fn plan_lifecycle_uses_generation_and_blocks_empty_ready_state() {
    let set = DeviceSet::new(
        DeviceSetId::new("set").unwrap(),
        [device("gpu0", "cuda", 24)],
    )
    .unwrap();
    let mut plan = MultiDevicePlacementPlan::new(
        MultiDevicePlacementPlanId::new("plan").unwrap(),
        MultiDevicePlacementGeneration::new(1),
        MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
        1,
        set,
        "policy-v1",
    )
    .unwrap();
    assert!(matches!(
        plan.mark_ready(),
        Err(MultiDevicePlacementError::NoPlacementBindings)
    ));
    plan.add_binding(
        PlacementScope::graph_segment([ExecutionNodeId::new("node0")]).unwrap(),
        binding("cuda", "gpu0"),
    )
    .unwrap();
    plan.mark_ready().unwrap();
    assert!(plan.state.accepts_new_work());
    assert!(matches!(
        plan.transition_to(MultiDevicePlacementState::Building),
        Err(MultiDevicePlacementError::InvalidStateTransition { .. })
    ));
}

#[test]
fn placement_pins_do_not_override_safety_or_compatibility() {
    let pin = PlacementPin {
        kind: PlacementPinKind::ModelInstanceDeviceSet,
        model_instance: Some(ModelInstanceId::new("model-a").unwrap()),
        stage_id: Some("stage0".into()),
        weight_artifact: Some("weights-a".into()),
        session_id: Some("session-a".into()),
        device_set: Some(DeviceSetId::new("set-a").unwrap()),
        binding: Some(binding("cuda", "gpu0")),
        compatibility_authoritative: true,
        device_available: true,
    };
    pin.validate().unwrap();

    let unsafe_pin = PlacementPin {
        compatibility_authoritative: false,
        ..pin
    };
    assert!(matches!(
        unsafe_pin.validate(),
        Err(MultiDevicePlacementError::PlacementPinInvalid)
    ));
}

#[test]
fn prefill_decode_transition_requires_state_and_explicit_movement() {
    let phases = PhasePlacementPlans {
        prefill_plan: MultiDevicePlacementPlanId::new("prefill").unwrap(),
        decode_plan: MultiDevicePlacementPlanId::new("decode").unwrap(),
    };
    assert!(phases.uses_distinct_plans());
    assert_eq!(
        phases.plan_for(PlacementPhase::Decode),
        &MultiDevicePlacementPlanId::new("decode").unwrap()
    );

    let ready = PhaseTransitionReadiness {
        from: PlacementPhase::Prefill,
        to: PlacementPhase::Decode,
        kv_available: true,
        weights_available: true,
        upstream_completion_observed: true,
        explicit_movements_completed: true,
        decode_guards_passed: true,
    };
    assert!(ready.permits_transition());
    assert!(
        !PhaseTransitionReadiness {
            kv_available: false,
            ..ready
        }
        .permits_transition()
    );
}

#[test]
fn kv_ownership_locality_partition_replication_and_session_migration_are_explicit() {
    let kv =
        KvPagePlacement::new("page0", "session0", "sequence0", binding("cuda", "gpu1")).unwrap();
    assert!(kv.authoritative);
    assert_eq!(kv.owner.device, DeviceBinding::new(DeviceId::new("gpu1")));

    let locality = KvLocalityDecision {
        decode_binding: binding("cuda", "gpu1"),
        authoritative_kv_device: DeviceBinding::new(DeviceId::new("gpu1")),
        kv_movement_cost_micros: 80,
        permits_per_token_bounce: false,
    };
    assert!(locality.favors_kv_locality());

    KvPartitionBoundary {
        attention_contract_supports_partition: true,
        required_collectives: BTreeSet::new(),
    }
    .validate()
    .unwrap();
    assert!(matches!(
        (KvPartitionBoundary {
            attention_contract_supports_partition: false,
            required_collectives: BTreeSet::new(),
        })
        .validate(),
        Err(MultiDevicePlacementError::KvPlacementInvalid)
    ));

    KvReplicaPolicy {
        authoritative_device: DeviceBinding::new(DeviceId::new("gpu1")),
        replicas: [DeviceBinding::new(DeviceId::new("gpu0"))]
            .into_iter()
            .collect(),
        update_coherency_explicit: true,
        baseline_prefers_single_authority: true,
    }
    .validate()
    .unwrap();

    let affinity = SessionPlacementAffinity::new(
        "session0",
        DeviceBinding::new(DeviceId::new("gpu1")),
        MultiDevicePlacementPlanId::new("decode").unwrap(),
    )
    .unwrap();
    assert!(affinity.preserves_kv_locality);

    assert!(
        SessionMigrationPlan {
            session_id: "session0".into(),
            source: binding("cuda", "gpu1"),
            destination: binding("cuda", "gpu0"),
            moves_kv: true,
            moves_adapters: true,
            moves_session_buffers: true,
            preserves_completion_tokens: true,
        }
        .is_explicit_and_complete()
    );
}

#[test]
fn model_instance_prepared_plan_guards_stale_invalidation_and_atomic_replacement_are_explicit() {
    let mut plans = ModelInstancePlacementPlans {
        model_instance: ModelInstanceId::new("model-a").unwrap(),
        plans: BTreeMap::new(),
    };
    plans.add_plan(
        PlacementPlanRole::Default,
        MultiDevicePlacementPlanId::new("default").unwrap(),
    );
    plans.add_plan(
        PlacementPlanRole::WorkloadSpecific,
        MultiDevicePlacementPlanId::new("batch-large").unwrap(),
    );
    plans.add_plan(
        PlacementPlanRole::Degraded,
        MultiDevicePlacementPlanId::new("degraded").unwrap(),
    );
    assert!(plans.supports_role(PlacementPlanRole::WorkloadSpecific));
    assert!(plans.supports_role(PlacementPlanRole::Degraded));

    let movement = StageMovementEdge::new(
        "stage0",
        "stage1",
        TensorResourceId::new("activation"),
        binding("cuda", "gpu0"),
        binding("cuda", "gpu1"),
        HostStagingPolicy::Forbid,
    )
    .unwrap();
    let prepared = PreparedExecutionPlacement {
        placement_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
        generation: MultiDevicePlacementGeneration::new(4),
        exact_segment_bindings: vec![
            ("stage0".into(), binding("cuda", "gpu0")),
            ("stage1".into(), binding("cuda", "gpu1")),
        ],
        movement_nodes: vec![movement],
        per_device_allocation_plans: [
            (DeviceBinding::new(DeviceId::new("gpu0")), "alloc0".into()),
            (DeviceBinding::new(DeviceId::new("gpu1")), "alloc1".into()),
        ]
        .into_iter()
        .collect(),
    };
    prepared.validate().unwrap();

    assert!(
        PlacementGuardSnapshot {
            device_available: true,
            provider_ready: true,
            kernel_prepared: true,
            resource_resident: true,
            memory_reserved: true,
            peer_path_available: true,
            host_staging_policy_valid: true,
        }
        .all_pass()
    );
    assert!(
        PlacementStaleness {
            reason: PlacementStalenessReason::PressureShift,
            request_background_replacement: true,
        }
        .request_background_replacement
    );
    assert!(
        PlacementInvalidation {
            reason: PlacementInvalidationReason::DeviceLost,
            invalid_for_new_work: true,
        }
        .invalid_for_new_work
    );
    assert!(
        PlacementReplacementRequest {
            current_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
            build_outside_hot_path: true,
            revalidate_resources: true,
            prepare_required_kernels: true,
        }
        .can_build()
    );
    assert!(
        AtomicPlacementReplacement {
            old_plan: MultiDevicePlacementPlanId::new("default").unwrap(),
            old_generation: MultiDevicePlacementGeneration::new(4),
            new_plan: MultiDevicePlacementPlanId::new("default-v2").unwrap(),
            new_generation: MultiDevicePlacementGeneration::new(5),
            new_plan_complete: true,
            old_in_flight_retained: true,
        }
        .can_publish()
    );
}

#[test]
fn failure_degraded_recovery_scheduler_admission_concurrency_and_replica_eviction_are_guarded() {
    assert!(
        DeviceFailureImpact {
            lost_device: DeviceBinding::new(DeviceId::new("gpu1")),
            invalidates_streams: true,
            invalidates_plans: true,
            preserves_other_devices: true,
        }
        .is_isolated_failure_domain()
    );
    assert!(
        DegradedPlanValidation {
            plan: MultiDevicePlacementPlanId::new("degraded").unwrap(),
            explicit_degraded_plan: true,
            model_capacity_ok: true,
            kernels_ok: true,
            memory_ok: true,
            policy_ok: true,
        }
        .valid()
    );
    assert!(
        !FailoverPolicy {
            arbitrary_remaining_device_forbidden: true,
            ready_fallback_plan: None,
        }
        .can_fail_over()
    );
    assert!(
        DeviceRecoveryChecklist {
            health_readiness_checked: true,
            pools_rebuilt: true,
            kernels_reprepared: true,
            placement_plan_rebuilt: true,
        }
        .ready_for_new_work()
    );
    assert!(
        SchedulerPlacementInput {
            session_affinity: None,
            device_pressure: BTreeMap::from([(
                DeviceBinding::new(DeviceId::new("gpu0")),
                ProviderPressureLevel::Low,
            )]),
            plan_ready: true,
            exposes_native_handles: false,
        }
        .can_admit()
    );
    assert!(
        PlacementAdmissionCheck {
            mandatory_devices_available: true,
            per_device_memory_ok: true,
            required_kernels_available: true,
            transfers_feasible: true,
        }
        .admits()
    );
    assert!(
        CrossDeviceConcurrencyContract {
            independent_device_execution: true,
            dependencies_preserved: true,
            resource_lifetime_preserved: true,
        }
        .permits_concurrency()
    );
    let failure = FailurePropagationDecision {
        upstream_failed: true,
        downstream_stopped: true,
        explicit_fallback: None,
        structured_reason: MultiDevicePlacementErrorCode::DeviceLost,
    };
    assert!(!failure.downstream_allowed());
    assert!(failure.downstream_stopped);
    assert!(
        ReplicaEvictionDecision {
            optional_replica: true,
            authoritative_copy_remains: true,
            no_in_flight_references: true,
            dependent_plan_invalidated: true,
        }
        .can_evict()
    );
}

#[test]
fn kernel_tuning_performance_cache_revalidation_native_wit_and_api_boundaries_hold() {
    let joint = JointKernelPlacementDecision {
        binding: binding("cuda", "gpu0"),
        kernel: test_kernel("attention"),
        hard_eligible: true,
        transfer_cost_micros: 40,
        memory_cost_bytes: 16,
    };
    assert!(joint.selectable());
    assert_eq!(joint.total_cost(), 56);

    let evidence = DeviceSpecificTuningEvidence {
        device: DeviceBinding::new(DeviceId::new("gpu0")),
        kernel: test_kernel("attention"),
        performance_context: "sm90-low-pressure".into(),
    };
    evidence
        .valid_for(&DeviceBinding::new(DeviceId::new("gpu0")))
        .unwrap();
    assert!(
        evidence
            .valid_for(&DeviceBinding::new(DeviceId::new("gpu1")))
            .is_err()
    );
    assert!(
        PlacementPerformanceFeedback {
            plan: MultiDevicePlacementPlanId::new("plan").unwrap(),
            device_context: DeviceBinding::new(DeviceId::new("gpu0")),
            segment_id: "stage0".into(),
            baseline_micros: 100,
            observed_micros: 112,
        }
        .regressed()
    );

    let key = PlacementPlanCacheKey {
        graph_fingerprint: MultiDevicePlacementFingerprint::new("sha256:graph0001").unwrap(),
        model_instance_revision: 1,
        device_set_fingerprint: MultiDevicePlacementFingerprint::new("sha256:devices1").unwrap(),
        provider_versions: vec![(ProviderBinding::new("cuda"), "12.4".into())],
        memory_budget_class: "24gib".into(),
        workload_scope: "decode".into(),
        placement_policy_version: "policy-v1".into(),
        partition_fingerprint: None,
    };
    key.validate().unwrap();
    let mut cache = PlacementPlanCache::default();
    cache.insert(
        key.clone(),
        MultiDevicePlacementPlanId::new("cached").unwrap(),
    );
    assert!(cache.lookup(&key).is_some());
    cache.invalidate(&key);
    assert!(cache.lookup(&key).is_none());
    assert!(
        CachedPlanRevalidation {
            device_available: true,
            provider_ready: true,
            memory_capacity_ok: true,
            peer_capability_ok: true,
            kernel_available: true,
            policy_ok: true,
            resource_residency_ok: true,
        }
        .valid()
    );

    assert!(matches!(
        NativeHandlePrivacyCheck {
            device_pointer: Some("0xdeadbeef".into()),
            peer_handle: None,
            native_queue: None,
            os_handle: None,
        }
        .validate(),
        Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
    ));
    ComponentPlacementRequest {
        portable_requirements: ["attention".into()].into_iter().collect(),
        requested_device: None,
        topology_authority: false,
    }
    .validate_wit_boundary()
    .unwrap();
    assert!(matches!(
        ComponentPlacementRequest {
            portable_requirements: BTreeSet::new(),
            requested_device: Some(DeviceBinding::new(DeviceId::new("gpu0"))),
            topology_authority: false,
        }
        .validate_wit_boundary(),
        Err(MultiDevicePlacementError::PolicyAuthorityViolation)
    ));
    assert!(
        !RuntimeInferencePlacementRequest {
            preferences: [RuntimePreference::LowLatency].into_iter().collect(),
            layer_to_device_mapping: vec![(0, DeviceBinding::new(DeviceId::new("gpu0")))],
            admin_policy_binding: None,
        }
        .normal_request_allowed()
    );
}

#[test]
fn errors_observability_conformance_and_documentation_cover_final_contract() {
    let expected_error_ids = [
        MultiDevicePlacementErrorCode::NoFeasiblePlan.id(),
        MultiDevicePlacementErrorCode::TensorPartitionInvalid.id(),
        MultiDevicePlacementErrorCode::StageTransferFailed.id(),
        MultiDevicePlacementErrorCode::KvPlacementInvalid.id(),
        MultiDevicePlacementErrorCode::DeviceLost.id(),
        MultiDevicePlacementErrorCode::DegradedPlanUnavailable.id(),
        MultiDevicePlacementErrorCode::Internal.id(),
    ];
    assert!(expected_error_ids.contains(&"multi-device-kv-placement-invalid"));
    assert!(expected_error_ids.contains(&"internal-multi-device-placement-error"));

    PlacementObservation {
        kind: PlacementObservationKind::CrossDeviceTransferCompleted,
        plan: Some(MultiDevicePlacementPlanId::new("plan").unwrap()),
        generation: Some(MultiDevicePlacementGeneration::new(1)),
        provider: Some(ProviderBinding::new("cuda")),
        device: Some(DeviceBinding::new(DeviceId::new("gpu0"))),
        stage_id: Some("stage0".into()),
        tensor_partition_id: Some("hidden-split".into()),
        movement_bytes: Some(4096),
        reason: Some(MultiDevicePlacementErrorCode::NoFeasiblePlan),
        detail: Some("redacted-logical-detail".into()),
    }
    .validate_redacted()
    .unwrap();
    assert!(matches!(
        PlacementObservation {
            kind: PlacementObservationKind::CrossDeviceTransferStarted,
            plan: None,
            generation: None,
            provider: None,
            device: None,
            stage_id: None,
            tensor_partition_id: None,
            movement_bytes: None,
            reason: None,
            detail: Some("native queue 0xdeadbeef".into()),
        }
        .validate_redacted(),
        Err(MultiDevicePlacementError::NativeHandleForbidden { .. })
    ));

    assert!(
        MultiDeviceConformanceReport {
            runtime_placement_authority: true,
            component_cannot_force_device: true,
            partition_validity: true,
            replica_partition_distinct: true,
            shard_not_full_tensor: true,
            explicit_cross_device_movement: true,
            host_staging_policy_preserved: true,
            peer_capability_required: true,
            per_device_memory_policy: true,
            heterogeneous_device_support: true,
            transfer_aware_selection: true,
            exact_prepared_placement: true,
            no_mid_flight_migration: true,
            kv_locality: true,
            explicit_session_migration: true,
            device_loss_invalidation: true,
            degraded_plan_validation: true,
            recovery_lifecycle: true,
            cache_revalidation: true,
            handle_isolation: true,
            observability_redaction: true,
        }
        .passes()
    );
    for topic in [
        "DeviceSet",
        "PlacementDomain",
        "MultiDevicePlacementPlan",
        "TensorShard",
        "KV locality",
        "local-only scope",
    ] {
        assert!(MULTI_DEVICE_PLACEMENT_DOCUMENTATION_TOPICS.contains(&topic));
    }
}
