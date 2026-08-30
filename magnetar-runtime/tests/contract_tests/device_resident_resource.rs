use magnetar_runtime::{
    CompletionTokenId, ComputeDType, DTypeDescriptor, DeviceBinding, DeviceId, FallbackClass,
    HostStagingPolicy, MappingCoherency, MemoryAllocationClass, MemoryAllocationOwner,
    MemoryAllocationRequest, MemoryCapabilityDescriptor, MemoryDomain, MemoryError, MemoryManager,
    MemoryObservationKind, MemoryPlacement, MemoryPressureAction, PeerAccessCapability,
    PeerAccessMode, ProviderBinding, ResidencyPin, ResidencyRequirement, ResidencySet,
    ResourceAffinity, ResourceExportPolicy, ResourceImportDescriptor, ResourceMappingAccess,
    ResourceMappingId, ResourceMappingState, ResourceMovement, ResourceMovementKind,
    ResourceResidency, ShapeDescriptor, TensorDescriptor, TensorResidency, TensorResourceId,
    ZeroCopyEligibility,
};
use std::collections::BTreeSet;

fn tensor_id(name: &str) -> TensorResourceId {
    TensorResourceId::new(name)
}

fn gpu(index: u64) -> DeviceBinding {
    DeviceBinding::new(DeviceId::new(format!("gpu-{index}")))
}

#[test]
fn memory_domain_covers_baseline_residency_classes_without_native_handles() {
    let device = gpu(0);

    assert_eq!(
        MemoryDomain::from_placement(&MemoryPlacement::HostOrdinary),
        MemoryDomain::Host
    );
    assert_eq!(
        MemoryDomain::from_placement(&MemoryPlacement::Device(device.clone())),
        MemoryDomain::DeviceLocal(device.clone())
    );
    assert!(MemoryDomain::HostVisibleDevice(device.clone()).is_host_visible());
    assert!(MemoryDomain::HostVisibleDevice(device).is_device_resident());
    assert!(MemoryDomain::External("future-interop".into()) >= MemoryDomain::External("".into()));
}

#[test]
fn residency_set_tracks_authoritative_current_and_stale_replicas() {
    let resource = tensor_id("weights");
    let gpu0 = MemoryDomain::DeviceLocal(gpu(0));
    let gpu1 = MemoryDomain::DeviceLocal(gpu(1));
    let stale_host = ResourceResidency::new(resource.clone(), MemoryDomain::Host).stale();

    let mut set = ResidencySet::new(resource.clone());
    set.add(ResourceResidency::new(resource.clone(), gpu0.clone()))
        .unwrap();
    set.add(ResourceResidency::new(resource.clone(), gpu1.clone()).replicated())
        .unwrap();
    set.add(stale_host).unwrap();

    assert_eq!(set.current_replicas().count(), 2);
    assert_eq!(set.stale_replicas().count(), 1);
    assert!(set.authoritative().unwrap().is_device_resident());
    assert!(set.readable_for(&gpu0).is_ok());
    assert!(set.readable_for(&MemoryDomain::Host).is_err());
}

#[test]
fn memory_manager_prevents_stale_replica_reads_and_maps_only_ready_resources() {
    let resource = tensor_id("activation");
    let mut memory = MemoryManager::default();
    let allocation = memory
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            1024,
            MemoryPlacement::Device(gpu(0)),
            MemoryAllocationOwner::Runtime,
        ))
        .unwrap();

    memory
        .record_resource_residency(
            ResourceResidency::new(resource.clone(), MemoryDomain::DeviceLocal(gpu(0)))
                .with_allocation(allocation.id),
        )
        .unwrap();
    memory
        .record_resource_residency(
            ResourceResidency::new(resource.clone(), MemoryDomain::Host).stale(),
        )
        .unwrap();

    assert!(
        memory
            .readable_residency(&resource, &MemoryDomain::DeviceLocal(gpu(0)))
            .is_ok()
    );
    assert!(
        memory
            .readable_residency(&resource, &MemoryDomain::Host)
            .is_err()
    );

    let mapping = memory
        .map_resource(
            resource,
            ResourceMappingAccess::Read,
            0..128,
            MemoryDomain::Host,
        )
        .unwrap();
    assert_eq!(mapping.state, ResourceMappingState::Active);
    memory.release_mapping(mapping.id).unwrap();
    assert_eq!(
        memory.resource_mappings().next().unwrap().state,
        ResourceMappingState::Released
    );
}

#[test]
fn zero_copy_requires_current_matching_residency_or_existing_host_visibility() {
    let resource = tensor_id("logits");
    let mut memory = MemoryManager::default();
    memory
        .record_resource_residency(ResourceResidency::new(
            resource.clone(),
            MemoryDomain::HostVisibleDevice(gpu(0)),
        ))
        .unwrap();

    assert!(
        memory
            .zero_copy_for_residency(&resource, &MemoryDomain::HostVisibleDevice(gpu(0)))
            .feasible
    );
    assert!(
        !memory
            .zero_copy_for_residency(&resource, &MemoryDomain::DeviceLocal(gpu(1)))
            .feasible
    );

    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let residency = TensorResidency::new(
        tensor_id("host-visible"),
        MemoryPlacement::UnifiedShared,
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_size_estimate(descriptor.byte_size().unwrap());

    assert!(
        memory
            .zero_copy_feasibility(&residency, &MemoryPlacement::HostOrdinary, None)
            .feasible
    );
}

#[test]
fn resource_views_validate_bounds_overlap_and_explicit_materialization() {
    let mut memory = MemoryManager::default();
    let view = memory
        .create_resource_view(
            tensor_id("parent"),
            2,
            ShapeDescriptor::new([2, 2]),
            [4, 1],
            "strided",
            16,
        )
        .unwrap();
    let overlapping = memory
        .create_resource_view(
            tensor_id("parent"),
            4,
            ShapeDescriptor::new([2, 2]),
            [4, 1],
            "strided",
            16,
        )
        .unwrap();

    assert!(view.overlaps(&overlapping).unwrap());
    assert!(
        memory
            .plan_materialization(&view, tensor_id("materialized"), false)
            .is_some()
    );
    assert!(
        memory
            .create_resource_view(
                tensor_id("parent"),
                u64::MAX,
                ShapeDescriptor::new([2]),
                [1],
                "bad",
                16,
            )
            .is_err()
    );
}

#[test]
fn mapping_hazards_lifetime_and_non_coherent_release_are_explicit() {
    let resource = tensor_id("mapped-resource");
    let mut memory = MemoryManager::default();
    memory
        .record_resource_residency(ResourceResidency::new(resource.clone(), MemoryDomain::Host))
        .unwrap();

    let first = memory
        .map_resource(
            resource.clone(),
            ResourceMappingAccess::Write,
            0..64,
            MemoryDomain::Host,
        )
        .unwrap();
    assert!(matches!(
        memory.map_resource(
            resource.clone(),
            ResourceMappingAccess::Read,
            32..96,
            MemoryDomain::Host
        ),
        Err(MemoryError::MappingConflict { .. })
    ));
    memory.release_mapping(first.id).unwrap();

    let write_mapping = memory
        .map_resource(
            resource,
            ResourceMappingAccess::Write,
            0..64,
            MemoryDomain::Host,
        )
        .unwrap()
        .with_coherency(MappingCoherency::NonCoherent);
    memory.release_mapping(write_mapping.id).unwrap();
    assert_eq!(
        memory.resource_mappings().last().unwrap().state,
        ResourceMappingState::Released
    );
    assert!(matches!(
        memory.release_mapping(ResourceMappingId::new(999)),
        Err(MemoryError::InvalidMappingHandle(_))
    ));
}

#[test]
fn zero_copy_eligibility_reports_specific_failed_gate() {
    let mut memory = MemoryManager::default();
    memory
        .validate_zero_copy_eligibility(ZeroCopyEligibility::compatible())
        .unwrap();

    let mut bad_layout = ZeroCopyEligibility::compatible();
    bad_layout.layout_compatible = false;
    assert!(matches!(
        memory.validate_zero_copy_eligibility(bad_layout),
        Err(MemoryError::ZeroCopyLayoutIncompatible { .. })
    ));

    let mut bad_coherency = ZeroCopyEligibility::compatible();
    bad_coherency.coherency_compatible = false;
    assert!(matches!(
        memory.validate_zero_copy_eligibility(bad_coherency),
        Err(MemoryError::ZeroCopyCoherencyUnsupported { .. })
    ));
}

#[test]
fn explicit_movement_can_elide_or_preserve_host_staging_policy_and_completion() {
    let resource = tensor_id("moving-resource");
    let mut memory = MemoryManager::default();
    memory
        .record_resource_residency(ResourceResidency::new(
            resource.clone(),
            MemoryDomain::DeviceLocal(gpu(0)),
        ))
        .unwrap();

    assert!(
        memory
            .movement_for_residency(
                &resource,
                MemoryDomain::DeviceLocal(gpu(0)),
                HostStagingPolicy::Forbid,
            )
            .unwrap()
            .is_none()
    );

    let movement = memory
        .movement_for_residency(
            &resource,
            MemoryDomain::DeviceLocal(gpu(1)),
            HostStagingPolicy::Forbid,
        )
        .unwrap()
        .unwrap();
    assert_eq!(movement.kind, ResourceMovementKind::DeviceToDevice);

    let denied = ResourceMovement::new(
        resource.clone(),
        &MemoryDomain::DeviceLocal(gpu(0)),
        MemoryDomain::DeviceLocal(gpu(1)),
        HostStagingPolicy::Forbid,
    )
    .requiring_host_staging();
    assert!(matches!(
        denied.validate(),
        Err(MemoryError::TransferHostStagingDenied { .. })
    ));

    let completed = memory
        .complete_movement(movement, CompletionTokenId::new(7, 1))
        .unwrap();
    assert_eq!(completed.memory_domain, MemoryDomain::DeviceLocal(gpu(1)));
}

#[test]
fn peer_access_import_export_pressure_and_redaction_are_contract_checked() {
    let mut memory = MemoryManager::default();
    let mut domains = BTreeSet::new();
    domains.insert(MemoryDomain::DeviceLocal(gpu(0)));
    domains.insert(MemoryDomain::Host);
    let capability = MemoryCapabilityDescriptor {
        memory_domains: domains,
        host_mapping: true,
        coherent_mapping: true,
        non_coherent_mapping: true,
        pinned_host_allocation: true,
        shared_memory: true,
        managed_memory: true,
        peer_access: vec![PeerAccessCapability::new(
            ProviderBinding::new("provider-a"),
            gpu(0),
            gpu(1),
            [PeerAccessMode::PeerRead, PeerAccessMode::PeerCopy],
        )],
        peer_transfer: true,
    };

    memory
        .validate_peer_zero_copy(&capability, &gpu(0), &gpu(1), PeerAccessMode::PeerRead)
        .unwrap();
    assert!(matches!(
        memory.validate_peer_zero_copy(&capability, &gpu(1), &gpu(0), PeerAccessMode::PeerRead),
        Err(MemoryError::PeerAccessUnsupported { .. })
    ));

    let import = ResourceImportDescriptor {
        size_bytes: 1024,
        alignment_bytes: 64,
        access: ResourceMappingAccess::ReadWrite,
        provider: ProviderBinding::new("provider-a"),
        device: Some(gpu(0)),
        lifetime_description: "logical external resource lifetime".into(),
    };
    memory
        .validate_resource_import(&import, &capability)
        .unwrap();

    let forbidden_import = ResourceImportDescriptor {
        lifetime_description: "cuda ipc 0xdeadbeef".into(),
        ..import
    };
    assert!(matches!(
        memory.validate_resource_import(&forbidden_import, &capability),
        Err(MemoryError::NativeHandleForbidden { .. })
    ));

    memory
        .validate_resource_export(&ResourceExportPolicy {
            allowed: true,
            policy_id: "test-policy".into(),
            exposes_native_handle: false,
        })
        .unwrap();
    assert!(matches!(
        memory.validate_resource_export(&ResourceExportPolicy {
            allowed: true,
            policy_id: "bad-policy".into(),
            exposes_native_handle: true,
        }),
        Err(MemoryError::NativeHandleForbidden { .. })
    ));

    memory
        .pin_residency(ResidencyPin {
            resource: tensor_id("pinned-kv"),
            domain: MemoryDomain::DeviceLocal(gpu(0)),
            capacity_class: MemoryAllocationClass::KvCache,
            bounded_bytes: 128,
        })
        .unwrap();
    let pressure = memory
        .pressure_action(
            tensor_id("pinned-kv"),
            &MemoryDomain::DeviceLocal(gpu(0)),
            MemoryDomain::Host,
            true,
            HostStagingPolicy::Permit,
        )
        .unwrap();
    assert!(matches!(pressure, MemoryPressureAction::RejectAdmission(_)));
    assert!(
        memory
            .observations()
            .iter()
            .any(|event| event.kind == MemoryObservationKind::PeerAccessUsed)
    );
}

#[test]
fn hard_residency_requirement_and_device_resident_roles_are_enforced() {
    let mut memory = MemoryManager::default();
    let weights = tensor_id("weights");
    let kv = tensor_id("kv");
    let workspace = tensor_id("workspace");
    for resource in [weights.clone(), kv.clone(), workspace.clone()] {
        memory
            .record_resource_residency(ResourceResidency::new(
                resource,
                MemoryDomain::DeviceLocal(gpu(0)),
            ))
            .unwrap();
    }
    let requirement = ResidencyRequirement {
        required_domain: MemoryDomain::DeviceLocal(gpu(0)),
        required_provider: None,
        required_device: Some(gpu(0)),
    };
    assert!(
        memory
            .validate_residency_requirement(&weights, &requirement)
            .is_ok()
    );
    assert!(
        memory
            .validate_residency_requirement(
                &kv,
                &ResidencyRequirement {
                    required_domain: MemoryDomain::DeviceLocal(gpu(1)),
                    required_provider: None,
                    required_device: Some(gpu(1)),
                },
            )
            .is_err()
    );
    assert!(
        memory
            .resource_residency(&workspace)
            .unwrap()
            .authoritative()
            .unwrap()
            .is_device_resident()
    );
}
