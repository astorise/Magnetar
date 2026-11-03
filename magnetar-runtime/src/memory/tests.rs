//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

use crate::affinity::{
    DeviceAvailability, ProviderHealth, ProviderHealthReport, ProviderStatusSnapshot,
};
use crate::affinity::{DeviceBinding, FallbackClass, ProviderBinding, ResourceAffinity};
use crate::compute::{ComputeDType, DTypeDescriptor, ShapeDescriptor, TensorDescriptor};
use crate::compute::{HostStagingPolicy, TensorResourceId};
use crate::device::DeviceId;
use crate::device::{DeviceMetadata, DeviceType};
use crate::tensor::TensorMemoryClass;
#[test]
fn memory_manager_admission_uses_pressure_and_queue_policy() {
    let manager = MemoryManager::default();
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        1024,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );
    let saturated = MemoryPressureSnapshot {
        runtime: MemoryPressureLevel::Saturated,
        ..MemoryPressureSnapshot::default()
    };

    assert!(matches!(
        manager.admit(MemoryAdmissionRequest {
            allocation: request.clone(),
            pressure: saturated.clone(),
            queue_allowed: false,
        }),
        MemoryAdmissionDecision::Reject { .. }
    ));
    assert!(matches!(
        manager.admit(MemoryAdmissionRequest {
            allocation: request,
            pressure: saturated,
            queue_allowed: true,
        }),
        MemoryAdmissionDecision::Queue { .. }
    ));
}

#[test]
fn memory_manager_reuses_cached_allocations_and_evicts_over_limit() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_cached_bytes: 512,
        ..MemoryManagerConfig::default()
    });
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        256,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );

    let first = manager.allocate(request.clone()).unwrap();
    manager.release(first.id).unwrap();
    let second = manager.allocate(request).unwrap();

    assert_eq!(first.id, second.id);
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == MemoryObservationKind::CacheHit
            && observation.allocation == Some(first.id)
    }));

    manager.release(second.id).unwrap();
    let other = manager
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::TemporaryWorkspace,
            768,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Runtime,
        ))
        .unwrap();
    manager.release(other.id).unwrap();

    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::CacheEviction })
    );
}

#[test]
fn running_allocation_totals_match_a_full_recount_through_reuse_and_eviction() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_cached_bytes: 512,
        max_runtime_bytes: Some(4096),
        ..MemoryManagerConfig::default()
    });
    let request = |class, size, placement| {
        MemoryAllocationRequest::new(class, size, placement, MemoryAllocationOwner::Runtime)
    };
    let mut live = Vec::new();
    for (class, size, placement) in [
        (
            MemoryAllocationClass::Tensor,
            256,
            MemoryPlacement::HostOrdinary,
        ),
        (
            MemoryAllocationClass::Tensor,
            128,
            MemoryPlacement::HostPinned,
        ),
        (
            MemoryAllocationClass::TemporaryWorkspace,
            64,
            MemoryPlacement::HostOrdinary,
        ),
    ] {
        live.push(
            manager
                .allocate(request(class, size, placement))
                .unwrap()
                .id,
        );
        manager.assert_allocation_totals_consistent();
    }
    manager.release(live[0]).unwrap();
    manager.assert_allocation_totals_consistent();
    let reused = manager
        .allocate(request(
            MemoryAllocationClass::Tensor,
            200,
            MemoryPlacement::HostOrdinary,
        ))
        .unwrap();
    assert_eq!(reused.id, live[0], "the cached allocation is reused");
    manager.assert_allocation_totals_consistent();
    for id in [reused.id, live[1], live[2]] {
        manager.release(id).unwrap();
        manager.assert_allocation_totals_consistent();
    }
    let large = manager
        .allocate(request(
            MemoryAllocationClass::TemporaryWorkspace,
            768,
            MemoryPlacement::HostOrdinary,
        ))
        .unwrap();
    manager.release(large.id).unwrap();
    manager.assert_allocation_totals_consistent();

    let mut uncached = MemoryManager::default();
    let id = uncached
        .allocate(request(
            MemoryAllocationClass::Tensor,
            32,
            MemoryPlacement::HostOrdinary,
        ))
        .unwrap()
        .id;
    uncached.assert_allocation_totals_consistent();
    uncached.release(id).unwrap();
    uncached.assert_allocation_totals_consistent();
}

#[test]
fn memory_manager_tracks_arena_growth_shrink_pressure_and_diagnostics() {
    let mut manager = MemoryManager::default();
    let arena = manager
        .create_arena(
            MemoryAllocationClass::TemporaryWorkspace,
            MemoryPlacement::HostOrdinary,
            128,
            MemoryArenaOwner::Runtime,
        )
        .unwrap();
    let id = arena.id;
    *manager.arena_mut(id).unwrap() = MemoryArena::new(
        id,
        MemoryAllocationClass::TemporaryWorkspace,
        MemoryPlacement::HostOrdinary,
        128,
        MemoryArenaOwner::Runtime,
    )
    .with_growth(MemoryArenaGrowthPolicy::GrowOnDemand {
        increment_bytes: 128,
    })
    .with_shrink(MemoryArenaShrinkPolicy::ReleaseReusable);

    manager.reserve_in_arena(id, 192).unwrap();
    manager.shrink_arena(id).unwrap();

    let arena = manager.arenas().find(|arena| arena.id == id).unwrap();
    assert_eq!(arena.used_bytes, 192);
    assert!(arena.capacity_bytes >= 192);
    assert!(matches!(
        arena.pressure,
        MemoryPressureLevel::High | MemoryPressureLevel::Saturated
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ArenaPressure })
    );
}

#[test]
fn memory_manager_pending_queue_times_out_cancels_and_retries() {
    let mut timeout_manager = MemoryManager::default();
    timeout_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            )
            .with_alignment(8)
            .with_deadline_millis(20),
            10,
        )
        .unwrap();
    let errors = timeout_manager.expire_pending_allocations(20);
    assert!(matches!(
        errors.as_slice(),
        [MemoryError::AllocationTimeout { .. }]
    ));

    let mut cancel_manager = MemoryManager::default();
    let pending = cancel_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            ),
            10,
        )
        .unwrap();
    assert!(matches!(
        cancel_manager.cancel_pending_allocation(pending.allocation.id),
        Err(MemoryError::AllocationCancelled { .. })
    ));

    let mut retry_manager = MemoryManager::default();
    retry_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            ),
            0,
        )
        .unwrap();
    let results = retry_manager.retry_pending_allocations();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_ok());
    assert!(retry_manager.pending_allocations().next().is_none());
    assert!(
        retry_manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::PendingQueueDelay })
    );
}

#[test]
fn memory_manager_observes_zero_copy_staging_pinned_and_browser_policy() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_pinned_host_bytes: 16,
        allow_browser_linear_memory: false,
        ..MemoryManagerConfig::default()
    });
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let source = TensorResidency::new(
        TensorResourceId::new("tensor:0"),
        MemoryPlacement::HostOrdinary,
        affinity,
    );

    let accepted =
        manager.observed_zero_copy_feasibility(&source, &MemoryPlacement::HostOrdinary, None);
    assert!(accepted.feasible);
    let rejected = manager.observed_zero_copy_feasibility(
        &source,
        &MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu:0"))),
        None,
    );
    assert!(!rejected.feasible);

    assert!(
        manager
            .observed_staging_feasibility(HostStagingPolicy::Permit, 8)
            .feasible
    );
    assert!(
        !manager
            .observed_staging_feasibility(HostStagingPolicy::Forbid, 8)
            .feasible
    );
    assert!(matches!(
        manager.allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::BrowserLinearMemory,
            8,
            MemoryPlacement::BrowserLinearMemory,
            MemoryAllocationOwner::Runtime,
        )),
        Err(MemoryError::UnsupportedPlacement(_))
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ZeroCopyAccepted })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ZeroCopyRejected })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::StagingInserted })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::StagingDenied })
    );
}

#[test]
fn memory_manager_tracks_allocation_lifetime_and_tensor_residency() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(4096),
        ..MemoryManagerConfig::default()
    });
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let allocation = manager
        .allocate(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                256,
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new("compute")),
                MemoryAllocationOwner::Provider(ProviderBinding::new("compute")),
            )
            .with_affinity(affinity.clone()),
        )
        .unwrap();
    let tensor = TensorResourceId::new("tensor:0");

    manager
        .record_tensor_residency(
            TensorResidency::new(
                tensor.clone(),
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new("compute")),
                affinity,
            )
            .with_allocation(allocation.id),
        )
        .unwrap();

    let residency = manager.tensor_residency(&tensor).unwrap();
    assert_eq!(residency.allocation, Some(allocation.id));
    assert!(residency.provider_owned);
    manager.release(allocation.id).unwrap();
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == MemoryObservationKind::AllocationReleased
            && observation.allocation == Some(allocation.id)
    }));
    // `release()` only changes the allocation's own state -- it does not
    // remove the tensor's residency record (`invalidate-tensor-residency-
    // on-release`), so the record is still present here until a caller
    // explicitly removes it.
    assert!(manager.tensor_residency(&tensor).is_some());
    let removed = manager.remove_tensor_residency(&tensor).unwrap();
    assert_eq!(removed.tensor, tensor);
    assert!(manager.tensor_residency(&tensor).is_none());
    assert!(manager.remove_tensor_residency(&tensor).is_none());
    assert!(matches!(
        manager.record_tensor_residency(
            TensorResidency::new(
                TensorResourceId::new("tensor:bad"),
                MemoryPlacement::HostOrdinary,
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .with_allocation(MemoryAllocationId::new(999)),
        ),
        Err(MemoryError::InvalidAllocationHandle(_))
    ));
}

#[test]
fn memory_manager_rejects_forbidden_staging_and_incompatible_zero_copy() {
    let manager = MemoryManager::default();
    let staging = manager.staging_feasibility(HostStagingPolicy::Forbid, 128);
    assert!(!staging.feasible);
    assert!(staging.reason.contains("forbidden"));

    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let source = TensorResidency::new(
        TensorResourceId::new("tensor:0"),
        MemoryPlacement::HostOrdinary,
        affinity,
    );
    let zero_copy = manager.zero_copy_feasibility(
        &source,
        &MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu:0"))),
        None,
    );
    assert!(!zero_copy.feasible);
    assert!(zero_copy.reason.contains("incompatible"));
}

#[test]
fn memory_manager_reports_provider_device_cache_and_class_pressure() {
    let provider = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("compute"),
        ProviderHealth::Saturated,
    ));
    let provider_pressure = MemoryManager::pressure_for_provider_status(&provider);
    assert_eq!(provider_pressure.runtime, MemoryPressureLevel::Saturated);
    assert_eq!(
        provider_pressure.provider,
        Some((
            ProviderBinding::new("compute"),
            MemoryPressureLevel::Saturated
        ))
    );

    let mut metadata =
        DeviceMetadata::new(DeviceId::new("gpu:0"), "GPU", DeviceType::Gpu, "compute");
    metadata.memory_capacity = 100;
    let device_pressure =
        MemoryManager::pressure_for_device_metadata(&metadata, 95, DeviceAvailability::Available);
    assert_eq!(device_pressure.runtime, MemoryPressureLevel::Saturated);

    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(100),
        max_cached_bytes: 100,
        ..MemoryManagerConfig::default()
    });
    manager
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::KvCache,
            80,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Session("session:0".into()),
        ))
        .unwrap();
    let pressure = manager.pressure_snapshot();
    assert_eq!(pressure.runtime, MemoryPressureLevel::High);
    assert_eq!(pressure.kv_cache, Some(MemoryPressureLevel::High));
    assert!(pressure.cache.is_some());
}

#[test]
fn tensor_residency_tracks_eviction_size_estimate_and_host_visibility() {
    let host = TensorResidency::new(
        TensorResourceId::new("residency-host"),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_eviction_eligible(true)
    .with_size_estimate(4096);
    assert!(host.eviction_eligible);
    assert_eq!(host.size_bytes_estimate, Some(4096));
    assert!(host.is_host_visible());
    assert_eq!(host.memory_class(), TensorMemoryClass::Host);

    let device = TensorResidency::new(
        TensorResourceId::new("residency-device"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-0"))),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert!(!device.is_host_visible());
    assert_eq!(device.memory_class(), TensorMemoryClass::Device);
}

#[test]
fn memory_manager_admits_tensor_computed_from_descriptor_size() {
    let manager = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let decision = manager.admit_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
        MemoryPressureSnapshot::default(),
    );
    assert!(matches!(decision, MemoryAdmissionDecision::Admit { .. }));
}

#[test]
fn memory_manager_rejects_tensor_admission_when_size_is_unknown() {
    let manager = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([u64::MAX, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let decision = manager.admit_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
        MemoryPressureSnapshot::default(),
    );
    assert!(matches!(decision, MemoryAdmissionDecision::Reject { .. }));
}

fn small_tensor_request() -> MemoryAllocationRequest {
    MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        64,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    )
}

#[test]
fn released_allocations_leave_the_live_table_but_stay_observable() {
    let mut manager = MemoryManager::default();
    let kept = manager.allocate(small_tensor_request()).unwrap();
    let released = manager.allocate(small_tensor_request()).unwrap();
    manager.release(released.id).unwrap();

    assert!(manager.is_live_allocation(kept.id));
    assert!(!manager.is_live_allocation(released.id));
    let states: Vec<_> = manager
        .allocations()
        .map(|allocation| (allocation.id, allocation.state))
        .collect();
    assert_eq!(
        states,
        vec![
            (kept.id, MemoryAllocationState::Active),
            (released.id, MemoryAllocationState::Released),
        ],
        "live and retired allocations are merged back in id order"
    );
    assert_eq!(manager.bookkeeping_sizes()["live_allocations"], 1);
    assert!(matches!(
        manager.release(released.id),
        Err(MemoryError::InvalidAllocationHandle(_))
    ));
    manager
        .record_tensor_residency(
            TensorResidency::new(
                TensorResourceId::new("retired"),
                MemoryPlacement::HostOrdinary,
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .with_allocation(released.id),
        )
        .expect("a retired allocation is still a known handle");
}

#[test]
fn retired_history_and_observation_logs_stay_bounded() {
    let mut manager = MemoryManager::default();
    let first = manager.allocate(small_tensor_request()).unwrap();
    manager.release(first.id).unwrap();
    for _ in 0..RETIRED_ALLOCATION_HISTORY_CAPACITY + 10 {
        let allocation = manager.allocate(small_tensor_request()).unwrap();
        manager.release(allocation.id).unwrap();
    }
    let sizes = manager.bookkeeping_sizes();
    assert_eq!(sizes["live_allocations"], 0);
    assert_eq!(
        sizes["retired_allocations"],
        RETIRED_ALLOCATION_HISTORY_CAPACITY
    );
    assert!(sizes["observations"] < 2 * MEMORY_OBSERVATION_LOG_CAPACITY);
    assert!(
        manager
            .allocations()
            .all(|allocation| allocation.id != first.id),
        "the oldest retired allocation was forgotten"
    );
    assert!(matches!(
        manager.release(first.id),
        Err(MemoryError::InvalidAllocationHandle(_))
    ));
}

#[test]
fn reusable_allocations_stay_live_and_evictions_retire() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_cached_bytes: 64,
        ..MemoryManagerConfig::default()
    });
    let a = manager.allocate(small_tensor_request()).unwrap();
    let b = manager.allocate(small_tensor_request()).unwrap();
    manager.release(a.id).unwrap();
    assert!(
        manager.is_live_allocation(a.id),
        "a reusable allocation stays live"
    );
    manager.release(b.id).unwrap();
    let live = [a.id, b.id]
        .iter()
        .filter(|id| manager.is_live_allocation(**id))
        .count();
    assert_eq!(
        live, 1,
        "exceeding the cache limit retires one reusable allocation"
    );
    let reused = manager.allocate(small_tensor_request()).unwrap();
    assert!(reused.id == a.id || reused.id == b.id);
}
