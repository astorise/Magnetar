//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

use crate::affinity::{DeviceBinding, FallbackClass, ProviderBinding, ResourceAffinity};
use crate::compute::{HostStagingPolicy, TensorResourceId};
use crate::device::DeviceId;
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
