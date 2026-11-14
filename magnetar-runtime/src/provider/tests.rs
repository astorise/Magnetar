//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::capability::CapabilityId;
use crate::device::{Device, DeviceDescriptor, DeviceId, DeviceMetadata, DeviceType};
use crate::{ComputeError, ComputeErrorCode, ComputeErrorPhase};
use crate::{
    HealthState, ProviderBinding, ProviderHealthReport, ProviderLifecycleState,
    ProviderReadinessState, ProviderStatusSnapshot,
};
use std::sync::Arc;

use crate::capability::{Capability, CapabilityDescriptor, CapabilityVersion};
use crate::component::WitInterface;

#[test]
fn device_registration_rejects_duplicate_ids_and_mismatched_owners() {
    let device = |id: &str, provider: &str| {
        Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            DeviceId::new(id),
            "test",
            DeviceType::Gpu,
            provider,
        ))) as Arc<dyn Device>
    };
    let mut registry = ProviderRegistry::default();
    registry
        .register_devices("cuda", [device("gpu:0", "cuda")])
        .unwrap();
    assert!(matches!(
        registry.register_devices("other", [device("gpu:0", "other")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:2", "cuda"), device("gpu:0", "cuda")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(registry.device(&DeviceId::new("gpu:2")).is_none());
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:1", "other")]),
        Err(ProviderError::DeviceProviderMismatch { .. })
    ));
}

#[test]
fn provider_lifecycle_transitions_and_drain_completion_are_explicit() {
    assert!(ProviderLifecycleState::Registered.can_transition_to(ProviderLifecycleState::Loading));
    assert!(
        ProviderLifecycleState::Loading.can_transition_to(ProviderLifecycleState::Initializing)
    );
    assert!(ProviderLifecycleState::Initializing.can_transition_to(ProviderLifecycleState::Ready));
    assert!(ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Draining));
    assert!(ProviderLifecycleState::Draining.can_transition_to(ProviderLifecycleState::Stopped));
    assert!(!ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Removed));

    let mut draining = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Draining,
    ));
    draining.lifecycle = ProviderLifecycleState::Draining;
    draining.readiness = ProviderReadinessState::Draining;
    draining.in_flight_operations = 1;
    assert!(!draining.is_drain_complete());
    assert!(draining.pinned_work_allowed_during_drain());
    draining.in_flight_operations = 0;
    assert!(draining.is_drain_complete());
}

#[test]
fn provider_abi_handles_lifecycle_and_errors_are_internal_runtime_contracts() {
    let instance = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderInstance,
        ProviderAbiHandle::new(7),
    );
    let resource = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderResource,
        ProviderAbiHandle::new(8),
    );
    let operation = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::Operation,
        ProviderAbiHandle::new(9),
    );

    assert!(instance.destroy_required);
    assert_eq!(resource.handle.as_u64(), 8);
    assert_eq!(operation.kind, ProviderAbiHandleKind::Operation);
    assert!(
        ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::LibraryLoaded)
    );
    assert!(
        !ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::Registered)
    );
    assert!(
        ProviderAbiLoadingLifecycle::Failed
            .can_transition_to(ProviderAbiLoadingLifecycle::Destroyed)
    );

    let categories = [
        ProviderAbiErrorCode::InvalidAbiDescriptor,
        ProviderAbiErrorCode::UnsupportedAbiVersion,
        ProviderAbiErrorCode::InvalidMetadata,
        ProviderAbiErrorCode::InvalidAdvertisement,
        ProviderAbiErrorCode::InvalidDeviceMetadata,
        ProviderAbiErrorCode::InitializationFailure,
        ProviderAbiErrorCode::ProviderNotReady,
        ProviderAbiErrorCode::ProviderDraining,
        ProviderAbiErrorCode::ProviderSaturated,
        ProviderAbiErrorCode::ExecutionRejected,
        ProviderAbiErrorCode::ExecutionFailed,
        ProviderAbiErrorCode::CancellationUnsupported,
        ProviderAbiErrorCode::CancellationFailed,
        ProviderAbiErrorCode::ResourceInvalid,
        ProviderAbiErrorCode::InternalProviderError,
        ProviderAbiErrorCode::PanicOrUnwindViolation,
    ];
    assert_eq!(categories.len(), 16);

    let compute_error = ComputeError::from(ProviderError::PanicOrUnwindViolation(
        "panic crossed boundary".into(),
    ));
    assert_eq!(compute_error.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(compute_error.phase, ComputeErrorPhase::Resolution);
}

fn capability(name: &str, version: CapabilityVersion) -> Capability {
    Capability::new(
        CapabilityId::new(name),
        version,
        CapabilityDescriptor::new("test capability")
            .with_contract(WitInterface::new(name, version.to_string())),
    )
}

#[test]
fn semantic_versions_select_the_latest_compatible_capability() {
    let mut registry = ProviderRegistry::default();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 0, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(2, 0, 0),
        ))
        .unwrap();

    let id = CapabilityId::new("magnetar:compute/run");
    assert_eq!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(1, 1, 0))
            .unwrap()
            .version,
        CapabilityVersion::new(1, 2, 0)
    );
    assert!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(3, 0, 0))
            .is_none()
    );
    assert!(!CapabilityVersion::new(0, 1, 1).is_compatible_with(CapabilityVersion::new(0, 1, 0)));
}

#[test]
fn provider_abi_descriptor_validates_version_layout_functions_and_ownership() {
    let descriptor = ProviderAbiDescriptor::current();
    descriptor.validate().unwrap();
    assert_eq!(descriptor.abi_version, ProviderAbiVersion::CURRENT);
    assert!(descriptor.features.contains(&ProviderAbiFeature::Execution));
    assert_eq!(
        descriptor.threading,
        ProviderAbiThreadingModel::RuntimeSynchronized
    );
    assert_eq!(
        descriptor.execution_behavior,
        ProviderAbiExecutionBehavior::Blocking
    );
    assert_eq!(
        descriptor.unload_policy,
        ProviderAbiUnloadPolicy::NeverUnload
    );
    assert!(descriptor.ownership.strings.release_required);
    assert!(!descriptor.ownership.runtime_buffers.release_required);

    let mut too_small = descriptor.clone();
    too_small.descriptor_size = 1;
    assert!(matches!(
        too_small.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut unsupported_major = descriptor.clone();
    unsupported_major.abi_version = ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION + 1, 0);
    assert!(matches!(
        unsupported_major.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut unsupported_minor = descriptor.clone();
    unsupported_minor.abi_version =
        ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION, PROVIDER_ABI_MINOR_VERSION + 1);
    assert!(matches!(
        unsupported_minor.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut missing_status = descriptor.clone();
    missing_status.functions.status = false;
    assert!(matches!(
        missing_status.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut cross_allocator = descriptor;
    cross_allocator.ownership.error_messages = ProviderAbiMemoryRule::runtime_borrowed();
    assert!(matches!(
        cross_allocator.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));
}

#[test]
fn capability_dependencies_must_resolve_compatibly() {
    let mut registry = ProviderRegistry::default();
    let dependent = Capability::new(
        CapabilityId::new("magnetar:app/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("dependent")
            .with_contract(WitInterface::new("magnetar:app/run", "1.0.0"))
            .with_dependency(
                CapabilityId::new("magnetar:compute/run"),
                CapabilityVersion::new(1, 1, 0),
            ),
    );
    registry.register_capability(dependent).unwrap();
    assert!(matches!(
        registry.validate_dependencies(),
        Err(ProviderError::MissingCapabilityDependency { .. })
    ));
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry.validate_dependencies().unwrap();
}
