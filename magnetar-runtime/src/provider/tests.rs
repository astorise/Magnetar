//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{
    CapabilityBinding, CapabilityHealth, HealthState, ProviderBinding, ProviderHealth,
    ProviderHealthReport, ProviderLifecycleState, ProviderReadinessState, ProviderStatusSnapshot,
};
use crate::capability::CapabilityId;
use crate::compute::{ComputeError, ComputeErrorCode, ComputeErrorPhase};
use crate::device::{Device, DeviceDescriptor, DeviceId, DeviceMetadata, DeviceType};
use crate::kernel::KernelAdvertisement;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::capability::{Capability, CapabilityDescriptor, CapabilityVersion};
use crate::component::WitInterface;
struct TestProvider {
    metadata: ProviderMetadata,
    initialized: AtomicBool,
    shut_down: AtomicBool,
    fail_initialization: bool,
    health: ProviderHealth,
    status_snapshot: Option<ProviderStatusSnapshot>,
    capability_health: BTreeMap<CapabilityId, HealthState>,
    devices: Vec<Arc<dyn Device>>,
    execution_api: Option<Arc<dyn ProviderExecutionApi>>,
    kernel_advertisements: Vec<KernelAdvertisement>,
}

impl TestProvider {
    fn new(name: &str) -> Self {
        Self {
            metadata: ProviderMetadata::new(name, "1", "test", "test"),
            initialized: AtomicBool::new(false),
            shut_down: AtomicBool::new(false),
            fail_initialization: false,
            health: ProviderHealth::Available,
            status_snapshot: None,
            capability_health: BTreeMap::new(),
            devices: Vec::new(),
            execution_api: None,
            kernel_advertisements: Vec::new(),
        }
    }
}

impl Provider for TestProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        self.kernel_advertisements.clone()
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn health(&self) -> ProviderHealth {
        self.health
    }
    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        self.status_snapshot
            .clone()
            .unwrap_or_else(|| ProviderStatusSnapshot::from_health_report(self.health_report()))
    }
    fn capability_health(&self, capability: &CapabilityBinding) -> Option<CapabilityHealth> {
        Some(CapabilityHealth::new(
            ProviderBinding::new(&self.metadata.name),
            capability.clone(),
            self.capability_health
                .get(capability.id())
                .copied()
                .unwrap_or(self.health),
        ))
    }
    fn initialize(&self) -> Result<(), ProviderError> {
        if self.fail_initialization {
            return Err(ProviderError::Lifecycle("unavailable".into()));
        }
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        self.devices.clone()
    }
    fn shutdown(&self) -> Result<(), ProviderError> {
        self.shut_down.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        self.execution_api.clone()
    }
}

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
fn reject_incompatible() {
    let mut p = TestProvider::new("old");
    p.metadata.api_version += 1;
    assert!(matches!(
        ProviderLoader::new().register_provider(Arc::new(p)),
        Err(ProviderError::IncompatibleApiVersion { .. })
    ));
}

#[test]
fn reject_duplicate() {
    let mut m = ProviderLoader::new();
    m.register_provider(Arc::new(TestProvider::new("same")))
        .unwrap();
    assert!(matches!(
        m.register_provider(Arc::new(TestProvider::new("same"))),
        Err(ProviderError::ProviderAlreadyRegistered(_))
    ));
}

#[test]
fn dynamic_provider_loading_denies_paths_by_default() {
    let mut loader = ProviderLoader::new();
    let path = std::path::PathBuf::from("target/test-provider.dll");

    let result = unsafe { loader.load_dynamic(&path) };

    assert!(matches!(
        result,
        Err(ProviderError::ProviderPathDenied { path: denied }) if denied == path
    ));
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

#[test]
fn load_valid_provider() {
    let p = Arc::new(TestProvider::new("valid"));
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("valid").is_some());
    assert!(p.initialized.load(Ordering::SeqCst));
}

#[test]
fn provider_shutdown_releases_registered_provider() {
    let p = TestProvider::new("provider");
    let p = Arc::new(p);
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("provider").is_some());
    m.shutdown().unwrap();
    assert!(p.shut_down.load(Ordering::SeqCst));
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
fn registers_capabilities_and_resolves_fallbacks_by_name() {
    let capability = capability("magnetar:runtime/execute", CapabilityVersion::new(1, 0, 0));
    let mut primary = TestProvider::new("a-primary");
    primary.metadata.capabilities.insert(capability.clone());
    let mut fallback = TestProvider::new("z-fallback");
    fallback.metadata.capabilities.insert(capability.clone());
    let mut loader = ProviderLoader::new();
    loader.register_provider(Arc::new(fallback)).unwrap();
    loader.register_provider(Arc::new(primary)).unwrap();

    assert!(loader.registry().has_capability(&capability));
    let names = loader
        .resolve_providers(&capability)
        .into_iter()
        .map(|provider| provider.metadata().name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["a-primary", "z-fallback"]);
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
fn dynamic_provider_loading_rejects_legacy_trait_object_factory_contract() {
    let mut loader = ProviderLoader::new();
    let root = std::path::PathBuf::from("target");
    let path = root.join("test-provider.dll");
    let policy = ProviderLoadingPolicy::development([root]);

    let result = unsafe { loader.load_dynamic_with_policy(&path, &policy) };

    assert!(matches!(
        result,
        Err(ProviderError::UnsupportedDynamicAbi {
            expected_symbol: PROVIDER_ABI_FACTORY_SYMBOL_V1,
            expected_descriptor,
            ..
        }) if expected_descriptor.abi_version == ProviderAbiVersion::CURRENT
    ));
    assert!(
        !include_str!("../provider.rs").contains("Box<dyn Provider>"),
        "stable dynamic loading must not use Rust trait-object factories"
    );
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
