//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{
    CapabilityBinding, CapabilityHealth, HealthState, ProviderBinding, ProviderHealth,
    ProviderStatusSnapshot,
};
use crate::capability::CapabilityId;
use crate::device::Device;
use crate::kernel::KernelAdvertisement;
use crate::provider::{
    Provider, ProviderError, ProviderExecutionApi, ProviderLoadingPolicy, ProviderMetadata,
    ProviderRegistry,
};
use crate::reference_cpu::ReferenceCpuProvider;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

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
fn provider_conformance_suite_fails_invalid_public_metadata() {
    let mut provider = TestProvider::new("bad provider");
    provider.metadata.vendor.clear();
    provider.metadata.description = "raw handle 0xdeadbeef".into();

    let report = ProviderConformanceSuite::default()
        .run(ProviderConformanceTarget::mock(Arc::new(provider)));

    assert!(!report.is_conformant());
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "ProviderId syntax"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "vendor metadata"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "metadata redaction"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
}

#[test]
fn provider_conformance_suite_reports_dynamic_loading_policy() {
    let path = std::env::temp_dir().join("magnetar-provider-fixture.dll");
    let denied = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderDynamicAbi]),
    )
    .run(ProviderConformanceTarget::dynamic_library(
        &path,
        ProviderLoadingPolicy::dynamic_library([std::env::temp_dir().join("allowed")]),
    ));
    assert!(!denied.is_conformant());
    assert!(denied.failed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "allowed path loading"
    }));

    let allowed = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderDynamicAbi]),
    )
    .run(ProviderConformanceTarget::development(
        &path,
        ProviderLoadingPolicy::development([std::env::temp_dir()]),
    ));
    assert!(allowed.is_conformant(), "{allowed:#?}");
    assert!(allowed.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "ABI descriptor structure"
    }));
    assert!(allowed.skipped_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "factory symbol exists"
    }));
}

#[test]
fn provider_conformance_profile_ids_mark_hardware_profiles_optional() {
    let profile_ids = provider_conformance_profile_ids([
        ProviderConformanceProfile::ProviderCore,
        ProviderConformanceProfile::Cuda,
        ProviderConformanceProfile::Metal,
        ProviderConformanceProfile::OpenVino,
        ProviderConformanceProfile::Qnn,
    ]);

    assert!(profile_ids["provider-core"]);
    assert!(!profile_ids["provider-hardware-cuda"]);
    assert!(!profile_ids["provider-hardware-metal"]);
    assert!(!profile_ids["provider-hardware-openvino"]);
    assert!(!profile_ids["provider-hardware-qnn"]);
}

#[test]
fn reference_cpu_provider_passes_generic_conformance_core_profile() {
    let suite =
        ProviderConformanceSuite::new(ProviderConformanceConfig::default().with_profiles([
            ProviderConformanceProfile::ProviderCore,
            ProviderConformanceProfile::ProviderObservability,
        ]));
    let report = suite.run(ProviderConformanceTarget::BuiltIn {
        provider: Arc::new(ReferenceCpuProvider::new()),
    });
    assert!(
        report.is_conformant(),
        "Reference CPU Provider failed conformance: {:?}",
        report.failed_tests
    );
}
