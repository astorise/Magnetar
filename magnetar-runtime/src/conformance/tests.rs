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

use crate::component::MAGNETAR_RUNTIME_VERSION;
use crate::compute::{
    COMPUTE_CAPABILITY_VERSION, ComputeCapabilitySupport, ComputeDType, ComputeDataMovementKind,
    ComputeDataMovementSupport, ComputeLayout, ComputeOperationFamily, ComputeOperationSupport,
    ComputePrecision, DataMovementSupport, HostBufferEncoding, OperationFamilySupport,
    ProviderComputeAdvertisement, compute_capability,
};
use crate::runtime::Runtime;
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

#[test]
fn provider_conformance_suite_reports_core_compute_and_data_movement_success() {
    let mut provider = TestProvider::new("magnetar.test.conformant");
    provider.metadata.capabilities.insert(compute_capability());
    let operation_support = ComputeOperationSupport::new()
        .with_dtypes([ComputeDType::Float32])
        .with_layouts([ComputeLayout::Dense])
        .with_precision_modes([ComputePrecision::Default]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_operation_family(OperationFamilySupport::from_operation_support(
            ComputeOperationFamily::Elementwise,
            operation_support,
        ))
        .with_data_movement(DataMovementSupport::from_compute_support(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_host_encodings([HostBufferEncoding::RawBytes]),
        ));

    let suite =
        ProviderConformanceSuite::new(ProviderConformanceConfig::default().with_profiles([
            ProviderConformanceProfile::ProviderCore,
            ProviderConformanceProfile::ProviderCompute,
            ProviderConformanceProfile::ProviderDataMovement,
            ProviderConformanceProfile::ProviderObservability,
        ]));
    let report = suite.run(ProviderConformanceTarget::mock(Arc::new(provider)));

    assert!(report.is_conformant(), "{report:#?}");
    assert_eq!(report.suite_version, PROVIDER_CONFORMANCE_SUITE_VERSION);
    assert_eq!(report.runtime_version, MAGNETAR_RUNTIME_VERSION);
    assert!(report.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderCompute
            && result.requirement.contains("elementwise")
    }));
    assert!(report.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDataMovement
            && result.requirement.contains("upload")
    }));

    let json = provider_conformance_report_json(&report).unwrap();
    assert!(json.contains("\"provider_identity\": \"magnetar.test.conformant\""));
    assert!(json.contains("\"suite_version\""));
}

#[test]
fn first_native_model_execution_profile_validation_rejects_incomplete_profiles() {
    let mut profile = first_native_model_execution_profile();
    profile.version.clear();
    assert!(matches!(
        profile.validate(),
        Err(FirstNativeModelExecutionProfileError::MissingVersion)
    ));

    let mut profile = first_native_model_execution_profile();
    profile
        .mandatory_capabilities
        .remove(&FirstNativeProfileCapability::KernelRegistry);
    assert!(matches!(
        profile.validate(),
        Err(
            FirstNativeModelExecutionProfileError::MissingMandatoryCapability(
                FirstNativeProfileCapability::KernelRegistry
            )
        )
    ));

    let mut profile = first_native_model_execution_profile();
    profile
        .deferred_capabilities
        .remove(&FirstNativeDeferredCapability::TensorParallel);
    assert!(matches!(
        profile.validate(),
        Err(
            FirstNativeModelExecutionProfileError::MissingDeferredCapability(
                FirstNativeDeferredCapability::TensorParallel
            )
        )
    ));
}

#[test]
fn first_native_model_execution_profile_declares_versioned_mandatory_capabilities() {
    let profile = first_native_model_execution_profile();

    assert_eq!(
        profile.version,
        FIRST_NATIVE_MODEL_EXECUTION_PROFILE_VERSION
    );
    assert!(profile.validate().is_ok());

    let mandatory = profile.mandatory_ids();
    for expected in [
        "local-runtime",
        "platform-component-engine",
        "wasmtime-component-engine",
        "model-wasm-component",
        "model-artifact",
        "tokenizer",
        "operator-catalog",
        "execution-graph",
        "prepared-execution-plan",
        "kernel-registry",
        "reference-cpu-provider",
        "logical-cpu-device",
        "f32-execution",
        "tensor-resource",
        "runtime-memory-manager",
        "kv-cache",
        "incremental-decode",
        "greedy-sampling",
        "streaming-output",
        "observability-redaction",
    ] {
        assert!(
            mandatory.contains(expected),
            "missing mandatory capability {expected}"
        );
    }
}

#[test]
fn first_native_model_execution_profile_defers_advanced_capabilities() {
    let profile = first_native_model_execution_profile();
    let deferred = profile.deferred_ids();

    for expected in [
        "multi-device-placement",
        "tensor-parallel",
        "collectives",
        "generated-kernels",
        "provider-runtime-compilation",
        "kernel-artifact-ingestion",
        "hot-swap",
        "runtime-autotuning",
        "adaptive-performance-feedback",
        "performance-model-replacement",
        "accelerated-providers",
        "reduced-precision",
        "quantization",
        "cross-provider-zero-copy",
        "advanced-memory-pools",
        "paged-kv-cache",
        "prefix-cache-optimization",
        "production-continuous-batching",
        "advanced-async-execution-streams",
    ] {
        assert!(
            deferred.contains(expected),
            "missing deferred capability {expected}"
        );
        assert!(
            !profile.mandatory_ids().contains(expected),
            "deferred capability {expected} must not be mandatory"
        );
    }
}

#[test]
fn first_native_single_host_topology_accepts_one_reference_cpu_runtime() {
    let runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();

    assert!(validate_first_native_single_host_topology(&runtime).is_ok());
}

#[test]
fn first_native_single_host_topology_requires_reference_cpu_provider_and_device() {
    let runtime = Runtime::builder().build().unwrap();
    assert!(matches!(
        validate_first_native_single_host_topology(&runtime),
        Err(FirstNativeModelExecutionProfileError::ReferenceCpuProviderMissing)
    ));
}
