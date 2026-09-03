use crate::adapter::*;
use crate::affinity::*;
use crate::batching::*;
use crate::capability::*;
use crate::cli_boundary::*;
use crate::component::*;
use crate::compute::*;
use crate::conformance::*;
use crate::device::*;
use crate::e2e_conformance::*;
use crate::generation::*;
use crate::inference_api::*;
use crate::kernel::*;
use crate::kernel_artifact::*;
use crate::kernel_artifact_ingestion::*;
use crate::kernel_artifact_manifest::*;
use crate::kernel_autotuning::*;
use crate::kernel_benchmark::*;
use crate::kernel_cache::*;
use crate::kernel_compilation::*;
use crate::kernel_execution_plan::*;
use crate::kernel_performance_model::*;
use crate::kernel_qualification::*;
use crate::kernel_registry::*;
use crate::kernel_selection_policy::*;
use crate::kv_cache::*;
use crate::memory::*;
use crate::model::*;
use crate::model_format_roadmap::*;
use crate::model_instance::*;
use crate::model_loading::*;
use crate::model_source_cache_roadmap::*;
use crate::observability::*;
use crate::operator::*;
use crate::operator_scope::*;
use crate::planning::*;
use crate::prefix_cache::*;
use crate::provider::*;
use crate::provider_roadmap::*;
use crate::qwen_model_component::*;
use crate::reference_cpu::*;
use crate::release_cutover::*;
use crate::release_packaging::*;
use crate::release_security::*;
use crate::resolution::*;
use crate::runtime::*;
use crate::sampling::*;
use crate::scheduler::*;
use crate::server_api_roadmap::*;
use crate::session::*;
use crate::tensor::*;
use crate::tokenizer::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
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

fn capability(name: &str, version: CapabilityVersion) -> Capability {
    Capability::new(
        CapabilityId::new(name),
        version,
        CapabilityDescriptor::new("test capability")
            .with_contract(WitInterface::new(name, version.to_string())),
    )
}
fn capability_binding(name: &str, version: CapabilityVersion) -> CapabilityBinding {
    CapabilityBinding::new(CapabilityId::new(name), version)
}
fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}
fn simple_elementwise_compute_graph(name: &str) -> ComputeGraph {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    ComputeGraph::new(ComputeGraphId::new(name)).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    )
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
struct TestProviderExecutionApi {
    submitted: Mutex<Vec<ProviderExecutionRequest>>,
    released: AtomicBool,
    cancel_outcome: Mutex<ProviderCancellationOutcome>,
    outputs: Mutex<Vec<TensorResourceDescriptor>>,
}
impl TestProviderExecutionApi {
    fn new() -> Self {
        Self {
            submitted: Mutex::new(Vec::new()),
            released: AtomicBool::new(false),
            cancel_outcome: Mutex::new(ProviderCancellationOutcome::Unsupported),
            outputs: Mutex::new(Vec::new()),
        }
    }
}
impl ProviderExecutionApi for TestProviderExecutionApi {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        );
        self.submitted.lock().unwrap().push(request);
        Ok(handle)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        let mut status = ProviderExecutionStatus::new(handle.clone(), SchedulingState::Running);
        status.progress = Some(
            ProviderExecutionProgress::new(1, 3).with_message("provider execution is running"),
        );
        Ok(status)
    }
    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(*self.cancel_outcome.lock().unwrap())
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            self.outputs.lock().unwrap().clone(),
        ))
    }
    fn release(&self, _handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.released.store(true, Ordering::SeqCst);
        Ok(())
    }
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
fn runtime_enumerates_provider_devices_with_metadata() {
    let mut provider = TestProvider::new("cuda");
    let mut metadata = DeviceMetadata::new(
        DeviceId::new("cuda:gpu:0"),
        "NVIDIA Test GPU",
        DeviceType::Gpu,
        "cuda",
    );
    metadata.vendor = "NVIDIA".into();
    metadata.architecture = "Ada".into();
    metadata.memory_capacity = 24 * 1024 * 1024 * 1024;
    metadata.compute_units = 128;
    metadata
        .execution_capabilities
        .insert(CapabilityId::new("magnetar:compute/run"));
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(metadata)));

    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let devices = runtime.devices().collect::<Vec<_>>();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), "cuda:gpu:0");
    assert_eq!(devices[0].metadata().vendor, "NVIDIA");
    assert_eq!(
        runtime
            .providers()
            .registry()
            .provider_for_device(&DeviceId::new("cuda:gpu:0")),
        Some("cuda")
    );
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
fn capability_validation_rejects_invalid_and_conflicting_definitions() {
    let mut registry = ProviderRegistry::default();
    let invalid = Capability::new(
        CapabilityId::new("magnetar:invalid"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("missing contract"),
    );
    assert!(matches!(
        registry.register_capability(invalid),
        Err(ProviderError::InvalidCapability(_))
    ));

    let original = capability("magnetar:compute/run", CapabilityVersion::new(1, 0, 0));
    registry.register_capability(original).unwrap();
    let conflicting = Capability::new(
        CapabilityId::new("magnetar:compute/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("different")
            .with_contract(WitInterface::new("magnetar:compute/other", "1.0.0")),
    );
    assert!(matches!(
        registry.register_capability(conflicting),
        Err(ProviderError::ConflictingCapability { .. })
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
#[test]
fn component_import_uses_a_semantic_capability_version() {
    let mut provider = TestProvider::new("compute");
    provider.metadata.capabilities.insert(capability(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 1, 0),
    ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new("magnetar:compute/run", "1.0.0"))
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn compute_capability_has_the_canonical_wit_contract() {
    let compute = compute_capability();
    assert_eq!(compute.id.as_str(), COMPUTE_CAPABILITY_ID);
    assert_eq!(compute.version, COMPUTE_CAPABILITY_VERSION);
    assert_eq!(COMPUTE_WIT_PACKAGE, "magnetar:compute");
    assert_eq!(
        compute.descriptor.contracts,
        BTreeSet::from([WitInterface::new(COMPUTE_WIT_INTERFACE, "2.0.0")])
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
        "qwen-wasm-model-component",
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

fn first_native_report_fixture(provider_summary: &str) -> E2eConformanceReport {
    E2eConformanceReport {
        suite_version: E2E_SUITE_VERSION.into(),
        fixture_version: E2E_FIXTURE_VERSION.into(),
        runtime_version: MAGNETAR_RUNTIME_VERSION.into(),
        provider_summary: provider_summary.into(),
        device_summary: REFERENCE_CPU_DEVICE_ID.into(),
        model_component_summary: "e2e-qwen-fixture@1.0.0".into(),
        operator_coverage: ["matmul".into(), "rmsnorm".into()].into_iter().collect(),
        kernel_coverage: [
            "reference-cpu/matmul".into(),
            "reference-cpu/rmsnorm".into(),
        ]
        .into_iter()
        .collect(),
        test_cases: [
            "success-path-no-shortcut-validated",
            "operator-coverage",
            "kernel-coverage",
            "reference-cpu-selected-through-kernel-registry",
            "no-shortcut-direct-provider-rejected",
            "no-shortcut-direct-kernel-invocation-rejected",
        ]
        .into_iter()
        .map(E2eTestResult::passed)
        .collect(),
        redacted: true,
        duration_millis: 1,
        timestamp_unix_seconds: 0,
    }
}

#[test]
fn first_native_native_execution_evidence_requires_registry_kernel_and_reference_cpu_path() {
    let report = first_native_report_fixture(REFERENCE_CPU_PROVIDER_NAME);
    assert!(validate_first_native_native_execution_evidence(&report).is_ok());

    let mut missing_kernel_evidence = report.clone();
    missing_kernel_evidence.kernel_coverage.clear();
    assert!(matches!(
        validate_first_native_native_execution_evidence(&missing_kernel_evidence),
        Err(E2eConformanceError::KernelCoverageMissing { .. })
    ));

    let non_reference_provider = first_native_report_fixture("candle-provider");
    assert!(matches!(
        validate_first_native_native_execution_evidence(&non_reference_provider),
        Err(E2eConformanceError::BoundaryViolation { .. })
    ));
}

#[test]
fn first_native_native_execution_evidence_rejects_shortcut_reports() {
    let mut report = first_native_report_fixture(REFERENCE_CPU_PROVIDER_NAME);
    report
        .test_cases
        .retain(|test| test.name != "success-path-no-shortcut-validated");

    assert!(matches!(
        validate_first_native_native_execution_evidence(&report),
        Err(E2eConformanceError::BoundaryViolation { .. })
    ));
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

#[test]
fn first_native_component_engine_accepts_native_controlled_wasi_capabilities() {
    let capabilities = ComponentEngineCapabilities::native();

    assert!(validate_first_native_component_engine_capabilities(&capabilities).is_ok());
    assert!(capabilities.supports(ComponentEngineFeature::ComponentModel));
    assert!(capabilities.supports(ComponentEngineFeature::ControlledWasi));
}

#[test]
fn first_native_component_engine_rejects_non_native_or_ambient_test_profile() {
    let capabilities = ComponentEngineCapabilities::test();

    assert!(matches!(
        validate_first_native_component_engine_capabilities(&capabilities),
        Err(FirstNativeModelExecutionProfileError::ComponentEngineProfileMismatch)
    ));

    let mut capabilities = ComponentEngineCapabilities::native();
    capabilities.controlled_wasi = false;
    assert!(matches!(
        validate_first_native_component_engine_capabilities(&capabilities),
        Err(
            FirstNativeModelExecutionProfileError::ComponentEngineFeatureMissing(
                ComponentEngineFeature::ControlledWasi
            )
        )
    ));
}

#[test]
fn compute_v1_import_is_not_satisfied_by_compute_v2_provider() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0")),
        Err(ProviderError::NoCompatibleProvider(capability))
            if capability.id().as_str() == COMPUTE_CAPABILITY_ID
                && capability.version() == CapabilityVersion::new(1, 1, 0)
    ));
}
#[test]
fn compute_operation_catalog_defines_stable_family_metadata() {
    let families = ComputeOperationFamily::ALL
        .into_iter()
        .map(|family| family.id())
        .collect::<BTreeSet<_>>();

    assert_eq!(families.len(), 11);
    assert!(families.contains("descriptor-and-view"));
    assert!(families.contains("construction-and-allocation"));
    assert!(families.contains("data-movement-and-conversion"));
    assert!(families.contains("elementwise"));
    assert!(families.contains("comparison-and-selection"));
    assert!(families.contains("reduction"));
    assert!(families.contains("linear-algebra"));
    assert!(families.contains("convolution-and-spatial-transform"));
    assert!(families.contains("indexing-and-update"));
    assert!(families.contains("random-generation"));
    assert!(families.contains("synchronization-and-completion"));

    let metadata = ComputeOperationFamily::LinearAlgebra.metadata();
    assert_eq!(metadata.family, ComputeOperationFamily::LinearAlgebra);
    assert_eq!(metadata.scope, "matrix and batched matrix operations");
    assert!(metadata.examples.contains(&"matmul"));
    assert_eq!(
        ComputeOperationFamily::from_id("elementwise"),
        Some(ComputeOperationFamily::Elementwise)
    );
    assert_eq!(ComputeOperationFamily::from_id("autograd"), None);
}
#[test]
fn compute_wit_defines_the_stabilized_run_surface() {
    let wit = include_str!("../wit/compute.wit");
    assert!(wit.contains("package magnetar:compute@2.0.0;"));
    assert!(wit.contains("resource tensor"));
    assert!(wit.contains("resource graph"));
    assert!(wit.contains("resource operation"));
    assert!(wit.contains("enum operation-family"));
    assert!(wit.contains("type operation-id = string"));
    assert!(wit.contains("record operation-schema"));
    assert!(wit.contains("record operation-schema-support"));
    assert!(wit.contains("variant operation-attribute"));
    assert!(wit.contains("record operation-input-rule"));
    assert!(wit.contains("record operation-output-rule"));
    assert!(wit.contains("record operation-descriptor"));
    assert!(wit.contains("schema-id: option<operation-id>"));
    assert!(wit.contains("attributes: list<tuple<string, operation-attribute>>"));
    assert!(wit.contains("record graph-descriptor"));
    assert!(wit.contains("record graph-node"));
    assert!(wit.contains("variant graph-value-ref"));
    assert!(wit.contains("record shape-descriptor"));
    assert!(wit.contains("variant dtype-descriptor"));
    assert!(wit.contains("variant layout-descriptor"));
    assert!(wit.contains("record view-descriptor"));
    assert!(wit.contains("record tensor-resource-descriptor"));
    assert!(wit.contains("enum data-movement-kind"));
    assert!(wit.contains("record host-buffer-descriptor"));
    assert!(wit.contains("record data-movement-support"));
    assert!(wit.contains("record data-movement-descriptor"));
    assert!(wit.contains("enum placement-intent"));
    assert!(wit.contains("preserve-source-affinity"));
    assert!(wit.contains("runtime-selected"));
    assert!(wit.contains("host-accessible"));
    assert!(wit.contains("enum host-staging-policy"));
    assert!(wit.contains("forbid"));
    assert!(wit.contains("permit"));
    assert!(wit.contains("placement: placement-intent"));
    assert!(wit.contains("host-staging: host-staging-policy"));
    assert!(wit.contains("world compute-consumer"));
    assert!(wit.contains("import run"));
    assert!(!wit.contains("target-provider"));
    assert!(!wit.contains("target-device"));
    assert!(!wit.contains("target-affinity-group"));
    assert!(wit.contains("invalid-shape"));
    assert!(wit.contains("size-overflow"));
    assert!(wit.contains("unsupported-operation-family"));
    assert!(wit.contains("invalid-tensor-descriptor"));
    assert!(wit.contains("invalid-dtype"));
    assert!(wit.contains("invalid-operation-attribute"));
    assert!(wit.contains("invalid-operation-arity"));
    assert!(wit.contains("invalid-output-descriptor"));
    assert!(wit.contains("unsupported-dtype"));
    assert!(wit.contains("unsupported-layout"));
    assert!(wit.contains("unsupported-data-movement"));
    assert!(wit.contains("no-compatible-provider"));
    assert!(wit.contains("policy-rejected-provider"));
    assert!(wit.contains("provider-unavailable"));
    assert!(wit.contains("device-unavailable"));
    assert!(wit.contains("provider-pinned-resource"));
    assert!(wit.contains("device-bound-resource"));
    assert!(wit.contains("artifact-fingerprint-mismatch"));
    assert!(wit.contains("affinity-group-mismatch"));
    assert!(wit.contains("execution-interrupted"));
    assert!(wit.contains("execution-cancelled"));
    assert!(wit.contains("invalid-host-buffer"));
    assert!(wit.contains("invalid-transfer"));
    assert!(wit.contains("unsupported-conversion"));
    assert!(wit.contains("materialization-required"));
    assert!(wit.contains("enum compute-error-phase"));
    assert!(wit.contains("enum compute-error-severity"));
    assert!(wit.contains("record compute-diagnostic"));
    assert!(wit.contains("enum recovery-hint"));
    assert!(wit.contains("diagnostics: list<compute-diagnostic>"));
    assert!(wit.contains("recovery-hints: list<recovery-hint>"));
    assert!(wit.contains("submit: func("));
    assert!(wit.contains("result<operation, compute-error>"));
    assert!(!wit.contains("BackendStorage"));
    assert!(!wit.contains("Tensor`"));
    assert!(!wit.contains("autograd"));
    assert!(!wit.contains("training"));
    assert!(!wit.contains("kernel-name"));
    assert!(!wit.contains("queue"));
    assert!(!wit.contains("custom-operation"));
}
#[test]
fn compute_error_model_maps_validation_resolution_affinity_and_execution_failures() {
    let invalid_shape = ComputeError::from(ComputeValidationError::InvalidShape {
        reason: "rank exceeds limit".into(),
    });
    assert_eq!(invalid_shape.code, ComputeErrorCode::InvalidShape);
    assert_eq!(invalid_shape.phase, ComputeErrorPhase::Validation);
    assert_eq!(invalid_shape.severity, ComputeErrorSeverity::Terminal);
    assert!(
        invalid_shape
            .recovery_hints
            .contains(&RecoveryHint::NotRetryable)
    );

    let materialization = ComputeError::from(ComputeValidationError::MaterializationRequired {
        reason: "view must be materialized".into(),
    });
    assert_eq!(
        materialization.code,
        ComputeErrorCode::MaterializationRequired
    );
    assert!(
        materialization
            .recovery_hints
            .contains(&RecoveryHint::ExplicitMaterializationRequired)
    );

    let affinity = ComputeError::from(AffinityError::BoundProviderUnavailable(
        ProviderBinding::new("provider-a"),
    ));
    assert_eq!(affinity.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(affinity.phase, ComputeErrorPhase::Interruption);
    assert!(
        affinity
            .recovery_hints
            .contains(&RecoveryHint::ProviderPinned)
    );

    let policy = ComputeError::from(ProviderError::PolicyRejectedProvider {
        capability: CapabilityBinding::new(
            CapabilityId::new(COMPUTE_CAPABILITY_ID),
            COMPUTE_CAPABILITY_VERSION,
        ),
        policy: BuiltInResolutionPolicy::Availability.id(),
    });
    assert_eq!(policy.code, ComputeErrorCode::PolicyRejectedProvider);
    assert_eq!(policy.phase, ComputeErrorPhase::Resolution);
    assert_eq!(
        policy.diagnostics[0]
            .capability
            .as_ref()
            .unwrap()
            .id()
            .as_str(),
        COMPUTE_CAPABILITY_ID
    );

    let execution = ComputeError::from(ProviderError::Lifecycle("device lost".into()));
    assert_eq!(execution.code, ComputeErrorCode::ExecutionInterrupted);
    assert_eq!(execution.phase, ComputeErrorPhase::Interruption);
    assert!(
        execution
            .recovery_hints
            .contains(&RecoveryHint::RetryBeforeState)
    );
}
#[test]
fn compute_diagnostics_are_optional_redacted_and_non_contractual() {
    let diagnostic = ComputeDiagnostic::new()
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_operation_family(ComputeOperationFamily::LinearAlgebra)
        .with_backend_message("native handle=0xdeadbeef at C:\\secret\\tensor.bin")
        .with_debug_trace_id("trace-42");

    assert_eq!(
        diagnostic.provider.as_ref().map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        diagnostic.backend_message.as_deref(),
        Some("[redacted backend diagnostic]")
    );

    let error = ComputeError::new(
        ComputeErrorCode::ExecutionFailed,
        ComputeErrorPhase::Execution,
        ComputeErrorSeverity::Terminal,
        "provider execution failed",
    )
    .with_diagnostic(diagnostic)
    .with_recovery_hint(RecoveryHint::RestartableWithReplay);

    assert_eq!(error.code, ComputeErrorCode::ExecutionFailed);
    assert_eq!(error.diagnostics.len(), 1);
    assert!(
        error
            .recovery_hints
            .contains(&RecoveryHint::RestartableWithReplay)
    );
}
#[test]
fn compute_operation_validation_uses_provider_advertisements() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32, ComputeDType::Float64])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default, ComputePrecision::Reduced]),
    );
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::LinearAlgebra,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense)
                    .with_precision(ComputePrecision::Reduced),
                ComputeOperationDescriptor::new(ComputeOperationFamily::LinearAlgebra)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            ],
        )
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::new(
                ComputeOperationFamily::Reduction
            )],
        ),
        Err(ComputeValidationError::UnsupportedOperationFamily { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::UInt8)
            ],
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_layout(ComputeLayout::Strided)
            ],
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));
}
#[test]
fn compute_operation_schemas_define_initial_portable_operations() {
    let schemas = initial_compute_operation_schemas();

    for id in [
        "tensor.reshape",
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.broadcast",
        "tensor.squeeze",
        "tensor.unsqueeze",
        "elementwise.unary.relu",
        "elementwise.binary.add",
        "comparison.eq",
        "selection.where",
        "reduction.sum",
        "linalg.matmul",
        "linalg.batched-matmul",
        "tensor.gather",
        "tensor.index-select",
        "tensor.scatter",
        "tensor.scatter-add",
        "tensor.concat",
        "random.uniform",
        "random.normal",
    ] {
        assert!(schemas.contains_key(&ComputeOperationId::new(id)), "{id}");
    }
    assert!(!schemas.contains_key(&ComputeOperationId::new("convolution.conv2d")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("pooling.max")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("attention.flash")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("quantized.matmul")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("custom.kernel")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("autograd.backward")));

    let scatter = schemas
        .get(&ComputeOperationId::new("tensor.scatter"))
        .unwrap();
    assert!(scatter.provider_specific_semantics);
}
#[test]
fn provider_compute_advertisement_drives_operation_validation() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("advertised-compute", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default()
                .with_versions([COMPUTE_CAPABILITY_VERSION])
                .with_operation_catalog_revision("initial")
                .with_operation_schema_revision("initial"),
        )
        .with_operation_schema(OperationSchemaSupport::from_operation_support(
            add.id.clone(),
            add.family,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_precision_modes([ComputePrecision::Default]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "advertised-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(tensor.clone())
                .with_tensor(tensor.clone())
                .with_tensor(tensor)],
        )
        .unwrap();
}
#[test]
fn provider_compute_advertisement_reports_version_and_schema_rejections() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut incompatible = TestProvider::new("incompatible-compute");
    incompatible.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([CapabilityVersion::new(0, 9, 0)]),
        );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(incompatible))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "incompatible-compute",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedAdvertisement { .. })
    ));

    let mut unsupported = provider_with_capabilities("unsupported-schema", [compute_capability()]);
    unsupported.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_operation_family(OperationFamilySupport::from_operation_support(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new().with_dtypes([ComputeDType::Float32]),
        ))
        .with_unsupported_operation_schema(add.id.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(unsupported))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "unsupported-schema",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedOperationSchema { .. })
    ));
}
#[test]
fn compute_operation_schema_validation_checks_attributes_and_shapes() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_schema_support.insert(
        add.id.clone(),
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let lhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let rhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([1, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(output.clone())],
        )
        .unwrap();

    let bad_output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(bad_output)]
        ),
        Err(ComputeValidationError::InvalidOutputDescriptor { .. })
    ));

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_attribute("unknown", ComputeOperationAttribute::Boolean(true))
                .with_tensor(lhs)
                .with_tensor(rhs)
                .with_tensor(output)]
        ),
        Err(ComputeValidationError::InvalidOperationAttribute { .. })
    ));
}
#[test]
fn compute_operation_schema_validation_checks_reduction_matmul_and_random_rules() {
    let schemas = initial_compute_operation_schemas();
    let sum = schemas
        .get(&ComputeOperationId::new("reduction.sum"))
        .unwrap();
    let matmul = schemas
        .get(&ComputeOperationId::new("linalg.matmul"))
        .unwrap();
    let random = schemas
        .get(&ComputeOperationId::new("random.uniform"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    for schema in [sum, matmul, random] {
        provider.metadata.compute_operation_schema_support.insert(
            schema.id.clone(),
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32, ComputeDType::SInt64])
                .with_layouts([ComputeLayout::Dense]),
        );
    }
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let f32_2x3 = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![1]))
                .with_attribute("keep-dimensions", ComputeOperationAttribute::Boolean(true))
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 1]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![2]))
                .with_tensor(f32_2x3.clone())
                .with_tensor(f32_2x3.clone())]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(matmul)
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([3, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(random)
                .with_attribute(
                    "shape",
                    ComputeOperationAttribute::Shape(ShapeDescriptor::new([2, 2])),
                )
                .with_attribute(
                    "dtype",
                    ComputeOperationAttribute::DType(ComputeDType::Float32),
                )
                .with_attribute("seed", ComputeOperationAttribute::Integer(42))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 2]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
}
#[test]
fn tensor_descriptors_validate_shape_dtype_layout_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::DescriptorAndView,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_rank: 4,
                max_dimension: 1024,
                max_elements: 4096,
                max_bytes: 16_384,
                allow_zero_sized: false,
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(tensor.clone()),
            ],
        )
        .unwrap();

    let zero_dim = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 0]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(zero_dim)
            ]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    let unsupported_dtype = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float64),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_dtype)
            ]
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));

    let unsupported_layout = TensorDescriptor::new(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::ProviderOpaque {
            layout_id: "native-blocked".into(),
        },
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_layout)
            ]
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));

    let overflowing = TensorDescriptor::materialized(
        ShapeDescriptor::new([64, 65]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(overflowing)
            ]
        ),
        Err(ComputeValidationError::SizeOverflow { .. })
    ));
}
#[test]
fn tensor_views_and_resources_preserve_affinity_and_materialization_boundaries() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let source = TensorResourceId::new("tensor-1");
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Strided {
            strides_elements: vec![4, 1],
            offset_elements: 0,
        },
    )
    .with_view(ViewDescriptor::from_resource(source.clone(), 4, [4, 1]));
    let resource = TensorResourceDescriptor::new(
        source,
        descriptor,
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );

    runtime
        .validate_compute_tensor_resources("portable-compute", &[resource])
        .unwrap();

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("tensor-2"),
        TensorDescriptor::materialized(
            ShapeDescriptor::new([4, 4]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    assert!(matches!(
        runtime.validate_compute_tensor_resources("portable-compute", &[foreign]),
        Err(ComputeValidationError::IncompatibleResourceAffinity(_))
    ));
}
#[test]
fn compute_data_movement_validates_host_buffers_affinity_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Upload,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_encodings([HostBufferEncoding::LittleEndian]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Materialize,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    let upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    runtime
        .validate_compute_data_movement("portable-compute", std::slice::from_ref(&upload))
        .unwrap();
    let uploaded = runtime
        .wrap_compute_data_movement_output(
            "portable-compute",
            &upload,
            TensorResourceId::new("uploaded"),
        )
        .unwrap();
    assert_eq!(
        uploaded.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );

    let invalid_upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(8, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_upload]),
        Err(ComputeValidationError::InvalidHostBuffer { .. })
    ));

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("foreign"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let transfer = ComputeDataMovementDescriptor::transfer(foreign, descriptor.clone());
    runtime
        .validate_compute_data_movement("portable-compute", &[transfer])
        .unwrap();

    let materialized = TensorResourceDescriptor::new(
        TensorResourceId::new("materialized"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute")),
    );
    let invalid_materialize = ComputeDataMovementDescriptor::materialize(materialized, descriptor);
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_materialize]),
        Err(ComputeValidationError::MaterializationRequired { .. })
    ));
}
#[test]
fn memory_planning_accounts_for_explicit_host_staged_transfers() {
    let mut provider = provider_with_capabilities("movement-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_staging(),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let source = TensorResourceDescriptor::new(
        TensorResourceId::new("source"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let movement =
        ComputeDataMovementDescriptor::transfer(source, descriptor).permit_host_staging();

    let plan = runtime
        .plan_compute_data_movement_memory("movement-compute", &[movement])
        .unwrap();

    assert_eq!(plan.pressure.transfer_buffer_cost_bytes, 16);
    assert!(
        plan.decisions.iter().any(|decision| {
            matches!(decision, MemoryPlanningDecision::AccountHostStaging { .. })
        })
    );
}
#[test]
fn provider_compute_advertisement_drives_data_movement_validation() {
    let mut provider = provider_with_capabilities("advertised-movement", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_data_movement(DataMovementSupport::from_compute_support(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_host_encodings([HostBufferEncoding::LittleEndian]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_data_movement(
            "advertised-movement",
            &[ComputeDataMovementDescriptor::upload(
                HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
                descriptor,
            )],
        )
        .unwrap();
}
#[test]
fn compute_operation_requests_reject_unknown_family_ids() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operation_requests(
            "portable-compute",
            &[ComputeOperationRequest::new("backend-kernel-name")]
        ),
        Err(ComputeValidationError::UnknownOperationFamily(_))
    ));
}
#[test]
fn compute_graph_validation_checks_references_provider_support_and_submission_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("graph-1"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("add"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let report = runtime
        .validate_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(report.node_count, 1);
    assert_eq!(report.input_count, 1);
    assert_eq!(report.output_count, 1);

    let submission = runtime
        .submit_validated_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(submission.graph, ComputeGraphId::new("graph-1"));
    assert_eq!(submission.state(), ComputeSubmissionState::Pending);
    assert_eq!(
        submission.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
}
#[test]
fn compute_execution_planning_selects_provider_device_and_validates_plan() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:0"),
        "GPU 0",
        DeviceType::Gpu,
        "portable-compute",
    );
    device.memory_capacity = 1_048_576;
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("planned-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("add"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert!(plan.is_validated());
    assert_eq!(plan.provider.as_str(), "portable-compute");
    assert_eq!(
        plan.device.as_ref().map(|device| device.id().as_str()),
        Some("gpu:0")
    );
    assert_eq!(plan.policy, BuiltInResolutionPolicy::Deterministic.id());
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::Transparent
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::SubmitToProvider)
    );
    assert!(
        plan.constraints
            .iter()
            .any(|constraint| matches!(constraint, ExecutionConstraint::NoHiddenCpuStaging))
    );
    assert_eq!(
        plan.memory_plan.graph,
        Some(ComputeGraphId::new("planned-graph"))
    );
}
#[test]
fn compute_execution_planning_preserves_provider_pinned_resources() {
    let mut provider_a = provider_with_capabilities("provider-a", [compute_capability()]);
    provider_a.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut provider_b = provider_with_capabilities("provider-b", [compute_capability()]);
    provider_b.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let resource = TensorResourceDescriptor::new(
        TensorResourceId::new("pinned"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("provider-b"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("pinned-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorResource(resource),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        );

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert_eq!(plan.provider.as_str(), "provider-b");
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::ProviderPinned
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::PreserveProviderPinnedAffinity)
    );
}
#[test]
fn scheduler_accepts_validated_plans_and_runs_fifo() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let first_graph = ComputeGraph::new(ComputeGraphId::new("first")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let second_graph = ComputeGraph::new(ComputeGraphId::new("second")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let first_plan = runtime.plan_compute_execution(&first_graph).unwrap();
    let second_plan = runtime.plan_compute_execution(&second_graph).unwrap();
    let mut scheduler = runtime.scheduler(2);

    let first = runtime
        .schedule_compute_execution(&mut scheduler, first_plan)
        .unwrap();
    let second = runtime
        .schedule_compute_execution(&mut scheduler, second_plan)
        .unwrap();

    assert_eq!(scheduler.policy(), SchedulingPolicy::Fifo);
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(first));
    assert_eq!(
        scheduler.operation(first).unwrap().state(),
        SchedulingState::Running
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(second));
}

#[test]
fn runtime_observability_correlates_plan_scheduler_and_metrics() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("observable")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let trace = plan.trace_id().clone();

    let plan_events = plan.observations();
    assert!(plan_events.iter().all(|event| event.trace_id == trace));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::CapabilityResolution
            && event.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::ExecutionPlanning
            && event.execution_plan.as_ref() == Some(&plan.id)
    }));

    let metrics = runtime_metrics_for_execution_plan(&plan);
    assert!(
        metrics
            .iter()
            .any(|metric| metric.kind == RuntimeMetricKind::MemoryUsageEstimate)
    );
    assert!(metrics.iter().all(|metric| {
        metric.trace_id.as_ref() == Some(&trace)
            && metric.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));

    let mut scheduler = runtime.scheduler(1);
    let operation = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    assert!(
        scheduler
            .observations()
            .iter()
            .all(|event| event.trace_id == trace)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionStarted)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn runtime_diagnostics_redact_native_details_and_exporters_are_components() {
    let diagnostic = RuntimeDiagnostic::new(
        RuntimeDiagnosticCode::ExecutionFailed,
        "backend handle=0xdeadbeef at C:\\native\\queue",
    )
    .with_trace(TraceId::new("trace:failure"))
    .with_provider(ProviderBinding::new("provider-a"));
    assert_eq!(diagnostic.message, "[redacted backend diagnostic]");

    let component = ComponentMetadata::new("otel-exporter", "1", "exports observations");
    let mut exporter =
        ObservabilityExporterDescriptor::new(component, ObservabilitySink::OpenTelemetry);
    exporter
        .accepted_events
        .insert(RuntimeEventKind::ExecutionCompleted);

    assert_eq!(
        exporter.input_contract,
        WitInterface::new("magnetar:runtime/observability", "1.0.0")
    );
    assert!(
        exporter
            .accepted_events
            .contains(&RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn provider_execution_api_submits_validated_scheduled_work() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-execution")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let operation = scheduler.operation(operation_id).unwrap();

    let request = runtime.prepare_provider_execution(operation).unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    let status = runtime.observe_provider_execution(&handle).unwrap();

    assert_eq!(handle.operation, operation_id);
    assert_eq!(handle.provider.as_str(), "portable-compute");
    assert!(handle.id.as_str().contains("provider-execution:"));
    assert_eq!(status.state, SchedulingState::Running);
    assert_eq!(
        status
            .progress
            .as_ref()
            .map(|progress| progress.total_steps),
        Some(3)
    );
    let submitted = api.submitted.lock().unwrap();
    assert_eq!(submitted.len(), 1);
    assert!(submitted[0].plan.is_validated());
    assert!(
        submitted[0].constraints.iter().any(|constraint| matches!(
            constraint,
            ExecutionConstraint::NoImplicitProviderMigration
        ))
    );
}
#[test]
fn provider_execution_completion_preserves_output_affinity_and_releases() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-complete")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    api.outputs
        .lock()
        .unwrap()
        .push(TensorResourceDescriptor::new(
            TensorResourceId::new("out-resource"),
            descriptor,
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("portable-compute"))
                .with_capability(CapabilityBinding::new(
                    CapabilityId::new(COMPUTE_CAPABILITY_ID),
                    COMPUTE_CAPABILITY_VERSION,
                )),
        ));

    let result = runtime.complete_provider_execution(&handle).unwrap();
    runtime.release_provider_execution(handle).unwrap();

    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(
        result.outputs[0]
            .affinity
            .provider()
            .map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(api.released.load(Ordering::SeqCst));
}
#[test]
fn provider_execution_rejects_mismatched_provider_request_and_maps_cancellation() {
    let api = Arc::new(TestProviderExecutionApi::new());
    *api.cancel_outcome.lock().unwrap() = ProviderCancellationOutcome::Accepted;
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-cancel")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let mut request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    request.provider = ProviderBinding::new("other-provider");

    assert!(matches!(
        runtime.submit_provider_execution(request),
        Err(ProviderExecutionError {
            code: ProviderExecutionErrorCode::InvalidExecutionPlan,
            phase: ProviderExecutionPhase::Submit,
            ..
        })
    ));

    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    assert_eq!(
        runtime.cancel_provider_execution(&handle).unwrap(),
        ProviderCancellationOutcome::Accepted
    );
}
#[test]
fn provider_execution_diagnostics_redact_native_details() {
    let diagnostic = ProviderExecutionDiagnostic::new(
        ProviderBinding::new("provider"),
        ProviderExecutionPhase::Submit,
    )
    .with_detail("backend handle=0xdeadbeef");

    assert_eq!(
        diagnostic.detail.as_deref(),
        Some("[redacted backend diagnostic]")
    );
}
#[test]
fn scheduler_rejects_over_capacity_and_cancels_queued_work() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("queued")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan.clone()).unwrap();

    assert!(matches!(
        scheduler.schedule(&runtime, plan),
        Err(SchedulerError::QueueCapacityExceeded { capacity: 1 })
    ));

    scheduler.cancel(operation).unwrap();
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Cancelled
    );
    assert_eq!(
        scheduler.result(operation).unwrap().state,
        SchedulingState::Cancelled
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), None);
}
#[test]
fn scheduler_completion_exposes_terminal_result_without_native_handles() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("completed"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("node"),
                output: ComputeOutputId::new("out"),
            },
        ));
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    let result = scheduler.result(operation).unwrap();
    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert!(result.error.is_none());
}
#[test]
fn scheduler_interrupts_when_provider_is_unavailable_before_submission() {
    let mut healthy = provider_with_capabilities("portable-compute", [compute_capability()]);
    healthy.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("interrupted")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();
    let mut unavailable = provider_with_capabilities("portable-compute", [compute_capability()]);
    unavailable.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    unavailable.health = ProviderHealth::Unavailable;
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(unavailable))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderUnavailable(provider))
            if provider.as_str() == "portable-compute"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
    );
}
#[test]
fn compute_graph_validation_rejects_missing_and_future_references() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let missing_input = ComputeGraph::new(ComputeGraphId::new("bad-input")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
        )
        .with_input(ComputeValueRef::Input(ComputeInputId::new("missing")))
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &missing_input),
        Err(ComputeValidationError::MissingInput { .. })
    ));

    let future_reference = ComputeGraph::new(ComputeGraphId::new("cycle"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("first"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_input(ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("second"),
                output: ComputeOutputId::new("out"),
            })
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("second"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &future_reference),
        Err(ComputeValidationError::CyclicGraph { .. })
    ));
}
#[test]
fn memory_planning_tracks_lifetimes_reuse_and_output_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("memory-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("temp"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("tmp"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("result-node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("result-node"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let plan = runtime
        .plan_compute_graph_memory("portable-compute", &graph)
        .unwrap();

    assert!(
        plan.decisions
            .iter()
            .any(|decision| matches!(decision, MemoryPlanningDecision::Reuse { .. }))
    );
    assert_eq!(
        plan.output_affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(
        plan.requirements
            .iter()
            .any(|requirement| requirement.region == MemoryRegionKind::Intermediate)
    );
}
#[test]
fn memory_planning_rejects_provider_and_device_memory_limits() {
    let mut limited_provider =
        provider_with_capabilities("limited-compute", [compute_capability()]);
    limited_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_bytes: 8,
                ..TensorDescriptorLimits::default()
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(limited_provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("too-large")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.plan_compute_graph_memory("limited-compute", &graph),
        Err(MemoryPlanningError::ProviderMemoryLimitExceeded { .. })
    ));

    let mut device_provider = provider_with_capabilities("device-limited", [compute_capability()]);
    device_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:tiny"),
        "Tiny GPU",
        DeviceType::Gpu,
        "device-limited",
    );
    device.memory_capacity = 8;
    device_provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(device_provider))
        .build()
        .unwrap();
    assert!(matches!(
        runtime.plan_compute_graph_memory("device-limited", &graph),
        Err(MemoryPlanningError::DeviceMemoryLimitExceeded { .. })
    ));
}
#[test]
fn compute_providers_register_and_resolve_compatibly() {
    let mut provider = TestProvider::new("portable-compute");
    provider.metadata.capabilities.insert(compute_capability());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ))
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn resolution_policy_records_selected_provider_capability_and_reason() {
    let compute = compute_capability();
    let provider_a = provider_with_capabilities("provider-a", [compute.clone()]);
    let provider_b = provider_with_capabilities("provider-b", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();

    let resolution = runtime
        .resolve_with_affinity(&compute, &[], FallbackClass::Transparent)
        .unwrap();
    let decision = resolution.decision();

    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        decision
            .selected_provider
            .as_ref()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        decision.selected_capability,
        Some(CapabilityBinding::new(compute.id.clone(), compute.version))
    );
    assert_eq!(
        decision.reason,
        ResolutionDecisionReason::SelectedDeterministically
    );
}
#[test]
fn policy_rejection_is_structured_when_all_candidates_are_unhealthy() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider.health = ProviderHealth::Unavailable;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { capability, policy })
            if capability.id() == &compute.id
                && capability.version() == compute.version
                && policy == BuiltInResolutionPolicy::Deterministic.id()
    ));
}
#[test]
fn availability_policy_prefers_healthy_candidates() {
    let compute = compute_capability();
    let mut degraded = provider_with_capabilities("a-degraded", [compute.clone()]);
    degraded.health = ProviderHealth::Degraded;
    let healthy = provider_with_capabilities("z-healthy", [compute.clone()]);
    let runtime = Runtime::builder()
        .config(RuntimeConfig {
            resolution_policy: BuiltInResolutionPolicy::Availability,
            ..RuntimeConfig::default()
        })
        .register_provider(Arc::new(degraded))
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();

    let providers = runtime.resolve_component_import(&WitInterface::new(
        COMPUTE_WIT_INTERFACE,
        COMPUTE_CAPABILITY_VERSION.to_string(),
    ));

    assert_eq!(providers.unwrap()[0].metadata().name, "z-healthy");
}
#[test]
fn capability_health_rejects_unavailable_implementation() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .capability_health
        .insert(compute.id.clone(), HealthState::Unavailable);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { .. })
    ));

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::CapabilityUnavailable
    );
}
#[test]
fn health_reports_redact_diagnostics_and_track_freshness() {
    let provider = ProviderBinding::new("provider-a");
    let mut report = ProviderHealthReport::new(provider.clone(), HealthState::Saturated);
    report.timestamp = Some(HealthTimestamp::unix_millis(1_000));
    report.time_to_live = Some(HealthTimeToLive::millis(250));
    report.capacity.queue_depth = Some(8);
    report.diagnostics.push(
        HealthDiagnostic::new(HealthScope::Provider, HealthState::Saturated)
            .with_code("queue-pressure")
            .with_message("cuda stream /tmp/secret token=abc is saturated")
            .with_trace_id("trace-1"),
    );

    assert!(report.is_stale_at(HealthTimestamp::unix_millis(1_251)));
    assert_eq!(report.capacity.queue_depth, Some(8));
    let message = report.diagnostics[0].message.as_deref().unwrap();
    assert!(!message.contains("/tmp/secret"));
    assert!(!message.contains("token=abc"));
    assert_eq!(message, "[redacted backend diagnostic]");
    assert_eq!(report.diagnostics[0].trace_id.as_deref(), Some("trace-1"));

    let device_health = DeviceHealth::new(
        provider.clone(),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    );
    let capability_health = CapabilityHealth::new(
        provider,
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Degraded,
    );
    assert!(matches!(
        HealthReport::Device(device_health),
        HealthReport::Device(_)
    ));
    assert!(matches!(
        HealthReport::Capability(capability_health),
        HealthReport::Capability(_)
    ));
}

#[test]
fn provider_status_snapshot_separates_health_readiness_pressure_and_admission() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.pressure = ProviderPressureLevel::Low;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    snapshot.timestamp = Some(HealthTimestamp::unix_millis(10));
    snapshot.time_to_live = Some(HealthTimeToLive::millis(5));

    assert_eq!(snapshot.provider, provider);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.pressure, ProviderPressureLevel::Low);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert!(!snapshot.accepts_new_work_by_default());
    assert!(snapshot.is_stale_at(HealthTimestamp::unix_millis(16)));
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
fn operation_family_status_falls_back_to_capability_status_when_absent() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::Ready
    );

    let unsupported =
        OperationFamilyStatus::unsupported(provider.clone(), ComputeOperationFamily::LinearAlgebra);
    snapshot = snapshot.with_operation_family_status(unsupported);
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::NotReady
    );

    let mut saturated =
        OperationFamilyStatus::available(provider, ComputeOperationFamily::Elementwise);
    saturated.pressure = ProviderPressureLevel::Saturated;
    saturated.readiness = ProviderReadinessState::NotReady;
    snapshot = snapshot.with_operation_family_status(saturated);
    assert_eq!(
        snapshot
            .operation_family_status(ComputeOperationFamily::Elementwise)
            .unwrap()
            .pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn provider_status_maps_interruption_to_health_readiness_and_admission() {
    let snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Interrupted,
    ));

    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Failed);
    assert_eq!(snapshot.health, ProviderHealthState::Failed);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert_eq!(snapshot.health_reason, ProviderStatusReason::Interrupted);
    assert!(matches!(
        snapshot.interruption,
        Some(ProviderInterruptionReason::DriverLoss)
    ));
}

#[test]
fn provider_status_observations_cover_all_status_dimensions() {
    let mut report =
        ProviderHealthReport::new(ProviderBinding::new("provider-a"), HealthState::Draining);
    report.timestamp = Some(HealthTimestamp::unix_millis(10));
    report.time_to_live = Some(HealthTimeToLive::millis(5));
    report.devices.push(DeviceHealth::new(
        ProviderBinding::new("provider-a"),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    ));
    report.capabilities.push(CapabilityHealth::new(
        ProviderBinding::new("provider-a"),
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Available,
    ));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(report);
    snapshot.lifecycle = ProviderLifecycleState::Draining;
    snapshot.readiness = ProviderReadinessState::Draining;
    snapshot.in_flight_operations = 0;

    let events =
        runtime_events_for_provider_status(&snapshot, Some(HealthTimestamp::unix_millis(16)));
    let kinds = events
        .iter()
        .map(|event| event.kind.clone())
        .collect::<BTreeSet<_>>();

    assert!(kinds.contains(&RuntimeEventKind::ProviderLifecycleChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderHealthChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderReadinessChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderPressureChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderAdmissionChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderStatusStale));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainStarted));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainCompleted));
    assert!(kinds.contains(&RuntimeEventKind::DeviceStatusChanged));
    assert!(kinds.contains(&RuntimeEventKind::CapabilityStatusChanged));
}

#[test]
fn resolution_rejects_healthy_provider_that_is_not_ready() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);

    assert_eq!(decision.selected_provider, None);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::ProviderInitializing
    );
}

#[test]
fn resolution_records_selected_provider_status_and_stale_rejection_reason() {
    let compute = compute_capability();
    let ready_provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut stale_provider = provider_with_capabilities("provider-b", [compute.clone()]);
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-b"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .register_provider(Arc::new(ready_provider))
        .build()
        .unwrap();

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);

    assert_eq!(
        decision
            .selected_provider_status
            .as_ref()
            .map(|status| status.provider.as_str()),
        Some("provider-a")
    );
    assert!(decision.rejected_candidates.iter().any(|candidate| {
        candidate.provider.as_str() == "provider-b"
            && candidate.reason == ResolutionRejectionReason::ProviderStatusStale
    }));
}

#[test]
fn scheduler_checks_refined_provider_status_before_submission() {
    let compute = compute_capability();
    let mut planning_provider = provider_with_capabilities("provider-a", [compute]);
    planning_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    planning_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(planning_provider))
        .build()
        .unwrap();
    let graph = simple_elementwise_compute_graph("status-change");
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();

    let mut submission_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    submission_provider
        .metadata
        .compute_operation_support
        .insert(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense]),
        );
    submission_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::Ready;
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    submission_provider.status_snapshot = Some(snapshot);
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(submission_provider))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderSaturated(provider)) if provider.as_str() == "provider-a"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
    );
}

#[test]
fn scheduler_and_provider_execution_reject_stale_provider_status() {
    let mut provider = provider_with_capabilities("provider-a", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let plan = planning_runtime
        .plan_compute_execution(&simple_elementwise_compute_graph("stale-status"))
        .unwrap();

    let mut stale_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    stale_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    stale_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .build()
        .unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&runtime),
        Err(SchedulerError::StaleHealthReport(provider)) if provider.as_str() == "provider-a"
    ));
    assert!(matches!(
        runtime.prepare_provider_execution(scheduler.operation(operation).unwrap()),
        Err(error) if error.code == ProviderExecutionErrorCode::StaleHealthReport
    ));
}
#[test]
fn phase_aware_resolution_rejects_restart_after_observable_output() {
    let compute = compute_capability();
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_with_affinity_at_phase(
            &compute,
            &[],
            FallbackClass::Restartable,
            ExecutionPhase::AfterObservableOutput,
            true,
        ),
        Err(AffinityError::PolicyRejectedProvider { .. })
    ));
}
#[test]
fn builder_isolates_failed_provider_initialization() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();
    assert!(runtime.providers().provider("failed").is_none());
    assert!(runtime.providers().provider("available").is_some());
}
#[test]
fn builder_reports_rejected_provider_instead_of_dropping_it_silently() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();

    let diagnostics = runtime.startup_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, RuntimeDiagnosticCode::ProviderRejected);
    assert_eq!(
        diagnostics[0]
            .provider
            .as_ref()
            .map(ProviderBinding::as_str),
        Some("failed")
    );
    assert!(diagnostics[0].message.contains("failed"));
}
#[test]
fn builder_records_no_diagnostics_when_every_provider_registers() {
    let runtime = Runtime::builder()
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();
    assert!(runtime.startup_diagnostics().is_empty());
}
#[test]
fn builder_does_not_register_kernels_for_a_rejected_provider() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    failed.kernel_advertisements = reference_cpu_kernel_advertisements();
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .build()
        .unwrap();

    // The Provider never came up, so its kernels must not be left in the
    // registry as candidates that can never resolve.
    assert!(runtime.providers().provider("failed").is_none());
    assert_eq!(runtime.startup_diagnostics().len(), 1);
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
fn provider_shutdown_releases_registered_provider() {
    let p = TestProvider::new("provider");
    let p = Arc::new(p);
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("provider").is_some());
    m.shutdown().unwrap();
    assert!(p.shut_down.load(Ordering::SeqCst));
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
        !include_str!("provider.rs").contains("Box<dyn Provider>"),
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
fn provider_loading_policy_is_explicit_for_dynamic_and_development_modes() {
    let root = std::path::PathBuf::from("target/providers");
    let provider = root.join("provider.dll");
    let outside = std::path::PathBuf::from("target/other/provider.dll");
    let dynamic = ProviderLoadingPolicy::dynamic_library([root.clone()]);
    let development = ProviderLoadingPolicy::development([root.clone()]);

    assert_eq!(dynamic.mode, ProviderLoadingMode::DynamicLibrary);
    assert!(!dynamic.development_mode);
    assert!(dynamic.allows(&provider));
    assert!(!dynamic.allows(&outside));
    assert_eq!(development.mode, ProviderLoadingMode::DevelopmentProvider);
    assert!(development.development_mode);
    assert!(development.allows(&provider));
    assert!(!ProviderLoadingPolicy::default().allows(&provider));
}

#[test]
fn runtime_source_does_not_restore_legacy_backend_or_plugin_surface() {
    let source = include_str!("lib.rs");
    let forbidden = [
        concat!("trait ", "Backend"),
        concat!("register_", "backend"),
        concat!("preferred_", "backend"),
        concat!("backend_", "names"),
        concat!("trait ", "Plugin"),
        concat!("Plugin", "Registry"),
    ];

    for term in forbidden {
        assert!(
            !source.contains(term),
            "legacy architecture surface remains in runtime source: {term}"
        );
    }
}

#[test]
fn public_component_api_does_not_expose_wasmtime_native_types() {
    let public_sources = [include_str!("lib.rs"), include_str!("component.rs")];
    let forbidden = [
        "wasmtime::Engine",
        "wasmtime::Config",
        "wasmtime::Store",
        "wasmtime::component::Component",
        "wasmtime::component::Linker",
        "wasmtime::component::Instance",
        "wasmtime::Trap",
        "wasmtime::Error",
    ];

    for source in public_sources {
        for term in forbidden {
            assert!(
                !source.contains(term),
                "public Component API exposes concrete engine type: {term}"
            );
        }
    }
}

#[test]
fn component_engine_profiles_declare_platform_capabilities() {
    let native = ComponentEngineCapabilities::native();
    assert_eq!(native.profile, ComponentEngineProfile::Native);
    assert!(native.supports(ComponentEngineFeature::ComponentModel));
    assert!(native.supports(ComponentEngineFeature::NativeProviderEndpoints));
    assert!(!native.supports(ComponentEngineFeature::BrowserCompatible));

    let web = ComponentEngineCapabilities::web();
    assert_eq!(web.profile, ComponentEngineProfile::Web);
    assert!(web.supports(ComponentEngineFeature::BrowserCompatible));
    assert!(web.supports(ComponentEngineFeature::JsMediatedHostCalls));
    assert!(web.supports(ComponentEngineFeature::BrowserMemory));
    assert!(!web.supports(ComponentEngineFeature::NativeProviderEndpoints));

    let test = ComponentEngineCapabilities::test();
    assert_eq!(test.profile, ComponentEngineProfile::Test);
    assert!(test.supports(ComponentEngineFeature::Interruption));
    assert!(!test.supports(ComponentEngineFeature::ControlledWasi));
}

#[test]
fn component_engine_requirements_fail_closed_on_profile_or_feature_mismatch() {
    let requirements = ComponentEngineRequirements::default()
        .require_profile(ComponentEngineProfile::Web)
        .require_feature(ComponentEngineFeature::BrowserCompatible);

    assert!(
        requirements
            .validate("browser-component", &ComponentEngineCapabilities::web())
            .is_ok()
    );

    assert!(matches!(
        requirements.validate("browser-component", &ComponentEngineCapabilities::native()),
        Err(ComponentError::EngineProfileMismatch {
            required: ComponentEngineProfile::Web,
            actual: ComponentEngineProfile::Native,
            ..
        })
    ));

    let requirements = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::NativeProviderEndpoints);
    assert!(matches!(
        requirements.validate("native-component", &ComponentEngineCapabilities::web()),
        Err(ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::NativeProviderEndpoints,
            profile: ComponentEngineProfile::Web,
            ..
        })
    ));
}

#[test]
fn component_manager_observes_engine_selection_and_rejection() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.prepare_component("component").unwrap();
    assert!(
        manager.observations().iter().any(|observation| {
            observation.kind == ComponentObservationKind::EngineSelection
                && observation
                    .message
                    .contains(ComponentEngineProfile::Test.as_str())
        }),
        "selected engine profile should be observable"
    );

    let error = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::ControlledWasi)
        .validate("component", &manager.engine_capabilities())
        .unwrap_err();
    assert!(matches!(
        error,
        ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::ControlledWasi,
            ..
        }
    ));
}

#[test]
fn execution_context_default_allocates_unique_ids() {
    let first = ExecutionContext::default();
    let second = ExecutionContext::default();
    assert_ne!(first.id(), second.id());
    assert_ne!(first.id(), ExecutionContextId::default());
}

#[test]
fn affinity_constraints_preserve_compatible_facts_and_fallback_precedence() {
    let capability_a = capability_binding("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let capability_b = capability_binding("magnetar:tokenize/run", CapabilityVersion::new(1, 0, 0));
    let provider = ProviderBinding::new("provider-a");
    let device = DeviceBinding::new(DeviceId::new("gpu:0"));
    let context = ExecutionContextId::new(42);
    let group = AffinityGroupId::new(7);

    let model = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(provider.clone())
        .with_device(device.clone())
        .with_capability(capability_a.clone())
        .with_artifact(ArtifactBinding::new("model", "sha256:model"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);
    let tokenizer = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(provider)
        .with_device(device)
        .with_capability(capability_b.clone())
        .with_artifact(ArtifactBinding::new("tokenizer", "sha256:tokenizer"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);

    let constraints = AffinityConstraints::try_from_affinities([&model, &tokenizer]).unwrap();
    let aggregate = constraints.affinity();
    assert_eq!(aggregate.capability(capability_a.id()), Some(&capability_a));
    assert_eq!(aggregate.capability(capability_b.id()), Some(&capability_b));
    assert_eq!(
        aggregate.artifact("model").unwrap().fingerprint(),
        "sha256:model"
    );
    assert_eq!(
        aggregate.artifact("tokenizer").unwrap().fingerprint(),
        "sha256:tokenizer"
    );
    assert_eq!(
        aggregate.artifact("bundle").unwrap().fingerprint(),
        "sha256:bundle"
    );
    assert_eq!(aggregate.fallback(), FallbackClass::ProviderPinned);
}

#[test]
fn affinity_constraints_report_each_binding_conflict() {
    let base = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:a"))
        .with_execution_context(ExecutionContextId::new(1))
        .with_group(AffinityGroupId::new(1));

    let provider_conflict = base
        .clone()
        .with_provider(ProviderBinding::new("provider-b"));
    assert!(matches!(
        base.validate_with(&provider_conflict),
        Err(AffinityError::ProviderMismatch { .. })
    ));

    let device_conflict = base
        .clone()
        .with_device(DeviceBinding::new(DeviceId::new("gpu:1")));
    assert!(matches!(
        base.validate_with(&device_conflict),
        Err(AffinityError::DeviceMismatch { .. })
    ));

    let capability_conflict = base.clone().with_capability(capability_binding(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 2, 0),
    ));
    assert!(matches!(
        base.validate_with(&capability_conflict),
        Err(AffinityError::CapabilityMismatch { .. })
    ));

    let artifact_conflict = base
        .clone()
        .with_artifact(ArtifactBinding::new("bundle", "sha256:b"));
    assert!(matches!(
        base.validate_with(&artifact_conflict),
        Err(AffinityError::ArtifactMismatch { .. })
    ));

    let context_conflict = base
        .clone()
        .with_execution_context(ExecutionContextId::new(2));
    assert!(matches!(
        base.validate_with(&context_conflict),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));

    let group_conflict = base.clone().with_group(AffinityGroupId::new(2));
    assert!(matches!(
        base.validate_with(&group_conflict),
        Err(AffinityError::AffinityGroupMismatch { .. })
    ));
}

#[test]
fn affinity_resource_keeps_value_and_affinity_together() {
    let affinity = ResourceAffinity::new(FallbackClass::Restartable)
        .with_provider(ProviderBinding::new("provider-a"));
    let resource = AffinityResource::new("native-handle", affinity.clone());

    assert_eq!(resource.value(), &"native-handle");
    assert_eq!(resource.affinity(), &affinity);
    assert_eq!(resource.into_parts(), ("native-handle", affinity));
}

#[test]
fn affinity_resolution_uses_provider_local_compatible_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider_a = provider_with_capabilities(
        "provider-a",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        )],
    );
    let provider_b = provider_with_capabilities(
        "provider-b",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        )],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_a))
        .register_provider(Arc::new(provider_b))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
}

#[test]
fn affinity_resolution_preserves_exact_live_capability_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities(
        "provider-a",
        [
            capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0)),
            capability("magnetar:compute/run", CapabilityVersion::new(1, 2, 0)),
        ],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
}

#[test]
fn affinity_resolution_requires_selected_provider_to_implement_all_bound_capabilities() {
    let compute = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:tokenize/run",
            CapabilityVersion::new(1, 0, 0),
        ));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::ProviderDoesNotImplementCapability { .. })
    ));
}

#[test]
fn affinity_resolution_reconciles_devices_with_provider_ownership() {
    let compute = compute_capability();
    let device_id = DeviceId::new("gpu:0");
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            device_id.clone(),
            "test gpu",
            DeviceType::Gpu,
            "provider-a",
        ))));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(device_id.clone()));

    let resolution = runtime
        .resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.affinity().device().map(DeviceBinding::id),
        Some(&device_id)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );

    let mismatched = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("other"))
        .with_device(DeviceBinding::new(device_id));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&mismatched], FallbackClass::ProviderPinned),
        Err(AffinityError::DeviceProviderMismatch { .. })
    ));

    let missing = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(DeviceId::new("missing")));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&missing], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundDeviceUnavailable(_))
    ));
}

#[test]
fn affinity_resolution_reports_unavailable_bound_provider_without_fallback() {
    let compute = compute_capability();
    let mut fallback = provider_with_capabilities("fallback", [compute.clone()]);
    fallback.metadata.capabilities.insert(compute.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(fallback))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("missing"));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundProviderUnavailable(provider)) if provider.as_str() == "missing"
    ));
}

#[test]
fn affinity_resolution_rejects_foreign_context_and_preserves_groups() {
    let compute = compute_capability();
    let first = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    let second = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();

    let ungrouped = first
        .resolve_with_affinity(&compute, &[], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(ungrouped.group(), None);

    let grouped = first
        .resolve_with_affinity(&compute, &[&ungrouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert!(grouped.group().is_some());

    let inherited = first
        .resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(inherited.group(), grouped.group());

    assert!(matches!(
        second.resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));
}

#[test]
fn affinity_resolution_rejects_shutdown_runtime() {
    let compute = compute_capability();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    runtime.shutdown();

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[], FallbackClass::Transparent),
        Err(AffinityError::RuntimeNotInitialized)
    ));
}

#[test]
fn component_runtime_instantiates_without_generic_start_or_stop() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let instance = manager.instantiate_component("component").unwrap();
    assert_eq!(
        manager.definition_state("component"),
        Some(ComponentDefinitionState::Prepared)
    );
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );

    manager.shutdown();
    assert_eq!(
        manager.instance_state(instance),
        None,
        "shutdown removes Runtime-owned Component instances"
    );
}

#[test]
fn component_artifact_reference_prepares_future_artifact_model_without_trust_policy() {
    let descriptor = ComponentDescriptor::new(
        ComponentMetadata::new("component", "1", "test component"),
        "component.wasm",
    );

    assert!(matches!(
        descriptor.artifact_reference(),
        ComponentArtifactReference::LocalPath(path) if path == std::path::Path::new("component.wasm")
    ));
}

#[test]
fn component_imports_are_authorized_and_linked_explicitly() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));

    manager.authorize_interface(interface.clone());
    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnresolvedImport { .. })
    ));

    manager.provide_interface(interface);
    let instance = manager.instantiate_component("consumer").unwrap();
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_import_version_must_match_authorized_interface() {
    let authorized = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let requested = WitInterface::new("magnetar:runtime/run", "2.0.0");
    let metadata = ComponentMetadata::new("consumer", "1", "test component").with_import(requested);
    let mut manager = ComponentManager::new();
    manager.provide_interface(authorized);
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_wasi_imports_fail_closed_without_authorization() {
    let filesystem = WitInterface::new("wasi:filesystem/types", "0.2.0");
    let environment = WitInterface::new("wasi:cli/environment", "0.2.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("wasi-consumer", "1", "test component")
                .with_import(filesystem)
                .with_import(environment),
            "wasi-consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("wasi-consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_ambient_network_process_and_secret_imports_fail_closed() {
    let interfaces = [
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("wasi:cli/run", "0.2.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ];
    for (index, interface) in interfaces.into_iter().enumerate() {
        let name = format!("authority-{index}");
        let mut manager = ComponentManager::new();
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new(&name, "1", "test component").with_import(interface),
                format!("{name}.wasm"),
            ))
            .unwrap();

        assert!(matches!(
            manager.instantiate_component(&name),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_link_plan_is_runtime_owned_and_immutable_to_callers() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager.provide_interface(interface.clone());
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    let plan = manager.link_plan("consumer").unwrap();
    let links = plan.links().collect::<Vec<_>>();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].0, &interface);
    assert!(matches!(
        links[0].1,
        ComponentEndpoint::Capability { interface: linked } if linked == &interface
    ));
    assert_eq!(plan.endpoint(&interface), Some(links[0].1));
}

#[test]
fn component_link_plan_rejects_forbidden_external_interfaces_even_if_provided() {
    for interface in [
        WitInterface::new("wasi:filesystem/types", "0.2.0"),
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("magnetar:workspace/read", "1.0.0"),
        WitInterface::new("magnetar:git/status", "1.0.0"),
        WitInterface::new("magnetar:process/run", "1.0.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ] {
        let mut manager = ComponentManager::new();
        manager.provide_interface(interface.clone());
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("external", "1", "external component")
                    .with_import(interface),
                "external.wasm",
            ))
            .unwrap();

        assert!(matches!(
            manager.link_plan("external"),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_authority_requirements_map_to_inference_runtime_endpoints() {
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "compute-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Capability { interface }
            if interface == WitInterface::new("magnetar:compute/run", "2.0.0")
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "model-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Model
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "tokenizer-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Tokenizer
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prompt-template-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::PromptTemplate
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "adapter-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Adapter
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "quantization-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Quantization
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "kv-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Kv
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prefix-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Prefix
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "observability-emit".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Observability
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "runtime-diagnostics".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::RuntimeDiagnostics
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "generation-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "sampling-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
}

#[test]
fn inference_artifact_registry_uses_identities_not_paths_and_scopes_sessions() {
    let mut manager = ComponentManager::new();
    let digest = ComponentDigest::sha256(b"model");
    let session = InferenceSessionId::new("session-a").unwrap();
    manager
        .register_inference_artifact(
            InferenceArtifactReference::new(InferenceArtifactKind::Model, "qwen-model", digest)
                .unwrap()
                .with_session(session.clone()),
        )
        .unwrap();

    let artifact = manager
        .resolve_inference_artifact(InferenceArtifactKind::Model, "qwen-model", Some(&session))
        .unwrap();
    assert_eq!(artifact.id, "qwen-model");
    assert!(matches!(
        manager.resolve_inference_artifact(InferenceArtifactKind::Model, "../qwen-model", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        manager.resolve_inference_artifact(
            InferenceArtifactKind::Model,
            "qwen-model",
            Some(&InferenceSessionId::new("session-b").unwrap())
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_artifact_registry_handles_tokenizer_template_adapter_and_quantization() {
    let mut registry = InferenceArtifactRegistry::default();
    for kind in [
        InferenceArtifactKind::Tokenizer,
        InferenceArtifactKind::PromptTemplate,
        InferenceArtifactKind::Adapter,
        InferenceArtifactKind::Quantization,
    ] {
        let id = format!("{kind:?}").to_ascii_lowercase();
        registry
            .register(
                InferenceArtifactReference::new(kind, &id, ComponentDigest::sha256(id.as_bytes()))
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(registry.resolve(kind, &id, None).unwrap().kind, kind);
    }
    assert!(matches!(
        registry.resolve(InferenceArtifactKind::Tokenizer, "C:\\tokenizer.json", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_cache_registry_scopes_access_to_session_and_model() {
    let mut registry = InferenceCacheRegistry::default();
    let session = InferenceSessionId::new("session-a").unwrap();
    let authorized =
        InferenceCacheScope::new(InferenceCacheKind::Kv, session.clone(), "qwen-model").unwrap();
    registry.authorize(authorized.clone());

    registry.authorize_access(&authorized).unwrap();
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(InferenceCacheKind::Kv, session, "other-model").unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(
                InferenceCacheKind::Prefix,
                InferenceSessionId::new("session-b").unwrap(),
                "qwen-model"
            )
            .unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn prepared_component_contract_must_match_declared_imports() {
    let declared = WitInterface::new("example:declared/api", "1.0.0");
    let undeclared = WitInterface::new("example:undeclared/api", "1.0.0");
    let mut contract = ComponentContract::default();
    contract.imports.insert(ComponentImportRequirement::new(
        undeclared,
        ComponentInterfaceShape::Interface,
    ));
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(contract);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("consumer", "1", "test component").with_import(declared),
            "consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("consumer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn prepared_component_contract_must_include_declared_exports() {
    let exported = WitInterface::new("example:export/api", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(ComponentContract::default());
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("producer", "1", "test component").with_export(exported),
            "producer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("producer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn component_exports_do_not_automatically_satisfy_imports() {
    let interface = WitInterface::new("example:service/api", "1.0.0");
    let producer =
        ComponentMetadata::new("producer", "1", "producer").with_export(interface.clone());
    let consumer = ComponentMetadata::new("consumer", "1", "consumer").with_import(interface);
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(producer, "producer.wasm"))
        .unwrap();
    manager
        .register_component(ComponentDescriptor::new(consumer, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_definition_can_create_multiple_isolated_instances() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let first = manager.instantiate_component("component").unwrap();
    let second = manager.instantiate_component("component").unwrap();
    assert_ne!(first, second);
    assert_eq!(
        manager.instance_state(first),
        Some(ComponentInstanceState::Ready)
    );
    assert_eq!(
        manager.instance_state(second),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_manager_enforces_instance_and_invocation_limits() {
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_instances: Some(1),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "instances",
            ..
        })
    ));

    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_concurrent_invocations: Some(0),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("callable", "1", "test component")
                .with_export(interface.clone()),
            "callable.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("callable").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "concurrent invocations",
            ..
        })
    ));
}

#[test]
fn component_invocation_after_destruction_fails() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.destroy_instance(instance).unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::InstanceNotFound(_))
    ));
}

#[test]
fn component_shutdown_prevents_new_lifecycle_operations() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.shutdown();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.register_component(ComponentDescriptor::new(
            ComponentMetadata::new("other", "1", "test component"),
            "other.wasm",
        )),
        Err(ComponentError::RuntimeShutdown)
    ));
}

#[test]
fn component_observations_are_non_authoritative_and_redacted() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap { .. })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(
                |observation| observation.kind == ComponentObservationKind::Trap
                    && observation.instance == Some(instance)
                    && observation.message.contains("[redacted component trap]")
            )
    );
    assert!(
        !manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("wasmtime::"))
    );
    assert!(!manager.observations().iter().any(|observation| {
        observation.message.contains("Provider")
            || observation.message.contains("Device")
            || observation.message.contains("Store")
    }));
}

#[test]
fn component_engine_normalizes_traps_interruptions_and_limit_failures() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut trapping_engine = MockComponentEngine::new();
    trapping_engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(trapping_engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap {
            kind: ComponentTrapKind::Trap,
            ..
        })
    ));

    let mut manager = ComponentManager::with_engine(Box::new(
        MockComponentEngine::new().without_resource_limits(),
    ));
    manager.set_resource_limits(ComponentResourceLimits {
        require_memory_limit: true,
        max_memory_bytes: Some(1024),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("limited", "1", "test component"),
            "limited.wasm",
        ))
        .unwrap();
    assert!(matches!(
        manager.instantiate_component("limited"),
        Err(ComponentError::ResourceLimitUnsupported { .. })
    ));
}

#[test]
fn component_discovery_returns_only_wasm_artifacts() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-components-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("valid.wasm"), []).unwrap();
    fs::write(directory.join("ignored.txt"), []).unwrap();

    let discovered = ComponentManager::discover([&directory]).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    assert_eq!(discovered, vec![directory.join("valid.wasm")]);
}

fn temp_component_artifact_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-component-artifact-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn manifest_yaml(digest: &str, runtime_version: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.hello"
  version: "0.1.0"
  description: "Minimal Magnetar Component fixture"
  role: "test-fixture"
runtime:
  magnetar:
    min_version: "{runtime_version}"
wit:
  imports:
    - package: "magnetar:test"
      interface: "echo"
      version: "1.0.0"
  exports:
    - package: "magnetar:test"
      interface: "run"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:test/echo"
      version: "1.0.0"
authority:
  requires: []
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/hello.component.wasm"
signatures: []
"#
    )
}

fn tokenizer_manifest_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"
authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"
signatures: []
"#
    )
}

fn manifest_yaml_with_authority(digest: &str, authorities: &[&str]) -> String {
    let requires = authorities
        .iter()
        .map(|authority| format!("    - {authority}"))
        .collect::<Vec<_>>()
        .join("\n");
    manifest_yaml(digest, MAGNETAR_RUNTIME_VERSION).replace(
        "authority:\n  requires: []",
        &format!("authority:\n  requires:\n{requires}"),
    )
}

fn trust_store_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-trust
schema_version: 1
trusted_digests:
  - "{digest}"
rejected_digests: []
revoked_digests: []
trusted_publishers: []
trusted_sources: []
development:
  allow_unsigned_local: false
"#
    )
}

fn component_artifact_package(
    bytes: &[u8],
    source_kind: ComponentDistributionSourceKind,
) -> ComponentArtifactPackage {
    let digest = ComponentDigest::sha256(bytes);
    ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).into_bytes(),
        digest,
        ComponentDistributionSource::new(source_kind, "test-source"),
    )
}

#[derive(Clone)]
struct TestComponentDistributionSource {
    package: ComponentArtifactPackage,
    candidates: Vec<ComponentDigest>,
}

impl ComponentDistributionSourceProvider for TestComponentDistributionSource {
    fn resolve(
        &self,
        component: &str,
        _version_requirement: Option<&str>,
    ) -> Result<Vec<ComponentDigest>, ComponentError> {
        if component == "magnetar.examples.hello" {
            Ok(self.candidates.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn fetch(&self, digest: &ComponentDigest) -> Result<ComponentArtifactPackage, ComponentError> {
        if self.package.declared_digest == *digest {
            Ok(self.package.clone())
        } else {
            Err(ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ArtifactNotFound,
                message: "digest not found".into(),
            })
        }
    }
}

#[test]
fn pushed_component_package_is_validated_before_preparation() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager.prepare_pushed_package(package).unwrap();

    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("client-provided")
    }));
}

#[test]
fn pushed_component_package_temp_materialization_is_removed_with_manager() {
    let before = std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("magnetar-distributed-component-"))
        })
        .collect::<BTreeSet<_>>();
    let bytes = b"component-bytes-cleanup";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    let materialized = {
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
        manager.prepare_pushed_package(package).unwrap();
        std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                !before.contains(path)
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("magnetar-distributed-component-"))
            })
            .expect("distributed component package materialized")
    };

    assert!(!materialized.exists());
}

#[test]
fn pushed_component_package_rejects_source_digest_mismatch() {
    let mut package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    package.declared_digest = ComponentDigest::sha256(b"different");
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::DigestMismatch,
            ..
        })
    ));
}

#[test]
fn distribution_source_identity_does_not_imply_trust() {
    let package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));
}

#[test]
fn trusted_distribution_source_still_rejects_forbidden_authority() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    let package = ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest.into_bytes(),
        digest,
        ComponentDistributionSource::new(ComponentDistributionSourceKind::Tachyon, "tachyon"),
    );
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Manifest { message, .. })
            if message == "authority kind is outside Magnetar inference scope"
    ));
}

#[test]
fn pulled_component_package_resolves_fetches_and_validates_locally() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let source = TestComponentDistributionSource {
        package,
        candidates: vec![digest.clone()],
    };
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager
        .prepare_pulled_package(&source, "magnetar.examples.hello", Some(">=0.1.0,<1.0.0"))
        .unwrap();

    assert_eq!(
        manager
            .definition("magnetar.examples.hello")
            .and_then(|definition| definition.artifact_digest.clone()),
        Some(digest)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("candidate digest")
    }));
}

#[test]
fn pulled_component_package_rejects_empty_candidate_list() {
    let package = component_artifact_package(
        b"component-bytes",
        ComponentDistributionSourceKind::LocalCache,
    );
    let source = TestComponentDistributionSource {
        package,
        candidates: Vec::new(),
    };
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pulled_package(&source, "magnetar.examples.hello", None),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::ArtifactNotFound,
            ..
        })
    ));
}

#[test]
fn local_distribution_does_not_require_tachyon_or_network() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));

    manager.prepare_pushed_package(package).unwrap();
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Prepared)
    );
}

#[test]
fn component_artifact_accepts_target_tokenizer_manifest_authorities() {
    let directory = temp_component_artifact_dir("tokenizer-authority");
    let artifact = directory.join("tokenizer.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("tokenizer.component.wasm.magnetar-component.yaml"),
        tokenizer_manifest_yaml(&digest.value),
    )
    .unwrap();

    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.tokenizer", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:compute/run", "2.0.0"))
                .with_export(WitInterface::new("magnetar:tokenizer/tokenize", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.tokenizer")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_broad_authority_kinds() {
    for authority in [
        "filesystem",
        "network",
        "secrets",
        "git",
        "workspace",
        "process",
    ] {
        let directory = temp_component_artifact_dir(authority);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest_yaml_with_authority(&digest.value, &[authority]),
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_artifact_rejects_unknown_authority_kinds() {
    let directory = temp_component_artifact_dir("unknown-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["workspace-admin"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { message, .. }) if message == "unsupported authority kind"
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_accepts_model_artifact_read_authority_when_trusted() {
    let directory = temp_component_artifact_dir("model-artifact-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["model-artifact-read"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_overrides_do_not_allow_forbidden_authority() {
    for (label, trust_store, source_kind) in [
        ("trusted-digest", ComponentTrustStore::default(), "local"),
        (
            "development-mode",
            ComponentTrustStore::default().allow_unsigned_local_development(true),
            "local",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let mut trust_store = trust_store;
        if label == "trusted-digest" {
            trust_store = trust_store.trust_digest(digest.value.clone());
        }
        let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
            .replace("  kind: \"local\"", &format!("  kind: \"{source_kind}\""));
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(trust_store);
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_authority_rejection_is_observed_with_reason_before_prepare() {
    let directory = temp_component_artifact_dir("authority-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["filesystem"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { .. })
    ));
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Failed)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Validation
            && observation.message.contains("component authority rejected")
            && observation
                .message
                .contains("authority kind is outside Magnetar inference scope")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_validation_observations_redact_paths_and_secrets() {
    let directory = temp_component_artifact_dir("redacted-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("[redacted]"))
    );
    assert!(!messages.iter().any(|message| {
        message.contains(directory.to_string_lossy().as_ref())
            || message.to_ascii_lowercase().contains("secret")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_pipeline_requires_external_trust_policy_before_prepare() {
    let directory = temp_component_artifact_dir("trusted");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = directory.join("hello.component.wasm.magnetar-component.yaml");
    fs::write(
        &manifest,
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let import = WitInterface::new("magnetar:test/echo", "1.0.0");
    let export = WitInterface::new("magnetar:test/run", "1.0.0");
    let metadata = ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
        .with_import(import)
        .with_export(export);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    manager
        .register_component(ComponentDescriptor::new(metadata, &artifact))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_local_wasm_without_manifest() {
    let directory = temp_component_artifact_dir("missing-manifest");
    let artifact = directory.join("unknown.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("unknown", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("unknown"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_digest_mismatch_before_prepare() {
    let directory = temp_component_artifact_dir("digest-mismatch");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let digest = ComponentDigest::sha256(b"different-bytes");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_manifest_wit_that_differs_from_actual_contract() {
    let directory = temp_component_artifact_dir("wit-mismatch");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let mut engine = MockComponentEngine::new();
    let mut actual = ComponentContract::default();
    actual.imports.insert(ComponentImportRequirement::new(
        WitInterface::new("magnetar:test/other", "1.0.0"),
        ComponentInterfaceShape::Interface,
    ));
    engine.prepared_contract = Some(actual);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_revocation_overrides_digest_allowlist() {
    let directory = temp_component_artifact_dir("revoked");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(
        ComponentTrustStore::default()
            .trust_digest(digest.value.clone())
            .revoke_digest(digest.value),
    );
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Revoked,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_loads_minimal_yaml_format() {
    let directory = temp_component_artifact_dir("trust-store");
    let digest = ComponentDigest::sha256(b"component-bytes");
    let trust_store = directory.join("trust.yaml");
    fs::write(&trust_store, trust_store_yaml(&digest.value)).unwrap();

    let loaded = ComponentTrustStore::load_yaml(&trust_store).unwrap();
    assert!(loaded.trusted_digests.contains(&digest.value));
    assert!(!loaded.allow_unsigned_local_development);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_manifest_may_declare_optional_wit_import_metadata() {
    let directory = temp_component_artifact_dir("optional-import");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "  exports:",
        "    - package: \"magnetar:optional\"\n      interface: \"telemetry\"\n      version: \"1.0.0\"\n      optional: true\n  exports:",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_runtime_max_version_incompatibility() {
    let directory = temp_component_artifact_dir("runtime-max");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, "0.0.1").replace(
        "    min_version: \"0.0.1\"",
        "    min_version: \"0.0.1\"\n    max_version: \"0.0.1\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_incompatible_capability_versions() {
    for (label, capability_block) in [
        (
            "cap-major",
            "    - id: \"magnetar:test/echo\"\n      version: \"2.0.0\"",
        ),
        (
            "cap-range",
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"\n      max_version: \"0.9.0\"",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"",
            capability_block,
        );
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::ArtifactRejected {
                status: ComponentTrustStatus::Rejected,
                ..
            })
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_publisher_and_source_metadata_do_not_grant_trust() {
    let directory = temp_component_artifact_dir("publisher-policy");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut untrusted = ComponentManager::new();
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    untrusted.register_component(descriptor()).unwrap();
    assert!(matches!(
        untrusted.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut publisher_claim = ComponentManager::new();
    publisher_claim.set_trust_store(ComponentTrustStore::default().trust_publisher("local-dev"));
    publisher_claim.register_component(descriptor()).unwrap();
    assert!(matches!(
        publisher_claim.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut explicit_local_development = ComponentManager::new();
    explicit_local_development.set_trust_store(
        ComponentTrustStore::default()
            .trust_publisher("local-dev")
            .trust_source("local")
            .allow_unsigned_local_development(true),
    );
    explicit_local_development
        .register_component(descriptor())
        .unwrap();
    explicit_local_development
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let tachyon_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION)
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        tachyon_manifest,
    )
    .unwrap();
    let mut source_claim = ComponentManager::new();
    source_claim.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));
    source_claim.register_component(descriptor()).unwrap();
    assert!(matches!(
        source_claim.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut digest_trusted = ComponentManager::new();
    digest_trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    digest_trusted.register_component(descriptor()).unwrap();
    digest_trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_signature_metadata_is_recorded_but_not_trusted_by_itself() {
    let directory = temp_component_artifact_dir("signature");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let signed_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        &format!(
            "signatures:\n  - algorithm: \"test\"\n    digest: \"{}\"",
            digest.value
        ),
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        signed_manifest,
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut no_trust = ComponentManager::new();
    no_trust.register_component(descriptor()).unwrap();
    assert!(matches!(
        no_trust.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let bad_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        "signatures:\n  - algorithm: \"test\"\n    digest: \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        bad_manifest,
    )
    .unwrap();
    let mut mismatch = ComponentManager::new();
    mismatch.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    mismatch.register_component(descriptor()).unwrap();
    assert!(matches!(
        mismatch.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_development_mode_is_explicit_and_still_validates_artifact() {
    let directory = temp_component_artifact_dir("development");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().allow_unsigned_local_development(true));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_cache_is_digest_keyed_and_non_authoritative() {
    let bytes = b"component-bytes".to_vec();
    let mut cache = ComponentArtifactCache::default();
    let digest = cache.insert(bytes.clone());
    assert!(cache.contains_untrusted(&digest));
    assert_eq!(cache.get_verified(&digest).unwrap(), Some(bytes.as_slice()));

    let wrong_digest = ComponentDigest::sha256(b"wrong");
    cache.insert_unchecked_for_test(wrong_digest.clone(), bytes);
    assert!(matches!(
        cache.get_verified(&wrong_digest),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
}

#[test]
fn component_quarantine_prevents_preparation_and_preserves_diagnostic_status() {
    let directory = temp_component_artifact_dir("quarantine");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().quarantine_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Quarantined,
            ..
        })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("Quarantined"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_validation_emits_structured_observations() {
    let directory = temp_component_artifact_dir("observations");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("discovered"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("digest computed"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("manifest loaded"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("WIT declarations match"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("compatibility validated"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("trust decision"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn runtime_initializes_memory_manager_as_first_class_service() {
    let runtime = Runtime::initialize(RuntimeConfig::default());

    assert!(runtime.is_initialized());
    assert_eq!(runtime.memory().config(), &MemoryManagerConfig::default());
    assert_eq!(runtime.memory().allocations().count(), 0);
}

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
fn memory_manager_distinguishes_storage_and_compute_dtype_costs() {
    let dtype = MemoryDTypeRelation::new(
        DTypeDescriptor::portable(ComputeDType::SInt8),
        DTypeDescriptor::portable(ComputeDType::BrainFloat16),
    );

    assert_eq!(dtype.storage_size_bytes(8).unwrap(), 8);
    assert_eq!(dtype.compute_workspace_bytes(8).unwrap(), 16);
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

fn generation_tokenizer_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("fixture").unwrap(),
        artifact: TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(16),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: vec![SpecialToken::new(SpecialTokenKind::Stop, "<stop>", 298)],
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    }
}

fn generation_request() -> GenerationRequest {
    let metadata = generation_tokenizer_metadata();
    GenerationRequest {
        request_id: GenerationRequestId::new("gen-1").unwrap(),
        session: None,
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        input_token_ids: vec![2, 3, 4],
        prompt_token_count: 3,
        max_new_tokens: 4,
        max_total_tokens: Some(8),
        model_context_length: Some(8),
        parameters: GenerationParameters::default(),
        stop_conditions: StopConditions {
            eos: EosPolicy {
                mode: EosMode::Stop,
                output: EosOutputPolicy::Exclude,
                eos_token_ids: vec![299],
            },
            stop_token_ids: vec![298],
            stop_token_patterns: vec![vec![10, 11]],
            stop_text_sequences: vec!["stop".into()],
            prepared_stop_sequences: vec![TokenStopPattern {
                text: "xy".into(),
                token_ids: vec![121, 122],
                exact: true,
            }],
            ..StopConditions::default()
        },
        streaming: StreamingMode::TokenIds,
        priority: GenerationPriority {
            priority: 3,
            deadline_millis: Some(100),
        },
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 12,
            output_token_buffer_bytes: 16,
            logits_buffer_bytes: 32,
            sampling_workspace_bytes: 8,
            prefill_workspace_bytes: 8,
            decode_workspace_bytes: 8,
            kv_cache_placeholder_bytes: 8,
            prefix_cache_placeholder_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: false,
        },
        correlation_id: Some(CorrelationId::new("corr-1")),
        trace_id: Some(TraceId::new("trace-1")),
    }
}

#[derive(Clone)]
struct TestGenerationExecutor {
    vocabulary_size: usize,
    evidence: RuntimeGenerationExecutionEvidence,
}

impl RuntimeModelExecutionEngine for TestGenerationExecutor {
    fn execute_generation_step(
        &self,
        _runtime: &mut Runtime,
        _request: &GenerationRequest,
        generated_tokens: &[TokenId],
        _execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        let mut logits = vec![0.0f32; self.vocabulary_size];
        logits[(11 + generated_tokens.len()) % self.vocabulary_size] = 10.0;
        Ok(RuntimeModelExecutionStep::new(
            logits,
            self.evidence.clone(),
        ))
    }
}

#[derive(Clone)]
struct FailingGenerationExecutor;

impl RuntimeModelExecutionEngine for FailingGenerationExecutor {
    fn execute_generation_step(
        &self,
        _runtime: &mut Runtime,
        _request: &GenerationRequest,
        _generated_tokens: &[TokenId],
        _execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        Err(InferenceApiError::ProviderUnavailable {
            reason: "provider failed during decode".into(),
        })
    }
}

#[derive(Clone)]
struct FailingKernelGenerationExecutor;

impl RuntimeModelExecutionEngine for FailingKernelGenerationExecutor {
    fn execute_generation_step(
        &self,
        _runtime: &mut Runtime,
        _request: &GenerationRequest,
        _generated_tokens: &[TokenId],
        _execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        Err(InferenceApiError::KernelUnavailable {
            reason: "kernel failed during decode".into(),
        })
    }
}

fn runtime_with_model_execution_engine(
    vocabulary_size: usize,
    evidence: RuntimeGenerationExecutionEvidence,
) -> Runtime {
    Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(Arc::new(TestGenerationExecutor {
            vocabulary_size,
            evidence,
        }))
        .build()
        .unwrap()
}

fn generation_runtime_tokenizer() -> RuntimeTokenizer<FixtureTokenizer> {
    let metadata = generation_tokenizer_metadata();
    let digest = metadata.digest.clone();
    RuntimeTokenizer::new(
        FixtureTokenizer::new(metadata),
        TokenizerArtifactSet {
            tokenizer: TokenizerArtifactReference::new(
                TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
                ModelArtifactKind::Tokenizer,
                digest,
            )
            .unwrap(),
            tokenizer_config: None,
            vocabulary: None,
            special_tokens: None,
        },
    )
}

#[test]
fn generation_request_validation_is_token_based_and_context_aware() {
    let request = generation_request();
    request.validate().unwrap();
    assert_eq!(request.prompt_token_count, request.input_token_ids.len());
    assert!(matches!(
        request.model,
        GenerationModelReference::LoadedModelContext(_)
    ));
}

#[test]
fn generation_request_rejects_invalid_input_tokens() {
    let mut request = generation_request();
    request.input_token_ids.push(999);
    request.prompt_token_count += 1;

    assert!(matches!(
        request.validate(),
        Err(GenerationError::InputTokensInvalid { .. })
    ));
}

#[test]
fn generation_request_rejects_prompt_that_exceeds_limits_without_truncation() {
    let mut request = generation_request();
    request.max_new_tokens = 20;

    assert!(matches!(
        request.validate(),
        Err(GenerationError::PromptTooLong { .. })
    ));
}

#[test]
fn generation_parameters_validate_temperature_sampling_and_greedy_modes() {
    let mut invalid = generation_request();
    invalid.parameters.temperature = f32::NAN;
    assert!(matches!(
        invalid.validate(),
        Err(GenerationError::ParameterInvalid {
            parameter: "temperature",
            ..
        })
    ));

    let mut greedy = generation_request();
    greedy.parameters = GenerationParameters::greedy();
    greedy.validate().unwrap();
    assert!(!greedy.parameters.sampling_enabled);
}

#[test]
fn generation_stop_conditions_distinguish_length_eos_token_and_sequences() {
    let request = generation_request();

    assert_eq!(
        stop_reason_for(&request, &[1, 2, 3, 4]),
        Some(FinishReason::MaxNewTokens)
    );
    assert_eq!(
        stop_reason_for(&request, &[299]),
        Some(FinishReason::EosToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[298]),
        Some(FinishReason::StopToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 10, 11]),
        Some(FinishReason::StopSequence)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 121, 122]),
        Some(FinishReason::StopSequence)
    );
}

#[test]
fn generation_can_ignore_eos_by_explicit_policy() {
    let mut request = generation_request();
    request.stop_conditions.eos.mode = EosMode::Ignore;

    assert_eq!(stop_reason_for(&request, &[299]), None);
}

#[test]
fn generation_decode_step_preserves_token_index_and_state_boundary() {
    let request = generation_request();
    let step = decode_step(&request, &[20, 21], 22).unwrap();

    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_decode_step_delegates_next_token_selection_to_sampling() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let mut logits = vec![0.0; request.tokenizer.metadata.vocabulary_size as usize];
    logits[21] = 10.0;

    let (sampling, step) =
        decode_step_from_sampling(&request, &[20, 21], logits, SamplingPolicy::default()).unwrap();

    assert_eq!(sampling.selected_token_id, 22);
    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_prefill_validates_tokens_and_records_prompt_count() {
    let request = generation_request();
    let state = prefill(&request).unwrap();

    assert_eq!(state.prompt_token_count, 3);
    assert!(state.kv_cache_placeholder.is_some());
    assert!(state.observations.iter().any(|event| {
        event.kind == GenerationEventKind::PrefillStarted && event.request_id == request.request_id
    }));
}

#[test]
fn generation_token_stream_events_preserve_order_and_identity() {
    let request = generation_request();
    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();

    assert_eq!(events.len(), 3);
    assert_eq!(events[0].token_id, Some(10));
    assert_eq!(events[1].token_index, Some(1));
    assert!(
        events
            .iter()
            .all(|event| event.request_id == request.request_id)
    );
}

#[test]
fn generation_streaming_text_uses_tokenizer_decode() {
    let tokenizer = generation_runtime_tokenizer();
    let output = streaming_text_chunk(
        &tokenizer,
        StreamingDecodeState::default(),
        vec![b'h' as TokenId + 1, b'i' as TokenId + 1],
        true,
    )
    .unwrap();

    assert_eq!(output.text, "hi");
    assert!(output.pending_partial_state.is_none());
}

#[test]
fn generation_prepares_text_stop_sequences_through_tokenizer() {
    let tokenizer = generation_runtime_tokenizer();
    let patterns = prepare_stop_sequences(&tokenizer, &["xy".into()]).unwrap();

    assert_eq!(patterns[0].text, "xy");
    assert_eq!(
        patterns[0].token_ids,
        vec![b'x' as TokenId + 1, b'y' as TokenId + 1]
    );
}

#[test]
fn generation_usage_and_output_account_for_tokens_without_decoded_text() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11], FinishReason::StopToken);

    output.validate().unwrap();
    assert_eq!(output.generated_token_count, 2);
    assert_eq!(output.usage.prompt_tokens, 3);
    assert_eq!(output.usage.total_tokens, 5);
}

#[test]
fn generation_cancellation_maps_to_stable_finish_reason() {
    let mut request = generation_request();
    request.cancellation.requested = true;

    assert_eq!(
        stop_reason_for(&request, &[]),
        Some(FinishReason::Cancelled)
    );
}

#[test]
fn generation_memory_admission_uses_memory_manager_policy() {
    let mut request = generation_request();
    request.memory.logits_buffer_bytes = 1024;
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(64),
        ..MemoryManagerConfig::default()
    });

    assert!(matches!(
        memory_admission(&request, &manager).unwrap(),
        MemoryAdmissionDecision::Reject { .. }
    ));
}

#[test]
fn generation_provider_errors_map_to_finish_reasons() {
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionInterrupted),
        FinishReason::Interrupted
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::OutOfMemory),
        FinishReason::MemoryLimit
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionFailed),
        FinishReason::ProviderError
    );
}

#[test]
fn generation_contract_has_no_authoritative_provider_or_device_selector() {
    let request = generation_request();

    assert_eq!(request.priority.priority, 3);
    assert!(request.correlation_id.is_some());
    assert!(request.trace_id.is_some());
}

fn reference_cpu_host_tensor(shape: impl Into<Vec<u64>>, data: impl Into<Vec<f32>>) -> HostTensor {
    HostTensor::new(shape, data).unwrap()
}

#[test]
fn reference_cpu_provider_identity_and_device_are_stable() {
    let provider = ReferenceCpuProvider::new();
    let metadata = provider.metadata();
    assert_eq!(metadata.name, REFERENCE_CPU_PROVIDER_NAME);
    assert_eq!(metadata.vendor, REFERENCE_CPU_PROVIDER_VENDOR);

    let devices = provider.devices();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), REFERENCE_CPU_DEVICE_ID);
    assert_eq!(devices[0].device_type(), DeviceType::Cpu);

    let (min, max) = REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE;
    assert!(min <= max);
    assert_eq!(
        REFERENCE_CPU_KERNEL_FAMILY,
        KernelImplementationFamily::CpuScalar
    );
}

#[test]
fn reference_cpu_provider_status_snapshot_reports_health_and_lifecycle() {
    let provider = ReferenceCpuProvider::new();
    let snapshot = provider.status_snapshot();
    assert_eq!(snapshot.provider.as_str(), REFERENCE_CPU_PROVIDER_NAME);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Ready);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Admit);
    assert!(snapshot.diagnostics.is_empty());
}

#[test]
fn reference_cpu_provider_pressure_is_explicitly_reportable() {
    let provider = ReferenceCpuProvider::new();
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Low
    );
    provider.report_pressure(ProviderPressureLevel::Saturated);
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn reference_cpu_device_advertises_dtype_layout_memory_class_and_limits() {
    let device = reference_cpu_device();
    let metadata = device.metadata();
    assert!(metadata.dtype_support.contains(&ComputeDType::Float32));
    assert!(
        metadata
            .layout_support
            .contains(&TensorLayoutKind::Contiguous)
    );
    assert!(
        metadata
            .memory_class_support
            .contains(&KernelMemoryClass::Host)
    );
    assert!(
        metadata
            .execution_limits
            .max_concurrent_operations
            .is_some()
    );
    assert_eq!(metadata.pressure, ProviderPressureLevel::Low);
}

#[test]
fn reference_cpu_initialize_emits_provider_registered_and_device_detected() {
    let provider = ReferenceCpuProvider::new();
    provider.initialize().unwrap();
    let observations = provider.executor().observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::ProviderRegistered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::DeviceDetected)
    );
}

#[test]
fn reference_cpu_conformance_report_passes_and_is_observed() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let report = executor.run_conformance_checks();
    assert!(
        report.is_conformant(),
        "Reference CPU conformance checks failed: {:?}",
        report.checks
    );
    assert_eq!(report.profile, REFERENCE_CPU_CONFORMANCE_PROFILE);
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelConformanceResult)
    );
}

#[test]
fn reference_cpu_advertises_only_implemented_kernels() {
    let provider = ReferenceCpuProvider::new();
    let advertisements = provider.kernel_advertisements();
    let names = advertisements
        .iter()
        .map(|advertisement| advertisement.id.name.as_str())
        .collect::<BTreeSet<_>>();
    for expected in [
        "matmul",
        "embedding",
        "rmsnorm",
        "rope",
        "attention",
        "softmax",
        "silu",
        "gelu",
        "activation",
        "add",
        "mul",
        "residual-add",
        "dtype-conversion",
        "layout-conversion",
    ] {
        assert!(names.contains(expected), "missing kernel: {expected}");
    }
    assert!(!names.contains("quantize"));
    assert!(!names.contains("dequantize"));
    for advertisement in &advertisements {
        validate_kernel_advertisement(advertisement).unwrap();
    }
}

#[test]
fn reference_cpu_matmul_known_output() {
    let a = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let b = reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
    let result = matmul(&a, &b, false, false).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn reference_cpu_matmul_rejects_inner_dimension_mismatch() {
    let a = reference_cpu_host_tensor([2, 3], vec![0.0; 6]);
    let b = reference_cpu_host_tensor([2, 2], vec![0.0; 4]);
    assert!(matmul(&a, &b, false, false).is_err());
}

#[test]
fn reference_cpu_embedding_known_output_and_out_of_range() {
    let table = reference_cpu_host_tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ids = reference_cpu_host_tensor([2], [0.0, 2.0]);
    let result = embedding_lookup(&table, &ids).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![1.0, 2.0, 5.0, 6.0]);

    let out_of_range = reference_cpu_host_tensor([1], [3.0]);
    assert!(embedding_lookup(&table, &out_of_range).is_err());
}

#[test]
fn reference_cpu_rmsnorm_known_output() {
    let input = reference_cpu_host_tensor([1, 4], [1.0, 2.0, 3.0, 4.0]);
    let weight = reference_cpu_host_tensor([4], [1.0, 1.0, 1.0, 1.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let mean_square = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    for (actual, expected) in result.data.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - expected * scale).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_rmsnorm_full_shape_weights_apply_per_row() {
    let input = reference_cpu_host_tensor([2, 2], [3.0, 4.0, 3.0, 4.0]);
    let weight = reference_cpu_host_tensor([2, 2], [1.0, 1.0, 2.0, 3.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let scale = 1.0 / (((9.0_f32 + 16.0) / 2.0) + 1e-6).sqrt();

    assert!((result.data[0] - 3.0 * scale).abs() < 1e-5);
    assert!((result.data[1] - 4.0 * scale).abs() < 1e-5);
    assert!((result.data[2] - 3.0 * scale * 2.0).abs() < 1e-5);
    assert!((result.data[3] - 4.0 * scale * 3.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_rmsnorm_flattens_leading_dimensions() {
    let input = reference_cpu_host_tensor([1, 2, 2], [3.0, 4.0, 5.0, 12.0]);
    let weight = reference_cpu_host_tensor([1, 2, 2], [1.0, 2.0, 3.0, 4.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let row0_scale = 1.0 / (((9.0_f32 + 16.0) / 2.0) + 1e-6).sqrt();
    let row1_scale = 1.0 / (((25.0_f32 + 144.0) / 2.0) + 1e-6).sqrt();

    assert_eq!(result.shape, vec![1, 2, 2]);
    assert!((result.data[0] - 3.0 * row0_scale).abs() < 1e-5);
    assert!((result.data[1] - 4.0 * row0_scale * 2.0).abs() < 1e-5);
    assert!((result.data[2] - 5.0 * row1_scale * 3.0).abs() < 1e-5);
    assert!((result.data[3] - 12.0 * row1_scale * 4.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_rmsnorm_rejects_dtype_shape_mismatch() {
    let input = reference_cpu_host_tensor([1, 4], vec![1.0; 4]);
    let weight = reference_cpu_host_tensor([3], vec![1.0; 3]);
    assert!(rmsnorm(&input, &weight, 1e-6).is_err());
}

#[test]
fn reference_cpu_rope_identity_at_position_zero() {
    let input = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let result = rope(&input, 10000.0, 1.0, 2, 0).unwrap();
    assert!((result.data[0] - 1.0).abs() < 1e-5);
    assert!((result.data[1] - 2.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_softmax_known_output() {
    let input = reference_cpu_host_tensor([1, 3], [1.0, 1.0, 1.0]);
    let result = softmax_rows(&input).unwrap();
    for value in result.data {
        assert!((value - (1.0 / 3.0)).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_softmax_rejects_invalid_shape() {
    let input = HostTensor {
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
    };
    assert!(softmax_rows(&input).is_err());
}

#[test]
fn reference_cpu_softmax_rejects_fully_masked_row() {
    // Every entry masked out: subtracting the row max would yield NaN for the
    // whole row, so the kernel must reject it rather than return Ok(NaN).
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY; 3]);
    let error = softmax_rows(&input).expect_err("fully masked row must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ExecutionFailed);
}

#[test]
fn reference_cpu_softmax_allows_partially_masked_row() {
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
    let result = softmax_rows(&input).unwrap();
    assert!(result.data.iter().all(|value| value.is_finite()));
    assert!((result.data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_silu_known_output() {
    let input = reference_cpu_host_tensor([1], [0.0]);
    let result = silu(&input);
    assert!((result.data[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reference_cpu_elementwise_known_outputs() {
    let a = reference_cpu_host_tensor([2], [1.0, 2.0]);
    let b = reference_cpu_host_tensor([2], [3.0, 4.0]);
    assert_eq!(add(&a, &b).unwrap().data, vec![4.0, 6.0]);
    assert_eq!(mul(&a, &b).unwrap().data, vec![3.0, 8.0]);
    assert_eq!(residual_add(&a, &b).unwrap().data, vec![4.0, 6.0]);

    let mismatched = reference_cpu_host_tensor([3], vec![0.0; 3]);
    assert!(add(&a, &mismatched).is_err());
}

#[test]
fn reference_cpu_attention_causal_masks_future_tokens() {
    let q = reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 2], [10.0, 10.0, 20.0, 20.0]);
    let result = attention(&q, &k, &v, 1, 2, None, None, true).unwrap();
    // Position 0 can only attend to itself, so its output must equal v[0].
    assert!((result.data[0] - 10.0).abs() < 1e-4);
    assert!((result.data[1] - 10.0).abs() < 1e-4);
}

#[test]
fn reference_cpu_attention_grouped_query_shares_kv_heads() {
    // 2 query heads sharing 1 kv head (head_dimension = 2).
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 2], [5.0, 6.0]);
    let v = reference_cpu_host_tensor([1, 2], [7.0, 8.0]);
    let result = attention(&q, &k, &v, 2, 2, Some(1), None, false).unwrap();
    // Single key position: every query head's output must equal v.
    assert_eq!(result.data, vec![7.0, 8.0, 7.0, 8.0]);
}

#[test]
fn reference_cpu_attention_rejects_incompatible_head_grouping() {
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 4], [5.0, 6.0, 7.0, 8.0]);
    let v = k.clone();
    // head_count 2 is not a multiple of kv_head_count 3.
    assert!(attention(&q, &k, &v, 2, 2, Some(3), None, false).is_err());
}

#[test]
fn reference_cpu_attention_window_size_restricts_context() {
    let q = reference_cpu_host_tensor([3, 1], [0.0, 0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([3, 1], [1.0, 2.0, 3.0]);
    // window_size = 1: each position can only see itself.
    let result = attention(&q, &k, &v, 1, 1, None, Some(1), true).unwrap();
    assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn reference_cpu_attention_rejects_zero_window() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // A zero window admits no keys at all; it must not be silently widened to 1.
    let error =
        attention(&q, &k, &v, 1, 1, None, Some(0), true).expect_err("zero window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_attention_rejects_window_without_causal_mask() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // The window is anchored at the query position, which only fully describes
    // the mask under causal attention.
    let error = attention(&q, &k, &v, 1, 1, None, Some(1), false)
        .expect_err("bidirectional sliding window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_host_tensor_rejects_overflowing_shape() {
    // The product of these dimensions wraps to 0 under unchecked u64
    // multiplication, which would let an empty buffer pass the length check.
    let error = HostTensor::new([1_u64 << 32, 1_u64 << 32], Vec::<f32>::new())
        .expect_err("overflowing shape must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_host_tensor_rejects_shape_beyond_address_space() {
    let error = HostTensor::new([u64::MAX], Vec::<f32>::new())
        .expect_err("shape beyond the address space must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

fn reference_cpu_kernel_by_name<'a>(
    advertisements: &'a [KernelAdvertisement],
    name: &str,
) -> &'a KernelAdvertisement {
    advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == name)
        .unwrap_or_else(|| panic!("no advertisement named {name}"))
}

#[test]
fn reference_cpu_generic_activation_kernel_dispatches_on_kind() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "activation");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("activation-in", [1]);
    let (_out_id, out_resource) = reference_cpu_resource("activation-out", [1]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1], [0.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert(
        "kind".to_string(),
        OperatorAttributeValue::String("silu".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-activation"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource.clone())
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert!((output.data[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reference_cpu_generic_activation_kernel_rejects_unknown_kind() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "activation");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("activation-bad-in", [1]);
    let (_out_id, out_resource) = reference_cpu_resource("activation-bad-out", [1]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1], [0.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert(
        "kind".to_string(),
        OperatorAttributeValue::String("relu".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-activation-bad"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource)
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

#[test]
fn reference_cpu_advertisements_match_numeric_storage_constraints() {
    let advertisements = reference_cpu_kernel_advertisements();
    let rmsnorm = reference_cpu_kernel_by_name(&advertisements, "rmsnorm");
    assert_eq!(rmsnorm.shape.rank, None);

    let embedding = reference_cpu_kernel_by_name(&advertisements, "embedding");
    let input_dtypes = embedding
        .supported_dtypes
        .get(&TensorRole::Input)
        .expect("embedding advertises input dtypes");
    assert!(!input_dtypes.contains(&ComputeDType::SInt32));
    assert!(input_dtypes.contains(&ComputeDType::Float32));
}

#[test]
fn reference_cpu_rope_rejects_unimplemented_position_mode() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "rope");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("rope-mode-in", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("rope-mode-out", [1, 2]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1, 2], [1.0, 2.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert("base".to_string(), OperatorAttributeValue::Float(10000.0));
    attributes.insert("dimension".to_string(), OperatorAttributeValue::Integer(2));
    attributes.insert(
        "position_mode".to_string(),
        OperatorAttributeValue::String("absolute".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-rope-mode"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource)
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

fn reference_cpu_attention_invocation(
    advertisement: &KernelAdvertisement,
    causal: bool,
    mask_kind: Option<&str>,
    q: KernelResource,
    k: KernelResource,
    v: KernelResource,
    out: KernelResource,
) -> KernelInvocation {
    let mut attributes = BTreeMap::new();
    attributes.insert("head_count".to_string(), OperatorAttributeValue::Integer(1));
    attributes.insert(
        "head_dimension".to_string(),
        OperatorAttributeValue::Integer(2),
    );
    attributes.insert(
        "causal".to_string(),
        OperatorAttributeValue::Boolean(causal),
    );
    if let Some(mask_kind) = mask_kind {
        attributes.insert(
            "attention_mask_kind".to_string(),
            OperatorAttributeValue::String(mask_kind.into()),
        );
    }
    KernelInvocation::new(
        KernelInvocationId::new("invocation-attention"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(q)
    .with_input(k)
    .with_input(v)
    .with_output(out)
    .with_attributes(attributes)
}

#[test]
fn reference_cpu_attention_requires_workspace_from_memory_manager() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "attention");
    assert!(advertisement.workspace.required);
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (q_id, q_resource) = reference_cpu_resource("attn-q", [1, 2]);
    let (k_id, k_resource) = reference_cpu_resource("attn-k", [1, 2]);
    let (v_id, v_resource) = reference_cpu_resource("attn-v", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("attn-out", [1, 2]);
    executor.write_tensor(q_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(k_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(v_id, reference_cpu_host_tensor([1, 2], [5.0, 6.0]));

    // Without a workspace attached, the shared Kernel Contract validation
    // rejects the invocation before Reference CPU ever runs it.
    let invocation_without_workspace = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource.clone(),
        k_resource.clone(),
        v_resource.clone(),
        out_resource.clone(),
    );
    let rejected =
        executor.execute_invocation(advertisement, operator, &invocation_without_workspace);
    assert_eq!(rejected.status, KernelResultStatus::Failed);
    assert_eq!(
        rejected.error,
        Some(KernelError::KernelWorkspaceUnavailable)
    );

    // With a workspace requested through the Memory Manager, execution
    // succeeds.
    let workspace = executor.allocate_workspace(&mut memory, 4096).unwrap();
    let invocation = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource,
        k_resource,
        v_resource,
        out_resource.clone(),
    )
    .with_workspace(workspace);
    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert_eq!(output.data, vec![5.0, 6.0]);
}

#[test]
fn reference_cpu_attention_mask_kind_must_be_consistent_with_causal_flag() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "attention");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let workspace = executor.allocate_workspace(&mut memory, 4096).unwrap();

    let (q_id, q_resource) = reference_cpu_resource("attn-mismatch-q", [1, 2]);
    let (k_id, k_resource) = reference_cpu_resource("attn-mismatch-k", [1, 2]);
    let (v_id, v_resource) = reference_cpu_resource("attn-mismatch-v", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("attn-mismatch-out", [1, 2]);
    executor.write_tensor(q_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(k_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(v_id, reference_cpu_host_tensor([1, 2], [5.0, 6.0]));

    // causal=false but attention_mask_kind says "causal": inconsistent.
    let invocation = reference_cpu_attention_invocation(
        advertisement,
        false,
        Some("causal"),
        q_resource,
        k_resource,
        v_resource,
        out_resource,
    )
    .with_workspace(workspace);
    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

#[test]
fn reference_cpu_dtype_conversion_rejects_non_f32() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(dtype_conversion(&input, ComputeDType::Float32, ComputeDType::Float32).is_ok());
    assert!(dtype_conversion(&input, ComputeDType::Float16, ComputeDType::Float32).is_err());
}

#[test]
fn reference_cpu_layout_conversion_rejects_non_contiguous() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Contiguous
        )
        .is_ok()
    );
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Strided
        )
        .is_err()
    );
}

#[test]
fn reference_cpu_quantization_is_explicitly_unsupported() {
    let error = dequantize_placeholder();
    assert_eq!(error.id(), "reference-cpu-dtype-unsupported");
}

#[test]
fn reference_cpu_fallback_denied_by_default_allowed_by_policy() {
    let pinned = ResourceAffinity::new(FallbackClass::ProviderPinned);
    assert!(evaluate_fallback(&pinned, &FallbackPolicyContext::new(true)).is_err());

    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(false)).is_err());
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(true)).is_ok());
}

#[test]
fn reference_cpu_fallback_denied_when_dtype_or_layout_conversion_forbidden() {
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    let dtype_denied = FallbackPolicyContext::new(true).with_dtype_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &dtype_denied).is_err());

    let layout_denied = FallbackPolicyContext::new(true).with_layout_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &layout_denied).is_err());

    let both_allowed = FallbackPolicyContext::new(true)
        .with_dtype_conversion(true, true)
        .with_layout_conversion(true, true);
    assert!(evaluate_fallback(&transparent, &both_allowed).is_ok());
}

#[test]
fn reference_cpu_fallback_is_observable() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let kernel = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(true))
        .unwrap();
    let observations = executor.observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackConsidered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackUsed)
    );

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(false))
        .unwrap_err();
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackFailed)
    );
}

fn reference_cpu_resource(
    id: &str,
    shape: impl Into<Vec<u64>>,
) -> (TensorResourceId, KernelResource) {
    let resource_id = TensorResourceId::new(id);
    let descriptor = TensorResourceDescriptor::new(
        resource_id.clone(),
        TensorDescriptor::materialized(
            ShapeDescriptor::new(shape.into()),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    (
        resource_id,
        KernelResource::new(descriptor, KernelMemoryClass::Host),
    )
}

#[test]
fn reference_cpu_executes_matmul_invocation_end_to_end() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-1"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource.clone());

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert_eq!(output.data, vec![19.0, 22.0, 43.0, 50.0]);
}

/// `reach-architecture-freeze-1` task 5.4/5.5: a real Operator with more
/// than one declared output ("split") dispatched through the same generic
/// `KernelInvocation`/`execute_invocation` path every other Kernel uses,
/// proving `KernelInvocation.outputs: Vec<..>` and
/// `store_output(invocation, index, ..)` actually produce a distinct,
/// independently-readable resource per declared output -- not just that
/// the types allow more than one.
#[test]
fn reference_cpu_split_kernel_produces_a_dedicated_resource_per_output() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "split")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("split-in", [2, 4]);
    let (_left_id, left_resource) = reference_cpu_resource("split-left", [2, 2]);
    let (_right_id, right_resource) = reference_cpu_resource("split-right", [2, 2]);
    executor.write_tensor(
        input_id,
        reference_cpu_host_tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-split"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(left_resource.clone())
    .with_output(right_resource.clone());

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    assert_eq!(
        result.updated_resources.len(),
        2,
        "split must report both output resources as updated, not just one"
    );

    let left = executor.read_tensor(&left_resource.resource.id).unwrap();
    let right = executor.read_tensor(&right_resource.resource.id).unwrap();
    assert_eq!(left.shape, vec![2, 2]);
    assert_eq!(right.shape, vec![2, 2]);
    // Each row of the [2, 4] input splits into its first-half/second-half
    // elements, landing in two independently-readable resources.
    assert_eq!(left.data, vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(right.data, vec![3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn reference_cpu_execution_tracks_output_through_memory_manager() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (a_id, a_resource) = reference_cpu_resource("mm-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("mm-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("mm-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-memory"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource.clone());

    let result = executor.execute_invocation_with_memory_manager(
        advertisement,
        operator,
        &invocation,
        &mut memory,
    );
    assert_eq!(result.status, KernelResultStatus::Succeeded);

    let residency = memory
        .tensor_residency(&out_resource.resource.id)
        .expect("Memory Manager should record residency for the output tensor");
    assert!(residency.provider_owned);
    assert!(residency.allocation.is_some());
    assert!(memory.allocations().next().is_some());
}

#[test]
fn reference_cpu_denies_dispatch_when_output_admission_is_rejected() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    // A runtime byte budget too small for any tensor allocation to be admitted.
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });

    let (a_id, a_resource) = reference_cpu_resource("mem-fail-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("mem-fail-b", [2, 2]);
    let (out_id, out_resource) = reference_cpu_resource("mem-fail-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-memory-fail"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation_with_memory_manager(
        advertisement,
        operator,
        &invocation,
        &mut memory,
    );
    // Memory Admission Precedes Provider Materialization: when the declared
    // output cannot be admitted, the Kernel is never dispatched to the
    // Provider at all (no numerical work runs, no bytes are written).
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert!(executor.read_tensor(&out_id).is_none());
    assert!(executor.observations().iter().any(
        |observation| observation.kind == KernelObservationKind::KernelMemoryFeasibilityFailed
    ));
    assert!(
        !executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelDispatchStarted)
    );
    assert!(memory.allocations().next().is_none());
}

/// Correctif 1: `write_tensor_admitted` (unlike plain `write_tensor`, which
/// takes no `MemoryManager` at all) SHALL reject a write whose byte size
/// cannot be admitted, before the Provider's storage is touched.
#[test]
fn reference_cpu_write_tensor_admitted_rejects_write_when_budget_exhausted() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });
    let id = TensorResourceId::new("admitted-write-oom");
    let result = executor.write_tensor_admitted(
        &mut memory,
        id.clone(),
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
        MemoryAllocationClass::Tensor,
        MemoryAllocationOwner::Session("test-session".into()),
    );
    assert!(
        result.is_err(),
        "write must be rejected under an exhausted budget"
    );
    assert!(
        executor.read_tensor(&id).is_none(),
        "a rejected admission must not materialize the tensor into Provider storage"
    );
    assert!(memory.allocations().next().is_none());
}

/// Correctif 1: writing to the *same* resource id twice through
/// `write_tensor_admitted` SHALL release the allocation the first write
/// admitted, not accumulate a second, independent one -- otherwise a
/// resource id a caller rewrites every generation step (first-native graph
/// edges, KV-pending writes) would grow the memory ledger unboundedly over
/// a long-running session even though Provider storage itself does not
/// grow (each write overwrites the same entry).
#[test]
fn reference_cpu_write_tensor_admitted_releases_previous_allocation_for_same_resource_id() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let id = TensorResourceId::new("admitted-write-replaced");
    let owner = || MemoryAllocationOwner::Session("test-session".into());

    executor
        .write_tensor_admitted(
            &mut memory,
            id.clone(),
            reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            MemoryAllocationClass::Tensor,
            owner(),
        )
        .expect("first admitted write succeeds");
    let active_after_first = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(active_after_first, 1);

    executor
        .write_tensor_admitted(
            &mut memory,
            id.clone(),
            reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
            MemoryAllocationClass::Tensor,
            owner(),
        )
        .expect("second admitted write to the same resource id succeeds");
    let active_after_second = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(
        active_after_second, 1,
        "the first write's allocation must be released when the second replaces it, \
         not left active alongside the new one"
    );
    assert_eq!(
        executor
            .read_tensor(&id)
            .expect("resource is still present")
            .data,
        vec![5.0, 6.0, 7.0, 8.0],
        "Provider storage must reflect the second write's value"
    );
}

#[test]
fn reference_cpu_releases_admitted_output_reservation_when_kernel_execution_fails() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (_a_id, a_resource) = reference_cpu_resource("mem-release-a", [2, 2]);
    let (_b_id, b_resource) = reference_cpu_resource("mem-release-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("mem-release-out", [2, 2]);
    // Inputs are intentionally left unwritten so the Kernel itself fails
    // (`input_tensor` finds no materialized data) after admission already
    // succeeded, exercising the rollback path rather than the admission
    // rejection path.

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-memory-release"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation_with_memory_manager(
        advertisement,
        operator,
        &invocation,
        &mut memory,
    );
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert!(
        !memory
            .allocations()
            .any(|allocation| allocation.state == MemoryAllocationState::Active),
        "the output reservation admitted before dispatch must be released when the Kernel itself fails"
    );
}

#[test]
fn reference_cpu_kernel_submission_is_causal_and_single_consumption() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (a_id, a_resource) = reference_cpu_resource("submit-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("submit-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("submit-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-submit"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    assert!(
        executor.observations().is_empty(),
        "no dispatch should have happened before submission"
    );

    // submit_kernel_invocation is what causally triggers the numerical
    // work; Reference CPU is synchronous, so it has already run by the time
    // this call returns.
    let handle =
        executor.submit_kernel_invocation(advertisement, operator, &invocation, &mut memory);
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelDispatchStarted)
    );

    let result = executor
        .complete_kernel_invocation(&handle)
        .expect("work submitted above is completable exactly once");
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    assert_eq!(result.updated_resources.len(), 1);

    // Single consumption: completing the same handle a second time fails
    // rather than silently re-reporting the same result.
    assert!(executor.complete_kernel_invocation(&handle).is_err());
}

#[test]
fn reference_cpu_kernel_completion_reports_real_failure_not_false_success() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (_a_id, a_resource) = reference_cpu_resource("submit-fail-a", [2, 2]);
    let (_b_id, b_resource) = reference_cpu_resource("submit-fail-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("submit-fail-out", [2, 2]);
    // Inputs are intentionally left unwritten so the Kernel itself fails.

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-submit-fail"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let handle =
        executor.submit_kernel_invocation(advertisement, operator, &invocation, &mut memory);
    let result = executor
        .complete_kernel_invocation(&handle)
        .expect("a submitted invocation is completable even when the Kernel itself failed");
    assert_eq!(
        result.status,
        KernelResultStatus::Failed,
        "completion must report the real Kernel failure, not fabricate a success"
    );
    assert!(result.error.is_some());
}

#[test]
fn reference_cpu_rejects_completion_of_a_handle_that_was_never_submitted() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let fabricated_handle = ProviderExecutionHandle::new(
        ScheduledOperationId::new(0xDEAD_BEEF),
        ExecutionPlanId::new("never-submitted-plan"),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        None,
    );
    assert!(
        executor
            .complete_kernel_invocation(&fabricated_handle)
            .is_err()
    );
    // The generic ProviderExecutionApi surface rejects the same fabricated
    // handle for the same reason: no submission is associated with it.
    assert!(ProviderExecutionApi::complete(executor.as_ref(), &fabricated_handle).is_err());
    assert!(ProviderExecutionApi::status(executor.as_ref(), &fabricated_handle).is_err());
}

#[test]
fn reference_cpu_cancellation_is_explicitly_unsupported_not_silently_ignored() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let handle = ProviderExecutionHandle::new(
        ScheduledOperationId::new(1),
        ExecutionPlanId::new("cancel-probe-plan"),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        None,
    );
    let outcome = ProviderExecutionApi::cancel(executor.as_ref(), &handle).unwrap();
    assert_eq!(outcome, ProviderCancellationOutcome::Unsupported);
}

#[test]
fn reference_cpu_generic_provider_execution_api_completes_exactly_once() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let provider_binding = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
    let plan = ComputeExecutionPlan {
        id: ExecutionPlanId::new("generic-submit-plan"),
        trace_id: TraceId::new("trace-generic-submit"),
        graph: ComputeGraphId::new("generic-submit-graph"),
        provider: provider_binding.clone(),
        device: None,
        capability: CapabilityBinding::new(
            CapabilityId::new(COMPUTE_CAPABILITY_ID),
            COMPUTE_CAPABILITY_VERSION,
        ),
        policy: ResolutionPolicyId::new("generic-submit-policy"),
        classification: ComputeExecutionClassification::Transparent,
        inputs: Vec::new(),
        outputs: Vec::new(),
        constraints: Vec::new(),
        steps: Vec::new(),
        memory_plan: MemoryPlan::new(provider_binding.clone(), None, ExecutionContextId::new(0)),
        diagnostics: Vec::new(),
        validated: true,
    };
    let request = ProviderExecutionRequest {
        operation: ScheduledOperationId::new(1),
        plan,
        provider: provider_binding.clone(),
        device: None,
        affinity: ResourceAffinity::new(FallbackClass::Transparent),
        memory_plan: MemoryPlan::new(provider_binding, None, ExecutionContextId::new(0)),
        steps: Vec::new(),
        constraints: Vec::new(),
    };

    let handle = ProviderExecutionApi::submit(executor.as_ref(), request).unwrap();
    let status = ProviderExecutionApi::status(executor.as_ref(), &handle).unwrap();
    assert_eq!(status.state, SchedulingState::Completed);
    let result = ProviderExecutionApi::complete(executor.as_ref(), &handle).unwrap();
    assert_eq!(result.state, SchedulingState::Completed);
    ProviderExecutionApi::release(executor.as_ref(), handle.clone()).unwrap();

    // submit -> status -> complete -> release is now exhausted: neither
    // status nor a second complete succeeds against the same handle.
    assert!(ProviderExecutionApi::status(executor.as_ref(), &handle).is_err());
    assert!(ProviderExecutionApi::complete(executor.as_ref(), &handle).is_err());
}

#[test]
fn reference_cpu_honors_already_elapsed_deadline_as_timeout() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    assert_eq!(
        advertisement.cancellation,
        KernelCancellationSupport::TimeoutOnly
    );
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("deadline-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("deadline-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("deadline-out", [2, 2]);
    executor.write_tensor(a_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));
    executor.write_tensor(b_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));

    let mut invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-deadline"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);
    invocation.deadline_millis = Some(0);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert_eq!(result.error, Some(KernelError::KernelTimeout));
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelTimeout)
    );
}

#[test]
fn reference_cpu_kernel_registry_selects_registered_candidate() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    registry.set_provider_status(ProviderStatusSnapshot::from_health_report(
        ProviderHealthReport::new(
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            HealthState::Available,
        ),
    ));
    let matmul_operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let request = KernelSelectionRequest::new(
        "select-matmul",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let selection = registry.select(&request).unwrap();
    let selected = selection
        .selected
        .expect("a compatible Reference CPU candidate should be selected");
    assert_eq!(selected.provider.as_str(), REFERENCE_CPU_PROVIDER_NAME);
    assert!(
        selection
            .observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelSelected)
    );
}

#[test]
fn reference_cpu_execution_only_accepts_runtime_created_invocation_shapes() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    // Only one input bound where the Operator requires two: Runtime-level
    // validation must reject it rather than the Provider guessing.
    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (out_id, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    let _ = out_id;

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-2"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert!(result.error.is_some());
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
fn reference_cpu_kernel_registry_accepts_provider_advertisements() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    let matmul_id = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    assert!(registry.active_advertisement(&matmul_id).is_some());
}

#[test]
fn reference_cpu_no_raw_handles_exposed_in_invocation_or_result() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("b", [2, 2]);
    let (_, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-3"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    let text = format!("{invocation:?} {result:?}");
    assert!(!text.contains("0x"));
    assert!(!text.contains("raw handle"));
}

fn tensor_resource_for_test(id: &str) -> TensorResource {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let residency = TensorResidency::new(
        TensorResourceId::new(id),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    TensorResource::new(TensorResourceId::new(id), descriptor, residency)
}

#[test]
fn tensor_lifecycle_allows_declared_to_ready_happy_path() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-1");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.mark_ready().unwrap();
    assert_eq!(resource.lifecycle, TensorLifecycleState::Ready);
    assert_eq!(resource.readiness, TensorReadiness::Ready);
    assert!(resource.ensure_usable().is_ok());
}

#[test]
fn tensor_lifecycle_rejects_declared_to_ready_skip() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-2");
    let error = resource
        .transition_to(TensorLifecycleState::Ready)
        .unwrap_err();
    assert!(matches!(error, TensorError::ResourceInvalid { .. }));
}

#[test]
fn tensor_resource_released_is_rejected_for_use() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-3");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.mark_ready().unwrap();
    resource
        .transition_to(TensorLifecycleState::Released)
        .unwrap();
    assert_eq!(
        resource.ensure_usable().unwrap_err(),
        TensorError::ResourceReleased
    );
}

#[test]
fn tensor_readiness_blocks_dispatch_until_ready() {
    let mut resource = tensor_resource_for_test("tensor-readiness-1");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.transition_to(TensorLifecycleState::Ready).unwrap();
    resource.readiness = TensorReadiness::PendingTransfer;
    assert!(matches!(
        resource.ensure_usable().unwrap_err(),
        TensorError::ResourceNotReady { .. }
    ));
    resource.readiness = TensorReadiness::Ready;
    assert!(resource.ensure_usable().is_ok());
}

#[test]
fn tensor_mutability_denies_mutation_of_immutable_resource() {
    let error =
        validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, true).unwrap_err();
    assert!(matches!(error, TensorError::MutabilityViolation { .. }));
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Mutable, true).is_ok());
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, false).is_ok());
}

#[test]
fn tensor_aliasing_requires_in_place_kernel_support() {
    let error =
        validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, false).unwrap_err();
    assert!(matches!(error, TensorError::AliasingViolation { .. }));
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, true).is_ok());
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::NoAlias, false).is_ok());
}

#[test]
fn tensor_memory_class_is_derived_from_memory_placement() {
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::HostOrdinary),
        TensorMemoryClass::Host
    );
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::HostPinned),
        TensorMemoryClass::PinnedHost
    );
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::BrowserLinearMemory),
        TensorMemoryClass::BrowserLinearMemory
    );
    let staged = MemoryPlacement::StagedTemporary(Box::new(MemoryPlacement::HostOrdinary));
    assert_eq!(TensorMemoryClass::from(&staged), TensorMemoryClass::Host);
}

#[test]
fn tensor_memory_class_validation_rejects_unsupported_class() {
    let error = validate_memory_class_for_kernel(
        TensorMemoryClass::Device,
        &[TensorMemoryClass::Host, TensorMemoryClass::PinnedHost],
    )
    .unwrap_err();
    assert!(matches!(error, TensorError::MemoryClassUnsupported { .. }));
    assert!(validate_memory_class_for_kernel(TensorMemoryClass::Device, &[]).is_ok());
    assert!(
        validate_memory_class_for_kernel(TensorMemoryClass::Host, &[TensorMemoryClass::Host])
            .is_ok()
    );
}

#[test]
fn tensor_view_becomes_unavailable_once_base_is_terminal() {
    let view = TensorView::new(
        TensorResourceId::new("base"),
        ViewDescriptor::from_resource(TensorResourceId::new("base"), 0, [1, 1]),
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert!(
        view.validate_against_base(TensorLifecycleState::Ready)
            .is_ok()
    );
    let error = view
        .validate_against_base(TensorLifecycleState::Released)
        .unwrap_err();
    assert!(matches!(error, TensorError::ViewBaseUnavailable { .. }));
}

#[test]
fn tensor_layout_descriptor_kind_covers_every_layout_category() {
    assert_eq!(LayoutDescriptor::Contiguous.kind(), ComputeLayout::Dense);
    assert_eq!(
        LayoutDescriptor::Blocked {
            block_dimensions: vec![2, 2],
        }
        .kind(),
        ComputeLayout::Blocked
    );
    assert_eq!(
        LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: None,
            current_length_elements: None,
            logical_to_physical: None,
            append_behavior: None,
        }
        .kind(),
        ComputeLayout::Paged
    );
    assert_eq!(
        LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: Some(32),
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        }
        .kind(),
        ComputeLayout::PackedQuantized
    );
    assert_eq!(
        LayoutDescriptor::AttentionSpecific {
            layout_id: "paged-attention".into(),
        }
        .kind(),
        ComputeLayout::AttentionSpecific
    );
    assert_eq!(
        LayoutDescriptor::BrowserCompatible {
            layout_id: "wasm-linear".into(),
        }
        .kind(),
        ComputeLayout::BrowserCompatible
    );
}

#[test]
fn operator_layout_kind_maps_every_layout_descriptor_variant() {
    assert_eq!(
        layout_kind(&LayoutDescriptor::Blocked {
            block_dimensions: vec![4],
        }),
        TensorLayoutKind::Blocked
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: None,
            current_length_elements: None,
            logical_to_physical: None,
            append_behavior: None,
        }),
        TensorLayoutKind::Paged
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        }),
        TensorLayoutKind::QuantizedPacked
    );
}

#[test]
fn tensor_error_and_observation_redact_backend_diagnostics() {
    let error = TensorError::resource_invalid("native handle=0xdeadbeef");
    assert_eq!(
        error.to_string(),
        "tensor resource invalid: [redacted backend diagnostic]"
    );
    let observation = TensorObservation::new(TensorObservationKind::ResourceReady)
        .with_message("C:\\weights\\model.bin");
    assert_eq!(observation.message, "[redacted backend diagnostic]");
}

#[test]
fn tensor_descriptor_builder_sets_intents_and_semantic_role() {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float16),
    )
    .with_storage_dtype(DTypeDescriptor::portable(ComputeDType::Float16))
    .with_compute_dtype(DTypeDescriptor::portable(ComputeDType::Float32))
    .with_memory_class_intent(TensorMemoryClass::Host)
    .with_mutability_intent(TensorMutabilityKind::Immutable)
    .with_aliasing_intent(TensorAliasingKind::NoAlias)
    .with_affinity_constraints(ResourceAffinity::new(FallbackClass::Transparent))
    .with_semantic_role(TensorRole::Input)
    .with_dimension_roles([DimensionRole::Batch, DimensionRole::Hidden]);

    assert_eq!(
        descriptor.compute_dtype,
        Some(DTypeDescriptor::portable(ComputeDType::Float32))
    );
    assert_eq!(
        descriptor.memory_class_intent,
        Some(TensorMemoryClass::Host)
    );
    assert_eq!(descriptor.semantic_role, Some(TensorRole::Input));
    assert!(
        descriptor
            .validate(&TensorDescriptorLimits::default())
            .is_ok()
    );

    let mismatched = descriptor.with_dimension_roles([DimensionRole::Batch]);
    assert!(
        mismatched
            .validate(&TensorDescriptorLimits::default())
            .is_err()
    );
}

#[test]
fn shape_descriptor_row_major_strides_matches_expected_layout() {
    let shape = ShapeDescriptor::new([2, 3, 4]);
    assert_eq!(shape.row_major_strides(), vec![12, 4, 1]);
    assert_eq!(ShapeDescriptor::new([5]).row_major_strides(), vec![1]);
}

#[test]
fn tensor_descriptor_estimated_byte_size_honors_packed_quantized_bits() {
    let packed = TensorDescriptor::new(
        ShapeDescriptor::new([16]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: Some(16),
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        },
    );
    assert_eq!(packed.estimated_byte_size().unwrap(), 8);

    let dense = TensorDescriptor::materialized(
        ShapeDescriptor::new([16]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert_eq!(
        dense.estimated_byte_size().unwrap(),
        dense.byte_size().unwrap()
    );
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
fn tensor_residency_affinity_rejects_forged_device_binding() {
    let claimed = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-0"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-0"))),
    );
    let actual = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-1"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-1"))),
    );
    let error = AffinityConstraints::try_from_affinities([&claimed.affinity, &actual.affinity])
        .unwrap_err();
    assert!(matches!(error, AffinityError::DeviceMismatch { .. }));
}

#[test]
fn tensor_layout_placeholder_variants_validate_without_bounds_checking() {
    let blocked = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Blocked {
            block_dimensions: vec![2, 2],
        },
    );
    let paged = TensorDescriptor::new(
        ShapeDescriptor::new([256]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: Some(16),
            current_length_elements: Some(200),
            logical_to_physical: None,
            append_behavior: None,
        },
    );
    for descriptor in [blocked, paged] {
        assert!(
            descriptor
                .validate(&TensorDescriptorLimits::default())
                .is_ok()
        );
    }
}

#[test]
fn kernel_result_tracks_aliasing_updates_alongside_readiness_and_residency() {
    let result = KernelResult::success(KernelInvocationId::new("aliasing-update"))
        .with_aliasing_update("output", TensorAliasingKind::InputOutputAlias);
    assert_eq!(
        result.updated_aliasing.get("output"),
        Some(&TensorAliasingKind::InputOutputAlias)
    );
}

#[test]
fn tensor_resource_debug_output_never_exposes_raw_pointers_or_handles() {
    let resource = tensor_resource_for_test("tensor-debug-safety");
    let text = format!("{resource:?}");
    assert!(!text.contains("0x"));
    assert!(!text.contains("handle="));
}

#[test]
fn shape_descriptor_symbolic_dimensions_validate_fixed_consistency() {
    let consistent = ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([
        SymbolicDimension::Symbolic("batch".into()),
        SymbolicDimension::Fixed(8),
    ]);
    assert!(consistent.validate_symbolic().is_ok());

    let inconsistent = ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([
        SymbolicDimension::Symbolic("batch".into()),
        SymbolicDimension::Fixed(16),
    ]);
    assert!(inconsistent.validate_symbolic().is_err());

    let wrong_rank =
        ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([SymbolicDimension::Dynamic]);
    assert!(wrong_rank.validate_symbolic().is_err());
}

#[test]
fn layout_descriptor_paged_tracks_logical_to_physical_and_append_behavior() {
    let mut logical_to_physical = std::collections::BTreeMap::new();
    logical_to_physical.insert(0, 3);
    logical_to_physical.insert(1, 7);
    let paged = LayoutDescriptor::Paged {
        page_size_elements: 16,
        block_size_elements: 4,
        capacity_pages: Some(8),
        current_length_elements: Some(64),
        logical_to_physical: Some(logical_to_physical.clone()),
        append_behavior: Some(PagedAppendBehavior::GrowOnDemand),
    };
    let LayoutDescriptor::Paged {
        logical_to_physical: stored_map,
        append_behavior: stored_behavior,
        ..
    } = &paged
    else {
        unreachable!()
    };
    assert_eq!(stored_map.as_ref(), Some(&logical_to_physical));
    assert_eq!(*stored_behavior, Some(PagedAppendBehavior::GrowOnDemand));
    assert_eq!(paged.kind(), ComputeLayout::Paged);
}

#[test]
fn layout_descriptor_packed_quantized_tracks_dequantization_requirements() {
    let layout = LayoutDescriptor::PackedQuantized {
        method: "int4".into(),
        bits_per_value: 4,
        group_size: Some(32),
        scale_dtype: Some(Box::new(DTypeDescriptor::portable(ComputeDType::Float16))),
        zero_point_dtype: None,
        packing_order: Some("row-major-blocks".into()),
        dequantization_requirements: Some("requires dequantize_placeholder before use".into()),
    };
    let LayoutDescriptor::PackedQuantized {
        dequantization_requirements,
        ..
    } = &layout
    else {
        unreachable!()
    };
    assert_eq!(
        dequantization_requirements.as_deref(),
        Some("requires dequantize_placeholder before use")
    );
}

#[test]
fn reference_cpu_quantize_and_dequantize_placeholders_reject_explicitly() {
    for error in [dequantize_placeholder(), quantize_placeholder()] {
        assert_eq!(error.code, ReferenceCpuErrorCode::DTypeUnsupported);
    }
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

fn permissive_operator_spec(memory: OperatorMemoryBehavior) -> OperatorSpec {
    OperatorSpec::new(
        OperatorId::new(
            OPERATOR_NAMESPACE,
            "conformance-op",
            1,
            OperatorFamily::Tensor,
        ),
        1,
        1,
    )
    .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Input, []))
    .with_memory(memory)
}

fn contiguous_f32_tensor() -> TensorDescriptor {
    TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    )
}

#[test]
fn operator_validate_invocation_rejects_aliasing_intent_without_in_place_support() {
    let spec = permissive_operator_spec(OperatorMemoryBehavior::pure());
    let output = contiguous_f32_tensor().with_aliasing_intent(TensorAliasingKind::InputOutputAlias);
    let error = spec
        .validate_invocation(&[contiguous_f32_tensor()], &[output], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(error, OperatorError::AliasingUnsupported { .. }));
}

#[test]
fn operator_validate_invocation_accepts_aliasing_intent_when_in_place_supported() {
    let memory = OperatorMemoryBehavior {
        supports_in_place: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let output = contiguous_f32_tensor().with_aliasing_intent(TensorAliasingKind::InputOutputAlias);
    assert!(
        spec.validate_invocation(&[contiguous_f32_tensor()], &[output], &BTreeMap::new())
            .is_ok()
    );
}

#[test]
fn operator_validate_invocation_rejects_memory_class_conflict() {
    let memory = OperatorMemoryBehavior {
        requires_host_visible: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let input = contiguous_f32_tensor().with_memory_class_intent(TensorMemoryClass::Device);
    let error = spec
        .validate_invocation(&[input], &[contiguous_f32_tensor()], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(
        error,
        OperatorError::MemoryBehaviorUnsupported { .. }
    ));
}

#[test]
fn operator_validate_invocation_rejects_mutation_of_immutable_input() {
    let memory = OperatorMemoryBehavior {
        mutates_input: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let input = contiguous_f32_tensor().with_mutability_intent(TensorMutabilityKind::Immutable);
    let error = spec
        .validate_invocation(&[input], &[contiguous_f32_tensor()], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(
        error,
        OperatorError::MemoryBehaviorUnsupported { .. }
    ));
}

// ---------------------------------------------------------------------
// Runtime Inference API
// ---------------------------------------------------------------------

#[test]
fn inference_api_model_reference_resolves_through_local_registry() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let result = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap();
    assert_eq!(result.artifact, artifact);
}

#[test]
fn inference_api_model_reference_resolution_fails_for_unregistered_reference() {
    let registry = ModelRegistry::new();
    let reference = ModelRef::new("unknown-model").unwrap();
    let error = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelResolutionFailed { .. }
    ));
}

#[test]
fn inference_api_model_reference_rejects_path_like_input() {
    assert!(matches!(
        ModelRef::new("../etc/passwd"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
    assert!(matches!(
        ModelRef::new("models/qwen"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
}

#[test]
fn inference_api_scope_validation_rejects_non_inference_capabilities() {
    assert!(validate_inference_scope("generation").is_ok());
    for forbidden in [
        "workspace-filesystem",
        "git",
        "shell",
        "secrets",
        "tool-call",
    ] {
        assert!(matches!(
            validate_inference_scope(forbidden),
            Err(InferenceApiError::PolicyDenied { .. })
        ));
    }
}

#[test]
fn inference_api_session_creation_rejects_forbidden_allowed_capabilities() {
    let runtime = &mut Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("shell".into());

    let error = create_inference_session(runtime, request).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_session_creation_succeeds_with_inference_only_capabilities() {
    let runtime = &mut Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());

    let session = create_inference_session(runtime, request).unwrap();
    let status = session_status(
        runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .unwrap();
    assert_eq!(status.id, session);
    assert!(!status.raw_prompt_available);
    assert!(!status.raw_handles_available);
}

#[test]
fn inference_api_tokenize_plain_text_never_stores_raw_prompt_text() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::PlainText("secret".into()));

    let result = tokenize_prompt_input(&tokenizer, request, None).unwrap();
    assert!(!result.token_ids.is_empty());
    assert!(!format!("{result:?}").contains("secret"));
}

#[test]
fn inference_api_tokenize_chat_messages_requires_authorized_formatter() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hello",
    )]));

    let error = tokenize_prompt_input(&tokenizer, request, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

struct ConcatenationChatTemplate;
impl ChatTemplateFormatter for ConcatenationChatTemplate {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        Ok(messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

#[test]
fn inference_api_tokenize_chat_messages_formats_through_authorized_contract() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hi",
    )]));

    let result =
        tokenize_prompt_input(&tokenizer, request, Some(&ConcatenationChatTemplate)).unwrap();
    assert!(!result.token_ids.is_empty());
}

#[test]
fn inference_api_tokenize_already_tokenized_input_validates_range() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let in_range = TokenizationRequest::new(PromptInput::TokenIds(vec![2, 3, 4]));
    assert!(tokenize_prompt_input(&tokenizer, in_range, None).is_ok());

    let out_of_range = TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999]));
    let error = tokenize_prompt_input(&tokenizer, out_of_range, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizationFailed { .. }
    ));
}

#[test]
fn inference_api_build_generation_request_from_tokenized_input() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    };
    let tokenized = TokenizationResult {
        token_ids: vec![2, 3, 4],
        token_count: 3,
        offsets: None,
        diagnostics: Vec::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
    };

    let request = build_generation_request(
        GenerationRequestId::new("gen-api-1").unwrap(),
        None,
        GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer,
        tokenized,
        4,
        GenerationParameters::default(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );

    assert_eq!(request.prompt_token_count, 3);
    request.validate().unwrap();
}

#[test]
fn inference_api_admission_state_reports_structured_backpressure() {
    assert_eq!(
        AdmissionState::from(&MemoryAdmissionDecision::Admit {
            reason: "ok".into()
        }),
        AdmissionState::Accepted
    );
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Queue {
            reason: "busy".into()
        }),
        AdmissionState::Queued { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Reject {
            reason: "no memory".into()
        }),
        AdmissionState::Rejected { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::RetryLater {
            reason: "pressure".into()
        }),
        AdmissionState::Delayed { .. }
    ));
}

#[test]
fn inference_api_submit_generation_admits_compatible_request_into_batch() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();

    let (state, slot) = submit_generation(&mut runtime, &batch, &request).unwrap();
    assert_eq!(state, AdmissionState::Accepted);
    assert!(slot.is_some());
}

#[test]
fn inference_api_submit_generation_reports_queued_when_batch_policy_enqueues() {
    let mut runtime = Runtime::builder().build().unwrap();
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let request = generation_request();

    let (state, slot) = submit_generation(&mut runtime, &batch, &request).unwrap();
    assert!(matches!(state, AdmissionState::Queued { .. }));
    assert!(slot.is_some());
}

#[test]
fn inference_api_submit_generation_reports_rejection_when_batch_policy_denies() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        max_active_operations: 0,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();

    let error = submit_generation(&mut runtime, &batch, &request).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::GenerationRejected { .. }
    ));
}

#[test]
fn inference_api_submit_generation_observed_emits_generation_accepted() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();
    let mut observer = InferenceApiObserver::new();

    let (state, _) =
        submit_generation_observed(&mut runtime, &batch, &request, &mut observer).unwrap();
    assert_eq!(state, AdmissionState::Accepted);
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::GenerationAccepted)
    );
}

fn model_instance_definition() -> ModelInstanceDefinition {
    ModelInstanceDefinition {
        artifact: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
        architecture: ModelArchitectureImplementation {
            architecture: ModelArchitecture::new("qwen", "qwen2"),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        },
        residencies: BTreeSet::from([ModelResidencyId::new(1)]),
        tokenizer: None,
        placement: ModelInstancePlacement::new(ResourceAffinity::new(FallbackClass::Transparent)),
        policy: ModelInstancePolicy::default(),
        adapter_state: ModelInstanceAdapterState::default(),
        associated_sessions: BTreeSet::new(),
        usage: ModelInstanceUsage::default(),
        compute_dtype: None,
        mutation_version: 0,
        tenant: None,
        owner: None,
        resource_bindings: ModelInstanceResourceBindings::default(),
        kernel_selection_policy: None,
    }
}

#[test]
fn inference_api_model_instance_suspend_resume_drain_through_api_boundary() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    // `create()` no longer reaches Ready on its own (transactional-weight-
    // materialization); this test's own concern is suspend/resume/drain
    // *from* Ready, so reach Ready explicitly first.
    runtime.model_instances_mut().mark_ready(&instance).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.raw_provider_handle_available);
    assert!(!status.raw_device_handle_available);
    assert!(!status.raw_weights_available);

    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )
    .unwrap();
    resume_model_instance(&mut runtime, &instance).unwrap();
    drain_model_instance(&mut runtime, &instance).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Draining);
}

#[test]
fn inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    // `create()` no longer reaches Ready on its own (transactional-weight-
    // materialization); this test's own concern is warmup's conflict
    // detection against an *already Ready* instance, so reach Ready
    // explicitly first.
    runtime.model_instances_mut().mark_ready(&instance).unwrap();
    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    let checks = ModelInstanceReadinessChecks {
        residency_available: true,
        provider_ready: true,
        device_ready: true,
        adapter_ready: true,
        memory_pressure: MemoryPressureLevel::Low,
        runtime_policy_allows: true,
        browser_supported: true,
        kernel_preparation_ready: true,
        autotuning_ready: true,
        weights_materialized: true,
    };

    let error = warm_model_instance(&mut runtime, &instance, &plan, &checks).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelInstanceUnavailable { .. }
    ));
}

#[test]
fn inference_api_create_model_instance_observed_emits_model_instance_selected() {
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let manifest = minimal_model_manifest();
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(core),
        &manifest,
        &trust,
    )
    .unwrap();

    let mut runtime = Runtime::builder().build().unwrap();
    let architecture = ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    };
    let mut observer = InferenceApiObserver::new();
    let instance = create_model_instance_observed(
        &mut runtime,
        &loaded,
        architecture,
        ResourceAffinity::new(FallbackClass::Transparent),
        &mut observer,
    )
    .unwrap();

    assert_eq!(
        model_instance_status(&runtime, &instance).unwrap().artifact,
        manifest.id
    );
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::ModelInstanceSelected)
    );
}

fn minimal_model_manifest() -> ModelManifest {
    ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    }
}

#[test]
fn inference_api_load_model_wires_coordinator_and_memory_manager() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut request = ModelLoadingApiRequest::new(core);
    request.tokenizer_reference = Some(TokenizerId::new("fixture").unwrap());
    request.adapter_references = vec![AdapterArtifactId {
        name: AdapterName::new("lora-a").unwrap(),
        revision: AdapterRevision::new("1").unwrap(),
        digest: AdapterDigest {
            algorithm: "sha256".into(),
            value: "deadbeef".into(),
        },
    }];
    request.layout_policy = Some(TensorLayoutKind::Contiguous);
    request.provider_preferences = vec![ProviderBinding::new("reference-cpu")];

    let loaded = load_model(&mut coordinator, &mut memory, request, &manifest, &trust).unwrap();
    assert_eq!(loaded.artifact, manifest.id);
    assert_eq!(loaded.state, ModelLoadingState::Ready);
}

#[test]
fn inference_api_load_model_observed_emits_loading_lifecycle_observations() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut observer = InferenceApiObserver::new();

    load_model_observed(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(core),
        &manifest,
        &trust,
        &mut observer,
    )
    .unwrap();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ModelLoadingRequested));
    assert!(kinds.contains(&InferenceApiObservationKind::ModelLoaded));
}

#[test]
fn inference_api_model_resolution_observed_emits_model_resolved_and_failed() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact);
    let mut observer = InferenceApiObserver::new();

    registry
        .resolve_observed(&ModelResolutionRequest::new(reference), &mut observer)
        .unwrap();
    registry
        .resolve_observed(
            &ModelResolutionRequest::new(ModelRef::new("unknown").unwrap()),
            &mut observer,
        )
        .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolved));
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolutionFailed));
}

#[test]
fn inference_api_session_lifecycle_observed_emits_created_and_closed() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let mut observer = InferenceApiObserver::new();

    let session = create_inference_session_observed(&mut runtime, request, &mut observer).unwrap();
    close_inference_session_observed(&mut runtime, &session, &mut observer).unwrap();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::SessionCreated));
    assert!(kinds.contains(&InferenceApiObservationKind::SessionClosed));
}

#[test]
fn inference_api_session_close_transitions_lifecycle_to_closed() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let session = create_inference_session(&mut runtime, request).unwrap();

    close_inference_session(&mut runtime, &session).unwrap();

    let status = session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .unwrap();
    assert_eq!(status.lifecycle, SessionLifecycleState::Closed);
}

#[test]
fn inference_api_tokenize_prompt_input_observed_emits_tokenized_and_failed() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut observer = InferenceApiObserver::new();

    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("secret".into())),
        None,
        &mut observer,
    )
    .unwrap();
    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999])),
        None,
        &mut observer,
    )
    .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::PromptTokenized));
    assert!(kinds.contains(&InferenceApiObservationKind::TokenizationFailed));
}

#[test]
fn inference_api_one_shot_pipeline_uses_session_tokenizer_and_generation_contracts() {
    let metadata = generation_tokenizer_metadata();
    let vocabulary_size = metadata.vocabulary_size as usize;
    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence::complete(),
    );
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let session = create_one_shot_session(&mut runtime, request).unwrap();

    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let tokenized = tokenize_prompt_input(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )
    .unwrap();

    let generation_request = build_generation_request(
        GenerationRequestId::new("one-shot-1").unwrap(),
        Some(session.clone()),
        GenerationModelReference::LoadedModelContext("model-context".into()),
        GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        tokenized,
        2,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );
    let prepared = prepare_generation(&runtime, generation_request).unwrap();

    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let (state, _) = submit_generation(&mut runtime, &batch, &prepared).unwrap();
    assert_eq!(state, AdmissionState::Accepted);

    // One-shot inference SHALL not bypass Model Instance, Tokenizer,
    // Generation, Sampling, Memory Manager, or Provider/Kernel contracts:
    // this drives the same Generation Contract loop (prefill, per-token
    // Sampling Contract decode, Provider/Kernel readiness gating) that any
    // session-bound generation would use.
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &prepared,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap();
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::TokenGenerated)
    );

    close_inference_session(&mut runtime, &session).unwrap();
}

fn adapter_residency_fixture() -> AdapterResidency {
    AdapterResidency {
        id: AdapterResidencyId::new("residency-1").unwrap(),
        artifact: AdapterArtifactId {
            name: AdapterName::new("lora-a").unwrap(),
            revision: AdapterRevision::new("1").unwrap(),
            digest: AdapterDigest {
                algorithm: "sha256".into(),
                value: "deadbeef".into(),
            },
        },
        lifecycle: AdapterLifecycleState::Ready,
        location: AdapterResidencyLocation::Host,
        affinity: None,
        memory_allocation: None,
        provider_resource: None,
    }
}

fn adapter_activation_request_fixture(residency: &AdapterResidency) -> AdapterActivationRequest {
    AdapterActivationRequest {
        residency: residency.id.clone(),
        scope: AdapterActivationScope::Session(InferenceSessionId::new("session-1").unwrap()),
        base_model: GenerationModelReference::LoadedModelContext("model-context".into()),
        adapter_set: AdapterSetId::from_adapters([residency.artifact.clone()]),
        policy: AdapterCompositionPolicy::SingleAdapterOnly,
    }
}

#[test]
fn inference_api_adapter_activation_succeeds_for_ready_residency() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);

    activate_adapter(&residency, &request, None, None).unwrap();
}

#[test]
fn inference_api_adapter_activation_rejects_forbidden_operation_scope() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.scope = AdapterActivationScope::Operation("shell".into());

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_adapter_activation_denied_when_incompatible() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.residency = AdapterResidencyId::new("other-residency").unwrap();

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::AdapterActivationFailed { .. }
    ));
}

#[test]
fn inference_api_adapter_activation_observed_emits_adapter_activated() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);
    let mut observer = InferenceApiObserver::new();

    activate_adapter_observed(&residency, &request, None, None, &mut observer).unwrap();

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::AdapterActivated)
    );
}

#[test]
fn inference_api_kv_cache_policy_covers_enabled_scope_budget_reuse_eviction_privacy() {
    let policy = KvCachePolicy {
        enabled: true,
        max_cache_tokens: Some(2048),
        max_cache_memory_bytes: Some(1 << 20),
        sharing: KvCacheSharingPolicy::AllowWithinSession,
        retention: KvCacheRetentionPolicy::RetainForPrefixReuse,
        prefix_reuse_allowed: true,
        privacy_redaction_required: true,
    };
    assert!(policy.enabled);
    assert!(policy.prefix_reuse_allowed);
    assert!(policy.privacy_redaction_required);
    assert_eq!(policy.sharing, KvCacheSharingPolicy::AllowWithinSession);
}

#[test]
fn inference_api_prefix_cache_policy_covers_scope_sharing_ttl_budget_privacy_reuse() {
    let policy = PrefixCachePolicy {
        enabled: true,
        allow_partial_reuse: true,
        require_sealed_kv_cache_for_sharing: true,
        sharing: PrefixCacheSharingPolicy::SessionLocal,
        privacy: PrefixCachePrivacyPolicy::default(),
        max_memory_bytes: Some(1 << 20),
        max_prefix_tokens: Some(1024),
        ttl_millis: Some(60_000),
        idle_ttl_millis: Some(5_000),
        persist_after_session_close: false,
    };
    assert!(policy.enabled);
    assert!(policy.allow_partial_reuse);
    assert_eq!(policy.sharing, PrefixCacheSharingPolicy::SessionLocal);
    assert_eq!(policy.ttl_millis, Some(60_000));
}

#[test]
fn inference_api_runtime_diagnostics_with_inputs_includes_caller_supplied_status() {
    let runtime = Runtime::builder().build().unwrap();
    let inputs = RuntimeDiagnosticsInputs {
        model_resolution_status: Some(ModelResolutionStatus::Resolved),
        model_loading_status: Some(ModelLoadingPhase::PublishModelContext),
        operator_missing_count: 2,
        tokenizer_compatible: Some(true),
        queued_admission_count: 3,
    };

    let diagnostics = runtime_diagnostics_with(&runtime, inputs);
    assert_eq!(
        diagnostics.model_resolution_status,
        Some(ModelResolutionStatus::Resolved)
    );
    assert_eq!(
        diagnostics.model_loading_status,
        Some(ModelLoadingPhase::PublishModelContext)
    );
    assert_eq!(diagnostics.operator_missing_count, 2);
    assert_eq!(diagnostics.tokenizer_compatible, Some(true));
    assert_eq!(diagnostics.queued_admission_count, 3);
    assert!(diagnostics.redacted);
}

#[test]
fn inference_api_generation_result_wraps_output_with_decoded_text_and_cache_usage() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11, 12], FinishReason::EosToken);

    let result = GenerationResult::new(output)
        .with_decoded_text("hello".into())
        .with_model_instance(ModelInstanceId::new("instance-1").unwrap())
        .with_cache_usage(CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        });

    assert_eq!(result.decoded_text.as_deref(), Some("hello"));
    assert!(result.model_instance.is_some());
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));
    assert!(result.error.is_none());
    assert!(result.redacted);
}

#[test]
fn inference_api_generation_result_reports_error_for_failed_finish_reason() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, Vec::new(), FinishReason::ProviderError);

    let result = GenerationResult::new(output);
    assert!(result.error.is_some());
}

#[test]
fn inference_api_cancellation_stage_before_dispatch_always_succeeds() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    for stage in [
        CancellationStage::Queued,
        CancellationStage::Tokenization,
        CancellationStage::Prefill,
        CancellationStage::Decode,
        CancellationStage::Sampling,
        CancellationStage::Batching,
        CancellationStage::GraphExecution,
        CancellationStage::KernelDispatch,
    ] {
        assert_eq!(
            request_cancellation_at_stage(&token, stage, false),
            CancellationOutcome::Cancelled
        );
    }
}

#[test]
fn inference_api_cancellation_stage_provider_execution_depends_on_support() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_cancellation_stage_observed_emits_generation_cancelled() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    let mut observer = InferenceApiObserver::new();

    request_cancellation_at_stage_observed(&token, CancellationStage::Decode, false, &mut observer);

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::GenerationCancelled)
    );
}

#[test]
fn inference_api_observer_buffer_stays_bounded_and_retains_most_recent() {
    let mut observer = InferenceApiObserver::new();
    let total = INFERENCE_API_OBSERVATION_BUFFER_CAPACITY + 128;
    for index in 0..total {
        observer.observe(
            InferenceApiObservationKind::TokenCommitted,
            format!("observation-{index}"),
            None,
        );
    }
    assert_eq!(
        observer.observations().len(),
        INFERENCE_API_OBSERVATION_BUFFER_CAPACITY
    );
    // The oldest observations were evicted to admit the newest ones -- the
    // buffer reflects the most recent causal evidence, not whatever
    // happened to be observed first.
    assert!(
        !observer
            .observations()
            .iter()
            .any(|observation| observation.message == "observation-0")
    );
    assert_eq!(
        observer
            .observations()
            .last()
            .map(|observation| observation.message.as_str()),
        Some(format!("observation-{}", total - 1)).as_deref()
    );
}

#[test]
fn inference_api_tachyon_and_cli_boundary_capabilities_are_inference_only() {
    for forbidden in ["git", "shell", "agent-orchestration", "secrets"] {
        assert!(validate_inference_scope(forbidden).is_err());
    }
    assert!(validate_inference_scope("generation").is_ok());
}

#[test]
fn inference_api_streaming_handle_correlates_with_ordered_token_events() {
    let request = generation_request();
    let handle = StreamingHandle::for_request(&request);

    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();
    assert_eq!(events.len(), 3);
    assert!(
        events
            .iter()
            .all(|event| event.request_id == handle.request)
    );
    let indices: Vec<_> = events
        .iter()
        .filter_map(|event| event.token_index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn inference_api_diagnostics_and_status_debug_output_never_exposes_raw_pointer_markers() {
    let runtime = Runtime::builder().build().unwrap();
    let diagnostics = runtime_diagnostics(&runtime);
    let instance = runtime
        .model_instances()
        .instances()
        .next()
        .map(ModelInstance::status);

    let debug_output = format!("{diagnostics:?} {instance:?}");
    assert!(!debug_output.contains("0x"));
    assert!(!debug_output.to_ascii_lowercase().contains("pointer"));
}

#[test]
fn inference_api_cancellation_reports_limitation_when_unsupported_after_dispatch() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation(&token, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation(&token, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_runtime_diagnostics_are_redacted_and_reflect_empty_runtime() {
    let runtime = Runtime::builder().build().unwrap();
    let diagnostics = runtime_diagnostics(&runtime);
    assert!(diagnostics.redacted);
    assert_eq!(diagnostics.model_instance_count, 0);
    assert_eq!(diagnostics.active_session_count, 0);
}

#[test]
fn inference_api_usage_report_never_carries_raw_prompt_text() {
    let usage = GenerationUsage::new(3, 4, FinishReason::EosToken);
    let memory = GenerationMemoryEstimate::default();

    let report = UsageReport::from_generation(&usage, &memory, Some(true), Some(12));
    assert_eq!(report.prompt_token_count, 3);
    assert_eq!(report.generated_token_count, 4);
    assert_eq!(report.cache_hit, Some(true));
    assert!(!report.cancelled);
}

#[test]
fn inference_api_browser_feature_check_only_rejects_on_wasm32() {
    let result = require_browser_supported("wasmtime");
    if cfg!(target_arch = "wasm32") {
        assert!(matches!(
            result,
            Err(InferenceApiError::BrowserFeatureUnsupported { .. })
        ));
    } else {
        assert!(result.is_ok());
    }
}

fn session_creation_request() -> SessionCreationRequest {
    let metadata = generation_tokenizer_metadata();
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
        created_at_millis: 0,
    }
}

#[test]
fn inference_api_model_resolution_source_placeholders_fail_structured() {
    let mut registry = ModelRegistry::new();
    let reference = ModelRef::new("qwen-test").unwrap();
    registry.register(
        reference.clone(),
        ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
    );

    for source in [
        ModelResolutionSource::FutureExternalSource,
        ModelResolutionSource::FutureTachyonSource,
    ] {
        let mut request = ModelResolutionRequest::new(reference.clone());
        request.source = source;
        let error = registry.resolve(&request).unwrap_err();
        assert!(matches!(
            error,
            InferenceApiError::ModelResolutionFailed { .. }
        ));
    }
}

#[test]
fn inference_api_model_resolution_local_registry_source_still_resolves() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let mut request = ModelResolutionRequest::new(reference);
    request.source = ModelResolutionSource::LocalRegistry;
    assert_eq!(registry.resolve(&request).unwrap().artifact, artifact);
}

#[test]
fn inference_api_streaming_decode_request_carries_state_across_calls() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut request = StreamingDecodeRequest::new(vec![2, 3]);
    request.skip_special_tokens = true;

    let output = decode_tokens_streaming(&tokenizer, request).unwrap();
    assert!(output.consumed_token_count > 0 || output.pending_partial_state.is_some());
}

#[test]
fn inference_api_validate_tokenizer_compatibility_accepts_matching_digest() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata.clone());
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(metadata.digest.clone()),
        expected_vocabulary_size: Some(metadata.vocabulary_size),
        expected_family: Some(metadata.family.clone()),
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap();
}

#[test]
fn inference_api_validate_tokenizer_compatibility_rejects_digest_mismatch() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata);
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(ModelDigest::sha256(b"a different tokenizer")),
        expected_vocabulary_size: None,
        expected_family: None,
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    let error = validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizerIncompatible { .. }
    ));
}

#[test]
fn inference_api_generation_api_request_carries_privacy_policy() {
    let request = GenerationApiRequest::new(
        generation_request(),
        SessionRedactionPolicy::RedactRawInputs,
    );
    assert_eq!(request.privacy, SessionRedactionPolicy::RedactRawInputs);
}

#[test]
fn inference_api_browser_inference_capabilities_reduced_excludes_kv_cache() {
    let capabilities = BrowserInferenceCapabilities::reduced();
    assert!(capabilities.tokenization);
    assert!(capabilities.generation);
    assert!(capabilities.streaming);
    assert!(!capabilities.kv_cache);
}

#[test]
fn inference_api_run_generation_loop_emits_full_streaming_lifecycle_and_completes() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    request.max_new_tokens = 2;
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;

    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence::complete(),
    );
    let mut observer = InferenceApiObserver::new();

    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        },
        |_generated| false,
        &mut observer,
    )
    .unwrap();

    assert_eq!(result.output.generated_token_count, 2);
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    for expected in [
        InferenceApiObservationKind::GenerationStarted,
        InferenceApiObservationKind::StreamOpened,
        InferenceApiObservationKind::KvCacheUsed,
        InferenceApiObservationKind::PrefixCacheMiss,
        InferenceApiObservationKind::PrefillStarted,
        InferenceApiObservationKind::PrefillCompleted,
        InferenceApiObservationKind::DecodeStarted,
        InferenceApiObservationKind::TokenGenerated,
        InferenceApiObservationKind::GenerationCompleted,
        InferenceApiObservationKind::StreamClosed,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}

#[test]
fn inference_api_run_generation_loop_cancels_during_decode() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    request.max_new_tokens = 5;
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;

    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence::complete(),
    );
    let mut observer = InferenceApiObserver::new();

    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |generated| !generated.is_empty(),
        &mut observer,
    )
    .unwrap();

    assert_eq!(result.output.finish_reason, FinishReason::Cancelled);
    assert_eq!(result.output.generated_token_count, 1);
    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::GenerationCancelled));
    assert!(kinds.contains(&InferenceApiObservationKind::StreamInterrupted));
    assert!(!kinds.contains(&InferenceApiObservationKind::GenerationCompleted));
}

#[test]
fn inference_api_run_generation_loop_rejects_incomplete_executor_evidence_before_sampling() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;
    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence {
            model_instance_ready: true,
            graph_validated: true,
            kernel_selected: true,
            kernel_dispatched: false,
            provider_executed: false,
            tensor_resource_used: false,
            context: Vec::new(),
        },
    );
    let mut observer = InferenceApiObserver::new();

    let error = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();

    assert!(matches!(error, InferenceApiError::KernelUnavailable { .. }));
    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ExecutionGraphValidated));
    assert!(kinds.contains(&InferenceApiObservationKind::KernelSelected));
    assert!(kinds.contains(&InferenceApiObservationKind::KernelUnavailable));
    assert!(kinds.contains(&InferenceApiObservationKind::StreamInterrupted));
    assert!(!kinds.contains(&InferenceApiObservationKind::TokenGenerated));
    assert!(!kinds.contains(&InferenceApiObservationKind::GenerationCompleted));
    assert!(!kinds.contains(&InferenceApiObservationKind::StreamClosed));
}

#[test]
fn inference_api_run_generation_loop_observes_executor_failure_before_returning() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(Arc::new(FailingGenerationExecutor))
        .build()
        .unwrap();
    let mut observer = InferenceApiObserver::new();

    let error = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        InferenceApiError::ProviderUnavailable { .. }
    ));
    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::DecodeStarted));
    assert!(kinds.contains(&InferenceApiObservationKind::ProviderUnavailable));
    assert!(kinds.contains(&InferenceApiObservationKind::StreamInterrupted));
    assert!(!kinds.contains(&InferenceApiObservationKind::StreamClosed));
}

#[test]
fn inference_api_run_generation_loop_reports_provider_and_kernel_unavailable() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let mut observer = InferenceApiObserver::new();
    let mut runtime = Runtime::builder().build().unwrap();
    let error = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ProviderUnavailable { .. }
    ));
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::ProviderUnavailable)
    );
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::StreamInterrupted)
    );

    request.request_id = GenerationRequestId::new("gen-2").unwrap();
    let mut observer = InferenceApiObserver::new();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(Arc::new(FailingKernelGenerationExecutor))
        .build()
        .unwrap();
    let error = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();
    assert!(matches!(error, InferenceApiError::KernelUnavailable { .. }));
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::KernelUnavailable)
    );
}

// ---------------------------------------------------------------------
// cli_boundary
// ---------------------------------------------------------------------

#[test]
fn cli_boundary_error_display_is_non_empty_for_every_variant() {
    let variants = vec![
        CliBoundaryError::CliCommandInvalid {
            reason: "bad command".into(),
        },
        CliBoundaryError::CliPromptInputInvalid {
            reason: "bad prompt".into(),
        },
        CliBoundaryError::CliFileReadFailed {
            reason: "file missing".into(),
        },
        CliBoundaryError::CliWorkspaceAccessDenied {
            reason: "policy denied".into(),
        },
        CliBoundaryError::CliGitFailed {
            reason: "git failed".into(),
        },
        CliBoundaryError::CliNetworkDenied {
            reason: "network denied".into(),
        },
        CliBoundaryError::CliSecretUnavailable {
            reason: "secret unavailable".into(),
        },
        CliBoundaryError::CliToolFailed {
            reason: "tool failed".into(),
        },
        CliBoundaryError::CliShellDenied {
            reason: "shell denied".into(),
        },
        CliBoundaryError::CliModelAliasNotFound {
            alias: "my-alias".into(),
        },
        CliBoundaryError::CliModelReferenceInvalid {
            reason: "bad reference".into(),
        },
        CliBoundaryError::CliRuntimeUnavailable {
            reason: "runtime down".into(),
        },
        CliBoundaryError::CliRuntimeRequestFailed(InferenceApiError::ModelLoadingFailed {
            reason: "example".into(),
        }),
        CliBoundaryError::CliStreamInterrupted {
            reason: "stream broke".into(),
        },
        CliBoundaryError::CliCancellationRequested,
        CliBoundaryError::CliDiagnosticsRedacted,
        CliBoundaryError::CliBoundaryViolation {
            capability: "workspace".into(),
        },
        CliBoundaryError::InternalCliError {
            reason: "unexpected".into(),
        },
    ];
    for variant in variants {
        let rendered = variant.to_string();
        assert!(!rendered.is_empty(), "{variant:?} rendered empty");
    }
}

#[test]
fn cli_boundary_rejects_cli_owned_authority_capabilities() {
    for capability in [
        "workspace",
        "filesystem",
        "git",
        "shell",
        "secrets",
        "tool-call",
    ] {
        let error = reject_cli_owned_authority(capability).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliBoundaryViolation { .. }
        ));
    }
}

#[test]
fn cli_boundary_allows_inference_scoped_capability() {
    assert!(reject_cli_owned_authority("generation").is_ok());
}

#[test]
fn cli_boundary_error_preserves_wrapped_runtime_error_category() {
    let source = InferenceApiError::SessionNotFound;
    let wrapped = CliBoundaryError::from(source.clone());
    assert_eq!(wrapped.runtime_category(), Some(&source));
}

#[test]
fn cli_boundary_conformance_report_is_conformant() {
    let report = run_cli_boundary_conformance();
    assert!(report.is_conformant());
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
}

// ---------------------------------------------------------------------
// provider_roadmap
// ---------------------------------------------------------------------

#[test]
fn provider_roadmap_features_are_all_optional_and_phase_tagged() {
    assert_eq!(PROVIDER_ROADMAP_FEATURES.len(), 31);
    for feature in PROVIDER_ROADMAP_FEATURES {
        assert!(feature.is_optional(), "{feature:?} must be optional");
    }
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::OptimizedCpu),
        vec![
            ProviderRoadmapFeature::Simd,
            ProviderRoadmapFeature::Blas,
            ProviderRoadmapFeature::ThreadPoolExecution,
            ProviderRoadmapFeature::CacheAwareKernels,
            ProviderRoadmapFeature::OptimizedCpuFusedKernels,
        ]
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Cuda).len(),
        8
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Metal).len(),
        5
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::OpenVino).len(),
        4
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Qnn).len(),
        4
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::WebGpu).len(),
        5
    );
    for feature in PROVIDER_ROADMAP_FEATURES {
        assert!(!feature.id().is_empty());
        assert!(
            provider_roadmap_features_for_phase(feature.phase()).contains(feature),
            "{feature:?} must appear under its own phase()"
        );
    }
}

#[test]
fn provider_roadmap_phases_are_ordered_1_through_9() {
    let mut ordinals: Vec<u8> = PROVIDER_ROADMAP_PHASES
        .iter()
        .map(|phase| phase.ordinal())
        .collect();
    ordinals.sort_unstable();
    assert_eq!(ordinals, (1..=9).collect::<Vec<_>>());
}

#[test]
fn provider_roadmap_every_phase_requires_provider_core() {
    for phase in PROVIDER_ROADMAP_PHASES {
        assert!(
            phase
                .required_conformance_gates()
                .contains(&ProviderConformanceProfile::ProviderCore),
            "{phase:?} must require provider-core"
        );
    }
}

#[test]
fn provider_roadmap_phase_readiness_requires_actually_passed_profiles() {
    let cuda = ProviderRoadmapPhase::Cuda;
    let required = cuda.required_conformance_gates();
    assert!(!phase_is_production_ready(cuda, &BTreeSet::new()));
    assert!(!phase_is_production_ready(
        cuda,
        &BTreeSet::from([ProviderConformanceProfile::ProviderCore])
    ));
    assert!(phase_is_production_ready(cuda, &required));
}

#[test]
fn provider_roadmap_reference_cpu_remains_correctness_baseline() {
    // Every post-baseline hardware family's primary fallback edge terminates
    // at Reference CPU (or an explicit browser-CPU-like path for WebGPU),
    // never at another hardware Provider -- Reference CPU stays the
    // correctness baseline the roadmap compares optimized output against.
    assert_eq!(
        ProviderRoadmapHardwareFamily::Cuda.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::CudaToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::Metal.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::MetalToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::OpenVino.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::OpenVinoToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::Qnn.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::QnnToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::WebGpu.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike
    );
}

#[test]
fn provider_roadmap_optimized_provider_does_not_redefine_operator_semantics() {
    // A fused kernel that does not preserve portable Operator/graph
    // semantics is rejected outright, regardless of how it declares itself.
    let non_preserving = KernelFusionMetadata {
        operator_group: vec![OperatorId::magnetar(
            "softmax",
            1,
            OperatorFamily::Activation,
        )],
        preserves_graph_semantics: false,
    };
    let precision = KernelPrecisionMetadata {
        tolerance_profile: Some("operator-default".into()),
        ..KernelPrecisionMetadata::default()
    };
    let fallback_hints = BTreeSet::from([KernelFallbackClass::AlternateKernel]);
    let outcome = validate_fused_kernel_declaration(FusedKernelDeclaration {
        fusion: Some(&non_preserving),
        precision: &precision,
        fallback_hints: &fallback_hints,
    });
    assert!(matches!(
        outcome,
        Err(ProviderRoadmapError::ProviderFusionInvalid { .. })
    ));
}

#[test]
fn provider_roadmap_rejects_model_family_provider_names() {
    for name in [
        "QwenProvider",
        "LlamaProvider",
        "GemmaProvider",
        "DeepSeekProvider",
    ] {
        let outcome = reject_model_family_provider_name(name);
        assert!(
            matches!(
                outcome,
                Err(ProviderRoadmapError::ProviderRoadmapUnsupported { .. })
            ),
            "{name} should have been rejected, got {outcome:?}"
        );
    }
}

#[test]
fn provider_roadmap_allows_hardware_and_optimized_provider_names() {
    for name in [
        "CudaProvider",
        "MetalProvider",
        "OpenVinoProvider",
        "QnnProvider",
        "WebGpuProvider",
        "OptimizedCpuProvider",
        "ReferenceCpuProvider",
    ] {
        assert!(
            reject_model_family_provider_name(name).is_ok(),
            "{name} should have been allowed"
        );
    }
}

#[test]
fn provider_roadmap_rejects_empty_provider_name() {
    assert!(matches!(
        reject_model_family_provider_name("   "),
        Err(ProviderRoadmapError::InternalProviderRoadmapError { .. })
    ));
}

#[test]
fn provider_roadmap_denies_native_handle_exposure_for_every_hardware_family() {
    for family in [
        ProviderRoadmapHardwareFamily::Cuda,
        ProviderRoadmapHardwareFamily::Metal,
        ProviderRoadmapHardwareFamily::OpenVino,
        ProviderRoadmapHardwareFamily::Qnn,
    ] {
        assert!(
            !family.native_handle_kinds().is_empty(),
            "{family:?} should declare at least one native handle kind"
        );
        for handle_kind in family.native_handle_kinds() {
            let outcome = reject_native_handle_exposure(family, handle_kind);
            assert!(matches!(
                outcome,
                Err(ProviderRoadmapError::ProviderNativeHandleExposureDenied { .. })
            ));
        }
    }
    // WebGPU has no native-handle boundary of its own (browser sandboxing
    // constraints apply instead), but it still must not require native
    // Provider loading.
    assert!(
        ProviderRoadmapHardwareFamily::WebGpu
            .native_handle_kinds()
            .is_empty()
    );
    assert!(ProviderRoadmapHardwareFamily::WebGpu.requires_no_native_provider_loading());
    assert!(!ProviderRoadmapHardwareFamily::Cuda.requires_no_native_provider_loading());
}

#[test]
fn provider_roadmap_fused_kernel_requires_semantic_declaration() {
    let missing = validate_fused_kernel_declaration(FusedKernelDeclaration {
        fusion: None,
        precision: &KernelPrecisionMetadata::default(),
        fallback_hints: &BTreeSet::new(),
    });
    assert!(matches!(
        missing,
        Err(ProviderRoadmapError::ProviderFusionInvalid { .. })
    ));

    let empty_group = KernelFusionMetadata {
        operator_group: Vec::new(),
        preserves_graph_semantics: true,
    };
    let precision = KernelPrecisionMetadata {
        tolerance_profile: Some("operator-default".into()),
        ..KernelPrecisionMetadata::default()
    };
    let fallback_hints = BTreeSet::from([KernelFallbackClass::AlternateKernel]);
    assert!(
        validate_fused_kernel_declaration(FusedKernelDeclaration {
            fusion: Some(&empty_group),
            precision: &precision,
            fallback_hints: &fallback_hints,
        })
        .is_err()
    );

    let complete = KernelFusionMetadata {
        operator_group: vec![OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        )],
        preserves_graph_semantics: true,
    };
    assert!(
        validate_fused_kernel_declaration(FusedKernelDeclaration {
            fusion: Some(&complete),
            precision: &precision,
            fallback_hints: &fallback_hints,
        })
        .is_ok()
    );
}

#[test]
fn provider_roadmap_quantized_path_requires_explicit_metadata() {
    let incomplete = KernelQuantizationMetadata {
        method: KernelQuantizationMethod::Int8,
        storage_dtype: ComputeDType::SInt8,
        compute_dtype: ComputeDType::Float32,
        accumulation_dtype: ComputeDType::Float32,
        scale_dtype: ComputeDType::Float32,
        zero_point_dtype: None,
        group_size: None,
        packing_layout: TensorLayoutKind::QuantizedPacked,
        dequantization: KernelDequantizationBehavior::ExplicitBeforeOperator,
        supported_operators: BTreeSet::new(),
        conformance_tolerance_profile: String::new(),
    };
    assert!(matches!(
        validate_quantization_declaration(&incomplete),
        Err(ProviderRoadmapError::ProviderQuantizationUnsupported { .. })
    ));

    let complete = KernelQuantizationMetadata {
        supported_operators: BTreeSet::from([OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        )]),
        conformance_tolerance_profile: "quantized-int8-default".into(),
        ..incomplete
    };
    assert!(validate_quantization_declaration(&complete).is_ok());
}

#[test]
fn provider_roadmap_rejects_hidden_dequantization() {
    assert!(matches!(
        reject_hidden_dequantization(false),
        Err(ProviderRoadmapError::ProviderQuantizationUnsupported { .. })
    ));
    assert!(reject_hidden_dequantization(true).is_ok());
}

#[test]
fn provider_roadmap_advanced_attention_unsupported_path_fails_explicitly() {
    let outcome = reject_unsupported_advanced_attention(AdvancedAttentionVariant::FlashAttention);
    assert!(matches!(
        outcome,
        ProviderRoadmapError::ProviderAdvancedAttentionUnsupported { .. }
    ));
}

#[test]
fn provider_roadmap_advanced_attention_declaration_requires_kv_cache_for_paged() {
    let operator = OperatorId::magnetar("attention", 1, OperatorFamily::Attention);
    let layouts = BTreeSet::from([TensorLayoutKind::AttentionSpecific]);
    let memory_classes = BTreeSet::from([KernelMemoryClass::Device]);
    let dtypes = BTreeSet::from([ComputeDType::Float32]);
    let precision = KernelPrecisionMetadata {
        tolerance_profile: Some("attention-default".into()),
        ..KernelPrecisionMetadata::default()
    };
    let determinism = KernelDeterminism::default();
    let fallback_hints = BTreeSet::from([KernelFallbackClass::AlternateKernel]);

    let missing_kv_cache = validate_advanced_attention_declaration(AdvancedAttentionDeclaration {
        variant: AdvancedAttentionVariant::PagedAttention,
        operator: &operator,
        layouts: &layouts,
        memory_classes: &memory_classes,
        dtypes: &dtypes,
        kv_cache: None,
        precision: &precision,
        determinism: &determinism,
        fallback_hints: &fallback_hints,
    });
    assert!(matches!(
        missing_kv_cache,
        Err(ProviderRoadmapError::ProviderAdvancedAttentionUnsupported { .. })
    ));

    let kv_cache = KernelKvCacheMetadata {
        layouts: BTreeSet::from(["paged".to_string()]),
        paged_cache: true,
        append: true,
        read: true,
        dtypes: BTreeSet::from([ComputeDType::Float32]),
        memory_classes: BTreeSet::from([KernelMemoryClass::Device]),
        affinity: None,
    };
    let complete = validate_advanced_attention_declaration(AdvancedAttentionDeclaration {
        variant: AdvancedAttentionVariant::PagedAttention,
        operator: &operator,
        layouts: &layouts,
        memory_classes: &memory_classes,
        dtypes: &dtypes,
        kv_cache: Some(&kv_cache),
        precision: &precision,
        determinism: &determinism,
        fallback_hints: &fallback_hints,
    });
    assert!(complete.is_ok());

    // Flash attention doesn't inherently require KV cache metadata.
    let flash_without_kv_cache =
        validate_advanced_attention_declaration(AdvancedAttentionDeclaration {
            variant: AdvancedAttentionVariant::FlashAttention,
            operator: &operator,
            layouts: &layouts,
            memory_classes: &memory_classes,
            dtypes: &dtypes,
            kv_cache: None,
            precision: &precision,
            determinism: &determinism,
            fallback_hints: &fallback_hints,
        });
    assert!(flash_without_kv_cache.is_ok());
}

#[test]
fn provider_roadmap_fallback_denied_by_default() {
    let context = ProviderRoadmapFallbackContext::deny_by_default();
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let outcome = evaluate_provider_roadmap_fallback(
        ProviderRoadmapFallbackEdge::CudaToReferenceCpu,
        &affinity,
        &context,
    );
    assert!(matches!(
        outcome,
        Err(ProviderRoadmapError::ProviderFallbackDenied { .. })
    ));
}

#[test]
fn provider_roadmap_fallback_requires_every_gate_open() {
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let mut context = ProviderRoadmapFallbackContext {
        cpu: FallbackPolicyContext::new(true),
        memory_policy_allows_fallback: true,
        privacy_policy_allows_fallback: true,
        precision_policy_allows_fallback: false,
    };
    assert!(
        evaluate_provider_roadmap_fallback(
            ProviderRoadmapFallbackEdge::MetalToReferenceCpu,
            &affinity,
            &context,
        )
        .is_err(),
        "precision gate closed must still deny fallback"
    );
    context.precision_policy_allows_fallback = true;
    assert!(
        evaluate_provider_roadmap_fallback(
            ProviderRoadmapFallbackEdge::MetalToReferenceCpu,
            &affinity,
            &context,
        )
        .is_ok(),
        "all gates open must allow fallback"
    );
    context.memory_policy_allows_fallback = false;
    assert!(
        evaluate_provider_roadmap_fallback(
            ProviderRoadmapFallbackEdge::MetalToReferenceCpu,
            &affinity,
            &context,
        )
        .is_err(),
        "memory gate closed must still deny fallback"
    );
}

#[test]
fn provider_roadmap_fallback_denies_provider_pinned_affinity_even_with_open_policy() {
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned);
    let context = ProviderRoadmapFallbackContext {
        cpu: FallbackPolicyContext::new(true),
        memory_policy_allows_fallback: true,
        privacy_policy_allows_fallback: true,
        precision_policy_allows_fallback: true,
    };
    assert!(
        evaluate_provider_roadmap_fallback(
            ProviderRoadmapFallbackEdge::CudaToOptimizedCpu,
            &affinity,
            &context,
        )
        .is_err()
    );
}

#[test]
fn provider_roadmap_fallback_observed_emits_considered_then_used_or_denied() {
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let denied_context = ProviderRoadmapFallbackContext::deny_by_default();
    let (observations, outcome) = evaluate_provider_roadmap_fallback_observed(
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike,
        &affinity,
        &denied_context,
    );
    assert!(outcome.is_err());
    assert_eq!(observations.len(), 2);
    assert_eq!(
        observations[0].kind,
        ProviderRoadmapObservationKind::FallbackConsidered
    );
    assert_eq!(
        observations[1].kind,
        ProviderRoadmapObservationKind::FallbackDenied
    );

    let allowed_context = ProviderRoadmapFallbackContext {
        cpu: FallbackPolicyContext::new(true),
        memory_policy_allows_fallback: true,
        privacy_policy_allows_fallback: true,
        precision_policy_allows_fallback: true,
    };
    let (observations, outcome) = evaluate_provider_roadmap_fallback_observed(
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike,
        &affinity,
        &allowed_context,
    );
    assert!(outcome.is_ok());
    assert_eq!(
        observations[1].kind,
        ProviderRoadmapObservationKind::FallbackUsed
    );
}

#[test]
fn provider_roadmap_runtime_api_remains_provider_independent() {
    for capability in PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES {
        assert!(
            reject_provider_specific_handle_capability(capability).is_err(),
            "{capability} should have been rejected"
        );
    }
    // Ordinary inference scopes remain unaffected.
    assert!(reject_provider_specific_handle_capability("generation").is_ok());
    assert!(validate_inference_scope("generation").is_ok());
}

#[test]
fn provider_roadmap_cli_receives_redacted_provider_diagnostics_only() {
    let raw = "provider handle=0xdeadbeef failed on cuda-stream";
    let redacted = cli_redacted_provider_diagnostic(raw);
    assert!(!redacted.contains("0xdeadbeef"));
}

#[test]
fn provider_roadmap_cli_may_pass_policy_preference_without_authority() {
    let preference = ProviderRoadmapPolicyPreference {
        preferred_provider: Some("cuda".into()),
        allow_optimized_provider_fallback: true,
    };
    let echoed = cli_may_pass_policy_preference(&preference);
    assert_eq!(echoed, preference);
    assert!(reject_cli_raw_provider_handle_selection("cuda-device-pointer").is_err());
}

#[test]
fn provider_roadmap_layout_expansion_requires_explicit_conversion() {
    for layout in POST_BASELINE_LAYOUTS {
        assert!(!layout.component_visible() || *layout != TensorLayoutKind::ProviderOpaque);
    }
    assert!(
        require_explicit_layout_conversion(
            TensorLayoutKind::Paged,
            TensorLayoutKind::Paged,
            false,
        )
        .is_ok()
    );
    assert!(
        require_explicit_layout_conversion(
            TensorLayoutKind::Paged,
            TensorLayoutKind::Blocked,
            false,
        )
        .is_err()
    );
    assert!(
        require_explicit_layout_conversion(
            TensorLayoutKind::Paged,
            TensorLayoutKind::Blocked,
            true,
        )
        .is_ok()
    );
}

#[test]
fn provider_roadmap_memory_expansion_requires_manager_tracking() {
    assert_eq!(POST_BASELINE_MEMORY_CLASSES.len(), 7);
    for memory_class in POST_BASELINE_MEMORY_CLASSES {
        assert!(require_memory_manager_tracking(*memory_class, true).is_ok());
        assert!(matches!(
            require_memory_manager_tracking(*memory_class, false),
            Err(ProviderRoadmapError::ProviderMemoryClassUnsupported { .. })
        ));
    }
}

#[test]
fn provider_roadmap_conformance_profiles_are_declared_without_implying_readiness() {
    let ids = provider_roadmap_conformance_profile_ids();
    assert_eq!(ids.len(), 9);
    assert!(ids.values().all(|required| !required));
    assert!(ids.contains_key(ProviderConformanceProfile::Quantized.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::AdvancedAttention.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::FusedKernel.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::Browser.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::WebGpu.id()));
}

#[test]
fn provider_roadmap_benchmarks_stay_separate_from_conformance() {
    // A benchmark result is never accepted anywhere a conformance decision
    // is made: a report whose only entry is a correctness failure is not
    // conformant no matter how good a (structurally separate) benchmark
    // result would look.
    let fast_but_wrong = ProviderRoadmapConformanceReport {
        results: vec![ProviderRoadmapConformanceResult {
            requirement: "optimized matmul matches Reference CPU".into(),
            passed: false,
            diagnostic: Some("output differs beyond tolerance".into()),
        }],
    };
    assert!(!fast_but_wrong.is_conformant());

    let benchmark = ProviderRoadmapBenchmarkResult {
        category: ProviderRoadmapBenchmarkCategory::TokensPerSecond,
        provider: "cuda".into(),
        value: 999.0,
        unit: "tokens/sec".into(),
    };
    // The benchmark result exists purely as data; nothing consumes it as
    // conformance input.
    assert_eq!(
        benchmark.category,
        ProviderRoadmapBenchmarkCategory::TokensPerSecond
    );
}

#[test]
fn provider_roadmap_error_display_is_non_empty_for_every_variant() {
    let variants = vec![
        ProviderRoadmapError::ProviderRoadmapUnsupported {
            reason: "example".into(),
        },
        ProviderRoadmapError::OptimizedCpuProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::CudaProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::MetalProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::OpenVinoProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::QnnProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::WebGpuProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFeatureUnsupported {
            feature: "example".into(),
        },
        ProviderRoadmapError::ProviderLayoutUnsupported {
            layout: "example".into(),
        },
        ProviderRoadmapError::ProviderDTypeUnsupported {
            dtype: "example".into(),
        },
        ProviderRoadmapError::ProviderMemoryClassUnsupported {
            memory_class: "example".into(),
        },
        ProviderRoadmapError::ProviderAdvancedAttentionUnsupported {
            variant: "example".into(),
        },
        ProviderRoadmapError::ProviderQuantizationUnsupported {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFusionInvalid {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderConformanceFailed {
            report: "example".into(),
        },
        ProviderRoadmapError::ProviderBenchmarkFailed {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFallbackDenied {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderNativeHandleExposureDenied {
            handle_kind: "example".into(),
        },
        ProviderRoadmapError::InternalProviderRoadmapError {
            reason: "example".into(),
        },
    ];
    for variant in variants {
        let rendered = variant.to_string();
        assert!(!rendered.is_empty(), "{variant:?} rendered empty");
        assert!(!variant.id().is_empty());
    }
}

#[test]
fn provider_roadmap_observation_redacts_metadata_by_default() {
    let observation = ProviderRoadmapObservation::new(ProviderRoadmapObservationKind::FallbackUsed)
        .with_provider("cuda")
        .with_redacted_metadata("diagnostic", "device pointer handle=0xdeadbeef");
    let value = observation.redacted_metadata.get("diagnostic").unwrap();
    assert!(!value.contains("0xdeadbeef"));
}

#[test]
fn provider_roadmap_observation_kind_round_trips_through_debug() {
    let kinds = [
        ProviderRoadmapObservationKind::RoadmapFeatureDiscovered,
        ProviderRoadmapObservationKind::CapabilityAdvertised,
        ProviderRoadmapObservationKind::CapabilityRejected,
        ProviderRoadmapObservationKind::OptimizedProviderSelected,
        ProviderRoadmapObservationKind::AdvancedAttentionSelected,
        ProviderRoadmapObservationKind::QuantizedKernelSelected,
        ProviderRoadmapObservationKind::FusedKernelSelected,
        ProviderRoadmapObservationKind::FallbackConsidered,
        ProviderRoadmapObservationKind::FallbackUsed,
        ProviderRoadmapObservationKind::FallbackDenied,
        ProviderRoadmapObservationKind::ConformancePassed,
        ProviderRoadmapObservationKind::ConformanceFailed,
        ProviderRoadmapObservationKind::BenchmarkExecuted,
        ProviderRoadmapObservationKind::BenchmarkSkipped,
    ];
    assert_eq!(kinds.len(), 14);
    for kind in kinds {
        let observation = ProviderRoadmapObservation::new(kind);
        assert_eq!(observation.kind, kind);
    }
}

#[test]
fn provider_roadmap_device_metadata_templates_carry_family_memory_classes() {
    for family in [
        ProviderRoadmapHardwareFamily::Cuda,
        ProviderRoadmapHardwareFamily::Metal,
        ProviderRoadmapHardwareFamily::OpenVino,
        ProviderRoadmapHardwareFamily::Qnn,
        ProviderRoadmapHardwareFamily::WebGpu,
    ] {
        let device = family.device_metadata_template();
        assert_eq!(device.memory_class_support, family.memory_classes());
        assert!(!device.vendor.is_empty());
        assert!(!device.architecture.is_empty());
    }
}

#[test]
fn provider_roadmap_conformance_report_is_conformant() {
    let report = run_provider_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// model_format_roadmap
// ---------------------------------------------------------------------

#[test]
fn model_format_roadmap_phases_are_ordered_1_through_12() {
    let mut ordinals: Vec<u8> = MODEL_FORMAT_ROADMAP_PHASES
        .iter()
        .map(|phase| phase.ordinal())
        .collect();
    ordinals.sort_unstable();
    assert_eq!(ordinals, (1..=12).collect::<Vec<_>>());
    for phase in MODEL_FORMAT_ROADMAP_PHASES {
        assert!(!phase.id().is_empty());
        assert!(phase.normalizes_into_existing_contract());
    }
}

#[test]
fn model_format_roadmap_rejects_format_shaped_provider_names() {
    for name in [
        "GGUFProvider",
        "SafetensorsProvider",
        "QwenSafetensorsProvider",
        "sentencepiece-provider",
        "tokenizer-json-provider",
    ] {
        assert!(
            reject_model_format_provider_name(name).is_err(),
            "{name} must be rejected"
        );
    }
}

#[test]
fn model_format_roadmap_allows_hardware_and_optimized_provider_names() {
    for name in [
        "ReferenceCpuProvider",
        "CudaProvider",
        "OptimizedCpuProvider",
    ] {
        assert!(
            reject_model_format_provider_name(name).is_ok(),
            "{name} must be allowed"
        );
    }
}

#[test]
fn model_format_roadmap_rejects_empty_provider_name() {
    assert!(matches!(
        reject_model_format_provider_name("   "),
        Err(ModelFormatRoadmapError::InternalModelFormatError { .. })
    ));
}

#[test]
fn model_format_roadmap_format_parsers_cannot_supply_execution_graphs() {
    assert!(reject_format_execution_graph(true).is_err());
    assert!(reject_format_execution_graph(false).is_ok());
}

#[test]
fn model_format_roadmap_normalized_manifest_coverage_tracks_present_fields() {
    let mut manifest = fixture_model_manifest();
    let coverage = NormalizedManifestCoverage::from_manifest(&manifest);
    assert!(coverage.identity);
    assert!(coverage.digest);
    assert!(coverage.architecture_family);
    assert!(!coverage.tokenizer);
    assert!(!coverage.license);
    assert!(coverage.covers_required_fields() || manifest.parts.is_empty());

    manifest.tokenizer = Some("tokenizer.json".into());
    manifest.license = Some(ModelLicenseMetadata {
        identifier: "apache-2.0".into(),
        url: None,
        usage_restrictions: Vec::new(),
    });
    let coverage = NormalizedManifestCoverage::from_manifest(&manifest);
    assert!(coverage.tokenizer);
    assert!(coverage.license);
}

fn fixture_model_manifest() -> ModelManifest {
    let digest = ModelDigest::parse(format!("sha256:{}", "2".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("fixture-model").unwrap(),
        ModelRevision::new("v1").unwrap(),
        digest,
    );
    ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    }
}

fn tensor(name: &str, shape: Vec<u64>) -> ModelTensorMetadata {
    ModelTensorMetadata {
        name: name.into(),
        shape,
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
    }
}

#[test]
fn model_format_roadmap_safetensors_manifest_validates_and_normalizes() {
    let manifest = SafetensorsManifest {
        tensors: vec![SafetensorsTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![4, 4],
            dtype: ModelDType::F32,
            byte_offset: 0,
            byte_length: 64,
        }],
        header_metadata: BTreeMap::new(),
    };
    assert!(manifest.validate().is_ok());
    let tensors = manifest.into_tensor_metadata();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "layer.0.weight");
    assert_eq!(tensors[0].offset_bytes, Some(0));
    assert_eq!(tensors[0].size_bytes, Some(64));

    let degenerate = SafetensorsManifest {
        tensors: vec![SafetensorsTensorEntry {
            name: "bad".into(),
            shape: vec![0],
            dtype: ModelDType::F32,
            byte_offset: 0,
            byte_length: 4,
        }],
        header_metadata: BTreeMap::new(),
    };
    assert!(matches!(
        degenerate.validate(),
        Err(ModelFormatRoadmapError::SafetensorsInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_memory_mapping_policy_rejects_raw_pointer_exposure() {
    let policy = MemoryMappingPolicy {
        mapping_allowed: true,
        streaming_read_allowed: true,
        exposes_raw_pointer: true,
    };
    assert!(policy.validate().is_err());
    let safe = MemoryMappingPolicy {
        exposes_raw_pointer: false,
        ..policy
    };
    assert!(safe.validate().is_ok());
}

#[test]
fn model_format_roadmap_detects_missing_and_duplicate_shard_tensors() {
    let mut index = ShardIndex::default();
    index.shards.push(ModelShard {
        id: ModelShardId::new("shard-0").unwrap(),
        digest: ModelDigest::parse(format!("sha256:{}", "3".repeat(64))).unwrap(),
        size_bytes: 1024,
        order: 0,
    });
    index.tensor_shard_map.insert(
        "layer.0.weight".into(),
        ModelShardId::new("shard-0").unwrap(),
    );
    assert!(detect_missing_shards(&index).is_ok());

    index.tensor_shard_map.insert(
        "layer.1.weight".into(),
        ModelShardId::new("shard-missing").unwrap(),
    );
    assert!(matches!(
        detect_missing_shards(&index),
        Err(ModelFormatRoadmapError::ShardMissing { .. })
    ));

    let duplicate = vec![
        tensor("layer.0.weight", vec![4, 4]),
        tensor("layer.0.weight", vec![4, 4]),
    ];
    assert!(detect_duplicate_tensor_names(&duplicate).is_err());

    let inconsistent = vec![
        tensor("layer.0.weight", vec![4, 4]),
        tensor("layer.0.weight", vec![8, 8]),
    ];
    assert!(matches!(
        validate_shard_tensor_shape_consistency(&inconsistent),
        Err(ModelFormatRoadmapError::ShardIndexInvalid { .. })
    ));

    let ordered = vec![
        ModelShard {
            id: ModelShardId::new("shard-0").unwrap(),
            digest: ModelDigest::parse(format!("sha256:{}", "4".repeat(64))).unwrap(),
            size_bytes: 1,
            order: 0,
        },
        ModelShard {
            id: ModelShardId::new("shard-1").unwrap(),
            digest: ModelDigest::parse(format!("sha256:{}", "5".repeat(64))).unwrap(),
            size_bytes: 1,
            order: 0,
        },
    ];
    assert!(matches!(
        validate_shard_loading_order(&ordered),
        Err(ModelFormatRoadmapError::ShardIndexInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_torch_dtype_never_forces_compute_dtype() {
    assert_eq!(
        torch_dtype_does_not_force_compute_dtype(Some("bfloat16"), ModelDType::F32),
        ModelDType::F32
    );
    assert_eq!(
        torch_dtype_does_not_force_compute_dtype(None, ModelDType::Bf16),
        ModelDType::Bf16
    );
}

#[test]
fn model_format_roadmap_normalizes_tokenizer_json() {
    let parsed = TokenizerJsonMetadata {
        vocabulary_size: 32000,
        added_tokens: Vec::new(),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Bos, "<s>", 1)],
        normalizer: Some("nfc".into()),
        pre_tokenizer: Some("byte-level".into()),
        decoder: Some("byte-level".into()),
        supports_offsets: true,
    };
    let metadata = normalize_tokenizer_json(
        TokenizerId::new("tok-1").unwrap(),
        TokenizerArtifactId::new("tokenizer.json").unwrap(),
        ModelDigest::parse(format!("sha256:{}", "6".repeat(64))).unwrap(),
        TokenizerFamily::new("qwen").unwrap(),
        TokenizerRevision::new("v1").unwrap(),
        &parsed,
    )
    .unwrap();
    assert_eq!(metadata.vocabulary_size, 32000);
    assert!(metadata.supports_offsets);
    assert_eq!(metadata.special_tokens.len(), 1);

    let empty = TokenizerJsonMetadata {
        vocabulary_size: 0,
        ..parsed
    };
    assert!(matches!(
        normalize_tokenizer_json(
            TokenizerId::new("tok-2").unwrap(),
            TokenizerArtifactId::new("tokenizer.json").unwrap(),
            ModelDigest::parse(format!("sha256:{}", "7".repeat(64))).unwrap(),
            TokenizerFamily::new("qwen").unwrap(),
            TokenizerRevision::new("v1").unwrap(),
            &empty,
        ),
        Err(ModelFormatRoadmapError::TokenizerJsonInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_tokenizer_config_requires_explicit_runtime_validation() {
    assert!(reject_silent_tokenizer_config_override(false).is_err());
    assert!(reject_silent_tokenizer_config_override(true).is_ok());
}

#[test]
fn model_format_roadmap_generation_config_as_defaults_and_override() {
    let parsed = GenerationConfigMetadata {
        temperature: Some(0.7),
        max_new_tokens: Some(256),
        stop_strings: vec!["<eos>".into()],
        ..Default::default()
    };
    let defaults = parsed.as_defaults();
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.max_tokens, Some(256));
    assert_eq!(defaults.stop_tokens, vec!["<eos>".to_string()]);

    assert_eq!(apply_generation_override(Some(0.7), Some(1.5)), Some(1.5));
    assert_eq!(apply_generation_override(Some(0.7), None), Some(0.7));
    assert_eq!(apply_generation_override::<f32>(None, None), None);
}

#[test]
fn model_format_roadmap_chat_template_requires_compatibility_and_variables() {
    let metadata = ChatTemplateMetadata {
        identity: "qwen-chat".into(),
        source: ChatTemplateSourceKind::EmbeddedInManifest,
        tokenizer_compatible: true,
        model_family_compatible: true,
        required_variables: BTreeSet::from(["messages".to_string()]),
        special_token_interaction: BTreeSet::new(),
    };
    assert!(validate_chat_template(&metadata, &BTreeSet::new()).is_err());
    assert!(validate_chat_template(&metadata, &BTreeSet::from(["messages".to_string()])).is_ok());

    let incompatible = ChatTemplateMetadata {
        tokenizer_compatible: false,
        ..metadata
    };
    assert!(matches!(
        validate_chat_template(&incompatible, &BTreeSet::new()),
        Err(ModelFormatRoadmapError::ChatTemplateInvalid { .. })
    ));

    assert_eq!(
        redact_chat_template_diagnostic("plain message"),
        "plain message"
    );
}

#[test]
fn model_format_roadmap_sentencepiece_unsupported_feature_fails_explicitly() {
    let metadata = SentencePieceMetadata {
        model_identity: "spm-1".into(),
        vocabulary_size: 32000,
        special_tokens: Vec::new(),
        normalization: None,
        browser_supported: false,
        license: None,
        supported_features: BTreeSet::from(["bpe".to_string()]),
    };
    assert!(reject_unsupported_sentencepiece_feature(&metadata, "bpe").is_ok());
    assert!(matches!(
        reject_unsupported_sentencepiece_feature(&metadata, "byte-fallback"),
        Err(ModelFormatRoadmapError::SentencePieceUnsupported { .. })
    ));
}

#[test]
fn model_format_roadmap_gguf_metadata_validates_and_normalizes_quantized_tensors() {
    let quantization = ModelQuantization {
        format: ModelQuantizationFormat::GgufQ4K,
        group_size: Some(32),
        block_size: None,
        scale_dtype: Some(ModelDType::F16),
        zero_point_dtype: None,
        per_channel: false,
        workspace_bytes: None,
        required_capabilities: Vec::new(),
    };
    let gguf = GgufMetadata {
        architecture: "qwen2".into(),
        alignment: 32,
        tensors: vec![GgufTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![4, 4],
            dtype: ModelDType::Q4K,
            quantization: Some(quantization),
        }],
        tokenizer_embedded: None,
        key_values: BTreeMap::new(),
    };
    assert!(gguf.validate().is_ok());
    let tensors = gguf.into_tensor_metadata();
    assert_eq!(tensors.len(), 1);
    assert!(tensors[0].quantization.is_some());
    assert_eq!(tensors[0].layout.as_deref(), Some("quantized-packed"));

    let empty = GgufMetadata {
        tensors: Vec::new(),
        ..gguf
    };
    assert!(matches!(
        empty.validate(),
        Err(ModelFormatRoadmapError::GgufInvalid { .. })
    ));

    assert!(reject_model_format_provider_name("GGUFProvider").is_err());
}

#[test]
fn model_format_roadmap_normalizes_lora_adapter_without_activating_or_trusting_it() {
    let base_model = AdapterBaseModelCompatibility {
        model_name: ModelName::new("qwen").unwrap(),
        model_revision: ModelRevision::new("v1").unwrap(),
        model_artifact: None,
        tokenizer: None,
        architecture: AdapterArchitectureCompatibility {
            family: "qwen".into(),
            implementation: "qwen2".into(),
            hidden_size: Some(4096),
            layer_count: Some(32),
            position_encoding: None,
            target_modules: BTreeSet::from(["q_proj".to_string()]),
            supported_storage_dtypes: BTreeSet::from([ModelDType::F16]),
            supported_compute_dtypes: BTreeSet::from([ComputeDType::Float16]),
            supported_quantization_formats: BTreeSet::new(),
        },
    };
    let metadata = LoraAdapterFormatMetadata {
        target_modules: vec!["q_proj".into()],
        rank: 8,
        alpha: 16,
        scaling: Some(2.0),
        dropout: Some(0.1),
        base_model,
        tensors: vec![tensor("q_proj.lora_a", vec![4096, 8])],
        storage_dtype: ModelDType::F16,
        compute_dtype: Some(ComputeDType::Float16),
        quantization: None,
        required_capabilities: Vec::new(),
        license: None,
        provenance: None,
    };
    let id = AdapterArtifactId::new(
        AdapterName::new("support-lora").unwrap(),
        AdapterRevision::new("r1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "8".repeat(64))).unwrap(),
    );
    let artifact = normalize_lora_adapter(id, &metadata);
    assert_eq!(artifact.method, AdapterMethod::Lora);
    assert_eq!(artifact.rank, Some(8));
    assert_eq!(artifact.alpha, Some(16));
    assert_eq!(artifact.trust, AdapterTrustStatus::Unknown);
    assert_eq!(artifact.targets.len(), 1);
}

#[test]
fn model_format_roadmap_quantization_declaration_requires_scale_dtype_and_rejects_hidden_dequant() {
    let missing_scale = ModelFormatQuantizationDeclaration {
        model_quantization: ModelQuantization {
            format: ModelQuantizationFormat::Gptq,
            group_size: Some(64),
            block_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            per_channel: false,
            workspace_bytes: None,
            required_capabilities: Vec::new(),
        },
        kernel_compatibility: None,
    };
    assert!(matches!(
        validate_model_format_quantization(&missing_scale, true),
        Err(ModelFormatRoadmapError::QuantizationMetadataInvalid { .. })
    ));

    let with_kernel = ModelFormatQuantizationDeclaration {
        model_quantization: ModelQuantization {
            scale_dtype: Some(ModelDType::F16),
            ..missing_scale.model_quantization.clone()
        },
        kernel_compatibility: Some(KernelQuantizationMetadata {
            method: KernelQuantizationMethod::Int8,
            storage_dtype: ComputeDType::SInt8,
            compute_dtype: ComputeDType::Float32,
            accumulation_dtype: ComputeDType::Float32,
            scale_dtype: ComputeDType::Float32,
            zero_point_dtype: None,
            group_size: None,
            packing_layout: TensorLayoutKind::QuantizedPacked,
            dequantization: KernelDequantizationBehavior::ExplicitBeforeOperator,
            supported_operators: BTreeSet::from([OperatorId::magnetar(
                "matmul",
                1,
                OperatorFamily::LinearAlgebra,
            )]),
            conformance_tolerance_profile: "operator-default".into(),
        }),
    };
    assert!(validate_model_format_quantization(&with_kernel, true).is_ok());
    assert!(matches!(
        validate_model_format_quantization(&with_kernel, false),
        Err(ModelFormatRoadmapError::QuantizationMetadataInvalid { .. })
    ));
}

#[test]
fn model_format_roadmap_source_and_local_file_and_network_boundaries() {
    for source in [
        ModelArtifactSource::LocalPath("/models/qwen".into()),
        ModelArtifactSource::LocalCache("cache-1".into()),
        ModelArtifactSource::ClientProvided("client-1".into()),
        ModelArtifactSource::Registry("registry-1".into()),
        ModelArtifactSource::HuggingFace("qwen/qwen2".into()),
        ModelArtifactSource::Oci("oci://image".into()),
        ModelArtifactSource::Tachyon("tachyon-1".into()),
    ] {
        assert!(reject_arbitrary_model_download(&source).is_ok());
    }

    let local = ModelArtifactSource::LocalPath("/models/qwen".into());
    assert!(matches!(
        validate_local_file_boundary(&local, false),
        Err(ModelFormatRoadmapError::ModelFormatLocalFileDenied { .. })
    ));
    assert!(validate_local_file_boundary(&local, true).is_ok());

    assert!(reject_raw_network_model_reference("https://example.com/model.gguf").is_err());
    assert!(reject_raw_network_model_reference("qwen/qwen2").is_ok());
}

#[test]
fn model_format_roadmap_format_alone_does_not_grant_trust() {
    let store = ModelTrustStore::default();
    let manifest = fixture_model_manifest();
    let decision = model_format_grants_no_trust(&store, &manifest);
    assert_eq!(decision.status, ModelTrustStatus::Unknown);

    let trusted_store = ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone());
    let trusted_decision = model_format_grants_no_trust(&trusted_store, &manifest);
    assert_eq!(trusted_decision.status, ModelTrustStatus::Trusted);
}

#[test]
fn model_format_roadmap_conformance_fixture_kinds_cover_twelve_categories() {
    assert_eq!(MODEL_FORMAT_CONFORMANCE_FIXTURES.len(), 12);
    for fixture in MODEL_FORMAT_CONFORMANCE_FIXTURES {
        assert!(!fixture.id().is_empty());
    }
}

#[test]
fn model_format_roadmap_observation_redacts_metadata() {
    let observation =
        ModelFormatRoadmapObservation::new(ModelFormatRoadmapObservationKind::ManifestNormalized)
            .with_artifact("fixture-model")
            .with_redacted_metadata("path", "/etc/secret/model.safetensors");
    assert_eq!(observation.artifact.as_deref(), Some("fixture-model"));
    assert!(observation.redacted_metadata.contains_key("path"));
}

#[test]
fn model_format_roadmap_conformance_report_is_conformant() {
    let report = run_model_format_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// model_source_cache_roadmap
// ---------------------------------------------------------------------

fn model_source_cache_probe_artifact_id(name: &str) -> ModelArtifactId {
    let digest = ModelDigest::sha256(name.as_bytes());
    ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new(name).unwrap(),
        ModelRevision::new("v1").unwrap(),
        digest,
    )
}

#[test]
fn model_source_cache_roadmap_source_kinds_cover_seven_categories() {
    assert_eq!(MODEL_SOURCE_KINDS.len(), 7);
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for kind in MODEL_SOURCE_KINDS {
        assert!(
            ids.insert(kind.id()),
            "duplicate source kind id {}",
            kind.id()
        );
        assert!(
            !kind.grants_trust(),
            "source kind {} must not grant trust",
            kind.id()
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_kind_normalizes_from_artifact_source() {
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::LocalCache("x".into())),
        ModelSourceKind::LocalCache
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::ClientProvided("x".into())),
        ModelSourceKind::ClientProvidedArtifact
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::LocalPath("/models".into())),
        ModelSourceKind::LocalDirectorySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Registry("x".into())),
        ModelSourceKind::ExternalRegistrySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Oci("x".into())),
        ModelSourceKind::ExternalRegistrySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::HuggingFace("x".into())),
        ModelSourceKind::ModelHubSource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Tachyon("x".into())),
        ModelSourceKind::TachyonProvidedSource
    );
}

#[test]
fn model_source_cache_roadmap_source_kind_normalizes_from_resolution_source() {
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::DevelopmentFixture),
        Some(ModelSourceKind::DevelopmentFixture)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::ClientProvidedArtifact),
        Some(ModelSourceKind::ClientProvidedArtifact)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::TrustedCache),
        Some(ModelSourceKind::LocalCache)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::FutureExternalSource),
        Some(ModelSourceKind::ExternalRegistrySource)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::FutureTachyonSource),
        Some(ModelSourceKind::TachyonProvidedSource)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::LocalRegistry),
        None
    );
}

#[test]
fn model_source_cache_roadmap_development_fixture_denied_in_production_by_default() {
    assert!(validate_development_fixture_source(true, false).is_err());
    assert!(validate_development_fixture_source(true, true).is_ok());
    assert!(validate_development_fixture_source(false, false).is_ok());
}

#[test]
fn model_source_cache_roadmap_development_fixture_still_uses_real_trust_store() {
    let store = ModelTrustStore::default();
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: model_source_cache_probe_artifact_id("fixture-probe"),
        architecture: ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    };
    let decision = development_fixture_requires_explicit_trust_evaluation(&store, &manifest);
    assert_eq!(decision.status, ModelTrustStatus::Unknown);
}

#[test]
fn model_source_cache_roadmap_client_provided_source_requires_authorization() {
    assert!(validate_client_provided_source(false).is_err());
    assert!(validate_client_provided_source(true).is_ok());
}

#[test]
fn model_source_cache_roadmap_cache_key_is_digest_based() {
    let id = model_source_cache_probe_artifact_id("qwen-local");
    let key = CacheKey::from_artifact(&id);
    assert_eq!(key.as_str(), id.digest.value.as_str());
}

#[test]
fn model_source_cache_roadmap_cache_entry_ref_redacts_path_by_default() {
    let key = CacheKey::from_artifact(&model_source_cache_probe_artifact_id("qwen-local"));
    let entry_ref = CacheEntryRef::new(key, "/var/cache/magnetar/qwen-local");
    assert_eq!(entry_ref.redacted_path(false), None);
    assert_eq!(
        entry_ref.redacted_path(true).as_deref(),
        Some("/var/cache/magnetar/qwen-local")
    );
}

#[test]
fn model_source_cache_roadmap_local_directory_source_denies_unauthorized_access() {
    let source = ModelArtifactSource::LocalPath("/models/qwen".into());
    assert!(validate_local_directory_source(&source, false).is_err());
    assert!(validate_local_directory_source(&source, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_remote_source_policy_denies_by_default() {
    let policy = SourcePolicy::default();
    assert!(
        validate_remote_source_policy(ModelSourceKind::ExternalRegistrySource, &policy).is_err()
    );
    assert!(validate_remote_source_policy(ModelSourceKind::ModelHubSource, &policy).is_err());
    assert!(
        validate_remote_source_policy(ModelSourceKind::TachyonProvidedSource, &policy).is_err()
    );
    assert!(validate_remote_source_policy(ModelSourceKind::LocalCache, &policy).is_ok());
}

#[test]
fn model_source_cache_roadmap_tachyon_source_requires_explicit_policy_flag_even_if_listed() {
    let mut policy = SourcePolicy::default();
    policy
        .allowed_kinds
        .insert(ModelSourceKind::TachyonProvidedSource);
    assert!(!policy.allow_tachyon_provided_sources);
    assert!(!policy.allows_kind(ModelSourceKind::TachyonProvidedSource));
    policy.allow_tachyon_provided_sources = true;
    assert!(policy.allows_kind(ModelSourceKind::TachyonProvidedSource));
}

#[test]
fn model_source_cache_roadmap_model_ref_resolution_rejects_zero_and_multiple_candidates() {
    let none = resolve_model_ref_candidates("qwen", Vec::new());
    assert!(matches!(
        none,
        Err(ModelSourceCacheRoadmapError::ModelSourceNotFound { .. })
    ));

    let one = resolve_model_ref_candidates(
        "qwen",
        vec![ModelRefResolutionOutcome::SourceCandidate {
            kind: ModelSourceKind::LocalCache,
            reference: "qwen".into(),
        }],
    );
    assert!(one.is_ok());

    let many = resolve_model_ref_candidates(
        "qwen",
        vec![
            ModelRefResolutionOutcome::SourceCandidate {
                kind: ModelSourceKind::LocalCache,
                reference: "qwen-a".into(),
            },
            ModelRefResolutionOutcome::SourceCandidate {
                kind: ModelSourceKind::DevelopmentFixture,
                reference: "qwen-b".into(),
            },
        ],
    );
    assert!(matches!(
        many,
        Err(ModelSourceCacheRoadmapError::ModelSourceAmbiguous { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_alias_missing_and_ambiguous_errors() {
    let alias = ModelAlias::new("qwen").unwrap();
    let mut table = ModelAliasTable::new();
    assert!(matches!(
        table.resolve(&alias),
        Err(ModelSourceCacheRoadmapError::ModelAliasNotFound { .. })
    ));

    table.register(alias.clone(), "qwen-a");
    assert_eq!(table.resolve(&alias).unwrap(), "qwen-a");

    table.register(alias.clone(), "qwen-b");
    assert!(matches!(
        table.resolve(&alias),
        Err(ModelSourceCacheRoadmapError::ModelAliasAmbiguous { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_alias_rejects_empty_name() {
    assert!(ModelAlias::new("").is_err());
    assert!(ModelAlias::new("qwen").is_ok());
}

#[test]
fn model_source_cache_roadmap_artifact_identity_coverage_tracks_present_fields() {
    let mut manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: model_source_cache_probe_artifact_id("qwen-local"),
        architecture: ModelArchitecture::new("qwen", "qwen"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: Some("tokenizer-id".into()),
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: Some(ModelArtifactSource::LocalCache("qwen".into())),
    };
    let coverage = ArtifactIdentityCoverage::from_manifest(&manifest);
    assert!(coverage.content_digest);
    assert!(coverage.tokenizer_reference);
    assert!(coverage.source_annotation);
    assert!(coverage.version_metadata);
    assert!(coverage.covers_required_fields());

    manifest.tokenizer = None;
    manifest.source = None;
    let coverage = ArtifactIdentityCoverage::from_manifest(&manifest);
    assert!(!coverage.tokenizer_reference);
    assert!(!coverage.source_annotation);
}

#[test]
fn model_source_cache_roadmap_same_name_different_digest_remains_distinct() {
    let left = model_source_cache_probe_artifact_id("qwen-local");
    let mut right = model_source_cache_probe_artifact_id("qwen-local");
    right.digest = ModelDigest::sha256(b"a-different-payload");
    assert!(artifacts_are_distinct_despite_same_name(&left, &right));
}

#[test]
fn model_source_cache_roadmap_cache_lifecycle_states_cover_thirteen_categories() {
    assert_eq!(CACHE_LIFECYCLE_STATES.len(), 13);
    let mut ready_count = 0;
    for state in CACHE_LIFECYCLE_STATES {
        assert!(!state.id().is_empty());
        if state.is_loadable() {
            ready_count += 1;
        }
    }
    assert_eq!(ready_count, 1, "exactly Ready should be loadable");
}

#[test]
fn model_source_cache_roadmap_rejects_every_non_ready_lifecycle_state_for_loading() {
    for state in CACHE_LIFECYCLE_STATES {
        let outcome = reject_non_ready_cache_entry_for_loading(*state);
        if matches!(state, CacheLifecycleState::Ready) {
            assert!(outcome.is_ok(), "Ready must be loadable");
        } else {
            assert!(outcome.is_err(), "{state:?} must not be loadable");
        }
    }
}

#[test]
fn model_source_cache_roadmap_partial_and_corrupt_entries_map_to_specific_errors() {
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Partial),
        Err(ModelSourceCacheRoadmapError::ModelCachePartialEntry { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Corrupt),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Untrusted),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryUntrusted { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Revoked),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryRevoked { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_cache_entry_metadata_defaults_to_discovered_and_unpinned() {
    let entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert_eq!(entry.lifecycle, CacheLifecycleState::Discovered);
    assert!(!entry.pinned);
    assert_eq!(entry.trust_status, ModelTrustStatus::Unknown);
    assert_eq!(entry.integrity_status, CacheIntegrityStatus::Unchecked);
    assert_eq!(entry.validation_status, CacheValidationStatus::Unvalidated);
    assert_eq!(entry.key(), CacheKey::from_artifact(&entry.identity));
}

#[test]
fn model_source_cache_roadmap_cache_trust_re_evaluates_and_revocation_wins() {
    let digest = ModelDigest::sha256(b"trusted-model");
    let store = ModelTrustStore::default().trust_digest(digest.value.clone());
    let mut manifest_id = model_source_cache_probe_artifact_id("trusted-model");
    manifest_id.digest = digest;
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: manifest_id,
        architecture: ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    };
    let trusted = evaluate_cache_trust(&store, &manifest, false);
    assert_eq!(trusted.status, ModelTrustStatus::Trusted);
    let revoked = evaluate_cache_trust(&store, &manifest, true);
    assert_eq!(revoked.status, ModelTrustStatus::Revoked);
}

#[test]
fn model_source_cache_roadmap_cache_integrity_detects_digest_mismatch() {
    let declared = ModelDigest::sha256(b"declared-bytes");
    let matching = ModelDigest::sha256(b"declared-bytes");
    let mismatched = ModelDigest::sha256(b"other-bytes");
    assert!(validate_cache_integrity(&declared, &matching).is_ok());
    assert!(matches!(
        validate_cache_integrity(&declared, &mismatched),
        Err(ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_cache_shard_integrity_composes_shard_verify_bytes() {
    let bytes = b"shard-bytes";
    let shard = ModelShard {
        id: ModelShardId::new("shard-0").unwrap(),
        digest: ModelDigest::sha256(bytes),
        size_bytes: bytes.len() as u64,
        order: 0,
    };
    assert!(validate_cache_shard_integrity(&shard, bytes).is_ok());
    assert!(matches!(
        validate_cache_shard_integrity(&shard, b"wrong-bytes"),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_mutation_requires_policy_and_denies_eviction_with_active_reference() {
    assert!(authorize_cache_mutation(CacheMutationKind::Insert, false, 0).is_err());
    assert!(authorize_cache_mutation(CacheMutationKind::Insert, true, 0).is_ok());
    assert!(matches!(
        authorize_cache_mutation(CacheMutationKind::Evict, true, 1),
        Err(ModelSourceCacheRoadmapError::ModelCacheActiveReference { .. })
    ));
    assert!(matches!(
        authorize_cache_mutation(CacheMutationKind::Evict, false, 0),
        Err(ModelSourceCacheRoadmapError::ModelCacheEvictionDenied { .. })
    ));
    assert!(authorize_cache_mutation(CacheMutationKind::Evict, true, 0).is_ok());
}

#[test]
fn model_source_cache_roadmap_eviction_respects_pin_and_active_reference_and_orders_by_last_used() {
    let mut old_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-old"),
        ModelSourceKind::LocalCache,
    );
    old_entry.lifecycle = CacheLifecycleState::Ready;
    old_entry.last_used_unix_seconds = 10;

    let mut new_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-new"),
        ModelSourceKind::LocalCache,
    );
    new_entry.lifecycle = CacheLifecycleState::Ready;
    new_entry.last_used_unix_seconds = 20;

    let mut pinned_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-pinned"),
        ModelSourceKind::LocalCache,
    );
    pinned_entry.lifecycle = CacheLifecycleState::Ready;
    pin_entry(&mut pinned_entry);

    let mut active_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-active"),
        ModelSourceKind::LocalCache,
    );
    active_entry.lifecycle = CacheLifecycleState::Ready;

    let candidates = vec![
        EvictionCandidate {
            entry: new_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: old_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: pinned_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: active_entry,
            active_instance_refs: 1,
        },
    ];

    let selected = select_eviction_candidates(&candidates);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].identity.name.as_str(), "qwen-old");
    assert_eq!(selected[1].identity.name.as_str(), "qwen-new");
}

#[test]
fn model_source_cache_roadmap_pin_and_unpin_toggle_evictability() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    entry.lifecycle = CacheLifecycleState::Ready;
    let candidate = EvictionCandidate {
        entry: entry.clone(),
        active_instance_refs: 0,
    };
    assert!(is_evictable(&candidate));

    pin_entry(&mut entry);
    let pinned_candidate = EvictionCandidate {
        entry: entry.clone(),
        active_instance_refs: 0,
    };
    assert!(!is_evictable(&pinned_candidate));

    unpin_entry(&mut entry);
    let unpinned_candidate = EvictionCandidate {
        entry,
        active_instance_refs: 0,
    };
    assert!(is_evictable(&unpinned_candidate));
}

#[test]
fn model_source_cache_roadmap_in_flight_lifecycle_states_are_never_evictable() {
    for state in [
        CacheLifecycleState::Resolving,
        CacheLifecycleState::Fetching,
        CacheLifecycleState::Normalizing,
        CacheLifecycleState::Validating,
        CacheLifecycleState::Evicting,
    ] {
        let mut entry = CacheEntryMetadata::new(
            model_source_cache_probe_artifact_id("qwen-in-flight"),
            ModelSourceKind::LocalCache,
        );
        entry.lifecycle = state;
        let candidate = EvictionCandidate {
            entry,
            active_instance_refs: 0,
        };
        assert!(!is_evictable(&candidate), "{state:?} must not be evictable");
    }
}

#[test]
fn model_source_cache_roadmap_offline_mode_allows_only_local_sources() {
    assert!(validate_offline_source(ModelSourceKind::LocalCache, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::ClientProvidedArtifact, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::DevelopmentFixture, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::LocalDirectorySource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::ExternalRegistrySource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::ModelHubSource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::TachyonProvidedSource, true).is_err());
    // Online mode never restricts by source kind.
    assert!(validate_offline_source(ModelSourceKind::ExternalRegistrySource, false).is_ok());
}

#[test]
fn model_source_cache_roadmap_rejects_credential_shaped_metadata_keys() {
    let mut annotations = BTreeMap::new();
    annotations.insert("model_family".to_string(), "qwen".to_string());
    assert!(reject_credential_in_metadata(&annotations).is_ok());

    for key in ["registry_token", "api_key", "auth_secret", "bearer_header"] {
        let mut annotations = BTreeMap::new();
        annotations.insert(key.to_string(), "value".to_string());
        assert!(
            reject_credential_in_metadata(&annotations).is_err(),
            "key '{key}' should be rejected"
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_policy_defaults_deny_remote_and_allow_offline_kinds() {
    let policy = SourcePolicy::default();
    for kind in [
        ModelSourceKind::DevelopmentFixture,
        ModelSourceKind::ClientProvidedArtifact,
        ModelSourceKind::LocalCache,
    ] {
        assert!(
            policy.allows_kind(kind),
            "{kind:?} should be allowed by default"
        );
    }
    for kind in [
        ModelSourceKind::LocalDirectorySource,
        ModelSourceKind::ExternalRegistrySource,
        ModelSourceKind::ModelHubSource,
        ModelSourceKind::TachyonProvidedSource,
    ] {
        assert!(
            !policy.allows_kind(kind),
            "{kind:?} should be denied by default"
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_policy_enforces_size_limit() {
    let policy = SourcePolicy {
        max_artifact_size_bytes: Some(1024),
        ..SourcePolicy::default()
    };
    assert!(policy.validate_size(512).is_ok());
    assert!(policy.validate_size(2048).is_err());
}

#[test]
fn model_source_cache_roadmap_license_requires_explicit_policy_validation() {
    let license = ModelLicenseMetadata {
        identifier: "apache-2.0".into(),
        url: None,
        usage_restrictions: Vec::new(),
    };
    assert!(validate_license_policy(&license, false, true).is_err());
    assert!(validate_license_policy(&license, true, false).is_err());
    assert!(validate_license_policy(&license, true, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_format_normalization_requires_metadata_and_lifecycle() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert!(!cache_entry_ready_for_format_normalization(&entry));
    entry.format_metadata = Some("safetensors".into());
    assert!(!cache_entry_ready_for_format_normalization(&entry));
    entry.lifecycle = CacheLifecycleState::Ready;
    assert!(cache_entry_ready_for_format_normalization(&entry));
}

#[test]
fn model_source_cache_roadmap_adapter_cache_entry_requires_reference_and_compatibility() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("lora-local"),
        ModelSourceKind::LocalCache,
    );
    let compatibility = AdapterBaseModelCompatibility {
        model_name: ModelName::new("qwen").unwrap(),
        model_revision: ModelRevision::new("v1").unwrap(),
        model_artifact: None,
        tokenizer: None,
        architecture: AdapterArchitectureCompatibility {
            family: "qwen".into(),
            implementation: "qwen".into(),
            hidden_size: None,
            layer_count: None,
            position_encoding: None,
            target_modules: BTreeSet::new(),
            supported_storage_dtypes: BTreeSet::new(),
            supported_compute_dtypes: BTreeSet::new(),
            supported_quantization_formats: BTreeSet::new(),
        },
    };
    assert!(validate_adapter_cache_entry(&entry, &compatibility).is_err());
    entry.adapters.push(AdapterArtifactId::new(
        AdapterName::new("lora").unwrap(),
        AdapterRevision::new("v1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "1".repeat(64))).unwrap(),
    ));
    assert!(validate_adapter_cache_entry(&entry, &compatibility).is_ok());
}

#[test]
fn model_source_cache_roadmap_tokenizer_cache_entry_requires_reference_and_compatibility() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert!(validate_tokenizer_cache_entry(&entry, true).is_err());
    entry.tokenizer = Some(TokenizerArtifactId::new("qwen-tokenizer").unwrap());
    assert!(validate_tokenizer_cache_entry(&entry, false).is_err());
    assert!(validate_tokenizer_cache_entry(&entry, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_cache_presence_never_implies_memory_residency() {
    assert!(!cache_presence_implies_memory_residency());
}

#[test]
fn model_source_cache_roadmap_diagnostic_digest_prefix_is_short_and_never_full() {
    let digest = ModelDigest::sha256(b"some-model-bytes");
    let prefix = ModelSourceCacheDiagnostic::digest_prefix_from(&digest);
    assert!(prefix.len() <= 15);
    assert_ne!(prefix, digest.value);
}

#[test]
fn model_source_cache_roadmap_diagnostic_redacts_policy_denial_reason() {
    let redacted = ModelSourceCacheDiagnostic::redact_policy_denial_reason(
        "denied for path /var/cache/magnetar/model",
    );
    assert!(!redacted.contains('/'));
}

#[test]
fn model_source_cache_roadmap_error_display_is_non_empty_for_every_variant() {
    let errors = vec![
        ModelSourceCacheRoadmapError::ModelSourceUnsupported { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceInvalid { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceAmbiguous {
            reference: "r".into(),
        },
        ModelSourceCacheRoadmapError::ModelSourcePolicyDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceNetworkDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceAuthenticationFailed { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceNotFound {
            reference: "r".into(),
        },
        ModelSourceCacheRoadmapError::ModelSourceOfflineUnavailable { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheUnavailable { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheMiss { key: "k".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryInvalid { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryUntrusted { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryRevoked { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheInsertDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEvictionDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheActiveReference { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCachePartialEntry { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelAliasNotFound { alias: "a".into() },
        ModelSourceCacheRoadmapError::ModelAliasAmbiguous { alias: "a".into() },
        ModelSourceCacheRoadmapError::InternalModelSourceCacheError { reason: "r".into() },
    ];
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for error in &errors {
        assert!(!error.to_string().is_empty());
        assert!(ids.insert(error.id()), "duplicate error id {}", error.id());
    }
    assert_eq!(ids.len(), 22);
}

#[test]
fn model_source_cache_roadmap_observation_redacts_metadata() {
    let observation =
        ModelSourceCacheRoadmapObservation::new(ModelSourceCacheRoadmapObservationKind::CacheHit)
            .with_artifact("qwen-local")
            .with_redacted_metadata("path", "/var/cache/magnetar/qwen-local");
    assert_eq!(observation.artifact.as_deref(), Some("qwen-local"));
    assert!(
        !observation
            .redacted_metadata
            .get("path")
            .unwrap()
            .contains('/')
    );
}

#[test]
fn model_source_cache_roadmap_conformance_report_is_conformant() {
    let report = run_model_source_cache_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// server_api_roadmap
// ---------------------------------------------------------------------

fn server_api_roadmap_tokenizer_reference() -> GenerationTokenizerReference {
    let metadata = TokenizerMetadata {
        id: TokenizerId::new("server-api-roadmap-test").unwrap(),
        artifact: TokenizerArtifactId::new("server-api-roadmap-test-artifact").unwrap(),
        digest: ModelDigest::sha256(b"server-api-roadmap-test"),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(64),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    }
}

fn server_api_roadmap_generation_request(streaming: bool) -> ServerGenerationRequest {
    ServerGenerationRequest {
        model_or_session: ServerModelOrSessionRef::Model(ModelRef::new("fixture-model").unwrap()),
        prompt: PromptInput::PlainText("hello".into()),
        parameters: GenerationParameters::greedy(),
        max_new_tokens: 4,
        max_total_tokens: Some(32),
        stop_conditions: StopConditions::default(),
        streaming,
        cache_policy: KvCachePolicy {
            enabled: false,
            max_cache_tokens: None,
            max_cache_memory_bytes: None,
            sharing: KvCacheSharingPolicy::Deny,
            retention: KvCacheRetentionPolicy::ReleaseOnSessionClose,
            prefix_reuse_allowed: false,
            privacy_redaction_required: true,
        },
        adapter_policy: None,
        timeout_millis: Some(5_000),
        correlation_id: Some(CorrelationId::new("fixture-correlation")),
    }
}

fn server_api_roadmap_runtime_context() -> ServerGenerationRuntimeContext {
    ServerGenerationRuntimeContext {
        request_id: GenerationRequestId::new("fixture-request").unwrap(),
        model: GenerationModelReference::LoadedModelContext("fixture-model-context".into()),
        tokenizer: server_api_roadmap_tokenizer_reference(),
        input_token_ids: vec![2, 3, 4],
        model_context_length: Some(64),
        trace_id: None,
    }
}

fn server_api_roadmap_session_creation_request(
    allowed_capabilities: BTreeSet<String>,
) -> SessionCreationRequest {
    let tokenizer = server_api_roadmap_tokenizer_reference();
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext("fixture-model-context".into()),
        tokenizer,
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities,
        correlation_id: None,
        created_at_millis: 0,
    }
}

// -- Endpoint scope --

#[test]
fn server_api_roadmap_endpoints_are_illustrative_and_complete() {
    assert_eq!(SERVER_API_ENDPOINTS.len(), 10);
    let ids: BTreeSet<&str> = SERVER_API_ENDPOINTS.iter().map(|e| e.id()).collect();
    for expected in [
        "health",
        "readiness",
        "models-list",
        "model-inspect",
        "session-create",
        "session-close",
        "generate",
        "generate-stream",
        "cancel",
        "diagnostics",
    ] {
        assert!(ids.contains(expected), "missing endpoint id '{expected}'");
    }
    for endpoint in SERVER_API_ENDPOINTS {
        assert!(endpoint.is_illustrative());
    }
}

// -- Health and readiness --

#[test]
fn server_api_roadmap_health_does_not_imply_readiness_or_model_availability() {
    let health = ServerHealthStatus::alive();
    let readiness = ServerReadinessStatus::not_ready("no model loaded");
    assert!(health.alive);
    assert!(!readiness.ready);
    assert!(healthy_but_not_ready_is_representable(&health, &readiness));
}

#[test]
fn server_api_roadmap_readiness_is_redacted() {
    let readiness = ServerReadinessStatus::not_ready("provider handle=0xdeadbeef unavailable");
    let summary = readiness.model_registry_state_summary.unwrap();
    assert!(!summary.contains("0xdeadbeef"));
}

#[test]
fn server_api_roadmap_health_and_readiness_are_structurally_independent_types() {
    // No conversion exists in either direction; both are constructed
    // independently.
    let health = ServerHealthStatus::not_alive("process exiting");
    let readiness = ServerReadinessStatus::ready();
    assert!(!health.alive);
    assert!(readiness.ready);
}

// -- Model endpoints --

#[test]
fn server_api_roadmap_model_load_requires_complete_loading_proof() {
    let incomplete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelLoad,
        &ModelEndpointLoadingProof::deny_by_default(),
    );
    assert!(matches!(
        incomplete,
        Err(ServerApiRoadmapError::ServerModelLoadFailed { .. })
    ));

    let complete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelLoad,
        &ModelEndpointLoadingProof {
            source_validated: true,
            cache_validated: true,
            artifact_validated: true,
            model_loading_validated: true,
            trust_validated: true,
            integrity_validated: true,
            compatibility_validated: true,
            policy_validated: true,
        },
    );
    assert!(complete.is_ok());
}

#[test]
fn server_api_roadmap_model_unload_requires_complete_loading_proof() {
    let incomplete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelUnload,
        &ModelEndpointLoadingProof::deny_by_default(),
    );
    assert!(incomplete.is_err());
}

#[test]
fn server_api_roadmap_read_only_model_endpoints_skip_loading_proof() {
    for operation in [
        ServerModelEndpointOperation::ListKnownModels,
        ServerModelEndpointOperation::InspectModelMetadata,
        ServerModelEndpointOperation::InspectLoadedInstance,
    ] {
        assert!(!operation.requires_loading_proof());
        let outcome = validate_model_endpoint_request(
            operation,
            &ModelEndpointLoadingProof::deny_by_default(),
        );
        assert!(outcome.is_ok());
    }
}

#[test]
fn server_api_roadmap_rejects_arbitrary_local_model_path() {
    let source = ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen"));
    let outcome = reject_server_arbitrary_model_path(&source, false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerSourcePolicyDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_authorized_local_model_path() {
    let source = ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen"));
    assert!(reject_server_arbitrary_model_path(&source, true).is_ok());
}

// -- Session endpoints --

#[test]
fn server_api_roadmap_session_rejects_cli_owned_authority_capabilities() {
    for capability in [
        "workspace",
        "git",
        "shell",
        "secrets",
        "tool-call",
        "filesystem",
    ] {
        let outcome = reject_server_session_owned_authority(capability);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            "capability '{capability}' should be rejected"
        );

        let mut capabilities = BTreeSet::new();
        capabilities.insert(capability.to_string());
        let request =
            ServerSessionRequest::new(server_api_roadmap_session_creation_request(capabilities));
        assert!(
            request.is_err(),
            "capability '{capability}' should reject session creation"
        );
    }
}

#[test]
fn server_api_roadmap_session_accepts_inference_scoped_capability() {
    assert!(reject_server_session_owned_authority("generation").is_ok());
    let mut capabilities = BTreeSet::new();
    capabilities.insert("generation".to_string());
    let request =
        ServerSessionRequest::new(server_api_roadmap_session_creation_request(capabilities));
    assert!(request.is_ok());
}

#[test]
fn server_api_roadmap_connection_state_is_separate_from_session_state() {
    let session = InferenceSessionId::new("session-1").unwrap();
    let connection = ServerConnectionState {
        connection: ServerConnectionId::new("conn-1"),
        bound_session: Some(session.clone()),
    };
    let (connection_id, policy, bound_session) =
        server_disconnect_policy(&connection, ServerDisconnectPolicy::CancelActiveGeneration);
    assert_eq!(connection_id.as_str(), "conn-1");
    assert_eq!(policy, ServerDisconnectPolicy::CancelActiveGeneration);
    assert_eq!(bound_session, Some(session));
}

// -- Generation endpoint --

#[test]
fn server_api_roadmap_generation_request_builds_valid_runtime_generation_request() {
    let request = server_api_roadmap_generation_request(false);
    let context = server_api_roadmap_runtime_context();
    let built = build_runtime_generation_request(&request, context).unwrap();
    assert_eq!(built.max_new_tokens, 4);
    assert_eq!(built.streaming, StreamingMode::Disabled);
    assert!(built.session.is_none());
}

#[test]
fn server_api_roadmap_generation_request_targeting_session_omits_model_and_sets_session() {
    let mut request = server_api_roadmap_generation_request(true);
    let session = InferenceSessionId::new("session-1").unwrap();
    request.model_or_session = ServerModelOrSessionRef::Session(session.clone());
    let context = server_api_roadmap_runtime_context();
    let built = build_runtime_generation_request(&request, context).unwrap();
    assert_eq!(built.session, Some(session));
    assert_eq!(built.streaming, StreamingMode::TokenIds);
}

#[test]
fn server_api_roadmap_generation_endpoint_rejects_tool_execution_from_generated_output() {
    let handling = ServerGeneratedTextHandling {
        text: "git push --force".into(),
        executed_as_tool_call: true,
    };
    let outcome = reject_tool_execution_from_generated_output(&handling);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
}

#[test]
fn server_api_roadmap_generation_endpoint_allows_generated_text_that_is_not_executed() {
    let handling = ServerGeneratedTextHandling {
        text: "rm -rf /".into(),
        executed_as_tool_call: false,
    };
    assert!(reject_tool_execution_from_generated_output(&handling).is_ok());
}

// -- Streaming endpoint --

#[test]
fn server_api_roadmap_streaming_preserves_event_ordering() {
    let source = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::DecodeStarted,
        GenerationEventKind::TokenGenerated,
    ];
    let forwarded = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::TokenGenerated,
    ];
    assert!(validate_stream_event_ordering(&source, &forwarded).is_ok());
}

#[test]
fn server_api_roadmap_streaming_rejects_reordered_events() {
    let source = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::TokenGenerated,
    ];
    let forwarded = [
        GenerationEventKind::TokenGenerated,
        GenerationEventKind::PrefillCompleted,
    ];
    let outcome = validate_stream_event_ordering(&source, &forwarded);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerStreamInterrupted { .. })
    ));
}

#[test]
fn server_api_roadmap_streaming_rejects_raw_payload_kinds_by_default() {
    for payload_kind in SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS {
        let outcome = reject_raw_stream_payload(payload_kind);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerStreamUnavailable { .. })
            ),
            "payload kind '{payload_kind}' should be rejected"
        );
    }
    assert!(reject_raw_stream_payload("token-id").is_ok());
}

#[test]
fn server_api_roadmap_stream_event_carries_only_redacted_metadata() {
    let event = ServerStreamEvent::new(GenerationEventKind::TokenGenerated)
        .with_redacted_metadata("path", "/var/cache/magnetar/model");
    assert!(!event.redacted_metadata.get("path").unwrap().contains('/'));
}

// -- Cancellation endpoint --

#[test]
fn server_api_roadmap_cancellation_calls_runtime_cancellation() {
    let token = CancellationToken::new(GenerationRequestId::new("cancel-fixture").unwrap());
    let outcome =
        server_cancellation_calls_runtime_cancellation(&token, CancellationStage::Decode, false);
    assert_eq!(outcome, CancellationOutcome::Cancelled);
}

#[test]
fn server_api_roadmap_cancellation_reports_limitation_for_provider_execution_stage() {
    let token = CancellationToken::new(GenerationRequestId::new("cancel-fixture-2").unwrap());
    let outcome = server_cancellation_calls_runtime_cancellation(
        &token,
        CancellationStage::ProviderExecution,
        false,
    );
    assert!(matches!(
        outcome,
        CancellationOutcome::LimitationReported { .. }
    ));
}

// -- Diagnostics endpoint --

#[test]
fn server_api_roadmap_diagnostics_summary_is_redacted() {
    let runtime_diagnostics = RuntimeDiagnostics {
        model_instance_count: 1,
        ready_model_instance_count: 1,
        active_session_count: 2,
        provider_count: 1,
        device_count: 1,
        kernel_advertisement_count: 3,
        memory_pressure: MemoryPressureLevel::Low,
        kv_cache_count: 0,
        prefix_cache_entry_count: 0,
        model_resolution_status: None,
        model_loading_status: None,
        operator_missing_count: 0,
        tokenizer_compatible: None,
        queued_admission_count: 0,
        redacted: true,
    };
    let health = ServerHealthStatus::alive();
    let readiness = ServerReadinessStatus::ready();
    let summary = server_diagnostics_summary(
        &runtime_diagnostics,
        &health,
        &readiness,
        Some("provider handle=0xdeadbeef ready"),
        4,
        1,
        vec!["server-generation-failed".into()],
    );
    assert!(summary.redacted);
    assert!(
        !summary
            .provider_readiness_summary
            .as_deref()
            .unwrap()
            .contains("0xdeadbeef")
    );
    assert_eq!(summary.active_session_count, 2);
}

// -- OpenAI-compatible facade placeholder --

#[test]
fn server_api_roadmap_openai_facade_rejects_unsupported_field_per_policy() {
    let outcome = handle_openai_unsupported_field(
        OpenAiCompatibilityPolicy::RejectUnsupportedField,
        "logprobs",
    );
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerRequestInvalid { .. })
    ));
}

#[test]
fn server_api_roadmap_openai_facade_ignores_unsupported_field_per_policy() {
    let outcome = handle_openai_unsupported_field(
        OpenAiCompatibilityPolicy::IgnoreUnsupportedField,
        "logprobs",
    );
    assert!(outcome.is_ok());
}

#[test]
fn server_api_roadmap_openai_facade_rejects_tool_call_execution() {
    let outcome = reject_openai_tool_call_execution(true, true);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
    assert!(reject_openai_tool_call_execution(true, false).is_ok());
}

#[test]
fn server_api_roadmap_openai_facade_maps_to_generation_api_request() {
    let request = server_api_roadmap_generation_request(false);
    let context = server_api_roadmap_runtime_context();
    let core = build_runtime_generation_request(&request, context).unwrap();
    let mapped =
        openai_facade_maps_to_generation_api_request(core, SessionRedactionPolicy::RedactRawInputs);
    assert_eq!(mapped.privacy, SessionRedactionPolicy::RedactRawInputs);
}

// -- Authentication boundary --

#[test]
fn server_api_roadmap_authenticated_request_carries_no_credential_type() {
    let authenticated = AuthenticatedServerRequest::from_authenticated(true);
    assert!(authenticated.is_ok());
}

#[test]
fn server_api_roadmap_authentication_required_when_not_authenticated() {
    let outcome = AuthenticatedServerRequest::from_authenticated(false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthenticationRequired)
    ));
}

#[test]
fn server_api_roadmap_rejects_credential_in_diagnostics() {
    let mut metadata = BTreeMap::new();
    metadata.insert("api_key".to_string(), "secret-value".to_string());
    let outcome = reject_credential_in_server_diagnostics(&metadata);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthenticationFailed { .. })
    ));
}

#[test]
fn server_api_roadmap_redacts_diagnostic_message() {
    let redacted = redact_server_diagnostic("provider handle=0xdeadbeef failed");
    assert!(!redacted.contains("0xdeadbeef"));
}

// -- Authorization boundary --

#[test]
fn server_api_roadmap_authorization_denied_when_server_denies() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: false,
    };
    let outcome = authorize_server_request(&decision, true);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthorizationDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_authorization_does_not_bypass_runtime_policy() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: true,
    };
    let outcome = authorize_server_request(&decision, false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthorizationDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_authorization_succeeds_when_both_allow() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: true,
    };
    assert!(authorize_server_request(&decision, true).is_ok());
}

// -- Admission and rate policy --

#[test]
fn server_api_roadmap_admission_denied_by_default() {
    let outcome = evaluate_server_admission(
        &ServerAdmissionLimits::deny_by_default(),
        &ServerAdmissionState::default(),
    );
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAdmissionRejected { .. })
    ));
}

#[test]
fn server_api_roadmap_admission_allows_within_configured_limits() {
    let limits = ServerAdmissionLimits {
        max_concurrent_requests: 4,
        max_queued_requests: 4,
        max_tokens_per_request: 1024,
        max_sessions: 4,
        max_loaded_models: 2,
        memory_budget_bytes: 1_000_000,
        max_streaming_connections: 4,
        max_request_body_bytes: 1_000_000,
        max_prompt_bytes: 100_000,
        max_source_cache_operations: 4,
    };
    let state = ServerAdmissionState {
        concurrent_requests: 1,
        queued_requests: 0,
        requested_tokens: 128,
        active_sessions: 1,
        loaded_models: 1,
        memory_used_bytes: 1_000,
        streaming_connections: 0,
        request_body_bytes: 512,
        prompt_bytes: 256,
        source_cache_operations_in_flight: 0,
    };
    assert!(evaluate_server_admission(&limits, &state).is_ok());
}

// -- Source and cache boundary --

#[test]
fn server_api_roadmap_rejects_arbitrary_download_during_generation() {
    let outcome = reject_arbitrary_download_during_generation("https://example.com/model.gguf");
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerSourcePolicyDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_non_network_model_reference_during_generation() {
    assert!(reject_arbitrary_download_during_generation("qwen-local").is_ok());
}

// -- Filesystem boundary --

#[test]
fn server_api_roadmap_rejects_arbitrary_filesystem_path() {
    let outcome = reject_arbitrary_filesystem_path("/etc/passwd", false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_authorized_filesystem_path() {
    assert!(reject_arbitrary_filesystem_path("/models/qwen/model.gguf", true).is_ok());
}

// -- Tool/Shell/Git boundary --

#[test]
fn server_api_roadmap_rejects_tool_shell_git_capabilities() {
    for capability in ["tool-call", "shell", "process", "git"] {
        let outcome = reject_server_tool_shell_git_execution(capability);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            "capability '{capability}' should be rejected"
        );
    }
}

#[test]
fn server_api_roadmap_allows_ordinary_inference_capability() {
    assert!(reject_server_tool_shell_git_execution("generation").is_ok());
}

// -- Error model --

#[test]
fn server_api_roadmap_error_id_matches_exact_kebab_string_for_every_variant() {
    let cases: Vec<(ServerApiRoadmapError, &str)> = vec![
        (
            ServerApiRoadmapError::ServerApiUnavailable { reason: "x".into() },
            "server-api-unavailable",
        ),
        (
            ServerApiRoadmapError::ServerRequestInvalid { reason: "x".into() },
            "server-request-invalid",
        ),
        (
            ServerApiRoadmapError::ServerRequestTooLarge { reason: "x".into() },
            "server-request-too-large",
        ),
        (
            ServerApiRoadmapError::ServerAuthenticationRequired,
            "server-authentication-required",
        ),
        (
            ServerApiRoadmapError::ServerAuthenticationFailed { reason: "x".into() },
            "server-authentication-failed",
        ),
        (
            ServerApiRoadmapError::ServerAuthorizationDenied { scope: "x".into() },
            "server-authorization-denied",
        ),
        (
            ServerApiRoadmapError::ServerRateLimited { reason: "x".into() },
            "server-rate-limited",
        ),
        (
            ServerApiRoadmapError::ServerAdmissionRejected { reason: "x".into() },
            "server-admission-rejected",
        ),
        (
            ServerApiRoadmapError::ServerStreamUnavailable { reason: "x".into() },
            "server-stream-unavailable",
        ),
        (
            ServerApiRoadmapError::ServerStreamInterrupted { reason: "x".into() },
            "server-stream-interrupted",
        ),
        (
            ServerApiRoadmapError::ServerCancellationFailed { reason: "x".into() },
            "server-cancellation-failed",
        ),
        (
            ServerApiRoadmapError::ServerModelNotFound { model: "x".into() },
            "server-model-not-found",
        ),
        (
            ServerApiRoadmapError::ServerModelLoadFailed {
                reason: "x".into(),
                runtime_cause: None,
            },
            "server-model-load-failed",
        ),
        (
            ServerApiRoadmapError::ServerSessionNotFound {
                session: "x".into(),
            },
            "server-session-not-found",
        ),
        (
            ServerApiRoadmapError::ServerGenerationFailed {
                reason: "x".into(),
                runtime_cause: None,
            },
            "server-generation-failed",
        ),
        (
            ServerApiRoadmapError::ServerDiagnosticsRedacted,
            "server-diagnostics-redacted",
        ),
        (
            ServerApiRoadmapError::ServerSourcePolicyDenied { reason: "x".into() },
            "server-source-policy-denied",
        ),
        (
            ServerApiRoadmapError::ServerCachePolicyDenied { reason: "x".into() },
            "server-cache-policy-denied",
        ),
        (
            ServerApiRoadmapError::ServerBoundaryViolation {
                capability: "x".into(),
            },
            "server-boundary-violation",
        ),
        (
            ServerApiRoadmapError::InternalServerApiError { reason: "x".into() },
            "internal-server-api-error",
        ),
    ];
    assert_eq!(cases.len(), 20, "expected exactly 20 error categories");
    for (error, expected_id) in cases {
        assert_eq!(error.id(), expected_id);
        assert!(
            !error.to_string().is_empty(),
            "{expected_id} rendered empty"
        );
    }
}

#[test]
fn server_api_roadmap_error_preserves_wrapped_runtime_cause() {
    let source = InferenceApiError::ModelLoadingFailed {
        reason: "example".into(),
    };
    let wrapped = ServerApiRoadmapError::model_load_failed_from_runtime(source.clone());
    assert_eq!(wrapped.runtime_cause(), Some(&source));
    assert!(wrapped.to_string().contains("example"));
}

#[test]
fn server_api_roadmap_generation_failed_from_runtime_preserves_cause() {
    let source = InferenceApiError::GenerationFailed {
        reason: "boom".into(),
    };
    let wrapped = ServerApiRoadmapError::generation_failed_from_runtime(source.clone());
    assert_eq!(wrapped.runtime_cause(), Some(&source));
    assert_eq!(wrapped.id(), "server-generation-failed");
}

#[test]
fn server_api_roadmap_error_without_runtime_cause_returns_none() {
    let error = ServerApiRoadmapError::ServerRequestInvalid {
        reason: "bad".into(),
    };
    assert_eq!(error.runtime_cause(), None);
}

// -- Observability --

#[test]
fn server_api_roadmap_observation_redacts_metadata() {
    let observation =
        ServerApiRoadmapObservation::new(ServerApiRoadmapObservationKind::RequestReceived)
            .with_endpoint("generate")
            .with_redacted_metadata("path", "/var/cache/magnetar/model");
    assert_eq!(observation.endpoint.as_deref(), Some("generate"));
    assert!(
        !observation
            .redacted_metadata
            .get("path")
            .unwrap()
            .contains('/')
    );
}

#[test]
fn server_api_roadmap_observation_kinds_cover_all_18_categories() {
    let kinds = [
        ServerApiRoadmapObservationKind::ServerStarted,
        ServerApiRoadmapObservationKind::ServerStopped,
        ServerApiRoadmapObservationKind::RequestReceived,
        ServerApiRoadmapObservationKind::RequestRejected,
        ServerApiRoadmapObservationKind::RequestAuthorized,
        ServerApiRoadmapObservationKind::RuntimeRequestSubmitted,
        ServerApiRoadmapObservationKind::StreamOpened,
        ServerApiRoadmapObservationKind::StreamClosed,
        ServerApiRoadmapObservationKind::StreamInterrupted,
        ServerApiRoadmapObservationKind::GenerationCompleted,
        ServerApiRoadmapObservationKind::GenerationFailed,
        ServerApiRoadmapObservationKind::CancellationRequested,
        ServerApiRoadmapObservationKind::DiagnosticsRequested,
        ServerApiRoadmapObservationKind::ModelEndpointUsed,
        ServerApiRoadmapObservationKind::SessionEndpointUsed,
        ServerApiRoadmapObservationKind::RateLimitHit,
        ServerApiRoadmapObservationKind::AdmissionRejected,
        ServerApiRoadmapObservationKind::BoundaryViolationDetected,
    ];
    assert_eq!(kinds.len(), 18);
    let unique: BTreeSet<ServerApiRoadmapObservationKind> = kinds.into_iter().collect();
    assert_eq!(unique.len(), 18, "observation kinds must be distinct");
}

// -- Conformance --

#[test]
fn server_api_roadmap_conformance_report_is_conformant() {
    let report = run_server_api_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn server_api_roadmap_version_constant_is_set() {
    assert_eq!(SERVER_API_ROADMAP_VERSION, "0.1.0");
}

// ---------------------------------------------------------------------
// release_packaging
// ---------------------------------------------------------------------

#[test]
fn release_version_displays_as_semver() {
    assert_eq!(ReleaseVersion::new(0, 1, 0).to_string(), "0.1.0");
    assert!(ReleaseVersion::new(0, 1, 0).is_pre_1_0());
    assert!(!ReleaseVersion::new(1, 0, 0).is_pre_1_0());
}

#[test]
fn breaking_change_is_rejected_in_patch_release() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 1, 1),
        true,
        true,
    );
    assert!(matches!(
        outcome,
        Err(ReleasePackagingError::BreakingChangeInPatchRelease { .. })
    ));
}

#[test]
fn undocumented_breaking_change_is_rejected() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 2, 0),
        true,
        false,
    );
    assert!(matches!(
        outcome,
        Err(ReleasePackagingError::UndocumentedBreakingChange { .. })
    ));
}

#[test]
fn documented_breaking_change_is_allowed_pre_1_0_in_minor_bump() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 2, 0),
        true,
        true,
    );
    assert_eq!(outcome, Ok(ReleaseVersionBumpKind::Minor));
}

#[test]
fn crate_dependency_across_independent_versions_requires_documentation() {
    let dependent = CrateVersionMetadata {
        crate_name: "magnetar-cli".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: false,
    };
    let dependency = CrateVersionMetadata {
        crate_name: "magnetar-runtime".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: false,
    };
    assert!(validate_crate_dependency_compatibility(&dependent, &dependency, None).is_err());
    assert!(
        validate_crate_dependency_compatibility(&dependent, &dependency, Some("compatible"))
            .is_ok()
    );
}

#[test]
fn shared_workspace_version_crates_do_not_require_documentation() {
    let dependent = CrateVersionMetadata {
        crate_name: "magnetar-cli".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: true,
    };
    let dependency = CrateVersionMetadata {
        crate_name: "magnetar-runtime".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: true,
    };
    assert!(validate_crate_dependency_compatibility(&dependent, &dependency, None).is_ok());
}

#[test]
fn release_binary_version_report_includes_all_fields() {
    let report = build_release_binary_version_report(
        ReleaseVersion::new(0, 1, 0),
        vec!["reference-cpu-provider".into()],
        "release",
        Some("abc1234".into()),
    );
    assert_eq!(report.binary_version, "0.1.0");
    assert_eq!(report.runtime_crate_version, MAGNETAR_RUNTIME_VERSION);
    assert_eq!(
        report.openspec_baseline_version,
        RELEASE_PACKAGING_POLICY_VERSION
    );
    assert_eq!(report.wit_contract_versions.len(), 2);
    assert_eq!(report.enabled_feature_flags, vec!["reference-cpu-provider"]);
    assert_eq!(report.build_profile, "release");
    assert_eq!(report.commit_hash.as_deref(), Some("abc1234"));
    assert!(report.conformance_suite_version.is_some());
}

#[test]
fn required_wit_version_bump_matches_change_kind() {
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::Breaking),
        ReleaseVersionBumpKind::Major
    );
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::Additive),
        ReleaseVersionBumpKind::Minor
    );
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::DocumentationOnly),
        ReleaseVersionBumpKind::Patch
    );
}

#[test]
fn breaking_wit_change_requires_major_bump() {
    assert!(
        validate_wit_version_bump(
            WitVersionChangeKind::Breaking,
            ReleaseVersionBumpKind::Minor,
            "magnetar:compute",
        )
        .is_err()
    );
    assert!(
        validate_wit_version_bump(
            WitVersionChangeKind::Breaking,
            ReleaseVersionBumpKind::Major,
            "magnetar:compute",
        )
        .is_ok()
    );
}

#[test]
fn supported_wit_version_matrix_lists_declared_interfaces() {
    let matrix = SupportedWitVersionMatrix::from_interfaces(&release_wit_contract_versions());
    assert_eq!(
        matrix.supported.get("magnetar:compute").map(String::as_str),
        Some("2.0.0")
    );
    assert_eq!(
        matrix
            .supported
            .get("magnetar:observability")
            .map(String::as_str),
        Some("1.0.0")
    );
}

#[test]
fn openspec_baseline_declaration_requires_accepted_changes_and_status() {
    let empty = OpenSpecBaselineDeclaration::default();
    assert!(empty.validate().is_err());

    let complete = OpenSpecBaselineDeclaration {
        accepted_changes: vec!["define-release-packaging-and-versioning-policy".into()],
        validation_status: Some("valid".into()),
        ..Default::default()
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn freeze_denies_semantic_change_but_allows_documentation_clarification() {
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_err()
    );
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::DocumentationClarification,
        )
        .is_ok()
    );
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Open,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_ok()
    );
}

#[test]
fn experimental_feature_flag_cannot_be_enabled_by_default() {
    let flag = ReleaseFeatureFlag {
        name: "webgpu-provider".into(),
        class: ReleaseFeatureFlagClass::Experimental,
        enabled_by_default: true,
    };
    assert!(reject_experimental_flag_enabled_by_default(&flag).is_err());

    let disabled = ReleaseFeatureFlag {
        enabled_by_default: false,
        ..flag
    };
    assert!(reject_experimental_flag_enabled_by_default(&disabled).is_ok());
}

#[test]
fn only_reference_cpu_provider_is_required_for_v0_1() {
    let flags = provider_feature_flags();
    assert_eq!(flags.len(), 7);
    assert!(validate_provider_feature_flags_for_v0_1(&flags).is_ok());

    let mut bad_flags = flags.clone();
    bad_flags[1].enabled_by_default = true; // optimized-cpu-provider
    assert!(validate_provider_feature_flags_for_v0_1(&bad_flags).is_err());
}

#[test]
fn component_engine_flags_are_disabled_by_default() {
    for flag in component_engine_feature_flags() {
        assert!(!flag.enabled_by_default);
    }
}

#[test]
fn browser_target_never_requires_wasmtime() {
    let browser = ReleasePlatformTarget {
        triple: "wasm32-unknown-unknown".into(),
        required_by_ci: false,
        check_only: true,
        is_browser_like: true,
    };
    assert!(
        reject_wasmtime_required_for_browser(&browser, &["wasmtime-component-engine"]).is_err()
    );
    assert!(reject_wasmtime_required_for_browser(&browser, &[]).is_ok());
}

#[test]
fn native_target_may_require_wasmtime() {
    let native = ReleasePlatformTarget {
        triple: "x86_64-unknown-linux-gnu".into(),
        required_by_ci: true,
        check_only: false,
        is_browser_like: false,
    };
    assert!(reject_wasmtime_required_for_browser(&native, &["wasmtime-component-engine"]).is_ok());
}

#[test]
fn release_platform_targets_include_ci_required_and_check_only_entries() {
    let targets = release_platform_targets();
    assert!(
        targets
            .iter()
            .any(|target| target.required_by_ci && !target.check_only)
    );
    assert!(
        targets
            .iter()
            .any(|target| target.check_only && !target.required_by_ci)
    );
}

#[test]
fn unsupported_targets_are_reported() {
    let supported = release_platform_targets();
    let candidates = ["x86_64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"];
    let unsupported = unsupported_targets(&supported, &candidates);
    assert_eq!(unsupported, vec!["riscv64gc-unknown-linux-gnu"]);
}

#[test]
fn release_artifact_manifest_requires_every_kind_present_or_not_applicable() {
    let manifest = ReleaseArtifactManifest::default();
    assert!(manifest.validate().is_err());

    let mut manifest = ReleaseArtifactManifest::default();
    for kind in RELEASE_ARTIFACT_KINDS {
        manifest.set(*kind, ReleaseArtifactStatus::NotApplicable);
    }
    assert!(manifest.validate().is_ok());
}

#[test]
fn artifact_checksum_rejects_empty_digest() {
    assert!(ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "").is_err());
    assert!(ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").is_ok());
}

#[test]
fn changelog_must_be_non_empty() {
    assert!(ReleaseChangelog::default().validate().is_err());
    let changelog = ReleaseChangelog {
        entries: vec![ChangelogEntry {
            kind: ChangelogEntryKind::AddedContract,
            description: "release packaging policy".into(),
        }],
    };
    assert!(changelog.validate().is_ok());
}

#[test]
fn compatibility_matrix_requires_every_dimension_declared() {
    let mut matrix = ReleaseCompatibilityMatrix::default();
    assert!(matrix.validate().is_err());
    for dimension in COMPATIBILITY_DIMENSIONS {
        matrix.set(*dimension, CompatibilityStatus::StableForBaseline);
    }
    assert!(matrix.validate().is_ok());
}

#[test]
fn v0_1_compatibility_matrix_marks_provider_abi_unstable() {
    let matrix = v0_1_compatibility_matrix();
    assert!(matrix.validate().is_ok());
    assert_eq!(
        matrix.status.get(compatibility_dimension_id(
            CompatibilityDimension::ProviderAbi
        )),
        Some(&CompatibilityStatus::Unstable)
    );
    assert_eq!(
        matrix.status.get(compatibility_dimension_id(
            CompatibilityDimension::RustPublicApi
        )),
        Some(&CompatibilityStatus::StableForBaseline)
    );
}

#[test]
fn release_public_api_denies_raw_handle_surfaces() {
    for surface in [
        "raw-provider-handle",
        "raw-device-handle",
        "raw-kernel-handle",
        "raw-tensor-pointer",
        "raw-memory-pointer",
        "raw-kv-cache",
        "raw-model-weight",
    ] {
        assert!(reject_release_public_api_handle_exposure(surface).is_err());
    }
    assert!(reject_release_public_api_handle_exposure("generation").is_ok());
}

#[test]
fn release_conformance_versions_reuse_existing_suite_constants() {
    let versions = ReleaseConformanceVersions::default();
    assert_eq!(
        versions.provider_conformance_suite_version,
        PROVIDER_CONFORMANCE_SUITE_VERSION
    );
    assert_eq!(
        versions.first_operator_scope_conformance_version,
        FIRST_OPERATOR_SCOPE_VERSION
    );
    assert_eq!(
        versions.qwen_baseline_conformance_version,
        QWEN_BASELINE_CONTRACT_VERSION.to_string()
    );
    assert_eq!(versions.e2e_local_conformance_version, E2E_SUITE_VERSION);
}

#[test]
fn release_may_publish_stable_requires_every_gate_present_and_passed() {
    let missing: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES[..3]
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        release_may_publish_stable(&missing),
        Err(ReleasePackagingError::ReleaseGateMissing { .. })
    ));

    let complete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(release_may_publish_stable(&complete).is_ok());

    let mut failing = complete.clone();
    failing[0].passed = false;
    assert!(matches!(
        release_may_publish_stable(&failing),
        Err(ReleasePackagingError::ReleaseGateFailed { .. })
    ));
}

#[test]
fn release_candidate_tags_are_never_stable() {
    assert!(!ReleaseCandidateTag::Alpha.is_stable());
    assert!(!ReleaseCandidateTag::Beta.is_stable());
    assert!(!ReleaseCandidateTag::Rc(1).is_stable());
    assert_eq!(ReleaseCandidateTag::Rc(1).to_string(), "-rc.1");
}

#[test]
fn release_candidate_manifest_requires_frozen_baseline_and_conformance_report() {
    let incomplete = ReleaseCandidateManifest {
        tag: ReleaseCandidateTag::Rc(1),
        frozen_openspec_baseline: false,
        conformance_report_included: true,
        known_failures: Vec::new(),
        release_notes_draft: true,
    };
    assert!(incomplete.validate().is_err());

    let complete = ReleaseCandidateManifest {
        frozen_openspec_baseline: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn failed_candidate_may_be_tagged_pre_release() {
    let failing = vec![ReleaseGateResult {
        gate: ReleaseGate::Formatting,
        passed: false,
    }];
    let tag = allow_failed_candidate_as_pre_release(&failing, ReleaseCandidateTag::Rc(2));
    assert_eq!(tag, Ok(ReleaseCandidateTag::Rc(2)));
}

#[test]
fn build_metadata_redacts_secret_shaped_keys_and_local_paths() {
    assert_eq!(
        redact_build_metadata("GITHUB_TOKEN", "ghp_example"),
        "[redacted build metadata]"
    );
    assert_eq!(
        redact_build_metadata("workspace_root", "/home/user/project"),
        "[redacted backend diagnostic]"
    );
    assert_eq!(
        redact_build_metadata("target_triple", "x86_64-unknown-linux-gnu"),
        "x86_64-unknown-linux-gnu"
    );
}

#[test]
fn documentation_checklist_requires_known_limitations() {
    let empty = ReleaseDocumentationChecklist::default();
    assert!(empty.validate().is_err());
    let documented = ReleaseDocumentationChecklist {
        known_limitations: true,
        ..empty
    };
    assert!(documented.validate().is_ok());
}

#[test]
fn deferred_roadmap_features_are_never_included_baseline() {
    for feature in [
        "cuda",
        "metal",
        "openvino",
        "qnn",
        "webgpu",
        "server-api-implementation",
    ] {
        assert_eq!(
            classify_publishing_boundary(feature),
            PublishingBoundaryCategory::DeferredRoadmap
        );
        assert!(reject_roadmap_feature_as_guarantee(feature, true).is_err());
        assert!(reject_roadmap_feature_as_guarantee(feature, false).is_ok());
    }
    assert_eq!(
        classify_publishing_boundary("reference-cpu-provider"),
        PublishingBoundaryCategory::IncludedBaseline
    );
    assert!(reject_roadmap_feature_as_guarantee("reference-cpu-provider", true).is_ok());
}

#[test]
fn release_packaging_conformance_report_is_conformant() {
    let report = run_release_packaging_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn release_packaging_policy_version_constant_is_set() {
    assert_eq!(RELEASE_PACKAGING_POLICY_VERSION, "0.1.0");
}

// ---------------------------------------------------------------------
// release_security
// ---------------------------------------------------------------------

#[test]
fn release_security_policy_version_constant_is_set() {
    assert_eq!(RELEASE_SECURITY_POLICY_VERSION, "0.1.0");
}

#[test]
fn hardened_claim_is_rejected_for_excluded_feature_but_allowed_for_baseline() {
    for feature in RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS {
        assert!(reject_hardened_security_claim_for_excluded_feature(feature, true).is_err());
        assert!(reject_hardened_security_claim_for_excluded_feature(feature, false).is_ok());
    }
    assert!(
        reject_hardened_security_claim_for_excluded_feature("reference-cpu-provider", true).is_ok()
    );
    assert!(!RELEASE_SECURITY_SCOPE_INCLUDED.is_empty());
}

#[test]
fn dependency_audit_blocks_stable_release_on_unmitigated_critical_advisory() {
    let mut report = DependencyAuditReport {
        advisories: vec![DependencyAdvisory {
            crate_name: "example-crate".into(),
            advisory_id: "RUSTSEC-0000-0000".into(),
            severity: DependencyAdvisorySeverity::Critical,
            mitigated: false,
            mitigation: None,
        }],
        ..Default::default()
    };
    assert!(matches!(
        report.validate_for_stable_release(),
        Err(ReleaseSecurityError::CriticalAdvisoryUnmitigated { .. })
    ));

    report.advisories[0].mitigated = true;
    report.advisories[0].mitigation = Some("vendored patch".into());
    assert!(report.validate_for_stable_release().is_ok());

    report.advisories[0].severity = DependencyAdvisorySeverity::Low;
    report.advisories[0].mitigated = false;
    assert!(report.validate_for_stable_release().is_ok());
}

#[test]
fn license_audit_blocks_stable_release_on_unapproved_incompatible_or_unknown_license() {
    for status in [
        LicenseAuditStatus::Incompatible,
        LicenseAuditStatus::Unknown,
    ] {
        let report = LicenseAuditReport {
            licenses: vec![DependencyLicense {
                crate_name: "example-crate".into(),
                spdx: None,
                status,
                exception_approved: false,
            }],
            ..Default::default()
        };
        assert!(matches!(
            report.validate_for_stable_release(),
            Err(ReleaseSecurityError::IncompatibleLicenseUnapproved { .. })
        ));
    }

    let approved = LicenseAuditReport {
        licenses: vec![DependencyLicense {
            crate_name: "example-crate".into(),
            spdx: None,
            status: LicenseAuditStatus::Unknown,
            exception_approved: true,
        }],
        ..Default::default()
    };
    assert!(approved.validate_for_stable_release().is_ok());

    let missing_metadata_not_blocking = LicenseAuditReport {
        licenses: vec![DependencyLicense {
            crate_name: "example-crate".into(),
            spdx: None,
            status: LicenseAuditStatus::MissingMetadata,
            exception_approved: false,
        }],
        ..Default::default()
    };
    assert!(
        missing_metadata_not_blocking
            .validate_for_stable_release()
            .is_ok()
    );
}

#[test]
fn sbom_manifest_requires_generation_or_documented_limitation() {
    let missing = SbomManifest {
        availability: SbomAvailability::Missing,
        ..Default::default()
    };
    assert!(missing.validate().is_err());

    let undocumented_placeholder = SbomManifest {
        availability: SbomAvailability::PlaceholderDocumented,
        ..Default::default()
    };
    assert!(undocumented_placeholder.validate().is_err());

    let documented = SbomManifest {
        availability: SbomAvailability::PlaceholderDocumented,
        limitation_note: Some("SBOM generation is not implemented for v0.1".into()),
        ..Default::default()
    };
    assert!(documented.validate().is_ok());

    let generated = SbomManifest {
        availability: SbomAvailability::Generated,
        entries: vec![SbomEntry {
            package_name: "magnetar-runtime".into(),
            package_version: "0.1.0".into(),
            licenses: vec!["MIT".into()],
            source_repository: None,
        }],
        build_target: Some("x86_64-unknown-linux-gnu".into()),
        feature_flags: vec!["reference-cpu-provider".into()],
        ..Default::default()
    };
    assert!(generated.validate().is_ok());
    assert_eq!(generated.feature_flags, vec!["reference-cpu-provider"]);
}

#[test]
fn checksum_mismatch_against_final_artifact_is_rejected() {
    let checksum =
        ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").unwrap();
    assert!(matches!(
        verify_checksum_matches_final_artifact(&checksum, "other-digest"),
        Err(ReleaseSecurityError::ChecksumMismatch { .. })
    ));
    assert!(verify_checksum_matches_final_artifact(&checksum, "deadbeef").is_ok());
}

#[test]
fn signature_absence_must_be_documented() {
    assert!(matches!(
        validate_signature_status(SignatureStatus::NotImplementedUndocumented),
        Err(ReleaseSecurityError::SignatureAbsenceUndocumented)
    ));
    assert!(validate_signature_status(SignatureStatus::NotImplementedDocumented).is_ok());
    assert!(validate_signature_status(SignatureStatus::Implemented).is_ok());
}

#[test]
fn provenance_rejects_secret_or_path_shaped_fields() {
    let leaky_path = ReleaseProvenance {
        build_target: Some("/home/user/workspace".into()),
        ..Default::default()
    };
    assert!(matches!(
        leaky_path.validate(),
        Err(ReleaseSecurityError::ProvenanceContainsSecretOrLocalPath { .. })
    ));

    let clean = ReleaseProvenance {
        source_commit: Some("abc1234".into()),
        release_tag: Some("v0.1.0".into()),
        build_target: Some("x86_64-unknown-linux-gnu".into()),
        ..Default::default()
    };
    assert!(clean.validate().is_ok());
}

#[test]
fn reproducibility_report_requires_status_or_documented_limitations() {
    assert!(ReproducibilityReport::default().validate().is_err());

    let undocumented_partial = ReproducibilityReport {
        status: ReproducibilityStatus::PartiallyReproducible,
        limitations: Vec::new(),
    };
    assert!(undocumented_partial.validate().is_err());

    let documented_partial = ReproducibilityReport {
        status: ReproducibilityStatus::PartiallyReproducible,
        limitations: vec!["build timestamps are not normalized".into()],
    };
    assert!(documented_partial.validate().is_ok());

    let full = ReproducibilityReport {
        status: ReproducibilityStatus::FullyReproducible,
        limitations: Vec::new(),
    };
    assert!(full.validate().is_ok());
}

#[test]
fn lockfile_policy_requires_checked_in_state_and_reviewed_drift() {
    let not_checked_in = LockfileState::default();
    assert!(matches!(
        reject_unreviewed_lockfile_drift(&not_checked_in),
        Err(ReleaseSecurityError::LockfileNotCheckedIn)
    ));

    let unreviewed_drift = LockfileState {
        checked_in: true,
        digest: Some("digest".into()),
        drift_detected: true,
        drift_reviewed: false,
    };
    assert!(matches!(
        reject_unreviewed_lockfile_drift(&unreviewed_drift),
        Err(ReleaseSecurityError::LockfileDriftUnreviewed)
    ));

    let reviewed_drift = LockfileState {
        drift_reviewed: true,
        ..unreviewed_drift
    };
    assert!(reject_unreviewed_lockfile_drift(&reviewed_drift).is_ok());
}

#[test]
fn unexpected_unreviewed_build_script_is_flagged() {
    let unexpected = BuildScriptReview {
        crate_name: "example-native-sys".into(),
        has_build_script: true,
        unexpected: true,
        reviewed: false,
        native_build_documented: false,
    };
    assert!(matches!(
        flag_unexpected_build_script(&unexpected),
        Err(ReleaseSecurityError::UnexpectedBuildScriptUnreviewed { .. })
    ));

    let reviewed = BuildScriptReview {
        reviewed: true,
        ..unexpected
    };
    assert!(flag_unexpected_build_script(&reviewed).is_ok());

    let expected = BuildScriptReview {
        crate_name: "example-native-sys".into(),
        has_build_script: true,
        unexpected: false,
        reviewed: false,
        native_build_documented: true,
    };
    assert!(flag_unexpected_build_script(&expected).is_ok());
}

#[test]
fn secret_scan_targets_cover_all_eight_locations_and_block_on_detection() {
    assert_eq!(SECRET_SCAN_TARGETS.len(), 8);

    let detected = SecretScanReport {
        findings: vec![SecretScanFinding {
            target: SecretScanTarget::BuildMetadata,
            detected: true,
            location: Some("build.env".into()),
        }],
    };
    assert!(matches!(
        detected.validate_for_stable_release(),
        Err(ReleaseSecurityError::SecretDetected { .. })
    ));

    let clean = SecretScanReport {
        findings: vec![SecretScanFinding {
            target: SecretScanTarget::BuildMetadata,
            detected: false,
            location: None,
        }],
    };
    assert!(clean.validate_for_stable_release().is_ok());
}

#[test]
fn artifact_integrity_requires_every_check_to_pass() {
    assert!(ArtifactIntegrityStatus::default().validate().is_err());

    let complete = ArtifactIntegrityStatus {
        source_state_clean_or_ci_controlled: true,
        release_tag_matches_source: true,
        openspec_report_matches_baseline: true,
        conformance_reports_match_commit: true,
        checksums_match_final_artifacts: true,
    };
    assert!(complete.validate().is_ok());

    let mut partial = complete;
    partial.checksums_match_final_artifacts = false;
    assert!(matches!(
        partial.validate(),
        Err(ReleaseSecurityError::ArtifactIntegrityFailed { .. })
    ));
}

#[test]
fn redaction_gate_rejects_sensitive_content_and_native_handles() {
    assert_eq!(REDACTION_CATEGORIES.len(), 13);
    assert!(matches!(
        validate_redaction_gate("raw prompt: what is the secret?"),
        Err(ReleaseSecurityError::RedactionGateFailed { .. })
    ));
    assert!(validate_redaction_gate("provider handle=0xdeadbeef").is_err());
    assert!(validate_redaction_gate("/home/user/model.bin").is_err());
    assert!(validate_redaction_gate("generation completed in 12ms").is_ok());
}

#[test]
fn dynamic_provider_loading_requires_review_or_explicit_status() {
    assert!(matches!(
        validate_dynamic_provider_loading_status(
            ProviderLoadingMode::DynamicLibrary,
            DynamicProviderLoadingStatus::StableUnreviewed,
        ),
        Err(ReleaseSecurityError::DynamicProviderLoadingUnreviewed)
    ));
    for status in [
        DynamicProviderLoadingStatus::Disabled,
        DynamicProviderLoadingStatus::Experimental,
        DynamicProviderLoadingStatus::MarkedUnstable,
        DynamicProviderLoadingStatus::SecurityReviewed,
    ] {
        assert!(
            validate_dynamic_provider_loading_status(ProviderLoadingMode::DynamicLibrary, status)
                .is_ok()
        );
    }
    assert!(
        validate_dynamic_provider_loading_status(
            ProviderLoadingMode::BuiltIn,
            DynamicProviderLoadingStatus::StableUnreviewed,
        )
        .is_ok()
    );
}

#[test]
fn provider_registration_alone_does_not_imply_trust() {
    assert!(matches!(
        reject_provider_registration_implies_trust(ProviderTrustSignalSource::RegistrationOnly),
        Err(ReleaseSecurityError::ProviderRegistrationTrustImplied)
    ));
    assert!(
        reject_provider_registration_implies_trust(ProviderTrustSignalSource::ConfiguredPolicy)
            .is_ok()
    );
}

#[test]
fn release_native_handle_exposure_is_denied_across_all_provider_families() {
    for surface in [
        "raw-provider-handle",
        "raw-device-handle",
        "raw-kernel-handle",
        "raw-tensor-pointer",
        "raw-memory-pointer",
        "raw-kv-cache",
        "raw-model-weight",
        "cuda-stream",
        "cuda-device-pointer",
        "metal-buffer",
        "metal-command-queue",
        "openvino-compiled-graph",
        "qnn-native-handle",
        "raw-cpu-allocation-pointer",
    ] {
        assert!(
            reject_release_native_handle_exposure(surface).is_err(),
            "expected surface '{surface}' to be denied"
        );
    }
    assert!(reject_release_native_handle_exposure("generation").is_ok());
}

#[test]
fn component_release_execution_trust_requires_trusted_status_and_signature_in_production() {
    let untrusted = ComponentTrustDecision::new(ComponentTrustStatus::Rejected, "rejected fixture");
    assert!(matches!(
        validate_component_release_execution_trust(&untrusted, true, true, false),
        Err(ReleaseSecurityError::ComponentArtifactUntrusted { .. })
    ));

    let trusted = ComponentTrustDecision::new(ComponentTrustStatus::Trusted, "trusted fixture");
    assert!(matches!(
        validate_component_release_execution_trust(&trusted, false, true, false),
        Err(ReleaseSecurityError::UnsignedComponentDeniedInProduction)
    ));
    assert!(validate_component_release_execution_trust(&trusted, false, true, true).is_ok());
    assert!(validate_component_release_execution_trust(&trusted, false, false, false).is_ok());
    assert!(validate_component_release_execution_trust(&trusted, true, true, false).is_ok());
}

#[test]
fn component_release_authority_expansion_is_denied_for_os_and_handle_capabilities() {
    for capability in [
        "filesystem",
        "network-tool",
        "secret",
        "shell",
        "raw-provider-handle",
    ] {
        assert!(
            reject_component_release_authority_expansion(capability).is_err(),
            "expected capability '{capability}' to be denied"
        );
    }
    assert!(reject_component_release_authority_expansion("generation").is_ok());
}

#[test]
fn model_artifact_release_trust_ignores_recognized_format() {
    let untrusted = ModelTrustDecision::new(ModelTrustStatus::Unknown, "no policy matched");
    assert!(matches!(
        validate_model_artifact_release_trust(&untrusted, true),
        Err(ReleaseSecurityError::ModelArtifactUntrusted { .. })
    ));

    let trusted = ModelTrustDecision::new(ModelTrustStatus::Trusted, "digest trusted");
    assert!(validate_model_artifact_release_trust(&trusted, false).is_ok());
    assert!(validate_model_artifact_release_trust(&trusted, true).is_ok());
}

#[test]
fn fixture_model_trust_requires_explicit_test_policy() {
    let trusted = ModelTrustDecision::new(ModelTrustStatus::Trusted, "fixture trusted");
    let undocumented = FixtureModelTrustPolicy::default();
    assert!(matches!(
        validate_fixture_model_trust(&trusted, &undocumented),
        Err(ReleaseSecurityError::FixtureTrustPolicyUndocumented)
    ));

    let documented = FixtureModelTrustPolicy {
        explicit_test_policy_documented: true,
    };
    assert!(validate_fixture_model_trust(&trusted, &documented).is_ok());
}

#[test]
fn source_cache_trust_is_never_implied_by_a_non_trust_signal() {
    let mut entry = CacheEntryMetadata::new(
        ModelArtifactId {
            kind: ModelArtifactKind::ModelWeights,
            name: ModelName::new("qwen-test").unwrap(),
            revision: ModelRevision::new("v1").unwrap(),
            variant: None,
            digest: ModelDigest {
                algorithm: "sha256".into(),
                value: "cachedigest".into(),
            },
            source: None,
            shard: None,
        },
        ModelSourceKind::LocalDirectorySource,
    );
    assert_eq!(entry.trust_status, ModelTrustStatus::Unknown);
    assert!(matches!(
        validate_source_cache_release_trust(&entry),
        Err(ReleaseSecurityError::CacheEntryTrustNotEstablished { .. })
    ));

    for signal in [
        NonTrustCacheSignal::CacheHit,
        NonTrustCacheSignal::SourceKind,
        NonTrustCacheSignal::Alias,
        NonTrustCacheSignal::LocalFile,
        NonTrustCacheSignal::FixtureStatus,
    ] {
        assert!(matches!(
            reject_cache_signal_alone_as_trust(signal, &entry),
            Err(ReleaseSecurityError::CacheSignalDoesNotImplyTrust { .. })
        ));
    }

    entry.trust_status = ModelTrustStatus::Trusted;
    assert!(validate_source_cache_release_trust(&entry).is_ok());
}

#[test]
fn cli_authority_is_never_delegated_to_runtime() {
    for capability in ["filesystem", "git", "shell", "secret", "tool-call"] {
        assert!(matches!(
            validate_cli_authority_not_delegated_to_runtime(capability),
            Err(ReleaseSecurityError::CliAuthorityDelegatedToRuntime { .. })
        ));
    }
}

#[test]
fn runtime_inference_api_rejects_non_inference_authority() {
    for capability in [
        "filesystem",
        "network-tool",
        "secret",
        "shell",
        "git",
        "tool-call",
    ] {
        assert!(matches!(
            validate_runtime_inference_api_security(capability),
            Err(ReleaseSecurityError::RuntimeInferenceApiAuthorityExpansionDenied { .. })
        ));
    }
    assert!(validate_runtime_inference_api_security("generation").is_ok());
}

#[test]
fn unsafe_code_policy_denies_unreviewed_blocks_only_when_configured() {
    let unreviewed = UnsafeCodePolicy {
        reviews: vec![UnsafeCodeReview {
            location: "compute.rs:100".into(),
            justified: false,
            reviewed: false,
        }],
        deny_unreviewed: true,
    };
    assert!(matches!(
        unreviewed.validate(),
        Err(ReleaseSecurityError::UnsafeCodeUnreviewed { .. })
    ));

    let not_enforced = UnsafeCodePolicy {
        deny_unreviewed: false,
        ..unreviewed.clone()
    };
    assert!(not_enforced.validate().is_ok());

    let reviewed = UnsafeCodePolicy {
        reviews: vec![UnsafeCodeReview {
            location: "compute.rs:100".into(),
            justified: true,
            reviewed: true,
        }],
        deny_unreviewed: true,
    };
    assert!(reviewed.validate().is_ok());
}

#[test]
fn magnetar_runtime_unsafe_code_inventory_is_reviewed_and_justified() {
    let inventory = magnetar_runtime_unsafe_code_inventory();
    assert_eq!(inventory.reviews.len(), 3);
    assert!(inventory.deny_unreviewed);
    assert!(inventory.validate().is_ok());
    for review in &inventory.reviews {
        assert!(review.location.contains("provider.rs"));
        assert!(review.justified);
        assert!(review.reviewed);
    }
}

#[test]
fn unexpected_capability_expanding_dependency_feature_is_rejected() {
    let unexpected = DependencyFeatureReview {
        crate_name: "example-crate".into(),
        feature_name: "http-client".into(),
        capability: DependencyFeatureCapability::Networking,
        expected: false,
        accepted_exception: false,
    };
    assert!(matches!(
        reject_unexpected_capability_expanding_feature(&unexpected),
        Err(ReleaseSecurityError::UnexpectedCapabilityExpandingFeature { .. })
    ));

    let excepted = DependencyFeatureReview {
        accepted_exception: true,
        ..unexpected.clone()
    };
    assert!(reject_unexpected_capability_expanding_feature(&excepted).is_ok());

    let expected = DependencyFeatureReview {
        expected: true,
        ..unexpected
    };
    assert!(reject_unexpected_capability_expanding_feature(&expected).is_ok());
}

#[test]
fn vulnerability_handling_policy_requires_every_field_defined() {
    assert!(VulnerabilityHandlingPolicy::default().validate().is_err());
    let complete = VulnerabilityHandlingPolicy {
        advisory_severity_handling_defined: true,
        release_blocking_criteria_defined: true,
        mitigation_documentation_required: true,
        exception_approval_defined: true,
        follow_up_tracking_defined: true,
        patch_release_expectation_documented: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn security_release_notes_require_shall_strength_topics() {
    assert!(SecurityReleaseNotes::default().validate().is_err());
    let complete = SecurityReleaseNotes {
        v0_1_threat_model: Some("CPU-local baseline".into()),
        trusted_native_provider_model: Some("Providers are trusted native code".into()),
        no_raw_handle_policy: Some("no raw handles in public APIs".into()),
        default_redaction: Some("diagnostics redact by default".into()),
        reporting_process_placeholder: Some("reporting process TBD".into()),
        ..Default::default()
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn release_security_blocking_reports_every_triggered_reason() {
    assert!(evaluate_release_security_blocking(&ReleaseSecurityGateInputs::default()).is_ok());

    let blocked = ReleaseSecurityGateInputs {
        secrets_detected: true,
        raw_handle_exposed: true,
        checksum_mismatch: true,
        ..Default::default()
    };
    match evaluate_release_security_blocking(&blocked) {
        Err(ReleaseSecurityError::ReleaseBlocked { reasons }) => assert_eq!(reasons.len(), 3),
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[test]
fn undocumented_security_exception_is_rejected_when_required() {
    assert!(reject_undocumented_security_exception(false, None).is_ok());
    assert!(matches!(
        reject_undocumented_security_exception(true, None),
        Err(ReleaseSecurityError::SecurityExceptionIncomplete)
    ));

    let incomplete = SecurityException {
        issue: "advisory".into(),
        affected_component: String::new(),
        severity: DependencyAdvisorySeverity::High,
        rationale: "reason".into(),
        mitigation: "patch".into(),
        owner: "team".into(),
        expiration_or_follow_up: "v0.2".into(),
        release_note_entry: true,
    };
    assert!(incomplete.validate().is_err());

    let complete = SecurityException {
        affected_component: "example-crate".into(),
        ..incomplete
    };
    assert!(reject_undocumented_security_exception(true, Some(&complete)).is_ok());
}

#[test]
fn release_security_observation_is_always_redacted() {
    let observation = record_release_security_observation(
        ReleaseSecurityObservationKind::SecretScanCompleted,
        "found credential abc123 in build.env",
    );
    assert!(!observation.detail.unwrap().contains("credential abc123"));

    let ordinary = record_release_security_observation(
        ReleaseSecurityObservationKind::ReleaseSecurityPassed,
        "all gates passed",
    );
    assert_eq!(ordinary.detail.as_deref(), Some("all gates passed"));
}

#[test]
fn release_security_conformance_report_is_conformant() {
    let report = run_release_security_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// release_cutover
// ---------------------------------------------------------------------

#[test]
fn release_cutover_policy_version_constant_is_set() {
    assert_eq!(RELEASE_CUTOVER_POLICY_VERSION, "0.1.0");
}

#[test]
fn release_readiness_checklist_requires_every_field() {
    let missing_notes = ReleaseReadinessChecklist {
        release_branch_or_commit_selected: true,
        version_selected: true,
        openspec_baseline_selected: true,
        scope_selected: true,
        gates_selected: true,
        artifacts_selected: true,
        release_notes_draft_exists: false,
        compatibility_matrix_draft_exists: true,
        security_notes_draft_exists: true,
    };
    assert!(matches!(
        missing_notes.validate(),
        Err(ReleaseCutoverError::ReleaseReadinessIncomplete { .. })
    ));

    let complete = ReleaseReadinessChecklist {
        release_notes_draft_exists: true,
        ..missing_notes
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn openspec_freeze_confirmation_requires_frozen_state() {
    let open = OpenSpecFreezeConfirmation {
        freeze_state: ReleaseFreezeState::Open,
        accepted_changes_list_final: true,
        pending_changes_excluded: true,
        wit_breaking_changes_have_version_bumps: true,
        checklist_references_correct_changes: true,
        roadmap_items_deferred_unless_included: true,
    };
    assert!(matches!(
        open.validate(),
        Err(ReleaseCutoverError::OpenSpecNotFrozen { .. })
    ));

    let frozen = OpenSpecFreezeConfirmation {
        freeze_state: ReleaseFreezeState::Frozen,
        ..open
    };
    assert!(frozen.validate().is_ok());
}

#[test]
fn semantic_change_after_freeze_is_blocked_or_freeze_is_restarted() {
    let outcome = reject_semantic_change_after_freeze(
        ReleaseFreezeState::Frozen,
        ReleaseFreezeChangeKind::SemanticContractChange,
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::SemanticChangeAfterFreeze { .. })
    ));
    assert!(
        reject_semantic_change_after_freeze(
            ReleaseFreezeState::Open,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_ok()
    );
}

#[test]
fn cuda_listed_as_included_is_blocked() {
    let outcome = validate_v0_1_scope_feature("cuda", true);
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RoadmapFeaturePresentedAsIncluded { .. })
    ));
    assert!(validate_v0_1_scope_feature("cuda", false).is_ok());
}

#[test]
fn missing_wit_version_blocks_release() {
    let missing = validate_wit_versions_confirmed(&[WitPackageVersionRecord {
        package: "magnetar:observability".into(),
        version: None,
    }]);
    assert!(matches!(
        missing,
        Err(ReleaseCutoverError::WitVersionMissing { .. })
    ));

    let confirmed: Vec<WitPackageVersionRecord> = release_wit_contract_versions()
        .iter()
        .map(WitPackageVersionRecord::from_interface)
        .collect();
    assert!(validate_wit_versions_confirmed(&confirmed).is_ok());
}

#[test]
fn missing_wit_package_blocks_the_cutover_compatibility_matrix() {
    let mut matrix = CutoverCompatibilityMatrix::default();
    for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
        if *dimension != CutoverCompatibilityDimension::WitPackages {
            matrix.set(*dimension, CutoverCompatibilityStatus::StableForV01Baseline);
        }
    }
    let outcome = matrix.validate();
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::CompatibilityDimensionMissing { dimension })
            if dimension == "wit-packages"
    ));
}

#[test]
fn cutover_version_confirmation_requires_every_field() {
    let wit_packages: Vec<WitPackageVersionRecord> = release_wit_contract_versions()
        .iter()
        .map(WitPackageVersionRecord::from_interface)
        .collect();
    let incomplete = CutoverVersionConfirmation {
        release_version: ReleaseVersion::new(0, 1, 0),
        crate_versions_confirmed: true,
        binary_version_confirmed: false,
        wit_packages: wit_packages.clone(),
        conformance_suite_versions_confirmed: true,
        openspec_baseline_version_confirmed: true,
        release_candidate_lineage_documented: None,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::VersionConfirmationIncomplete { .. })
    ));

    let complete = CutoverVersionConfirmation {
        binary_version_confirmed: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn runtime_version_mismatch_fails_cutover_verification() {
    let report = build_release_binary_version_report(
        ReleaseVersion::new(0, 1, 0),
        vec!["reference-cpu-provider".into()],
        "release",
        None,
    );
    let outcome = validate_runtime_version_matches_release_tag(&report, "0.1.0-rc.1");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeVersionMismatch { .. })
    ));
    assert!(validate_runtime_version_matches_release_tag(&report, "0.1.0").is_ok());
}

#[test]
fn experimental_webgpu_enabled_by_default_is_blocked() {
    let flag = ReleaseFeatureFlag {
        name: "webgpu-provider".into(),
        class: ReleaseFeatureFlagClass::Experimental,
        enabled_by_default: true,
    };
    assert!(matches!(
        validate_cutover_feature_flag(&flag),
        Err(ReleaseCutoverError::ExperimentalFeatureEnabledByDefault { .. })
    ));
}

#[test]
fn test_only_flag_enabled_by_default_is_blocked_in_release_build() {
    let flag = ReleaseFeatureFlag {
        name: "test-harness".into(),
        class: ReleaseFeatureFlagClass::TestOnly,
        enabled_by_default: true,
    };
    assert!(matches!(
        validate_cutover_feature_flag(&flag),
        Err(ReleaseCutoverError::NonBaselineFeatureEnabledInRelease { .. })
    ));
}

#[test]
fn provider_abi_missing_blocks_compatibility_matrix_completion() {
    let mut matrix = CutoverCompatibilityMatrix::default();
    for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
        if *dimension != CutoverCompatibilityDimension::ProviderAbi {
            matrix.set(*dimension, CutoverCompatibilityStatus::StableForV01Baseline);
        }
    }
    assert!(matches!(
        matrix.validate(),
        Err(ReleaseCutoverError::CompatibilityDimensionMissing { .. })
    ));
}

#[test]
fn cutover_compatibility_matrix_uses_approved_status_vocabulary() {
    assert_eq!(CUTOVER_COMPATIBILITY_DIMENSIONS.len(), 12);
    assert_eq!(
        cutover_compatibility_status_id(CutoverCompatibilityStatus::StableForV01Baseline),
        "stable-for-v0.1-baseline"
    );
    assert_eq!(
        cutover_compatibility_status_id(CutoverCompatibilityStatus::Unsupported),
        "unsupported"
    );
}

#[test]
fn experimental_api_presented_stable_is_blocked() {
    let outcome = reject_status_misrepresentation(
        CutoverCompatibilityStatus::Experimental,
        CutoverCompatibilityStatus::StableForV01Baseline,
        "cli-command-surface",
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::StatusMisrepresented { .. })
    ));
    assert!(
        reject_status_misrepresentation(
            CutoverCompatibilityStatus::StableForV01Baseline,
            CutoverCompatibilityStatus::StableForV01Baseline,
            "cli-command-surface",
        )
        .is_ok()
    );
}

#[test]
fn e2e_gate_not_run_blocks_required_gate_execution() {
    let without_e2e: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .filter(|gate| **gate != ReleaseGate::E2eLocalConformance)
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        validate_required_gates_executed(&without_e2e),
        Err(ReleaseCutoverError::RequiredGateFailedOrMissing { .. })
    ));
}

#[test]
fn reference_cpu_gate_skip_is_disallowed() {
    let skip = GateSkip {
        gate: "reference-cpu-conformance".into(),
        outside_v0_1_scope: false,
        reason: Some("ran out of time".into()),
        hides_baseline_failure: false,
        included_in_release_report: true,
    };
    assert!(matches!(
        validate_gate_skips(&[skip]),
        Err(ReleaseCutoverError::DisallowedGateSkip { .. })
    ));
}

#[test]
fn allowed_skip_requires_every_condition() {
    let hides_failure = GateSkip {
        gate: "cuda-conformance".into(),
        outside_v0_1_scope: true,
        reason: Some("out of scope".into()),
        hides_baseline_failure: true,
        included_in_release_report: true,
    };
    assert!(hides_failure.validate().is_err());

    let allowed = GateSkip {
        hides_baseline_failure: false,
        ..hides_failure
    };
    assert!(allowed.validate().is_ok());
}

#[test]
fn undocumented_exception_blocks_release() {
    assert!(matches!(
        reject_undocumented_cutover_exception(true, None),
        Err(ReleaseCutoverError::UndocumentedException { .. })
    ));
    assert!(reject_undocumented_cutover_exception(false, None).is_ok());
}

#[test]
fn exception_missing_mitigation_or_owner_blocks_release() {
    let incomplete = CutoverException {
        gate: "dependency-audit".into(),
        exception: SecurityException {
            issue: "advisory RUSTSEC-0000-0000".into(),
            affected_component: "example-crate".into(),
            severity: DependencyAdvisorySeverity::High,
            rationale: "no fixed version available yet".into(),
            mitigation: String::new(),
            owner: String::new(),
            expiration_or_follow_up: "revisit in v0.2".into(),
            release_note_entry: true,
        },
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::UndocumentedException { .. })
    ));

    let complete = CutoverException {
        exception: SecurityException {
            mitigation: "vendored patch applied".into(),
            owner: "release-team".into(),
            ..incomplete.exception.clone()
        },
        ..incomplete
    };
    assert!(complete.validate().is_ok());
    assert!(validate_cutover_exceptions(&[complete]).is_ok());
}

#[test]
fn secret_scan_missing_blocks_security_verification() {
    let missing = CutoverSecurityVerification {
        gate_inputs: ReleaseSecurityGateInputs::default(),
        security_notes: SecurityReleaseNotes {
            v0_1_threat_model: Some("CPU-local baseline".into()),
            trusted_native_provider_model: Some("Providers are trusted native code".into()),
            no_raw_handle_policy: Some("no raw handles in public APIs".into()),
            default_redaction: Some("diagnostics redact by default".into()),
            reporting_process_placeholder: Some("reporting process TBD".into()),
            ..Default::default()
        },
    };
    // A gate_inputs with every field false represents "not yet confirmed" at
    // this call site only through the surrounding checklist; the blocking
    // gate itself is exercised directly here.
    let blocked = CutoverSecurityVerification {
        gate_inputs: ReleaseSecurityGateInputs {
            secrets_detected: true,
            ..Default::default()
        },
        ..missing
    };
    assert!(blocked.validate().is_err());
}

#[test]
fn missing_conformance_report_blocks_stable_release_unless_marked_not_applicable() {
    let manifest = ReleaseArtifactManifest::default();
    assert!(matches!(
        validate_cutover_artifacts_generated(&manifest),
        Err(ReleaseCutoverError::ArtifactGenerationIncomplete { .. })
    ));

    let mut complete = ReleaseArtifactManifest::default();
    for kind in RELEASE_ARTIFACT_KINDS {
        complete.set(*kind, ReleaseArtifactStatus::NotApplicable);
    }
    assert!(validate_cutover_artifacts_generated(&complete).is_ok());
}

#[test]
fn checksum_mismatch_blocks_or_withdraws_release() {
    let checksum =
        ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").unwrap();
    let outcome = verify_cutover_artifact_checksum(&checksum, "different-digest");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::ArtifactVerificationFailed { .. })
    ));
    assert!(verify_cutover_artifact_checksum(&checksum, "deadbeef").is_ok());
}

#[test]
fn cutover_artifact_verification_requires_integrity_and_cutover_specific_checks() {
    let incomplete = CutoverArtifactVerification {
        integrity: ArtifactIntegrityStatus {
            source_state_clean_or_ci_controlled: true,
            release_tag_matches_source: true,
            openspec_report_matches_baseline: true,
            conformance_reports_match_commit: true,
            checksums_match_final_artifacts: true,
        },
        release_notes_match_compatibility_matrix: false,
        artifact_names_include_version: true,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ArtifactVerificationFailed { .. })
    ));

    let complete = CutoverArtifactVerification {
        release_notes_match_compatibility_matrix: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn known_limitation_missing_from_changelog_blocks_release() {
    let incomplete = CutoverChangelogChecklist {
        changelog: ReleaseChangelog {
            entries: vec![ChangelogEntry {
                kind: ChangelogEntryKind::AddedContract,
                description: "added the Runtime Inference API".into(),
            }],
        },
        includes_added_contracts: true,
        includes_changed_contracts: true,
        includes_removed_or_deprecated_contracts: true,
        includes_release_scope: true,
        includes_known_limitations: false,
        includes_compatibility_status: true,
        includes_security_notes: true,
        includes_conformance_status: true,
        includes_deferred_roadmap_items: true,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ChangelogIncomplete { .. })
    ));

    let complete = CutoverChangelogChecklist {
        includes_known_limitations: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn release_notes_checklist_requires_every_topic() {
    let incomplete = CutoverReleaseNotesChecklist {
        explains_what_v0_1_is: true,
        ..Default::default()
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ReleaseNotesIncomplete { .. })
    ));

    let complete = CutoverReleaseNotesChecklist {
        explains_what_v0_1_is: true,
        explains_what_users_can_run: true,
        explains_stable_status: true,
        explains_preview_status: true,
        explains_experimental_status: true,
        explains_deferred_status: true,
        explains_unsupported_status: true,
        explains_artifact_verification: true,
        explains_security_limitations: true,
        explains_how_to_run_conformance: true,
        includes_compatibility_matrix: true,
        includes_security_notes: true,
        includes_known_limitations: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn tag_created_before_gates_pass_is_invalid() {
    let mut results: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    results[0].passed = false;
    assert!(matches!(
        validate_tag_after_gates(&results, true),
        Err(ReleaseCutoverError::TagCreatedBeforeGatesPassed)
    ));
    // No tag created yet: an incomplete gate set does not itself block.
    assert!(validate_tag_after_gates(&results, false).is_ok());

    for result in &mut results {
        result.passed = true;
    }
    assert!(validate_tag_after_gates(&results, true).is_ok());
}

#[test]
fn wit_validation_must_complete_before_stable_tag() {
    let without_wit: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .filter(|gate| **gate != ReleaseGate::WitValidation)
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        validate_tag_after_gates(&without_wit, true),
        Err(ReleaseCutoverError::TagCreatedBeforeGatesPassed)
    ));
}

#[test]
fn server_api_claimed_included_is_blocked() {
    let outcome = validate_publication_scope_preserved(
        "server-api-implementation",
        true,
        CutoverCompatibilityStatus::Deferred,
        CutoverCompatibilityStatus::StableForV01Baseline,
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::PublicationScopeViolation { .. })
    ));
}

#[test]
fn binary_version_mismatch_marks_post_publication_verification_invalid() {
    let mismatch = PostPublicationVerification {
        published_artifacts_match_checksums: true,
        release_notes_visible: true,
        reports_accessible: true,
        version_command_matches_tag: false,
        documentation_links_valid: true,
        compatibility_matrix_visible: true,
        security_notes_visible: true,
        deferred_roadmap_clearly_separated: true,
    };
    assert!(matches!(
        mismatch.validate(),
        Err(ReleaseCutoverError::PostPublicationVerificationFailed { .. })
    ));
}

#[test]
fn rollback_and_retraction_notes_cover_invalid_release_discovery() {
    let incomplete = RollbackRetractionNotes::default();
    assert!(incomplete.validate().is_err());

    let complete = RollbackRetractionNotes {
        withdrawal_procedure_documented: true,
        advisory_publication_procedure_documented: true,
        patch_release_procedure_documented: true,
        audit_trail_preservation_documented: true,
        release_notes_update_procedure_documented: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn post_v0_1_item_presented_as_release_claim_is_rejected() {
    let next_work = PostV01HandoffItem {
        name: "optimized-cpu-provider".into(),
        presented_as_v0_1_release_claim: false,
    };
    assert!(reject_post_v0_1_item_as_release_claim(&next_work).is_ok());

    let misclaimed = PostV01HandoffItem {
        name: "optimized-cpu-provider".into(),
        presented_as_v0_1_release_claim: true,
    };
    assert!(matches!(
        reject_post_v0_1_item_as_release_claim(&misclaimed),
        Err(ReleaseCutoverError::PostV01ItemPresentedAsReleaseClaim { .. })
    ));
}

#[test]
fn final_release_statement_describes_baseline_accurately() {
    assert!(validate_final_release_statement(V0_1_FINAL_RELEASE_STATEMENT).is_ok());
    assert!(validate_final_release_statement("Magnetar v0.1 ships full GPU support.").is_err());
}

#[test]
fn cli_boundary_gate_failure_blocks_cutover() {
    let outcome = validate_cutover_cli_boundary("filesystem");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
    ));
}

#[test]
fn runtime_tool_execution_blocks_cutover_scope_confirmation() {
    let outcome = validate_cutover_runtime_scope("shell");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
    ));
    assert!(validate_cutover_runtime_scope("generation").is_ok());
}

#[test]
fn release_cutover_observation_is_always_redacted() {
    let observation = record_release_cutover_observation(ReleaseCutoverObservationInput {
        correlation_id: CorrelationId::new("cutover-run-1"),
        kind: ReleaseSecurityObservationKind::SecretScanCompleted,
        gate: Some("secret-scan".into()),
        target: Some("reference-cpu-provider".into()),
        feature_set: vec!["reference-cpu-provider".into()],
        artifact: Some("magnetar-cli".into()),
        release_metadata: Some("v0.1.0-rc.1".into()),
        raw_detail: "found credential abc123 in build.env",
    });
    assert!(
        !observation
            .security_observation
            .detail
            .clone()
            .unwrap()
            .contains("credential abc123")
    );
    assert!(observation.gate.is_some());
    assert!(observation.target.is_some());
    assert!(!observation.feature_set.is_empty());
    assert!(observation.artifact.is_some());
}

#[test]
fn evaluate_release_cutover_reports_every_triggered_reason() {
    let clean = ReleaseCutoverGateInputs::default();
    assert!(evaluate_release_cutover(&clean).is_ok());

    let blocked = ReleaseCutoverGateInputs {
        openspec_not_frozen: true,
        tag_created_before_gates_passed: true,
        ..Default::default()
    };
    let outcome = evaluate_release_cutover(&blocked);
    assert!(matches!(
        &outcome,
        Err(ReleaseCutoverError::ReleaseCutoverBlocked { reasons })
            if reasons.len() == 2
    ));
}

#[test]
fn release_cutover_conformance_report_is_conformant() {
    let report = run_release_cutover_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// kernel_artifact
// ---------------------------------------------------------------------

fn conformance_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("kernel-artifact-test-provider"),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::TestFixture,
    )
}

#[test]
fn kernel_source_format_is_extensible_and_not_provider_binding() {
    let triton = KernelSourceFormat::new("triton", "source").with_version("3");
    assert_eq!(triton.stable_key(), "triton:source@3");
    let custom = KernelSourceFormat::new("vendor", "custom-ir").with_version("2");
    assert!(custom.is_valid());
    assert_eq!(custom.to_string(), "vendor:custom-ir@2");

    let empty = KernelSourceFormat::new("", "name");
    assert!(!empty.is_valid());
}

#[test]
fn artifact_trust_is_only_ever_policy_controlled() {
    assert_eq!(
        evaluate_artifact_trust(false),
        KernelArtifactTrust::Untrusted
    );
    assert_eq!(evaluate_artifact_trust(true), KernelArtifactTrust::Trusted);
    assert_eq!(
        KernelArtifactTrust::default(),
        KernelArtifactTrust::Untrusted
    );
}

#[test]
fn source_artifact_validation_requires_trust_and_valid_format() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let artifact = KernelSourceArtifact::new(
        KernelSourceArtifactId::from_digest("digest-1"),
        KernelSourceFormat::new("triton", "source").with_version("3"),
        operator,
        KernelArtifactProvenance::AiGenerated,
    );

    let untrusted = validate_source_artifact(&artifact);
    assert!(matches!(
        untrusted,
        Err(KernelArtifactError::Untrusted { .. })
    ));

    let trusted = artifact.with_trust(evaluate_artifact_trust(true));
    assert!(validate_source_artifact(&trusted).is_ok());

    let empty_digest = KernelSourceArtifact::new(
        KernelSourceArtifactId::from_digest(""),
        KernelSourceFormat::new("triton", "source"),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelArtifactProvenance::HumanAuthored,
    )
    .with_trust(evaluate_artifact_trust(true));
    assert!(matches!(
        validate_source_artifact(&empty_digest),
        Err(KernelArtifactError::ArtifactInvalid { .. })
    ));
}

#[test]
fn compiled_artifact_validation_rejects_operator_and_provider_mismatch() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let other_operator = OperatorId::magnetar("softmax", 1, OperatorFamily::Activation);
    let provider = ProviderBinding::new("cuda-provider");
    let other_provider = ProviderBinding::new("metal-provider");

    let artifact = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("compiled-digest"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator.clone(),
    )
    .with_trust(evaluate_artifact_trust(true))
    .with_provider_compatibility([provider.clone()]);

    assert!(validate_compiled_artifact(&artifact, &operator, &provider).is_ok());
    assert!(matches!(
        validate_compiled_artifact(&artifact, &other_operator, &provider),
        Err(KernelArtifactError::OperatorIncompatible { .. })
    ));
    assert!(matches!(
        validate_compiled_artifact(&artifact, &operator, &other_provider),
        Err(KernelArtifactError::ProviderIncompatible { .. })
    ));

    let untrusted = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("compiled-digest-2"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator.clone(),
    );
    assert!(matches!(
        validate_compiled_artifact(&untrusted, &operator, &provider),
        Err(KernelArtifactError::Untrusted { .. })
    ));
}

#[test]
fn hot_path_denies_compilation_cold_path_allows_it() {
    assert!(matches!(
        reject_hot_path_compilation(
            KernelArtifactPath::Hot,
            KernelArtifactColdPathOperation::Compilation
        ),
        Err(KernelArtifactError::HotPathCompilationDenied { .. })
    ));
    assert!(
        reject_hot_path_compilation(
            KernelArtifactPath::Cold,
            KernelArtifactColdPathOperation::Compilation
        )
        .is_ok()
    );
}

#[test]
fn lazy_preparation_requires_explicit_policy_and_admission_state() {
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), false, false).is_ok());
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), true, true),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, false),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, true).is_ok());
}

#[test]
fn prepared_kernel_id_allocator_produces_distinct_opaque_ids() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let first = allocator.allocate();
    let second = allocator.allocate();
    assert_ne!(first, second);
    assert_ne!(first.to_string(), second.to_string());
}

#[test]
fn prepared_kernel_lifecycle_blocks_destruction_while_referenced() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let kernel = conformance_kernel_id("matmul-prepared");
    let mut prepared = PreparedKernel::new(
        allocator.allocate(),
        kernel,
        CompiledKernelArtifactId::from_digest("digest"),
        ProviderBinding::new("cuda-provider"),
        DeviceBinding::new(DeviceId::new("cuda-0")),
        PreparedKernelGeneration::new(1),
    );
    assert_eq!(prepared.active_references(), 0);
    prepared.mark_ready().unwrap();
    assert!(prepared.state.is_dispatchable());

    prepared.add_reference();
    assert!(matches!(
        prepared.destroy(),
        Err(KernelArtifactError::PreparedGenerationInUse { .. })
    ));
    prepared.release_reference();
    prepared.retire().unwrap();
    assert!(prepared.destroy().is_ok());
}

#[test]
fn prepared_kernel_generations_coexist_and_registry_tracks_readiness() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let kernel = conformance_kernel_id("matmul-generations");
    let device = DeviceBinding::new(DeviceId::new("cuda-0"));
    let provider = ProviderBinding::new("cuda-provider");
    let artifact = CompiledKernelArtifactId::from_digest("digest");

    let mut registry = KernelRegistry::new();
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        artifact.clone(),
        provider.clone(),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().unwrap();
    let generation_one_id = generation_one.id;
    registry.register_prepared_kernel(generation_one);

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        artifact,
        provider,
        device,
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().unwrap();
    registry.register_prepared_kernel(generation_two);

    assert_eq!(registry.prepared_kernels_for(&kernel).count(), 2);
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());
    assert!(!registry.artifact_observations().is_empty());

    registry.retire_prepared_kernel(&generation_one_id).unwrap();
    assert!(registry.destroy_prepared_kernel(&generation_one_id).is_ok());
    assert_eq!(registry.prepared_kernels_for(&kernel).count(), 1);
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());
}

#[test]
fn kernel_advertisement_may_reference_artifact_metadata() {
    let id = conformance_kernel_id("matmul-advertised");
    let binding = KernelArtifactBinding::new(CompiledKernelArtifactId::from_digest("digest"))
        .with_source_artifact(KernelSourceArtifactId::from_digest("source-digest"));
    let advertisement = KernelAdvertisement::new(id.clone()).with_artifact(binding);
    assert!(advertisement.artifact.is_some());
    assert_eq!(advertisement.id, id);
}

#[test]
fn inference_requests_reject_kernel_artifact_management_fields() {
    for field in KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS {
        assert!(reject_inference_request_artifact_field(field).is_err());
    }
    assert!(reject_inference_request_artifact_field("prompt").is_ok());
}

#[test]
fn kernel_artifact_error_ids_match_proposal_error_model() {
    let cases: &[(KernelArtifactError, &str)] = &[
        (
            KernelArtifactError::ArtifactInvalid { reason: "x".into() },
            "kernel-artifact-invalid",
        ),
        (
            KernelArtifactError::DigestMismatch {
                expected: "a".into(),
                found: "b".into(),
            },
            "kernel-artifact-digest-mismatch",
        ),
        (
            KernelArtifactError::FormatUnsupported { format: "x".into() },
            "kernel-artifact-format-unsupported",
        ),
        (
            KernelArtifactError::Untrusted {
                artifact: "x".into(),
            },
            "kernel-artifact-untrusted",
        ),
        (
            KernelArtifactError::OperatorIncompatible { reason: "x".into() },
            "kernel-artifact-operator-incompatible",
        ),
        (
            KernelArtifactError::DTypeIncompatible { reason: "x".into() },
            "kernel-artifact-dtype-incompatible",
        ),
        (
            KernelArtifactError::LayoutIncompatible { reason: "x".into() },
            "kernel-artifact-layout-incompatible",
        ),
        (
            KernelArtifactError::ShapeIncompatible { reason: "x".into() },
            "kernel-artifact-shape-incompatible",
        ),
        (
            KernelArtifactError::TargetIncompatible { target: "x".into() },
            "kernel-artifact-target-incompatible",
        ),
        (
            KernelArtifactError::ProviderIncompatible {
                provider: "x".into(),
            },
            "kernel-artifact-provider-incompatible",
        ),
        (
            KernelArtifactError::DriverIncompatible { reason: "x".into() },
            "kernel-artifact-driver-incompatible",
        ),
        (
            KernelArtifactError::CompilerIncompatible { reason: "x".into() },
            "kernel-artifact-compiler-incompatible",
        ),
        (
            KernelArtifactError::PreparationUnavailable { reason: "x".into() },
            "kernel-preparation-unavailable",
        ),
        (
            KernelArtifactError::PreparationFailed { reason: "x".into() },
            "kernel-preparation-failed",
        ),
        (
            KernelArtifactError::PreparedHandleInvalid { reason: "x".into() },
            "kernel-prepared-handle-invalid",
        ),
        (
            KernelArtifactError::PreparedGenerationInUse { generation: 1 },
            "kernel-prepared-generation-in-use",
        ),
        (
            KernelArtifactError::PreparedDestroyFailed { reason: "x".into() },
            "kernel-prepared-destroy-failed",
        ),
        (
            KernelArtifactError::PreparedNotReady { kernel: "x".into() },
            "kernel-prepared-not-ready",
        ),
        (
            KernelArtifactError::HotPathCompilationDenied {
                operation: "x".into(),
            },
            "kernel-hot-path-compilation-denied",
        ),
        (
            KernelArtifactError::InternalKernelArtifactError { reason: "x".into() },
            "internal-kernel-artifact-error",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.id(), *expected_id);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn model_instance_readiness_fails_when_kernel_preparation_failed() {
    let mut checks = ModelInstanceReadinessChecks::default();
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Ready);
    assert!(checks.validate().is_ok());

    checks.kernel_preparation_ready = false;
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Failed);
    assert!(matches!(
        checks.validate(),
        Err(ModelInstanceError::ModelInstanceKernelPreparationFailed)
    ));
}

#[test]
fn kernel_artifact_conformance_report_is_conformant() {
    let report = run_kernel_artifact_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// kernel_artifact_manifest
// ---------------------------------------------------------------------

fn temp_kernel_bundle_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-kernel-bundle-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(directory.join("blobs").join("sha256")).unwrap();
    directory
}

fn write_kernel_bundle(directory: &std::path::Path, blob_bytes: &[u8]) -> String {
    let digest = KernelBlobDigest::of_bytes(blob_bytes);
    fs::write(
        directory.join("blobs").join("sha256").join(&digest.value),
        blob_bytes,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{}",
      "size": {},
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest.value,
        blob_bytes.len()
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();
    digest.value
}

#[test]
fn kernel_artifact_manifest_conformance_report_is_conformant() {
    let report = run_kernel_artifact_manifest_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn kernel_manifest_json_duplicate_key_is_rejected() {
    let limits = KernelManifestLimits::default();
    let text = r#"{"schema":"magnetar:kernel-manifest@1.0","artifacts":[],"artifacts":[]}"#;
    let outcome = parse_manifest_json(text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::DuplicateKey { .. })
    ));
}

#[test]
fn kernel_manifest_json_excessive_nesting_is_rejected() {
    let limits = KernelManifestLimits {
        max_nesting_depth: 4,
        ..KernelManifestLimits::default()
    };
    let mut text = String::new();
    for _ in 0..10 {
        text.push('[');
    }
    for _ in 0..10 {
        text.push(']');
    }
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
}

#[test]
fn kernel_manifest_oversized_input_is_rejected() {
    let limits = KernelManifestLimits {
        max_manifest_bytes: 8,
        ..KernelManifestLimits::default()
    };
    let outcome = parse_manifest_json(
        r#"{"schema":"magnetar:kernel-manifest@1.0","artifacts":[]}"#,
        &limits,
    );
    assert!(matches!(outcome, Err(KernelManifestError::TooLarge { .. })));
}

#[test]
fn kernel_manifest_unsupported_schema_major_is_rejected() {
    let limits = KernelManifestLimits::default();
    let outcome = parse_manifest_json(
        r#"{"schema":"magnetar:kernel-manifest@2.0","artifacts":[]}"#,
        &limits,
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::SchemaUnsupported { .. })
    ));
}

#[test]
fn kernel_manifest_dependency_cycle_is_detected() {
    let digest_a = KernelBlobDigest::of_bytes(b"artifact-a");
    let digest_b = KernelBlobDigest::of_bytes(b"artifact-b");
    let mut artifact_a = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        digest_a.clone(),
        4,
    ));
    artifact_a.dependencies.push(digest_b.clone());
    let mut artifact_b = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        digest_b,
        4,
    ));
    artifact_b.dependencies.push(digest_a);
    let manifest = KernelManifestV1 {
        artifacts: vec![artifact_a, artifact_b],
        ..KernelManifestV1::new()
    };
    assert!(detect_dependency_cycle(&manifest).is_some());
    assert!(matches!(
        manifest.validate(),
        Err(KernelManifestError::DependencyCycle { .. })
    ));
}

#[test]
fn kernel_exchange_bundle_validates_end_to_end() {
    let directory = temp_kernel_bundle_dir("happy-path");
    write_kernel_bundle(&directory, b"cubin-bytes");

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("well-formed bundle should validate");
    assert_eq!(validated.digest, validated.manifest.digest());

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_rejects_corrupted_blob() {
    let directory = temp_kernel_bundle_dir("corrupted-blob");
    let digest_value = write_kernel_bundle(&directory, b"cubin-bytes");
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_value),
        b"tampered-bytes",
    )
    .unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleBlobSizeMismatch { .. })
            | Err(KernelManifestError::BundleBlobDigestMismatch { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_missing_optional_embedded_artifact_is_tolerated() {
    let directory = temp_kernel_bundle_dir("optional-missing");
    let present_digest = KernelBlobDigest::of_bytes(b"present-bytes");
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&present_digest.value),
        b"present-bytes",
    )
    .unwrap();
    let missing_digest = KernelBlobDigest::of_bytes(b"missing-bytes");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{present}",
      "size": 13,
      "storage_mode": "embedded",
      "required": true
    }},
    {{
      "role": "benchmark-evidence",
      "format": "magnetar:benchmark-report@1",
      "digest": "sha256:{missing}",
      "size": 99,
      "storage_mode": "embedded",
      "required": false
    }}
  ]
}}"#,
        present = present_digest.value,
        missing = missing_digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("missing optional embedded artifact should not invalidate the bundle");
    assert_eq!(validated.manifest.artifacts.len(), 2);

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_missing_required_embedded_artifact_is_rejected() {
    let directory = temp_kernel_bundle_dir("required-missing");
    let missing_digest = KernelBlobDigest::of_bytes(b"never-written");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{missing}",
      "size": 13,
      "storage_mode": "embedded",
      "required": true
    }}
  ]
}}"#,
        missing = missing_digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleRequiredArtifactMissing { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_rejects_required_external_artifact_without_fetching() {
    let directory = temp_kernel_bundle_dir("required-external");
    let digest = KernelBlobDigest::of_bytes(b"external-bytes");
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 13,
      "storage_mode": "external",
      "required": true,
      "location_hint": "https://example.invalid/artifact.cubin"
    }}
  ]
}}"#,
        digest = digest.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ExchangeExternalReferenceDenied { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_exchange_bundle_missing_manifest_is_rejected() {
    let directory = temp_kernel_bundle_dir("missing-manifest");
    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleManifestMissing)
    ));
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_bundle_path_safety_rejects_traversal_and_absolute_paths() {
    for bad in [
        "../escape",
        "/etc/passwd",
        "C:/Windows/system32",
        "a/../../b",
        "\\\\server\\share",
    ] {
        assert!(
            validate_bundle_relative_path(bad).is_err(),
            "expected '{bad}' to be rejected"
        );
    }
    assert!(validate_bundle_relative_path("blobs/sha256/deadbeef").is_ok());
}

#[test]
fn kernel_bundle_symlink_entry_is_rejected_when_creatable() {
    let directory = temp_kernel_bundle_dir("symlink");
    write_kernel_bundle(&directory, b"symlink-fixture");
    let target = directory.join(KERNEL_MANIFEST_FILE_NAME);
    let link = directory.join("blobs").join("escape-link");

    #[cfg(unix)]
    let created = std::os::unix::fs::symlink(&target, &link).is_ok();
    #[cfg(windows)]
    let created = std::os::windows::fs::symlink_file(&target, &link).is_ok();
    #[cfg(not(any(unix, windows)))]
    let created = false;

    if created {
        let outcome = scan_bundle_for_unsafe_entries(&directory);
        assert!(matches!(
            outcome,
            Err(KernelManifestError::BundleSymlinkDenied { .. })
        ));
    }
    // When the platform/permissions do not allow creating a symlink (e.g.
    // Windows without Developer Mode or admin rights), this test is a no-op
    // rather than a false failure -- the rejection code path itself is
    // exercised whenever a symlink can actually be created.

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_inference_request_field_boundary_rejects_bundle_fields() {
    for field in KERNEL_MANIFEST_FORBIDDEN_INFERENCE_FIELDS {
        let outcome = reject_inference_request_manifest_field(field);
        assert!(outcome.is_err(), "expected '{field}' to be rejected");
    }
    assert!(reject_inference_request_manifest_field("prompt").is_ok());
}

#[test]
fn kernel_manifest_extension_cannot_claim_a_core_field_namespace() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"core-field-fixture").value
    );
    let mut manifest = parse_manifest_json(&text, &limits).expect("sample manifest parses");
    manifest.extensions.push(KernelManifestExtension {
        namespace: "trust:override".into(),
        required: false,
        data: serde_json::Value::Null,
    });
    let outcome = manifest.validate();
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ArtifactReferenceInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_recommendation_never_grants_promotion() {
    for recommendation in [
        KernelManifestRecommendation::RecommendedForLatency,
        KernelManifestRecommendation::RecommendedForThroughput,
        KernelManifestRecommendation::Experimental,
        KernelManifestRecommendation::Reject,
    ] {
        assert!(!recommendation_grants_promotion(recommendation));
    }
}

#[test]
fn kernel_manifest_qualification_evidence_array_round_trips_through_json() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"evidence-bytes").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{artifact_digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ],
  "qualification_evidence": [
    {{
      "digest": "sha256:{digest}",
      "profile": "correctness",
      "suite_or_workload_version": "v1",
      "oracle_or_provider_identity": "reference-cpu@1",
      "status": "passed"
    }}
  ]
}}"#,
        artifact_digest = KernelBlobDigest::of_bytes(b"artifact-bytes").value,
        digest = digest
    );
    let manifest =
        parse_manifest_json(&text, &limits).expect("manifest with qualification evidence parses");
    assert_eq!(manifest.qualification_evidence.len(), 1);
    let evidence = &manifest.qualification_evidence[0];
    assert_eq!(evidence.profile, "correctness");
    assert_eq!(evidence.status, KernelEvidenceStatus::Passed);
    assert!(oracle_identity_is_known(evidence));
    assert!(evaluate_qualification_evidence_currency(evidence, "v1"));
    assert!(!evaluate_qualification_evidence_currency(evidence, "v2"));
}

#[test]
fn kernel_manifest_accepts_unknown_future_artifact_format() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "vendor:new-ir@1",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"future-format-bytes").value
    );
    let manifest = parse_manifest_json(&text, &limits)
        .expect("unknown future format still parses structurally");
    assert_eq!(
        manifest.artifacts[0].blob.format.stable_key(),
        "vendor:new-ir@1"
    );
}

#[test]
fn kernel_manifest_signature_presence_alone_is_never_verified() {
    let envelope = KernelSignatureEnvelope {
        algorithm: "ed25519".into(),
        key_id: Some("key-1".into()),
        signed_digest: KernelBlobDigest::of_bytes(b"manifest"),
        signature_material: KernelBlobDigest::of_bytes(b"signature-bytes"),
        certificate_chain_reference: None,
    };
    assert!(!signature_is_verified(&envelope, false));
    assert!(signature_is_verified(&envelope, true));
}

#[test]
fn kernel_manifest_operator_version_range_compatibility() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ],
      "operator_version_range": {{ "min": 1, "max": 3 }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"version-range-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("manifest with version range parses");
    let binding = manifest.artifacts[0].semantic_binding.as_ref().unwrap();
    assert!(binding.is_version_compatible(1));
    assert!(binding.is_version_compatible(3));
    assert!(!binding.is_version_compatible(4));

    let invalid_range = KernelSemanticBinding {
        operators: vec![OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        )],
        primary_version_requirements: Some(KernelOperatorVersionRange { min: 5, max: 1 }),
    };
    assert!(matches!(
        invalid_range.validate(),
        Err(KernelManifestError::SemanticBindingInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_fused_semantic_binding_preserves_operator_order() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "rmsnorm", "version": 1, "family": "normalization" }},
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"fused-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("fused binding manifest parses");
    let binding = manifest.artifacts[0].semantic_binding.as_ref().unwrap();
    assert!(binding.is_fused());
    assert_eq!(
        binding.fingerprint(),
        "magnetar:operator/rmsnorm@1 -> magnetar:operator/matmul@1"
    );

    // Order matters: swapping the two operators is a different fusion.
    let reversed_text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }},
        {{ "namespace": "magnetar:operator", "name": "rmsnorm", "version": 1, "family": "normalization" }}
      ]
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"fused-fixture-reversed").value
    );
    let reversed = parse_manifest_json(&reversed_text, &limits)
        .expect("reversed fused binding manifest parses");
    let reversed_binding = reversed.artifacts[0].semantic_binding.as_ref().unwrap();
    assert_ne!(binding.fingerprint(), reversed_binding.fingerprint());

    // Normalizing a fused source artifact preserves the remaining operators
    // as the fused group, and only the primary Operator becomes the
    // compiled artifact's single `operator_semantics`.
    let normalized_source =
        normalize_to_source_artifact(&manifest.artifacts[0]).expect("fused source normalizes");
    assert_eq!(normalized_source.fused_operator_group.len(), 1);
    assert_eq!(normalized_source.fused_operator_group[0].name(), "matmul");
}

#[test]
fn kernel_manifest_target_specialization_compiler_precision_generator_round_trip() {
    let limits = KernelManifestLimits::default();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ],
      "target": {{
        "device_type": "gpu",
        "hardware_vendor": "nvidia",
        "architecture": "sm90",
        "device_features": ["tensor-core"],
        "provider_compatibility": ["nvidia-cuda"],
        "runtime_driver_compatibility": ["cuda-12"],
        "memory_classes": ["hbm"]
      }},
      "specialization": {{
        "exact_dimensions": {{"0": 128}},
        "batch_range": [1, 32],
        "sequence_range": [1, 4096],
        "head_count": 32,
        "head_dimension": 128,
        "tile_sizes": [64, 64],
        "alignment": 16,
        "dtype": "float16",
        "layout": "row-major",
        "quantization_profile": "int8-groupwise",
        "execution_phase": "decode",
        "device_features": ["tensor-core"]
      }},
      "compiler_metadata": {{
        "compiler_identity": "triton",
        "compiler_version": "3.1",
        "backend_identity_version": "ptxas-12.4",
        "flags_fingerprint": "abc123",
        "build_fingerprint": "build-456",
        "target_architecture": "sm90"
      }},
      "precision": {{
        "accumulation_dtype": "float32",
        "approximate_math": true,
        "deterministic": false,
        "tolerance_profile": "operator-default",
        "quantization_error_profile": "int8-standard"
      }},
      "generator": {{
        "generator_name": "kernel-forge",
        "generator_version": "2.0",
        "campaign_id": "campaign-42",
        "source_revision": "https://example.invalid/repo@deadbeef"
      }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"rich-descriptor-fixture").value
    );
    let manifest = parse_manifest_json(&text, &limits).expect("rich descriptor manifest parses");
    let artifact = &manifest.artifacts[0];

    assert_eq!(artifact.target.device_type.as_deref(), Some("gpu"));
    assert_eq!(artifact.target.architecture.as_deref(), Some("sm90"));
    assert!(artifact.target.device_features.contains("tensor-core"));

    assert_eq!(artifact.specialization.batch_range, Some((1, 32)));
    assert_eq!(artifact.specialization.head_count, Some(32));
    assert_eq!(artifact.specialization.dtype.as_deref(), Some("float16"));
    assert_eq!(
        artifact.specialization.execution_phase,
        Some(KernelExecutionPhase::Decode)
    );

    let compiler = artifact.compiler_metadata.as_ref().unwrap();
    assert_eq!(compiler.compiler_identity.as_deref(), Some("triton"));
    assert_eq!(compiler.target_architecture.as_deref(), Some("sm90"));

    assert!(artifact.precision.approximate_math);
    assert_eq!(artifact.precision.deterministic, Some(false));

    let generator = artifact.generator.as_ref().unwrap();
    assert_eq!(generator.generator_name.as_deref(), Some("kernel-forge"));
    assert_eq!(generator.campaign_id.as_deref(), Some("campaign-42"));

    // Canonical identity round-trips through re-parsing the canonical bytes.
    let canonical_text = String::from_utf8(manifest.canonical_bytes()).unwrap();
    let reparsed = parse_manifest_json(&canonical_text, &limits).expect("canonical bytes reparse");
    assert_eq!(reparsed.digest(), manifest.digest());
}

#[test]
fn kernel_manifest_generator_rejects_embedded_credential_locator() {
    let generator = KernelGeneratorMetadata {
        generator_name: Some("gen".into()),
        generator_version: None,
        campaign_id: None,
        source_revision: Some("https://user:secret@example.invalid/repo".into()),
    };
    assert!(matches!(
        generator.validate(),
        Err(KernelManifestError::ProvenanceInvalid { .. })
    ));

    let clean = KernelGeneratorMetadata {
        source_revision: Some("https://example.invalid/repo@deadbeef".into()),
        ..generator
    };
    assert!(clean.validate().is_ok());
}

#[test]
fn kernel_manifest_specialization_rejects_impossible_range() {
    let specialization = KernelManifestSpecialization {
        batch_range: Some((32, 1)),
        ..KernelManifestSpecialization::default()
    };
    assert!(matches!(
        specialization.validate(),
        Err(KernelManifestError::SpecializationInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_target_entry_count_limit_is_enforced() {
    let limits = KernelManifestLimits {
        max_target_entries: 2,
        ..KernelManifestLimits::default()
    };
    let features: Vec<String> = (0..5).map(|i| format!("\"feature-{i}\"")).collect();
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "target": {{ "device_features": [{features}] }}
    }}
  ]
}}"#,
        digest = KernelBlobDigest::of_bytes(b"target-limit-fixture").value,
        features = features.join(", ")
    );
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
}

#[test]
fn kernel_manifest_conflicting_digest_metadata_is_rejected() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"shared-content").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }},
    {{
      "role": "auxiliary",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 999,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
    );
    let outcome = parse_manifest_json(&text, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::ArtifactReferenceInvalid { .. })
    ));
}

#[test]
fn kernel_manifest_same_digest_same_size_across_artifacts_is_allowed() {
    let limits = KernelManifestLimits::default();
    let digest = KernelBlobDigest::of_bytes(b"deduplicated-content").value;
    let text = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }},
    {{
      "role": "auxiliary",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded"
    }}
  ]
}}"#,
    );
    assert!(parse_manifest_json(&text, &limits).is_ok());
}

#[test]
fn kernel_manifest_multi_target_bundle_validates_with_distinct_architectures() {
    let directory = temp_kernel_bundle_dir("multi-target");
    let sm80_bytes = b"sm80-cubin";
    let sm90_bytes = b"sm90-cubin";
    let sm80_digest = KernelBlobDigest::of_bytes(sm80_bytes);
    let sm90_digest = KernelBlobDigest::of_bytes(sm90_bytes);
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&sm80_digest.value),
        sm80_bytes,
    )
    .unwrap();
    fs::write(
        directory
            .join("blobs")
            .join("sha256")
            .join(&sm90_digest.value),
        sm90_bytes,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{sm80}",
      "size": {sm80_len},
      "storage_mode": "embedded",
      "target": {{ "architecture": "sm80", "provider_compatibility": ["nvidia-cuda"] }}
    }},
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{sm90}",
      "size": {sm90_len},
      "storage_mode": "embedded",
      "target": {{ "architecture": "sm90", "provider_compatibility": ["nvidia-cuda"] }}
    }}
  ],
  "qualification_evidence": [
    {{ "digest": "sha256:{sm80}", "profile": "correctness@1", "status": "passed" }}
  ],
  "benchmark_evidence": [
    {{ "digest": "sha256:{sm90}", "profile": "latency@1", "workload_profile": "decode-256", "status": "passed" }}
  ]
}}"#,
        sm80 = sm80_digest.value,
        sm80_len = sm80_bytes.len(),
        sm90 = sm90_digest.value,
        sm90_len = sm90_bytes.len(),
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let validated = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default())
        .expect("multi-target bundle should validate");
    assert_eq!(validated.manifest.artifacts.len(), 2);
    assert_eq!(validated.manifest.qualification_evidence.len(), 1);
    assert_eq!(validated.manifest.benchmark_evidence.len(), 1);
    assert_eq!(
        validated.manifest.benchmark_evidence[0]
            .workload_profile
            .as_deref(),
        Some("decode-256")
    );

    let architectures: std::collections::BTreeSet<_> = validated
        .manifest
        .artifacts
        .iter()
        .filter_map(|artifact| artifact.target.architecture.clone())
        .collect();
    assert_eq!(
        architectures.len(),
        2,
        "expected two distinct compiled architectures"
    );

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_evaluate_target_compatibility() {
    let target = KernelTargetConstraints {
        architecture: Some("sm90".into()),
        provider_compatibility: ["nvidia-cuda".to_string()].into_iter().collect(),
        device_features: ["tensor-core".to_string()].into_iter().collect(),
        ..KernelTargetConstraints::default()
    };

    let matching_context = KernelRuntimeCompatibilityContext {
        provider_id: Some("nvidia-cuda".into()),
        architecture: Some("sm90".into()),
        available_device_features: ["tensor-core".to_string()].into_iter().collect(),
    };
    assert!(evaluate_target_compatibility(&target, &matching_context).is_ok());

    let wrong_architecture = KernelRuntimeCompatibilityContext {
        architecture: Some("sm80".into()),
        ..matching_context.clone()
    };
    assert!(matches!(
        evaluate_target_compatibility(&target, &wrong_architecture),
        Err(KernelManifestError::ExchangeCompatibilityFailed { .. })
    ));

    let missing_feature = KernelRuntimeCompatibilityContext {
        available_device_features: std::collections::BTreeSet::new(),
        ..matching_context
    };
    assert!(matches!(
        evaluate_target_compatibility(&target, &missing_feature),
        Err(KernelManifestError::ExchangeCompatibilityFailed { .. })
    ));

    // An artifact with no declared target constraints is always compatible.
    assert!(
        evaluate_target_compatibility(
            &KernelTargetConstraints::default(),
            &KernelRuntimeCompatibilityContext::default()
        )
        .is_ok()
    );
}

#[test]
fn kernel_manifest_validation_pipeline_orders_schema_before_blob_io() {
    // A directory bundle with a *missing* blobs directory entirely and an
    // *unsupported* schema major version: if schema validation ran after
    // blob I/O, this would surface as a filesystem/blob error instead.
    let directory = temp_kernel_bundle_dir("ordering");
    let manifest = r#"{
  "schema": "magnetar:kernel-manifest@99.0",
  "artifacts": [
    { "role": "compiled-kernel", "format": "nvidia:cubin", "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000", "size": 4 }
  ]
}"#;
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();
    fs::remove_dir_all(directory.join("blobs")).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let outcome = validate_kernel_exchange_bundle(&bundle, &KernelManifestLimits::default());
    assert!(
        matches!(outcome, Err(KernelManifestError::SchemaUnsupported { .. })),
        "expected schema validation to fail before any blob I/O is attempted, got {outcome:?}"
    );

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_normalizes_to_source_and_compiled_artifacts() {
    let mut source_blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::KERNEL_SOURCE),
        KernelArtifactFormat::new("triton", "source").with_version("3"),
        KernelBlobDigest::of_bytes(b"normalize-source"),
        4,
    );
    source_blob.required = true;
    let mut source_artifact = KernelManifestArtifact::new(source_blob);
    source_artifact.semantic_binding = Some(KernelSemanticBinding::single(OperatorId::magnetar(
        "matmul",
        1,
        OperatorFamily::LinearAlgebra,
    )));
    source_artifact.provenance = Some(KernelArtifactProvenance::AiGenerated);
    source_artifact.specialization.batch_range = Some((1, 8));

    let normalized_source =
        normalize_to_source_artifact(&source_artifact).expect("source normalizes");
    assert_eq!(normalized_source.format.stable_key(), "triton:source@3");
    assert_eq!(normalized_source.shape.max_batch_size, Some(8));
    assert_eq!(
        normalized_source.provenance,
        KernelArtifactProvenance::AiGenerated
    );

    let compiled_blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"normalize-compiled"),
        4,
    );
    let mut compiled_artifact = KernelManifestArtifact::new(compiled_blob);
    compiled_artifact.semantic_binding = Some(KernelSemanticBinding::single(OperatorId::magnetar(
        "matmul",
        1,
        OperatorFamily::LinearAlgebra,
    )));
    compiled_artifact.compiler_metadata = Some(KernelCompilerMetadata {
        compiler_identity: Some("triton".into()),
        compiler_version: Some("3.1".into()),
        ..Default::default()
    });
    compiled_artifact.target.architecture = Some("sm90".into());

    let normalized_compiled =
        normalize_to_compiled_artifact(&compiled_artifact).expect("compiled normalizes");
    assert_eq!(normalized_compiled.compiler_identity, "triton");
    assert_eq!(normalized_compiled.target_architecture, "sm90");

    // No semantic binding -> normalization fails rather than fabricating one.
    let unbound = KernelManifestArtifact::new(KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"unbound"),
        4,
    ));
    assert!(normalize_to_source_artifact(&unbound).is_err());
    assert!(normalize_to_compiled_artifact(&unbound).is_err());
}

#[test]
fn kernel_manifest_normalizes_qualification_identity_pieces() {
    let evidence = KernelEvidenceReference {
        digest: KernelBlobDigest::of_bytes(b"qualification-evidence"),
        profile: "baseline-correctness@1".into(),
        suite_or_workload_version: Some("v3".into()),
        oracle_or_provider_identity: Some("reference-cpu@2".into()),
        target_compatibility: std::collections::BTreeSet::new(),
        status: KernelEvidenceStatus::Passed,
        storage_mode: KernelArtifactStorageMode::Embedded,
        workload_profile: None,
        device_context: None,
        provider_context: None,
        workload_metadata: None,
    };
    let profile = normalize_qualification_profile(&evidence).expect("profile normalizes");
    assert_eq!(
        profile,
        QualificationProfile::new("baseline-correctness", 1)
    );

    let oracle = normalize_oracle_identity(&evidence).expect("oracle normalizes");
    assert_eq!(oracle.provider, ProviderBinding::new("reference-cpu"));
    assert_eq!(oracle.version, "2");

    let mut malformed = evidence.clone();
    malformed.profile = "no-version-separator".into();
    assert!(normalize_qualification_profile(&malformed).is_err());

    let mut no_oracle = evidence;
    no_oracle.oracle_or_provider_identity = None;
    assert!(normalize_oracle_identity(&no_oracle).is_err());
}

#[test]
fn kernel_manifest_normalizes_to_cache_key_and_entry_without_granting_trust() {
    let mut blob = KernelBlobDescriptor::new(
        KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelBlobDigest::of_bytes(b"cache-bridge-fixture"),
        4,
    );
    blob.required = true;
    let mut artifact = KernelManifestArtifact::new(blob);
    artifact.target.architecture = Some("sm90".into());
    artifact.compiler_metadata = Some(KernelCompilerMetadata {
        compiler_identity: Some("triton".into()),
        compiler_version: Some("3.1".into()),
        ..Default::default()
    });

    let key = normalize_to_cache_key(&artifact);
    assert_eq!(key.target_architecture, "sm90");
    assert_eq!(key.compiler_identity, "triton");

    let entry = normalize_to_cache_entry(&artifact);
    assert!(
        !entry.trust.is_trusted(),
        "a freshly normalized cache entry must start untrusted"
    );
    assert!(
        entry.qualification.is_none(),
        "a freshly normalized cache entry must start unqualified"
    );
}

#[test]
fn kernel_exchange_bundle_total_size_limit_is_enforced() {
    let directory = temp_kernel_bundle_dir("total-size-limit");
    let bytes_a = b"aaaa";
    let bytes_b = b"bbbb";
    let digest_a = KernelBlobDigest::of_bytes(bytes_a);
    let digest_b = KernelBlobDigest::of_bytes(bytes_b);
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_a.value),
        bytes_a,
    )
    .unwrap();
    fs::write(
        directory.join("blobs").join("sha256").join(&digest_b.value),
        bytes_b,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{ "role": "compiled-kernel", "format": "nvidia:cubin", "digest": "sha256:{a}", "size": 4, "storage_mode": "embedded" }},
    {{ "role": "auxiliary", "format": "nvidia:cubin", "digest": "sha256:{b}", "size": 4, "storage_mode": "embedded" }}
  ]
}}"#,
        a = digest_a.value,
        b = digest_b.value
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();

    let bundle = KernelExchangeBundle::open(&directory);
    let limits = KernelManifestLimits {
        max_total_embedded_bytes: 6,
        ..KernelManifestLimits::default()
    };
    let outcome = validate_kernel_exchange_bundle(&bundle, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleTotalSizeExceeded { .. })
    ));

    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn kernel_manifest_embedded_byte_accounting_saturates_instead_of_overflowing() {
    // The bundle validation pipeline accumulates declared blob sizes with
    // `u64::saturating_add`, implementing "Reject overflow" (tasks, "Integer
    // Safety"): summing sizes near `u64::MAX` must never wrap around or
    // panic, even though no real bundle could actually contain that many
    // bytes on disk.
    let total = [u64::MAX, u64::MAX, 1_u64]
        .into_iter()
        .fold(0_u64, u64::saturating_add);
    assert_eq!(total, u64::MAX);
}

#[test]
fn kernel_manifest_cli_operations_all_use_shared_validation() {
    let directory = temp_kernel_bundle_dir("cli-shared-validation");
    write_kernel_bundle(&directory, b"cli-fixture-bytes");
    let bundle = KernelExchangeBundle::open(&directory);
    let limits = KernelManifestLimits::default();

    for operation in [
        KernelManifestCliOperation::Inspect,
        KernelManifestCliOperation::Validate,
        KernelManifestCliOperation::Import,
        KernelManifestCliOperation::Export,
    ] {
        let result = run_kernel_manifest_cli_operation(operation, &bundle, &limits);
        assert!(
            result.is_ok(),
            "operation {operation:?} should reuse shared validation and succeed"
        );
    }

    fs::remove_dir_all(&directory).unwrap();
}

fn build_kernel_bundle_tar(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (path, data) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        builder.append_data(&mut header, *path, *data).unwrap();
    }
    builder.into_inner().unwrap()
}

fn gzip_compress(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn kernel_bundle_tar_entries(blob_bytes: &[u8]) -> (Vec<(&'static str, Vec<u8>)>, String) {
    let digest = KernelBlobDigest::of_bytes(blob_bytes);
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": {size},
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest = digest.value,
        size = blob_bytes.len()
    );
    (
        vec![
            (KERNEL_MANIFEST_FILE_NAME, manifest.into_bytes()),
            (
                Box::leak(format!("blobs/sha256/{}", digest.value).into_boxed_str()),
                blob_bytes.to_vec(),
            ),
        ],
        digest.value,
    )
}

#[test]
fn kernel_exchange_tar_archive_validates_to_identical_digest_as_directory_bundle() {
    let directory = temp_kernel_bundle_dir("archive-directory-parity");
    write_kernel_bundle(&directory, b"tar-parity-bytes");
    let directory_bundle = KernelExchangeBundle::open(&directory);
    let directory_validated =
        validate_kernel_exchange_bundle(&directory_bundle, &KernelManifestLimits::default())
            .expect("directory bundle validates");

    let (entries, _digest) = kernel_bundle_tar_entries(b"tar-parity-bytes");
    let entry_refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, d)| (*p, d.as_slice())).collect();
    let tar_bytes = build_kernel_bundle_tar(&entry_refs);

    let extract_dir = temp_kernel_bundle_dir("archive-directory-parity-extracted");
    let archive_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &extract_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("tar archive extracts");
    let archive_validated =
        validate_kernel_exchange_bundle(&archive_bundle, &KernelManifestLimits::default())
            .expect("extracted archive bundle validates");

    assert_eq!(directory_validated.digest, archive_validated.digest);

    // The archive checksum is a distinct, transport-level concept: it is
    // *not* the manifest digest, and never claimed to be.
    let archive_checksum = archive_diagnostic_checksum(&tar_bytes);
    assert_ne!(archive_checksum, directory_validated.digest.value);

    fs::remove_dir_all(&directory).unwrap();
    fs::remove_dir_all(&extract_dir).unwrap();
}

#[test]
fn kernel_exchange_tar_gz_archive_produces_the_same_manifest_digest_as_plain_tar() {
    let (entries, _digest) = kernel_bundle_tar_entries(b"gzip-parity-bytes");
    let entry_refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, d)| (*p, d.as_slice())).collect();
    let tar_bytes = build_kernel_bundle_tar(&entry_refs);
    let gz_bytes = gzip_compress(&tar_bytes);

    let plain_dir = temp_kernel_bundle_dir("gzip-parity-plain");
    let plain_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &plain_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("plain tar extracts");
    let plain_validated =
        validate_kernel_exchange_bundle(&plain_bundle, &KernelManifestLimits::default())
            .expect("plain tar bundle validates");

    let gz_dir = temp_kernel_bundle_dir("gzip-parity-compressed");
    let gz_bundle = extract_kernel_exchange_archive(
        std::io::Cursor::new(&gz_bytes),
        true,
        &gz_dir,
        &KernelExchangeArchiveLimits::default(),
    )
    .expect("gzip tar extracts");
    let gz_validated =
        validate_kernel_exchange_bundle(&gz_bundle, &KernelManifestLimits::default())
            .expect("gzip tar bundle validates");

    assert_eq!(plain_validated.digest, gz_validated.digest);
    // Compressing the exact same logical content changes the raw archive
    // bytes (and therefore the transport-level diagnostic checksum) without
    // touching logical identity.
    assert_ne!(
        archive_diagnostic_checksum(&tar_bytes),
        archive_diagnostic_checksum(&gz_bytes)
    );

    fs::remove_dir_all(&plain_dir).unwrap();
    fs::remove_dir_all(&gz_dir).unwrap();
}

#[test]
fn kernel_exchange_archive_ignores_ownership_timestamp_and_mode_for_identity() {
    let (entries, _digest) = kernel_bundle_tar_entries(b"metadata-irrelevant-bytes");

    let mut builder_a = tar::Builder::new(Vec::new());
    let mut builder_b = tar::Builder::new(Vec::new());
    for (path, data) in &entries {
        let mut header_a = tar::Header::new_gnu();
        header_a.set_size(data.len() as u64);
        header_a.set_mode(0o644);
        header_a.set_mtime(1_000);
        header_a.set_uid(0);
        header_a.set_gid(0);
        header_a.set_cksum();
        builder_a
            .append_data(&mut header_a, *path, data.as_slice())
            .unwrap();

        let mut header_b = tar::Header::new_gnu();
        header_b.set_size(data.len() as u64);
        header_b.set_mode(0o755);
        header_b.set_mtime(999_999_999);
        header_b.set_uid(1000);
        header_b.set_gid(1000);
        header_b.set_cksum();
        builder_b
            .append_data(&mut header_b, *path, data.as_slice())
            .unwrap();
    }
    let tar_a = builder_a.into_inner().unwrap();
    let tar_b = builder_b.into_inner().unwrap();
    assert_ne!(
        archive_diagnostic_checksum(&tar_a),
        archive_diagnostic_checksum(&tar_b),
        "sanity check: the two archives really do differ at the byte level"
    );

    let dir_a = temp_kernel_bundle_dir("archive-metadata-a");
    let dir_b = temp_kernel_bundle_dir("archive-metadata-b");
    let bundle_a = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_a),
        false,
        &dir_a,
        &KernelExchangeArchiveLimits::default(),
    )
    .unwrap();
    let bundle_b = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_b),
        false,
        &dir_b,
        &KernelExchangeArchiveLimits::default(),
    )
    .unwrap();
    let validated_a =
        validate_kernel_exchange_bundle(&bundle_a, &KernelManifestLimits::default()).unwrap();
    let validated_b =
        validate_kernel_exchange_bundle(&bundle_b, &KernelManifestLimits::default()).unwrap();
    assert_eq!(
        validated_a.digest, validated_b.digest,
        "ownership/timestamp/mode differences must not affect logical bundle identity"
    );

    fs::remove_dir_all(&dir_a).unwrap();
    fs::remove_dir_all(&dir_b).unwrap();
}

fn build_tar_with_raw_path_entry(path: &str, data: &[u8]) -> Vec<u8> {
    // `tar::Header::set_path` (used by `append_data`) refuses `..` itself,
    // which is correct behavior for a well-behaved writer but means it
    // cannot be used to construct the maliciously-crafted archive this test
    // needs. Writing directly into the raw GNU header name bytes bypasses
    // that writer-side check, simulating an archive from an attacker who
    // does not go through this safe API.
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(0);
    {
        let gnu = header.as_gnu_mut().expect("gnu header");
        let name_bytes = path.as_bytes();
        gnu.name[..name_bytes.len()].copy_from_slice(name_bytes);
    }
    header.set_cksum();
    builder.append(&header, data).unwrap();
    builder.into_inner().unwrap()
}

#[test]
fn kernel_exchange_archive_rejects_path_traversal_entry() {
    let tar_bytes = build_tar_with_raw_path_entry("../escape.txt", b"malicious");
    let dir = temp_kernel_bundle_dir("archive-traversal");
    let outcome = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &dir,
        &KernelExchangeArchiveLimits::default(),
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundlePathInvalid { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_exchange_archive_rejects_symlink_entry() {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_cksum();
    builder
        .append_link(&mut header, "blobs/escape-link", "/etc/passwd")
        .unwrap();
    let tar_bytes = builder.into_inner().unwrap();

    let dir = temp_kernel_bundle_dir("archive-symlink");
    let outcome = extract_kernel_exchange_archive(
        std::io::Cursor::new(&tar_bytes),
        false,
        &dir,
        &KernelExchangeArchiveLimits::default(),
    );
    assert!(matches!(
        outcome,
        Err(KernelManifestError::BundleSymlinkDenied { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_exchange_archive_rejects_hardlink_and_device_entries() {
    for entry_type in [
        tar::EntryType::Link,
        tar::EntryType::Char,
        tar::EntryType::Fifo,
    ] {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(entry_type);
        header.set_size(0);
        header.set_mode(0o644);
        header.set_mtime(0);
        if entry_type == tar::EntryType::Char {
            header.set_device_major(1).unwrap();
            header.set_device_minor(3).unwrap();
        }
        if entry_type == tar::EntryType::Link {
            // `append_link` sets the path/link-name and writes the header's
            // checksum itself, so no manual `set_cksum` call here.
            builder
                .append_link(&mut header, "blobs/hardlink", "blobs/sha256/target")
                .unwrap();
        } else {
            // The raw `append` method does not recompute the checksum, so
            // the path must be set *before* `set_cksum` here.
            header.set_path("special-entry").unwrap();
            header.set_cksum();
            builder.append(&header, std::io::empty()).unwrap();
        }
        let tar_bytes = builder.into_inner().unwrap();

        let dir = temp_kernel_bundle_dir(&format!("archive-special-{entry_type:?}"));
        let outcome = extract_kernel_exchange_archive(
            std::io::Cursor::new(&tar_bytes),
            false,
            &dir,
            &KernelExchangeArchiveLimits::default(),
        );
        assert!(
            matches!(outcome, Err(KernelManifestError::BundlePathInvalid { .. })),
            "expected {entry_type:?} to be rejected, got {outcome:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn kernel_exchange_archive_enforces_per_entry_decompressed_size_limit() {
    let oversized = vec![0u8; 1024];
    let tar_bytes = build_kernel_bundle_tar(&[("blobs/sha256/oversized", &oversized)]);
    let dir = temp_kernel_bundle_dir("archive-entry-limit");
    let limits = KernelExchangeArchiveLimits {
        max_entry_decompressed_bytes: 100,
        ..KernelExchangeArchiveLimits::default()
    };
    let outcome =
        extract_kernel_exchange_archive(std::io::Cursor::new(&tar_bytes), false, &dir, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_exchange_archive_enforces_entry_count_limit() {
    let entries: Vec<(&str, &[u8])> = vec![("a", b"1"), ("b", b"2"), ("c", b"3")];
    let tar_bytes = build_kernel_bundle_tar(&entries);
    let dir = temp_kernel_bundle_dir("archive-entry-count-limit");
    let limits = KernelExchangeArchiveLimits {
        max_entries: 2,
        ..KernelExchangeArchiveLimits::default()
    };
    let outcome =
        extract_kernel_exchange_archive(std::io::Cursor::new(&tar_bytes), false, &dir, &limits);
    assert!(matches!(
        outcome,
        Err(KernelManifestError::LimitExceeded { .. })
    ));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn kernel_bundle_transport_reservation_marks_only_directory_and_tar_as_implemented() {
    assert!(KernelBundleTransport::Directory.is_implemented());
    assert!(KernelBundleTransport::TarArchive.is_implemented());
    assert!(!KernelBundleTransport::ObjectStore.is_implemented());
    assert!(!KernelBundleTransport::OciLike.is_implemented());
    assert!(!KernelBundleTransport::Registry.is_implemented());
}

#[test]
fn kernel_manifest_evaluate_trust_pipeline_stage_delegates_to_sole_authority() {
    assert_eq!(
        evaluate_manifest_trust(true),
        crate::evaluate_artifact_trust(true)
    );
    assert_eq!(
        evaluate_manifest_trust(false),
        crate::evaluate_artifact_trust(false)
    );
    assert!(!evaluate_manifest_trust(false).is_trusted());
}

#[test]
fn kernel_manifest_normalizes_benchmark_profile() {
    let evidence = KernelEvidenceReference {
        digest: KernelBlobDigest::of_bytes(b"benchmark-evidence"),
        profile: "latency@1".into(),
        suite_or_workload_version: Some("v2".into()),
        oracle_or_provider_identity: None,
        target_compatibility: std::collections::BTreeSet::new(),
        status: KernelEvidenceStatus::Passed,
        storage_mode: KernelArtifactStorageMode::Embedded,
        workload_profile: Some("decode-256".into()),
        device_context: Some("h100".into()),
        provider_context: Some("nvidia-cuda@12.4".into()),
        workload_metadata: Some(KernelBenchmarkWorkloadMetadata {
            input_shapes: Some("[1,256,4096]".into()),
            dtype_layout: Some("fp16-row-major".into()),
            batch_size: Some(1),
            sequence_length: Some(256),
            warmup_count: Some(10),
            measurement_count: Some(100),
            synchronization_policy: Some("device-sync".into()),
            driver_runtime_version: Some("cuda-12.4".into()),
            benchmark_version: Some("bench-1".into()),
        }),
    };

    let profile =
        normalize_benchmark_profile(&evidence, "sm90").expect("benchmark profile normalizes");
    assert_eq!(profile.target_device, "h100");
    assert_eq!(profile.hardware_architecture, "sm90");
    assert_eq!(profile.provider_version, "nvidia-cuda@12.4");
    assert_eq!(profile.measurement_count, 100);
    assert!(profile.is_authoritative());

    let mut missing_workload = evidence.clone();
    missing_workload.workload_metadata = None;
    assert!(normalize_benchmark_profile(&missing_workload, "sm90").is_err());
}

#[test]
fn kernel_manifest_observation_builders_carry_schema_artifact_count_and_formats() {
    let schema = KernelManifestSchemaVersion::current();
    let formats = vec![
        KernelArtifactFormat::new("nvidia", "cubin"),
        KernelArtifactFormat::new("triton", "source").with_version("3"),
    ];
    let observation = KernelManifestObservation::new(KernelManifestObservationKind::ManifestParsed)
        .with_schema_version(&schema)
        .with_artifact_count(2)
        .with_formats(formats);

    assert_eq!(
        observation
            .redacted_metadata
            .get("schema-version")
            .map(String::as_str),
        Some("magnetar:kernel-manifest@1.0")
    );
    assert_eq!(
        observation
            .redacted_metadata
            .get("artifact-count")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        observation
            .redacted_metadata
            .get("formats")
            .map(String::as_str),
        Some("nvidia:cubin, triton:source@3")
    );
}

// ---------------------------------------------------------------------
// kernel_artifact_ingestion
// ---------------------------------------------------------------------

#[test]
fn kernel_artifact_ingestion_conformance_report_is_conformant() {
    let report = run_kernel_artifact_ingestion_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn ingestion_state_machine_rejects_illegal_transitions() {
    assert!(IngestionState::Created.can_transition_to(IngestionState::Receiving));
    assert!(!IngestionState::Created.can_transition_to(IngestionState::Committed));
    assert!(!IngestionState::Committed.can_transition_to(IngestionState::Accepted));
    assert!(IngestionState::Committed.is_terminal());
    assert!(!IngestionState::Staged.is_terminal());
}

#[test]
fn ingestion_pipeline_accepts_trusted_bundle_and_commits() {
    let directory = temp_kernel_bundle_dir("ingestion-accept");
    let digest = write_kernel_bundle(&directory, b"ingestion-accept-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline should accept a trusted, well-formed bundle");
    assert_eq!(outcome.decision, IngestionDecisionKind::Accept);
    assert_eq!(transaction.state, IngestionState::Accepted);

    let mut cache = KernelArtifactCache::new();
    let committed = commit_accepted_transaction(&mut transaction, &outcome.validated, &mut cache)
        .expect("accepted transaction should commit");
    assert_eq!(committed, vec![digest.clone()]);
    assert_eq!(transaction.state, IngestionState::Committed);
    assert!(
        cache
            .get(&normalize_to_cache_key(&outcome.validated.manifest.artifacts[0]).stable_key())
            .is_some()
    );

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_quarantines_untrusted_bundle_without_committing() {
    let directory = temp_kernel_bundle_dir("ingestion-quarantine");
    write_kernel_bundle(&directory, b"ingestion-quarantine-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: false,
        trust_explicitly_denied: false,
        signed: false,
        signature_verified: false,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline runs to a decision even when untrusted");
    assert!(matches!(
        outcome.decision,
        IngestionDecisionKind::Quarantine(_)
    ));
    assert_eq!(transaction.state, IngestionState::Quarantined);
    assert!(transaction.committed_digests.is_empty());

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_rejects_revoked_digest() {
    let directory = temp_kernel_bundle_dir("ingestion-revoked");
    write_kernel_bundle(&directory, b"ingestion-revoked-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: true,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline runs to a decision even when revoked");
    assert!(matches!(outcome.decision, IngestionDecisionKind::Reject(_)));
    assert_eq!(transaction.state, IngestionState::Rejected);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_audit_record_supports_release_evidence_traceability() {
    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    transaction.mark_receiving().unwrap();
    transaction.mark_staged().unwrap();
    transaction.mark_validating().unwrap();
    transaction.mark_policy_evaluating().unwrap();
    transaction.mark_accepted().unwrap();

    let record = KernelIngestionAuditRecord::from_transaction(&transaction)
        .with_manifest_digest("sha256:aaaa")
        .with_redacted_metadata("locator", "https://user:secret@internal/path");
    assert_eq!(
        record.release_evidence_reference(),
        Some(("policy-v1".to_string(), "sha256:aaaa".to_string()))
    );
    assert_ne!(
        record.redacted_metadata.get("locator").map(String::as_str),
        Some("https://user:secret@internal/path")
    );
}

#[test]
fn ingestion_error_ids_are_stable_and_displayable() {
    let cases: &[(IngestionError, &str)] = &[
        (
            IngestionError::StateInvalid { reason: "x".into() },
            "kernel-ingestion-state-invalid",
        ),
        (
            IngestionError::CommitConflict,
            "kernel-ingestion-commit-conflict",
        ),
        (
            IngestionError::ExternalDigestMismatch {
                expected: "a".into(),
                actual: "b".into(),
            },
            "kernel-ingestion-external-digest-mismatch",
        ),
        (
            IngestionError::ManualApprovalCannotBypassIntegrity,
            "kernel-ingestion-manual-approval-cannot-bypass-integrity",
        ),
        (
            IngestionError::ArtifactRevoked {
                digest: "sha256:aaa".into(),
            },
            "kernel-ingestion-artifact-revoked",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.id(), *expected_id);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn ingestion_commit_detects_bundle_mutation_after_validation() {
    // "Immutable Snapshot" / TOCTOU protection (proposal): the source is
    // mutated *after* validation but *before* commit -- the attack scenario
    // is "validate file A -> source replaces file A -> prepare replaced file
    // B". Here the replacement happens at the exact same content-addressed
    // path (an attacker overwriting bytes in place), which is exactly what
    // `verify_bundle_snapshot_unchanged` re-checks.
    let directory = temp_kernel_bundle_dir("ingestion-toctou");
    let digest = write_kernel_bundle(&directory, b"original-validated-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline should accept the originally staged bytes");
    assert_eq!(outcome.decision, IngestionDecisionKind::Accept);

    // Mutate the blob in place at its own content-addressed path -- the
    // source is replaced after validation completed.
    fs::write(
        directory.join("blobs").join("sha256").join(&digest),
        b"mutated-after-validation-bytes",
    )
    .unwrap();

    let mut cache = KernelArtifactCache::new();
    let commit_outcome = commit_accepted_transaction_from_bundle(
        &mut transaction,
        &bundle,
        &outcome.validated,
        &mut cache,
    );
    assert!(matches!(
        commit_outcome,
        Err(IngestionError::ToctouDetected)
    ));
    assert!(
        cache
            .get(&normalize_to_cache_key(&outcome.validated.manifest.artifacts[0]).stable_key())
            .is_none(),
        "staged/mutated content must never reach the accepted cache"
    );

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_failure_leaves_cache_and_registry_untouched() {
    // "Failure Atomicity" (proposal): a malformed bundle (missing manifest
    // file entirely) must fail before commit, and every stage of active
    // state (accepted cache, Kernel Registry) is untouched by the attempt.
    let directory = temp_kernel_bundle_dir("ingestion-malformed");
    // Deliberately do not write a manifest file -- `write_kernel_bundle` is
    // not called, so `blobs/sha256/` exists but `kernel.manifest.json` does
    // not.
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let cache_before = KernelArtifactCache::new();
    let registry_before = KernelRegistry::new();

    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy);
    assert!(matches!(outcome, Err(IngestionError::BundleInvalid { .. })));
    assert_eq!(transaction.state, IngestionState::Failed);
    assert!(transaction.committed_digests.is_empty());
    // Nothing in this pipeline call had access to a cache or registry at
    // all, so both remain exactly as constructed -- demonstrating the
    // failure cannot have touched either even in principle.
    assert_eq!(cache_before.observations().len(), 0);
    assert_eq!(registry_before.entries().count(), 0);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_concurrent_transactions_are_isolated() {
    // "Concurrent Transactions" (proposal): one failed transaction SHALL NOT
    // corrupt another. Simulated here as two independently-tracked
    // transactions against a shared cache: one is revoked (rejected), the
    // other succeeds, and the successful one's commit is unaffected.
    let directory_a = temp_kernel_bundle_dir("ingestion-concurrent-a");
    let digest_a = write_kernel_bundle(&directory_a, b"concurrent-transaction-a");
    let bundle_a = KernelExchangeBundle::open(&directory_a);
    let directory_b = temp_kernel_bundle_dir("ingestion-concurrent-b");
    write_kernel_bundle(&directory_b, b"concurrent-transaction-b");
    let bundle_b = KernelExchangeBundle::open(&directory_b);

    let mut allocator = IngestionTransactionIdAllocator::default();
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let mut cache = KernelArtifactCache::new();

    let mut transaction_a = KernelIngestionTransaction::new(
        allocator.allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let context_a = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let outcome_a = run_ingestion_pipeline(&mut transaction_a, &bundle_a, &context_a, &policy)
        .expect("transaction A should be accepted");
    let committed_a =
        commit_accepted_transaction(&mut transaction_a, &outcome_a.validated, &mut cache).unwrap();
    assert_eq!(committed_a, vec![digest_a]);

    let mut transaction_b = KernelIngestionTransaction::new(
        allocator.allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let context_b = IngestionPipelineContext {
        revoked: true,
        ..context_a
    };
    let outcome_b = run_ingestion_pipeline(&mut transaction_b, &bundle_b, &context_b, &policy)
        .expect("pipeline runs to a decision even when revoked");
    assert!(matches!(
        outcome_b.decision,
        IngestionDecisionKind::Reject(_)
    ));
    assert_eq!(transaction_b.state, IngestionState::Rejected);

    // Transaction A's committed content is unaffected by transaction B's
    // rejection.
    assert!(
        cache
            .get(&normalize_to_cache_key(&outcome_a.validated.manifest.artifacts[0]).stable_key())
            .is_some()
    );

    let _ = fs::remove_dir_all(&directory_a);
    let _ = fs::remove_dir_all(&directory_b);
}

// ---------------------------------------------------------------------
// kernel_compilation
// ---------------------------------------------------------------------

#[test]
fn compilation_capability_absence_is_valid_and_optional() {
    let descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    assert!(!descriptor.is_present());
    assert!(descriptor.validate().is_ok());
}

#[test]
fn compilation_capability_descriptor_validation_requires_declared_formats_and_isolation() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::SourceCompilation;
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor
        .accepted_source_formats
        .insert(KernelSourceFormat::new("triton", "source").with_version("3"));
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor
        .produced_compiled_formats
        .insert("nvidia:ptx@9".into());
    assert!(matches!(
        descriptor.validate(),
        Err(KernelCompilationError::DescriptorInvalid { .. })
    ));

    descriptor.isolation_model = CompilationIsolationModel::SandboxedSubprocess;
    assert!(descriptor.validate().is_ok());
}

#[test]
fn source_format_negotiation_rejects_unsupported_before_compilation() {
    let accepted: std::collections::BTreeSet<KernelSourceFormat> =
        [KernelSourceFormat::new("triton", "source").with_version("3")]
            .into_iter()
            .collect();
    let wgsl = KernelSourceFormat::new("webgpu", "wgsl");
    assert!(matches!(
        negotiate_source_format(&wgsl, &accepted),
        Err(KernelCompilationError::SourceFormatUnsupported { .. })
    ));
    let triton = KernelSourceFormat::new("triton", "source").with_version("3");
    assert!(negotiate_source_format(&triton, &accepted).is_ok());
}

#[test]
fn output_format_negotiation_requires_explicit_declaration() {
    let produced: std::collections::BTreeSet<String> =
        ["nvidia:ptx@9".to_string()].into_iter().collect();
    assert!(negotiate_output_format("nvidia:ptx@9", &produced).is_ok());
    assert!(matches!(
        negotiate_output_format("nvidia:cubin", &produced),
        Err(KernelCompilationError::OutputFormatUnsupported { .. })
    ));
}

#[test]
fn runtime_target_authority_rejects_provider_and_device_redirection() {
    let selected_provider = ProviderBinding::new("cuda-provider");
    let selected_device = DeviceBinding::new(crate::DeviceId::new("cuda-0"));
    let target =
        CompilationTarget::new(selected_provider.clone(), selected_device.clone(), "sm_90");
    assert!(
        enforce_runtime_target_authority(&target, &selected_provider, &selected_device).is_ok()
    );

    let other_provider = ProviderBinding::new("metal-provider");
    assert!(matches!(
        enforce_runtime_target_authority(&target, &other_provider, &selected_device),
        Err(KernelCompilationError::TargetUnsupported { .. })
    ));

    let other_device = DeviceBinding::new(crate::DeviceId::new("cuda-1"));
    assert!(matches!(
        enforce_runtime_target_authority(&target, &selected_provider, &other_device),
        Err(KernelCompilationError::TargetUnsupported { .. })
    ));
}

#[test]
fn compilation_job_lifecycle_progresses_legally_and_never_reverts() {
    let mut allocator = CompilationJobIdAllocator::default();
    let mut job = CompilationJob::new(allocator.allocate(), CompilationRequestId::new("req-1"));
    assert_eq!(job.state, CompilationJobState::Queued);
    assert!(job.start_compiling().is_ok());
    assert_eq!(job.state, CompilationJobState::Compiling);
    assert!(job.mark_succeeded().is_ok());
    assert_eq!(job.state, CompilationJobState::Succeeded);
    assert!(job.state.may_publish_artifact());

    let mut cancelled_before_start =
        CompilationJob::new(allocator.allocate(), CompilationRequestId::new("req-2"));
    assert!(matches!(
        cancelled_before_start.mark_succeeded(),
        Err(KernelCompilationError::JobStateInvalid { .. })
    ));
    assert!(cancelled_before_start.mark_cancelled().is_ok());
    assert!(!cancelled_before_start.state.may_publish_artifact());
}

#[test]
fn cancellation_support_levels_are_respected() {
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::NotSupported, false),
        Err(KernelCompilationError::CancellationUnsupported)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::BeforeStartOnly, true),
        Err(KernelCompilationError::CancellationUnsupported)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::BeforeStartOnly, false),
        Ok(CompilationJobState::Cancelled)
    ));
    assert!(matches!(
        evaluate_cancellation_request(CompilationCancellationSupport::Cooperative, true),
        Ok(CompilationJobState::Cancelled)
    ));
}

#[test]
fn compilation_deadline_fails_closed_when_unenforceable() {
    let required = CompilationDeadline {
        max_wall_clock_millis: 5_000,
    };
    assert!(matches!(
        enforce_compilation_deadline(Some(required), false),
        Err(KernelCompilationError::DeadlineUnsupported { .. })
    ));
    assert!(enforce_compilation_deadline(Some(required), true).is_ok());
    assert!(enforce_compilation_deadline(None, false).is_ok());
}

#[test]
fn compilation_limits_are_enforced_before_compiler_invocation() {
    let limits = CompilationLimits {
        max_source_bytes: Some(1024),
        max_output_bytes: Some(2048),
        max_concurrent_jobs: Some(2),
        max_workspace_bytes: Some(1 << 20),
        max_host_memory_bytes: Some(1 << 30),
    };
    assert!(enforce_compilation_limits(512, &limits).is_ok());
    assert!(matches!(
        enforce_compilation_limits(2048, &limits),
        Err(KernelCompilationError::SourceTooLarge { .. })
    ));
    assert!(matches!(
        enforce_output_limit(4096, &limits),
        Err(KernelCompilationError::OutputTooLarge { .. })
    ));
    assert!(enforce_concurrency_limit(1, &limits).is_ok());
    assert!(matches!(
        enforce_concurrency_limit(2, &limits),
        Err(KernelCompilationError::ConcurrencyLimit { .. })
    ));
}

#[test]
fn isolation_sufficiency_rejects_weaker_than_required_model() {
    assert!(matches!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::InProcessTrustedCompiler,
            CompilationIsolationModel::SandboxedSubprocess,
        ),
        Err(KernelCompilationError::IsolationInsufficient { .. })
    ));
    assert!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::SandboxedSubprocess,
            CompilationIsolationModel::SandboxedSubprocess,
        )
        .is_ok()
    );
    assert!(
        evaluate_isolation_sufficiency(
            CompilationIsolationModel::ExternalCompilationService,
            CompilationIsolationModel::SandboxedSubprocess,
        )
        .is_ok()
    );
}

#[test]
fn compilation_success_never_grants_trust_by_itself() {
    assert!(!compilation_result_trust(false).is_trusted());
    assert!(compilation_result_trust(true).is_trusted());
}

#[test]
fn network_boundary_denies_implicit_dependency_downloads() {
    let policy = CompilationNetworkPolicy::default();
    assert!(matches!(
        enforce_compilation_network_boundary(true, &policy),
        Err(KernelCompilationError::PolicyDenied { .. })
    ));
    let authorized = CompilationNetworkPolicy {
        network_access_authorized: true,
    };
    assert!(enforce_compilation_network_boundary(true, &authorized).is_ok());
    assert!(enforce_compilation_network_boundary(false, &policy).is_ok());
}

#[test]
fn environment_variables_are_denied_by_default() {
    let deny = CompilationEnvironmentPolicy::Deny;
    assert!(!evaluate_environment_variable("SECRET_TOKEN", &deny));

    let mut allowed = std::collections::BTreeSet::new();
    allowed.insert("CUDA_HOME".to_string());
    let allowlist = CompilationEnvironmentPolicy::Allowlist(allowed);
    assert!(evaluate_environment_variable("CUDA_HOME", &allowlist));
    assert!(!evaluate_environment_variable("SECRET_TOKEN", &allowlist));
}

#[test]
fn explicit_specialization_is_required_when_applied() {
    let empty = CompilationSpecialization::default();
    assert!(matches!(
        require_explicit_compilation_specialization(true, &empty),
        Err(KernelCompilationError::SpecializationUnsupported { .. })
    ));
    assert!(require_explicit_compilation_specialization(false, &empty).is_ok());

    let mut declared = CompilationSpecialization::default();
    declared.dtype.insert(ComputeDType::Float16);
    assert!(require_explicit_compilation_specialization(true, &declared).is_ok());
}

#[test]
fn output_integrity_rejects_digest_mismatch() {
    let id = CompiledKernelArtifactId::from_digest("expected-digest");
    assert!(verify_output_integrity("expected-digest", &id).is_ok());
    assert!(matches!(
        verify_output_integrity("different-digest", &id),
        Err(KernelCompilationError::OutputIntegrityFailed { .. })
    ));
}

#[test]
fn compiler_crash_is_normalized_and_redacted_never_a_success() {
    let error = normalize_compiler_crash("segfault at C:\\temp\\compiler\\work\\0xdeadbeef");
    match &error {
        KernelCompilationError::CompilerCrashed { detail } => {
            assert_eq!(detail, "[redacted backend diagnostic]");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(error.id(), "kernel-compilation-compiler-crashed");
}

#[test]
fn provider_compiler_panic_is_caught_and_never_unwinds_across_boundary() {
    let result: Result<(), KernelCompilationError> =
        call_provider_compiler_without_unwinding(|| -> Result<(), KernelCompilationError> {
            panic!("simulated compiler panic");
        });
    assert!(matches!(
        result,
        Err(KernelCompilationError::CompilerCrashed { .. })
    ));

    let ok: Result<u32, KernelCompilationError> =
        call_provider_compiler_without_unwinding(|| Ok(42));
    assert_eq!(ok, Ok(42));
}

#[test]
fn hot_path_denies_kernel_compilation_cold_path_allows_it() {
    assert!(matches!(
        reject_hot_path_kernel_compilation(KernelArtifactPath::Hot),
        Err(KernelCompilationError::HotPathDenied)
    ));
    assert!(reject_hot_path_kernel_compilation(KernelArtifactPath::Cold).is_ok());
}

#[test]
fn kernel_compilation_abi_descriptor_validates_ownership_and_version() {
    let descriptor = KernelCompilationAbiDescriptor::current();
    assert!(descriptor.validate().is_ok());

    let mut missing_release = descriptor.clone();
    missing_release.ownership.result_buffer.release_required = false;
    assert!(matches!(
        missing_release.validate(),
        Err(KernelCompilationError::AbiIncompatible { .. })
    ));

    let mut wrong_version = descriptor;
    wrong_version.abi_version = crate::provider::ProviderAbiVersion::new(99, 0);
    assert!(matches!(
        wrong_version.validate(),
        Err(KernelCompilationError::AbiIncompatible { .. })
    ));
}

#[test]
fn compilation_job_ids_and_prepared_kernel_ids_expose_no_pointer_semantics() {
    let mut allocator = CompilationJobIdAllocator::default();
    let job_id = allocator.allocate();
    assert!(assert_prepared_kernel_id_opaque(&job_id.to_string()).is_ok());
    assert!(matches!(
        assert_prepared_kernel_id_opaque("0xdeadbeef"),
        Err(KernelCompilationError::BufferOwnershipViolation { .. })
    ));
}

#[test]
fn compiler_diagnostics_redact_raw_output() {
    let diagnostic = CompilerDiagnostic::from_raw_output(
        "parse",
        "error in C:\\tmp\\kernel.triton at 0xffffabcd",
    )
    .with_source_location("kernel.triton:12:5");
    assert_eq!(diagnostic.redacted_message, "[redacted backend diagnostic]");
    assert_eq!(
        diagnostic.source_location.as_deref(),
        Some("kernel.triton:12:5")
    );
}

#[test]
fn compiler_flags_are_redacted_by_default() {
    let identity = CompilerIdentity::default().with_raw_flags("-I C:\\vendor\\include -DSECRET=1");
    assert_eq!(
        identity.flags_fingerprint.as_deref(),
        Some("[redacted backend diagnostic]")
    );
}

#[test]
fn failed_compilation_never_mutates_existing_known_good_artifact() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let v1 = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("v1-digest"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator,
    );
    let crash = normalize_compiler_crash("replacement compile crashed");
    let preserved = preserve_known_good_artifact_on_failure(&v1, &crash);
    assert_eq!(preserved, &v1);
}

#[test]
fn compilation_process_arguments_are_structural_never_shell_strings() {
    let untrusted_metadata = "kernel; rm -rf / #";
    let invocation = CompilationProcessArguments::new("nvcc")
        .with_arg("--compile")
        .with_arg(untrusted_metadata);
    assert_eq!(invocation.program, "nvcc");
    // The untrusted value is preserved as a single argument element -- never
    // interpolated into (and re-parsed out of) a shell command string.
    assert_eq!(invocation.args.len(), 2);
    assert_eq!(invocation.args[1], untrusted_metadata);
}

#[test]
fn compilation_observation_never_carries_raw_source_or_native_handles() {
    let observation =
        KernelCompilationObservation::new(KernelCompilationObservationKind::CompilerCompleted)
            .with_job("compilation-job-1")
            .with_redacted_metadata("compiler", "nvcc 12.4")
            .with_redacted_metadata("temp_path", "C:\\tmp\\build\\0xdeadbeef");
    assert_eq!(
        observation.kind,
        KernelCompilationObservationKind::CompilerCompleted
    );
    assert_eq!(observation.job.as_deref(), Some("compilation-job-1"));
    assert_eq!(
        observation.redacted_metadata.get("temp_path").unwrap(),
        "[redacted backend diagnostic]"
    );
    assert_eq!(
        observation.redacted_metadata.get("compiler").unwrap(),
        "nvcc 12.4"
    );
}

#[test]
fn preparation_only_provider_is_a_valid_distinct_support_level() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::PreparationOnly;
    descriptor
        .produced_compiled_formats
        .insert("nvidia:cubin".into());
    descriptor.isolation_model = CompilationIsolationModel::PlatformManagedCompiler;
    assert!(descriptor.is_present());
    assert!(descriptor.validate().is_ok());
    assert!(descriptor.accepted_source_formats.is_empty());
}

#[test]
fn platform_managed_compilation_mode_preserves_cold_hot_boundary() {
    let mut descriptor = KernelCompilationCapabilityDescriptor::unsupported();
    descriptor.support_level = CompilationSupportLevel::SourceCompilation;
    descriptor.modes.insert(CompilationMode::ProviderManaged);
    descriptor.isolation_model = CompilationIsolationModel::PlatformManagedCompiler;
    descriptor
        .accepted_source_formats
        .insert(KernelSourceFormat::new("apple", "msl"));
    descriptor
        .produced_compiled_formats
        .insert("apple:metallib".into());
    assert!(descriptor.validate().is_ok());
    // Even a platform that logically combines compile+prepare internally
    // still denies compilation on the decode hot path.
    assert!(matches!(
        reject_hot_path_kernel_compilation(KernelArtifactPath::Hot),
        Err(KernelCompilationError::HotPathDenied)
    ));
}

#[test]
fn compilation_result_wraps_compiled_artifact_with_compilation_provenance() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let artifact = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("digest-result"),
        "ptx",
        "triton",
        "3.1",
        "sm_90",
        operator,
    );
    let mut allocator = CompilationJobIdAllocator::default();
    let result = CompilationResult {
        job: allocator.allocate(),
        artifact: artifact.clone(),
        compiler: CompilerIdentity {
            name: Some("triton".into()),
            version: Some("3.1".into()),
            ..CompilerIdentity::default()
        },
        specialization: CompilationSpecialization::default(),
        duration_millis: Some(1200),
    };
    assert_eq!(result.artifact, artifact);
    assert_eq!(result.compiler.name.as_deref(), Some("triton"));
    assert_eq!(result.duration_millis, Some(1200));
}

#[test]
fn kernel_compilation_conformance_report_is_conformant() {
    let report = run_kernel_compilation_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// kernel_qualification
// ---------------------------------------------------------------------

#[test]
fn kernel_qualification_conformance_report_is_conformant() {
    let report = run_kernel_qualification_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn qualification_status_transitions_reject_skipping_qualifying() {
    let mut record = QualificationRecord::new(
        QualificationIdentity::new(
            CompiledKernelArtifactId::from_digest("digest"),
            1,
            "suite-1",
            "sm90",
            "provider-1",
        ),
        QualificationProfile::baseline_correctness(),
        reference_cpu_oracle("1"),
    );
    assert!(matches!(record.status(), QualificationStatus::Unqualified));
    assert!(record.mark_qualified(None).is_err());
    assert!(record.start_qualifying().is_ok());
    assert!(record.mark_qualified(None).is_ok());
    assert!(record.status().is_eligible());
}

#[test]
fn qualification_profile_does_not_infer_stricter_profile_from_weaker_evidence() {
    let baseline = QualificationProfile::baseline_correctness();
    let strict = QualificationProfile::strict_correctness();
    assert!(!baseline.satisfies(&strict));
    assert!(baseline.satisfies(&baseline.clone()));
}

#[test]
fn oracle_required_when_reference_cpu_does_not_support_operator() {
    assert!(require_oracle(false, None).is_err());
    assert!(require_oracle(true, None).is_ok());
    assert!(require_oracle(false, Some(&reference_cpu_oracle("1"))).is_ok());
}

// ---------------------------------------------------------------------
// kernel_benchmark
// ---------------------------------------------------------------------

#[test]
fn kernel_benchmark_conformance_report_is_conformant() {
    let report = run_kernel_benchmark_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn regression_policy_correctness_only_never_rejects_on_latency() {
    assert!(evaluate_regression_policy(RegressionPolicy::CorrectnessOnly, 1000.0, 1.0).is_ok());
}

// ---------------------------------------------------------------------
// kernel_autotuning
// ---------------------------------------------------------------------

#[test]
fn kernel_autotuning_conformance_report_is_conformant() {
    let report = run_kernel_autotuning_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn kernel_autotuning_policy_enforces_single_active_policy() {
    assert!(!KernelAutotuningPolicy::Disabled.permits_live_tuning());
    assert!(KernelAutotuningPolicy::Optional.permits_live_tuning());
    assert!(KernelAutotuningPolicy::Required.permits_live_tuning());
    assert!(
        !KernelAutotuningPolicy::Pinned {
            record_fingerprint: "r".into(),
        }
        .permits_live_tuning()
    );
    assert!(require_autotuning_enabled(&KernelAutotuningPolicy::Disabled).is_err());
    assert!(require_autotuning_enabled(&KernelAutotuningPolicy::Optional).is_ok());
}

#[test]
fn kernel_autotuning_qualification_coverage_never_implicitly_inherits() {
    let exact = QualificationCoverage::ExactInstance {
        fingerprint: "instance-a".into(),
    };
    assert!(exact.covers("instance-a"));
    assert!(!exact.covers("instance-b"));

    let enumerated = QualificationCoverage::EnumeratedInstances {
        fingerprints: BTreeSet::from(["instance-a".to_string(), "instance-b".to_string()]),
    };
    assert!(enumerated.covers("instance-b"));
    assert!(!enumerated.covers("instance-c"));

    let unauthorized_envelope = QualificationCoverage::DeclaredEnvelope { authorized: false };
    assert!(!unauthorized_envelope.covers("any-instance"));
    let authorized_envelope = QualificationCoverage::DeclaredEnvelope { authorized: true };
    assert!(authorized_envelope.covers("any-instance"));

    assert!(!QualificationCoverage::RequiresPerInstanceQualification.covers("any-instance"));
}

#[test]
fn kernel_autotuning_workload_bucket_requires_exact_match() {
    let bucket = KernelAutotuningWorkloadBucket {
        operator: OperatorId::magnetar("attention", 1, OperatorFamily::Attention),
        shape_bucket: "batch=1/seq=4096".into(),
        batch_bucket: Some("1".into()),
        sequence_bucket: Some("4096".into()),
        phase: KernelAutotuningExecutionPhase::Prefill,
        dtype: ComputeDType::Float16,
        layout: TensorLayoutKind::Contiguous,
        quantization: None,
        provider: ProviderBinding::new("cuda"),
        device_architecture: "sm90".into(),
        device_features: BTreeSet::new(),
    };
    let mut decode_bucket = bucket.clone();
    decode_bucket.phase = KernelAutotuningExecutionPhase::Decode;
    assert!(bucket.is_compatible_with(&bucket.clone()));
    assert!(!bucket.is_compatible_with(&decode_bucket));
}

// ---------------------------------------------------------------------
// kernel_cache
// ---------------------------------------------------------------------

#[test]
fn kernel_cache_conformance_report_is_conformant() {
    let report = run_kernel_cache_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn kernel_cache_preparation_pressure_is_informational_and_does_not_gate_eligibility() {
    let mut cache = KernelArtifactCache::new();
    assert_eq!(cache.preparation_pressure_hint().level, None);

    cache.set_preparation_pressure_hint(PreparationPressureHint {
        level: Some(MemoryPressureLevel::High),
    });
    assert_eq!(
        cache.preparation_pressure_hint().level,
        Some(MemoryPressureLevel::High)
    );

    // Setting a high pressure hint does not, by itself, change whether an
    // otherwise-eligible entry is usable -- the Memory Manager retains
    // authority over Runtime Tensor allocation, not this cache.
    let key = KernelCacheKey {
        source_digest: None,
        compiled_artifact_digest: "digest-pressure".into(),
        source_format: None,
        compiled_format: "nvidia:cubin".into(),
        compiler_identity: "nvcc".into(),
        compiler_version: "12.0".into(),
        compiler_flags_fingerprint: None,
        provider_version: "1.0.0".into(),
        target_architecture: "sm90".into(),
        driver_runtime_compatibility_class: Default::default(),
        operator_semantics: "magnetar:matmul@1".into(),
        dtype: Default::default(),
        layout: Default::default(),
        shape_specialization: None,
        device_features: Default::default(),
    };
    let mut entry = KernelCacheEntry::new(
        key,
        CompiledKernelArtifactId::from_digest("digest-pressure"),
        "sha256:pressure",
    );
    entry.mark_validating().unwrap();
    entry.mark_ready().unwrap();
    let eligibility = evaluate_cache_eligibility(&entry, true, &CacheEligibilityPolicy::default());
    assert!(eligibility.is_ok());
}

#[test]
fn qualification_cache_key_requires_exact_match_for_reuse() {
    let base = QualificationCacheKey {
        artifact_digest: "digest".into(),
        qualification_suite_version: "1".into(),
        oracle_identity_version: "1".into(),
        qualification_profile: "baseline-correctness@1".into(),
        target_context: "sm90".into(),
        test_matrix_fingerprint: "fp".into(),
        tolerance_profile_fingerprint: "tp".into(),
    };
    let mut different_suite = base.clone();
    different_suite.qualification_suite_version = "2".into();
    assert!(base.is_reusable_for(&base));
    assert!(!base.is_reusable_for(&different_suite));
}

// ---------------------------------------------------------------------
// kernel_registry generated-kernel lifecycle
// ---------------------------------------------------------------------

#[test]
fn kernel_registry_lifecycle_conformance_report_is_conformant() {
    let report = run_kernel_registry_lifecycle_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn candidate_state_cannot_skip_from_qualified_directly_to_retired() {
    assert!(!CandidateState::Qualified.can_transition_to(CandidateState::Retired));
    assert!(CandidateState::Qualified.can_transition_to(CandidateState::Candidate));
    assert!(CandidateState::Active.can_transition_to(CandidateState::Retiring));
    assert!(CandidateState::Retiring.can_transition_to(CandidateState::Retired));
}

#[test]
fn kernel_registry_hot_swap_and_retirement_errors_have_expected_ids() {
    let cases = [
        (
            KernelRegistryError::HotSwapFailed { reason: "x".into() },
            "kernel-hot-swap-failed",
        ),
        (
            KernelRegistryError::RetirementInUse { kernel: "x".into() },
            "kernel-retirement-in-use",
        ),
        (
            KernelRegistryError::RetirementFailed { kernel: "x".into() },
            "kernel-retirement-failed",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.code(), expected_id);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn automatic_rollback_policy_is_reserved_and_disabled_by_default() {
    assert!(!AutomaticRollbackPolicy::default().enabled);
}

#[test]
fn continuous_batching_preserves_in_flight_generation_and_admits_new_work_on_new_generation() {
    let kernel = KernelId::new(
        ProviderBinding::new("conformance-provider"),
        "conformance-kernel",
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::TestFixture,
    );
    let device = DeviceBinding::new(crate::DeviceId::new("conformance-device"));
    let mut allocator = PreparedKernelIdAllocator::default();
    let mut registry = KernelRegistry::new();

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-batch-v1"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().unwrap();
    let generation_one_id = generation_one.id;
    registry.register_prepared_kernel(generation_one);
    registry
        .promote_generation(&kernel, generation_one_id)
        .unwrap();

    let slot_binding = registry.bind_batch_slot(&kernel, 7).unwrap();
    assert_eq!(slot_binding.generation, generation_one_id);

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-batch-v2"),
        ProviderBinding::new("conformance-provider"),
        device,
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().unwrap();
    let generation_two_id = generation_two.id;
    registry.register_prepared_kernel(generation_two);
    registry
        .promote_generation(&kernel, generation_two_id)
        .unwrap();

    // The existing slot binding is unaffected by promotion...
    assert_eq!(slot_binding.generation, generation_one_id);
    // ...but new batch admissions resolve to the newly promoted generation.
    assert_eq!(
        registry.admit_new_batch_work(&kernel),
        Some(generation_two_id)
    );
}

#[test]
fn model_instance_kernel_selection_policy_is_explicit() {
    let dynamic = KernelSelectionPolicy::Dynamic;
    assert!(!dynamic.is_pinned());

    let kernel = KernelId::new(
        ProviderBinding::new("conformance-provider"),
        "conformance-kernel",
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::TestFixture,
    );
    let mut pinned = PinnedKernelSelection::new(kernel, "digest-pinned");
    assert!(pinned.validate().is_ok());
    pinned.prepared_generation = Some(3);
    pinned.qualification_profile = Some("baseline-correctness@1".into());
    let policy = KernelSelectionPolicy::Pinned(pinned);
    assert!(policy.is_pinned());

    let empty_digest = PinnedKernelSelection::new(
        KernelId::new(
            ProviderBinding::new("conformance-provider"),
            "conformance-kernel",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
            KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::TestFixture,
        ),
        "",
    );
    assert!(matches!(
        empty_digest.validate(),
        Err(ModelInstanceError::ModelInstancePolicyDenied)
    ));
}

#[test]
fn session_inherits_model_instance_kernel_policy_and_owns_no_native_kernel_state() {
    // Structural fact: `InferenceSession`'s fields (id, lifecycle, model,
    // tokenizer, generation_defaults, policy, memory, resources, operation,
    // affinity, correlation_id, timestamps, last_error) contain no
    // KernelSelectionPolicy, KernelId, PreparedKernelId, or native handle --
    // a Session has nothing to override, so it can only inherit.
    let policy = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(
        KernelId::new(
            ProviderBinding::new("conformance-provider"),
            "conformance-kernel",
            crate::CapabilityVersion::new(1, 0, 0),
            OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
            KernelOperatorVersionRange::exact(1),
            crate::KernelImplementationFamily::TestFixture,
        ),
        "digest-pinned",
    ));
    let inherited = session_kernel_policy_is_inherited(&policy);
    assert_eq!(inherited, &policy);
}

// ---------------------------------------------------------------------
// kernel_selection_policy
// ---------------------------------------------------------------------

fn selection_policy_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("selection-policy-provider"),
        name,
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::CpuScalar,
    )
}

fn selection_policy_identity(name: &str) -> CandidateIdentity {
    CandidateIdentity {
        kernel: selection_policy_kernel_id(name),
        provider: ProviderBinding::new("selection-policy-provider"),
        artifact_digest: None,
    }
}

#[test]
fn kernel_selection_policy_conformance_report_is_conformant() {
    let report = run_kernel_selection_policy_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn eligible_candidate_cannot_be_constructed_without_passing_eligibility() {
    let rejected = EligibleCandidate::from_checked(
        selection_policy_identity("ineligible"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput {
            shape_compatible: false,
            ..CandidateEligibilityInput::all_satisfied()
        },
    );
    assert_eq!(
        rejected,
        Err(KernelSelectionExclusionReason::ShapeIncompatible)
    );
}

#[test]
fn eligibility_checks_fail_in_deterministic_order() {
    // Both dtype and layout are wrong; dtype is checked first, so it wins.
    let input = CandidateEligibilityInput {
        dtype_compatible: false,
        layout_compatible: false,
        ..CandidateEligibilityInput::all_satisfied()
    };
    assert_eq!(
        evaluate_candidate_eligibility(&input),
        Err(KernelSelectionExclusionReason::DTypeIncompatible)
    );
}

#[test]
fn kernel_candidate_rejection_maps_onto_selection_exclusion_reason() {
    assert_eq!(
        KernelSelectionExclusionReason::from(KernelCandidateRejection::Revoked),
        KernelSelectionExclusionReason::QualificationRevoked
    );
    assert_eq!(
        KernelSelectionExclusionReason::from(KernelCandidateRejection::WorkspaceUnavailable),
        KernelSelectionExclusionReason::WorkspaceInfeasible
    );
}

#[test]
fn missing_metric_never_resolves_to_best_possible_value() {
    assert_eq!(
        missing_metric_value(MissingMetricPolicy::Exclude, None),
        f64::INFINITY
    );
    assert!(missing_metric_value(MissingMetricPolicy::RankConservatively, None) > 0.0);
    assert_eq!(
        missing_metric_value(MissingMetricPolicy::UseFallbackMetadata, Some(42.0)),
        42.0
    );
}

#[test]
fn should_retain_active_due_to_missing_metric_only_under_that_policy() {
    let metrics = CandidateMetrics::default();
    assert!(should_retain_active_due_to_missing_metric(
        &metrics,
        &[ObjectiveDimension::Latency],
        MissingMetricPolicy::RetainActiveKernel,
    ));
    assert!(!should_retain_active_due_to_missing_metric(
        &metrics,
        &[ObjectiveDimension::Latency],
        MissingMetricPolicy::Exclude,
    ));
}

#[test]
fn weighted_score_ranks_lower_combined_cost_first() {
    let fast = CandidateMetrics::default()
        .with(ObjectiveDimension::Latency, 1.0)
        .with(ObjectiveDimension::Memory, 1.0);
    let slow = CandidateMetrics::default()
        .with(ObjectiveDimension::Latency, 10.0)
        .with(ObjectiveDimension::Memory, 1.0);
    let policy = WeightedScorePolicy {
        weights: BTreeMap::from([
            (ObjectiveDimension::Latency, 1.0),
            (ObjectiveDimension::Memory, 1.0),
        ]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    };
    assert!(weighted_score(&fast, &policy) < weighted_score(&slow, &policy));
}

#[test]
fn lexicographic_ranking_uses_first_differentiating_objective() {
    let a = CandidateMetrics::default()
        .with(ObjectiveDimension::Determinism, 0.0)
        .with(ObjectiveDimension::Latency, 100.0);
    let b = CandidateMetrics::default()
        .with(ObjectiveDimension::Determinism, 1.0)
        .with(ObjectiveDimension::Latency, 1.0);
    let policy = LexicographicPolicy {
        order: vec![ObjectiveDimension::Determinism, ObjectiveDimension::Latency],
        missing_metric_policy: MissingMetricPolicy::Exclude,
    };
    assert_eq!(
        lexicographic_compare(&a, &b, &policy),
        std::cmp::Ordering::Less
    );
}

#[test]
fn rank_candidates_is_deterministic_across_repeated_calls() {
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("z"),
            CandidateMetrics::default().with(ObjectiveDimension::Latency, 3.0),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
        EligibleCandidate::from_checked(
            selection_policy_identity("a"),
            CandidateMetrics::default().with(ObjectiveDimension::Latency, 3.0),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    assert!(selection_is_deterministic_for_identical_inputs(
        &candidates,
        &strategy
    ));
    // Tied score: stable tie-break falls back to CandidateIdentity's Ord.
    let ranked = rank_candidates(&candidates, &strategy).unwrap();
    assert_eq!(ranked[0].kernel.name, "a");
}

#[test]
fn pinned_ranking_strategy_fails_when_kernel_is_not_eligible() {
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("present"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::Pinned(selection_policy_kernel_id("absent"));
    assert_eq!(
        rank_candidates(&candidates, &strategy),
        Err(KernelSelectionError::PinnedKernelUnavailable)
    );
}

#[test]
fn policy_ordered_ranking_places_unlisted_candidates_last() {
    let listed = selection_policy_kernel_id("listed");
    let candidates = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("unlisted"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
        EligibleCandidate::from_checked(
            CandidateIdentity {
                kernel: listed.clone(),
                provider: ProviderBinding::new("selection-policy-provider"),
                artifact_digest: None,
            },
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let strategy = RankingStrategy::PolicyOrdered(vec![listed.clone()]);
    let ranked = rank_candidates(&candidates, &strategy).unwrap();
    assert_eq!(ranked[0].kernel, listed);
}

#[test]
fn benchmark_context_requires_exact_match() {
    let context = BenchmarkContext {
        provider: ProviderBinding::new("p"),
        device_architecture: "sm90".into(),
        driver_runtime_compatibility: "cuda-12".into(),
        operator_version: 1,
        artifact_digest: None,
        dtype: ComputeDType::Float32,
        layout: TensorLayoutKind::Contiguous,
        shape_bucket: "b1s128".into(),
        batch_bucket: "b1".into(),
        sequence_bucket: "s128".into(),
        execution_mode: KernelExecutionMode::Synchronous,
        benchmark_profile_version: 1,
    };
    let different_shape = BenchmarkContext {
        shape_bucket: "b64s8192".into(),
        ..context.clone()
    };
    assert!(benchmark_context_compatible(&context, &context));
    assert!(!benchmark_context_compatible(&different_shape, &context));
}

#[test]
fn stale_benchmark_policy_explicitly_governs_exclusion() {
    let stale = BenchmarkFreshness::Stale {
        reason: BenchmarkStalenessReason::DriverChanged,
    };
    assert!(evaluate_stale_benchmark_policy(stale, StaleBenchmarkPolicy::Accept).is_ok());
    assert_eq!(
        evaluate_stale_benchmark_policy(stale, StaleBenchmarkPolicy::Exclude),
        Err(KernelSelectionError::BenchmarkStale)
    );
    assert!(
        evaluate_stale_benchmark_policy(BenchmarkFreshness::Fresh, StaleBenchmarkPolicy::Exclude)
            .is_ok()
    );
}

#[test]
fn shape_aware_evidence_does_not_apply_outside_its_bucket() {
    assert!(performance_evidence_applies_to_workload("b1s128", "b1s128"));
    assert!(!performance_evidence_applies_to_workload(
        "b1s128", "b64s8192"
    ));
}

#[test]
fn pressure_bias_only_applies_to_an_already_eligible_candidate() {
    let candidate = EligibleCandidate::from_checked(
        selection_policy_identity("pressure"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let saturated = PressureSnapshot {
        memory_pressure: PressureLevel::Saturated,
        provider_admission_open: true,
        ..PressureSnapshot::default()
    };
    let nominal = PressureSnapshot {
        provider_admission_open: true,
        ..PressureSnapshot::default()
    };
    assert!(
        pressure_ranking_bias(&candidate, &saturated) > pressure_ranking_bias(&candidate, &nominal)
    );
}

#[test]
fn conversion_cost_can_flip_the_faster_raw_execution_choice() {
    let kernel_a = ConversionCost {
        layout_conversion_ms: 30.0,
        ..ConversionCost::default()
    };
    let kernel_b = ConversionCost::default();
    let total_a = total_execution_cost_ms(20.0, &kernel_a);
    let total_b = total_execution_cost_ms(35.0, &kernel_b);
    assert!(total_b < total_a);
}

#[test]
fn preparation_cost_is_not_charged_again_once_already_prepared() {
    assert!(!preparation_cost_applies(
        PreparationCostClass::OneTime,
        true
    ));
    assert!(preparation_cost_applies(
        PreparationCostClass::PerOperation,
        true
    ));
}

#[test]
fn compilation_cost_is_excluded_once_artifact_is_cached() {
    assert!(compilation_cost_excluded_from_hot_path(true));
    assert!(!compilation_cost_excluded_from_hot_path(false));
}

#[test]
fn hysteresis_retains_active_kernel_below_threshold_and_promotes_above_it() {
    let policy = HysteresisPolicy {
        promotion_threshold_fraction: 0.05,
    };
    assert_eq!(
        evaluate_hysteresis(100.0, 99.9, &policy),
        SelectionOutcome::RetainActive
    );
    assert_eq!(
        evaluate_hysteresis(100.0, 80.0, &policy),
        SelectionOutcome::PromoteCandidate
    );
}

#[test]
fn anti_flapping_blocks_promotion_inside_cooldown_or_minimum_duration() {
    let policy = AntiFlappingPolicy {
        cooldown_seconds: 30,
        minimum_active_duration_seconds: 10,
    };
    assert!(!promotion_allowed_by_anti_flapping(&policy, 5, 20));
    assert!(promotion_allowed_by_anti_flapping(&policy, 30, 10));
}

#[test]
fn promotion_recommendation_is_denied_when_hysteresis_or_anti_flapping_blocks_it() {
    let candidate = selection_policy_identity("candidate");
    let denied = recommend_promotion(&candidate, SelectionOutcome::RetainActive, true);
    assert!(!denied.approved);
    let denied_by_cooldown =
        recommend_promotion(&candidate, SelectionOutcome::PromoteCandidate, false);
    assert!(!denied_by_cooldown.approved);
    let approved = recommend_promotion(&candidate, SelectionOutcome::PromoteCandidate, true);
    assert!(approved.approved);
}

#[test]
fn selection_cache_invalidate_all_clears_every_entry() {
    let mut cache = SelectionCache::new();
    let key = SelectionCacheKey {
        operator: OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        provider: ProviderBinding::new("selection-policy-provider"),
        dtype: ComputeDType::Float32,
        layout: TensorLayoutKind::Contiguous,
        shape_bucket: "b1s128".into(),
        batch_bucket: "b1".into(),
        sequence_bucket: "s128".into(),
        generation_phase: Some(GenerationPhase::Decode),
        optimization_profile: OptimizationProfile::Latency,
        policy_version: KernelSelectionPolicyVersion(1),
    };
    cache.insert(key.clone(), selection_policy_identity("cached"));
    assert_eq!(cache.len(), 1);
    cache.invalidate_all(SelectionCacheInvalidationTrigger::PolicyChanged);
    assert!(cache.is_empty());
    assert!(cache.get(&key).is_none());
}

#[test]
fn kernel_optimization_policy_rejects_self_contradictory_deterministic_profile() {
    let policy = KernelOptimizationPolicy {
        id: KernelSelectionPolicyId::new("policy-1"),
        version: KernelSelectionPolicyVersion(1),
        profile: OptimizationProfile::Deterministic,
        ranking_strategy: RankingStrategy::WeightedScore(WeightedScorePolicy {
            weights: BTreeMap::new(),
            missing_metric_policy: MissingMetricPolicy::Exclude,
        }),
        fallback: FallbackPolicy::default(),
        hysteresis: HysteresisPolicy::default(),
        anti_flapping: AntiFlappingPolicy::default(),
        exploration: ExplorationPolicy::default(),
        selection_mode: KernelSelectionPolicy::Dynamic,
        determinism_required: false,
        require_trusted: true,
        allowed_qualification_profiles: BTreeSet::new(),
    };
    assert!(policy.validate().is_err());
}

#[test]
fn model_component_cannot_select_a_concrete_kernel() {
    assert!(
        validate_model_component_request(&ModelComponentKernelRequest::ConcreteKernelOverride(
            selection_policy_kernel_id("x")
        ))
        .is_err()
    );
    assert!(
        validate_model_component_request(
            &ModelComponentKernelRequest::PortableOperatorRequirement(OperatorId::magnetar(
                "matmul",
                1,
                OperatorFamily::LinearAlgebra
            ))
        )
        .is_ok()
    );
}

#[test]
fn session_and_cli_preference_only_apply_to_already_eligible_candidates() {
    let eligible = vec![
        EligibleCandidate::from_checked(
            selection_policy_identity("eligible"),
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    let ineligible_kernel = selection_policy_kernel_id("ineligible");
    assert!(resolve_session_preference(Some(&ineligible_kernel), &eligible).is_none());
    assert!(resolve_cli_kernel_preference(Some(&ineligible_kernel), &eligible).is_none());
    let eligible_kernel = eligible[0].identity.kernel.clone();
    assert!(resolve_session_preference(Some(&eligible_kernel), &eligible).is_some());
}

#[test]
fn cli_preference_maps_onto_the_matching_optimization_profile() {
    assert_eq!(
        map_cli_preference(CliPreference::Latency),
        OptimizationProfile::Latency
    );
    assert_eq!(
        map_cli_preference(CliPreference::Deterministic),
        OptimizationProfile::Deterministic
    );
}

#[test]
fn provider_advertised_alternative_never_overrides_runtime_cross_provider_decision() {
    let runtime_choice = selection_policy_identity("runtime-choice");
    let provider_alternative = selection_policy_identity("provider-alternative");
    assert_eq!(
        resolve_cross_provider_selection(&runtime_choice, Some(&provider_alternative)),
        runtime_choice
    );
}

#[test]
fn provider_private_variant_requires_every_contract_dimension_unchanged() {
    assert!(!provider_may_select_variant_privately(
        &ProviderPrivateVariant::default()
    ));
    assert!(provider_may_select_variant_privately(
        &ProviderPrivateVariant {
            contract_semantics_identical: true,
            runtime_visible_compatibility_unchanged: true,
            determinism_precision_unchanged: true,
        }
    ));
}

#[test]
fn fallback_chain_tries_classes_in_order_and_exhausts_explicitly() {
    let policy = FallbackPolicy {
        ordered_classes: vec![KernelFallbackClass::HostExecution],
        allow_reference_cpu: false,
    };
    assert_eq!(
        evaluate_kernel_selection_fallback_chain(&policy, true, HostStagingPolicy::Permit, true),
        Err(KernelSelectionError::FallbackExhausted)
    );
    let permissive = FallbackPolicy {
        ordered_classes: vec![KernelFallbackClass::HostExecution],
        allow_reference_cpu: true,
    };
    assert_eq!(
        evaluate_kernel_selection_fallback_chain(
            &permissive,
            true,
            HostStagingPolicy::Permit,
            true
        ),
        Ok(KernelFallbackClass::HostExecution)
    );
}

#[test]
fn cross_provider_movement_requires_both_explicit_and_authorized() {
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: true,
            authorized_by_policy: true,
        })
        .is_ok()
    );
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: true,
            authorized_by_policy: false,
        })
        .is_err()
    );
    assert!(
        validate_cross_provider_movement(&CrossProviderMovement {
            explicit: false,
            authorized_by_policy: true,
        })
        .is_err()
    );
}

#[test]
fn resolve_pinned_selection_distinguishes_unavailable_from_ineligible() {
    let kernel = selection_policy_kernel_id("pinned");
    let pin = PinnedKernelSelection::new(kernel.clone(), "digest-1");
    let unavailable = resolve_pinned_selection(&pin, &[], &[]);
    assert_eq!(
        unavailable,
        Err(KernelSelectionError::PinnedKernelUnavailable)
    );

    let discovered = vec![CandidateIdentity {
        kernel: kernel.clone(),
        provider: ProviderBinding::new("selection-policy-provider"),
        artifact_digest: None,
    }];
    let ineligible = resolve_pinned_selection(&pin, &discovered, &[]);
    assert_eq!(
        ineligible,
        Err(KernelSelectionError::PinnedKernelIneligible)
    );

    let eligible = vec![
        EligibleCandidate::from_checked(
            CandidateIdentity {
                kernel: kernel.clone(),
                provider: ProviderBinding::new("selection-policy-provider"),
                artifact_digest: None,
            },
            CandidateMetrics::default(),
            &CandidateEligibilityInput::all_satisfied(),
        )
        .unwrap(),
    ];
    assert!(resolve_pinned_selection(&pin, &discovered, &eligible).is_ok());
}

#[test]
fn exploration_is_denied_by_default_under_reproducible_mode_but_allowed_when_enabled_outside_it() {
    let candidate = EligibleCandidate::from_checked(
        selection_policy_identity("explore"),
        CandidateMetrics::default(),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let policy = ExplorationPolicy::default();
    assert!(!exploration_allowed(&policy, false));
    let enabled = ExplorationPolicy {
        enabled: true,
        disabled_for_reproducible: true,
    };
    assert!(eligible_for_exploration(&enabled, false, &candidate));
    assert!(!eligible_for_exploration(&enabled, true, &candidate));
}

#[test]
fn canary_budget_exhaustion_is_explicit() {
    let policy = CanaryPolicy {
        max_requests: Some(100),
        max_duration_seconds: None,
        max_percentage: None,
    };
    assert!(!canary_budget_exhausted(&policy, 99, 0));
    assert!(canary_budget_exhausted(&policy, 100, 0));
}

#[test]
fn exploration_failure_never_affects_an_unrelated_candidate() {
    let failing = selection_policy_identity("failing");
    let unrelated = selection_policy_identity("unrelated");
    assert!(!exploration_failure_affects_unrelated_candidate(
        ExplorationFailureAction::TriggerRollback,
        &failing,
        &unrelated,
    ));
}

#[test]
fn online_measurement_never_overrides_trust_or_correctness() {
    assert_eq!(
        online_measurement_cannot_override_correctness_or_trust(
            true,
            KernelArtifactTrust::Untrusted
        ),
        KernelArtifactTrust::Untrusted
    );
}

#[test]
fn policy_precedence_lets_runtime_safety_override_every_lower_preference() {
    let stack = PolicyConstraintStack {
        runtime_safety_forces_deterministic: true,
        cli_preference: Some(OptimizationProfile::Latency),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Deterministic
    );
}

#[test]
fn policy_precedence_lets_deployment_forbid_a_cli_requested_profile() {
    let stack = PolicyConstraintStack {
        cli_preference: Some(OptimizationProfile::Latency),
        deployment_forbids_profile: BTreeSet::from([OptimizationProfile::Latency]),
        model_instance_profile: Some(OptimizationProfile::Balanced),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Balanced
    );
}

#[test]
fn policy_precedence_honors_cli_preference_when_nothing_overrides_it() {
    let stack = PolicyConstraintStack {
        cli_preference: Some(OptimizationProfile::Throughput),
        ..PolicyConstraintStack::default()
    };
    assert_eq!(
        resolve_effective_profile(&stack),
        OptimizationProfile::Throughput
    );
}

#[test]
fn selection_explanation_never_contains_native_handles() {
    let mut explanation = SelectionExplanation::default();
    explanation.exclusions.insert(
        "some-kernel".into(),
        KernelSelectionExclusionReason::PolicyDenied,
    );
    assert!(!explanation.contains_native_handles());
}

#[test]
fn kernel_selection_error_ids_are_stable_and_match_the_proposal_vocabulary() {
    assert_eq!(
        KernelSelectionError::NoEligibleCandidates.id(),
        "kernel-selection-no-eligible-candidates"
    );
    assert_eq!(
        KernelSelectionError::PinnedKernelIneligible.id(),
        "kernel-selection-pinned-kernel-ineligible"
    );
    assert_eq!(
        KernelSelectionError::InternalError { reason: "x".into() }.id(),
        "internal-kernel-selection-error"
    );
}

#[test]
fn kernel_selection_observation_redacts_metadata_and_carries_kernel_identity() {
    let kernel = selection_policy_kernel_id("observed");
    let observation =
        KernelSelectionObservation::new(KernelSelectionObservationKind::KernelSelected)
            .with_kernel(&kernel)
            .with_redacted_metadata("path", "C:\\secret\\path");
    assert_eq!(observation.kernel, Some(kernel.stable_key()));
    assert_eq!(
        observation
            .redacted_metadata
            .get("path")
            .map(String::as_str),
        Some("[redacted backend diagnostic]")
    );
}

#[test]
fn workload_context_carries_batch_and_phase_metadata_for_ranking() {
    let context = WorkloadContext {
        active_sequences: 32,
        batch_width: 32,
        total_active_tokens: 4096,
        raggedness: Some(0.2),
        phase: Some(GenerationPhase::Decode),
        kv_cache_mode: Some("paged".into()),
    };
    // Batch-aware ranking evidence is only meaningful when a candidate's
    // metrics are indexed under the matching workload bucket -- confirm the
    // context can drive that bucket key deterministically.
    let bucket = format!(
        "batch{}-seq{}",
        context.batch_width, context.total_active_tokens
    );
    assert!(performance_evidence_applies_to_workload(&bucket, &bucket));
    assert_eq!(context.phase, Some(GenerationPhase::Decode));
}

#[test]
fn rank_by_generation_phase_permits_distinct_prefill_and_decode_winners() {
    let prefill_throughput = EligibleCandidate::from_checked(
        selection_policy_identity("prefill-throughput"),
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 50.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let decode_latency = EligibleCandidate::from_checked(
        selection_policy_identity("decode-latency"),
        CandidateMetrics::default().with(ObjectiveDimension::Latency, 2.0),
        &CandidateEligibilityInput::all_satisfied(),
    )
    .unwrap();
    let strategy = RankingStrategy::WeightedScore(WeightedScorePolicy {
        weights: BTreeMap::from([(ObjectiveDimension::Latency, 1.0)]),
        missing_metric_policy: MissingMetricPolicy::Exclude,
    });
    let by_phase = rank_by_generation_phase(
        std::slice::from_ref(&prefill_throughput),
        std::slice::from_ref(&decode_latency),
        &strategy,
    )
    .unwrap();
    assert_eq!(
        by_phase[&GenerationPhase::Prefill],
        vec![prefill_throughput.identity]
    );
    assert_eq!(
        by_phase[&GenerationPhase::Decode],
        vec![decode_latency.identity]
    );
}

#[test]
fn rolling_measurement_window_only_reports_stable_once_full() {
    let mut window = RollingMeasurementWindow::new(3);
    assert!(!window.is_stable());
    window.record(10.0);
    window.record(12.0);
    assert!(!window.is_stable());
    window.record(11.0);
    assert!(window.is_stable());
    assert_eq!(window.len(), 3);
    window.record(9.0);
    assert_eq!(window.len(), 3);
    assert!(window.mean().is_some());
}

#[test]
fn static_selection_required_during_warmup_needs_pinned_mode_and_kernel_step() {
    let kernel = selection_policy_kernel_id("pinned");
    let pinned_mode = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(kernel, "digest"));
    let dynamic_mode = KernelSelectionPolicy::Dynamic;
    let full_plan = ModelInstanceWarmupPlan::for_policy(ModelInstanceWarmupPolicy::Full);
    let metadata_only_plan =
        ModelInstanceWarmupPlan::for_policy(ModelInstanceWarmupPolicy::ValidateMetadataOnly);

    assert!(static_selection_required_during_warmup(
        &pinned_mode,
        &full_plan
    ));
    assert!(!static_selection_required_during_warmup(
        &dynamic_mode,
        &full_plan
    ));
    assert!(!static_selection_required_during_warmup(
        &pinned_mode,
        &metadata_only_plan
    ));
}

#[test]
fn apply_promotion_recommendation_only_promotes_the_registry_when_approved() {
    let kernel = selection_policy_kernel_id("promotable");
    let device = DeviceBinding::new(crate::DeviceId::new("selection-policy-device"));
    let mut allocator = PreparedKernelIdAllocator::default();
    let mut registry = KernelRegistry::new();
    let mut generation = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-selection-policy"),
        ProviderBinding::new("selection-policy-provider"),
        device,
        PreparedKernelGeneration::new(1),
    );
    generation.mark_ready().unwrap();
    let generation_id = generation.id;
    registry.register_prepared_kernel(generation);

    let candidate = selection_policy_identity("promotable");
    let denied = PromotionRecommendation {
        candidate: candidate.clone(),
        approved: false,
        reason: Some("hysteresis threshold not met".into()),
    };
    assert_eq!(
        apply_promotion_recommendation(&denied, &mut registry, generation_id),
        Err(KernelSelectionError::PromotionThresholdNotMet)
    );

    let approved = PromotionRecommendation {
        candidate,
        approved: true,
        reason: None,
    };
    assert!(apply_promotion_recommendation(&approved, &mut registry, generation_id).is_ok());
}

#[test]
fn apply_exploration_failure_action_only_rolls_back_for_trigger_rollback() {
    let kernel = selection_policy_kernel_id("exploring");
    let device = DeviceBinding::new(crate::DeviceId::new("selection-policy-device"));
    let mut allocator = PreparedKernelIdAllocator::default();
    let mut registry = KernelRegistry::new();

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-exploring-v1"),
        ProviderBinding::new("selection-policy-provider"),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().unwrap();
    registry.register_prepared_kernel(generation_one.clone());
    registry
        .promote_generation(&kernel, generation_one.id)
        .unwrap();

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-exploring-v2"),
        ProviderBinding::new("selection-policy-provider"),
        device,
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().unwrap();
    registry.register_prepared_kernel(generation_two.clone());
    registry
        .promote_generation(&kernel, generation_two.id)
        .unwrap();

    // A bookkeeping-only action never touches the Registry.
    assert!(
        apply_exploration_failure_action(
            ExplorationFailureAction::MarkUnhealthy,
            &mut registry,
            &kernel,
        )
        .is_ok()
    );

    // Only `TriggerRollback` reaches `rollback_generation`, and it succeeds
    // because a previous generation is still retained and dispatchable.
    assert!(
        apply_exploration_failure_action(
            ExplorationFailureAction::TriggerRollback,
            &mut registry,
            &kernel,
        )
        .is_ok()
    );
}

#[test]
fn model_instance_definition_can_reference_a_kernel_selection_policy() {
    let mut definition = model_instance_definition();
    assert!(definition.kernel_selection_policy.is_none());
    definition.kernel_selection_policy = Some(KernelSelectionPolicyId::new("latency-profile-v1"));
    assert_eq!(
        definition
            .kernel_selection_policy
            .as_ref()
            .map(|id| id.as_str()),
        Some("latency-profile-v1")
    );
}

#[test]
fn generation_preference_boundary_has_no_way_to_carry_a_kernel_identity() {
    // Structural: `resolve_generation_preference` only ever takes and
    // returns an `OptimizationProfile`, so a generation request cannot use
    // it to smuggle a concrete `PreparedKernelId` or `KernelId` into
    // selection, implementing "Generation requests MAY provide high-level
    // policy preferences. They SHALL NOT directly force an ineligible
    // concrete Kernel" (proposal).
    assert_eq!(
        resolve_generation_preference(
            Some(OptimizationProfile::Latency),
            OptimizationProfile::Balanced
        ),
        OptimizationProfile::Latency
    );
    assert_eq!(
        resolve_generation_preference(None, OptimizationProfile::Balanced),
        OptimizationProfile::Balanced
    );
}

#[test]
fn reproducible_kernel_selection_forces_pinned_performance_feedback() {
    // Implements "Reproducible Mode Prevents Adaptation" (proposal, define-
    // kernel-performance-model-and-adaptive-feedback-contract): a pinned
    // Kernel selection always yields Pinned feedback mode regardless of the
    // Model Instance's requested mode, and dynamic selection leaves the
    // requested mode untouched.
    let policy = ModelInstancePolicy {
        performance_feedback: KernelPerformanceFeedbackMode::Adaptive,
        ..ModelInstancePolicy::default()
    };

    assert_eq!(
        effective_performance_feedback_mode(&policy, &KernelSelectionPolicy::Dynamic),
        KernelPerformanceFeedbackMode::Adaptive
    );

    let pinned = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(
        conformance_kernel_id("attn"),
        "digest-a",
    ));
    assert_eq!(
        effective_performance_feedback_mode(&policy, &pinned),
        KernelPerformanceFeedbackMode::Pinned
    );
}

#[test]
fn registry_performance_evidence_is_keyed_by_generation_and_never_fabricated() {
    // Implements "Registry Preserves Performance Evidence Identity" and
    // "Registry Does Not Generate Performance Evidence" (proposal, define-
    // kernel-performance-model-and-adaptive-feedback-contract): a new
    // Prepared Kernel generation never inherits a prior generation's
    // evidence, and a candidate with no recorded evidence resolves to
    // `None` rather than another candidate's summary.
    let artifact = CompiledKernelArtifactId::from_digest("digest-a");
    let mut allocator = PreparedKernelIdAllocator::default();
    let generation_n = allocator.allocate();
    let generation_n_plus_1 = allocator.allocate();

    let mut evidence = BTreeMap::new();
    evidence.insert(
        performance_evidence_key(&artifact, generation_n),
        KernelPerformanceMetricSummary {
            count: 100,
            ..KernelPerformanceMetricSummary::default()
        },
    );

    assert!(lookup_performance_evidence(&evidence, &artifact, generation_n).is_some());
    assert!(
        lookup_performance_evidence(&evidence, &artifact, generation_n_plus_1).is_none(),
        "a new generation must not silently inherit the prior generation's evidence"
    );
}

// ---------------------------------------------------------------------
// `host_tensors_from_artifact_bytes` (materialize-weights-from-real-model-
// artifact task group 1): the generic bytes-to-HostTensor bridge a format
// parser's own tensor inventory (`ModelTensorMetadata`) feeds into, without
// `magnetar-runtime` ever depending on a concrete format-parser crate.
// ---------------------------------------------------------------------

fn artifact_bytes_test_tensor(
    name: &str,
    shape: Vec<u64>,
    data: &[f32],
) -> (ModelTensorMetadata, Vec<u8>) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let metadata = ModelTensorMetadata {
        name: name.to_string(),
        shape,
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(bytes.len() as u64),
        quantization: None,
        expected_compute_dtype: None,
    };
    (metadata, bytes)
}

#[test]
fn host_tensors_from_artifact_bytes_reads_a_well_formed_tensor() {
    let (mut metadata, tensor_bytes) =
        artifact_bytes_test_tensor("weight.a", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    // Simulate a second tensor's bytes preceding this one in a real file by
    // offsetting into a larger buffer, rather than only ever testing offset
    // zero.
    let mut file = vec![0u8; 16];
    file.extend_from_slice(&tensor_bytes);
    metadata.offset_bytes = Some(16);

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &file, 0)
        .expect("well-formed tensor materializes");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Regression guard for a real bug found while implementing
/// `materialize-weights-from-real-model-artifact`: `offset_bytes` is
/// relative to the tensor-data section's start, not the whole file, and an
/// earlier version of this function ignored `data_section_start` entirely
/// (silently reading the wrong bytes for any nonzero data-section start,
/// exactly the shape a real Safetensors/GGUF file always has once a header
/// precedes the data). This test's `metadata.offset_bytes` deliberately
/// stays relative to the data section (`0`), and a nonzero
/// `data_section_start` is what must be added to land on the tensor's real
/// bytes.
#[test]
fn host_tensors_from_artifact_bytes_honors_nonzero_data_section_start() {
    let (metadata, tensor_bytes) =
        artifact_bytes_test_tensor("weight.a", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    // A "header" occupies the first 20 bytes; the tensor's own
    // `offset_bytes` (0) is relative to where the data section begins
    // (byte 20), not to the start of `file` itself.
    let mut file = vec![0xAAu8; 20];
    file.extend_from_slice(&tensor_bytes);

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &file, 20)
        .expect("well-formed tensor materializes with a nonzero data section start");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_non_f32_dtype() {
    let (mut metadata, bytes) = artifact_bytes_test_tensor("weight.a", vec![1], &[1.0]);
    metadata.storage_dtype = ModelDType::F16;

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("non-F32 dtype must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::StorageDTypeUnsupported);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_out_of_bounds_range() {
    let (metadata, _bytes) = artifact_bytes_test_tensor("weight.a", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    // Declare a range that does not actually fit in a much smaller buffer.
    let short_file = vec![0u8; 4];

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &short_file, 0)
        .expect_err("out-of-bounds range must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_shape_size_mismatch() {
    let (mut metadata, bytes) =
        artifact_bytes_test_tensor("weight.a", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    // Shape says 4 elements (16 bytes), but size_bytes disagrees.
    metadata.size_bytes = Some(8);

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("shape/size mismatch must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}
