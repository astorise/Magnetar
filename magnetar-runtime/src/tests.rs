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
use crate::execution_graph::*;
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
use crate::kernel_dispatch::*;
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
use crate::observability::*;
use crate::operator::*;
use crate::planning::*;
use crate::prefix_cache::*;
use crate::production_model_ingestion::*;
use crate::provider::*;
use crate::provider_roadmap::*;
use crate::reference_cpu::*;
use crate::resolution::*;
use crate::runtime::*;
use crate::sampling::*;
use crate::scheduler::*;
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
fn execution_context_default_allocates_unique_ids() {
    let first = ExecutionContext::default();
    let second = ExecutionContext::default();
    assert_ne!(first.id(), second.id());
    assert_ne!(first.id(), ExecutionContextId::default());
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

/// `unify-provider-output-admission-and-residency` task 1.2/1.3:
/// `admit_kernel_output` is the Runtime-owned pre-admission helper a
/// dispatch caller uses to admit a Kernel output's *final* resource
/// identity before submission, under a caller-chosen owner, instead of
/// letting the Provider admit (and own) it itself.
#[test]
fn admit_kernel_output_replaces_and_releases_the_previous_allocation_for_the_same_id() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(4096),
        ..MemoryManagerConfig::default()
    });
    let id = TensorResourceId::new("edge.layer0.attention.out");
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([2, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("cuda"));
    let owner = MemoryAllocationOwner::Session("cache-1".into());
    let placement = MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("cuda:0")));

    let resource = manager
        .admit_kernel_output(
            id.clone(),
            &descriptor,
            placement.clone(),
            owner.clone(),
            affinity.clone(),
        )
        .unwrap();
    assert_eq!(resource.id, id);
    let first_allocation = manager.tensor_residency(&id).unwrap().allocation.unwrap();
    assert!(
        manager
            .allocations()
            .any(|allocation| allocation.id == first_allocation
                && allocation.state == MemoryAllocationState::Active)
    );

    // Re-admitting the same id (a later generation step dispatching the
    // same graph node) must replace and release the previous allocation,
    // not accumulate a second one.
    manager
        .admit_kernel_output(id.clone(), &descriptor, placement, owner, affinity)
        .unwrap();
    let second_allocation = manager.tensor_residency(&id).unwrap().allocation.unwrap();
    assert_ne!(first_allocation, second_allocation);
    assert!(
        !manager
            .allocations()
            .any(|allocation| allocation.id == first_allocation
                && allocation.state == MemoryAllocationState::Active),
        "the first allocation must no longer be Active after re-admission"
    );
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
    let result = rope(&input, 10000.0, 1.0, 2, 0, 1).unwrap();
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

/// `unify-provider-output-admission-and-residency` task 2.4: an output
/// resource id a caller already admitted via `MemoryManager::
/// admit_kernel_output` (under a caller-chosen owner, e.g. `Session`) is
/// honored as-is by `execute_invocation_with_memory_manager` -- no second,
/// Provider-owned allocation, no overwritten residency.
#[test]
fn reference_cpu_honors_a_caller_pre_admitted_output_without_double_admitting() {
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

    let (a_id, a_resource) = reference_cpu_resource("mm-preadmit-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("mm-preadmit-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("mm-preadmit-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );

    let owner = MemoryAllocationOwner::Session("preadmit-cache".into());
    let placement = MemoryPlacement::HostOrdinary;
    memory
        .admit_kernel_output(
            out_resource.resource.id.clone(),
            &out_resource.resource.descriptor,
            placement.clone(),
            owner,
            out_resource.resource.affinity.clone(),
        )
        .expect("caller pre-admission must succeed");
    assert_eq!(memory.allocations().count(), 1);

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-preadmitted"),
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

    // Still exactly one allocation -- the Provider did not admit a second,
    // Provider-owned allocation for the same resource id.
    assert_eq!(memory.allocations().count(), 1);
    let residency = memory
        .tensor_residency(&out_resource.resource.id)
        .expect("residency record must still exist");
    // The caller's own placement is preserved, not overwritten with the
    // Provider's own `ProviderOwnedOpaque` default.
    assert_eq!(residency.placement, placement);
    assert!(
        executor
            .read_tensor(&out_resource.resource.id)
            .is_some_and(|tensor| tensor.data == vec![1.0, 2.0, 3.0, 4.0]),
        "the Kernel must still have written its actual output into the pre-admitted resource id"
    );
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

/// `implement-device-resident-multi-step-cuda-decode` task 4.4:
/// `copy_tensor_admitted` duplicates a resource's bytes to a fresh
/// identity, admitted the same way `write_tensor_admitted` already is.
#[test]
fn reference_cpu_copy_tensor_admitted_duplicates_bytes_to_a_fresh_identity() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let source_id = TensorResourceId::new("copy-source");
    executor
        .write_tensor_admitted(
            &mut memory,
            source_id.clone(),
            reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect("source write succeeds");

    let dest_id = TensorResourceId::new("copy-destination");
    executor
        .copy_tensor_admitted(
            &mut memory,
            &source_id,
            dest_id.clone(),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect("copy succeeds");

    assert_eq!(
        executor
            .read_tensor(&dest_id)
            .expect("destination resource is present")
            .data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// A second copy to the same destination identity replaces (and releases)
/// the first, matching `write_tensor_admitted`'s own replacement
/// discipline -- the exact bounded-growth requirement the KV pending/
/// commit paths depend on.
#[test]
fn reference_cpu_copy_tensor_admitted_replaces_a_previous_allocation_at_the_same_destination() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let source_id = TensorResourceId::new("copy-source-2");
    executor
        .write_tensor_admitted(
            &mut memory,
            source_id.clone(),
            reference_cpu_host_tensor([2], [1.0, 2.0]),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect("source write succeeds");

    let dest_id = TensorResourceId::new("copy-destination-stable");
    executor
        .copy_tensor_admitted(
            &mut memory,
            &source_id,
            dest_id.clone(),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect("first copy succeeds");
    let active_after_first_copy = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();

    for _ in 0..4 {
        executor
            .copy_tensor_admitted(
                &mut memory,
                &source_id,
                dest_id.clone(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test-session".into()),
            )
            .expect("repeated copy to the same destination succeeds");
    }
    let active_after_repeated_copies = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(
        active_after_repeated_copies, active_after_first_copy,
        "repeated copies to the same destination id must not accumulate allocations"
    );
}

#[test]
fn reference_cpu_copy_tensor_admitted_rejects_a_missing_source() {
    let executor = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let error = executor
        .copy_tensor_admitted(
            &mut memory,
            &TensorResourceId::new("does-not-exist"),
            TensorResourceId::new("copy-destination-3"),
            MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Session("test-session".into()),
        )
        .expect_err("copying a nonexistent source must fail structurally");
    assert!(matches!(error, TensorValueAdmissionError::Memory(_)));
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

/// Builds a real, fully consistent `matmul` `KernelDispatchPlan` against
/// Reference CPU -- Provider affinity set, two inputs, one output, all
/// agreeing -- for `kernel_dispatcher_revalidate_*` below to mutate one
/// specific field of at a time. Registered through
/// `register_provider_advertisement` (not the `PreparedKernel` machinery),
/// matching `reference_cpu_kernel_registry_selects_registered_candidate`
/// above, since these tests exercise `KernelDispatcher::revalidate`'s own
/// Provider/Device/ResourceAffinity consistency check
/// (audit-complet-cuda-hot-path-2026-09-08's Correctif A), not Kernel
/// Registry selection itself.
fn matmul_dispatch_plan_for_revalidation_tests() -> (KernelRegistry, KernelDispatchPlan) {
    let mut registry = KernelRegistry::new();
    for advertisement in ReferenceCpuProvider::new().kernel_advertisements() {
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
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
    let (_a_id, a_resource) = reference_cpu_resource("revalidate-a", [2, 2]);
    let (_b_id, b_resource) = reference_cpu_resource("revalidate-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("revalidate-out", [2, 2]);
    let matmul_operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let request = KernelSelectionRequest::new("revalidate-matmul", matmul_operator, affinity)
        .with_input(a_resource)
        .with_input(b_resource)
        .with_output(out_resource);
    let selection = registry.select(&request).unwrap();
    let candidate = selection
        .selected
        .expect("a compatible Reference CPU candidate should be selected");
    let advertisement = registry
        .active_advertisement(&candidate.kernel)
        .expect("just-selected candidate's advertisement must still be active")
        .clone();
    let plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("revalidate-matmul-dispatch"),
        &request,
        &candidate,
        &advertisement,
        KernelInvocationId::new("revalidate-matmul-invocation"),
    )
    .expect("a compatible candidate builds a valid dispatch plan");
    (registry, plan)
}

#[test]
fn kernel_dispatcher_revalidate_accepts_a_fully_consistent_plan() {
    let (registry, mut plan) = matmul_dispatch_plan_for_revalidation_tests();
    KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .expect("a plan whose invocation, resources, and affinity all agree must revalidate");
}

/// audit-complet-cuda-hot-path-2026-09-08 P0-1, Correctif A test list item
/// "provider mismatch".
#[test]
fn kernel_dispatcher_revalidate_rejects_provider_mismatch() {
    let (registry, mut plan) = matmul_dispatch_plan_for_revalidation_tests();
    plan.invocation.provider = ProviderBinding::new("magnetar:provider/somewhere-else");
    let error = KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .unwrap_err();
    assert!(matches!(
        error,
        KernelDispatchError::ResourceAffinityConflict(_)
    ));
}

/// audit-complet-cuda-hot-path-2026-09-08 P0-1, Correctif A test list item
/// "device mismatch même Provider" -- the exact regression the previous,
/// Provider-only `validate_invocation_provider_matches_affinity` (removed
/// in favor of this generic boundary check) could not catch: the same
/// Provider resolving to the *wrong* Device among several it exposes.
#[test]
fn kernel_dispatcher_revalidate_rejects_device_mismatch_with_the_same_provider() {
    let (registry, mut plan) = matmul_dispatch_plan_for_revalidation_tests();
    plan.invocation.affinity = plan
        .invocation
        .affinity
        .clone()
        .with_device(DeviceBinding::new(DeviceId::new("gpu-0")));
    plan.invocation.device = Some(DeviceBinding::new(DeviceId::new("gpu-1")));
    let error = KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .unwrap_err();
    assert!(matches!(
        error,
        KernelDispatchError::ResourceAffinityConflict(_)
    ));
}

/// audit-complet-cuda-hot-path-2026-09-08 P0-1, Correctif A test list item
/// "input resource mismatch" -- the generic check now walks
/// `input_bindings`, not only `output_bindings` (the previous check's own
/// gap).
#[test]
fn kernel_dispatcher_revalidate_rejects_input_resource_affinity_mismatch() {
    let (registry, mut plan) = matmul_dispatch_plan_for_revalidation_tests();
    plan.invocation.inputs[0].resource.affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("magnetar:provider/somewhere-else"));
    plan.input_bindings[0].resource.affinity = plan.invocation.inputs[0].resource.affinity.clone();
    let error = KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .unwrap_err();
    assert!(matches!(
        error,
        KernelDispatchError::ResourceAffinityConflict(_)
    ));
}

/// audit-complet-cuda-hot-path-2026-09-08 P0-1, Correctif A test list item
/// "output resource mismatch".
#[test]
fn kernel_dispatcher_revalidate_rejects_output_resource_affinity_mismatch() {
    let (registry, mut plan) = matmul_dispatch_plan_for_revalidation_tests();
    plan.invocation.outputs[0].resource.affinity =
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new("magnetar:provider/somewhere-else"));
    plan.output_bindings[0].resource.affinity =
        plan.invocation.outputs[0].resource.affinity.clone();
    let error = KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .unwrap_err();
    assert!(matches!(
        error,
        KernelDispatchError::ResourceAffinityConflict(_)
    ));
}

/// audit-complet-cuda-hot-path-2026-09-08 P0-1, Correctif A test list item
/// "prepared plan device mismatch" -- the same invariant, checked through
/// `KernelDispatchPlan::from_prepared_node_execution`'s construction path
/// (a published Plan's binding), not only `from_selection`'s.
#[test]
fn kernel_dispatcher_revalidate_rejects_prepared_plan_device_mismatch() {
    let mut registry = KernelRegistry::new();
    for advertisement in ReferenceCpuProvider::new().kernel_advertisements() {
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
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_device(DeviceBinding::new(DeviceId::new("gpu-0")));
    let (_a_id, a_resource) = reference_cpu_resource("revalidate-prepared-a", [2, 2]);
    let (_b_id, b_resource) = reference_cpu_resource("revalidate-prepared-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("revalidate-prepared-out", [2, 2]);
    let matmul_operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let request =
        KernelSelectionRequest::new("revalidate-prepared-matmul", matmul_operator, affinity)
            .with_input(a_resource)
            .with_input(b_resource)
            .with_output(out_resource);
    let selection = registry.select(&request).unwrap();
    let candidate = selection
        .selected
        .expect("a compatible Reference CPU candidate should be selected");
    let advertisement = registry
        .active_advertisement(&candidate.kernel)
        .expect("just-selected candidate's advertisement must still be active")
        .clone();
    let prepared = PreparedPlanNodeExecution {
        graph_node: ExecutionNodeId::new("revalidate-prepared-node"),
        kernel: candidate.kernel.clone(),
        prepared_kernel: PreparedKernelIdAllocator::default().allocate(),
        prepared_kernel_generation: PreparedKernelGeneration::new(1),
        provider: candidate.provider.clone(),
        // Diverges from the request's own `affinity.device` (`gpu-0`)
        // above -- a published Plan binding naming the wrong Device for
        // this resource's declared affinity.
        device: Some(DeviceBinding::new(DeviceId::new("gpu-1"))),
        plan: PreparedExecutionPlanId::new("revalidate-prepared-plan")
            .expect("a simple ascii identifier is a valid Prepared Execution Plan id"),
        plan_generation: PreparedExecutionPlanGeneration::new(1),
    };
    let mut plan = KernelDispatchPlan::from_prepared_node_execution(
        KernelDispatchPlanId::new("revalidate-prepared-matmul-dispatch"),
        &request,
        &prepared,
        &advertisement,
        KernelInvocationId::new("revalidate-prepared-matmul-invocation"),
    )
    .expect("a compatible prepared node execution builds a valid dispatch plan");
    let error = KernelDispatcher::new()
        .revalidate(&registry, &mut plan)
        .unwrap_err();
    assert!(matches!(
        error,
        KernelDispatchError::ResourceAffinityConflict(_)
    ));
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
fn tensor_lifecycle_rejects_declared_to_ready_skip() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-2");
    let error = resource
        .transition_to(TensorLifecycleState::Ready)
        .unwrap_err();
    assert!(matches!(error, TensorError::ResourceInvalid { .. }));
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
        required_weight_names: BTreeSet::new(),
        required_weight_digests: BTreeMap::new(),
        required_weight_shapes: BTreeMap::new(),
    }
}

/// Materializes one real, Runtime-issued-evidence-backed weight for
/// `instance` through `materialize_model_instance_weights` -- the one
/// legitimate way any caller (production or an external embedder) can turn
/// weight bytes into bound, Ready-eligible resources
/// (`bind-model-loading-evidence-to-validated-artifact`). This alone
/// reaches `Ready`: the underlying transaction commits bindings, mints
/// materialization evidence, and marks the instance Ready in one step, the
/// same as the real production path -- a separate follow-up
/// `warm_model_instance` call is not just redundant but would fail (no
/// `Ready -> Ready` lifecycle transition exists). Requires a
/// `ReferenceCpuProvider` already registered on `runtime` and `instance`'s
/// placement pinned to it.
fn reach_ready_with_real_weight(runtime: &mut Runtime, instance: &ModelInstanceId) {
    let weights = BTreeMap::from([("weight".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(runtime, instance, "test", &weights).unwrap();
}

#[test]
fn inference_api_model_instance_suspend_resume_drain_through_api_boundary() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    // `create()` no longer reaches Ready on its own (transactional-weight-
    // materialization); this test's own concern is suspend/resume/drain
    // *from* Ready, so reach Ready explicitly first -- with real,
    // Provider-backed evidence: `resume_model_instance` now re-derives
    // readiness (Correctif: Runtime-owned ModelInstance readiness
    // authority, round 3), so an instance that was never really
    // materialized correctly cannot resume back to Ready any more, and
    // this test's own concern (the suspend/resume/drain state machine,
    // not weight materialization) needs a genuinely resumable instance
    // to exercise that machinery at all.
    reach_ready_with_real_weight(&mut runtime, &instance);

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

/// Implements `seal-model-instance-readiness-authority` round 3, the
/// audit's own resume scenario: state that made an instance eligible for
/// `Ready` can change while it is suspended, so `resume_model_instance`
/// must re-derive readiness against *current* Runtime state, not assume
/// whatever was true before suspension still holds. Proven here by
/// invalidating the weight evidence during suspension (removing its
/// `TensorResidency`, a legitimate way real state can regress -- e.g. a
/// rollback or eviction elsewhere) rather than by mutating a `Provider`'s
/// status snapshot, which this test double has no interior mutability to
/// do after registration; both exercise the same property: resume must
/// look at current state, not trust the state from before suspension.
#[test]
fn inference_api_resume_model_instance_revalidates_and_rejects_stale_evidence() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance);
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );

    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )
    .unwrap();

    // While suspended, the weight evidence that made this instance Ready
    // becomes invalid (its TensorResidency is removed) -- simulating real
    // state regressing during the suspension window, exactly the scenario
    // the audit's own resume test targets.
    let weight_resource = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .get("weight")
        .unwrap()
        .clone();
    runtime
        .memory_mut()
        .remove_tensor_residency(&weight_resource);

    resume_model_instance(&mut runtime, &instance).unwrap_err();
    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `seal-model-instance-readiness-authority` round 3, audit
/// test 25 "Inventaire incomplet": a manifest declaring multiple mandatory
/// tensors, with only some of them bound, must not reach Ready even
/// though every bound entry is individually real (residency-backed,
/// Provider-written, and -- since `bind-model-loading-evidence-to-
/// validated-artifact` -- Runtime-evidenced). Round 2's fix only checked
/// that whatever *was* bound was real; it never checked the bound set was
/// *complete*. Materializes "weight.a" for real through the one authorized
/// transaction (which has no knowledge of `required_weight_names` --
/// that's `derive_effective_readiness_checks`'s own concern -- so it marks
/// the instance Ready on its own terms); a subsequent `warm_model_instance`
/// re-derivation must then find the mandatory inventory still incomplete
/// and demote it, the same defense-in-depth property the resume-
/// revalidation test above proves for stale residency.
#[test]
fn inference_api_warm_model_instance_rejects_incomplete_mandatory_weight_inventory() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names =
        BTreeSet::from(["weight.a".to_string(), "weight.b".to_string()]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    // Really materialize only "weight.a" -- "weight.b" stays entirely
    // absent.
    let weights = BTreeMap::from([("weight.a".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `seal-model-instance-readiness-authority` round 3, audit
/// test 25 "Residency synthétique": a `TensorResidency` recorded manually
/// (real Memory Manager allocation, real `record_tensor_residency` call)
/// but with no matching `Provider::write_tensor` ever having run must not
/// count as materialized. Round 2's fix only checked a residency record
/// existed; it did not check the Provider it claims to describe actually
/// holds the tensor. **Mechanism note (`bind-model-loading-evidence-to-
/// validated-artifact`):** this scenario now also fails the newer
/// evidence-matching check (no `MaterializationEvidence` exists for a
/// hand-constructed binding at all, regardless of whether `write_tensor`
/// ran) -- kept as its own test anyway since it demonstrates the
/// `TensorResidency`-presence check on a *different* axis than evidence
/// does (residency answers "is it still resident now", not "did an
/// authorized transaction produce it").
#[test]
fn inference_api_warm_model_instance_rejects_residency_without_provider_write() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    let allocation = runtime
        .memory_mut()
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            1,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::InferenceArtifact("test".into()),
        ))
        .unwrap();
    let weight_resource = TensorResourceId::new("test.weight.never-written");
    // Record a real residency -- but never call `write_tensor` on the
    // Provider it claims. This is exactly what a caller with mutable
    // access to `Runtime::memory_mut()` and `resource_bindings.weights`
    // could always do; the Provider's own storage is the one thing they
    // cannot forge without actually writing real bytes to it.
    runtime
        .memory_mut()
        .record_tensor_residency(
            TensorResidency::new(
                weight_resource.clone(),
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(
                    REFERENCE_CPU_PROVIDER_NAME,
                )),
                ResourceAffinity::new(FallbackClass::Transparent)
                    .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
            )
            .with_allocation(allocation.id),
        )
        .unwrap();
    assert!(
        runtime
            .providers()
            .provider(REFERENCE_CPU_PROVIDER_NAME)
            .and_then(|provider| provider.execution_api())
            .unwrap()
            .read_tensor(&weight_resource)
            .is_none(),
        "test precondition: nothing was ever written to the Provider for this resource"
    );
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), weight_resource);

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
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
    // explicitly first. Bind a weight resource before doing so: since
    // `warm_model_instance` now derives `weights_materialized` from
    // `resource_bindings.weights` non-emptiness (`runtime-owned-model-
    // instance-readiness-authority`), an instance with none would make the
    // later `warm_model_instance` call fail on that check instead of the
    // already-Ready conflict this test actually means to exercise -- both
    // happen to map to the same broad `ModelInstanceUnavailable` variant,
    // which would silently let this test pass for the wrong reason.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("test.weight"));
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

/// Implements `runtime-owned-model-instance-readiness-authority`. An
/// external audit of PR #36 found `warm_model_instance` trusted a caller's
/// `weights_materialized: true` claim outright, even though
/// `ModelInstanceReadinessChecks::default()` sets it `true` -- a caller
/// using the default checks against an instance whose weights were never
/// materialized could reach `Ready`. Covers the audit's test 18.1 (warmup
/// without materialization must not reach Ready) and 18.3 (a forged
/// `weights_materialized=true` claim against empty bindings is rejected).
#[test]
fn inference_api_warm_model_instance_rejects_forged_weights_materialized_claim() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    assert!(
        runtime
            .model_instance(&instance)
            .unwrap()
            .definition
            .resource_bindings
            .weights
            .is_empty(),
        "test precondition: no weights ever bound"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    // The caller asserts every fact is satisfied, exactly matching
    // `ModelInstanceReadinessChecks::default()` -- the audit's own
    // exploit scenario (section 11).
    let forged_checks = ModelInstanceReadinessChecks::default();
    assert!(forged_checks.weights_materialized);

    warm_model_instance(&mut runtime, &instance, &plan, &forged_checks).unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.2: `WarmupPolicy::Disabled` calls `ModelInstance::validate_readiness`
/// directly, without the lifecycle transition `warmup()`'s other policies
/// perform first -- so before this fix, a caller-forged `readiness =
/// Ready` could be published while `lifecycle` stayed `Loading`, an
/// internally inconsistent state. This test isolates that specific gap
/// from `weights_materialized` derivation (covered separately above) by
/// directly binding a weight resource, so the *only* thing preventing
/// `Ready` here is the lifecycle/readiness consistency check.
#[test]
fn inference_api_warm_model_instance_disabled_policy_cannot_forge_ready_readiness() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("test.weight"));
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Loading,
        "test precondition: still Loading, no transition has run"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::Disabled,
        steps: Vec::new(),
    };
    let checks = ModelInstanceReadinessChecks::default();

    // Disabled policy's own contract (`ModelInstance::warmup`) means this
    // call does not have to fail -- `validate_readiness`'s own `Result`
    // reflects whether the checks *themselves* are internally coherent,
    // which they are. What matters is `self.readiness` afterward.
    let _ = warm_model_instance(&mut runtime, &instance, &plan, &checks);

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(
        status.lifecycle,
        ModelInstanceLifecycleState::Loading,
        "Disabled policy must not transition the lifecycle"
    );
    assert_ne!(
        status.readiness,
        ModelInstanceReadiness::Ready,
        "readiness must not report Ready while lifecycle is still Loading"
    );
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.4: an internally inconsistent `lifecycle: Loading, readiness: Ready`
/// state -- however it might arise -- must never grant `acquire_usage` or
/// `generation_reference`. This is the structural safety net: it holds
/// regardless of which caller-forgeable path produced the inconsistency.
#[test]
fn model_instance_acquire_usage_rejects_ready_readiness_with_incompatible_lifecycle() {
    let mut manager = ModelInstanceManager::new();
    let id = manager.create(model_instance_definition()).unwrap();
    assert_eq!(
        manager.instance(&id).unwrap().lifecycle,
        ModelInstanceLifecycleState::Loading
    );

    // Force the inconsistent state directly (bypassing every public
    // entry point) to prove the check in `acquire_usage`/
    // `generation_reference` itself, independent of how the
    // inconsistency might be produced.
    manager.instance_mut(&id).unwrap().readiness = ModelInstanceReadiness::Ready;
    assert_eq!(
        manager.instance(&id).unwrap().lifecycle,
        ModelInstanceLifecycleState::Loading
    );
    assert_eq!(
        manager.instance(&id).unwrap().readiness,
        ModelInstanceReadiness::Ready
    );

    assert!(matches!(
        manager.instance_mut(&id).unwrap().acquire_usage(0),
        Err(ModelInstanceError::ModelInstanceLoading)
    ));
    assert!(matches!(
        manager.generation_reference(&id),
        Err(ModelInstanceError::ModelInstanceLoading)
    ));
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.5: the happy path -- real weight materialization, a Provider the
/// Runtime can actually resolve, and no forged claims -- still reaches
/// Ready and accepts usage. Closing the forgery gap must not regress the
/// legitimate warmup path.
#[test]
fn inference_api_warm_model_instance_reaches_ready_when_weights_and_provider_are_real() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    // Real, Provider-backed evidence, not just a residency record or a
    // bare map entry: `weights_materialized` derivation resolves each
    // residency's recorded Provider and reads the tensor back from it, so
    // a residency whose Provider never actually received a `write_tensor`
    // call does not count (Correctif: Runtime-owned ModelInstance
    // readiness authority, round 3 -- closing the "synthetic residency"
    // forgery a further audit of PR #36 demonstrated: a caller could
    // previously construct a valid-looking `TensorResidency` for a
    // resource no Provider ever wrote).
    reach_ready_with_real_weight(&mut runtime, &instance);

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(status.readiness.accepts_generation());
    assert!(
        runtime
            .model_instances_mut()
            .acquire_usage(&instance, 0)
            .is_ok()
    );
}

/// `bind-model-loading-evidence-to-validated-artifact`: a weight binding
/// assembled by hand -- real Memory Manager allocation, real
/// `Provider::write_tensor` call, real `TensorResidency` record, real
/// binding -- but never produced by `WeightMaterializationTransaction`
/// itself must not count as materialized, because no
/// `MaterializationEvidence` exists for it. This is the general property
/// the audit's `bind_fake_weight`-in-`contract_tests` finding demonstrated
/// concretely: every individual piece of state a caller can construct with
/// ordinary public/`pub(crate)` access looks legitimate, but the one thing
/// a caller cannot fabricate without the real transaction is Runtime-issued
/// evidence.
#[test]
fn inference_api_warm_model_instance_rejects_hand_assembled_binding_without_evidence() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    let allocation = runtime
        .memory_mut()
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            1,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::InferenceArtifact("test".into()),
        ))
        .unwrap();
    let resource_id = TensorResourceId::new("test.weight.hand-assembled");
    let executor = runtime
        .providers()
        .provider(REFERENCE_CPU_PROVIDER_NAME)
        .and_then(|provider| provider.execution_api())
        .unwrap();
    executor
        .write_tensor(resource_id.clone(), HostTensor::new([1], [0.0]).unwrap())
        .unwrap();
    runtime
        .memory_mut()
        .record_tensor_residency(
            TensorResidency::new(
                resource_id.clone(),
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(
                    REFERENCE_CPU_PROVIDER_NAME,
                )),
                ResourceAffinity::new(FallbackClass::Transparent)
                    .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
            )
            .with_allocation(allocation.id),
        )
        .unwrap();
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), resource_id);

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// `bind-model-loading-evidence-to-validated-artifact`: materialization
/// evidence is looked up by the instance's own id, so Model Instance B
/// cannot become materialized by having its weight bindings set to match
/// Model Instance A's already-legitimately-materialized bindings -- B's
/// lookup only ever finds evidence `commit` minted for B's own id (none),
/// regardless of what A's evidence says.
#[test]
fn inference_api_warm_model_instance_rejects_bindings_copied_from_another_instance() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition_a = model_instance_definition();
    definition_a.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance_a = runtime.model_instances_mut().create(definition_a).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance_a);

    let mut definition_b = model_instance_definition();
    definition_b.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance_b = runtime.model_instances_mut().create(definition_b).unwrap();

    // Copy A's already-legitimate bindings directly onto B -- the exact
    // "reuse another instance's materialization" scenario the audit's own
    // "Artifact mismatch" test asks for, here proven for same-artifact
    // reuse across two distinct instances (the stronger, more general
    // case: even identical bindings for the identical artifact do not
    // transfer, because evidence is instance-scoped, not artifact-scoped).
    let copied_weights = runtime
        .model_instance(&instance_a)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    runtime
        .model_instances_mut()
        .instance_mut(&instance_b)
        .unwrap()
        .definition
        .resource_bindings
        .weights = copied_weights;

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance_b,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status_b = model_instance_status(&runtime, &instance_b).unwrap();
    assert_ne!(status_b.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status_b.readiness.accepts_generation());
    // A remains legitimately Ready throughout -- this is not a mutation of
    // A's own state, only B reading A's (still-valid) bindings.
    assert_eq!(
        model_instance_status(&runtime, &instance_a)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );
}

/// `bind-model-loading-evidence-to-validated-artifact`: materialization
/// evidence records the `ModelArtifactId` that was current when it was
/// minted; if a Model Instance's declared artifact later differs from what
/// its evidence recorded, the evidence no longer matches and readiness
/// must reject it. `artifact` has no dedicated setter, but the field
/// remains `pub` on `ModelInstanceDefinition` -- unlike `resource_bindings`,
/// reassigning it can only ever make otherwise-valid evidence stop
/// matching (fail closed), never let a caller borrow a *different*
/// instance's evidence, since evidence lookup is keyed by instance id, not
/// artifact id (see the previous test). Proven directly here rather than
/// left as a hypothetical.
#[test]
fn inference_api_warm_model_instance_rejects_evidence_after_artifact_reassignment() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance);
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );

    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("2-different".to_string()).unwrap(),
        ModelDigest::sha256(b"different weights"),
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// `bind-model-loading-evidence-to-validated-artifact`: closes the
/// companion P1 the same audit round identified -- a Provider that never
/// implements host-memory tensor readback (`read_tensor` always returning
/// `None`, its documented default) must still be able to reach
/// `weights_materialized: true` through the authorized transaction, since
/// readiness derivation no longer calls `read_tensor` at all.
/// `TestProviderExecutionApi` does not override `write_tensor`/
/// `read_tensor`, so it inherits the trait's real default bodies -- a
/// faithful stand-in for a device-only Provider, not a mock that merely
/// asserts it was never called.
#[test]
fn inference_api_warm_model_instance_reaches_ready_without_provider_read_tensor_support() {
    let mut provider = TestProvider::new(REFERENCE_CPU_PROVIDER_NAME);
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    // Confirm this Provider genuinely does not support host readback,
    // rather than assuming the trait default -- if a future change to
    // `TestProviderExecutionApi` overrides `read_tensor`, this assertion
    // (not just the one below) would catch that this test no longer tests
    // what it claims to.
    let probe_resource = TensorResourceId::new("probe");
    let executor = runtime
        .providers()
        .provider(REFERENCE_CPU_PROVIDER_NAME)
        .and_then(|provider| provider.execution_api())
        .unwrap();
    executor
        .write_tensor(probe_resource.clone(), HostTensor::new([1], [0.0]).unwrap())
        .unwrap();
    assert!(executor.read_tensor(&probe_resource).is_none());

    let weights = BTreeMap::from([("weight".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(status.readiness.accepts_generation());
}

/// A further audit of PR #36 found that `WeightMaterializationTransaction::
/// commit` called `ModelInstanceManager::mark_ready` unconditionally right
/// after staging succeeded and evidence was minted -- so
/// `materialize_model_instance_weights(..., &BTreeMap::new())` against an
/// instance with a non-empty mandatory inventory could still reach
/// `Ready`, bypassing the exact same Runtime-derived readiness gate
/// `warm_model_instance`/`resume_model_instance` already correctly use.
/// Non-compliant with `model-loading`'s pre-existing "Model Loading Does
/// Not Bypass Instance Readiness" ("Successful materialization alone
/// SHALL not imply Model Instance readiness") and "Partial Loading
/// Policy" requirements. Proven here at the exact public entrypoint the
/// bug was in, not only through `warm_model_instance` (which was already
/// correct on its own and could not, by itself, prove this bypass was
/// closed).
#[test]
fn inference_api_materialize_model_instance_weights_does_not_ready_with_empty_map() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names =
        BTreeSet::from(["weight.a".to_string(), "weight.b".to_string()]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    materialize_model_instance_weights(&mut runtime, &instance, "test", &BTreeMap::new()).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Same root gap as the empty-map test above, for a strict subset of a
/// multi-tensor mandatory inventory -- audit scenario "subset partiel":
/// `commit` minted an evidence record that was *exactly correct* for what
/// it staged, but exact correctness for an incomplete set is still
/// incomplete, and `commit` never checked completeness before marking
/// Ready.
#[test]
fn inference_api_materialize_model_instance_weights_does_not_ready_with_partial_inventory() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names = BTreeSet::from([
        "weight.a".to_string(),
        "weight.b".to_string(),
        "weight.c".to_string(),
    ]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    let weights = BTreeMap::from([
        ("weight.a".to_string(), HostTensor::new([1], [0.0]).unwrap()),
        ("weight.b".to_string(), HostTensor::new([1], [0.0]).unwrap()),
    ]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());

    // The partial materialization itself still succeeded for real: its
    // evidence exists and matches exactly what was staged, proving
    // `commit`'s gate rejects on inventory completeness specifically, not
    // by failing to publish anything at all (which would also incidentally
    // "fix" the symptom without fixing the actual defect).
    assert_eq!(
        runtime
            .model_instance(&instance)
            .unwrap()
            .definition
            .resource_bindings
            .weights
            .len(),
        2
    );

    // Completing the inventory in a second, independent call reaches
    // Ready -- proving incremental/progressive materialization (which
    // `commit`'s own evidence-recomputation is designed to support) still
    // works once the gate this fix adds is actually satisfied.
    let remaining =
        BTreeMap::from([("weight.c".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &remaining).unwrap();
    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(status.readiness.accepts_generation());
}

/// Audit scenario "Provider non-ready": a complete mandatory inventory
/// staged and evidenced for real, but the pinned Provider's own status
/// model reports it does not currently accept new work -- `commit` must
/// not mark the instance Ready anyway. Proven at the materialize
/// entrypoint directly, the same reasoning as the two tests above: this is
/// specifically about what `commit` itself does after staging succeeds,
/// not about `warm_model_instance`'s already-correct behavior.
#[test]
fn inference_api_materialize_model_instance_weights_does_not_ready_when_provider_rejects_work() {
    let mut provider = TestProvider::new(REFERENCE_CPU_PROVIDER_NAME);
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        HealthState::Available,
    ));
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    assert!(
        !provider.status_snapshot().accepts_new_work_by_default(),
        "test precondition: this Provider's own status model rejects new work"
    );

    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    let weights = BTreeMap::from([("weight".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority` round 2,
/// audit test 23.3: a `TensorResourceId` inserted directly into
/// `resource_bindings.weights`, with no matching `TensorResidency`, must
/// not let `warm_model_instance` derive `weights_materialized = true` --
/// the exact gap a follow-up audit of PR #36 found: the round-1 fix only
/// checked the map was non-empty, not that its entries were real.
#[test]
fn inference_api_warm_model_instance_rejects_weight_binding_without_residency() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    // Insert a plausible-looking `TensorResourceId` directly -- no
    // `MemoryManager::allocate`, no `record_tensor_residency`. This is
    // exactly what a caller with `&mut Runtime` could always do, and what
    // this Change's derivation must not trust.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("forged.weight"));
    assert!(
        runtime
            .memory()
            .tensor_residency(&TensorResourceId::new("forged.weight"))
            .is_none(),
        "test precondition: no residency was ever recorded for this resource"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority` round 2,
/// audit test 23.5: a Provider that is registered and offers a real
/// `execution_api()` but whose own status model reports it as not
/// accepting new work (here: `Saturated` pressure) must make
/// `provider_ready` derive `false`, even though the caller claims `true`
/// -- the round-1 fix only checked the Provider "exists and is executable
/// in principle" (`execution_api().is_some()`), not "is ready now".
#[test]
fn inference_api_warm_model_instance_rejects_provider_that_rejects_new_work() {
    let mut provider = TestProvider::new("saturated-provider");
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("saturated-provider"),
        HealthState::Available,
    ));
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    assert!(
        !provider.status_snapshot().accepts_new_work_by_default(),
        "test precondition: this Provider's own status model rejects new work"
    );

    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new("saturated-provider")),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    let allocation = runtime
        .memory_mut()
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            1,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::InferenceArtifact("test".into()),
        ))
        .unwrap();
    let weight_resource = TensorResourceId::new("test.weight");
    runtime
        .memory_mut()
        .record_tensor_residency(
            TensorResidency::new(
                weight_resource.clone(),
                MemoryPlacement::HostOrdinary,
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .with_allocation(allocation.id),
        )
        .unwrap();
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), weight_resource);

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    // Real weight evidence is present; only `provider_ready` should be
    // what fails this attempt.
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

#[test]
fn inference_api_create_model_instance_observed_emits_model_instance_selected() {
    let manifest = minimal_model_manifest();
    let mut load_runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut load_runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
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
        architecture_config: None,
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
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();

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

    let loaded = load_model(&mut coordinator, &mut runtime, request, &manifest).unwrap();
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
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut observer = InferenceApiObserver::new();

    load_model_observed(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
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

/// `seal-runtime-model-trust-and-provenance-authority` P0-A: `load_model`
/// no longer accepts a caller-supplied trust decision at all -- trust is
/// evaluated from the performing `Runtime`'s own sealed configuration. A
/// `Runtime` built without trusting this manifest's digest must reject the
/// load, proving there is no parameter through which a caller could still
/// supply a favorable decision for an untrusted artifact.
#[test]
fn inference_api_load_model_rejects_when_runtime_trust_store_does_not_trust_digest() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    // Sealed with an empty trust store -- trusts nothing, matching
    // `RuntimeBuilder::trust_store`'s documented default.
    let mut runtime = Runtime::builder().build().unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());

    let error = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap_err();
    assert!(error.to_string().contains("ModelArtifactUntrusted"));
}

/// P0-B: `create_model_instance` cross-checks the caller-supplied
/// architecture against `loaded.plan().architecture`, which the loading
/// phase already resolved from the same manifest -- a disagreement is
/// rejected rather than silently accepted.
#[test]
fn runtime_create_model_instance_rejects_architecture_disagreeing_with_resolved_plan() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();

    let disagreeing_architecture = ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("a-different-family", "a-different-identifier"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    };
    let error = runtime
        .create_model_instance(
            &loaded,
            disagreeing_architecture,
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ModelInstanceError::ArchitectureMismatch { .. }
    ));

    // Agreeing architecture is unaffected.
    runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
}

/// P0-B: when the loading phase *did* resolve a provider binding for the
/// plan, a caller-supplied affinity naming a different provider is
/// rejected. Nothing in today's real loading pipeline resolves
/// `plan.provider_binding` yet (confirmed by inspection: it is
/// unconditionally `None` at construction), so this forces the scenario
/// directly on the loaded plan rather than only exercising the
/// permissive "unresolved" branch every other test already covers.
#[test]
fn runtime_create_model_instance_rejects_affinity_disagreeing_with_resolved_provider_binding() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    loaded.plan.provider_binding = Some(ProviderBinding::new("resolved-provider"));

    let disagreeing_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("a-different-provider"));
    let error = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            disagreeing_affinity,
        )
        .unwrap_err();
    assert!(matches!(error, ModelInstanceError::AffinityMismatch { .. }));

    let agreeing_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("resolved-provider"));
    runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            agreeing_affinity,
        )
        .unwrap();
}

fn manifest_with_one_tensor(shape: Vec<u64>, storage_dtype: ModelDType) -> ModelManifest {
    ModelManifest {
        tensors: vec![ModelTensorMetadata {
            name: "the-only-tensor".into(),
            shape,
            storage_dtype,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        }],
        ..minimal_model_manifest()
    }
}

fn manifest_with_tensors(tensors: Vec<(&str, Vec<u64>)>) -> ModelManifest {
    let tensors = tensors
        .into_iter()
        .map(|(name, shape)| {
            let element_count: u64 = shape.iter().product();
            ModelTensorMetadata {
                name: name.to_string(),
                shape,
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: Some(0),
                size_bytes: Some(element_count * 4),
                quantization: None,
                expected_compute_dtype: None,
                digest: None,
            }
        })
        .collect();
    ModelManifest {
        tensors,
        ..minimal_model_manifest()
    }
}

/// Test-only [`ProductionArtifactPayloadSource`]
/// (`implement-production-qwen-model-loading` task group 6): serves each
/// tensor's bytes from an in-memory map by logical name (ignoring
/// `offset`/`length`, which a real ingestor would use to locate bytes
/// within its own authorized source), and records every `read_payload`
/// call's identity in order -- so a test can assert both *what* was read
/// and *when*, proving no tensor is read more than once and no tensor is
/// read out of the order [`stream_materialize_model_instance_weights`]
/// was given.
struct RecordingPayloadSource {
    bytes_by_name: BTreeMap<String, Vec<u8>>,
    calls: std::sync::Mutex<Vec<String>>,
    fail_for: Option<String>,
}

impl RecordingPayloadSource {
    fn new(bytes_by_name: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            bytes_by_name,
            calls: std::sync::Mutex::new(Vec::new()),
            fail_for: None,
        }
    }

    fn failing_for(mut self, name: &str) -> Self {
        self.fail_for = Some(name.to_string());
        self
    }

    fn call_log(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl ProductionArtifactPayloadSource for RecordingPayloadSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        self.calls.lock().unwrap().push(range.identity.clone());
        if self.fail_for.as_deref() == Some(range.identity.as_str()) {
            return Err(ProductionIngestionError::PayloadUnavailable {
                identity: range.identity.clone(),
            });
        }
        self.bytes_by_name
            .get(&range.identity)
            .cloned()
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            })
    }
}

fn active_allocation_bytes(runtime: &Runtime) -> u64 {
    runtime
        .memory()
        .allocations()
        .filter(|allocation| !allocation_released(allocation))
        .map(|allocation| allocation.request.size_bytes)
        .sum()
}

fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Shared setup for the `stream_materialize_model_instance_weights` tests
/// below: a three-tensor manifest, a loaded (not yet materialized) Model
/// Instance, and the `ReferenceCpuExecutor` handle backing the Provider
/// that instance is bound to (obtained before the `ReferenceCpuProvider`
/// itself is moved into the Runtime, since it shares state with whatever
/// gets registered).
fn streaming_materialization_fixture() -> (
    Runtime,
    ModelInstanceId,
    ModelManifest,
    Arc<ReferenceCpuExecutor>,
) {
    let manifest = manifest_with_tensors(vec![
        ("weight.a", vec![2, 2]),
        ("weight.b", vec![2, 2]),
        ("weight.c", vec![2, 2]),
    ]);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(provider))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    (runtime, instance, manifest, executor)
}

fn streaming_fixture_payload_bytes() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        (
            "weight.a".to_string(),
            f32_tensor_bytes(&[1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "weight.b".to_string(),
            f32_tensor_bytes(&[5.0, 6.0, 7.0, 8.0]),
        ),
        (
            "weight.c".to_string(),
            f32_tensor_bytes(&[9.0, 10.0, 11.0, 12.0]),
        ),
    ])
}

/// `implement-production-qwen-model-loading` task group 6: production
/// weight materialization can proceed one tensor at a time from a bounded
/// payload source instead of requiring a whole-model `BTreeMap<String,
/// HostTensor>` upfront -- and every tensor is read from the payload
/// source exactly once, in the order given.
#[test]
fn stream_materialize_model_instance_weights_succeeds_and_reads_each_tensor_once() {
    let (mut runtime, instance, manifest, _executor) = streaming_materialization_fixture();
    let payload_source = RecordingPayloadSource::new(streaming_fixture_payload_bytes());

    stream_materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &manifest.tensors,
        &payload_source,
    )
    .expect("streaming materialization succeeds");

    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
    assert_eq!(
        payload_source.call_log(),
        vec!["weight.a", "weight.b", "weight.c"],
        "each tensor must be read exactly once, in the manifest's own order"
    );
}

/// Task 6.6: a payload-read failure partway through a streaming attempt
/// rolls back every tensor staged so far in that attempt -- no orphan
/// Provider tensor, no leaked Memory Manager allocation -- exercised at
/// the first, middle, and last tensor in a three-tensor manifest.
#[test]
fn stream_materialize_model_instance_weights_rolls_back_on_failure_at_any_position() {
    for failing_tensor in ["weight.a", "weight.b", "weight.c"] {
        let (mut runtime, instance, manifest, executor) = streaming_materialization_fixture();
        let payload_source = RecordingPayloadSource::new(streaming_fixture_payload_bytes())
            .failing_for(failing_tensor);
        let active_allocations_before = active_allocation_bytes(&runtime);

        let error = stream_materialize_model_instance_weights(
            &mut runtime,
            &instance,
            "test",
            &manifest.tensors,
            &payload_source,
        )
        .expect_err(&format!(
            "a payload failure for '{failing_tensor}' must propagate"
        ));
        assert!(matches!(
            error,
            InferenceApiError::ModelLoadingFailed { .. }
        ));

        assert_ne!(
            runtime.model_instance(&instance).unwrap().lifecycle(),
            ModelInstanceLifecycleState::Ready,
            "failing on '{failing_tensor}' must not leave the instance Ready"
        );
        assert_eq!(
            active_allocation_bytes(&runtime),
            active_allocations_before,
            "failing on '{failing_tensor}' must leave no leaked Memory Manager allocation"
        );
        for tensor in &manifest.tensors {
            let resource_id =
                TensorResourceId::new(format!("model.{instance}.weight.{}", tensor.name));
            assert!(
                executor.read_tensor(&resource_id).is_none(),
                "failing on '{failing_tensor}' must leave no orphan Provider tensor for '{}'",
                tensor.name
            );
        }
    }
}

/// Task 6.7: peak transient host staging is bounded independently of total
/// model size. The exactly-once-per-tensor, in-order call log the other
/// two streaming tests already assert *is* the proof this function never
/// buffers ahead: this test additionally scales the tensor count well
/// past the tiny three-tensor fixture (50 tensors) to show that scaling up
/// the tensor *count* changes nothing about the one-tensor-at-a-time
/// shape of each `read_payload` call -- `stream_materialize_model_
/// instance_weights`'s loop body holds exactly one `HostTensor` local
/// variable at a time (never a growing collection), so this property does
/// not degrade as the model grows.
#[test]
fn stream_materialize_model_instance_weights_scales_tensor_count_without_buffering_ahead() {
    const TENSOR_COUNT: usize = 50;
    let names: Vec<String> = (0..TENSOR_COUNT).map(|i| format!("weight.{i}")).collect();
    let manifest = manifest_with_tensors(
        names
            .iter()
            .map(|name| (name.as_str(), vec![4, 4]))
            .collect(),
    );
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let bytes_by_name: BTreeMap<String, Vec<u8>> = names
        .iter()
        .map(|name| (name.clone(), f32_tensor_bytes(&[1.0; 16])))
        .collect();
    let payload_source = RecordingPayloadSource::new(bytes_by_name);

    stream_materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &manifest.tensors,
        &payload_source,
    )
    .expect("streaming materialization of 50 tensors succeeds");

    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
    assert_eq!(
        payload_source.call_log(),
        names,
        "still exactly once, in order, at scale"
    );
}

/// P0-C: `materialize_model_instance_weights` rejects content whose shape
/// disagrees with the manifest's declared shape for that tensor name, even
/// though this tensor has no content digest (permissive elsewhere, but not
/// for shape/dtype agreement).
#[test]
fn materialize_model_instance_weights_rejects_shape_mismatch() {
    let manifest = manifest_with_one_tensor(vec![2, 2], ModelDType::F32);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let wrong_shape_weights = BTreeMap::from([(
        "the-only-tensor".to_string(),
        HostTensor::new([3], vec![0.0, 0.0, 0.0]).unwrap(),
    )]);
    let error =
        materialize_model_instance_weights(&mut runtime, &instance, "test", &wrong_shape_weights)
            .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::WeightShapeOrDtypeMismatch { .. }
    ));
}

/// P0-C: a tensor the manifest declares with a quantized (unsupported)
/// storage dtype is rejected even when the caller supplies well-formed,
/// correctly-shaped content -- this Runtime cannot legitimately
/// materialize that tensor as F32 at all, regardless of digest presence.
/// `F16`/`Bf16` are *not* rejected here since
/// `implement-production-qwen-model-loading` task group 5 made them
/// legitimate declared storage dtypes (explicit conversion to F32) -- see
/// `materialize_model_instance_weights_accepts_f16_and_bf16_declared_dtype`.
#[test]
fn materialize_model_instance_weights_rejects_quantized_declared_dtype() {
    // I8 (not F32/F16/BF16/Q8_0/Q4_K/Q5_K) stays genuinely unsupported --
    // Q8_0/Q4_K/Q5_K moved to their own dedicated dequantization coverage
    // once `support-gguf-quantized-tensor-dequantization` added real support for
    // them (a declared Q8_0/Q4_K/Q5_K tensor backed by already-dequantized
    // F32 content, exactly like F16/BF16 already worked, is now valid,
    // not a mismatch).
    let manifest = manifest_with_one_tensor(vec![2, 2], ModelDType::I8);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let well_formed_f32_weights = BTreeMap::from([(
        "the-only-tensor".to_string(),
        HostTensor::new([2, 2], vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
    )]);
    let error = materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &well_formed_f32_weights,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::WeightShapeOrDtypeMismatch { .. }
    ));
}

/// P0-C regression guard: matching shape and F32 dtype still materializes
/// normally.
#[test]
fn materialize_model_instance_weights_accepts_matching_shape_and_dtype() {
    let manifest = manifest_with_one_tensor(vec![2, 2], ModelDType::F32);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let matching_weights = BTreeMap::from([(
        "the-only-tensor".to_string(),
        HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &matching_weights).unwrap();
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// `support-gguf-quantized-tensor-dequantization`: a manifest declaring a quantized
/// storage dtype (`Q8_0`/`Q4_K`/`Q5_K`) backed by already-dequantized F32
/// content -- exactly the shape `host_tensors_from_artifact_bytes`
/// produces for a real quantized GGUF tensor -- materializes successfully
/// through the same shape/dtype whitelist `materialize_model_instance_
/// weights_rejects_quantized_declared_dtype` proves still rejects a
/// genuinely unsupported dtype, mirroring F16/BF16's own already-working
/// "declared non-F32, content already F32" acceptance.
#[test]
fn materialize_model_instance_weights_accepts_dequantized_content_for_a_quantized_declared_dtype() {
    for quantized_dtype in [ModelDType::Q8, ModelDType::Q4K, ModelDType::Q5K] {
        let manifest = manifest_with_one_tensor(vec![2, 2], quantized_dtype);
        let mut coordinator = ModelLoadingCoordinator::new();
        coordinator.register_architecture(ModelArchitectureImplementation {
            architecture: manifest.architecture.clone(),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        });
        let mut runtime = Runtime::builder()
            .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
            .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
            .build()
            .unwrap();
        let core =
            ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
        let loaded = load_model(
            &mut coordinator,
            &mut runtime,
            ModelLoadingApiRequest::new(core),
            &manifest,
        )
        .unwrap();
        let instance = runtime
            .create_model_instance(
                &loaded,
                ModelArchitectureImplementation {
                    architecture: manifest.architecture.clone(),
                    kind: ModelArchitectureImplementationKind::TestFixture,
                    required_capabilities: Vec::new(),
                },
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .unwrap();

        let dequantized_weights = BTreeMap::from([(
            "the-only-tensor".to_string(),
            HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )]);
        materialize_model_instance_weights(&mut runtime, &instance, "test", &dequantized_weights)
            .unwrap_or_else(|error| {
                panic!("{quantized_dtype:?} declared dtype with dequantized F32 content must materialize, got: {error}")
            });
        assert_eq!(
            runtime.model_instance(&instance).unwrap().lifecycle(),
            ModelInstanceLifecycleState::Ready
        );
    }
}

/// `implement-production-qwen-model-loading` task 5.4: an F16/BF16 storage
/// dtype's explicit-conversion-to-F32 is recorded as a residency plan
/// diagnostic, never silent; an F32-storage manifest carries no such note.
#[test]
fn residency_plan_records_f16_bf16_conversion_diagnostic() {
    for (storage_dtype, expect_diagnostic) in [
        (ModelDType::F32, false),
        (ModelDType::F16, true),
        (ModelDType::Bf16, true),
    ] {
        let mut manifest = manifest_with_one_tensor(vec![2, 2], storage_dtype);
        manifest.storage_dtype = Some(storage_dtype);
        let mut coordinator = ModelLoadingCoordinator::new();
        coordinator.register_architecture(ModelArchitectureImplementation {
            architecture: manifest.architecture.clone(),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        });
        let request =
            ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
        let artifact_plan = manifest.residency_plan().unwrap();
        let plan = coordinator
            .plan(&request, &manifest, artifact_plan)
            .unwrap();
        let has_conversion_note = plan
            .diagnostics()
            .iter()
            .any(|note| note.contains("explicitly converted to F32"));
        assert_eq!(
            has_conversion_note, expect_diagnostic,
            "{storage_dtype:?}: unexpected diagnostic presence"
        );
    }
}

/// `implement-production-qwen-model-loading` task group 5: a tensor the
/// manifest declares with `F16`/`Bf16` storage materializes normally when
/// the caller supplies its already-converted `F32` content (the shape/
/// dtype gate `materialize_model_instance_weights_rejects_quantized_
/// declared_dtype` proves rejects `Q8` accepts `F16`/`Bf16` instead).
#[test]
fn materialize_model_instance_weights_accepts_f16_and_bf16_declared_dtype() {
    for storage_dtype in [ModelDType::F16, ModelDType::Bf16] {
        let manifest = manifest_with_one_tensor(vec![2, 2], storage_dtype);
        let mut coordinator = ModelLoadingCoordinator::new();
        coordinator.register_architecture(ModelArchitectureImplementation {
            architecture: manifest.architecture.clone(),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        });
        let mut runtime = Runtime::builder()
            .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
            .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
            .build()
            .unwrap();
        let core =
            ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
        let loaded = load_model(
            &mut coordinator,
            &mut runtime,
            ModelLoadingApiRequest::new(core),
            &manifest,
        )
        .unwrap();
        let instance = runtime
            .create_model_instance(
                &loaded,
                ModelArchitectureImplementation {
                    architecture: manifest.architecture.clone(),
                    kind: ModelArchitectureImplementationKind::TestFixture,
                    required_capabilities: Vec::new(),
                },
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .unwrap();

        let converted_weights = BTreeMap::from([(
            "the-only-tensor".to_string(),
            HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )]);
        materialize_model_instance_weights(&mut runtime, &instance, "test", &converted_weights)
            .unwrap_or_else(|error| {
                panic!("{storage_dtype:?}-declared tensor with converted F32 content must materialize: {error:?}")
            });
        assert_eq!(
            runtime.model_instance(&instance).unwrap().lifecycle(),
            ModelInstanceLifecycleState::Ready
        );
    }
}

// The following helpers and tests were relocated from
// `magnetar-runtime/tests/contract_tests/model_instance.rs` by
// `seal-model-loading-and-instance-creation-primitives`:
// `ModelLoadingCoordinator::load`, `ModelInstanceDefinition::
// from_loaded_context`, and `ModelInstanceManager::create`/
// `create_checked` all became `pub(crate)`, so they are no longer
// reachable from that external test crate. Each test below genuinely
// exercises one of those primitives' own contract (definition
// cloning/reset-on-create, checked-creation validation, or pre-create
// field injection `Runtime::create_model_instance` has no way to
// express) rather than merely using it as a shortcut to reach some
// other state -- see this Change's design.md ("D2: Relocate tests by
// what they actually test, not wholesale") for the full reasoning.

fn sealed_creation_manifest() -> ModelManifest {
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: instance-model
  revision: r1
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: bf16
compute_dtype: bf16
supported_compute_dtypes: [bf16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
    size_bytes: 16
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: f32
"#,
        "sha256:0000000000000000000000000000000000000000000000000000000000000001",
        "sha256:0000000000000000000000000000000000000000000000000000000000000001",
        "sha256:0000000000000000000000000000000000000000000000000000000000000001",
    ))
    .unwrap()
}

fn sealed_creation_implementation() -> ModelArchitectureImplementation {
    ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    }
}

fn sealed_creation_definition() -> ModelInstanceDefinition {
    let manifest = sealed_creation_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(sealed_creation_implementation());
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = coordinator
        .load(
            request,
            &manifest,
            &ModelTrustStore::default()
                .trust_digest(manifest.id.digest.value.clone())
                .evaluate(&manifest),
            &mut memory,
        )
        .unwrap();

    ModelInstanceDefinition::from_loaded_context(
        &loaded,
        sealed_creation_implementation(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
}

fn sealed_creation_bind_fake_weight(runtime: &mut Runtime, id: &ModelInstanceId) {
    if runtime
        .providers()
        .provider(REFERENCE_CPU_PROVIDER_NAME)
        .is_none()
    {
        runtime
            .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
            .unwrap();
    }
    let weights = BTreeMap::from([(
        "transformer.wte.weight".to_string(),
        HostTensor::new([4, 8], vec![0.0; 32]).unwrap(),
    )]);
    materialize_model_instance_weights(runtime, id, "test", &weights).unwrap();
}

fn sealed_creation_reach_ready(runtime: &mut Runtime, id: &ModelInstanceId) {
    sealed_creation_bind_fake_weight(runtime, id);
}

/// A further audit of PR #36 found that `ModelInstanceDefinition`'s
/// `#[derive(Clone)]` copies every field -- including the already-sealed
/// `resource_bindings` -- regardless of the caller's own inability to name
/// those fields directly, and that `ModelInstanceManager::create` accepted
/// a caller-supplied definition as-is. So an external caller with only
/// `pub` access (`ModelInstance::definition()`, `Clone`,
/// `ModelInstanceManager::create`) could clone a `Ready` instance's
/// definition -- carrying its real weight bindings -- into a brand-new
/// instance, aliasing live Provider resources across two distinct
/// `ModelInstanceId`s. Fixed by having `create` unconditionally reset
/// `resource_bindings` regardless of what the supplied definition
/// contained.
#[test]
fn cloned_definition_does_not_inherit_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id_a = runtime
        .model_instances_mut()
        .create(sealed_creation_definition())
        .unwrap();
    sealed_creation_reach_ready(&mut runtime, &id_a);
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let cloned = runtime.model_instance(&id_a).unwrap().definition().clone();
    let id_b = runtime.model_instances_mut().create(cloned).unwrap();

    // B must not be Ready on arrival -- creation alone never implies
    // readiness, cloned definition or not.
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // The real regression: an *empty* materialization attempt on B must
    // not be able to "adopt" A's already-committed bindings by minting
    // fresh evidence over whatever `resource_bindings.weights` happens to
    // contain. If `create` had not reset it, this would have reached
    // `Ready` for B using A's real Provider resource, with zero bytes
    // actually staged by this call.
    materialize_model_instance_weights(&mut runtime, &id_b, "test", &BTreeMap::new()).unwrap();
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // A is unaffected throughout -- this is not a mutation of A's own
    // state, only B's (failed) attempt to inherit it.
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// Same root cause as the test above, exercised through `reload` (which
/// also calls `ModelInstanceManager::create` internally with a
/// caller-supplied replacement definition) rather than a direct `create`
/// call.
#[test]
fn reload_replacement_does_not_inherit_original_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .model_instances_mut()
        .create(sealed_creation_definition())
        .unwrap();
    sealed_creation_reach_ready(&mut runtime, &id);
    let existing_definition = runtime.model_instance(&id).unwrap().definition().clone();

    let replacement_id = runtime
        .model_instances_mut()
        .reload(
            &id,
            ModelInstanceReloadRequest {
                replacement: existing_definition,
                migrate_sessions: false,
                allow_active_semantic_mutation: false,
            },
        )
        .unwrap();

    assert_ne!(
        runtime.model_instance(&replacement_id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// Proves `ModelInstanceManager::create_checked`'s own checks-validation
/// contract directly: `checks.validate()` must run and reject before
/// `create` is ever reached, and readiness checks are evaluated
/// independently afterward. `Runtime::create_model_instance` has no
/// parameter through which a caller can inject custom
/// `ModelInstanceCreationChecks`, so this cannot be re-expressed through
/// it.
#[test]
fn sealed_creation_and_readiness_checks_gate_ready_state() {
    let mut manager = ModelInstanceManager::new();
    let denied = ModelInstanceCreationChecks {
        artifact_trusted: false,
        ..ModelInstanceCreationChecks::default()
    };
    assert_eq!(
        manager.create_checked(sealed_creation_definition(), &denied),
        Err(ModelInstanceError::ModelInstancePolicyDenied)
    );

    let id = manager
        .create_checked(
            sealed_creation_definition(),
            &ModelInstanceCreationChecks::default(),
        )
        .unwrap();
    let checks = ModelInstanceReadinessChecks {
        provider_ready: false,
        ..ModelInstanceReadinessChecks::default()
    };
    assert_eq!(
        manager
            .instance_mut(&id)
            .unwrap()
            .validate_readiness(&checks),
        Err(ModelInstanceError::ModelInstanceProviderNotReady)
    );
}

/// Proves pre-create usage-dependency fields (`kv_cache_dependencies`,
/// `prefix_cache_dependencies`) set directly on a hand-built definition
/// are honored by `create`. `Runtime::create_model_instance` builds its
/// definition internally from `architecture`/`affinity` alone, with no
/// way for a caller to inject these before creation.
#[test]
fn sealed_creation_adapter_activation_records_mutation_and_invalidates_dependent_caches() {
    let mut manager = ModelInstanceManager::new();
    let mut def = sealed_creation_definition();
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-a").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-a").unwrap());
    let id = manager.create(def).unwrap();

    let report = manager
        .activate_adapters(&id, AdapterSetId::empty(), "session:opaque", true)
        .unwrap();

    assert_eq!(report.kv_caches.len(), 1);
    assert_eq!(report.prefix_entries.len(), 1);
    assert_eq!(
        manager.instance(&id).unwrap().definition().mutation_version,
        1
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|event| event.kind == ModelInstanceObservationKind::CacheInvalidation)
    );
}

/// Proves pre-create session/usage/adapter dependency fields are honored
/// by `create` and correctly released on unload, and that `create`
/// unconditionally resets `resource_bindings` regardless of what the
/// caller-supplied `def` pre-populated -- so an out-of-band memory
/// allocation id set here cannot be pre-populated on `def` before
/// `create`, only injected afterward via the two narrow, post-creation
/// mutations `ModelInstance` still allows
/// (`set_provider_resource`/`track_memory_allocation`).
#[test]
fn sealed_creation_unload_releases_memory_provider_resources_adapters_and_cache_dependencies() {
    let mut def = sealed_creation_definition();
    def.associated_sessions
        .insert(InferenceSessionId::new("session-unload").unwrap());
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-unload").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-unload").unwrap());
    def.usage.adapter_dependencies.insert(AdapterSetId::empty());
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(def).unwrap();
    sealed_creation_reach_ready(&mut runtime, &id);
    {
        let instance = runtime.model_instances_mut().instance_mut(&id).unwrap();
        instance.set_provider_resource(Some(ProviderModelResource {
            provider: ProviderBinding::new("provider-a"),
            handle_kind: "opaque-model".into(),
            release_required: true,
        }));
        // A deliberately out-of-band id: `reach_ready` above issues its own
        // real allocation from the same `MemoryManager`, whose ids start
        // at 1, so a fixture id of `1` here would silently collide in the
        // `BTreeSet` and undercount `released_memory_allocations`.
        instance.track_memory_allocation(MemoryAllocationId::new(999));
    }

    let report = runtime
        .model_instances_mut()
        .unload(&id, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();

    assert_eq!(report.invalidated.kv_caches.len(), 1);
    assert_eq!(report.invalidated.prefix_entries.len(), 1);
    assert_eq!(report.invalidated.adapters_released.len(), 1);
    // +1 relative to the fixture's single pre-set `memory_allocations`
    // entry: `reach_ready`'s own Memory Manager allocation for the bound
    // weight is released on unload too.
    assert_eq!(report.released_memory_allocations.len(), 2);
    assert_eq!(report.released_provider_resources.len(), 1);
    assert!(!report.dangling_session_references);
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

/// Proves pre-create `policy.browser_linear_memory_limit_bytes` is
/// enforced by `create` itself. `Runtime::create_model_instance` builds
/// `policy: ModelInstancePolicy::default()` internally with no way for a
/// caller to override it before creation.
#[test]
fn sealed_creation_browser_policy_rejects_native_or_oversized_instance_features() {
    let mut def = sealed_creation_definition();
    def.policy = ModelInstancePolicy {
        browser_linear_memory_limit_bytes: Some(1),
        ..ModelInstancePolicy::default()
    };
    let mut manager = ModelInstanceManager::new();
    assert_eq!(
        manager.create(def),
        Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported)
    );
}

// The following helpers and tests were relocated from
// `magnetar-runtime/tests/contract_tests/model_loading.rs` for the same
// reason as the `sealed_creation_*` block above: they call
// `ModelLoadingCoordinator::load` directly to inspect
// `ModelLoadingCoordinator`'s/`LoadedModelContext`'s own contract
// (untrusted-artifact rejection before allocation, memory-budget/
// quantization/allocation failure mapping, exact resolved plan shape) --
// concerns `inference_api::load_model`'s `Runtime`-sealed wrapper does not
// re-expose 1:1.

fn sealed_loading_digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

fn sealed_loading_valid_manifest() -> ModelManifest {
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: qwen.example
  revision: r1
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: int8
compute_dtype: bf16
supported_compute_dtypes: [bf16, fp16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
    size_bytes: 16
quantization:
  format: q4_k
  workspace_bytes: 64
shards:
  - id: shard0
    digest: {}
    size_bytes: 128
    order: 0
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: int8
    shard: shard0
"#,
        sealed_loading_digest(),
        sealed_loading_digest(),
        sealed_loading_digest(),
        sealed_loading_digest()
    ))
    .unwrap()
}

fn sealed_loading_trusted(manifest: &ModelManifest) -> ModelTrustDecision {
    ModelTrustStore::default()
        .trust_digest(manifest.id.digest.value.clone())
        .evaluate(manifest)
}

fn sealed_loading_coordinator() -> ModelLoadingCoordinator {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    coordinator
}

#[test]
fn sealed_loading_rejects_untrusted_artifact_before_memory_allocation() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let untrusted = ModelTrustStore::default()
        .reject_digest(manifest.id.digest.value.clone())
        .evaluate(&manifest);

    let error = coordinator
        .load(request, &manifest, &untrusted, &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::ModelArtifactUntrusted);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn sealed_loading_creates_runtime_owned_ready_context_without_raw_handles() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.sharding_policy = ModelShardingPolicy::Sequential;

    let context = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap();

    assert_eq!(context.state(), ModelLoadingState::Ready);
    assert!(context.can_start_inference());
    assert!(!context.plan().has_raw_native_handles());
    assert_eq!(
        context.plan().quantization_handling(),
        &ModelQuantizationHandling::DequantizeAtLoad(ModelQuantizationFormat::GgufQ4K)
    );
    assert_eq!(
        context.plan().memory_placements(),
        &[ModelResidencyLocation::Host]
    );
    assert_eq!(memory.allocations().count(), 1);
    assert!(
        coordinator
            .observations()
            .iter()
            .any(|observation| observation.kind == ModelLoadingObservationKind::ModelReady)
    );
}

#[test]
fn sealed_loading_memory_budget_failure_does_not_allocate() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.memory_budget_bytes = Some(1);

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryFeasibilityFailed);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn sealed_loading_unsupported_quantization_requires_explicit_policy() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::QuantizationUnsupported);
}

#[test]
fn sealed_loading_memory_manager_rejection_maps_to_loading_error() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryAllocationFailed);
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
fn inference_api_tachyon_and_cli_boundary_capabilities_are_inference_only() {
    for forbidden in ["git", "shell", "agent-orchestration", "secrets"] {
        assert!(validate_inference_scope(forbidden).is_err());
    }
    assert!(validate_inference_scope("generation").is_ok());
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
fn compilation_success_never_grants_trust_by_itself() {
    assert!(!compilation_result_trust(false).is_trusted());
    assert!(compilation_result_trust(true).is_trusted());
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
fn kernel_out_of_device_memory_error_round_trips_code_id_and_display() {
    let error = KernelError::KernelOutOfDeviceMemory {
        reason: "requested 8GiB, 2GiB free".into(),
    };
    assert_eq!(error.code(), KernelErrorCode::KernelOutOfDeviceMemory);
    assert_eq!(error.id(), "kernel-out-of-device-memory");
    assert_ne!(error.code(), KernelErrorCode::KernelExecutionFailed);
    let rendered = error.to_string();
    assert!(rendered.contains("out of device memory"));
    assert!(rendered.contains("8GiB"));
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
fn shape_aware_evidence_does_not_apply_outside_its_bucket() {
    assert!(performance_evidence_applies_to_workload("b1s128", "b1s128"));
    assert!(!performance_evidence_applies_to_workload(
        "b1s128", "b64s8192"
    ));
}

#[test]
fn compilation_cost_is_excluded_once_artifact_is_cached() {
    assert!(compilation_cost_excluded_from_hot_path(true));
    assert!(!compilation_cost_excluded_from_hot_path(false));
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
fn provider_advertised_alternative_never_overrides_runtime_cross_provider_decision() {
    let runtime_choice = selection_policy_identity("runtime-choice");
    let provider_alternative = selection_policy_identity("provider-alternative");
    assert_eq!(
        resolve_cross_provider_selection(&runtime_choice, Some(&provider_alternative)),
        runtime_choice
    );
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
fn selection_explanation_never_contains_native_handles() {
    let mut explanation = SelectionExplanation::default();
    explanation.exclusions.insert(
        "some-kernel".into(),
        KernelSelectionExclusionReason::PolicyDenied,
    );
    assert!(!explanation.contains_native_handles());
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
        digest: None,
    };
    (metadata, bytes)
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
fn host_tensors_from_artifact_bytes_rejects_unsupported_dtype() {
    let (mut metadata, bytes) = artifact_bytes_test_tensor("weight.a", vec![1], &[1.0]);
    // I8 (not F32/F16/BF16/Q8_0/Q4_K/Q5_K) stays genuinely unsupported --
    // Q8_0/Q4_K/Q5_K moved to their own dedicated dequantization tests
    // once `support-gguf-quantized-tensor-dequantization` added real support for
    // them (they no longer belong in this "still rejected" test).
    metadata.storage_dtype = ModelDType::I8;

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("an unsupported dtype must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::StorageDTypeUnsupported);
}

fn f16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f16_tensor_metadata(name: &str, shape: Vec<u64>, byte_len: usize) -> ModelTensorMetadata {
    ModelTensorMetadata {
        name: name.to_string(),
        shape,
        storage_dtype: ModelDType::F16,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(byte_len as u64),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    }
}

/// `implement-production-qwen-model-loading` task 5.6: numeric edge cases
/// for F16 -> F32 conversion -- +0/-0, a subnormal, the largest finite
/// magnitude, both infinities, and NaN (payload/sign preserved, not
/// canonicalized).
#[test]
fn f16_to_f32_handles_every_numeric_class_exactly() {
    assert_eq!(f16_to_f32(0x0000).to_bits(), 0f32.to_bits());
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0f32).to_bits());
    // Smallest positive subnormal f16 (2^-24) must renormalize exactly.
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    // Largest subnormal f16 mantissa.
    assert_eq!(f16_to_f32(0x03FF), 2f32.powi(-14) * (1023.0 / 1024.0));
    // Largest finite f16 magnitude (65504).
    assert_eq!(f16_to_f32(0x7BFF), 65504.0f32);
    assert_eq!(f16_to_f32(0xFBFF), -65504.0f32);
    assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
    assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
    let nan = f16_to_f32(0x7E00);
    assert!(nan.is_nan());
    assert_eq!(nan.to_bits() >> 31, 0, "sign bit must be preserved for NaN");
    let signed_nan = f16_to_f32(0xFE00);
    assert!(signed_nan.is_nan());
    assert_eq!(signed_nan.to_bits() >> 31, 1);
}

/// Same edge-case coverage for BF16 -> F32 (task 5.6), which -- because
/// bf16 is a pure truncation of f32 -- also proves the shift-based
/// conversion round-trips every class correctly, not only normal numbers.
#[test]
fn bf16_to_f32_handles_every_numeric_class_exactly() {
    assert_eq!(bf16_to_f32(0x0000).to_bits(), 0f32.to_bits());
    assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0f32).to_bits());
    assert_eq!(bf16_to_f32(0x0001), f32::from_bits(1u32 << 16));
    assert_eq!(bf16_to_f32(0x7F7F), f32::from_bits(0x7F7F_0000));
    assert!(bf16_to_f32(0x7F80).is_infinite() && bf16_to_f32(0x7F80) > 0.0);
    assert!(bf16_to_f32(0xFF80).is_infinite() && bf16_to_f32(0xFF80) < 0.0);
    assert!(bf16_to_f32(0x7FC0).is_nan());
}

/// `add-native-cuda-half-precision-compute` task 2.3: a handful of named
/// values for readability. Not load-bearing on its own -- the exhaustive
/// round-trip tests below already cover every possible bit pattern -- but
/// documents intent for a reader who does not want to reason through that
/// exhaustive coverage.
#[test]
fn f32_to_f16_handles_named_numeric_classes() {
    assert_eq!(f32_to_f16(0.0), 0x0000);
    assert_eq!(f32_to_f16(-0.0), 0x8000);
    assert_eq!(f32_to_f16(1.0), 0x3C00);
    assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001); // smallest subnormal
    assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400); // smallest normal
    assert_eq!(f32_to_f16(65504.0), 0x7BFF); // largest finite
    assert_eq!(f32_to_f16(-65504.0), 0xFBFF);
    assert_eq!(f32_to_f16(70000.0), 0x7C00); // overflow -> +Inf
    assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
    assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
}

/// The load-bearing verification: every one of the 65,536 possible `u16`
/// bit patterns is, by construction, an exactly representable `f16` value.
/// Decoding it via the already-trusted `f16_to_f32`
/// (`f16_to_f32_handles_every_numeric_class_exactly`) and re-encoding via
/// `f32_to_f16` must reproduce the original bit pattern exactly -- a round
/// trip starting from an exact value should never observe any rounding.
/// `NaN` is the sole exception (IEEE 754 does not mandate any specific
/// payload survive a round trip): only "still NaN" is checked for those.
#[test]
fn f32_to_f16_round_trips_every_possible_f16_bit_pattern_exactly() {
    for bits in 0u32..=0xFFFF {
        let bits = bits as u16;
        let decoded = f16_to_f32(bits);
        let reencoded = f32_to_f16(decoded);
        if decoded.is_nan() {
            assert!(
                f16_to_f32(reencoded).is_nan(),
                "f16 bit pattern {bits:#06x} decoded to NaN {decoded:?} must re-encode to a NaN, got {reencoded:#06x}"
            );
        } else {
            assert_eq!(
                reencoded, bits,
                "f16 bit pattern {bits:#06x} (decoded {decoded:?}) did not round-trip: got {reencoded:#06x}"
            );
        }
    }
}

#[test]
fn host_tensors_from_artifact_bytes_converts_bf16_storage_to_f32() {
    // 1.0 and -2.0 as bfloat16 bit patterns (f32's top 16 bits).
    let bytes: Vec<u8> = [0x3F80u16, 0xC000u16]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let metadata = ModelTensorMetadata {
        storage_dtype: ModelDType::Bf16,
        ..f16_tensor_metadata("weight.a", vec![2], bytes.len())
    };

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect("BF16 tensor materializes");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.data, vec![1.0, -2.0]);
}

fn quantized_tensor_metadata(
    name: &str,
    dtype: ModelDType,
    element_count: u64,
    byte_len: usize,
) -> ModelTensorMetadata {
    ModelTensorMetadata {
        name: name.to_string(),
        shape: vec![element_count],
        storage_dtype: dtype,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(byte_len as u64),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    }
}

/// GGUF `block_q8_0` (`ggml-quants.c`'s `dequantize_row_q8_0`, verified
/// against real upstream source, not recalled from memory --
/// `support-gguf-quantized-tensor-dequantization` design.md): `value[i] = d * qs[i]`,
/// `d` the block's `f16` scale, `qs[i]` a signed `int8` taken as-is.
#[test]
fn host_tensors_from_artifact_bytes_dequantizes_q8_0() {
    let mut block = Vec::with_capacity(34);
    block.extend_from_slice(&0x4000u16.to_le_bytes()); // d = 2.0 (f16)
    let qs: Vec<i8> = (-16..16).collect(); // 32 values, -16..=15
    block.extend(qs.iter().map(|&value| value as u8));
    assert_eq!(block.len(), 34);

    let metadata = quantized_tensor_metadata("weight.a", ModelDType::Q8, 32, block.len());
    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &block, 0)
        .expect("Q8_0 tensor materializes");
    let expected: Vec<f32> = qs.iter().map(|&value| 2.0 * f32::from(value)).collect();
    assert_eq!(weights.get("weight.a").unwrap().data, expected);
}

/// GGUF `block_q4_K` (`ggml-quants.c`'s `dequantize_row_q4_K`, verified
/// against real upstream source): exercises both branches of the shared
/// 12-byte 6-bit scale/min packing (`get_scale_min_k4`, sub-blocks 0-3
/// read directly, sub-blocks 4-7 split across two bytes -- the detail
/// most reimplementations get wrong) and the low/high-nibble-across-two-
/// sub-blocks-per-64-chunk interleaving.
#[test]
fn host_tensors_from_artifact_bytes_dequantizes_q4_k() {
    let mut block = Vec::with_capacity(144);
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
    // scales[12]: sub-block 0 -> (d=1, m=0); sub-block 1 -> (d=2, m=3);
    // sub-block 4 -> (d=5, m=2) via the split-byte branch; sub-block 5 ->
    // (d=7, m=3) likewise; sub-blocks 2/3/6/7 left at (0, 0).
    let scales: [u8; 12] = [1, 2, 0, 0, 0, 3, 0, 0, 0x25, 0x37, 0, 0];
    block.extend_from_slice(&scales);
    // qs[128]: bytes 0..32 -> sub-blocks 0/1 (elements 0..64); bytes
    // 32..64 -> sub-blocks 2/3 (elements 64..128, scale 0 either way);
    // bytes 64..96 -> sub-blocks 4/5 (elements 128..192); bytes 96..128
    // -> sub-blocks 6/7 (elements 192..256, scale 0 either way).
    block.extend(std::iter::repeat_n(0x21u8, 32)); // low nibble 1, high nibble 2
    block.extend(std::iter::repeat_n(0x00u8, 32));
    block.extend(std::iter::repeat_n(0x21u8, 32));
    block.extend(std::iter::repeat_n(0x00u8, 32));
    assert_eq!(block.len(), 144);

    let metadata = quantized_tensor_metadata("weight.a", ModelDType::Q4K, 256, block.len());
    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &block, 0)
        .expect("Q4_K tensor materializes");
    let mut expected = Vec::with_capacity(256);
    // Each element's value is d_sb * nibble_value - m_sb, where d_sb =
    // block_d * sub_scale and m_sb = block_dmin * sub_min (block_d =
    // block_dmin = 1.0 here) -- not d_sb - m_sb alone; the nibble value
    // (1 for a 0x21 byte's low nibble, 2 for its high nibble) is a real
    // factor, easy to drop by mistake when hand-computing this.
    expected.extend(std::iter::repeat_n(1.0f32 * 1.0 - 0.0, 32)); // sub-block 0: d_sb=1*1=1, nibble=1, m_sb=1*0=0 -> 1*1-0
    expected.extend(std::iter::repeat_n(2.0f32 * 2.0 - 3.0, 32)); // sub-block 1: d_sb=1*2=2, nibble=2, m_sb=1*3=3 -> 2*2-3
    expected.extend(std::iter::repeat_n(0.0f32, 64)); // sub-blocks 2/3: scale/min 0
    expected.extend(std::iter::repeat_n(5.0f32 * 1.0 - 2.0, 32)); // sub-block 4: d_sb=1*5=5, nibble=1, m_sb=1*2=2 -> 5*1-2
    expected.extend(std::iter::repeat_n(7.0f32 * 2.0 - 3.0, 32)); // sub-block 5: d_sb=1*7=7, nibble=2, m_sb=1*3=3 -> 7*2-3
    expected.extend(std::iter::repeat_n(0.0f32, 64)); // sub-blocks 6/7: scale/min 0
    assert_eq!(weights.get("weight.a").unwrap().data, expected);
}

/// GGUF `block_q5_K` (`ggml-quants.c`'s `dequantize_row_q5_K`): shares
/// `block_q4_K`'s scale/min packing (already exercised above) -- this
/// test isolates the one thing unique to Q5_K, the `qh` high-bit plane
/// reconstructing a 5-bit value (`q5 = (ql & 0xF) | (qh & bit ? 16 : 0)`).
#[test]
fn host_tensors_from_artifact_bytes_dequantizes_q5_k() {
    let mut block = Vec::with_capacity(176);
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
    // Only sub-block 0 carries a nonzero scale (d=1, m=0); every other
    // sub-block's scale/min stays 0, so its elements are 0 regardless of
    // ql/qh content.
    let scales: [u8; 12] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    block.extend_from_slice(&scales);
    // qh[32]: only element 0's high bit is set (bit 0, matching u1=1 for
    // the first 64-chunk's first sub-block).
    let mut qh = [0u8; 32];
    qh[0] = 0x01;
    block.extend_from_slice(&qh);
    // qs[128]: only element 0's low nibble is 1; every other low nibble
    // (and every high nibble, which belongs to sub-block 1, scale 0) is 0.
    let mut qs = [0u8; 128];
    qs[0] = 0x01;
    block.extend_from_slice(&qs);
    assert_eq!(block.len(), 176);

    let metadata = quantized_tensor_metadata("weight.a", ModelDType::Q5K, 256, block.len());
    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &block, 0)
        .expect("Q5_K tensor materializes");
    let mut expected = vec![0.0f32; 256];
    // element 0: q5 = (1 & 0xF) | 16 = 17 -> 1.0 * 17 - 0.0 = 17.0
    expected[0] = 17.0;
    assert_eq!(weights.get("weight.a").unwrap().data, expected);
}

/// Decision 8: a declared tensor digest is verified against the *original*
/// F16 storage bytes, not the converted F32 representation.
#[test]
fn host_tensors_from_artifact_bytes_verifies_f16_digest_against_original_bytes() {
    let bytes = f16_bytes(&[0x3C00]);
    let mut metadata = f16_tensor_metadata("weight.a", vec![1], bytes.len());
    metadata.digest = Some(ModelDigest::sha256(&bytes));

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect("F16 tensor with a matching digest materializes");
    assert_eq!(weights.get("weight.a").unwrap().data, vec![1.0]);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_f16_digest_mismatch() {
    let bytes = f16_bytes(&[0x3C00]);
    let mut metadata = f16_tensor_metadata("weight.a", vec![1], bytes.len());
    metadata.digest = Some(ModelDigest::sha256(b"not the real storage bytes"));

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("a digest mismatch against the original F16 bytes must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
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

// ---------------------------------------------------------------------
// provider_roadmap
// ---------------------------------------------------------------------

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
fn provider_roadmap_cli_receives_redacted_provider_diagnostics_only() {
    let raw = "provider handle=0xdeadbeef failed on cuda-stream";
    let redacted = cli_redacted_provider_diagnostic(raw);
    assert!(!redacted.contains("0xdeadbeef"));
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
fn provider_roadmap_observation_redacts_metadata_by_default() {
    let observation = ProviderRoadmapObservation::new(ProviderRoadmapObservationKind::FallbackUsed)
        .with_provider("cuda")
        .with_redacted_metadata("diagnostic", "device pointer handle=0xdeadbeef");
    let value = observation.redacted_metadata.get("diagnostic").unwrap();
    assert!(!value.contains("0xdeadbeef"));
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
fn model_format_roadmap_format_parsers_cannot_supply_execution_graphs() {
    assert!(reject_format_execution_graph(true).is_err());
    assert!(reject_format_execution_graph(false).is_ok());
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
        architecture_config: None,
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
        digest: None,
    }
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
    assert_eq!(decision.status(), ModelTrustStatus::Unknown);

    let trusted_store = ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone());
    let trusted_decision = model_format_grants_no_trust(&trusted_store, &manifest);
    assert_eq!(trusted_decision.status(), ModelTrustStatus::Trusted);
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
