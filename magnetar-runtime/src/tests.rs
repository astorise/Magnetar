use crate::adapter::*;
use crate::affinity::*;
use crate::batching::*;
use crate::capability::*;
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
use crate::kernel_cache::*;
use crate::kernel_compilation::*;
use crate::kernel_dispatch::*;
use crate::kernel_execution_plan::*;
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
        artifact_format: ArtifactFormat::HuggingFace,
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
fn inference_api_generation_result_reports_error_for_failed_finish_reason() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, Vec::new(), FinishReason::ProviderError);

    let result = GenerationResult::new(output);
    assert!(result.error.is_some());
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
fn inference_api_generation_api_request_carries_privacy_policy() {
    let request = GenerationApiRequest::new(
        generation_request(),
        SessionRedactionPolicy::RedactRawInputs,
    );
    assert_eq!(request.privacy, SessionRedactionPolicy::RedactRawInputs);
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

// ---------------------------------------------------------------------
// kernel_artifact
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// kernel_artifact_ingestion
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// kernel_qualification
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_benchmark
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_autotuning
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// kernel_cache
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_registry generated-kernel lifecycle
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// provider_roadmap
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// model_format_roadmap
// ---------------------------------------------------------------------

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
