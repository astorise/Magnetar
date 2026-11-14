use crate::affinity::*;

use crate::capability::*;

use crate::component::*;

use crate::compute::*;

use crate::device::*;

use crate::e2e_conformance::*;

use crate::execution_graph::*;

use crate::generation::*;

use crate::inference_api::*;

use crate::kernel::*;

use crate::kernel_artifact::*;

use crate::kernel_artifact_ingestion::*;

use crate::kernel_artifact_manifest::*;

use crate::kernel_cache::*;

use crate::kernel_compilation::*;

use crate::kernel_dispatch::*;

use crate::kernel_execution_plan::*;

use crate::kernel_registry::*;

use crate::kernel_selection_policy::*;

use crate::memory::*;

use crate::model::*;

use crate::model_instance::*;

use crate::model_loading::*;

use crate::observability::*;

use crate::operator::*;

use crate::planning::*;

use crate::production_model_ingestion::*;

use crate::provider::*;

use crate::reference_cpu::*;

use crate::resolution::*;

use crate::runtime::*;

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

pub(crate) struct TestProvider {
    pub(crate) metadata: ProviderMetadata,
    pub(crate) initialized: AtomicBool,
    pub(crate) shut_down: AtomicBool,
    pub(crate) fail_initialization: bool,
    pub(crate) health: ProviderHealth,
    pub(crate) status_snapshot: Option<ProviderStatusSnapshot>,
    pub(crate) capability_health: BTreeMap<CapabilityId, HealthState>,
    pub(crate) devices: Vec<Arc<dyn Device>>,
    pub(crate) execution_api: Option<Arc<dyn ProviderExecutionApi>>,
    pub(crate) kernel_advertisements: Vec<KernelAdvertisement>,
}

impl TestProvider {
    pub(crate) fn new(name: &str) -> Self {
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

pub(crate) fn capability(name: &str, version: CapabilityVersion) -> Capability {
    Capability::new(
        CapabilityId::new(name),
        version,
        CapabilityDescriptor::new("test capability")
            .with_contract(WitInterface::new(name, version.to_string())),
    )
}

pub(crate) fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}

pub(crate) fn simple_elementwise_compute_graph(name: &str) -> ComputeGraph {
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

pub(crate) struct TestProviderExecutionApi {
    pub(crate) submitted: Mutex<Vec<ProviderExecutionRequest>>,
    pub(crate) released: AtomicBool,
    pub(crate) cancel_outcome: Mutex<ProviderCancellationOutcome>,
    pub(crate) outputs: Mutex<Vec<TensorResourceDescriptor>>,
}

impl TestProviderExecutionApi {
    pub(crate) fn new() -> Self {
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

pub(crate) fn first_native_report_fixture(provider_summary: &str) -> E2eConformanceReport {
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
    let public_sources = [
        include_str!("lib.rs"),
        include_str!("../../magnetar-engine/src/lib.rs"),
        include_str!("../../magnetar-engine/src/component.rs"),
    ];
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

pub(crate) fn temp_component_artifact_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-component-artifact-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

pub(crate) fn trust_store_yaml(digest: &str) -> String {
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

    // Re-admitting the same id with the same request (a later generation
    // step dispatching the same graph node) keeps the still-active
    // allocation: one allocation, not one per step.
    manager
        .admit_kernel_output(
            id.clone(),
            &descriptor,
            placement.clone(),
            owner.clone(),
            affinity.clone(),
        )
        .unwrap();
    assert_eq!(
        manager.tensor_residency(&id).unwrap().allocation,
        Some(first_allocation)
    );
    let active_for_owner = |manager: &MemoryManager| {
        manager
            .allocations()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.owner == owner
            })
            .count()
    };
    assert_eq!(active_for_owner(&manager), 1);

    // Re-admitting it with a different request (here a larger tensor) must
    // replace and release the previous allocation, not accumulate a second
    // one.
    let larger = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    manager
        .admit_kernel_output(id.clone(), &larger, placement, owner.clone(), affinity)
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
    assert_eq!(active_for_owner(&manager), 1);
}

pub(crate) fn generation_tokenizer_metadata() -> TokenizerMetadata {
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

#[derive(Clone)]
pub(crate) struct FailingGenerationExecutor;

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
pub(crate) struct FailingKernelGenerationExecutor;

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

pub(crate) fn reference_cpu_host_tensor(
    shape: impl Into<Vec<u64>>,
    data: impl Into<Vec<f32>>,
) -> HostTensor {
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

pub(crate) fn reference_cpu_resource(
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
pub(crate) fn matmul_dispatch_plan_for_revalidation_tests() -> (KernelRegistry, KernelDispatchPlan)
{
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

pub(crate) fn tensor_resource_for_test(id: &str) -> TensorResource {
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

pub(crate) fn permissive_operator_spec(memory: OperatorMemoryBehavior) -> OperatorSpec {
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

pub(crate) fn contiguous_f32_tensor() -> TensorDescriptor {
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

pub(crate) fn model_instance_definition() -> ModelInstanceDefinition {
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
pub(crate) fn reach_ready_with_real_weight(runtime: &mut Runtime, instance: &ModelInstanceId) {
    let weights = BTreeMap::from([("weight".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(runtime, instance, "test", &weights).unwrap();
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

pub(crate) fn minimal_model_manifest() -> ModelManifest {
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

pub(crate) fn manifest_with_one_tensor(
    shape: Vec<u64>,
    storage_dtype: ModelDType,
) -> ModelManifest {
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

pub(crate) fn manifest_with_tensors(tensors: Vec<(&str, Vec<u64>)>) -> ModelManifest {
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
pub(crate) struct RecordingPayloadSource {
    pub(crate) bytes_by_name: BTreeMap<String, Vec<u8>>,
    pub(crate) calls: std::sync::Mutex<Vec<String>>,
    pub(crate) fail_for: Option<String>,
}

impl RecordingPayloadSource {
    pub(crate) fn new(bytes_by_name: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            bytes_by_name,
            calls: std::sync::Mutex::new(Vec::new()),
            fail_for: None,
        }
    }

    pub(crate) fn failing_for(mut self, name: &str) -> Self {
        self.fail_for = Some(name.to_string());
        self
    }

    pub(crate) fn call_log(&self) -> Vec<String> {
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

pub(crate) fn active_allocation_bytes(runtime: &Runtime) -> u64 {
    runtime
        .memory()
        .allocations()
        .filter(|allocation| !allocation_released(allocation))
        .map(|allocation| allocation.request.size_bytes)
        .sum()
}

pub(crate) fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
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
pub(crate) fn streaming_materialization_fixture() -> (
    Runtime,
    ModelInstanceId,
    ModelManifest,
    Arc<ReferenceCpuExecutor>,
) {
    streaming_materialization_fixture_for(manifest_with_tensors(vec![
        ("weight.a", vec![2, 2]),
        ("weight.b", vec![2, 2]),
        ("weight.c", vec![2, 2]),
    ]))
}

/// [`streaming_materialization_fixture`] for an arbitrary manifest.
pub(crate) fn streaming_materialization_fixture_for(
    manifest: ModelManifest,
) -> (
    Runtime,
    ModelInstanceId,
    ModelManifest,
    Arc<ReferenceCpuExecutor>,
) {
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

pub(crate) fn streaming_fixture_payload_bytes() -> BTreeMap<String, Vec<u8>> {
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

/// `execute-direct-quantized-weights`: a block-quantized, transposed-storage
/// weight is first offered packed; a Provider that declines it (Reference
/// CPU computes only from f32) receives the dequantized weight in its
/// declared, untransposed shape, admitted at its f32 size.
#[test]
fn stream_materialization_dequantizes_a_declined_packed_weight_into_its_declared_shape() {
    let mut manifest = manifest_with_tensors(vec![("weight.q", vec![32, 2])]);
    manifest.tensors[0].storage_dtype = ModelDType::Q8;
    manifest.tensors[0].layout = Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT.into());
    manifest.tensors[0].size_bytes = Some(68);
    let (mut runtime, instance, manifest, executor) =
        streaming_materialization_fixture_for(manifest);
    // Two stored rows of 32 quants: d = 0.5 then d = 2.0.
    let mut bytes = 0x3800_u16.to_le_bytes().to_vec();
    bytes.extend((0..32).map(|i| (i as i8 - 16) as u8));
    bytes.extend(0x4000_u16.to_le_bytes());
    bytes.extend((0..32).map(|i| (100 - i as i8) as u8));
    let payload_source =
        RecordingPayloadSource::new(BTreeMap::from([("weight.q".to_string(), bytes)]));

    stream_materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &manifest.tensors,
        &payload_source,
    )
    .expect("the declined packed weight materializes dequantized");

    assert_eq!(payload_source.call_log(), vec!["weight.q"], "read once");
    let resource = runtime
        .model_instance(&instance)
        .unwrap()
        .definition()
        .resource_bindings
        .weights["weight.q"]
        .clone();
    let stored = executor.read_tensor(&resource).expect("stored as f32");
    assert_eq!(stored.shape, vec![32, 2]);
    for k in 0..32 {
        assert_eq!(stored.data[k * 2], 0.5 * (k as f32 - 16.0));
        assert_eq!(stored.data[k * 2 + 1], 2.0 * (100.0 - k as f32));
    }
    assert_eq!(active_allocation_bytes(&runtime), 64 * 4);
}

/// A transposed-storage layout on anything but a matrix, or an unknown
/// layout, is refused rather than read as untransposed bytes.
#[test]
fn materialization_rejects_unusable_storage_layouts() {
    for (shape, layout) in [
        (vec![64], crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT),
        (vec![2, 2], "column-major-ish"),
    ] {
        let mut manifest = manifest_with_tensors(vec![("weight.a", shape)]);
        manifest.tensors[0].layout = Some(layout.into());
        let element_count: u64 = manifest.tensors[0].shape.iter().product();
        let payload_source = RecordingPayloadSource::new(BTreeMap::from([(
            "weight.a".to_string(),
            f32_tensor_bytes(&vec![1.0; element_count as usize]),
        )]));
        let error = crate::model_loading::host_tensor_from_payload_source(
            &manifest.tensors[0],
            &payload_source,
        )
        .expect_err("the layout is refused");
        assert!(
            error.to_string().contains("layout") || error.to_string().contains("transposed"),
            "{error}"
        );
    }
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

pub(crate) fn sealed_creation_manifest() -> ModelManifest {
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

pub(crate) fn sealed_creation_implementation() -> ModelArchitectureImplementation {
    ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    }
}

pub(crate) fn sealed_creation_definition() -> ModelInstanceDefinition {
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

pub(crate) fn sealed_creation_bind_fake_weight(runtime: &mut Runtime, id: &ModelInstanceId) {
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

pub(crate) fn sealed_creation_reach_ready(runtime: &mut Runtime, id: &ModelInstanceId) {
    sealed_creation_bind_fake_weight(runtime, id);
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

pub(crate) fn session_creation_request() -> SessionCreationRequest {
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

// ---------------------------------------------------------------------
// cli_boundary
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_artifact
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_artifact_manifest
// ---------------------------------------------------------------------

pub(crate) fn temp_kernel_bundle_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-kernel-bundle-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(directory.join("blobs").join("sha256")).unwrap();
    directory
}

pub(crate) fn write_kernel_bundle(directory: &std::path::Path, blob_bytes: &[u8]) -> String {
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
    wrong_version.abi_version = crate::provider_abi::ProviderAbiVersion::new(99, 0);
    assert!(matches!(
        wrong_version.validate(),
        Err(KernelCompilationError::AbiIncompatible { .. })
    ));
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

// ---------------------------------------------------------------------
// kernel_cache
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// kernel_registry generated-kernel lifecycle
// ---------------------------------------------------------------------

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

pub(crate) fn selection_policy_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("selection-policy-provider"),
        name,
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::CpuScalar,
    )
}

pub(crate) fn selection_policy_identity(name: &str) -> CandidateIdentity {
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

pub(crate) fn artifact_bytes_test_tensor(
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

pub(crate) fn f16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(crate) fn f16_tensor_metadata(
    name: &str,
    shape: Vec<u64>,
    byte_len: usize,
) -> ModelTensorMetadata {
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

pub(crate) fn quantized_tensor_metadata(
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

// ---------------------------------------------------------------------
// model_format_roadmap
// ---------------------------------------------------------------------

pub(crate) fn tensor(name: &str, shape: Vec<u64>) -> ModelTensorMetadata {
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
