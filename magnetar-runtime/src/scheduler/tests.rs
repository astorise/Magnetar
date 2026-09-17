//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::ProviderBinding;

use crate::affinity::{
    CapabilityBinding, CapabilityHealth, DeviceBinding, DeviceHealth, HealthState,
    HealthTimeToLive, HealthTimestamp, ProviderHealth, ProviderHealthReport,
    ProviderLifecycleState, ProviderReadinessState, ProviderStatusSnapshot,
};
use crate::capability::{Capability, CapabilityId};
use crate::compute::{
    COMPUTE_CAPABILITY_VERSION, ComputeDType, ComputeGraph, ComputeGraphId, ComputeLayout,
    ComputeNode, ComputeNodeId, ComputeNodeOutput, ComputeOperationDescriptor,
    ComputeOperationFamily, ComputeOperationSupport, ComputeOutput, ComputeOutputId,
    ComputeValueRef, DTypeDescriptor, ShapeDescriptor, TensorDescriptor, compute_capability,
};
use crate::device::{Device, DeviceId};
use crate::kernel::KernelAdvertisement;
use crate::observability::RuntimeEventKind;
use crate::provider::{
    Provider, ProviderError, ProviderExecutionApi, ProviderMetadata, ProviderRegistry,
};
use crate::runtime::Runtime;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
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

fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}

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
