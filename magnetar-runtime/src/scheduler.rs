// The Provider execution types live in `execution`; they keep their paths here.
use crate::affinity::*;
use crate::device::Device;
pub(crate) use crate::execution::*;
pub use crate::execution::{
    ProviderCancellationOutcome, ProviderExecutionHandle, ProviderExecutionId,
    ProviderExecutionProgress, ProviderExecutionRequest, ProviderExecutionResult,
    ProviderExecutionStatus, ScheduledOperation, ScheduledOperationResult, SchedulerError,
    SchedulingDiagnostic, SchedulingState,
};
use crate::memory::*;
use crate::observability::*;
pub use crate::provider_status::{
    ProviderExecutionDiagnostic, ProviderExecutionError, ProviderExecutionErrorCode,
    ProviderExecutionPhase, ScheduledOperationId, SchedulerErrorCode,
};
use crate::runtime::next_scheduled_operation_id;
use crate::runtime::*;
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulingPolicy {
    #[default]
    Fifo,
    Priority,
    Deadline,
    ResourceAware,
    BatchAware,
    Fairness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerQueue {
    capacity: usize,
    order: VecDeque<ScheduledOperationId>,
}
impl SchedulerQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    fn push(&mut self, id: ScheduledOperationId) -> Result<usize, SchedulerError> {
        if self.order.len() >= self.capacity {
            return Err(SchedulerError::QueueCapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.order.push_back(id);
        Ok(self.order.len() - 1)
    }
    fn pop_next(&mut self) -> Option<ScheduledOperationId> {
        self.order.pop_front()
    }
    fn remove(&mut self, id: ScheduledOperationId) -> bool {
        let Some(position) = self.order.iter().position(|candidate| *candidate == id) else {
            return false;
        };
        self.order.remove(position);
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Scheduler {
    queue: SchedulerQueue,
    policy: SchedulingPolicy,
    operations: BTreeMap<ScheduledOperationId, ScheduledOperation>,
    observations: Vec<RuntimeEvent>,
    next_order: u64,
}
impl Scheduler {
    pub fn new(policy: SchedulingPolicy, capacity: usize) -> Self {
        Self {
            queue: SchedulerQueue::new(capacity),
            policy,
            operations: BTreeMap::new(),
            observations: Vec::new(),
            next_order: 0,
        }
    }
    pub fn policy(&self) -> SchedulingPolicy {
        self.policy
    }
    pub fn queue(&self) -> &SchedulerQueue {
        &self.queue
    }
    pub fn operation(&self, id: ScheduledOperationId) -> Option<&ScheduledOperation> {
        self.operations.get(&id)
    }
    pub fn observations(&self) -> &[RuntimeEvent] {
        &self.observations
    }
    pub fn schedule(
        &mut self,
        runtime: &Runtime,
        plan: ComputeExecutionPlan,
    ) -> Result<ScheduledOperationId, SchedulerError> {
        runtime.validate_scheduler_plan(&plan).map_err(|error| {
            SchedulerError::InvalidExecutionPlan {
                reason: error.to_string(),
            }
        })?;
        if let MemoryAdmissionDecision::Reject { reason } =
            runtime.memory_admission_for_plan(&plan.memory_plan, true)
        {
            return Err(SchedulerError::MemoryPlanInvalid { reason });
        }
        let id = next_scheduled_operation_id();
        let accepted_order = self.next_order;
        self.next_order += 1;
        let mut operation = ScheduledOperation {
            id,
            plan,
            state: SchedulingState::Accepted,
            accepted_order,
            diagnostics: vec![
                SchedulingDiagnostic::Accepted { operation: id },
                SchedulingDiagnostic::QueueTime { accepted_order },
            ],
            result: None,
        };
        let position = match self.queue.push(id) {
            Ok(position) => position,
            Err(error) => {
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::StableFailureReason(error.code()));
                self.observations.push(
                    RuntimeEvent::new(
                        operation.plan.trace_id.clone(),
                        SpanId::new(format!("schedule:{id}")),
                        RuntimeObservationPhase::Scheduling,
                        RuntimeEventKind::SchedulerBackpressure,
                        "scheduler queue capacity exceeded",
                    )
                    .with_plan(operation.plan.id.clone())
                    .with_operation(id)
                    .with_provider(operation.plan.provider.clone())
                    .with_diagnostic_code(RuntimeDiagnosticCode::SchedulerBackpressure),
                );
                return Err(error);
            }
        };
        operation.state = SchedulingState::Queued;
        operation.diagnostics.push(SchedulingDiagnostic::Queued {
            operation: id,
            position,
        });
        operation
            .diagnostics
            .push(SchedulingDiagnostic::SelectedProvider(
                operation.plan.provider.clone(),
            ));
        if let Some(device) = &operation.plan.device {
            operation
                .diagnostics
                .push(SchedulingDiagnostic::SelectedDevice(device.clone()));
        }
        self.observations.push(
            RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("schedule:{id}")),
                RuntimeObservationPhase::Scheduling,
                RuntimeEventKind::Scheduled,
                "scheduled operation queued",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone()),
        );
        self.operations.insert(id, operation);
        Ok(id)
    }
    pub fn submit_next(
        &mut self,
        runtime: &Runtime,
    ) -> Result<Option<ScheduledOperationId>, SchedulerError> {
        let Some(id) = self.queue.pop_next() else {
            return Ok(None);
        };
        let provider = self
            .operations
            .get(&id)
            .map(|operation| operation.plan.provider.clone())
            .ok_or_else(|| SchedulerError::SubmissionFailed {
                operation: id,
                reason: "operation is not registered".into(),
            })?;
        let provider_health = runtime
            .providers()
            .provider(provider.as_str())
            .map(|provider| provider.status_snapshot())
            .ok_or_else(|| SchedulerError::ProviderUnavailable(provider.clone()))?;
        if matches!(provider_health.health_reason, ProviderStatusReason::Stale) {
            self.interrupt_operation(id, "selected Provider status is stale before submission");
            return Err(SchedulerError::StaleHealthReport(provider));
        }
        let provider_health = provider_health.provider_health_compat();
        if let Some(error) = scheduler_error_for_provider_health(&provider, provider_health) {
            self.interrupt_operation(
                id,
                format!("selected Provider is {provider_health:?} before submission"),
            );
            return Err(error);
        }
        if let Some(device) = self
            .operations
            .get(&id)
            .and_then(ScheduledOperation::device)
            && !runtime
                .device(device.id())
                .map(Device::availability)
                .unwrap_or(DeviceAvailability::Unavailable)
                .accepts_new_work_by_default()
        {
            let device = device.clone();
            let health = runtime
                .device(device.id())
                .map(Device::availability)
                .unwrap_or(DeviceAvailability::Unavailable);
            self.interrupt_operation(id, "selected Device is unavailable before submission");
            return Err(scheduler_error_for_device_health(&device, health)
                .unwrap_or(SchedulerError::DeviceUnavailable(device)));
        }
        let submitted_event = {
            let operation = self
                .operations
                .get_mut(&id)
                .expect("operation checked above");
            operation.state = SchedulingState::Ready;
            operation.state = SchedulingState::Submitted;
            RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("submit:{id}")),
                RuntimeObservationPhase::ProviderExecution,
                RuntimeEventKind::ProviderSubmission,
                "scheduled operation submitted to Provider",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone())
        };
        self.observations.push(submitted_event);
        let started_event = {
            let operation = self
                .operations
                .get_mut(&id)
                .expect("operation checked above");
            operation.state = SchedulingState::Running;
            RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("run:{id}")),
                RuntimeObservationPhase::ProviderExecution,
                RuntimeEventKind::ExecutionStarted,
                "scheduled operation execution started",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone())
        };
        self.observations.push(started_event);
        Ok(Some(id))
    }
    pub fn complete(&mut self, id: ScheduledOperationId) -> Result<(), SchedulerError> {
        let event = {
            let operation = self.operation_mut(id)?;
            if operation.state.is_terminal() {
                return Ok(());
            }
            operation.state = SchedulingState::Completed;
            operation
                .diagnostics
                .push(SchedulingDiagnostic::TerminalState(
                    SchedulingState::Completed,
                ));
            operation.result = Some(ScheduledOperationResult {
                state: SchedulingState::Completed,
                outputs: operation.plan.outputs.clone(),
                diagnostics: operation.diagnostics.clone(),
                error: None,
            });
            RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("complete:{id}")),
                RuntimeObservationPhase::ProviderExecution,
                RuntimeEventKind::ExecutionCompleted,
                "scheduled operation execution completed",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone())
        };
        self.observations.push(event);
        Ok(())
    }
    pub fn fail(
        &mut self,
        id: ScheduledOperationId,
        reason: impl Into<String>,
    ) -> Result<(), SchedulerError> {
        let reason = reason.into();
        let error = SchedulerError::ExecutionFailed {
            operation: id,
            reason: reason.clone(),
        };
        let event = {
            let operation = self.operation_mut(id)?;
            if operation.state.is_terminal() {
                return Ok(());
            }
            operation.state = SchedulingState::Failed;
            operation
                .diagnostics
                .push(SchedulingDiagnostic::StableFailureReason(error.code()));
            operation
                .diagnostics
                .push(SchedulingDiagnostic::TerminalState(SchedulingState::Failed));
            operation.result = Some(ScheduledOperationResult {
                state: SchedulingState::Failed,
                outputs: Vec::new(),
                diagnostics: operation.diagnostics.clone(),
                error: Some(error),
            });
            RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("fail:{id}")),
                RuntimeObservationPhase::ProviderExecution,
                RuntimeEventKind::DiagnosticEmitted,
                "scheduled operation execution failed",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone())
            .with_diagnostic_code(RuntimeDiagnosticCode::ExecutionFailed)
        };
        self.observations.push(event);
        Ok(())
    }
    pub fn cancel(&mut self, id: ScheduledOperationId) -> Result<(), SchedulerError> {
        let state = self.operation_mut(id)?.state;
        match state {
            SchedulingState::Accepted | SchedulingState::Queued | SchedulingState::Ready => {
                self.queue.remove(id);
                let event = {
                    let operation = self.operation_mut(id)?;
                    operation.state = SchedulingState::Cancelled;
                    operation
                        .diagnostics
                        .push(SchedulingDiagnostic::CancellationRequested);
                    operation
                        .diagnostics
                        .push(SchedulingDiagnostic::TerminalState(
                            SchedulingState::Cancelled,
                        ));
                    operation.result = Some(ScheduledOperationResult {
                        state: SchedulingState::Cancelled,
                        outputs: Vec::new(),
                        diagnostics: operation.diagnostics.clone(),
                        error: None,
                    });
                    RuntimeEvent::new(
                        operation.plan.trace_id.clone(),
                        SpanId::new(format!("cancel:{id}")),
                        RuntimeObservationPhase::ProviderExecution,
                        RuntimeEventKind::ExecutionCancelled,
                        "scheduled operation cancelled",
                    )
                    .with_plan(operation.plan.id.clone())
                    .with_operation(id)
                    .with_provider(operation.plan.provider.clone())
                    .with_diagnostic_code(RuntimeDiagnosticCode::ExecutionCancelled)
                };
                self.observations.push(event);
                Ok(())
            }
            SchedulingState::Submitted | SchedulingState::Running => {
                let operation = self.operation_mut(id)?;
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::CancellationRequested);
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::CancellationForwardedToProvider(
                        operation.plan.provider.clone(),
                    ));
                Err(SchedulerError::CancellationUnsupported(id))
            }
            SchedulingState::Completed
            | SchedulingState::Cancelled
            | SchedulingState::Failed
            | SchedulingState::Interrupted => Ok(()),
        }
    }
    pub fn result(&self, id: ScheduledOperationId) -> Option<&ScheduledOperationResult> {
        self.operations.get(&id)?.result.as_ref()
    }
    fn operation_mut(
        &mut self,
        id: ScheduledOperationId,
    ) -> Result<&mut ScheduledOperation, SchedulerError> {
        self.operations
            .get_mut(&id)
            .ok_or_else(|| SchedulerError::InvalidExecutionPlan {
                reason: format!("scheduled operation '{id}' is unknown"),
            })
    }
    fn interrupt_operation(&mut self, id: ScheduledOperationId, reason: impl Into<String>) {
        if let Some(operation) = self.operations.get_mut(&id) {
            let error = SchedulerError::ExecutionInterrupted {
                operation: id,
                reason: reason.into(),
            };
            operation.state = SchedulingState::Interrupted;
            operation
                .diagnostics
                .push(SchedulingDiagnostic::StableFailureReason(error.code()));
            operation
                .diagnostics
                .push(SchedulingDiagnostic::TerminalState(
                    SchedulingState::Interrupted,
                ));
            operation.result = Some(ScheduledOperationResult {
                state: SchedulingState::Interrupted,
                outputs: Vec::new(),
                diagnostics: operation.diagnostics.clone(),
                error: Some(error),
            });
            let event = RuntimeEvent::new(
                operation.plan.trace_id.clone(),
                SpanId::new(format!("interrupt:{id}")),
                RuntimeObservationPhase::ProviderExecution,
                RuntimeEventKind::ExecutionInterrupted,
                "scheduled operation execution interrupted",
            )
            .with_plan(operation.plan.id.clone())
            .with_operation(id)
            .with_provider(operation.plan.provider.clone())
            .with_diagnostic_code(RuntimeDiagnosticCode::ExecutionInterrupted);
            self.observations.push(event);
        }
    }
}

pub fn runtime_metrics_for_execution_plan(plan: &ComputeExecutionPlan) -> Vec<RuntimeMetric> {
    let mut metrics = vec![
        RuntimeMetric::new(
            RuntimeMetricKind::MemoryUsageEstimate,
            plan.memory_plan.pressure.estimated_required_bytes,
            "bytes",
        ),
        RuntimeMetric::new(
            RuntimeMetricKind::TransferVolume,
            plan.memory_plan.pressure.transfer_buffer_cost_bytes,
            "bytes",
        ),
        RuntimeMetric::new(
            RuntimeMetricKind::MaterializationCount,
            plan.memory_plan
                .decisions
                .iter()
                .filter(|decision| {
                    matches!(
                        decision,
                        MemoryPlanningDecision::RequireMaterialization { .. }
                    )
                })
                .count() as u64,
            "count",
        ),
    ];
    for metric in &mut metrics {
        metric.trace_id = Some(plan.trace_id.clone());
        metric.provider = Some(plan.provider.clone());
        metric.device = plan.device.clone();
    }
    metrics
}

pub fn runtime_event_for_provider_health(report: &ProviderHealthReport) -> RuntimeEvent {
    RuntimeEvent::new(
        TraceId::new(format!("trace:health:{}", report.provider)),
        SpanId::new(format!("health:provider:{}", report.provider)),
        RuntimeObservationPhase::Health,
        RuntimeEventKind::ProviderHealthChanged,
        format!("Provider health changed to {:?}", report.state),
    )
    .with_provider(report.provider.clone())
    .with_diagnostic_code(RuntimeDiagnosticCode::ProviderHealthChanged)
}

pub fn runtime_event_for_device_health(report: &DeviceHealth) -> RuntimeEvent {
    RuntimeEvent::new(
        TraceId::new(format!("trace:health:{}", report.device)),
        SpanId::new(format!("health:device:{}", report.device)),
        RuntimeObservationPhase::Health,
        RuntimeEventKind::DeviceHealthChanged,
        format!("Device health changed to {:?}", report.state),
    )
    .with_provider(report.provider.clone())
    .with_device(report.device.clone())
    .with_diagnostic_code(RuntimeDiagnosticCode::DeviceHealthChanged)
}

pub fn runtime_events_for_provider_status(
    status: &ProviderStatusSnapshot,
    now: Option<HealthTimestamp>,
) -> Vec<RuntimeEvent> {
    let trace = TraceId::new(format!("trace:status:{}", status.provider));
    let base = |kind, code, label: &str| {
        RuntimeEvent::new(
            trace.clone(),
            SpanId::new(format!("status:{}:{kind:?}", status.provider)),
            RuntimeObservationPhase::Health,
            kind,
            label,
        )
        .with_provider(status.provider.clone())
        .with_diagnostic_code(code)
    };
    let mut events = vec![
        base(
            RuntimeEventKind::ProviderLifecycleChanged,
            RuntimeDiagnosticCode::ProviderLifecycleChanged,
            format!("Provider lifecycle is {:?}", status.lifecycle).as_str(),
        ),
        base(
            RuntimeEventKind::ProviderHealthChanged,
            RuntimeDiagnosticCode::ProviderHealthChanged,
            format!("Provider health is {:?}", status.health).as_str(),
        ),
        base(
            RuntimeEventKind::ProviderReadinessChanged,
            RuntimeDiagnosticCode::ProviderReadinessChanged,
            format!("Provider readiness is {:?}", status.readiness).as_str(),
        ),
        base(
            RuntimeEventKind::ProviderPressureChanged,
            RuntimeDiagnosticCode::ProviderPressureChanged,
            format!("Provider pressure is {:?}", status.pressure).as_str(),
        ),
        base(
            RuntimeEventKind::ProviderAdmissionChanged,
            RuntimeDiagnosticCode::ProviderAdmissionChanged,
            format!("Provider admission is {:?}", status.admission).as_str(),
        ),
    ];
    if now.is_some_and(|timestamp| status.is_stale_at(timestamp))
        || matches!(status.health_reason, ProviderStatusReason::Stale)
    {
        events.push(base(
            RuntimeEventKind::ProviderStatusStale,
            RuntimeDiagnosticCode::ProviderStatusStale,
            "Provider status is stale",
        ));
    }
    if matches!(status.lifecycle, ProviderLifecycleState::Draining) {
        events.push(base(
            RuntimeEventKind::ProviderDrainStarted,
            RuntimeDiagnosticCode::ProviderDrainStarted,
            "Provider drain is active",
        ));
        if status.is_drain_complete() {
            events.push(base(
                RuntimeEventKind::ProviderDrainCompleted,
                RuntimeDiagnosticCode::ProviderDrainCompleted,
                "Provider drain is complete",
            ));
        }
    }
    for device in &status.devices {
        events.push(
            RuntimeEvent::new(
                trace.clone(),
                SpanId::new(format!(
                    "status:{}:device:{}",
                    status.provider, device.device
                )),
                RuntimeObservationPhase::Health,
                RuntimeEventKind::DeviceStatusChanged,
                format!("Device status is {:?}", device.readiness),
            )
            .with_provider(status.provider.clone())
            .with_device(device.device.clone())
            .with_diagnostic_code(RuntimeDiagnosticCode::DeviceStatusChanged),
        );
    }
    for capability in &status.capabilities {
        events.push(
            RuntimeEvent::new(
                trace.clone(),
                SpanId::new(format!(
                    "status:{}:capability:{}",
                    status.provider, capability.capability
                )),
                RuntimeObservationPhase::Health,
                RuntimeEventKind::CapabilityStatusChanged,
                format!("Capability status is {:?}", capability.readiness),
            )
            .with_provider(status.provider.clone())
            .with_capability(capability.capability.clone())
            .with_diagnostic_code(RuntimeDiagnosticCode::CapabilityStatusChanged),
        );
    }
    events
}

pub(crate) fn scheduler_error_for_provider_health(
    provider: &ProviderBinding,
    health: ProviderHealth,
) -> Option<SchedulerError> {
    match health {
        HealthState::Unknown => Some(SchedulerError::ProviderHealthUnknown(provider.clone())),
        HealthState::Initializing => Some(SchedulerError::ProviderInitializing(provider.clone())),
        HealthState::Available | HealthState::Degraded => None,
        HealthState::Saturated => Some(SchedulerError::ProviderSaturated(provider.clone())),
        HealthState::Draining => Some(SchedulerError::ProviderDraining(provider.clone())),
        HealthState::Unavailable => Some(SchedulerError::ProviderUnavailable(provider.clone())),
        HealthState::Interrupted => Some(SchedulerError::ProviderInterrupted(provider.clone())),
    }
}

pub(crate) fn scheduler_error_for_device_health(
    device: &DeviceBinding,
    health: DeviceAvailability,
) -> Option<SchedulerError> {
    match health {
        HealthState::Unknown | HealthState::Initializing => {
            Some(SchedulerError::DeviceHealthUnknown(device.clone()))
        }
        HealthState::Available | HealthState::Degraded | HealthState::Draining => None,
        HealthState::Saturated => Some(SchedulerError::DeviceSaturated(device.clone())),
        HealthState::Unavailable | HealthState::Interrupted => {
            Some(SchedulerError::DeviceUnavailable(device.clone()))
        }
    }
}

#[cfg(test)]
mod tests;
