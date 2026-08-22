use crate::compute::redact_backend_diagnostic;
use crate::planning::provider_execution_id;
use crate::runtime::next_scheduled_operation_id;
use crate::*;
use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt,
};
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScheduledOperationId(u64);
impl ScheduledOperationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for ScheduledOperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulingState {
    Accepted,
    Queued,
    Ready,
    Submitted,
    Running,
    Completed,
    Cancelled,
    Failed,
    Interrupted,
}
impl SchedulingState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed | Self::Interrupted
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulingDiagnostic {
    Accepted {
        operation: ScheduledOperationId,
    },
    Queued {
        operation: ScheduledOperationId,
        position: usize,
    },
    SelectedProvider(ProviderBinding),
    SelectedDevice(DeviceBinding),
    QueueTime {
        accepted_order: u64,
    },
    CancellationRequested,
    CancellationForwardedToProvider(ProviderBinding),
    TerminalState(SchedulingState),
    StableFailureReason(SchedulerErrorCode),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulerErrorCode {
    InvalidExecutionPlan,
    QueueCapacityExceeded,
    ProviderHealthUnknown,
    ProviderInitializing,
    ProviderDegradedRejected,
    ProviderSaturated,
    ProviderDraining,
    ProviderUnavailable,
    ProviderInterrupted,
    DeviceHealthUnknown,
    DeviceSaturated,
    DeviceUnavailable,
    StaleHealthReport,
    ResourceAffinityConflict,
    MemoryPlanInvalid,
    SubmissionFailed,
    CancellationUnsupported,
    CancellationFailed,
    ExecutionFailed,
    ExecutionInterrupted,
    OperationTimeout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    InvalidExecutionPlan {
        reason: String,
    },
    QueueCapacityExceeded {
        capacity: usize,
    },
    ProviderHealthUnknown(ProviderBinding),
    ProviderInitializing(ProviderBinding),
    ProviderDegradedRejected(ProviderBinding),
    ProviderSaturated(ProviderBinding),
    ProviderDraining(ProviderBinding),
    ProviderUnavailable(ProviderBinding),
    ProviderInterrupted(ProviderBinding),
    DeviceHealthUnknown(DeviceBinding),
    DeviceSaturated(DeviceBinding),
    DeviceUnavailable(DeviceBinding),
    StaleHealthReport(ProviderBinding),
    ResourceAffinityConflict {
        reason: String,
    },
    MemoryPlanInvalid {
        reason: String,
    },
    SubmissionFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    CancellationUnsupported(ScheduledOperationId),
    CancellationFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    ExecutionFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    ExecutionInterrupted {
        operation: ScheduledOperationId,
        reason: String,
    },
    OperationTimeout(ScheduledOperationId),
}
impl SchedulerError {
    pub const fn code(&self) -> SchedulerErrorCode {
        match self {
            Self::InvalidExecutionPlan { .. } => SchedulerErrorCode::InvalidExecutionPlan,
            Self::QueueCapacityExceeded { .. } => SchedulerErrorCode::QueueCapacityExceeded,
            Self::ProviderHealthUnknown(_) => SchedulerErrorCode::ProviderHealthUnknown,
            Self::ProviderInitializing(_) => SchedulerErrorCode::ProviderInitializing,
            Self::ProviderDegradedRejected(_) => SchedulerErrorCode::ProviderDegradedRejected,
            Self::ProviderSaturated(_) => SchedulerErrorCode::ProviderSaturated,
            Self::ProviderDraining(_) => SchedulerErrorCode::ProviderDraining,
            Self::ProviderUnavailable(_) => SchedulerErrorCode::ProviderUnavailable,
            Self::ProviderInterrupted(_) => SchedulerErrorCode::ProviderInterrupted,
            Self::DeviceHealthUnknown(_) => SchedulerErrorCode::DeviceHealthUnknown,
            Self::DeviceSaturated(_) => SchedulerErrorCode::DeviceSaturated,
            Self::DeviceUnavailable(_) => SchedulerErrorCode::DeviceUnavailable,
            Self::StaleHealthReport(_) => SchedulerErrorCode::StaleHealthReport,
            Self::ResourceAffinityConflict { .. } => SchedulerErrorCode::ResourceAffinityConflict,
            Self::MemoryPlanInvalid { .. } => SchedulerErrorCode::MemoryPlanInvalid,
            Self::SubmissionFailed { .. } => SchedulerErrorCode::SubmissionFailed,
            Self::CancellationUnsupported(_) => SchedulerErrorCode::CancellationUnsupported,
            Self::CancellationFailed { .. } => SchedulerErrorCode::CancellationFailed,
            Self::ExecutionFailed { .. } => SchedulerErrorCode::ExecutionFailed,
            Self::ExecutionInterrupted { .. } => SchedulerErrorCode::ExecutionInterrupted,
            Self::OperationTimeout(_) => SchedulerErrorCode::OperationTimeout,
        }
    }
}
impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExecutionPlan { reason } => {
                write!(f, "invalid execution plan: {reason}")
            }
            Self::QueueCapacityExceeded { capacity } => {
                write!(f, "scheduler queue capacity {capacity} exceeded")
            }
            Self::ProviderHealthUnknown(provider) => {
                write!(
                    f,
                    "provider '{provider}' health is unknown before submission"
                )
            }
            Self::ProviderInitializing(provider) => {
                write!(f, "provider '{provider}' is initializing before submission")
            }
            Self::ProviderDegradedRejected(provider) => {
                write!(
                    f,
                    "provider '{provider}' is degraded and rejected by policy"
                )
            }
            Self::ProviderSaturated(provider) => {
                write!(f, "provider '{provider}' is saturated before submission")
            }
            Self::ProviderDraining(provider) => {
                write!(f, "provider '{provider}' is draining before submission")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable before submission")
            }
            Self::ProviderInterrupted(provider) => {
                write!(f, "provider '{provider}' is interrupted before submission")
            }
            Self::DeviceHealthUnknown(device) => {
                write!(f, "device '{device}' health is unknown before submission")
            }
            Self::DeviceSaturated(device) => {
                write!(f, "device '{device}' is saturated before submission")
            }
            Self::DeviceUnavailable(device) => {
                write!(f, "device '{device}' is unavailable before submission")
            }
            Self::StaleHealthReport(provider) => {
                write!(f, "provider '{provider}' health report is stale")
            }
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "resource affinity conflict: {reason}")
            }
            Self::MemoryPlanInvalid { reason } => write!(f, "memory plan invalid: {reason}"),
            Self::SubmissionFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' submission failed: {reason}"
                )
            }
            Self::CancellationUnsupported(operation) => {
                write!(
                    f,
                    "scheduled operation '{operation}' cancellation is unsupported"
                )
            }
            Self::CancellationFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' cancellation failed: {reason}"
                )
            }
            Self::ExecutionFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' execution failed: {reason}"
                )
            }
            Self::ExecutionInterrupted { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' execution interrupted: {reason}"
                )
            }
            Self::OperationTimeout(operation) => {
                write!(f, "scheduled operation '{operation}' timed out")
            }
        }
    }
}
impl Error for SchedulerError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOperationResult {
    pub state: SchedulingState,
    pub outputs: Vec<ExecutionOutput>,
    pub diagnostics: Vec<SchedulingDiagnostic>,
    pub error: Option<SchedulerError>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOperation {
    pub id: ScheduledOperationId,
    pub plan: ComputeExecutionPlan,
    pub state: SchedulingState,
    pub accepted_order: u64,
    pub diagnostics: Vec<SchedulingDiagnostic>,
    pub result: Option<ScheduledOperationResult>,
}
impl ScheduledOperation {
    pub fn provider(&self) -> &ProviderBinding {
        &self.plan.provider
    }
    pub fn device(&self) -> Option<&DeviceBinding> {
        self.plan.device.as_ref()
    }
    pub fn state(&self) -> SchedulingState {
        self.state
    }
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderExecutionId(String);
impl ProviderExecutionId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ProviderExecutionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionHandle {
    pub id: ProviderExecutionId,
    pub operation: ScheduledOperationId,
    pub plan: ExecutionPlanId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
}
impl ProviderExecutionHandle {
    pub fn new(
        operation: ScheduledOperationId,
        plan: ExecutionPlanId,
        provider: ProviderBinding,
        device: Option<DeviceBinding>,
    ) -> Self {
        Self {
            id: provider_execution_id(operation, &plan, &provider),
            operation,
            plan,
            provider,
            device,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionRequest {
    pub operation: ScheduledOperationId,
    pub plan: ComputeExecutionPlan,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub affinity: ResourceAffinity,
    pub memory_plan: MemoryPlan,
    pub steps: Vec<ExecutionStep>,
    pub constraints: Vec<ExecutionConstraint>,
}
impl ProviderExecutionRequest {
    pub fn from_operation(operation: &ScheduledOperation) -> Self {
        Self {
            operation: operation.id,
            plan: operation.plan.clone(),
            provider: operation.plan.provider.clone(),
            device: operation.plan.device.clone(),
            affinity: operation.plan.memory_plan.output_affinity.clone(),
            memory_plan: operation.plan.memory_plan.clone(),
            steps: operation.plan.steps.clone(),
            constraints: operation.plan.constraints.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderExecutionPhase {
    Prepare,
    Submit,
    Observe,
    Cancel,
    Complete,
    Release,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderExecutionErrorCode {
    ProviderHealthUnknown,
    ProviderInitializing,
    ProviderDegradedRejected,
    ProviderSaturated,
    ProviderDraining,
    ProviderUnavailable,
    ProviderInterrupted,
    DeviceHealthUnknown,
    DeviceSaturated,
    DeviceUnavailable,
    StaleHealthReport,
    CapabilityUnavailable,
    InvalidExecutionPlan,
    IncompatibleResourceAffinity,
    MemoryPlanRejected,
    UnsupportedOperation,
    UnsupportedDType,
    UnsupportedLayout,
    DataMovementFailed,
    MaterializationFailed,
    SubmissionFailed,
    ExecutionFailed,
    ExecutionInterrupted,
    CancellationUnsupported,
    CancellationFailed,
    ResourceExhausted,
    OutOfMemory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionDiagnostic {
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub phase: ProviderExecutionPhase,
    pub stable_reason: Option<ProviderExecutionErrorCode>,
    pub detail: Option<String>,
    pub trace_id: Option<String>,
}
impl ProviderExecutionDiagnostic {
    pub fn new(provider: ProviderBinding, phase: ProviderExecutionPhase) -> Self {
        Self {
            provider,
            device: None,
            phase,
            stable_reason: None,
            detail: None,
            trace_id: None,
        }
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
        self
    }
    pub fn with_reason(mut self, reason: ProviderExecutionErrorCode) -> Self {
        self.stable_reason = Some(reason);
        self
    }
    pub fn with_detail(mut self, detail: impl AsRef<str>) -> Self {
        self.detail = Some(redact_backend_diagnostic(detail.as_ref()));
        self
    }
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionProgress {
    pub completed_steps: u32,
    pub total_steps: u32,
    pub message: Option<String>,
}
impl ProviderExecutionProgress {
    pub fn new(completed_steps: u32, total_steps: u32) -> Self {
        Self {
            completed_steps,
            total_steps,
            message: None,
        }
    }
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionStatus {
    pub handle: ProviderExecutionHandle,
    pub state: SchedulingState,
    pub progress: Option<ProviderExecutionProgress>,
    pub diagnostics: Vec<ProviderExecutionDiagnostic>,
}
impl ProviderExecutionStatus {
    pub fn new(handle: ProviderExecutionHandle, state: SchedulingState) -> Self {
        Self {
            handle,
            state,
            progress: None,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderCancellationOutcome {
    Accepted,
    Unsupported,
    AlreadyTerminal(SchedulingState),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionResult {
    pub handle: ProviderExecutionHandle,
    pub state: SchedulingState,
    pub outputs: Vec<TensorResourceDescriptor>,
    pub diagnostics: Vec<ProviderExecutionDiagnostic>,
}
impl ProviderExecutionResult {
    pub fn completed(
        handle: ProviderExecutionHandle,
        outputs: impl IntoIterator<Item = TensorResourceDescriptor>,
    ) -> Self {
        Self {
            handle,
            state: SchedulingState::Completed,
            outputs: outputs.into_iter().collect(),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExecutionError {
    pub code: ProviderExecutionErrorCode,
    pub phase: ProviderExecutionPhase,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub message: String,
    pub diagnostics: Vec<ProviderExecutionDiagnostic>,
}
impl ProviderExecutionError {
    pub fn new(
        code: ProviderExecutionErrorCode,
        phase: ProviderExecutionPhase,
        provider: ProviderBinding,
        device: Option<DeviceBinding>,
        message: impl Into<String>,
    ) -> Self {
        let diagnostic = ProviderExecutionDiagnostic::new(provider.clone(), phase)
            .with_device(device.clone())
            .with_reason(code);
        Self {
            code,
            phase,
            provider,
            device,
            message: message.into(),
            diagnostics: vec![diagnostic],
        }
    }
    pub fn with_diagnostic(mut self, diagnostic: ProviderExecutionDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
    pub fn scheduler_code(&self) -> SchedulerErrorCode {
        match self.code {
            ProviderExecutionErrorCode::ProviderHealthUnknown => {
                SchedulerErrorCode::ProviderHealthUnknown
            }
            ProviderExecutionErrorCode::ProviderInitializing => {
                SchedulerErrorCode::ProviderInitializing
            }
            ProviderExecutionErrorCode::ProviderDegradedRejected => {
                SchedulerErrorCode::ProviderDegradedRejected
            }
            ProviderExecutionErrorCode::ProviderSaturated => SchedulerErrorCode::ProviderSaturated,
            ProviderExecutionErrorCode::ProviderDraining => SchedulerErrorCode::ProviderDraining,
            ProviderExecutionErrorCode::ProviderUnavailable => {
                SchedulerErrorCode::ProviderUnavailable
            }
            ProviderExecutionErrorCode::ProviderInterrupted => {
                SchedulerErrorCode::ProviderInterrupted
            }
            ProviderExecutionErrorCode::DeviceHealthUnknown => {
                SchedulerErrorCode::DeviceHealthUnknown
            }
            ProviderExecutionErrorCode::DeviceSaturated => SchedulerErrorCode::DeviceSaturated,
            ProviderExecutionErrorCode::DeviceUnavailable => SchedulerErrorCode::DeviceUnavailable,
            ProviderExecutionErrorCode::StaleHealthReport => SchedulerErrorCode::StaleHealthReport,
            ProviderExecutionErrorCode::InvalidExecutionPlan => {
                SchedulerErrorCode::InvalidExecutionPlan
            }
            ProviderExecutionErrorCode::IncompatibleResourceAffinity => {
                SchedulerErrorCode::ResourceAffinityConflict
            }
            ProviderExecutionErrorCode::MemoryPlanRejected
            | ProviderExecutionErrorCode::OutOfMemory
            | ProviderExecutionErrorCode::ResourceExhausted => {
                SchedulerErrorCode::MemoryPlanInvalid
            }
            ProviderExecutionErrorCode::CancellationUnsupported => {
                SchedulerErrorCode::CancellationUnsupported
            }
            ProviderExecutionErrorCode::CancellationFailed => {
                SchedulerErrorCode::CancellationFailed
            }
            ProviderExecutionErrorCode::ExecutionInterrupted => {
                SchedulerErrorCode::ExecutionInterrupted
            }
            ProviderExecutionErrorCode::CapabilityUnavailable
            | ProviderExecutionErrorCode::UnsupportedOperation
            | ProviderExecutionErrorCode::UnsupportedDType
            | ProviderExecutionErrorCode::UnsupportedLayout
            | ProviderExecutionErrorCode::DataMovementFailed
            | ProviderExecutionErrorCode::MaterializationFailed
            | ProviderExecutionErrorCode::SubmissionFailed
            | ProviderExecutionErrorCode::ExecutionFailed => SchedulerErrorCode::ExecutionFailed,
        }
    }
}
impl fmt::Display for ProviderExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "provider execution error {:?} during {:?} for provider '{}': {}",
            self.code, self.phase, self.provider, self.message
        )
    }
}
impl Error for ProviderExecutionError {}

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
            .map(Provider::health)
            .unwrap_or(ProviderHealth::Unavailable);
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

pub(crate) fn runtime_events_for_execution_plan(plan: &ComputeExecutionPlan) -> Vec<RuntimeEvent> {
    let mut events = Vec::new();
    let base = |phase, kind, message: &str| {
        RuntimeEvent::new(
            plan.trace_id.clone(),
            SpanId::new(format!("plan:{}:{kind:?}", plan.id)),
            phase,
            kind,
            message,
        )
        .with_plan(plan.id.clone())
        .with_provider(plan.provider.clone())
        .with_capability(plan.capability.clone())
    };

    events.push(base(
        RuntimeObservationPhase::Resolution,
        RuntimeEventKind::CapabilityResolution,
        "capability resolution completed",
    ));
    events.push(base(
        RuntimeObservationPhase::Resolution,
        RuntimeEventKind::ProviderSelected,
        "Provider selected for execution plan",
    ));
    if let Some(device) = &plan.device {
        events.last_mut().expect("event exists").device = Some(device.clone());
    }

    for diagnostic in &plan.diagnostics {
        match diagnostic {
            ExecutionDiagnostic::RejectedProviderCandidate { provider, reason } => {
                events.push(
                    RuntimeEvent::new(
                        plan.trace_id.clone(),
                        SpanId::new(format!("plan:{}:rejected:{provider}", plan.id)),
                        RuntimeObservationPhase::Resolution,
                        RuntimeEventKind::ProviderRejected,
                        format!("Provider rejected: {reason:?}"),
                    )
                    .with_plan(plan.id.clone())
                    .with_provider(provider.clone())
                    .with_capability(plan.capability.clone())
                    .with_diagnostic_code(RuntimeDiagnosticCode::ProviderRejected),
                );
            }
            ExecutionDiagnostic::TransferRequired { resource, .. } => {
                events.push(
                    base(
                        RuntimeObservationPhase::ResourceAffinity,
                        RuntimeEventKind::TransferRequired,
                        &format!("transfer required for resource '{resource}'"),
                    )
                    .with_diagnostic_code(RuntimeDiagnosticCode::TransferRequired),
                );
            }
            ExecutionDiagnostic::MaterializationRequired { source } => {
                events.push(
                    base(
                        RuntimeObservationPhase::ResourceAffinity,
                        RuntimeEventKind::MaterializationRequired,
                        &format!("materialization required for '{source}'"),
                    )
                    .with_diagnostic_code(RuntimeDiagnosticCode::MaterializationRequired),
                );
            }
            ExecutionDiagnostic::Memory(_) => {
                events.push(base(
                    RuntimeObservationPhase::MemoryPlanning,
                    RuntimeEventKind::MemoryPlanning,
                    "memory planning diagnostic recorded",
                ));
            }
            ExecutionDiagnostic::SelectedProvider(_)
            | ExecutionDiagnostic::SelectedDevice(_)
            | ExecutionDiagnostic::SelectedCapability(_)
            | ExecutionDiagnostic::ResolutionDecision(_)
            | ExecutionDiagnostic::PolicyDecisionReason(_) => {}
        }
    }

    for constraint in &plan.constraints {
        if matches!(constraint, ExecutionConstraint::ResourceAffinity(_)) {
            events.push(base(
                RuntimeObservationPhase::ResourceAffinity,
                RuntimeEventKind::ResourceAffinityDecision,
                "resource affinity constraint preserved",
            ));
        }
    }

    events.push(base(
        RuntimeObservationPhase::ExecutionPlanning,
        RuntimeEventKind::ExecutionPlanning,
        "execution plan created",
    ));
    events
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
