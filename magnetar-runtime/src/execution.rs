//! Execution plans and Provider execution requests.
//!
//! The data a Runtime hands to a Provider and gets back: the execution plan
//! and the memory plan built for a Capability, the scheduled operation, the
//! request submitted to a Provider, its handle, progress, status and result,
//! and the structured errors and diagnostics around them. Defined below the
//! code that plans and schedules (`planning`, `scheduler`) so that a
//! Provider can name them without depending on it.

use crate::affinity::*;
use crate::capability::*;
use crate::compute::*;
use crate::provider_status::{
    ProviderExecutionDiagnostic, ScheduledOperationId, SchedulerErrorCode,
};
use crate::resolution::*;
use crate::runtime_event::*;
use crate::trace_id::*;
use std::{error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionPlanId(String);

impl ExecutionPlanId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExecutionPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionPhase {
    Validation,
    Resolution,
    Planning,
    DataMovement,
    Materialization,
    MemoryAllocation,
    ProviderSubmission,
    Execution,
    Completion,
    Cancellation,
    Interruption,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionClassification {
    Transparent,
    Restartable,
    ProviderPinned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionStepKind {
    ValidateGraph,
    ResolveProvider,
    ResolveDevice,
    ValidateCapabilityVersion,
    ValidateOperationSchema,
    ValidateDType,
    ValidateLayout,
    ValidatePrecisionPolicy,
    ValidateDeterminism,
    BindInputResource,
    BindOutputResource,
    PreserveProviderPinnedAffinity,
    PreserveDeviceBoundAffinity,
    PreserveAffinityGroup,
    RejectIncompatibleResourceChain,
    Upload,
    Download,
    Copy,
    Transfer,
    Materialize,
    AllocateMemory,
    ValidateMemory,
    SubmitToProvider,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionStep {
    pub id: String,
    pub phase: ComputeExecutionPhase,
    pub kind: ExecutionStepKind,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub dependencies: Vec<String>,
}

impl ExecutionStep {
    pub fn new(
        id: impl Into<String>,
        phase: ComputeExecutionPhase,
        kind: ExecutionStepKind,
        provider: ProviderBinding,
    ) -> Self {
        Self {
            id: id.into(),
            phase,
            kind,
            provider,
            device: None,
            dependencies: Vec::new(),
        }
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
        self
    }
    pub fn depends_on(mut self, dependency: impl Into<String>) -> Self {
        self.dependencies.push(dependency.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionInput {
    pub id: ComputeInputId,
    pub descriptor: TensorDescriptor,
    pub resource: Option<TensorResourceId>,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionConstraint {
    ResolutionPolicy(ResolutionPolicyId),
    CapabilityVersion(CapabilityBinding),
    Provider(ProviderBinding),
    Device(DeviceBinding),
    ResourceAffinity(ResourceAffinity),
    AffinityGroup(AffinityGroupId),
    OperationSchema(ComputeOperationId),
    DType(ComputeDType),
    Layout(ComputeLayout),
    PrecisionPolicy(ComputePrecision),
    DeterministicBehavior,
    MemoryRequirement(String),
    ExplicitTransferRequired(TensorResourceId),
    ExplicitTransferRequirement(String),
    ExplicitMaterializationRequired(String),
    NoHiddenCpuStaging,
    NoImplicitProviderMigration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionDiagnostic {
    SelectedProvider(ProviderBinding),
    SelectedDevice(DeviceBinding),
    SelectedCapability(CapabilityBinding),
    ResolutionDecision(ResolutionDecision),
    RejectedProviderCandidate {
        provider: ProviderBinding,
        reason: ResolutionRejectionReason,
    },
    Memory(MemoryPlanningDiagnostic),
    TransferRequired {
        resource: TensorResourceId,
        from: ResourceAffinity,
        to: ResourceAffinity,
    },
    MaterializationRequired {
        source: String,
    },
    PolicyDecisionReason(ResolutionDecisionReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionPlan {
    pub id: ExecutionPlanId,
    pub trace_id: TraceId,
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub capability: CapabilityBinding,
    pub policy: ResolutionPolicyId,
    pub classification: ComputeExecutionClassification,
    pub inputs: Vec<ExecutionInput>,
    pub outputs: Vec<ExecutionOutput>,
    pub constraints: Vec<ExecutionConstraint>,
    pub steps: Vec<ExecutionStep>,
    pub memory_plan: MemoryPlan,
    pub diagnostics: Vec<ExecutionDiagnostic>,
    pub(crate) validated: bool,
}

impl ComputeExecutionPlan {
    pub fn is_validated(&self) -> bool {
        self.validated
    }
    pub fn trace_id(&self) -> &TraceId {
        &self.trace_id
    }
    pub fn observations(&self) -> Vec<RuntimeEvent> {
        runtime_events_for_execution_plan(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryRegionKind {
    GraphInput,
    GraphOutput,
    Intermediate,
    Temporary,
    Materialization,
    Transfer,
    HostStaging,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRequirement {
    pub id: String,
    pub region: MemoryRegionKind,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reusable: bool,
}

impl MemoryRequirement {
    pub fn new(
        id: impl Into<String>,
        region: MemoryRegionKind,
        byte_size: u64,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id: id.into(),
            region,
            byte_size,
            affinity,
            reusable: false,
        }
    }
    pub const fn reusable(mut self) -> Self {
        self.reusable = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLifetime {
    pub id: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
}

impl TensorLifetime {
    pub fn overlaps(&self, other: &Self) -> bool {
        self.first_step <= other.last_step && other.first_step <= self.last_step
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferLifetime {
    pub id: String,
    pub source: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reuses: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDecision {
    Allocate { requirement: String },
    Reuse { requirement: String, buffer: String },
    PreservePinnedResource { resource: TensorResourceId },
    RequireMaterialization { requirement: String },
    RequireTransfer { requirement: String },
    AccountHostStaging { requirement: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDiagnostic {
    EstimatedRequirement {
        requirement: String,
        bytes: u64,
    },
    PeakBytes {
        bytes: u64,
    },
    ProviderLimit {
        provider: ProviderBinding,
        max_bytes: u64,
    },
    DeviceLimit {
        device: DeviceBinding,
        max_bytes: u64,
    },
    MaterializationCost {
        bytes: u64,
    },
    TransferCost {
        bytes: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPlan {
    pub provider: ProviderBinding,
    pub graph: Option<ComputeGraphId>,
    pub requirements: Vec<MemoryRequirement>,
    pub tensor_lifetimes: Vec<TensorLifetime>,
    pub buffer_lifetimes: Vec<BufferLifetime>,
    pub pressure: MemoryPressureReport,
    pub decisions: Vec<MemoryPlanningDecision>,
    pub diagnostics: Vec<MemoryPlanningDiagnostic>,
    pub output_affinity: ResourceAffinity,
}

impl MemoryPlan {
    pub(crate) fn new(
        provider: ProviderBinding,
        graph: Option<ComputeGraphId>,
        execution_context: ExecutionContextId,
    ) -> Self {
        let output_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(execution_context);
        Self {
            provider: provider.clone(),
            graph,
            requirements: Vec::new(),
            tensor_lifetimes: Vec::new(),
            buffer_lifetimes: Vec::new(),
            pressure: MemoryPressureReport {
                selected_provider: Some(provider),
                ..MemoryPressureReport::default()
            },
            decisions: Vec::new(),
            diagnostics: Vec::new(),
            output_affinity,
        }
    }
    pub(crate) fn add_requirement(
        &mut self,
        requirement: MemoryRequirement,
    ) -> Result<(), MemoryPlanningError> {
        self.pressure.estimated_required_bytes = self
            .pressure
            .estimated_required_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "total memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.pressure.estimated_peak_bytes = self
            .pressure
            .estimated_peak_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "peak memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.diagnostics
            .push(MemoryPlanningDiagnostic::EstimatedRequirement {
                requirement: requirement.id.clone(),
                bytes: requirement.byte_size,
            });
        self.diagnostics.push(MemoryPlanningDiagnostic::PeakBytes {
            bytes: self.pressure.estimated_peak_bytes,
        });
        self.requirements.push(requirement);
        Ok(())
    }
    pub(crate) fn find_reusable_buffer(&self, lifetime: &TensorLifetime) -> Option<String> {
        self.buffer_lifetimes
            .iter()
            .find(|buffer| {
                buffer.byte_size >= lifetime.byte_size
                    && buffer.last_step < lifetime.first_step
                    && buffer.affinity.validate_with(&lifetime.affinity).is_ok()
            })
            .map(|buffer| buffer.id.clone())
    }
}

pub(crate) fn memory_bytes(
    descriptor: &TensorDescriptor,
    report: &MemoryPressureReport,
) -> Result<u64, MemoryPlanningError> {
    descriptor
        .byte_size()
        .map_err(|error| MemoryPlanningError::SizeOverflow {
            reason: error.to_string(),
            report: report.clone(),
        })
}

pub(crate) fn provider_execution_id(
    operation: ScheduledOperationId,
    plan: &ExecutionPlanId,
    provider: &ProviderBinding,
) -> ProviderExecutionId {
    ProviderExecutionId::new(format!("provider-execution:{operation}:{plan}:{provider}"))
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
