//! Runtime observation events.
//!
//! The structured event a Runtime step, a Provider execution or a plan
//! produces: phase, kind, identifiers of the plan, Provider, Capability and
//! Device it concerns, and a redacted message. Carried by execution plans
//! and by Provider execution results, so defined below the code that emits
//! them.

use crate::binding::*;
use crate::compute_types::redact_backend_diagnostic;
use crate::execution::*;
use crate::provider_status::*;
use crate::trace_id::*;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeObservationPhase {
    Resolution,
    ResourceAffinity,
    MemoryPlanning,
    ExecutionPlanning,
    Scheduling,
    ProviderExecution,
    Health,
    Diagnostic,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeEventKind {
    CapabilityResolution,
    ProviderSelected,
    ProviderRejected,
    ResourceAffinityDecision,
    TransferRequired,
    MaterializationRequired,
    MemoryPlanning,
    ExecutionPlanning,
    Scheduled,
    SchedulerBackpressure,
    ProviderSubmission,
    ExecutionStarted,
    ExecutionCompleted,
    ExecutionCancelled,
    ExecutionInterrupted,
    ProviderHealthChanged,
    DeviceHealthChanged,
    ProviderLifecycleChanged,
    ProviderReadinessChanged,
    ProviderPressureChanged,
    ProviderAdmissionChanged,
    ProviderStatusStale,
    ProviderDrainStarted,
    ProviderDrainCompleted,
    DeviceStatusChanged,
    CapabilityStatusChanged,
    DiagnosticEmitted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeEvent {
    pub sequence: u64,
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub correlation_id: Option<CorrelationId>,
    pub phase: RuntimeObservationPhase,
    pub kind: RuntimeEventKind,
    pub execution_plan: Option<ExecutionPlanId>,
    pub scheduled_operation: Option<ScheduledOperationId>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub capability: Option<CapabilityBinding>,
    pub diagnostic_code: Option<RuntimeDiagnosticCode>,
    pub message: String,
}

impl RuntimeEvent {
    pub fn new(
        trace_id: TraceId,
        span_id: SpanId,
        phase: RuntimeObservationPhase,
        kind: RuntimeEventKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            sequence: next_runtime_observation_sequence(),
            trace_id,
            span_id,
            correlation_id: None,
            phase,
            kind,
            execution_plan: None,
            scheduled_operation: None,
            provider: None,
            device: None,
            capability: None,
            diagnostic_code: None,
            message: redact_backend_diagnostic(&message.into()),
        }
    }
    pub fn with_correlation(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
    pub fn with_plan(mut self, plan: ExecutionPlanId) -> Self {
        self.execution_plan = Some(plan);
        self
    }
    pub fn with_operation(mut self, operation: ScheduledOperationId) -> Self {
        self.scheduled_operation = Some(operation);
        self
    }
    pub fn with_provider(mut self, provider: ProviderBinding) -> Self {
        self.provider = Some(provider);
        self
    }
    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }
    pub fn with_capability(mut self, capability: CapabilityBinding) -> Self {
        self.capability = Some(capability);
        self
    }
    pub fn with_diagnostic_code(mut self, code: RuntimeDiagnosticCode) -> Self {
        self.diagnostic_code = Some(code);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeDiagnosticCode {
    CapabilityResolutionFailed,
    ProviderRejected,
    ResourceAffinityConflict,
    TransferRequired,
    MaterializationRequired,
    MemoryPlanningFailed,
    ExecutionPlanningFailed,
    SchedulerBackpressure,
    ProviderUnavailable,
    DeviceUnavailable,
    ExecutionFailed,
    ExecutionInterrupted,
    ExecutionCancelled,
    ProviderHealthChanged,
    DeviceHealthChanged,
    ProviderLifecycleChanged,
    ProviderReadinessChanged,
    ProviderPressureChanged,
    ProviderAdmissionChanged,
    ProviderStatusStale,
    ProviderDrainStarted,
    ProviderDrainCompleted,
    DeviceStatusChanged,
    CapabilityStatusChanged,
}

pub(crate) static NEXT_RUNTIME_OBSERVATION_SEQUENCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

pub(crate) fn next_runtime_observation_sequence() -> u64 {
    NEXT_RUNTIME_OBSERVATION_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
