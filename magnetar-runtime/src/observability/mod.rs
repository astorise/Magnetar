mod exporter;
use crate::compute::redact_backend_diagnostic;
use crate::runtime::next_runtime_observation_sequence;
pub use exporter::{
    CustomEventRecord, CustomLogRecord, CustomMetricKind, CustomMetricRecord,
    DeviceMetricsSnapshot, ExporterRuntimeStatus, LogSeverity, MetricTags,
    OBSERVABILITY_CAPABILITY_VERSION, OBSERVABILITY_EMIT_INTERFACE, OBSERVABILITY_READER_INTERFACE,
    OBSERVABILITY_STREAM_INTERFACE, ObservabilityComponentDescriptor, ObservabilityComponentRole,
    ObservabilityComponentState, ObservabilityError, ObservabilityErrorCode,
    ObservabilityMetricsSnapshot, ObservabilityPolicy, ObservabilityPolicyField,
    ObservabilityPriority, ObservabilitySinkDependency, ObservabilitySnapshot, ObservationBatch,
    ObservationBus, ObservationCategory, ObservationFilter, ObservationOverflowPolicy,
    ObservationRecord, ObservationStream, ProviderMetricsSnapshot, RuntimeMetricsSnapshot,
    SchedulerMetricsSnapshot, jaeger_exporter_component, jsonl_exporter_component,
    observability_emit_capability, observability_emit_wit, observability_reader_capability,
    observability_reader_wit, observability_stream_capability, observability_stream_wit,
    opentelemetry_exporter_component, prometheus_exposer_component, prometheus_snapshot_lines,
};

use crate::*;
use std::{collections::BTreeSet, fmt};
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CorrelationId(String);
impl CorrelationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TraceId(String);
impl TraceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpanId(String);
impl SpanId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeTrace {
    pub trace_id: TraceId,
    pub events: Vec<RuntimeEvent>,
}
impl RuntimeTrace {
    pub fn new(trace_id: TraceId) -> Self {
        Self {
            trace_id,
            events: Vec::new(),
        }
    }
    pub fn push(&mut self, event: RuntimeEvent) {
        if event.trace_id == self.trace_id {
            self.events.push(event);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeMetricKind {
    QueueLatency,
    PlanningLatency,
    ExecutionLatency,
    MemoryUsageEstimate,
    TransferVolume,
    MaterializationCount,
    ProviderUtilization,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeMetric {
    pub kind: RuntimeMetricKind,
    pub value: u64,
    pub unit: &'static str,
    pub trace_id: Option<TraceId>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
}
impl RuntimeMetric {
    pub fn new(kind: RuntimeMetricKind, value: u64, unit: &'static str) -> Self {
        Self {
            kind,
            value,
            unit,
            trace_id: None,
            provider: None,
            device: None,
        }
    }
    pub fn with_trace(mut self, trace_id: TraceId) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
    pub fn with_provider(mut self, provider: ProviderBinding) -> Self {
        self.provider = Some(provider);
        self
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeDiagnostic {
    pub code: RuntimeDiagnosticCode,
    pub message: String,
    pub trace_id: Option<TraceId>,
    pub correlation_id: Option<CorrelationId>,
    pub execution_plan: Option<ExecutionPlanId>,
    pub scheduled_operation: Option<ScheduledOperationId>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
}
impl RuntimeDiagnostic {
    pub fn new(code: RuntimeDiagnosticCode, message: impl AsRef<str>) -> Self {
        Self {
            code,
            message: redact_backend_diagnostic(message.as_ref()),
            trace_id: None,
            correlation_id: None,
            execution_plan: None,
            scheduled_operation: None,
            provider: None,
            device: None,
        }
    }
    pub fn with_trace(mut self, trace_id: TraceId) -> Self {
        self.trace_id = Some(trace_id);
        self
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservabilityExporterDescriptor {
    pub component: ComponentMetadata,
    pub input_contract: WitInterface,
    pub accepted_events: BTreeSet<RuntimeEventKind>,
    pub sink: ObservabilitySink,
}
impl ObservabilityExporterDescriptor {
    pub fn new(component: ComponentMetadata, sink: ObservabilitySink) -> Self {
        Self {
            component,
            input_contract: runtime_observability_wit(),
            accepted_events: BTreeSet::new(),
            sink,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservabilitySink {
    OpenTelemetry,
    Prometheus,
    Jaeger,
    Custom(String),
}

pub fn runtime_observability_wit() -> WitInterface {
    WitInterface::new("magnetar:runtime/observability", "1.0.0")
}
