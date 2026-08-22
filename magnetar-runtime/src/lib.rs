//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod affinity;
pub mod capability;
pub mod component;
pub mod compute;
pub mod device;
pub mod observability;
pub mod planning;
pub mod provider;
pub mod resolution;
pub mod runtime;
pub mod scheduler;

pub use affinity::{
    AffinityConstraints, AffinityError, AffinityGroupId, AffinityResolution, AffinityResource,
    ArtifactBinding, CapabilityBinding, CapabilityHealth, DeviceAvailability, DeviceBinding,
    DeviceHealth, ExecutionContextId, ExecutionPhase, FallbackClass, HealthCapacityHints,
    HealthDiagnostic, HealthReport, HealthScope, HealthState, HealthTimeToLive, HealthTimestamp,
    ProviderBinding, ProviderHealth, ProviderHealthReport, ResourceAffinity,
};
pub use capability::{Capability, CapabilityDescriptor, CapabilityId, CapabilityVersion};
pub use component::{
    Component, ComponentDescriptor, ComponentError, ComponentManager, ComponentMetadata,
    ComponentState, WitInterface,
};
pub use compute::{
    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION, COMPUTE_WIT_INTERFACE, COMPUTE_WIT_PACKAGE,
    ComputeAttributeKind, ComputeCapabilitySupport, ComputeDType, ComputeDataMovementDescriptor,
    ComputeDataMovementKind, ComputeDataMovementSource, ComputeDataMovementSupport,
    ComputeDiagnostic, ComputeError, ComputeErrorCode, ComputeErrorPhase, ComputeErrorSeverity,
    ComputeGraph, ComputeGraphId, ComputeGraphValidationReport, ComputeInput, ComputeInputId,
    ComputeInputValue, ComputeLayout, ComputeNode, ComputeNodeId, ComputeNodeOutput,
    ComputeOperationAttribute, ComputeOperationAttributeRule, ComputeOperationDescriptor,
    ComputeOperationFamily, ComputeOperationFamilyMetadata, ComputeOperationId,
    ComputeOperationInputRule, ComputeOperationOutputRule, ComputeOperationRequest,
    ComputeOperationSchema, ComputeOperationSupport, ComputeOperationValidationResult,
    ComputeOutput, ComputeOutputDTypeRule, ComputeOutputId, ComputeOutputShapeRule,
    ComputePlacementIntent, ComputePrecision, ComputeSubmission, ComputeSubmissionState,
    ComputeValidationError, ComputeValueRef, DTypeDescriptor, DTypeSupport, DataMovementSupport,
    DeviceComputeSupport, HostBufferDescriptor, HostBufferEncoding, HostStagingPolicy,
    LayoutDescriptor, LayoutSupport, OperationFamilySupport, OperationSchemaSupport,
    PrecisionSupport, ProviderComputeAdvertisement, RecoveryHint, ShapeDescriptor,
    ShapeLimitSupport, TensorDescriptor, TensorDescriptorLimits, TensorResourceDescriptor,
    TensorResourceId, TensorViewSource, ViewDescriptor, compute_capability,
    initial_compute_operation_schemas,
};
pub use device::{Device, DeviceDescriptor, DeviceId, DeviceMetadata, DeviceType};
pub use observability::{
    CorrelationId, CustomEventRecord, CustomLogRecord, CustomMetricKind, CustomMetricRecord,
    DeviceMetricsSnapshot, ExporterRuntimeStatus, LogSeverity, MetricTags,
    OBSERVABILITY_CAPABILITY_VERSION, OBSERVABILITY_EMIT_INTERFACE, OBSERVABILITY_READER_INTERFACE,
    OBSERVABILITY_STREAM_INTERFACE, ObservabilityComponentDescriptor, ObservabilityComponentRole,
    ObservabilityComponentState, ObservabilityError, ObservabilityErrorCode,
    ObservabilityExporterDescriptor, ObservabilityMetricsSnapshot, ObservabilityPolicy,
    ObservabilityPolicyField, ObservabilityPriority, ObservabilitySink,
    ObservabilitySinkDependency, ObservabilitySnapshot, ObservationBatch, ObservationBus,
    ObservationCategory, ObservationFilter, ObservationOverflowPolicy, ObservationRecord,
    ObservationStream, ProviderMetricsSnapshot, RuntimeDiagnostic, RuntimeDiagnosticCode,
    RuntimeEvent, RuntimeEventKind, RuntimeMetric, RuntimeMetricKind, RuntimeMetricsSnapshot,
    RuntimeObservationPhase, RuntimeTrace, SchedulerMetricsSnapshot, SpanId, TraceId,
    jaeger_exporter_component, jsonl_exporter_component, observability_emit_capability,
    observability_emit_wit, observability_reader_capability, observability_reader_wit,
    observability_stream_capability, observability_stream_wit, opentelemetry_exporter_component,
    prometheus_exposer_component, prometheus_snapshot_lines, runtime_observability_wit,
};
pub use planning::{
    BufferLifetime, ComputeExecutionClassification, ComputeExecutionPhase, ComputeExecutionPlan,
    ComputePlanningError, ExecutionConstraint, ExecutionDiagnostic, ExecutionInput,
    ExecutionOutput, ExecutionPlanId, ExecutionStep, ExecutionStepKind, MemoryPlan,
    MemoryPlanningDecision, MemoryPlanningDiagnostic, MemoryPlanningError, MemoryPressureReport,
    MemoryRegionKind, MemoryRequirement, TensorLifetime,
};
pub use provider::{
    PROVIDER_API_VERSION, Provider, ProviderDescriptor, ProviderError, ProviderExecutionApi,
    ProviderLoader, ProviderMetadata, ProviderRegistry,
};
pub use resolution::{
    BuiltInResolutionPolicy, ResolutionCandidate, ResolutionCandidateRejection, ResolutionContext,
    ResolutionDecision, ResolutionDecisionReason, ResolutionPolicy, ResolutionPolicyId,
    ResolutionRejectionReason,
};
pub use runtime::{ExecutionContext, Runtime, RuntimeBuilder, RuntimeConfig};
pub use scheduler::{
    ProviderCancellationOutcome, ProviderExecutionDiagnostic, ProviderExecutionError,
    ProviderExecutionErrorCode, ProviderExecutionHandle, ProviderExecutionId,
    ProviderExecutionPhase, ProviderExecutionProgress, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus, ScheduledOperation, ScheduledOperationId,
    ScheduledOperationResult, Scheduler, SchedulerError, SchedulerErrorCode, SchedulerQueue,
    SchedulingDiagnostic, SchedulingPolicy, SchedulingState, runtime_event_for_device_health,
    runtime_event_for_provider_health, runtime_metrics_for_execution_plan,
};

#[cfg(test)]
mod tests;
