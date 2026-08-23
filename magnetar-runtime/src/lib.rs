//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod affinity;
pub mod capability;
pub mod component;
#[cfg(feature = "wasmtime-component-engine")]
pub mod component_wasmtime;
pub mod compute;
pub mod conformance;
pub mod device;
pub mod memory;
pub mod observability;
pub mod planning;
pub mod provider;
pub mod resolution;
pub mod runtime;
pub mod scheduler;

pub use affinity::{
    AffinityConstraints, AffinityError, AffinityGroupId, AffinityResolution, AffinityResource,
    ArtifactBinding, CapabilityBinding, CapabilityHealth, CapabilityStatus, DeviceAvailability,
    DeviceBinding, DeviceHealth, DeviceStatus, ExecutionContextId, ExecutionPhase, FallbackClass,
    HealthCapacityHints, HealthDiagnostic, HealthReport, HealthScope, HealthState,
    HealthTimeToLive, HealthTimestamp, OperationFamilyStatus, ProviderAdmission,
    ProviderAdmissionDecision, ProviderBinding, ProviderHealth, ProviderHealthReport,
    ProviderHealthState, ProviderInterruptionReason, ProviderLifecycleState, ProviderPressureLevel,
    ProviderReadinessState, ProviderStatusReason, ProviderStatusScope, ProviderStatusSeverity,
    ProviderStatusSnapshot, ResourceAffinity, provider_admission_from_dimensions,
};
pub use capability::{Capability, CapabilityDescriptor, CapabilityId, CapabilityVersion};
pub use component::{
    COMPONENT_ARTIFACT_SCHEMA, COMPONENT_ARTIFACT_SCHEMA_VERSION, COMPONENT_TRUST_SCHEMA,
    ComponentArtifactCache, ComponentArtifactPackage, ComponentArtifactReference,
    ComponentAuthorityEndpoint, ComponentAuthorityRequirement, ComponentCapabilityRequirement,
    ComponentContract, ComponentDefinition, ComponentDefinitionId, ComponentDefinitionState,
    ComponentDescriptor, ComponentDigest, ComponentDistributionErrorCategory,
    ComponentDistributionSource, ComponentDistributionSourceKind,
    ComponentDistributionSourceProvider, ComponentEndpoint, ComponentEngine,
    ComponentEngineCapabilities, ComponentEngineInstance, ComponentError,
    ComponentExportDescription, ComponentImportRequirement, ComponentInstance, ComponentInstanceId,
    ComponentInstanceState, ComponentInterfaceShape, ComponentInterruptionReason,
    ComponentInvocation, ComponentInvocationResult, ComponentLinkPlan, ComponentManager,
    ComponentManifest, ComponentMetadata, ComponentObservation, ComponentObservationKind,
    ComponentProvenance, ComponentPublisher, ComponentResourceLimits, ComponentSignature,
    ComponentSource, ComponentTrapKind, ComponentTrustDecision, ComponentTrustStatus,
    ComponentTrustStore, ComponentValue, InferenceArtifactKind, InferenceArtifactReference,
    InferenceArtifactRegistry, InferenceCacheKind, InferenceCacheRegistry, InferenceCacheScope,
    InferenceSessionId, MAGNETAR_RUNTIME_VERSION, MockComponentEngine, PreparedComponent,
    WitInterface,
};
#[cfg(feature = "wasmtime-component-engine")]
pub use component_wasmtime::WasmtimeComponentEngine;
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
pub use conformance::{
    PROVIDER_CONFORMANCE_SUITE_VERSION, ProviderConformanceConfig, ProviderConformanceProfile,
    ProviderConformanceReport, ProviderConformanceSuite, ProviderConformanceTarget,
    ProviderConformanceTargetKind, ProviderConformanceTestResult, ProviderConformanceTestStatus,
    provider_conformance_profile_ids, provider_conformance_report_json,
};
pub use device::{Device, DeviceDescriptor, DeviceId, DeviceMetadata, DeviceType};
pub use memory::{
    MemoryAdmissionDecision, MemoryAdmissionRequest, MemoryAllocation, MemoryAllocationClass,
    MemoryAllocationId, MemoryAllocationLifetime, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryAllocationState, MemoryArena, MemoryArenaGrowthPolicy, MemoryArenaId, MemoryArenaOwner,
    MemoryArenaShrinkPolicy, MemoryDTypeRelation, MemoryError, MemoryFeasibility, MemoryManager,
    MemoryManagerConfig, MemoryObservation, MemoryObservationKind, MemoryPlacement,
    MemoryPressureLevel, MemoryPressureSnapshot, PendingMemoryAllocation, StagingFeasibility,
    TensorResidency, ZeroCopyFeasibility,
};
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
    PROVIDER_ABI_FACTORY_SYMBOL_V1, PROVIDER_ABI_MAJOR_VERSION, PROVIDER_ABI_MINOR_VERSION,
    PROVIDER_API_VERSION, Provider, ProviderAbiDescriptor, ProviderAbiErrorCode,
    ProviderAbiExecutionBehavior, ProviderAbiFeature, ProviderAbiFunctionTable, ProviderAbiHandle,
    ProviderAbiHandleDescriptor, ProviderAbiHandleKind, ProviderAbiLoadingLifecycle,
    ProviderAbiMemoryOwner, ProviderAbiMemoryRule, ProviderAbiOwnershipRules,
    ProviderAbiRetentionPolicy, ProviderAbiThreadingModel, ProviderAbiUnloadPolicy,
    ProviderAbiVersion, ProviderDescriptor, ProviderError, ProviderExecutionApi, ProviderLoader,
    ProviderLoadingMode, ProviderLoadingPolicy, ProviderMetadata, ProviderRegistry,
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
    runtime_event_for_provider_health, runtime_events_for_provider_status,
    runtime_metrics_for_execution_plan,
};

#[cfg(test)]
mod tests;
