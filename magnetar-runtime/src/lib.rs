//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod adapter;
pub mod affinity;
pub mod batching;
pub mod capability;
pub mod component;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub mod component_wasmtime;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub mod component_web;
pub mod compute;
pub mod conformance;
pub mod device;
pub mod execution_graph;
pub mod generation;
pub mod kv_cache;
pub mod memory;
pub mod model;
pub mod model_instance;
pub mod model_loading;
pub mod observability;
pub mod operator;
pub mod planning;
pub mod prefix_cache;
pub mod provider;
pub mod resolution;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod session;
pub mod tokenizer;

pub use adapter::{
    AdapterActivationPolicy, AdapterActivationRequest, AdapterActivationScope,
    AdapterArchitectureCompatibility, AdapterArtifact, AdapterArtifactId,
    AdapterBaseModelCompatibility, AdapterBatchCompatibility, AdapterCacheCompatibility,
    AdapterCompositionPolicy, AdapterDeactivationRequest, AdapterDigest, AdapterError,
    AdapterGenerationContext, AdapterLayerSelector, AdapterLifecycleState, AdapterLoadingRequest,
    AdapterLoadingRequestId, AdapterMemoryEstimate, AdapterMergePolicy, AdapterMergeRecord,
    AdapterMethod, AdapterName, AdapterObservation, AdapterObservationKind, AdapterResidency,
    AdapterResidencyId, AdapterResidencyLocation, AdapterResidencyPolicy, AdapterRevision,
    AdapterSessionPolicy, AdapterSetId, AdapterSharingPolicy, AdapterTargetModule,
    AdapterTargetModuleRole, AdapterTargetUsage, AdapterTrustStatus, AdapterUnloadPolicy,
    ProviderAdapterCapabilities, ProviderAdapterResource, activation_uses_adapter,
    adapter_memory_feasibility, apply_adapter_deactivation, validate_adapter_activation,
    validate_adapter_compatibility,
};
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
pub use batching::{
    BatchAdmission, BatchCompatibility, BatchExecutionStep, BatchId, BatchMemoryEstimate,
    BatchObservation, BatchObservationKind, BatchPhase, BatchSchedulingMode, BatchSlot,
    BatchSlotId, BatchedOperationState, BatchingError, BatchingErrorCode, BatchingPolicy,
    ContinuousBatch, ContinuousBatchingManager, portable_batch_dtype,
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
    ComponentEngineCapabilities, ComponentEngineFeature, ComponentEngineInstance,
    ComponentEngineProfile, ComponentEngineRequirements, ComponentError,
    ComponentExportDescription, ComponentImportRequirement, ComponentInstance, ComponentInstanceId,
    ComponentInstanceState, ComponentInterfaceShape, ComponentInterruptionReason,
    ComponentInvocation, ComponentInvocationResult, ComponentLinkPlan, ComponentManager,
    ComponentManifest, ComponentMetadata, ComponentObservation, ComponentObservationKind,
    ComponentProvenance, ComponentPublisher, ComponentResourceLimits, ComponentSignature,
    ComponentSource, ComponentTrapKind, ComponentTrustDecision, ComponentTrustStatus,
    ComponentTrustStore, ComponentValue, InferenceArtifactKind, InferenceArtifactReference,
    InferenceArtifactRegistry, InferenceCacheKind, InferenceCacheRegistry, InferenceCacheScope,
    MAGNETAR_RUNTIME_VERSION, MockComponentEngine, PreparedComponent, WitInterface,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub use component_wasmtime::WasmtimeComponentEngine;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub use component_web::WebComponentEngine;
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
pub use execution_graph::{
    ExecutionGraph, ExecutionGraphId, ExecutionGraphPhase, ExecutionGraphPlan,
    ExecutionGraphProducer, ExecutionGraphVersion, ExecutionNode, ExecutionNodeId,
    GraphAdapterMetadata, GraphError, GraphKvCacheBehavior, GraphKvCacheMetadata,
    GraphModelCompatibility, GraphObservation, GraphObservationKind, GraphPlanStep,
    GraphPlanningPolicy, GraphPrefixCacheMetadata, GraphValidationReport, TensorAliasing,
    TensorEdge, TensorEdgeId, TensorLifetimeHint, TensorMutability, TensorResidencyConstraint,
    default_graph_catalog, execute_graph_boundary, plan_execution_graph, validate_execution_graph,
};
pub use generation::{
    CancellationMetadata, DecodeStepInput, DecodeStepOutput, EosMode, EosOutputPolicy, EosPolicy,
    FinishReason, GenerationDiagnostic, GenerationDiagnosticKind, GenerationError, GenerationEvent,
    GenerationEventKind, GenerationMemoryEstimate, GenerationModelReference, GenerationOutput,
    GenerationParameters, GenerationPriority, GenerationRequest, GenerationRequestId,
    GenerationTokenizerReference, LogitsProcessorReference, PrefillState, Probability,
    StopConditions, StreamingMode, decode_step, decode_step_from_sampling,
    finish_reason_from_provider_error, memory_admission, prefill, prepare_stop_sequences,
    stop_reason_for, streaming_text_chunk, token_stream_events,
};
pub use kv_cache::{
    KvCache, KvCacheCompatibility, KvCacheError, KvCacheId, KvCacheLayoutFormat,
    KvCacheLayoutMetadata, KvCacheLifecycleState, KvCacheManager, KvCacheObservation,
    KvCacheObservationKind, KvCachePageMetadata, KvCachePolicy, KvCacheQuantization,
    KvCacheResidency, KvCacheRetentionPolicy, KvCacheScope, KvCacheSharingPolicy,
    PrefixFingerprint, ProviderKvCacheResource,
};
pub use memory::{
    MemoryAdmissionDecision, MemoryAdmissionRequest, MemoryAllocation, MemoryAllocationClass,
    MemoryAllocationId, MemoryAllocationLifetime, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryAllocationState, MemoryArena, MemoryArenaGrowthPolicy, MemoryArenaId, MemoryArenaOwner,
    MemoryArenaShrinkPolicy, MemoryDTypeRelation, MemoryError, MemoryFeasibility, MemoryManager,
    MemoryManagerConfig, MemoryObservation, MemoryObservationKind, MemoryPlacement,
    MemoryPressureLevel, MemoryPressureSnapshot, PendingMemoryAllocation, StagingFeasibility,
    TensorResidency, ZeroCopyFeasibility,
};
pub use model::{
    MODEL_ARTIFACT_SCHEMA, MODEL_ARTIFACT_SCHEMA_VERSION, ModelAdapterCompatibility,
    ModelArchitecture, ModelArtifactError, ModelArtifactId, ModelArtifactKind,
    ModelArtifactObserver, ModelArtifactPart, ModelArtifactRecord, ModelArtifactSource,
    ModelComponentRequirement, ModelDType, ModelDigest, ModelGenerationDefaults,
    ModelLicenseMetadata, ModelManifest, ModelName, ModelObservation, ModelObservationKind,
    ModelProvenance, ModelQuantization, ModelQuantizationFormat, ModelResidencyPlan, ModelRevision,
    ModelShard, ModelShardId, ModelSignature, ModelSourceIdentity, ModelTensorMetadata,
    ModelTrustDecision, ModelTrustStatus, ModelTrustStore, ModelVariant,
};
pub use model_instance::{
    ModelInstance, ModelInstanceAdapterState, ModelInstanceCreationChecks, ModelInstanceDefinition,
    ModelInstanceError, ModelInstanceId, ModelInstanceInvalidationReport,
    ModelInstanceLifecycleState, ModelInstanceManager, ModelInstanceMutationKind,
    ModelInstanceObservation, ModelInstanceObservationKind, ModelInstancePlacement,
    ModelInstancePolicy, ModelInstanceReadiness, ModelInstanceReadinessChecks,
    ModelInstanceReloadRequest, ModelInstanceResourceBindings, ModelInstanceSharingContext,
    ModelInstanceSharingPolicy, ModelInstanceStatus, ModelInstanceSuspensionReason,
    ModelInstanceUnloadPolicy, ModelInstanceUnloadReport, ModelInstanceUsage,
    ModelInstanceWarmupPlan, ModelInstanceWarmupPolicy, ModelInstanceWarmupStep,
    ProviderModelResource, readiness_error, readiness_for_lifecycle,
};
pub use model_loading::{
    ArtifactResidencyPlan, LoadedModelContext, ModelArchitectureImplementation,
    ModelArchitectureImplementationKind, ModelLoadingCachePolicy, ModelLoadingCoordinator,
    ModelLoadingError, ModelLoadingErrorCode, ModelLoadingObservation, ModelLoadingObservationKind,
    ModelLoadingPhase, ModelLoadingRequest, ModelLoadingRequestId, ModelLoadingResidencyPlan,
    ModelLoadingResidencyPolicy, ModelLoadingState, ModelLoadingTargetUsage,
    ModelPlacementPreference, ModelQuantizationHandling, ModelQuantizationPolicy,
    ModelReloadRequest, ModelResidencyId, ModelResidencyLocation, ModelShardingPolicy,
    ModelStorageHandling, ModelUnloadPolicy, ModelUnloadRequest, allocation_released,
    compute_dtype_supported, invalidates_kv_cache_on_unload, reload_is_new_loading_process,
    storage_to_compute_relation,
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
pub use operator::{
    OPERATOR_CATALOG_VERSION, OPERATOR_NAMESPACE, OperatorAttributeKind, OperatorAttributeRule,
    OperatorAttributeSchema, OperatorAttributeValue, OperatorCatalog, OperatorDTypeContract,
    OperatorDeterminism, OperatorError, OperatorFamily, OperatorId, OperatorLayoutContract,
    OperatorMemoryBehavior, OperatorObservation, OperatorObservationKind, OperatorSpec, ShapeRule,
    TensorLayoutKind, TensorRole, initial_operator_catalog, layout_kind,
    validate_affinity_compatibility,
};
pub use planning::{
    BufferLifetime, ComputeExecutionClassification, ComputeExecutionPhase, ComputeExecutionPlan,
    ComputePlanningError, ExecutionConstraint, ExecutionDiagnostic, ExecutionInput,
    ExecutionOutput, ExecutionPlanId, ExecutionStep, ExecutionStepKind, MemoryPlan,
    MemoryPlanningDecision, MemoryPlanningDiagnostic, MemoryPlanningError, MemoryPressureReport,
    MemoryRegionKind, MemoryRequirement, TensorLifetime,
};
pub use prefix_cache::{
    PrefixCacheBackingKvCache, PrefixCacheCompatibility, PrefixCacheEntry, PrefixCacheEntryId,
    PrefixCacheError, PrefixCacheFingerprint, PrefixCacheLifecycleState, PrefixCacheLookupRequest,
    PrefixCacheLookupResult, PrefixCacheManager, PrefixCacheMatchKind, PrefixCacheObservation,
    PrefixCacheObservationKind, PrefixCachePolicy, PrefixCachePrivacyPolicy, PrefixCacheScope,
    PrefixCacheSharingPolicy,
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
pub use sampling::{
    LogitsProcessorAuthority, LogitsProcessorConfig, LogitsProcessorKind, LogitsReference,
    SamplingDiagnostic, SamplingDiagnosticKind, SamplingError, SamplingErrorKind,
    SamplingFinishHint, SamplingObservation, SamplingObservationKind, SamplingPolicy,
    SamplingRequest, SamplingRequestId, SamplingResult, SamplingRngState, SamplingSelectionMode,
    SamplingStopMetadata, TemperatureZeroPolicy, processor_order, sampling_observation,
    sampling_workspace_requests, select_next_token,
};
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
pub use session::{
    InferenceSession, InferenceSessionId, SessionAccessPolicy, SessionConcurrencyPolicy,
    SessionCreationRequest, SessionError, SessionGenerationParameter, SessionLifecycleState,
    SessionMemoryBudget, SessionMemoryUsage, SessionObservation, SessionObservationKind,
    SessionOperationAdmission, SessionOperationState, SessionPolicy, SessionRedactionPolicy,
    SessionResources, SessionStatus, SessionStreamingStatus, runtime_session_affinity,
};
pub use tokenizer::{
    BatchEncodeInput, BatchEncodeOutput, DecodeInput, DecodeOutput, EncodeInput, EncodeOutput,
    FixtureTokenizer, PaddingPolicy, RuntimeTokenizer, SpecialToken, SpecialTokenKind,
    SpecialTokenPolicy, StreamingDecodeState, TokenId, TokenIdRange, TokenOffset, TokenStopPattern,
    Tokenizer, TokenizerArtifactId, TokenizerArtifactReference, TokenizerArtifactSet,
    TokenizerCompatibility, TokenizerDiagnostic, TokenizerDiagnosticKind, TokenizerError,
    TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerObservation,
    TokenizerObservationKind, TokenizerObserver, TokenizerRevision, TruncationPolicy,
    tokenizer_component_artifact_reference, tokenizer_memory_feasibility,
};

#[cfg(test)]
mod tests;
