//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod adapter;
pub mod affinity;
pub mod batching;
pub mod capability;
pub mod cli_boundary;
pub mod component;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub mod component_wasmtime;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub mod component_web;
pub mod compute;
pub mod conformance;
pub mod device;
pub mod e2e_conformance;
pub mod execution_graph;
pub mod generation;
pub mod inference_api;
pub mod kernel;
pub mod kernel_dispatch;
pub mod kernel_registry;
pub mod kv_cache;
pub mod memory;
pub mod model;
pub mod model_component;
pub mod model_format_roadmap;
pub mod model_instance;
pub mod model_loading;
pub mod model_source_cache_roadmap;
pub mod observability;
pub mod operator;
pub mod operator_scope;
pub mod planning;
pub mod prefix_cache;
pub mod provider;
pub mod provider_roadmap;
pub mod qwen_model_component;
pub mod reference_cpu;
pub mod release_cutover;
pub mod release_packaging;
pub mod release_security;
pub mod resolution;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod server_api_roadmap;
pub mod session;
pub mod tensor;
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
pub use cli_boundary::{
    CliBoundaryConformanceReport, CliBoundaryConformanceResult, CliBoundaryError,
    reject_cli_owned_authority, run_cli_boundary_conformance,
};
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
    PagedAppendBehavior, PrecisionSupport, ProviderComputeAdvertisement, RecoveryHint,
    ShapeDescriptor, ShapeLimitSupport, SymbolicDimension, TensorDescriptor,
    TensorDescriptorLimits, TensorResourceDescriptor, TensorResourceId, TensorViewSource,
    ViewDescriptor, compute_capability, initial_compute_operation_schemas,
};
pub use conformance::{
    PROVIDER_CONFORMANCE_SUITE_VERSION, ProviderConformanceConfig, ProviderConformanceProfile,
    ProviderConformanceReport, ProviderConformanceSuite, ProviderConformanceTarget,
    ProviderConformanceTargetKind, ProviderConformanceTestResult, ProviderConformanceTestStatus,
    provider_conformance_profile_ids, provider_conformance_report_json,
};
pub use device::{
    Device, DeviceDescriptor, DeviceExecutionLimits, DeviceId, DeviceMetadata, DeviceType,
};
pub use e2e_conformance::{
    E2E_EXERCISED_OPERATORS, E2E_FIXTURE_VERSION, E2E_SUITE_VERSION, E2eConformanceError,
    E2eConformanceReport, E2eFixture, E2eTestResult, E2eTestStatus, e2e_conformance_report_json,
    e2e_fixture, e2e_fixture_config, e2e_fixture_identity, e2e_fixture_manifest,
    e2e_fixture_tokenizer, e2e_fixture_weights, e2e_forward, run_e2e_local_inference_conformance,
    validate_e2e_no_shortcuts,
};
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
    GenerationTokenizerReference, GenerationUsage, LogitsProcessorReference, PrefillState,
    Probability, StopConditions, StreamingMode, decode_step, decode_step_from_sampling,
    finish_reason_from_provider_error, memory_admission, prefill, prepare_stop_sequences,
    stop_reason_for, streaming_text_chunk, token_stream_events,
};
pub use inference_api::{
    AdmissionState, BrowserInferenceCapabilities, CacheUsageSummary, CancellationOutcome,
    CancellationStage, CancellationToken, ChatMessage, ChatTemplateFormatter,
    FORBIDDEN_INFERENCE_API_SCOPES, GenerationApiRequest, GenerationResult, InferenceApiError,
    InferenceApiObservation, InferenceApiObservationKind, InferenceApiObserver,
    ModelLoadingApiRequest, ModelRef, ModelRegistry, ModelResolutionRequest, ModelResolutionResult,
    ModelResolutionSource, ModelResolutionStatus, PromptInput, RuntimeDiagnostics,
    RuntimeDiagnosticsInputs, StreamingDecodeRequest, StreamingHandle, TokenizationRequest,
    TokenizationResult, UNSUPPORTED_BROWSER_FEATURES, UsageReport, activate_adapter,
    activate_adapter_observed, build_generation_request, cancel_inference_session,
    close_inference_session, close_inference_session_observed, create_inference_session,
    create_inference_session_observed, create_model_instance, create_model_instance_observed,
    create_one_shot_session, decode_tokens, decode_tokens_streaming, drain_model_instance,
    load_model, load_model_observed, model_instance_status, prepare_generation,
    request_cancellation, request_cancellation_at_stage, request_cancellation_at_stage_observed,
    require_browser_supported, resume_model_instance, run_generation_loop, runtime_diagnostics,
    runtime_diagnostics_with, session_status, submit_generation, submit_generation_observed,
    suspend_model_instance, tokenize_prompt_input, tokenize_prompt_input_observed,
    unload_model_instance, validate_inference_scope, validate_inference_scopes,
    validate_tokenizer_compatibility, warm_model_instance,
};
pub use kernel::{
    KernelAdapterMetadata, KernelAdvertisement, KernelAliasing, KernelBatchMetadata,
    KernelCancellationSupport, KernelConformanceProfile, KernelDequantizationBehavior,
    KernelDeterminism, KernelError, KernelErrorCode, KernelExecutionMode, KernelFallbackClass,
    KernelFusionMetadata, KernelId, KernelImplementationFamily, KernelInvocation,
    KernelInvocationId, KernelKvCacheMetadata, KernelMemoryClass, KernelObservation,
    KernelObservationKind, KernelOperatorVersionRange, KernelPrecisionMetadata,
    KernelPrefixCacheMetadata, KernelQuantizationMetadata, KernelQuantizationMethod,
    KernelResource, KernelResult, KernelResultStatus, KernelShapeConstraints,
    KernelWorkspaceLifetime, KernelWorkspaceRequirements, KernelWorkspaceReuse,
};
pub use kernel_dispatch::{
    KernelDispatchError, KernelDispatchLifecycleState, KernelDispatchPlan, KernelDispatchPlanId,
    KernelDispatchResult, KernelDispatchRevalidationContext, KernelDispatcher, KernelFallbackStep,
    kernel_dispatch_error_from_provider_execution,
};
pub use kernel_registry::{
    KernelCandidate, KernelCandidateRejection, KernelRegistrationAuthority, KernelRegistry,
    KernelRegistryEntry, KernelRegistryError, KernelSelectionRequest, KernelSelectionResult,
    validate_kernel_advertisement,
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
pub use model_component::{
    ActivationKind, AttentionVariant, GraphProductionRequest, GraphProductionResult,
    MODEL_COMPONENT_CONTRACT_VERSION, MODEL_COMPONENT_ROLE, ModelComponentArchitectureMetadata,
    ModelComponentAuthority, ModelComponentCapabilityKind, ModelComponentCapabilityRequirement,
    ModelComponentConformanceProfile, ModelComponentDescriptor, ModelComponentError,
    ModelComponentId, ModelComponentIdentity, ModelComponentImplementationKind,
    ModelComponentKvCacheMetadata, ModelComponentModelType, ModelComponentObservation,
    ModelComponentObservationKind, ModelComponentProvenance,
    ModelComponentQuantizationCompatibility, ModelComponentSignatureState,
    ModelComponentTokenizerCompatibility, ModelComponentTrustStatus, ModelComponentVersion,
    NormalizationKind, OperatorRequirement, PositionEncodingKind, TargetModuleMetadata,
    TargetModuleRole, browser_feature_supported, device_handle_access_error,
    kernel_handle_access_error, memory_pointer_access_error, provider_handle_access_error,
    provider_owned_resource_access_error, validate_model_component_authority,
    validate_model_component_config_data, validate_model_component_role,
};
pub use model_format_roadmap::{
    ChatTemplateMetadata, ChatTemplateSourceKind, GenerationConfigMetadata, GgufMetadata,
    GgufTensorEntry, HfConfigMetadata, HfRopeMetadata, LoraAdapterFormatMetadata,
    MODEL_FORMAT_CONFORMANCE_FIXTURES, MODEL_FORMAT_ROADMAP_PHASES, MODEL_FORMAT_ROADMAP_VERSION,
    MemoryMappingPolicy, ModelFormatConformanceFixtureKind, ModelFormatQuantizationDeclaration,
    ModelFormatRoadmapConformanceReport, ModelFormatRoadmapConformanceResult,
    ModelFormatRoadmapError, ModelFormatRoadmapObservation, ModelFormatRoadmapObservationKind,
    ModelFormatRoadmapPhase, NormalizedManifestCoverage, PaddingSide, SafetensorsManifest,
    SafetensorsTensorEntry, SentencePieceMetadata, ShardIndex, TokenizerConfigMetadata,
    TokenizerJsonMetadata, TruncationSide, apply_generation_override,
    detect_duplicate_tensor_names, detect_missing_shards, model_format_grants_no_trust,
    normalize_lora_adapter, normalize_tokenizer_json, redact_chat_template_diagnostic,
    reject_arbitrary_model_download, reject_format_execution_graph,
    reject_model_format_provider_name, reject_raw_network_model_reference,
    reject_silent_tokenizer_config_override, reject_unsupported_sentencepiece_feature,
    run_model_format_roadmap_conformance, torch_dtype_does_not_force_compute_dtype,
    validate_chat_template, validate_local_file_boundary, validate_model_format_quantization,
    validate_shard_loading_order, validate_shard_tensor_shape_consistency,
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
pub use model_source_cache_roadmap::{
    ArtifactIdentityCoverage, CACHE_LIFECYCLE_STATES, CacheEntryMetadata, CacheEntryRef,
    CacheIntegrityStatus, CacheKey, CacheLifecycleState, CacheMutationKind, CacheValidationStatus,
    EvictionCandidate, MODEL_SOURCE_CACHE_ROADMAP_VERSION, MODEL_SOURCE_KINDS, ModelAlias,
    ModelAliasTable, ModelRefResolutionOutcome, ModelSourceCacheDiagnostic,
    ModelSourceCacheRoadmapConformanceReport, ModelSourceCacheRoadmapConformanceResult,
    ModelSourceCacheRoadmapError, ModelSourceCacheRoadmapObservation,
    ModelSourceCacheRoadmapObservationKind, ModelSourceKind, SourcePolicy,
    artifacts_are_distinct_despite_same_name, authorize_cache_mutation,
    cache_entry_ready_for_format_normalization, cache_presence_implies_memory_residency,
    development_fixture_requires_explicit_trust_evaluation, evaluate_cache_trust, is_evictable,
    pin_entry, reject_credential_in_metadata, reject_non_ready_cache_entry_for_loading,
    resolve_model_ref_candidates, run_model_source_cache_roadmap_conformance,
    select_eviction_candidates, unpin_entry, validate_adapter_cache_entry,
    validate_cache_integrity, validate_cache_shard_integrity, validate_client_provided_source,
    validate_development_fixture_source, validate_license_policy, validate_local_directory_source,
    validate_offline_source, validate_remote_source_policy, validate_tokenizer_cache_entry,
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
pub use operator_scope::{
    FIRST_OPERATOR_SCOPE_VERSION, FirstScopeDTypeTier, FirstScopeError, FirstScopeErrorCode,
    FirstScopeLayoutTier, FirstScopeObservation, FirstScopeObservationKind,
    FutureOptimizedOperator, OperatorScopeEntry, OperatorScopeTier, first_operator_scope,
    first_scope_dtype_tier, first_scope_layout_tier, first_scope_required_fixture_names,
    future_optimized_operators, operator_scope_entry, validate_first_scope_dtype,
    validate_first_scope_graph, validate_first_scope_kernel_selection_request,
    validate_first_scope_layout, validate_model_component_first_scope_requirements,
    validate_no_placeholder_kernel_advertisements, validate_reference_cpu_required_kernel_coverage,
    validate_required_now_operator,
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
pub use provider_roadmap::{
    AdvancedAttentionDeclaration, AdvancedAttentionVariant, FusedKernelDeclaration,
    POST_BASELINE_LAYOUTS, POST_BASELINE_MEMORY_CLASSES, PROVIDER_ROADMAP_FEATURES,
    PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES, PROVIDER_ROADMAP_PHASES,
    PROVIDER_ROADMAP_VERSION, ProviderRoadmapBenchmarkCategory, ProviderRoadmapBenchmarkResult,
    ProviderRoadmapConformanceReport, ProviderRoadmapConformanceResult, ProviderRoadmapError,
    ProviderRoadmapFallbackContext, ProviderRoadmapFallbackEdge, ProviderRoadmapFeature,
    ProviderRoadmapHardwareFamily, ProviderRoadmapObservation, ProviderRoadmapObservationKind,
    ProviderRoadmapPhase, ProviderRoadmapPolicyPreference, cli_may_pass_policy_preference,
    cli_redacted_provider_diagnostic, evaluate_provider_roadmap_fallback,
    evaluate_provider_roadmap_fallback_observed, phase_is_production_ready,
    provider_roadmap_conformance_profile_ids, provider_roadmap_features_for_phase,
    reject_cli_raw_provider_handle_selection, reject_hidden_dequantization,
    reject_model_family_provider_name, reject_native_handle_exposure,
    reject_provider_specific_handle_capability, reject_unsupported_advanced_attention,
    require_explicit_layout_conversion, require_memory_manager_tracking,
    run_provider_roadmap_conformance, validate_advanced_attention_declaration,
    validate_fused_kernel_declaration, validate_quantization_declaration,
};
pub use qwen_model_component::{
    QWEN_ARCHITECTURE_FAMILY, QWEN_BASELINE_CONTRACT_VERSION, QwenComponentError, QwenConfig,
    QwenRopeConfig, QwenRopePositionMode, qwen_adapter_architecture_compatibility,
    qwen_architecture_metadata, qwen_authority, qwen_browser_supported,
    qwen_component_compatibility_key, qwen_component_descriptor, qwen_component_identity,
    qwen_conformance_fixture_names, qwen_decode_graph, qwen_expected_tensor_names,
    qwen_expected_tensor_shape, qwen_kv_cache_metadata, qwen_operator_requirements,
    qwen_prefill_graph, qwen_quantization_compatibility, qwen_target_modules,
    qwen_tokenizer_compatibility, qwen_validate_generation_defaults, qwen_validate_model_artifact,
    qwen_validate_reference_cpu_coverage, qwen_validate_tensor_inventory,
    qwen_validate_tensor_shapes, qwen_validate_tokenizer_compatibility,
};
pub use reference_cpu::{
    FallbackPolicyContext, HostTensor, REFERENCE_CPU_BUILT_IN, REFERENCE_CPU_CONFORMANCE_PROFILE,
    REFERENCE_CPU_DEVICE_ID, REFERENCE_CPU_KERNEL_FAMILY, REFERENCE_CPU_PROVIDER_NAME,
    REFERENCE_CPU_PROVIDER_VENDOR, REFERENCE_CPU_PROVIDER_VERSION,
    REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE, ReferenceCpuConformanceCheck,
    ReferenceCpuConformanceReport, ReferenceCpuError, ReferenceCpuErrorCode, ReferenceCpuExecutor,
    ReferenceCpuFeatureFlags, ReferenceCpuProvider, add, attention, dequantize_placeholder,
    dtype_conversion, embedding_lookup, evaluate_fallback, gelu, layout_conversion, matmul, mul,
    quantize_placeholder, reference_cpu_device, reference_cpu_kernel_advertisements,
    reference_cpu_provider_metadata, residual_add, rmsnorm, rope, silu, softmax_rows,
};
pub use release_cutover::{
    CUTOVER_COMPATIBILITY_DIMENSIONS, CutoverArtifactVerification, CutoverChangelogChecklist,
    CutoverCompatibilityDimension, CutoverCompatibilityMatrix, CutoverCompatibilityStatus,
    CutoverException, CutoverReleaseNotesChecklist, CutoverSecurityVerification,
    CutoverVersionConfirmation, GateSkip, OpenSpecFreezeConfirmation, POST_V0_1_HANDOFF_CANDIDATES,
    PostPublicationVerification, PostV01HandoffItem, RELEASE_CUTOVER_POLICY_VERSION,
    ReleaseCutoverConformanceReport, ReleaseCutoverConformanceResult, ReleaseCutoverError,
    ReleaseCutoverGateInputs, ReleaseCutoverObservation, ReleaseReadinessChecklist,
    RollbackRetractionNotes, V0_1_FINAL_RELEASE_STATEMENT, V0_1_INCLUDED_SCOPE,
    WitPackageVersionRecord, cutover_compatibility_dimension_id, cutover_compatibility_status_id,
    evaluate_release_cutover, record_release_cutover_observation,
    reject_post_v0_1_item_as_release_claim, reject_semantic_change_after_freeze,
    reject_status_misrepresentation, reject_undocumented_cutover_exception,
    run_release_cutover_conformance, validate_cutover_artifacts_generated,
    validate_cutover_cli_boundary, validate_cutover_exceptions, validate_cutover_feature_flag,
    validate_cutover_provider_feature_flags, validate_cutover_runtime_scope,
    validate_final_release_statement, validate_gate_skips, validate_publication_scope_preserved,
    validate_required_gates_executed, validate_runtime_version_matches_release_tag,
    validate_tag_after_gates, validate_v0_1_scope_feature, validate_wit_versions_confirmed,
    verify_cutover_artifact_checksum,
};
pub use release_packaging::{
    ArtifactChecksum, CLI_BOUNDARY_CONFORMANCE_VERSION, COMPATIBILITY_DIMENSIONS, ChangelogEntry,
    ChangelogEntryKind, ChecksumAlgorithm, CompatibilityDimension, CompatibilityStatus,
    CrateVersionMetadata, OpenSpecBaselineDeclaration, PublishingBoundaryCategory,
    RELEASE_ARTIFACT_KINDS, RELEASE_PACKAGING_POLICY_VERSION, REQUIRED_RELEASE_GATES,
    RUNTIME_INFERENCE_API_CONFORMANCE_VERSION, ReleaseArtifactKind, ReleaseArtifactManifest,
    ReleaseArtifactStatus, ReleaseBinaryVersionReport, ReleaseBuildMetadata,
    ReleaseCandidateManifest, ReleaseCandidateTag, ReleaseChangelog, ReleaseCompatibilityMatrix,
    ReleaseConformanceVersions, ReleaseDocumentationChecklist, ReleaseFeatureFlag,
    ReleaseFeatureFlagClass, ReleaseFreezeChangeKind, ReleaseFreezeState, ReleaseGate,
    ReleaseGateResult, ReleasePackagingConformanceReport, ReleasePackagingConformanceResult,
    ReleasePackagingError, ReleasePlatformTarget, ReleaseSecurityNotes, ReleaseVersion,
    ReleaseVersionBumpKind, SupportedWitVersionMatrix, WitVersionChangeKind,
    allow_failed_candidate_as_pre_release, artifact_kind_id, build_release_binary_version_report,
    classify_publishing_boundary, compatibility_dimension_id, component_engine_feature_flags,
    evaluate_version_bump, provider_feature_flags, redact_build_metadata,
    reject_change_after_freeze, reject_experimental_flag_enabled_by_default,
    reject_release_public_api_handle_exposure, reject_roadmap_feature_as_guarantee,
    reject_wasmtime_required_for_browser, release_may_publish_stable, release_platform_targets,
    release_wit_contract_versions, required_wit_version_bump, run_release_packaging_conformance,
    unsupported_targets, v0_1_compatibility_matrix, validate_crate_dependency_compatibility,
    validate_provider_feature_flags_for_v0_1, validate_wit_version_bump,
};
pub use release_security::{
    ArtifactIntegrityStatus, BuildScriptReview, DependencyAdvisory, DependencyAdvisorySeverity,
    DependencyAuditReport, DependencyFeatureCapability, DependencyFeatureReview, DependencyLicense,
    DynamicProviderLoadingStatus, FixtureModelTrustPolicy, LicenseAuditReport, LicenseAuditStatus,
    LockfileState, NonTrustCacheSignal, ProviderTrustModel, ProviderTrustSignalSource,
    REDACTION_CATEGORIES, RELEASE_SECURITY_POLICY_VERSION,
    RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS, RELEASE_SECURITY_SCOPE_INCLUDED,
    RedactionCategory, ReleaseProvenance, ReleaseSecurityConformanceReport,
    ReleaseSecurityConformanceResult, ReleaseSecurityError, ReleaseSecurityGateInputs,
    ReleaseSecurityObservation, ReleaseSecurityObservationKind, ReproducibilityReport,
    ReproducibilityStatus, SECRET_SCAN_TARGETS, SbomAvailability, SbomEntry, SbomManifest,
    SecretScanFinding, SecretScanReport, SecretScanTarget, SecurityException, SecurityReleaseNotes,
    SignatureStatus, UnsafeCodePolicy, UnsafeCodeReview, VulnerabilityHandlingPolicy,
    evaluate_release_security_blocking, flag_unexpected_build_script,
    magnetar_runtime_unsafe_code_inventory, record_release_security_observation,
    reject_cache_signal_alone_as_trust, reject_component_release_authority_expansion,
    reject_hardened_security_claim_for_excluded_feature,
    reject_provider_registration_implies_trust, reject_release_native_handle_exposure,
    reject_undocumented_security_exception, reject_unexpected_capability_expanding_feature,
    reject_unreviewed_lockfile_drift, run_release_security_conformance, secret_scan_target_id,
    validate_cli_authority_not_delegated_to_runtime, validate_component_release_execution_trust,
    validate_dynamic_provider_loading_status, validate_fixture_model_trust,
    validate_model_artifact_release_trust, validate_redaction_gate,
    validate_runtime_inference_api_security, validate_signature_status,
    validate_source_cache_release_trust, verify_checksum_matches_final_artifact,
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
pub use server_api_roadmap::{
    AuthenticatedServerRequest, ModelEndpointLoadingProof, OpenAiCompatibilityPolicy,
    SERVER_API_ENDPOINTS, SERVER_API_ROADMAP_VERSION, SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS,
    ServerAdmissionLimits, ServerAdmissionState, ServerApiEndpoint,
    ServerApiRoadmapConformanceReport, ServerApiRoadmapConformanceResult, ServerApiRoadmapError,
    ServerApiRoadmapObservation, ServerApiRoadmapObservationKind, ServerAuthorizationDecision,
    ServerAuthorizationScope, ServerConnectionId, ServerConnectionState, ServerDiagnosticsSummary,
    ServerDisconnectPolicy, ServerGeneratedTextHandling, ServerGenerationRequest,
    ServerGenerationRuntimeContext, ServerHealthStatus, ServerModelEndpointOperation,
    ServerModelOrSessionRef, ServerReadinessStatus, ServerSessionRequest, ServerStreamEvent,
    ServerStreamingTransport, authorize_server_request, build_runtime_generation_request,
    evaluate_server_admission, handle_openai_unsupported_field,
    healthy_but_not_ready_is_representable, openai_facade_maps_to_generation_api_request,
    redact_server_diagnostic, reject_arbitrary_download_during_generation,
    reject_arbitrary_filesystem_path, reject_credential_in_server_diagnostics,
    reject_openai_tool_call_execution, reject_raw_stream_payload,
    reject_server_arbitrary_model_path, reject_server_session_owned_authority,
    reject_server_tool_shell_git_execution, reject_tool_execution_from_generated_output,
    run_server_api_roadmap_conformance, server_cancellation_calls_runtime_cancellation,
    server_diagnostics_summary, server_disconnect_policy, validate_model_endpoint_request,
    validate_stream_event_ordering,
};
pub use session::{
    InferenceSession, InferenceSessionId, SessionAccessPolicy, SessionConcurrencyPolicy,
    SessionCreationRequest, SessionError, SessionGenerationParameter, SessionLifecycleState,
    SessionMemoryBudget, SessionMemoryUsage, SessionObservation, SessionObservationKind,
    SessionOperationAdmission, SessionOperationState, SessionPolicy, SessionRedactionPolicy,
    SessionResources, SessionStatus, SessionStreamingStatus, runtime_session_affinity,
};
pub use tensor::{
    DimensionRole, TensorAliasingKind, TensorError, TensorLifecycleState, TensorMemoryClass,
    TensorMutabilityKind, TensorObservation, TensorObservationKind, TensorOwnerSubsystem,
    TensorReadiness, TensorResource, TensorView, validate_aliasing_for_dispatch,
    validate_memory_class_for_kernel, validate_mutability_for_dispatch,
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
