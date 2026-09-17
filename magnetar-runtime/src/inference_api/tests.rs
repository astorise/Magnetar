//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::adapter::{
    AdapterActivationRequest, AdapterActivationScope, AdapterArtifactId, AdapterCompositionPolicy,
    AdapterDigest, AdapterLifecycleState, AdapterName, AdapterResidency, AdapterResidencyId,
    AdapterResidencyLocation, AdapterRevision, AdapterSetId,
};
use crate::generation::{GenerationModelReference, GenerationRequestId};
use crate::model::ModelDigest;
use crate::session::InferenceSessionId;
use crate::tokenizer::{
    FixtureTokenizer, SpecialToken, SpecialTokenKind, TokenIdRange, TokenizerArtifactId,
    TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerRevision,
};

use crate::affinity::{FallbackClass, ProviderBinding, ResourceAffinity};
use crate::batching::BatchingPolicy;
use crate::compute::TensorResourceId;
use crate::generation::GenerationOutput;
use crate::generation::{
    CancellationMetadata, EosMode, EosOutputPolicy, EosPolicy, FinishReason,
    GenerationMemoryEstimate, GenerationParameters, GenerationPriority, GenerationRequest,
    GenerationTokenizerReference, StopConditions, StreamingMode, token_stream_events,
};
use crate::kernel_execution_plan::PreparedExecutionPlan;
use crate::materialize_model_instance_weights;
use crate::memory::MemoryPlacement;
use crate::memory::MemoryPressureLevel;
use crate::model::ModelArchitecture;
use crate::model::{ModelArtifactId, ModelArtifactKind, ModelName, ModelRevision};
use crate::model_instance::ModelInstanceId;
use crate::model_instance::{
    ModelInstanceAdapterState, ModelInstanceDefinition, ModelInstanceLifecycleState,
    ModelInstancePlacement, ModelInstancePolicy, ModelInstanceReadinessChecks,
    ModelInstanceResourceBindings, ModelInstanceSuspensionReason, ModelInstanceUsage,
    ModelInstanceWarmupPlan, ModelInstanceWarmupPolicy,
};
use crate::model_loading::ModelLoadingPhase;
use crate::model_loading::{
    ModelArchitectureImplementation, ModelArchitectureImplementationKind, ModelResidencyId,
};
use crate::observability::{CorrelationId, TraceId};
use crate::reference_cpu::REFERENCE_CPU_PROVIDER_NAME;
use crate::reference_cpu::ReferenceCpuProvider;
use crate::runtime::Runtime;
use crate::sampling::SamplingPolicy;
use crate::session::{SessionCreationRequest, SessionMemoryBudget, SessionPolicy};
use crate::tensor::HostTensor;
use crate::tokenizer::TokenizerCompatibility;
use crate::tokenizer::{TokenId, TokenStopPattern};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
fn adapter_activation_request_fixture(residency: &AdapterResidency) -> AdapterActivationRequest {
    AdapterActivationRequest {
        residency: residency.id.clone(),
        scope: AdapterActivationScope::Session(InferenceSessionId::new("session-1").unwrap()),
        base_model: GenerationModelReference::LoadedModelContext("model-context".into()),
        adapter_set: AdapterSetId::from_adapters([residency.artifact.clone()]),
        policy: AdapterCompositionPolicy::SingleAdapterOnly,
    }
}

fn generation_tokenizer_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("fixture").unwrap(),
        artifact: TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(16),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: vec![SpecialToken::new(SpecialTokenKind::Stop, "<stop>", 298)],
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    }
}

struct ConcatenationChatTemplate;

impl ChatTemplateFormatter for ConcatenationChatTemplate {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        Ok(messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

fn adapter_residency_fixture() -> AdapterResidency {
    AdapterResidency {
        id: AdapterResidencyId::new("residency-1").unwrap(),
        artifact: AdapterArtifactId {
            name: AdapterName::new("lora-a").unwrap(),
            revision: AdapterRevision::new("1").unwrap(),
            digest: AdapterDigest {
                algorithm: "sha256".into(),
                value: "deadbeef".into(),
            },
        },
        lifecycle: AdapterLifecycleState::Ready,
        location: AdapterResidencyLocation::Host,
        affinity: None,
        memory_allocation: None,
        provider_resource: None,
    }
}

#[test]
fn inference_api_scope_validation_rejects_non_inference_capabilities() {
    assert!(validate_inference_scope("generation").is_ok());
    for forbidden in [
        "workspace-filesystem",
        "git",
        "shell",
        "secrets",
        "tool-call",
    ] {
        assert!(matches!(
            validate_inference_scope(forbidden),
            Err(InferenceApiError::PolicyDenied { .. })
        ));
    }
}

#[test]
fn inference_api_tokenize_plain_text_never_stores_raw_prompt_text() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::PlainText("secret".into()));

    let result = tokenize_prompt_input(&tokenizer, request, None).unwrap();
    assert!(!result.token_ids.is_empty());
    assert!(!format!("{result:?}").contains("secret"));
}

#[test]
fn inference_api_tokenize_chat_messages_formats_through_authorized_contract() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hi",
    )]));

    let result =
        tokenize_prompt_input(&tokenizer, request, Some(&ConcatenationChatTemplate)).unwrap();
    assert!(!result.token_ids.is_empty());
}

#[test]
fn inference_api_tokenize_already_tokenized_input_validates_range() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let in_range = TokenizationRequest::new(PromptInput::TokenIds(vec![2, 3, 4]));
    assert!(tokenize_prompt_input(&tokenizer, in_range, None).is_ok());

    let out_of_range = TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999]));
    let error = tokenize_prompt_input(&tokenizer, out_of_range, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizationFailed { .. }
    ));
}

#[test]
fn inference_api_tokenize_prompt_input_observed_emits_tokenized_and_failed() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut observer = InferenceApiObserver::new();

    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("secret".into())),
        None,
        &mut observer,
    )
    .unwrap();
    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999])),
        None,
        &mut observer,
    )
    .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::PromptTokenized));
    assert!(kinds.contains(&InferenceApiObservationKind::TokenizationFailed));
}

#[test]
fn inference_api_adapter_activation_observed_emits_adapter_activated() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);
    let mut observer = InferenceApiObserver::new();

    activate_adapter_observed(&residency, &request, None, None, &mut observer).unwrap();

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::AdapterActivated)
    );
}

#[test]
fn inference_api_cancellation_stage_before_dispatch_always_succeeds() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    for stage in [
        CancellationStage::Queued,
        CancellationStage::Tokenization,
        CancellationStage::Prefill,
        CancellationStage::Decode,
        CancellationStage::Sampling,
        CancellationStage::Batching,
        CancellationStage::GraphExecution,
        CancellationStage::KernelDispatch,
    ] {
        assert_eq!(
            request_cancellation_at_stage(&token, stage, false),
            CancellationOutcome::Cancelled
        );
    }
}

#[test]
fn inference_api_cancellation_stage_provider_execution_depends_on_support() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_cancellation_stage_observed_emits_generation_cancelled() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    let mut observer = InferenceApiObserver::new();

    request_cancellation_at_stage_observed(&token, CancellationStage::Decode, false, &mut observer);

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::GenerationCancelled)
    );
}

#[test]
fn inference_api_observer_buffer_stays_bounded_and_retains_most_recent() {
    let mut observer = InferenceApiObserver::new();
    let total = INFERENCE_API_OBSERVATION_BUFFER_CAPACITY + 128;
    for index in 0..total {
        observer.observe(
            InferenceApiObservationKind::TokenCommitted,
            format!("observation-{index}"),
            None,
        );
    }
    assert_eq!(
        observer.observations().len(),
        INFERENCE_API_OBSERVATION_BUFFER_CAPACITY
    );
    // The oldest observations were evicted to admit the newest ones -- the
    // buffer reflects the most recent causal evidence, not whatever
    // happened to be observed first.
    assert!(
        !observer
            .observations()
            .iter()
            .any(|observation| observation.message == "observation-0")
    );
    assert_eq!(
        observer
            .observations()
            .last()
            .map(|observation| observation.message.as_str()),
        Some(format!("observation-{}", total - 1)).as_deref()
    );
}

fn generation_request() -> GenerationRequest {
    let metadata = generation_tokenizer_metadata();
    GenerationRequest {
        request_id: GenerationRequestId::new("gen-1").unwrap(),
        session: None,
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        input_token_ids: vec![2, 3, 4],
        prompt_token_count: 3,
        max_new_tokens: 4,
        max_total_tokens: Some(8),
        model_context_length: Some(8),
        parameters: GenerationParameters::default(),
        stop_conditions: StopConditions {
            eos: EosPolicy {
                mode: EosMode::Stop,
                output: EosOutputPolicy::Exclude,
                eos_token_ids: vec![299],
            },
            stop_token_ids: vec![298],
            stop_token_patterns: vec![vec![10, 11]],
            stop_text_sequences: vec!["stop".into()],
            prepared_stop_sequences: vec![TokenStopPattern {
                text: "xy".into(),
                token_ids: vec![121, 122],
                exact: true,
            }],
            ..StopConditions::default()
        },
        streaming: StreamingMode::TokenIds,
        priority: GenerationPriority {
            priority: 3,
            deadline_millis: Some(100),
        },
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 12,
            output_token_buffer_bytes: 16,
            logits_buffer_bytes: 32,
            sampling_workspace_bytes: 8,
            prefill_workspace_bytes: 8,
            decode_workspace_bytes: 8,
            kv_cache_placeholder_bytes: 8,
            prefix_cache_placeholder_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: false,
        },
        correlation_id: Some(CorrelationId::new("corr-1")),
        trace_id: Some(TraceId::new("trace-1")),
    }
}

fn runtime_with_model_execution_engine(
    vocabulary_size: usize,
    evidence: RuntimeGenerationExecutionEvidence,
) -> Runtime {
    Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(Arc::new(TestGenerationExecutor {
            vocabulary_size,
            evidence,
        }))
        .build()
        .unwrap()
}

#[derive(Clone)]
struct TestGenerationExecutor {
    vocabulary_size: usize,
    evidence: RuntimeGenerationExecutionEvidence,
}

impl RuntimeModelExecutionEngine for TestGenerationExecutor {
    fn execute_generation_step(
        &self,
        _runtime: &mut Runtime,
        _request: &GenerationRequest,
        generated_tokens: &[TokenId],
        _execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        let mut logits = vec![0.0f32; self.vocabulary_size];
        logits[(11 + generated_tokens.len()) % self.vocabulary_size] = 10.0;
        Ok(RuntimeModelExecutionStep::new(
            logits,
            self.evidence.clone(),
        ))
    }
}

#[test]
fn inference_api_model_reference_rejects_path_like_input() {
    assert!(matches!(
        ModelRef::new("../etc/passwd"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
    assert!(matches!(
        ModelRef::new("models/qwen"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
}

#[test]
fn inference_api_tokenize_chat_messages_requires_authorized_formatter() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hello",
    )]));

    let error = tokenize_prompt_input(&tokenizer, request, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_streaming_handle_correlates_with_ordered_token_events() {
    let request = generation_request();
    let handle = StreamingHandle::for_request(&request);

    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();
    assert_eq!(events.len(), 3);
    assert!(
        events
            .iter()
            .all(|event| event.request_id == handle.request)
    );
    let indices: Vec<_> = events
        .iter()
        .filter_map(|event| event.token_index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn inference_api_cancellation_reports_limitation_when_unsupported_after_dispatch() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation(&token, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation(&token, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_browser_feature_check_only_rejects_on_wasm32() {
    let result = require_browser_supported("wasmtime");
    if cfg!(target_arch = "wasm32") {
        assert!(matches!(
            result,
            Err(InferenceApiError::BrowserFeatureUnsupported { .. })
        ));
    } else {
        assert!(result.is_ok());
    }
}

#[test]
fn inference_api_streaming_decode_request_carries_state_across_calls() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut request = StreamingDecodeRequest::new(vec![2, 3]);
    request.skip_special_tokens = true;

    let output = decode_tokens_streaming(&tokenizer, request).unwrap();
    assert!(output.consumed_token_count > 0 || output.pending_partial_state.is_some());
}

#[test]
fn inference_api_run_generation_loop_emits_full_streaming_lifecycle_and_completes() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    request.max_new_tokens = 2;
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;

    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence::complete(),
    );
    let mut observer = InferenceApiObserver::new();

    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        },
        |_generated| false,
        &mut observer,
    )
    .unwrap();

    assert_eq!(result.output.generated_token_count, 2);
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    for expected in [
        InferenceApiObservationKind::GenerationStarted,
        InferenceApiObservationKind::StreamOpened,
        InferenceApiObservationKind::KvCacheUsed,
        InferenceApiObservationKind::PrefixCacheMiss,
        InferenceApiObservationKind::PrefillStarted,
        InferenceApiObservationKind::PrefillCompleted,
        InferenceApiObservationKind::DecodeStarted,
        InferenceApiObservationKind::TokenGenerated,
        InferenceApiObservationKind::GenerationCompleted,
        InferenceApiObservationKind::StreamClosed,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}

#[test]
fn inference_api_run_generation_loop_rejects_incomplete_executor_evidence_before_sampling() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;
    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence {
            model_instance_ready: true,
            graph_validated: true,
            kernel_selected: true,
            kernel_dispatched: false,
            provider_executed: false,
            tensor_resource_used: false,
            context: Vec::new(),
        },
    );
    let mut observer = InferenceApiObserver::new();

    let error = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();

    assert!(matches!(error, InferenceApiError::KernelUnavailable { .. }));
    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ExecutionGraphValidated));
    assert!(kinds.contains(&InferenceApiObservationKind::KernelSelected));
    assert!(kinds.contains(&InferenceApiObservationKind::KernelUnavailable));
    assert!(kinds.contains(&InferenceApiObservationKind::StreamInterrupted));
    assert!(!kinds.contains(&InferenceApiObservationKind::TokenGenerated));
    assert!(!kinds.contains(&InferenceApiObservationKind::GenerationCompleted));
    assert!(!kinds.contains(&InferenceApiObservationKind::StreamClosed));
}

#[test]
fn inference_api_adapter_activation_succeeds_for_ready_residency() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);

    activate_adapter(&residency, &request, None, None).unwrap();
}

#[test]
fn inference_api_tachyon_and_cli_boundary_capabilities_are_inference_only() {
    for forbidden in ["git", "shell", "agent-orchestration", "secrets"] {
        assert!(validate_inference_scope(forbidden).is_err());
    }
    assert!(validate_inference_scope("generation").is_ok());
}

#[test]
fn inference_api_validate_tokenizer_compatibility_accepts_matching_digest() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata.clone());
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(metadata.digest.clone()),
        expected_vocabulary_size: Some(metadata.vocabulary_size),
        expected_family: Some(metadata.family.clone()),
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap();
}

fn session_creation_request() -> SessionCreationRequest {
    let metadata = generation_tokenizer_metadata();
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
        created_at_millis: 0,
    }
}

#[test]
fn inference_api_build_generation_request_from_tokenized_input() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    };
    let tokenized = TokenizationResult {
        token_ids: vec![2, 3, 4],
        token_count: 3,
        offsets: None,
        diagnostics: Vec::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
    };

    let request = build_generation_request(
        GenerationRequestId::new("gen-api-1").unwrap(),
        None,
        GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer,
        tokenized,
        4,
        GenerationParameters::default(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );

    assert_eq!(request.prompt_token_count, 3);
    request.validate().unwrap();
}

#[test]
fn inference_api_model_resolution_observed_emits_model_resolved_and_failed() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact);
    let mut observer = InferenceApiObserver::new();

    registry
        .resolve_observed(&ModelResolutionRequest::new(reference), &mut observer)
        .unwrap();
    registry
        .resolve_observed(
            &ModelResolutionRequest::new(ModelRef::new("unknown").unwrap()),
            &mut observer,
        )
        .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolved));
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolutionFailed));
}

#[test]
fn inference_api_session_lifecycle_observed_emits_created_and_closed() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let mut observer = InferenceApiObserver::new();

    let session = create_inference_session_observed(&mut runtime, request, &mut observer).unwrap();
    close_inference_session_observed(&mut runtime, &session, &mut observer).unwrap();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::SessionCreated));
    assert!(kinds.contains(&InferenceApiObservationKind::SessionClosed));
}

#[test]
fn inference_api_one_shot_pipeline_uses_session_tokenizer_and_generation_contracts() {
    let metadata = generation_tokenizer_metadata();
    let vocabulary_size = metadata.vocabulary_size as usize;
    let mut runtime = runtime_with_model_execution_engine(
        vocabulary_size,
        RuntimeGenerationExecutionEvidence::complete(),
    );
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let session = create_one_shot_session(&mut runtime, request).unwrap();

    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let tokenized = tokenize_prompt_input(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )
    .unwrap();

    let generation_request = build_generation_request(
        GenerationRequestId::new("one-shot-1").unwrap(),
        Some(session.clone()),
        GenerationModelReference::LoadedModelContext("model-context".into()),
        GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        tokenized,
        2,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );
    let prepared = prepare_generation(&runtime, generation_request).unwrap();

    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let (state, _) = submit_generation(&mut runtime, &batch, &prepared).unwrap();
    assert_eq!(state, AdmissionState::Accepted);

    // One-shot inference SHALL not bypass Model Instance, Tokenizer,
    // Generation, Sampling, Memory Manager, or Provider/Kernel contracts:
    // this drives the same Generation Contract loop (prefill, per-token
    // Sampling Contract decode, Provider/Kernel readiness gating) that any
    // session-bound generation would use.
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &prepared,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| false,
        &mut observer,
    )
    .unwrap();
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::TokenGenerated)
    );

    close_inference_session(&mut runtime, &session).unwrap();
}

#[test]
fn inference_api_adapter_activation_rejects_forbidden_operation_scope() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.scope = AdapterActivationScope::Operation("shell".into());

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_adapter_activation_denied_when_incompatible() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.residency = AdapterResidencyId::new("other-residency").unwrap();

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::AdapterActivationFailed { .. }
    ));
}

#[test]
fn inference_api_runtime_diagnostics_with_inputs_includes_caller_supplied_status() {
    let runtime = Runtime::builder().build().unwrap();
    let inputs = RuntimeDiagnosticsInputs {
        model_resolution_status: Some(ModelResolutionStatus::Resolved),
        model_loading_status: Some(ModelLoadingPhase::PublishModelContext),
        operator_missing_count: 2,
        tokenizer_compatible: Some(true),
        queued_admission_count: 3,
    };

    let diagnostics = runtime_diagnostics_with(&runtime, inputs);
    assert_eq!(
        diagnostics.model_resolution_status,
        Some(ModelResolutionStatus::Resolved)
    );
    assert_eq!(
        diagnostics.model_loading_status,
        Some(ModelLoadingPhase::PublishModelContext)
    );
    assert_eq!(diagnostics.operator_missing_count, 2);
    assert_eq!(diagnostics.tokenizer_compatible, Some(true));
    assert_eq!(diagnostics.queued_admission_count, 3);
    assert!(diagnostics.redacted);
}

#[test]
fn inference_api_generation_result_wraps_output_with_decoded_text_and_cache_usage() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11, 12], FinishReason::EosToken);

    let result = GenerationResult::new(output)
        .with_decoded_text("hello".into())
        .with_model_instance(ModelInstanceId::new("instance-1").unwrap())
        .with_cache_usage(CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        });

    assert_eq!(result.decoded_text.as_deref(), Some("hello"));
    assert!(result.model_instance.is_some());
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));
    assert!(result.error.is_none());
    assert!(result.redacted);
}

#[test]
fn inference_api_runtime_diagnostics_are_redacted_and_reflect_empty_runtime() {
    let runtime = Runtime::builder().build().unwrap();
    let diagnostics = runtime_diagnostics(&runtime);
    assert!(diagnostics.redacted);
    assert_eq!(diagnostics.model_instance_count, 0);
    assert_eq!(diagnostics.active_session_count, 0);
}

#[test]
fn inference_api_validate_tokenizer_compatibility_rejects_digest_mismatch() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata);
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(ModelDigest::sha256(b"a different tokenizer")),
        expected_vocabulary_size: None,
        expected_family: None,
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    let error = validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizerIncompatible { .. }
    ));
}

#[test]
fn inference_api_browser_inference_capabilities_reduced_excludes_kv_cache() {
    let capabilities = BrowserInferenceCapabilities::reduced();
    assert!(capabilities.tokenization);
    assert!(capabilities.generation);
    assert!(capabilities.streaming);
    assert!(!capabilities.kv_cache);
}

/// Materializes one real, Runtime-issued-evidence-backed weight for
/// `instance` through `materialize_model_instance_weights` -- the one
/// legitimate way any caller (production or an external embedder) can turn
/// weight bytes into bound, Ready-eligible resources
/// (`bind-model-loading-evidence-to-validated-artifact`). This alone
/// reaches `Ready`: the underlying transaction commits bindings, mints
/// materialization evidence, and marks the instance Ready in one step, the
/// same as the real production path -- a separate follow-up
/// `warm_model_instance` call is not just redundant but would fail (no
/// `Ready -> Ready` lifecycle transition exists). Requires a
/// `ReferenceCpuProvider` already registered on `runtime` and `instance`'s
/// placement pinned to it.
fn reach_ready_with_real_weight(runtime: &mut Runtime, instance: &ModelInstanceId) {
    let weights = BTreeMap::from([("weight".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(runtime, instance, "test", &weights).unwrap();
}

fn model_instance_definition() -> ModelInstanceDefinition {
    ModelInstanceDefinition {
        artifact: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
        architecture: ModelArchitectureImplementation {
            architecture: ModelArchitecture::new("qwen", "qwen2"),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        },
        residencies: BTreeSet::from([ModelResidencyId::new(1)]),
        tokenizer: None,
        placement: ModelInstancePlacement::new(ResourceAffinity::new(FallbackClass::Transparent)),
        policy: ModelInstancePolicy::default(),
        adapter_state: ModelInstanceAdapterState::default(),
        associated_sessions: BTreeSet::new(),
        usage: ModelInstanceUsage::default(),
        compute_dtype: None,
        mutation_version: 0,
        tenant: None,
        owner: None,
        resource_bindings: ModelInstanceResourceBindings::default(),
        kernel_selection_policy: None,
        required_weight_names: BTreeSet::new(),
        required_weight_digests: BTreeMap::new(),
        required_weight_shapes: BTreeMap::new(),
    }
}

#[test]
fn inference_api_model_instance_suspend_resume_drain_through_api_boundary() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    // `create()` no longer reaches Ready on its own (transactional-weight-
    // materialization); this test's own concern is suspend/resume/drain
    // *from* Ready, so reach Ready explicitly first -- with real,
    // Provider-backed evidence: `resume_model_instance` now re-derives
    // readiness (Correctif: Runtime-owned ModelInstance readiness
    // authority, round 3), so an instance that was never really
    // materialized correctly cannot resume back to Ready any more, and
    // this test's own concern (the suspend/resume/drain state machine,
    // not weight materialization) needs a genuinely resumable instance
    // to exercise that machinery at all.
    reach_ready_with_real_weight(&mut runtime, &instance);

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.raw_provider_handle_available);
    assert!(!status.raw_device_handle_available);
    assert!(!status.raw_weights_available);

    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )
    .unwrap();
    resume_model_instance(&mut runtime, &instance).unwrap();
    drain_model_instance(&mut runtime, &instance).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Draining);
}

#[test]
fn inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    // `create()` no longer reaches Ready on its own (transactional-weight-
    // materialization); this test's own concern is warmup's conflict
    // detection against an *already Ready* instance, so reach Ready
    // explicitly first. Bind a weight resource before doing so: since
    // `warm_model_instance` now derives `weights_materialized` from
    // `resource_bindings.weights` non-emptiness (`runtime-owned-model-
    // instance-readiness-authority`), an instance with none would make the
    // later `warm_model_instance` call fail on that check instead of the
    // already-Ready conflict this test actually means to exercise -- both
    // happen to map to the same broad `ModelInstanceUnavailable` variant,
    // which would silently let this test pass for the wrong reason.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("test.weight"));
    runtime.model_instances_mut().mark_ready(&instance).unwrap();
    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    let checks = ModelInstanceReadinessChecks {
        residency_available: true,
        provider_ready: true,
        device_ready: true,
        adapter_ready: true,
        memory_pressure: MemoryPressureLevel::Low,
        runtime_policy_allows: true,
        browser_supported: true,
        kernel_preparation_ready: true,
        autotuning_ready: true,
        weights_materialized: true,
    };

    let error = warm_model_instance(&mut runtime, &instance, &plan, &checks).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelInstanceUnavailable { .. }
    ));
}
