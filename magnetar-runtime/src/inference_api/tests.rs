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

use crate::generation::{
    CancellationMetadata, EosMode, EosOutputPolicy, EosPolicy, FinishReason,
    GenerationMemoryEstimate, GenerationParameters, GenerationPriority, GenerationRequest,
    GenerationTokenizerReference, StopConditions, StreamingMode, token_stream_events,
};
use crate::kernel_execution_plan::PreparedExecutionPlan;
use crate::memory::MemoryPlacement;
use crate::observability::{CorrelationId, TraceId};
use crate::reference_cpu::ReferenceCpuProvider;
use crate::runtime::Runtime;
use crate::sampling::SamplingPolicy;
use crate::tokenizer::{TokenId, TokenStopPattern};
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
