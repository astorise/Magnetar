//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::memory::MemoryPlacement;
use crate::model::ModelDigest;
use crate::observability::{CorrelationId, TraceId};
use crate::sampling::SamplingPolicy;
use crate::scheduler::ProviderExecutionErrorCode;
use crate::tokenizer::{
    SpecialToken, SpecialTokenKind, TokenIdRange, TokenStopPattern, TokenizerArtifactId,
    TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerRevision,
};

use crate::memory::{MemoryAdmissionDecision, MemoryManager, MemoryManagerConfig};
use crate::model::ModelArtifactKind;
use crate::tokenizer::{
    FixtureTokenizer, RuntimeTokenizer, StreamingDecodeState, TokenId, TokenizerArtifactReference,
    TokenizerArtifactSet,
};
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

#[test]
fn generation_parameters_validate_temperature_sampling_and_greedy_modes() {
    let mut invalid = generation_request();
    invalid.parameters.temperature = f32::NAN;
    assert!(matches!(
        invalid.validate(),
        Err(GenerationError::ParameterInvalid {
            parameter: "temperature",
            ..
        })
    ));

    let mut greedy = generation_request();
    greedy.parameters = GenerationParameters::greedy();
    greedy.validate().unwrap();
    assert!(!greedy.parameters.sampling_enabled);
}

#[test]
fn generation_stop_conditions_distinguish_length_eos_token_and_sequences() {
    let request = generation_request();

    assert_eq!(
        stop_reason_for(&request, &[1, 2, 3, 4]),
        Some(FinishReason::MaxNewTokens)
    );
    assert_eq!(
        stop_reason_for(&request, &[299]),
        Some(FinishReason::EosToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[298]),
        Some(FinishReason::StopToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 10, 11]),
        Some(FinishReason::StopSequence)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 121, 122]),
        Some(FinishReason::StopSequence)
    );
}

#[test]
fn generation_decode_step_delegates_next_token_selection_to_sampling() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let mut logits = vec![0.0; request.tokenizer.metadata.vocabulary_size as usize];
    logits[21] = 10.0;

    let (sampling, step) =
        decode_step_from_sampling(&request, &[20, 21], logits, SamplingPolicy::default()).unwrap();

    assert_eq!(sampling.selected_token_id, 22);
    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_provider_errors_map_to_finish_reasons() {
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionInterrupted),
        FinishReason::Interrupted
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::OutOfMemory),
        FinishReason::MemoryLimit
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionFailed),
        FinishReason::ProviderError
    );
}

fn generation_runtime_tokenizer() -> RuntimeTokenizer<FixtureTokenizer> {
    let metadata = generation_tokenizer_metadata();
    let digest = metadata.digest.clone();
    RuntimeTokenizer::new(
        FixtureTokenizer::new(metadata),
        TokenizerArtifactSet {
            tokenizer: TokenizerArtifactReference::new(
                TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
                ModelArtifactKind::Tokenizer,
                digest,
            )
            .unwrap(),
            tokenizer_config: None,
            vocabulary: None,
            special_tokens: None,
        },
    )
}

#[test]
fn generation_can_ignore_eos_by_explicit_policy() {
    let mut request = generation_request();
    request.stop_conditions.eos.mode = EosMode::Ignore;

    assert_eq!(stop_reason_for(&request, &[299]), None);
}

#[test]
fn generation_decode_step_preserves_token_index_and_state_boundary() {
    let request = generation_request();
    let step = decode_step(&request, &[20, 21], 22).unwrap();

    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_prefill_validates_tokens_and_records_prompt_count() {
    let request = generation_request();
    let state = prefill(&request).unwrap();

    assert_eq!(state.prompt_token_count, 3);
    assert!(state.kv_cache_placeholder.is_some());
    assert!(state.observations.iter().any(|event| {
        event.kind == GenerationEventKind::PrefillStarted && event.request_id == request.request_id
    }));
}

#[test]
fn generation_token_stream_events_preserve_order_and_identity() {
    let request = generation_request();
    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();

    assert_eq!(events.len(), 3);
    assert_eq!(events[0].token_id, Some(10));
    assert_eq!(events[1].token_index, Some(1));
    assert!(
        events
            .iter()
            .all(|event| event.request_id == request.request_id)
    );
}

#[test]
fn generation_streaming_text_uses_tokenizer_decode() {
    let tokenizer = generation_runtime_tokenizer();
    let output = streaming_text_chunk(
        &tokenizer,
        StreamingDecodeState::default(),
        vec![b'h' as TokenId + 1, b'i' as TokenId + 1],
        true,
    )
    .unwrap();

    assert_eq!(output.text, "hi");
    assert!(output.pending_partial_state.is_none());
}

#[test]
fn generation_prepares_text_stop_sequences_through_tokenizer() {
    let tokenizer = generation_runtime_tokenizer();
    let patterns = prepare_stop_sequences(&tokenizer, &["xy".into()]).unwrap();

    assert_eq!(patterns[0].text, "xy");
    assert_eq!(
        patterns[0].token_ids,
        vec![b'x' as TokenId + 1, b'y' as TokenId + 1]
    );
}

#[test]
fn generation_usage_and_output_account_for_tokens_without_decoded_text() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11], FinishReason::StopToken);

    output.validate().unwrap();
    assert_eq!(output.generated_token_count, 2);
    assert_eq!(output.usage.prompt_tokens, 3);
    assert_eq!(output.usage.total_tokens, 5);
}

#[test]
fn generation_cancellation_maps_to_stable_finish_reason() {
    let mut request = generation_request();
    request.cancellation.requested = true;

    assert_eq!(
        stop_reason_for(&request, &[]),
        Some(FinishReason::Cancelled)
    );
}

#[test]
fn generation_memory_admission_uses_memory_manager_policy() {
    let mut request = generation_request();
    request.memory.logits_buffer_bytes = 1024;
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(64),
        ..MemoryManagerConfig::default()
    });

    assert!(matches!(
        memory_admission(&request, &manager).unwrap(),
        MemoryAdmissionDecision::Reject { .. }
    ));
}
