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
