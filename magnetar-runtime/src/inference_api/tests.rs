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
