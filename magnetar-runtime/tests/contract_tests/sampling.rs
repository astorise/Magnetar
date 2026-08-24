use magnetar_runtime::{
    GenerationParameters, LogitsProcessorAuthority, LogitsProcessorConfig, LogitsProcessorKind,
    LogitsReference, MemoryManager, MemoryManagerConfig, SamplingDiagnosticKind, SamplingError,
    SamplingRequest, SamplingRequestId, SamplingRngState, SamplingSelectionMode,
    SamplingStopMetadata, SpecialToken, SpecialTokenKind, TemperatureZeroPolicy, TokenIdRange,
    TokenizerArtifactId, TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerRevision,
    processor_order, sampling_workspace_requests, select_next_token,
};

fn digest() -> magnetar_runtime::ModelDigest {
    magnetar_runtime::ModelDigest::parse(
        "sha256:0000000000000000000000000000000000000000000000000000000000000001",
    )
    .unwrap()
}

fn metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("sampling.tokenizer").unwrap(),
        artifact: TokenizerArtifactId::new("tokenizer").unwrap(),
        digest: digest(),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("r1").unwrap(),
        vocabulary_size: 8,
        added_token_count: 1,
        token_id_range: TokenIdRange::new(0, 7),
        model_max_length: Some(16),
        special_tokens: vec![
            SpecialToken::new(SpecialTokenKind::Bos, "<s>", 0),
            SpecialToken::new(SpecialTokenKind::Eos, "</s>", 1),
            SpecialToken::new(SpecialTokenKind::Pad, "<pad>", 2),
            SpecialToken::new(SpecialTokenKind::Unknown, "<unk>", 3),
        ],
        additional_special_tokens: vec![SpecialToken::new(
            SpecialTokenKind::Additional,
            "<tool>",
            7,
        )],
        byte_fallback: true,
        normalization: Some("identity".into()),
        pre_tokenizer: Some("bytes".into()),
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    }
}

fn request(scores: Vec<f32>) -> SamplingRequest {
    let mut request = SamplingRequest::host_scores(
        SamplingRequestId::new("sampling.request").unwrap(),
        scores,
        metadata(),
    );
    request.parameters = GenerationParameters::greedy();
    request
}

#[test]
fn sampling_greedy_selects_highest_valid_token_without_decoding() {
    let mut request = request(vec![0.0, 1.0, 9.0, 2.0, 4.0, 6.0, 5.0, 8.0]);
    request.policy.allow_probability_metadata = true;

    let result = select_next_token(&request).unwrap();

    assert_eq!(result.selected_token_id, 5);
    assert_eq!(result.selection_mode, SamplingSelectionMode::Greedy);
    assert_eq!(result.token_rank, Some(1));
    assert!(result.token_probability.is_some());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == SamplingDiagnosticKind::RedactedLogits)
    );
}

#[test]
fn sampling_validates_temperature_and_reserved_modes() {
    let mut invalid_temperature = request(vec![0.0; 8]);
    invalid_temperature.parameters = GenerationParameters {
        temperature: 0.0,
        greedy: false,
        sampling_enabled: true,
        ..GenerationParameters::default()
    };
    invalid_temperature.policy.temperature_zero = TemperatureZeroPolicy::Invalid;
    assert!(matches!(
        select_next_token(&invalid_temperature),
        Err(SamplingError::TemperatureInvalid { .. })
    ));

    let mut typical = request(vec![0.0; 8]);
    typical.parameters.typical_p = Some(0.9);
    assert!(matches!(
        select_next_token(&typical),
        Err(SamplingError::TypicalPUnsupported)
    ));
}

#[test]
fn sampling_applies_top_k_top_p_and_penalties_in_stable_order() {
    let mut top_k = request(vec![0.0, 0.1, 0.2, 0.3, 4.0, 3.0, 2.0, 1.0]);
    top_k.parameters.top_k = Some(2);
    assert_eq!(select_next_token(&top_k).unwrap().selected_token_id, 4);

    let mut penalty = request(vec![0.0, 0.1, 0.2, 0.3, 4.0, 3.9, 2.0, 1.0]);
    penalty.token_history = vec![4, 4];
    penalty.parameters.repetition_penalty = Some(1.0);
    penalty.parameters.top_k = Some(2);
    assert_eq!(select_next_token(&penalty).unwrap().selected_token_id, 5);

    let order = processor_order(&penalty.parameters);
    assert!(
        order
            .iter()
            .position(|kind| *kind == LogitsProcessorKind::RepetitionPenalty)
            < order
                .iter()
                .position(|kind| *kind == LogitsProcessorKind::TopK)
    );
}

#[test]
fn sampling_enforces_banned_allowed_special_and_minimum_length_constraints() {
    let mut request = request(vec![0.0, 9.0, 8.0, 7.0, 3.0, 4.0, 5.0, 10.0]);
    request.step_index = 1;
    request.banned_token_ids = vec![6];
    request.parameters.allowed_token_ids = Some(vec![1, 4, 5, 6]);
    request.stop = SamplingStopMetadata {
        eos_token_ids: vec![1],
        stop_token_ids: vec![5],
        minimum_generated_tokens: Some(3),
        mask_stop_tokens_before_minimum: true,
    };

    let result = select_next_token(&request).unwrap();

    assert_eq!(result.selected_token_id, 4);
    assert_eq!(result.finish_hint, None);
}

#[test]
fn sampling_reports_no_eligible_token_and_vocab_mismatch() {
    let mut none = request(vec![1.0; 8]);
    none.parameters.allowed_token_ids = Some(vec![2]);
    assert!(matches!(
        select_next_token(&none),
        Err(SamplingError::NoEligibleToken)
    ));

    let mut mismatch = request(vec![1.0; 7]);
    mismatch.vocabulary_size = 8;
    assert!(matches!(
        select_next_token(&mismatch),
        Err(SamplingError::VocabularyMismatch { .. })
    ));
}

#[test]
fn sampling_keeps_provider_and_device_logits_opaque() {
    let mut request = request(vec![1.0; 8]);
    request.logits = Some(LogitsReference::RuntimeTensor {
        id: "runtime.logits".into(),
    });
    request.policy.allow_logits_materialization = false;

    assert!(matches!(
        select_next_token(&request),
        Err(SamplingError::LogitsMaterializationDenied)
    ));
}

#[test]
fn sampling_processors_are_inference_scoped() {
    let mut request = request(vec![0.0, 1.0, 2.0, 3.0, 8.0, 4.0, 5.0, 6.0]);
    request.processors = vec![LogitsProcessorConfig {
        id: "bad.processor".into(),
        kind: LogitsProcessorKind::Custom,
        authority: LogitsProcessorAuthority {
            filesystem: true,
            ..LogitsProcessorAuthority::default()
        },
    }];

    assert!(matches!(
        select_next_token(&request),
        Err(SamplingError::ProcessorUnsupported { .. })
    ));
}

#[test]
fn sampling_stochastic_mode_uses_seed_and_opaque_rng_state() {
    let mut request = request(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    request.parameters = GenerationParameters {
        temperature: 1.0,
        greedy: false,
        sampling_enabled: true,
        seed: Some(7),
        ..GenerationParameters::default()
    };
    request.rng_seed = Some(7);

    let result = select_next_token(&request).unwrap();

    assert_eq!(result.selection_mode, SamplingSelectionMode::Stochastic);
    assert!(
        result
            .updated_rng_state
            .as_ref()
            .unwrap()
            .expose()
            .is_none()
    );
    assert!(
        SamplingRngState::inspectable(vec![1, 2, 3])
            .expose()
            .is_some()
    );
}

#[test]
fn sampling_workspace_uses_memory_manager_policy() {
    let mut request = request(vec![1.0; 8]);
    request.parameters.top_p = Some(0.9);
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(8),
        ..MemoryManagerConfig::default()
    });

    assert!(matches!(
        sampling_workspace_requests(&request, &manager),
        Err(SamplingError::MemoryAllocationFailed { .. })
    ));
}
