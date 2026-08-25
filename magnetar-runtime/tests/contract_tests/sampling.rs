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

/// Builds a stochastic request over `n` equiprobable tokens.
///
/// `step_index` starts at 1 in these tests so BOS suppression is constant
/// across steps: at step 0 the BOS token is still selectable, which would
/// otherwise show up as "variation" that has nothing to do with the RNG.
fn stochastic_request(seed: u64, step_index: usize) -> SamplingRequest {
    let mut request = request(vec![1.0; 8]);
    request.parameters = GenerationParameters {
        temperature: 1.0,
        greedy: false,
        sampling_enabled: true,
        seed: Some(seed),
        ..GenerationParameters::default()
    };
    request.rng_seed = Some(seed);
    request.step_index = step_index;
    request
}

/// Asserts a run of draws looks like sampling rather than a constant.
///
/// The fixture's eight tokens are not all selectable: policy masks BOS after
/// the first step, plus pad, unknown and the additional special token, leaving
/// four. A working sampler visits all four at roughly even frequency; a
/// sampler stuck on one CDF quantile returns exactly one token every time.
/// Both bounds below are far from the expected values (4 distinct, 25% share),
/// so this stays deterministic rather than flaky.
fn assert_distribution_is_spread(draws: &[u32], context: &str) {
    let mut frequency = std::collections::BTreeMap::new();
    for token_id in draws {
        *frequency.entry(*token_id).or_insert(0_usize) += 1;
    }
    let distinct = frequency.len();
    let most_common = frequency.values().copied().max().unwrap_or(0);

    assert!(
        distinct >= 4,
        "{context}: only {distinct} distinct token(s) over {} draws ({frequency:?})",
        draws.len()
    );
    assert!(
        most_common * 2 < draws.len(),
        "{context}: one token took {most_common} of {} draws ({frequency:?})",
        draws.len()
    );
}

/// A fixed seed must not pin every step to the same point in the CDF.
///
/// Before the sampling RNG became a stream, the seed resolved to the same
/// value on every step, so the threshold was constant and every step with the
/// same logits picked the same token. Over 128 draws from 7 selectable
/// equiprobable tokens, a working sampler hits essentially all of them; the
/// old behaviour hit exactly one.
#[test]
fn sampling_stochastic_mode_varies_across_steps_under_a_fixed_seed() {
    let selected = (1..=128)
        .map(|step_index| {
            let result = select_next_token(&stochastic_request(7, step_index)).unwrap();
            assert_eq!(result.selection_mode, SamplingSelectionMode::Stochastic);
            result.selected_token_id
        })
        .collect::<Vec<_>>();

    assert_distribution_is_spread(&selected, "fixed seed did not advance the sampling stream");
}

/// Threading `updated_rng_state` must resume the stream, not restart it.
///
/// `step_index` is held constant here, so the threaded state is the only thing
/// that can carry the stream forward.
#[test]
fn sampling_threaded_rng_state_advances_the_stream() {
    let mut state = None;
    let selected = (0..128)
        .map(|_| {
            let mut request = stochastic_request(11, 1);
            request.rng_state = state.take();
            let result = select_next_token(&request).unwrap();
            state = result.updated_rng_state.clone();
            result.selected_token_id
        })
        .collect::<Vec<_>>();

    assert_distribution_is_spread(&selected, "threaded rng state did not advance the stream");
}

/// Advancing the stream must not cost reproducibility: the same seed and the
/// same step must still yield the same token.
#[test]
fn sampling_stochastic_mode_is_reproducible_for_a_given_seed() {
    for step_index in 1..=32 {
        let first = select_next_token(&stochastic_request(99, step_index)).unwrap();
        let second = select_next_token(&stochastic_request(99, step_index)).unwrap();
        assert_eq!(first.selected_token_id, second.selected_token_id);
        assert_eq!(first.updated_rng_state, second.updated_rng_state);
    }
}

/// Two different seeds must not produce the same stream.
#[test]
fn sampling_stochastic_mode_separates_streams_by_seed() {
    let draw = |seed: u64| {
        (1..=64)
            .map(|step_index| {
                select_next_token(&stochastic_request(seed, step_index))
                    .unwrap()
                    .selected_token_id
            })
            .collect::<Vec<_>>()
    };

    assert_ne!(draw(3), draw(4));
}
