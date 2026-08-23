use magnetar_runtime::{
    BatchEncodeInput, ComponentDigest, DecodeInput, EncodeInput, FixtureTokenizer,
    InferenceArtifactKind, InferenceArtifactReference, InferenceArtifactRegistry, MemoryManager,
    MemoryManagerConfig, MemoryPlacement, ModelArtifactKind, ModelDigest, ModelManifest,
    PaddingPolicy, RuntimeTokenizer, SpecialToken, SpecialTokenKind, TokenIdRange, Tokenizer,
    TokenizerArtifactId, TokenizerArtifactReference, TokenizerArtifactSet, TokenizerCompatibility,
    TokenizerError, TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerObservationKind,
    TokenizerObserver, TokenizerRevision, TruncationPolicy, tokenizer_memory_feasibility,
};

fn digest() -> ModelDigest {
    ModelDigest::parse("sha256:0000000000000000000000000000000000000000000000000000000000000001")
        .unwrap()
}

fn metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("fixture.tokenizer").unwrap(),
        artifact: TokenizerArtifactId::new("tokenizer").unwrap(),
        digest: digest(),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("r1").unwrap(),
        vocabulary_size: 512,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(0, 1024),
        model_max_length: Some(8),
        special_tokens: vec![
            SpecialToken::new(SpecialTokenKind::Bos, "<s>", 0),
            SpecialToken::new(SpecialTokenKind::Eos, "</s>", 256),
            SpecialToken::new(SpecialTokenKind::Pad, "<pad>", 257),
        ],
        additional_special_tokens: vec![SpecialToken::new(SpecialTokenKind::Mask, "<mask>", 258)],
        byte_fallback: true,
        normalization: Some("identity".into()),
        pre_tokenizer: Some("bytes".into()),
        supports_offsets: true,
        supports_token_type_ids: true,
        supports_browser: true,
    }
}

fn artifact_set() -> TokenizerArtifactSet {
    TokenizerArtifactSet {
        tokenizer: TokenizerArtifactReference::new(
            TokenizerArtifactId::new("tokenizer").unwrap(),
            ModelArtifactKind::Tokenizer,
            digest(),
        )
        .unwrap(),
        tokenizer_config: Some(
            TokenizerArtifactReference::new(
                TokenizerArtifactId::new("tokenizer-config").unwrap(),
                ModelArtifactKind::TokenizerConfig,
                digest(),
            )
            .unwrap(),
        ),
        vocabulary: Some(
            TokenizerArtifactReference::new(
                TokenizerArtifactId::new("vocabulary").unwrap(),
                ModelArtifactKind::Vocabulary,
                digest(),
            )
            .unwrap(),
        ),
        special_tokens: Some(
            TokenizerArtifactReference::new(
                TokenizerArtifactId::new("special-tokens").unwrap(),
                ModelArtifactKind::SpecialTokens,
                digest(),
            )
            .unwrap(),
        ),
    }
}

fn registry() -> InferenceArtifactRegistry {
    let mut registry = InferenceArtifactRegistry::default();
    for id in [
        "tokenizer",
        "tokenizer-config",
        "vocabulary",
        "special-tokens",
    ] {
        registry
            .register(
                InferenceArtifactReference::new(
                    InferenceArtifactKind::Tokenizer,
                    id,
                    ComponentDigest::sha256(id.as_bytes()),
                )
                .unwrap(),
            )
            .unwrap();
    }
    registry
}

fn model_manifest() -> ModelManifest {
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: tokenizer.model
  revision: r1
architecture:
  family: fixture
  identifier: fixture
artifacts:
  weights:
    kind: model-weights
    digest: {}
  config:
    kind: model-config
    digest: {}
  tokenizer:
    kind: tokenizer
    digest: {}
tokenizer: tokenizer
"#,
        digest().value,
        digest().value,
        digest().value,
        digest().value
    ))
    .unwrap()
}

#[test]
fn tokenizer_metadata_identity_and_special_tokens_validate() {
    let metadata = metadata();

    metadata.validate().unwrap();
    assert_eq!(
        metadata.special_token(SpecialTokenKind::Eos).unwrap().id,
        256
    );
    assert!(TokenizerId::new("C:\\tokenizer.json").is_err());
}

#[test]
fn tokenizer_artifacts_must_be_runtime_registered() {
    let artifacts = artifact_set();

    artifacts.validate_registered(&registry()).unwrap();
    assert!(matches!(
        artifacts.validate_registered(&InferenceArtifactRegistry::default()),
        Err(TokenizerError::TokenizerArtifactMissing { .. })
    ));
}

#[test]
fn tokenizer_runtime_loads_and_checks_model_compatibility() {
    let tokenizer = RuntimeTokenizer::new(FixtureTokenizer::new(metadata()), artifact_set());
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(digest()),
        expected_vocabulary_size: Some(512),
        expected_family: Some(TokenizerFamily::new("fixture").unwrap()),
        expected_model_max_length: Some(8),
        expected_added_tokens: Some(2),
        expected_special_tokens: vec![SpecialTokenKind::Eos, SpecialTokenKind::Pad],
        expected_normalization: Some("identity".into()),
    };
    let mut observer = TokenizerObserver::default();

    tokenizer.load(&registry(), &mut observer).unwrap();
    tokenizer
        .validate_for_model(&model_manifest(), &compatibility, &mut observer)
        .unwrap();

    let kinds = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect::<Vec<_>>();
    assert!(kinds.contains(&TokenizerObservationKind::Loaded));
    assert!(kinds.contains(&TokenizerObservationKind::CompatibilityChecked));
}

#[test]
fn tokenizer_rejects_incompatible_model_digest_and_missing_special_token() {
    let mut wrong_metadata = metadata();
    wrong_metadata.special_tokens.clear();

    let compatibility = TokenizerCompatibility {
        expected_digest: Some(digest()),
        expected_vocabulary_size: Some(512),
        expected_family: Some(TokenizerFamily::new("fixture").unwrap()),
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: vec![SpecialTokenKind::Eos],
        expected_normalization: None,
    };

    assert!(matches!(
        wrong_metadata.validate_compatibility(&compatibility),
        Err(TokenizerError::SpecialTokenMissing)
    ));
}

#[test]
fn tokenizer_encode_decode_offsets_masks_and_token_type_ids() {
    let tokenizer = RuntimeTokenizer::new(FixtureTokenizer::new(metadata()), artifact_set());
    let mut observer = TokenizerObserver::default();
    let mut input = EncodeInput::new("hi");
    input.add_special_tokens = true;
    input.return_offsets = true;

    let output = tokenizer.encode(input, &mut observer).unwrap();

    assert_eq!(output.token_count, output.token_ids.len());
    assert_eq!(
        output.attention_mask.as_ref().unwrap().len(),
        output.token_ids.len()
    );
    assert_eq!(
        output.token_type_ids.as_ref().unwrap().len(),
        output.token_ids.len()
    );

    let decoded = tokenizer
        .decode(
            DecodeInput {
                token_ids: output.token_ids,
                skip_special_tokens: true,
                clean_up_tokenization_spaces: false,
                streaming_state: None,
            },
            &mut observer,
        )
        .unwrap();

    assert_eq!(decoded.text, "hi");
    assert!(observer.observations().iter().all(|observation| {
        !observation.message.contains("hi") && !observation.message.contains("prompt")
            || observation.kind == TokenizerObservationKind::PromptTooLong
    }));
}

#[test]
fn tokenizer_rejects_invalid_token_id_and_unsupported_offsets() {
    let tokenizer = FixtureTokenizer::new(metadata());
    assert!(matches!(
        tokenizer.decode(DecodeInput {
            token_ids: vec![2048],
            skip_special_tokens: false,
            clean_up_tokenization_spaces: false,
            streaming_state: None,
        }),
        Err(TokenizerError::InvalidTokenId { token_id: 2048 })
    ));

    let mut metadata = metadata();
    metadata.supports_offsets = false;
    let tokenizer = FixtureTokenizer::new(metadata);
    let mut input = EncodeInput::new("hi");
    input.return_offsets = true;
    assert!(matches!(
        tokenizer.encode(input),
        Err(TokenizerError::OffsetsUnsupported)
    ));
}

#[test]
fn tokenizer_truncation_padding_batch_and_stop_patterns_are_explicit() {
    let tokenizer = FixtureTokenizer::new(metadata());
    let mut overlong = EncodeInput::new("too long for limit");
    overlong.max_tokens = Some(3);
    assert!(matches!(
        tokenizer.encode(overlong),
        Err(TokenizerError::PromptTooLong { .. })
    ));

    let mut truncated = EncodeInput::new("abcdef");
    truncated.max_tokens = Some(3);
    truncated.truncation = TruncationPolicy::Right;
    let output = tokenizer.encode(truncated).unwrap();
    assert_eq!(output.token_ids.len(), 3);
    assert!(
        output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("truncation"))
    );

    let batch = tokenizer.batch_encode(BatchEncodeInput {
        inputs: vec![EncodeInput::new("a"), EncodeInput::new("b")],
        padding: PaddingPolicy::Longest,
    });
    assert_eq!(batch.outputs.len(), 2);
    assert_eq!(batch.outputs[0].as_ref().unwrap().token_ids, vec![98]);
    assert_eq!(batch.outputs[1].as_ref().unwrap().token_ids, vec![99]);

    let stop = tokenizer.resolve_stop_sequence("stop").unwrap();
    assert!(stop.exact);
    assert_eq!(stop.text, "stop");
}

#[test]
fn tokenizer_memory_feasibility_uses_memory_manager_policy() {
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(16),
        ..MemoryManagerConfig::default()
    });

    let feasibility =
        tokenizer_memory_feasibility(&metadata(), &manager, MemoryPlacement::HostOrdinary).unwrap();

    assert!(!feasibility.feasible);
}
