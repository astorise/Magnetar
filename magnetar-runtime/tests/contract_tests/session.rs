use magnetar_runtime::{
    CancellationMetadata, CorrelationId, GenerationMemoryEstimate, GenerationModelReference,
    GenerationParameters, GenerationPriority, GenerationRequest, GenerationRequestId,
    GenerationTokenizerReference, InferenceSessionId, MemoryManagerConfig, MemoryPlacement,
    ModelArtifactId, ModelArtifactKind, ModelDigest, ModelName, ModelRevision, Runtime,
    SessionAccessPolicy, SessionConcurrencyPolicy, SessionCreationRequest, SessionError,
    SessionLifecycleState, SessionMemoryBudget, SessionObservationKind, SessionOperationAdmission,
    SessionPolicy, SpecialToken, SpecialTokenKind, StopConditions, StreamingMode, TokenIdRange,
    TokenizerArtifactId, TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerRevision,
};

fn model_reference() -> GenerationModelReference {
    GenerationModelReference::ModelArtifact(ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("session-model").unwrap(),
        ModelRevision::new("r1").unwrap(),
        ModelDigest::parse(
            "sha256:0000000000000000000000000000000000000000000000000000000000000001",
        )
        .unwrap(),
    ))
}

fn tokenizer_reference() -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: TokenizerId::new("session-tokenizer").unwrap(),
        metadata: TokenizerMetadata {
            id: TokenizerId::new("session-tokenizer").unwrap(),
            artifact: TokenizerArtifactId::new("tokenizer-artifact").unwrap(),
            digest: ModelDigest::parse(
                "sha256:0000000000000000000000000000000000000000000000000000000000000001",
            )
            .unwrap(),
            revision: TokenizerRevision::new("r1").unwrap(),
            family: TokenizerFamily::new("fixture").unwrap(),
            vocabulary_size: 512,
            added_token_count: 2,
            token_id_range: TokenIdRange::new(0, 1024),
            model_max_length: Some(256),
            special_tokens: vec![
                SpecialToken::new(SpecialTokenKind::Bos, "<s>", 0),
                SpecialToken::new(SpecialTokenKind::Eos, "</s>", 256),
            ],
            additional_special_tokens: Vec::new(),
            byte_fallback: true,
            normalization: Some("identity".into()),
            pre_tokenizer: Some("bytes".into()),
            supports_offsets: true,
            supports_token_type_ids: true,
            supports_browser: true,
        },
    }
}

fn creation_request() -> SessionCreationRequest {
    SessionCreationRequest {
        model: model_reference(),
        tokenizer: tokenizer_reference(),
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy {
            max_prompt_tokens: Some(64),
            max_generated_tokens: Some(32),
            max_total_tokens: Some(96),
            memory_budget_bytes: Some(1024),
            idle_ttl_millis: Some(10),
            total_ttl_millis: Some(100),
            ..SessionPolicy::default()
        },
        memory: SessionMemoryBudget {
            input_token_buffer_bytes: 16,
            output_token_buffer_bytes: 16,
            temporary_workspace_bytes: 16,
            placement: MemoryPlacement::HostOrdinary,
            ..SessionMemoryBudget::default()
        },
        allowed_capabilities: Default::default(),
        correlation_id: Some(CorrelationId::new("session-correlation")),
        created_at_millis: 1,
    }
}

fn generation_request(session: Option<InferenceSessionId>) -> GenerationRequest {
    GenerationRequest {
        request_id: GenerationRequestId::new("session-generation").unwrap(),
        session,
        model: model_reference(),
        tokenizer: tokenizer_reference(),
        input_token_ids: vec![1, 2, 3],
        prompt_token_count: 3,
        max_new_tokens: 8,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters::default(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: None,
        trace_id: None,
    }
}

#[test]
fn session_creation_issues_runtime_owned_opaque_id_and_status_is_redacted() {
    let mut runtime = Runtime::initialize(Default::default());

    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();
    let access = SessionAccessPolicy::authorize(session.clone());
    let status = runtime.session_status(&session, &access).unwrap();

    assert!(session.as_str().starts_with("session-"));
    assert_eq!(status.lifecycle, SessionLifecycleState::Ready);
    assert!(!status.raw_prompt_available);
    assert!(!status.raw_handles_available);
    assert_eq!(
        runtime.session_observations()[0].kind,
        SessionObservationKind::CreateRequested
    );
}

#[test]
fn session_id_does_not_grant_authority_by_itself() {
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();
    let access = SessionAccessPolicy::authorize(InferenceSessionId::new("other-session").unwrap());

    assert!(matches!(
        runtime.session_status(&session, &access),
        Err(SessionError::Unauthorized)
    ));
}

#[test]
fn session_lifecycle_concurrency_and_drain_are_enforced() {
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();

    assert_eq!(
        runtime.start_session_operation(&session).unwrap(),
        SessionOperationAdmission::Started
    );
    assert!(matches!(
        runtime.start_session_operation(&session),
        Err(SessionError::ConcurrencyViolation)
    ));
    runtime.finish_session_operation(&session).unwrap();
    runtime.drain_inference_session(&session).unwrap();
    assert!(matches!(
        runtime.start_session_operation(&session),
        Err(SessionError::SessionDraining)
    ));
}

#[test]
fn queued_operation_policy_queues_second_operation() {
    let mut request = creation_request();
    request.policy.concurrency = SessionConcurrencyPolicy::QueueOperations;
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime.create_inference_session(request).unwrap();

    assert_eq!(
        runtime.start_session_operation(&session).unwrap(),
        SessionOperationAdmission::Started
    );
    assert_eq!(
        runtime.start_session_operation(&session).unwrap(),
        SessionOperationAdmission::Queued
    );
}

#[test]
fn session_policy_and_memory_are_applied_to_generation() {
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();
    let mut generation = generation_request(Some(session.clone()));

    runtime
        .apply_session_to_generation(&mut generation)
        .unwrap();
    let admission = runtime.session_memory_admission(&session).unwrap();

    assert_eq!(generation.max_total_tokens, Some(96));
    assert!(admission.is_admitted());
}

#[test]
fn session_policy_rejects_token_limit_violations() {
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();
    let mut generation = generation_request(Some(session));
    generation.max_new_tokens = 128;

    assert!(matches!(
        runtime.apply_session_to_generation(&mut generation),
        Err(SessionError::SessionPolicyDenied { .. })
    ));
}

#[test]
fn session_expiration_releases_transient_state() {
    let mut runtime = Runtime::initialize(Default::default());
    let session = runtime
        .create_inference_session(creation_request())
        .unwrap();

    let expired = runtime.expire_inference_sessions(200);
    let status = runtime
        .session_status(&session, &SessionAccessPolicy::authorize(session.clone()))
        .unwrap();

    assert_eq!(expired, vec![session]);
    assert_eq!(status.lifecycle, SessionLifecycleState::Expired);
    assert!(!status.streaming.streaming_decode_active);
}

#[test]
fn session_model_stays_platform_neutral() {
    let request = creation_request();
    let tokenizer = tokenizer_reference();

    assert_eq!(request.memory.placement, MemoryPlacement::HostOrdinary);
    assert!(tokenizer.metadata.supports_browser);
    assert_eq!(tokenizer.metadata.id.as_str(), "session-tokenizer");
}

#[test]
fn session_memory_budget_rejects_oversized_creation() {
    let mut request = creation_request();
    request.policy.memory_budget_bytes = Some(1);
    let mut runtime = Runtime::initialize(magnetar_runtime::RuntimeConfig {
        memory: MemoryManagerConfig {
            max_runtime_bytes: Some(64),
            ..MemoryManagerConfig::default()
        },
        ..magnetar_runtime::RuntimeConfig::default()
    });

    assert!(matches!(
        runtime.create_inference_session(request),
        Err(SessionError::MemoryBudgetExceeded { .. })
    ));
}
