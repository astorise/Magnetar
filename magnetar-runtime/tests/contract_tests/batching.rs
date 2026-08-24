use magnetar_runtime::{
    BatchId, BatchMemoryEstimate, BatchObservationKind, BatchPhase, BatchSchedulingMode,
    BatchSlotId, BatchedOperationState, BatchingError, BatchingErrorCode, BatchingPolicy,
    CancellationMetadata, GenerationMemoryEstimate, GenerationModelReference, GenerationParameters,
    GenerationPriority, GenerationRequest, GenerationRequestId, GenerationTokenizerReference,
    KvCacheId, MemoryAdmissionDecision, MemoryAllocationClass, MemoryManagerConfig,
    MemoryPlacement, ModelArtifactId, ModelArtifactKind, ModelDigest, ModelName, ModelRevision,
    Runtime, RuntimeConfig, SpecialToken, SpecialTokenKind, StopConditions, StreamingMode,
    TokenIdRange, TokenizerArtifactId, TokenizerFamily, TokenizerId, TokenizerMetadata,
    TokenizerRevision,
};

fn model_reference(name: &str) -> GenerationModelReference {
    GenerationModelReference::ModelArtifact(ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new(name).unwrap(),
        ModelRevision::new("r1").unwrap(),
        ModelDigest::parse(
            "sha256:0000000000000000000000000000000000000000000000000000000000000001",
        )
        .unwrap(),
    ))
}

fn tokenizer_reference(name: &str) -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: TokenizerId::new(name).unwrap(),
        metadata: TokenizerMetadata {
            id: TokenizerId::new(name).unwrap(),
            artifact: TokenizerArtifactId::new("batching-tokenizer-artifact").unwrap(),
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

fn generation_request(id: &str, model: &str, tokenizer: &str) -> GenerationRequest {
    GenerationRequest {
        request_id: GenerationRequestId::new(id).unwrap(),
        session: None,
        model: model_reference(model),
        tokenizer: tokenizer_reference(tokenizer),
        input_token_ids: vec![1, 2, 3],
        prompt_token_count: 3,
        max_new_tokens: 4,
        max_total_tokens: Some(16),
        model_context_length: Some(32),
        parameters: GenerationParameters::default(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::TokenIds,
        priority: GenerationPriority {
            priority: 7,
            deadline_millis: Some(100),
        },
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 24,
            output_token_buffer_bytes: 24,
            logits_buffer_bytes: 128,
            sampling_workspace_bytes: 64,
            prefill_workspace_bytes: 64,
            decode_workspace_bytes: 64,
            kv_cache_placeholder_bytes: 128,
            prefix_cache_placeholder_bytes: 16,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: true,
        },
        correlation_id: None,
        trace_id: None,
    }
}

#[test]
fn batch_id_and_slot_id_are_opaque_and_not_authority() {
    assert!(BatchId::new("client-batch").is_ok());
    assert!(BatchId::new("provider:0x1234").is_err());
    assert!(BatchSlotId::new("device-slot").is_err());

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());

    assert!(batch.as_str().starts_with("batch-"));
    assert!(matches!(
        runtime
            .batching()
            .batch(&BatchId::new("batch-999").unwrap()),
        Err(BatchingError::OperationNotFound)
    ));
}

#[test]
fn operation_lifecycle_enforces_prefill_decode_streaming_and_completion() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let request = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let slot = runtime.admit_generation_to_batch(&batch, &request).unwrap();

    let prefill = runtime.schedule_batch_prefill(&batch, 8).unwrap();
    assert_eq!(prefill.phase, BatchPhase::Prefill);
    assert_eq!(prefill.slots, vec![slot.clone()]);
    assert_eq!(
        runtime.batching().slot(&slot).unwrap().state,
        BatchedOperationState::Prefilling
    );

    runtime
        .batching_mut()
        .transition_slot(&slot, BatchedOperationState::DecodePending)
        .unwrap();
    let decode = runtime.schedule_batch_decode(&batch, 8).unwrap();
    assert_eq!(decode.phase, BatchPhase::Decode);

    runtime
        .batching_mut()
        .transition_slot(&slot, BatchedOperationState::Streaming)
        .unwrap();
    runtime
        .batching_mut()
        .record_streamed_token(&slot, 0)
        .unwrap();
    runtime
        .batching_mut()
        .transition_slot(&slot, BatchedOperationState::Completed)
        .unwrap();

    assert_eq!(
        runtime.batching().slot(&slot).unwrap().state,
        BatchedOperationState::Completed
    );
}

#[test]
fn batching_rejects_incompatible_model_or_tokenizer_in_same_execution_group() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let first = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let wrong_model = generation_request("batch-op-b", "other-model", "batch-tokenizer");
    let wrong_tokenizer = generation_request("batch-op-c", "batch-model", "other-tokenizer");

    runtime.admit_generation_to_batch(&batch, &first).unwrap();
    assert!(matches!(
        runtime.admit_generation_to_batch(&batch, &wrong_model),
        Err(BatchingError::ModelIncompatible)
    ));
    assert!(matches!(
        runtime.admit_generation_to_batch(&batch, &wrong_tokenizer),
        Err(BatchingError::TokenizerIncompatible)
    ));
}

#[test]
fn memory_admission_uses_memory_manager_without_scheduler_allocation() {
    let runtime = Runtime::initialize(RuntimeConfig {
        memory: MemoryManagerConfig {
            max_runtime_bytes: Some(1024),
            ..MemoryManagerConfig::default()
        },
        ..RuntimeConfig::default()
    });
    let mut runtime = runtime;
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let estimate = BatchMemoryEstimate {
        input_buffer_bytes: 64,
        output_buffer_bytes: 64,
        logits_buffer_bytes: 128,
        attention_mask_bytes: 32,
        position_buffer_bytes: 32,
        sampling_workspace_bytes: 64,
        kv_cache_block_bytes: 128,
        prefix_cache_lookup_bytes: 32,
        temporary_staging_bytes: 64,
        provider_workspace_bytes: 64,
        placement: MemoryPlacement::HostOrdinary,
        queue_allowed: false,
    };
    let request = runtime
        .batching()
        .memory_admission_request(&batch, &estimate)
        .unwrap();

    assert_eq!(
        request.allocation.class,
        MemoryAllocationClass::TemporaryWorkspace
    );
    assert!(matches!(
        runtime.batch_memory_admission(&batch, &estimate).unwrap(),
        MemoryAdmissionDecision::Admit { .. }
    ));
}

#[test]
fn kv_cache_and_prefix_cache_are_references_not_raw_owned_state() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let request = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let slot = runtime.admit_generation_to_batch(&batch, &request).unwrap();
    let cache = KvCacheId::new("kv-cache-reference").unwrap();

    runtime
        .batching_mut()
        .assign_kv_cache(&slot, cache.clone())
        .unwrap();

    assert_eq!(
        runtime.batching().slot(&slot).unwrap().kv_cache,
        Some(cache)
    );
    assert!(
        runtime
            .batching()
            .observations()
            .iter()
            .any(|event| event.kind == BatchObservationKind::KvCacheAssigned)
    );
}

#[test]
fn streaming_order_and_slow_consumer_policy_are_per_operation() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let request = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let slot = runtime.admit_generation_to_batch(&batch, &request).unwrap();

    runtime
        .batching_mut()
        .record_streamed_token(&slot, 0)
        .unwrap();
    assert!(matches!(
        runtime.batching_mut().record_streamed_token(&slot, 2),
        Err(BatchingError::StreamingBackpressure { .. })
    ));
}

#[test]
fn cancellation_and_failure_are_isolated_by_slot() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let first = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let second = generation_request("batch-op-b", "batch-model", "batch-tokenizer");
    let slot_a = runtime.admit_generation_to_batch(&batch, &first).unwrap();
    let slot_b = runtime.admit_generation_to_batch(&batch, &second).unwrap();

    runtime.batching_mut().cancel_slot(&slot_a).unwrap();
    runtime.batching_mut().fail_slot(&slot_b).unwrap();

    assert_eq!(
        runtime.batching().slot(&slot_a).unwrap().state,
        BatchedOperationState::Cancelled
    );
    assert_eq!(
        runtime.batching().slot(&slot_b).unwrap().state,
        BatchedOperationState::Failed
    );
    assert!(
        runtime
            .batching()
            .observations()
            .iter()
            .all(|event| !event.raw_prompt_available
                && !event.raw_logits_available
                && !event.raw_kv_cache_available
                && !event.raw_provider_handle_available)
    );
}

#[test]
fn scheduling_policy_limits_queue_and_reports_stable_errors() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let batch = runtime.create_continuous_batch(BatchingPolicy {
        mode: BatchSchedulingMode::Priority,
        max_batch_sequences: 1,
        ..BatchingPolicy::default()
    });
    let first = generation_request("batch-op-a", "batch-model", "batch-tokenizer");
    let second = generation_request("batch-op-b", "batch-model", "batch-tokenizer");

    runtime.admit_generation_to_batch(&batch, &first).unwrap();
    let error = runtime
        .admit_generation_to_batch(&batch, &second)
        .unwrap_err();

    assert_eq!(error.code(), BatchingErrorCode::QueueFull);
}
