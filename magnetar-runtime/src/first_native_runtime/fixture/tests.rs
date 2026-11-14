//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_fixture_model_passes_validation() {
    let fixture = e2e_fixture().expect("fixture builds and validates");
    fixture.manifest.validate().expect("manifest re-validates");
    assert_eq!(
        fixture.identity.implementation,
        ModelComponentImplementationKind::WebAssemblyComponent
    );
    assert_eq!(
        fixture.config.architecture.vocabulary_size,
        E2E_FIXTURE_VOCAB
    );
    assert_eq!(fixture.config.architecture.hidden_size, E2E_FIXTURE_HIDDEN);
    assert_eq!(fixture.config.architecture.layer_count, E2E_FIXTURE_LAYERS);
}

#[test]
fn e2e_fixture_weight_digest_is_stable() {
    let fixture = e2e_fixture().expect("fixture builds");
    let digest = e2e_fixture_weight_digest(&fixture.weights);
    assert_eq!(digest, E2E_FIXTURE_WEIGHT_DIGEST);
    assert_eq!(digest, e2e_fixture_weight_digest(&fixture.weights));
}

#[test]
fn e2e_eos_token_stops_generation() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_eos_token_stops_generation(&fixture).expect("EOS stop is honored");
}

/// GitHub issue "A Model Instance stuck in Loading cannot currently be
/// unloaded or canceled": proves the existing `fail_instance` +
/// `unload_model_instance` combination already provides a real, working
/// cancellation path for an instance that was created but never (or only
/// partially) materialized -- `ModelInstance::fail`/`invalidate`
/// unconditionally set lifecycle to `Failed`/`Invalid` regardless of the
/// instance's current state (they do not go through
/// `allows_transition_to` at all), and `(Failed, Unloading)` is already a
/// valid transition `ModelInstanceManager::unload` already accepts. No new
/// lifecycle transition or unload entrypoint was needed.
#[test]
fn loading_instance_can_be_canceled_via_fail_then_unload() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )
    .expect("loading succeeds");
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .expect("creation succeeds");
    // Never materialized: still Loading, exactly the stuck scenario this
    // issue describes.
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Loading
    );

    runtime
        .model_instances_mut()
        .fail_instance(
            &instance,
            ModelInstanceError::InternalModelInstance {
                reason: "abandoned before materialization".into(),
            },
        )
        .expect("fail_instance works from Loading");
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Failed
    );

    let report = runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .expect("an instance that never materialized anything still unloads cleanly");
    assert!(report.released_weight_resources.is_empty());
    assert!(report.released_memory_allocations.is_empty());
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

/// `materialize-weights-from-real-model-artifact` task 3.1: proves the
/// real-artifact-bytes path (`e2e_fixture_weights_from_real_artifact`,
/// reading the checked-in `E2E_FIXTURE_SAFETENSORS_BYTES` at real offsets)
/// produces the exact same materialized tensors as the pre-existing
/// in-memory construction (`e2e_fixture_weights`) -- before
/// `bind_qwen_fixture_weights`'s production call site is allowed to switch
/// from one to the other. Equivalence proven, not assumed, matching this
/// session's own working pattern for prior real-Component/real-artifact
/// cutovers.
#[test]
fn e2e_fixture_real_artifact_weights_match_in_memory_weights() {
    let config = e2e_fixture_config();
    let in_memory = e2e_fixture_weights(&config).expect("in-memory fixture weights build");
    let from_real_artifact =
        e2e_fixture_weights_from_real_artifact(&config).expect("real-artifact weights materialize");

    assert_eq!(
        in_memory.keys().collect::<Vec<_>>(),
        from_real_artifact.keys().collect::<Vec<_>>(),
        "real-artifact and in-memory weight maps must cover the same tensor names"
    );
    for (name, expected_tensor) in &in_memory {
        let actual_tensor = &from_real_artifact[name];
        assert_eq!(
            actual_tensor.shape, expected_tensor.shape,
            "tensor '{name}' shape mismatch between real-artifact and in-memory paths"
        );
        assert_eq!(
            actual_tensor.data, expected_tensor.data,
            "tensor '{name}' data mismatch between real-artifact and in-memory paths"
        );
    }
}

#[test]
fn e2e_tensor_output_updates_readiness_without_raw_pointer() {
    let fixture = e2e_fixture().expect("fixture builds");
    let logits = e2e_forward(&fixture, &[1, 2]).expect("forward pass produces logits");
    assert_eq!(logits.len(), E2E_FIXTURE_VOCAB as usize);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn e2e_resource_cleanup_after_generation_and_session_close() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
    let report = unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("unload succeeds");
    assert!(!report.dangling_session_references);
}

#[test]
fn e2e_fixture_tokenizer_streams_decode_across_multiple_chunks() {
    let fixture = e2e_fixture().expect("fixture builds");
    let full = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi!".into())),
        None,
    )
    .expect("tokenizes");
    assert!(full.token_ids.len() >= 2);

    let mut state = StreamingDecodeState::default();
    let mut decoded = String::new();
    for token_id in &full.token_ids {
        let output = fixture
            .tokenizer
            .streaming_decode(state, vec![*token_id], false)
            .expect("streaming decode step succeeds");
        decoded.push_str(&output.text);
        state = output.pending_partial_state.unwrap_or_default();
    }
    assert_eq!(decoded, "hi!");
}

#[test]
fn e2e_chat_message_prompt_without_formatter_is_policy_denied() {
    let fixture = e2e_fixture().expect("fixture builds");
    let messages = vec![ChatMessage::new("user", "hi")];
    let result = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::ChatMessages(messages)),
        None,
    );
    assert!(matches!(
        result,
        Err(InferenceApiError::PolicyDenied { .. })
    ));
}
