//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;
use std::sync::atomic::Ordering;

#[test]
fn e2e_graph_executor_logits_provenance_requires_declared_output_edge() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_executor_logits_provenance_requires_declared_output_edge(&fixture)
        .expect("graph executor never fabricates a 'logits' binding");
}

#[test]
fn e2e_generation_loop_executes_published_plan_bindings() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_loop_executes_published_plan_bindings(&fixture)
        .expect("generation loop fails closed when a published plan binding is missing");
}

#[test]
fn e2e_kv_sampling_failure_leaves_cache_uncommitted() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_sampling_failure_leaves_cache_uncommitted(&fixture)
        .expect("a KV write with no commit call stays pending, never promoted");
}

#[test]
fn e2e_kv_cancelled_decode_does_not_corrupt_committed_cache() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_cancelled_decode_does_not_corrupt_committed_cache(&fixture)
        .expect("a cancelled decode's pending KV write does not alter the committed cache");
}

#[test]
fn e2e_kv_pending_write_is_memory_admitted_for_its_concatenated_size() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_pending_write_is_memory_admitted_for_its_concatenated_size(&fixture)
        .expect("decode's concatenated pending KV write is memory-admitted at its real size");
}

#[test]
fn e2e_kv_pending_write_allocation_is_released_on_discard() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_pending_write_allocation_is_released_on_discard(&fixture)
        .expect("discarding a pending KV state releases its admitted allocations");
}

#[test]
fn e2e_kv_partial_layer_failure_during_commit_rolls_back_cleanly() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_partial_layer_failure_during_commit_rolls_back_cleanly(&fixture).expect(
        "a partial-layer commit failure rolls back cleanly, without a mixed-generation cache",
    );
}

#[test]
fn failed_prefill_commit_releases_pending_kv_on_request_cleanup() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "prefill-commit-failure",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let step = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    let pending = engine
        .pending_kv_states
        .lock()
        .unwrap()
        .get(&E2eRuntimeModelExecutionEngine::kv_state_key(&request))
        .unwrap()
        .layer_kv
        .get(&0)
        .unwrap()
        .clone();
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    assert!(provider.read_tensor(&pending.k).is_some());
    let _ = provider.release_tensor(&pending.v);
    assert!(
        engine
            .commit_generation_step(&mut runtime, &request, &[], 1, &step)
            .is_err()
    );
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
    assert!(
        provider.read_tensor(&pending.k).is_none(),
        "failed prefill leaked its pending key"
    );
}

#[test]
fn failed_prefill_graph_releases_partially_written_kv_on_request_cleanup() {
    let fixture = e2e_fixture().unwrap();
    for (copy_based, failed_weight) in [
        (true, "layers.0.self_attn.v_proj"),
        (true, "layers.0.mlp.down_proj"),
        (false, "layers.0.self_attn.v_proj"),
        (false, "layers.0.mlp.down_proj"),
    ] {
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);
        let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
        let engine = if copy_based {
            new_copy_based_kv_lifecycle_engine(&fixture)
        } else {
            new_kv_lifecycle_engine(&fixture)
        };
        let request = kv_lifecycle_test_request(
            &fixture,
            &runtime,
            &instance,
            "partial-prefill-failure",
            None,
            &[1, 2],
        )
        .unwrap();
        let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
        let weight = runtime
            .model_instances()
            .instance(&instance)
            .unwrap()
            .definition()
            .resource_bindings
            .weights[failed_weight]
            .clone();
        let provider = resolve_kernel_execution_provider(
            &runtime,
            &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        )
        .unwrap();
        assert!(provider.release_tensor(&weight).unwrap());
        let error = engine
            .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
            .unwrap_err();
        assert!(error.to_string().contains(failed_weight), "{error}");
        let cache = runtime
            .kv_caches()
            .observations()
            .iter()
            .find_map(|observation| observation.cache.clone())
            .unwrap();
        let suffix = if copy_based { "pending" } else { "store" };
        let key = TensorResourceId::new(format!("kv.{cache}.layer0.k.{suffix}"));
        let value = TensorResourceId::new(format!("kv.{cache}.layer0.v.{suffix}"));
        engine
            .release_request_state(&mut runtime, &request)
            .unwrap();
        for id in [key, value] {
            assert!(provider.read_tensor(&id).is_none(), "leaked {id}");
            assert!(
                runtime.memory().tensor_residency(&id).is_none(),
                "leaked {id}"
            );
        }
        assert!(
            runtime.kv_cache(&cache).is_err(),
            "failed prefill left its cache"
        );
        let owned = runtime
            .memory()
            .allocations()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.owner == MemoryAllocationOwner::Session(cache.to_string())
            })
            .count();
        assert_eq!(owned, 0, "failed prefill left cache-owned allocations");
    }
}

#[test]
fn abandoned_prefill_cleanup_accepts_an_already_forgotten_cache() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "forgotten-prefill",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let step = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    let cache = prefill_cache_id_from_step(&step).unwrap();
    let key = TensorResourceId::new(format!("kv.{cache}.layer0.k.pending"));
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    assert!(provider.read_tensor(&key).is_some());
    runtime.release_kv_cache(&cache).unwrap();
    runtime.kv_caches_mut().forget_released(&cache).unwrap();
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
    assert!(provider.read_tensor(&key).is_none());
}

#[test]
fn failed_decode_graph_preserves_committed_kv_and_can_retry() {
    let fixture = e2e_fixture().unwrap();
    for copy_based in [true, false] {
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);
        let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
        let engine = if copy_based {
            new_copy_based_kv_lifecycle_engine(&fixture)
        } else {
            new_kv_lifecycle_engine(&fixture)
        };
        let request = kv_lifecycle_test_request(
            &fixture,
            &runtime,
            &instance,
            "partial-decode-failure",
            None,
            &[1, 2],
        )
        .unwrap();
        let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
        let prefill = engine
            .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
            .unwrap();
        engine
            .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
            .unwrap();
        let cache = prefill_cache_id_from_step(&prefill).unwrap();
        let before = runtime.kv_cache(&cache).unwrap().clone();
        let provider = resolve_kernel_execution_provider(
            &runtime,
            &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        )
        .unwrap();
        let binding = &before.layer_resources[&0];
        let committed_k = provider.read_tensor_rows(&binding.k, 0, 2).unwrap();
        let committed_v = provider.read_tensor_rows(&binding.v, 0, 2).unwrap();
        let expected = engine
            .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
            .unwrap()
            .logits;
        engine
            .discard_pending_kv_state(&mut runtime, &request)
            .unwrap();
        let weight = runtime
            .model_instances()
            .instance(&instance)
            .unwrap()
            .definition()
            .resource_bindings
            .weights["layers.0.mlp.down_proj"]
            .clone();
        let saved = provider.read_tensor(&weight).unwrap();
        assert!(provider.release_tensor(&weight).unwrap());
        let error = engine
            .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
            .unwrap_err();
        assert!(error.to_string().contains("down_proj"), "{error}");
        assert_eq!(runtime.kv_cache(&cache).unwrap(), &before);
        assert_eq!(
            provider.read_tensor_rows(&binding.k, 0, 2).unwrap(),
            committed_k
        );
        assert_eq!(
            provider.read_tensor_rows(&binding.v, 0, 2).unwrap(),
            committed_v
        );
        for role in ["k", "v"] {
            let id = TensorResourceId::new(format!("kv.{cache}.layer0.{role}.pending"));
            assert!(provider.read_tensor(&id).is_none(), "leaked {id}");
            assert!(
                runtime.memory().tensor_residency(&id).is_none(),
                "leaked {id}"
            );
        }
        provider.write_tensor(weight, saved).unwrap();
        let retried = engine
            .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
            .unwrap();
        assert_eq!(retried.logits, expected);
        engine
            .commit_generation_step(&mut runtime, &request, &[1], 2, &retried)
            .unwrap();
        engine
            .release_request_state(&mut runtime, &request)
            .unwrap();
    }
}

#[test]
fn failed_prefill_commit_can_retry_the_same_pending_state() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "prefill-retry",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let step = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    let pending = engine
        .pending_kv_states
        .lock()
        .unwrap()
        .get(&E2eRuntimeModelExecutionEngine::kv_state_key(&request))
        .unwrap()
        .layer_kv
        .get(&0)
        .unwrap()
        .clone();
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let value = provider.read_tensor(&pending.v).unwrap();
    let _ = provider.release_tensor(&pending.v);
    assert!(
        engine
            .commit_generation_step(&mut runtime, &request, &[], 1, &step)
            .is_err()
    );
    provider.write_tensor(pending.v.clone(), value).unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &step)
        .unwrap();
    let cache = prefill_cache_id_from_step(&step).unwrap();
    assert!(
        runtime
            .kv_cache(&cache)
            .unwrap()
            .layer_resources
            .contains_key(&0)
    );
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

#[test]
fn sealed_cache_rejects_decode_without_replacing_committed_kv() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "sealed-commit",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let cache = prefill_cache_id_from_step(&prefill).unwrap();
    let previous = runtime.kv_cache(&cache).unwrap().layer_resources.clone();
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let before = provider.read_tensor(&previous[&0].k).unwrap();
    let decode = engine
        .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
        .unwrap();
    runtime.seal_kv_cache(&cache).unwrap();
    assert!(
        engine
            .commit_generation_step(&mut runtime, &request, &[1], 2, &decode)
            .is_err()
    );
    assert_eq!(runtime.kv_cache(&cache).unwrap().layer_resources, previous);
    assert_eq!(provider.read_tensor(&previous[&0].k).unwrap(), before);
    assert!(
        engine
            .pending_kv_states
            .lock()
            .unwrap()
            .contains_key(&E2eRuntimeModelExecutionEngine::kv_state_key(&request))
    );
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

#[test]
fn over_capacity_prefill_keeps_cache_empty_and_pending_state_reachable() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "capacity-commit",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let step = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    let cache = prefill_cache_id_from_step(&step).unwrap();
    runtime
        .kv_caches_mut()
        .cache_mut(&cache)
        .unwrap()
        .layout
        .token_capacity = 1;
    let before = runtime.kv_cache(&cache).unwrap().clone();
    let mut candidate = before.clone();
    assert!(candidate.complete_prefill(2).is_err());
    assert_eq!(candidate, before);
    assert!(
        engine
            .commit_generation_step(&mut runtime, &request, &[], 1, &step)
            .is_err()
    );
    assert_eq!(runtime.kv_cache(&cache).unwrap(), &before);
    assert!(
        engine
            .pending_kv_states
            .lock()
            .unwrap()
            .contains_key(&E2eRuntimeModelExecutionEngine::kv_state_key(&request))
    );
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

#[test]
fn e2e_kv_double_commit_second_call_is_rejected() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_double_commit_second_call_is_rejected(&fixture)
        .expect("a second commit for an already-committed generation step is rejected");
}

#[test]
fn e2e_kv_double_abort_is_idempotent() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_double_abort_is_idempotent(&fixture)
        .expect("discarding a pending KV state twice in a row is idempotent");
}

#[test]
fn e2e_kv_stale_pending_state_does_not_survive_a_failed_retry() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_stale_pending_state_does_not_survive_a_failed_retry(&fixture)
        .expect("a stale pending KV write is discarded before a failed retry, not left to be wrongly committed");
}

#[test]
fn e2e_kv_wrong_session_reuse_is_rejected_by_compatibility() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_kv_wrong_session_reuse_is_rejected_by_compatibility(&fixture)
        .expect("reusing one session's committed KV cache under another session's compatibility is rejected");
}

#[test]
fn same_request_id_cannot_reuse_or_discard_another_sessions_kv() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let session_a = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(&fixture, &instance),
    )
    .unwrap();
    let session_b = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(&fixture, &instance),
    )
    .unwrap();
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "same-request-id",
        Some(session_a),
        &[1, 2],
    )
    .unwrap();
    let mut other_session = request.clone();
    other_session.session = Some(session_b);
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    let key = E2eRuntimeModelExecutionEngine::kv_state_key(&request);
    assert!(
        engine
            .discard_pending_kv_state(&mut runtime, &other_session)
            .is_err()
    );
    assert!(
        engine
            .commit_generation_step(&mut runtime, &other_session, &[], 1, &prefill)
            .is_err()
    );
    assert!(
        engine
            .execute_generation_step(&mut runtime, &other_session, &[], Some(&mut plans.prefill))
            .is_err()
    );
    assert!(
        engine
            .release_request_state(&mut runtime, &other_session)
            .is_err()
    );
    assert!(engine.pending_kv_states.lock().unwrap().contains_key(&key));
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let cache = prefill_cache_id_from_step(&prefill).unwrap();
    let committed = runtime.kv_cache(&cache).unwrap().clone();
    assert_eq!(
        committed.compatibility.tokenizer_vocabulary_size,
        request.tokenizer.metadata.vocabulary_size
    );
    assert!(committed.compatibility.model_architecture.is_some());

    let mut other_component = engine.clone();
    other_component.component_digest = Some(ComponentDigest::sha256(b"different component"));
    assert!(
        other_component
            .decode_kv_state(&mut runtime, &request)
            .is_err()
    );
    assert!(
        other_component
            .release_request_state(&mut runtime, &request)
            .is_err()
    );
    assert_eq!(runtime.kv_cache(&cache).unwrap(), &committed);

    let mut no_session = request.clone();
    no_session.session = None;
    let mut other_model = request.clone();
    other_model.model = GenerationModelReference::LoadedModelContext("other-model".into());
    let mut other_tokenizer = request.clone();
    other_tokenizer.tokenizer.tokenizer_id = TokenizerId::new("other-tokenizer").unwrap();
    let mut other_tokenizer_revision = request.clone();
    other_tokenizer_revision.tokenizer.metadata.revision = TokenizerRevision::new("r2").unwrap();
    let mut other_prompt = request.clone();
    other_prompt.input_token_ids[0] = 3;
    for (label, foreign) in [
        ("session", &other_session),
        ("missing session", &no_session),
        ("model", &other_model),
        ("tokenizer", &other_tokenizer),
        ("tokenizer revision", &other_tokenizer_revision),
        ("prompt", &other_prompt),
    ] {
        assert!(
            engine.decode_kv_state(&mut runtime, foreign).is_err(),
            "accepted {label}"
        );
        assert!(
            engine.release_request_state(&mut runtime, foreign).is_err(),
            "released {label}"
        );
        assert_eq!(runtime.kv_cache(&cache).unwrap(), &committed);
    }

    let decode = engine
        .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
        .unwrap();
    let key = E2eRuntimeModelExecutionEngine::kv_state_key(&request);
    let pending = engine.pending_kv_states.lock().unwrap()[&key].clone();
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let pending_key = pending.layer_kv[&0].k.clone();
    assert!(provider.read_tensor(&pending_key).is_some());
    assert!(
        engine
            .discard_pending_kv_state(&mut runtime, &other_session)
            .is_err()
    );
    assert!(
        engine
            .commit_generation_step(&mut runtime, &other_session, &[1], 2, &decode)
            .is_err()
    );
    assert!(
        engine
            .execute_generation_step(&mut runtime, &other_session, &[1], Some(&mut plans.decode))
            .is_err()
    );
    assert!(
        engine
            .release_request_state(&mut runtime, &other_session)
            .is_err()
    );
    assert!(engine.pending_kv_states.lock().unwrap().contains_key(&key));
    assert!(provider.read_tensor(&pending_key).is_some());
    assert_eq!(runtime.kv_cache(&cache).unwrap(), &committed);

    engine
        .commit_generation_step(&mut runtime, &request, &[1], 2, &decode)
        .unwrap();
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

#[test]
fn kv_steps_require_the_committed_generated_token_history() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "kv-token-history",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let cache = prefill_cache_id_from_step(&prefill).unwrap();
    let committed = runtime.kv_cache(&cache).unwrap().clone();
    assert!(
        engine
            .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
            .is_err()
    );
    assert!(
        engine
            .execute_generation_step(&mut runtime, &request, &[2], Some(&mut plans.decode))
            .is_err()
    );
    assert_eq!(runtime.kv_cache(&cache).unwrap(), &committed);

    let decode = engine
        .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
        .unwrap();
    let key = E2eRuntimeModelExecutionEngine::kv_state_key(&request);
    assert!(
        engine
            .commit_generation_step(&mut runtime, &request, &[2], 2, &decode)
            .is_err()
    );
    assert!(
        engine
            .execute_generation_step(&mut runtime, &request, &[2], Some(&mut plans.decode))
            .is_err()
    );
    assert!(engine.pending_kv_states.lock().unwrap().contains_key(&key));
    assert_eq!(runtime.kv_cache(&cache).unwrap(), &committed);
    engine
        .commit_generation_step(&mut runtime, &request, &[1], 2, &decode)
        .unwrap();

    assert!(
        engine
            .execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode))
            .is_err()
    );
    assert!(
        engine
            .execute_generation_step(&mut runtime, &request, &[3, 2], Some(&mut plans.decode))
            .is_err()
    );
    let next = engine
        .execute_generation_step(&mut runtime, &request, &[1, 2], Some(&mut plans.decode))
        .unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[1, 2], 3, &next)
        .unwrap();
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

#[test]
fn released_coarse_kv_allocation_does_not_skip_layer_cleanup() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let engine = new_copy_based_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "kv-coarse-already-released",
        None,
        &[1, 2],
    )
    .unwrap();
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2).unwrap();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let cache = prefill_cache_id_from_step(&prefill).unwrap();
    let binding = runtime.kv_cache(&cache).unwrap().layer_resources[&0].clone();
    let coarse = runtime.allocate_kv_cache_memory(&cache).unwrap();
    runtime.memory_mut().release(coarse).unwrap();
    assert!(runtime.memory().is_live_allocation(binding.k_allocation));
    assert!(runtime.memory().is_live_allocation(binding.v_allocation));
    runtime.release_kv_cache(&cache).unwrap();
    assert!(!runtime.memory().is_live_allocation(binding.k_allocation));
    assert!(!runtime.memory().is_live_allocation(binding.v_allocation));
    engine
        .release_request_state(&mut runtime, &request)
        .unwrap();
}

/// `turboquant-kv-history`: a prefill records its stores' precisions in
/// the history's compatibility and KV layout metadata (none for f32, as
/// before), and a history is not continued once this model's stores
/// resolve to other precisions.
#[test]
fn a_kv_history_is_only_continued_in_its_recorded_precisions() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
    let engine = new_kv_lifecycle_engine(&fixture);
    let request = kv_lifecycle_test_request(
        &fixture,
        &runtime,
        &instance,
        "kv-precisions",
        None,
        &[1, 2],
    )
    .expect("request");
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        &fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )
    .expect("plans");
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .expect("prefill");
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .expect("commit");
    let cache = prefill_cache_id_from_step(&prefill).expect("cache");
    let recorded = runtime.kv_cache(&cache).expect("the history's cache");
    assert_eq!(recorded.compatibility.quantization_mode, None);
    assert_eq!(recorded.layout.quantization, None);

    engine
        .kv_pool
        .resolve_precisions_to(KvPrecisions::uniform(KvStorePrecision::Tq4));
    match engine.execute_generation_step(&mut runtime, &request, &[1], Some(&mut plans.decode)) {
        Err(error) => assert!(format!("{error:?}").contains("dtype mismatch"), "{error:?}"),
        Ok(_) => panic!("an f32 history was continued with tq4 stores"),
    }
}

#[test]
fn e2e_generation_step_rechecks_model_instance_readiness() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_step_rechecks_model_instance_readiness(&fixture)
        .expect("a generation step re-checks model instance readiness for itself");
}

/// `cache-and-optimize-prepared-graphs`: a loaded model builds its graphs
/// once per prompt length, plans and every step of the request use those
/// same graphs, and the cache keeps at most
/// `COMPONENT_GRAPH_CACHE_CAPACITY` prompt lengths, most recent last.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn resident_production_model_reuses_component_graphs_per_prompt_length() {
    use crate::phase_attribution::{StepKind, StepPhase};

    let (fixture, payload_source) = production_shaped_canonical_fixture();
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("production loading succeeds");
    // Exact-length prefills: one graph per prompt length.
    model.set_prefill_buckets(Vec::new());
    let request = |prompt: String| ProductionGenerationRequest {
        prompt: PromptInput::PlainText(prompt),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(3),
        max_generation_millis: None,
    };
    let first = model.generate(request("hi".into()), None).unwrap();
    let prompt_tokens = first.result.output.usage.prompt_tokens as u64;
    assert_eq!(model.graph_cache.prompt_lengths(), vec![prompt_tokens]);

    model.enable_phase_attribution();
    let again = model.generate(request("hi".into()), None).unwrap();
    assert_eq!(
        again.result.output.generated_token_ids, first.result.output.generated_token_ids,
        "cached graphs generate the same tokens"
    );
    assert_eq!(model.graph_cache.prompt_lengths(), vec![prompt_tokens]);
    let report = model.take_phase_attribution();
    for step in &report.steps {
        let graph_builds = step
            .phases
            .iter()
            .filter(|(key, _)| key.phase == StepPhase::GraphBuild)
            .count();
        assert_eq!(graph_builds, 1, "{step:?}");
    }
    assert_eq!(report.summary(StepKind::Decode).unwrap().steps, 2);

    let mut lengths = vec![prompt_tokens];
    for words in 2..=COMPONENT_GRAPH_CACHE_CAPACITY + 3 {
        let prompt = "h".repeat(words);
        let outcome = model.generate(request(prompt), None).unwrap();
        let length = outcome.result.output.usage.prompt_tokens as u64;
        lengths.retain(|known| *known != length);
        lengths.push(length);
    }
    let expected: Vec<u64> = lengths
        .iter()
        .copied()
        .skip(lengths.len().saturating_sub(COMPONENT_GRAPH_CACHE_CAPACITY))
        .collect();
    assert!(
        lengths.len() > COMPONENT_GRAPH_CACHE_CAPACITY,
        "{lengths:?}"
    );
    assert_eq!(model.graph_cache.prompt_lengths(), expected);
}

/// `append-kv-history-in-place`: in-place KV history (Reference CPU
/// advertises `kv-write` and `key-count` attention) produces the same logits
/// bit for bit as the copy-based history, a decode step executed and
/// discarded leaves the committed rows untouched (executing it again gives
/// the same logits), and the committed layer binding is the layer's store.
#[test]
fn in_place_kv_history_matches_copy_based_history_and_survives_a_discarded_step() {
    let fixture = e2e_fixture().expect("fixture builds");
    let run = |engine: E2eRuntimeModelExecutionEngine| -> (Vec<Vec<u32>>, Runtime, String) {
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);
        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
        let request =
            kv_lifecycle_test_request(&fixture, &runtime, &instance, "kv-in-place", None, &[1, 2])
                .unwrap();
        let mut plans = first_native_plans_for_prompt(
            &runtime,
            &fixture,
            &instance,
            request.input_token_ids.len() as u64,
        )
        .unwrap();
        let bits = |logits: &[f32]| logits.iter().map(|value| value.to_bits()).collect();
        let mut steps = Vec::new();
        let prefill = engine
            .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
            .unwrap();
        steps.push(bits(&prefill.logits));
        engine
            .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
            .unwrap();
        let cache = prefill_cache_id_from_step(&prefill).unwrap().to_string();
        let mut generated = vec![1];
        for token in [2, 1] {
            let step = engine
                .execute_generation_step(
                    &mut runtime,
                    &request,
                    &generated,
                    Some(&mut plans.decode),
                )
                .unwrap();
            // Discarded (never committed), then executed again.
            let again = engine
                .execute_generation_step(
                    &mut runtime,
                    &request,
                    &generated,
                    Some(&mut plans.decode),
                )
                .unwrap();
            assert_eq!(bits(&step.logits), bits(&again.logits));
            steps.push(bits(&again.logits));
            engine
                .commit_generation_step(&mut runtime, &request, &generated, token, &again)
                .unwrap();
            generated.push(token);
        }
        (steps, runtime, cache)
    };
    let (in_place, runtime, cache) = run(new_kv_lifecycle_engine(&fixture));
    let (copy_based, _, _) = run(new_copy_based_kv_lifecycle_engine(&fixture));
    assert_eq!(in_place, copy_based);
    let cache = runtime.kv_cache(&KvCacheId::new(cache).unwrap()).unwrap();
    for binding in cache.layer_resources.values() {
        assert!(binding.k.as_str().ends_with(".k.store"), "{}", binding.k);
        assert!(binding.v.as_str().ends_with(".v.store"), "{}", binding.v);
    }
}

/// A prefill materializes only its last logits row, so its LM head runs on
/// the last hidden row alone, and that row's logits are the full graph's
/// (Reference CPU computes each row independently, in the same order).
#[test]
fn prefill_lm_head_runs_on_the_last_row_only() {
    let (steps, executor, _) = run_replaying_decode(false, true, 1, |_, _| {});
    // One prefill of the 2-token prompt, then one decode step.
    assert_eq!(*executor.logits_rows.lock().unwrap(), vec![1, 1]);
    let fixture = e2e_fixture().expect("fixture builds");
    let expected = e2e_forward(&fixture, &[1, 2]).unwrap();
    let prefill: Vec<f32> = steps[0].iter().map(|bits| f32::from_bits(*bits)).collect();
    assert_eq!(prefill.len(), expected.len());
    for (actual, expected) in prefill.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-4, "{actual} vs {expected}");
    }
}

/// A Provider that zeroes the in-place KV stores itself
/// (`write_zeroed_tensor`) gets one call per layer and role at prefill, and
/// every step's logits are bit-identical to those of host-written zeros.
#[test]
fn provider_zeroed_kv_stores_match_host_written_zeros() {
    let decode_steps = 4;
    let (host, host_executor, _) = run_replaying_decode(true, true, decode_steps, |_, _| {});
    assert_eq!(host_executor.zeroed_tensors.load(Ordering::SeqCst), 0);
    let executor = Arc::new(ReplayingExecutor::new());
    executor.zero_on_provider.store(true, Ordering::SeqCst);
    let (zeroed, executor, _) =
        run_replaying_decode_on(executor, true, true, decode_steps, |_, _| {});
    assert_eq!(zeroed, host);
    assert_eq!(
        executor.zeroed_tensors.load(Ordering::SeqCst),
        2 * E2E_FIXTURE_LAYERS
    );
}

/// `keep-sampling-on-device`: a replayed plain greedy decode step can have
/// its token selected by the Provider instead of returning its logits, and
/// the selected token is the one host sampling picks from those logits.
#[test]
fn replayed_greedy_steps_select_their_token_on_the_provider() {
    let decode_steps = 6;
    // The logits each step produces, with host selection.
    let (logits, _, _) = run_replaying_decode(true, true, decode_steps, |_, _| {});
    let fixture = e2e_fixture().expect("fixture builds");
    let executor = Arc::new(ReplayingExecutor::new());
    executor.select_greedy.store(true, Ordering::SeqCst);
    let provider = Arc::new(ReplayingProvider {
        inner: ReferenceCpuProvider::new(),
        executor: executor.clone(),
    });
    let mut runtime =
        build_runtime_with_model_execution_engine_and_provider(&fixture, provider).unwrap();
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(vec![1, 2])),
        None,
    )
    .unwrap();
    let request = prepare_generation(
        &runtime,
        build_generation_request(
            GenerationRequestId::new("step-replay").unwrap(),
            None,
            GenerationModelReference::ModelInstance(instance.clone()),
            generation_tokenizer_reference(&fixture),
            tokenized,
            decode_steps + 1,
            GenerationParameters::greedy(),
            StopConditions::default(),
            StreamingMode::Disabled,
        ),
    )
    .unwrap();
    let engine = new_kv_lifecycle_engine(&fixture);
    engine.step_replay.set_enabled(true);
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        &fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )
    .unwrap();
    let prefill = engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .unwrap();
    engine
        .commit_generation_step(&mut runtime, &request, &[], 1, &prefill)
        .unwrap();
    let mut generated = vec![1];
    let mut selected_steps = 0;
    for (index, step_logits) in logits.iter().enumerate().skip(1) {
        let greedy = crate::generation::sampling_request_for(
            &request,
            &generated,
            Vec::new(),
            SamplingPolicy::default(),
            None,
        )
        .unwrap();
        assert!(
            crate::sampling::device_greedy_plan(&greedy, step_logits.len()).is_some(),
            "a greedy request without logits is a Device selection candidate"
        );
        let step = engine
            .execute_generation_step_selecting(
                &mut runtime,
                &request,
                &generated,
                Some(&mut plans.decode),
                &greedy,
            )
            .unwrap();
        // What host sampling picks from this step's logits.
        let host_logits: Vec<f32> = step_logits
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect();
        let (host, _) =
            decode_step_from_sampling(&request, &generated, host_logits, SamplingPolicy::default())
                .unwrap();
        match step.device_greedy_column {
            Some(column) => {
                selected_steps += 1;
                assert!(step.logits.is_empty(), "step {index}");
                let (device, _) = crate::generation::decode_step_from_device_greedy(
                    &request, &generated, &greedy, column,
                )
                .unwrap();
                assert_eq!(
                    device.selected_token_id, host.selected_token_id,
                    "step {index}"
                );
                assert_eq!(device.token_rank, host.token_rank, "step {index}");
                assert_eq!(device.finish_hint, host.finish_hint, "step {index}");
            }
            None => assert_eq!(step.logits.len(), step_logits.len(), "step {index}"),
        }
        let token = (index % 3 + 1) as TokenId;
        engine
            .commit_generation_step(&mut runtime, &request, &generated, token, &step)
            .unwrap();
        generated.push(token);
    }
    // Only replayed steps (after the capture at the second decode step)
    // select on the Provider.
    assert_eq!(selected_steps, decode_steps - 2);
    assert_eq!(
        executor.greedy_selections.load(Ordering::SeqCst),
        selected_steps as u64
    );
}

#[test]
fn replayed_decode_steps_match_per_node_steps_bit_for_bit() {
    let (per_node, executor, _) = run_replaying_decode(false, true, 6, |_, _| {});
    assert_eq!(executor.counts(), (0, 0, 0));
    let (replayed, executor, _) = run_replaying_decode(true, true, 6, |_, _| {});
    assert_eq!(per_node, replayed);
    // Captured at the second decode step, replayed at the four after it,
    // released with the request.
    assert_eq!(executor.counts(), (1, 4, 0));
}

#[test]
fn an_invalidated_replay_runs_per_node_and_captures_again() {
    let (per_node, _, _) = run_replaying_decode(false, true, 6, |_, _| {});
    let (replayed, executor, _) = run_replaying_decode(true, true, 6, |executor, index| {
        if index == 4 {
            executor
                .invalidate_next_replay
                .store(true, Ordering::SeqCst);
        }
    });
    assert_eq!(per_node, replayed);
    // Step 3 replays; step 4's replay is invalidated, so it runs per node
    // and step 5 captures again; step 6 replays.
    assert_eq!(executor.counts(), (2, 2, 0));
}

#[test]
fn a_failed_capture_runs_the_step_per_node_and_stops_capturing() {
    let (per_node, _, _) = run_replaying_decode(false, true, 5, |_, _| {});
    let (replayed, executor, _) = run_replaying_decode(true, true, 5, |executor, _| {
        executor.fail_capture.store(true, Ordering::SeqCst);
    });
    assert_eq!(per_node, replayed);
    assert_eq!(executor.counts(), (0, 0, 0));
}

#[test]
fn dropping_the_engine_releases_captures_its_requests_never_released() {
    let (_, executor, _) = run_replaying_decode(true, false, 4, |_, _| {});
    // Captured and replayed, then released when the engine dropped.
    assert_eq!(executor.counts(), (1, 2, 0));
}

/// A request abandoned between two replayed steps gives its capture back
/// with its state, and the next request captures its own: nothing of the
/// first one's capture is replayed for the second
/// (`capture-and-replay-cuda-graphs` 3.3: cancellation, invalid state reuse).
#[test]
fn a_request_released_between_replayed_steps_gives_its_capture_back() {
    let (per_node, _, _) = run_replaying_decode(false, true, 4, |_, _| {});
    let mut lifecycle = ReplayingLifecycle::new();
    let mut first = lifecycle.start("abandoned", None);
    lifecycle.decode(&mut first, 3);
    // Captured at the second decode step, replayed at the third.
    assert_eq!(lifecycle.executor.counts(), (1, 1, 1));
    lifecycle
        .engine
        .release_request_state(&mut lifecycle.runtime, &first.request)
        .unwrap();
    assert_eq!(lifecycle.executor.counts(), (1, 1, 0));
    let mut second = lifecycle.start("next", None);
    lifecycle.decode(&mut second, 4);
    // The second request captured and replayed its own.
    assert_eq!(lifecycle.executor.counts(), (2, 3, 1));
    assert_eq!(second.steps, per_node);
    assert_eq!(first.steps[..], per_node[..first.steps.len()]);
}

/// Two requests of two sessions replay their own captures, whatever the
/// order of their steps; another session's release of one's id is refused
/// and leaves its capture in place
/// (`capture-and-replay-cuda-graphs` 3.3: tenant and session isolation).
#[test]
fn requests_of_two_sessions_replay_their_own_captures() {
    let (per_node, _, _) = run_replaying_decode(false, true, 4, |_, _| {});
    let mut lifecycle = ReplayingLifecycle::new();
    let session_a = create_inference_session(
        &mut lifecycle.runtime,
        kv_lifecycle_session_request(&lifecycle.fixture, &lifecycle.instance),
    )
    .unwrap();
    let session_b = create_inference_session(
        &mut lifecycle.runtime,
        kv_lifecycle_session_request(&lifecycle.fixture, &lifecycle.instance),
    )
    .unwrap();
    let mut a = lifecycle.start("tenant-a", Some(session_a));
    let mut b = lifecycle.start("tenant-b", Some(session_b.clone()));
    for _ in 0..3 {
        lifecycle.decode(&mut a, 1);
        lifecycle.decode(&mut b, 1);
    }
    // Each captured at its second decode step and replayed at its third.
    assert_eq!(lifecycle.executor.counts(), (2, 2, 2));
    // Another session cannot release the capture of `tenant-a`.
    let mut foreign = a.request.clone();
    foreign.session = Some(session_b);
    assert!(
        lifecycle
            .engine
            .release_request_state(&mut lifecycle.runtime, &foreign)
            .is_err()
    );
    assert_eq!(lifecycle.executor.counts(), (2, 2, 2));
    lifecycle.decode(&mut a, 1);
    lifecycle.decode(&mut b, 1);
    assert_eq!(lifecycle.executor.counts(), (2, 4, 2));
    assert_eq!(a.steps, per_node);
    assert_eq!(b.steps, per_node);
}

/// A Provider that cannot reserve the memory a capture needs: every step
/// runs per node, with the same values, and the request does not try
/// again (`capture-and-replay-cuda-graphs` 3.3: memory pressure).
#[test]
fn a_capture_refused_for_memory_runs_every_step_per_node() {
    let (per_node, _, _) = run_replaying_decode(false, true, 5, |_, _| {});
    let (refused, executor, _) = run_replaying_decode(true, true, 5, |executor, _| {
        executor.refuse_capture_memory.store(true, Ordering::SeqCst);
    });
    assert_eq!(per_node, refused);
    assert_eq!(executor.counts(), (0, 0, 0));
    assert_eq!(executor.capture_attempts.load(Ordering::SeqCst), 1);
}

/// A request reuses the plans prepared for its graphs while the Model
/// Instance, the Provider and the Kernel Registry are unchanged, and
/// prepares them again when the registry changes.
#[test]
fn cached_graphs_reuse_prepared_plans_until_the_registry_changes() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let graphs = CachedGraphs::new(
        first_native_component_graphs_for_prompt(&fixture, 2).expect("component graphs build"),
    );
    let provider = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
    let first = graphs
        .prepared_plans(&runtime, &instance, 2, &provider)
        .unwrap();
    let second = graphs
        .prepared_plans(&runtime, &instance, 2, &provider)
        .unwrap();
    assert_eq!(graphs.preparations(), 1);
    assert_eq!(first.prefill, second.prefill);
    assert_eq!(first.decode, second.decode);
    // The copy is what a fresh preparation returns.
    let fresh = prepare_first_native_execution_plans_for_provider(
        &runtime,
        &instance,
        FirstNativeComponentGraphs::clone(&graphs),
        2,
        &provider,
    )
    .unwrap();
    assert_eq!(fresh.prefill, second.prefill);
    assert_eq!(fresh.decode, second.decode);
    // Any registry change invalidates the cached plans.
    let before = runtime.kernel_registry().revision();
    runtime.kernel_registry_mut().set_provider_features(
        provider.clone(),
        std::collections::BTreeSet::<String>::new(),
    );
    assert_ne!(runtime.kernel_registry().revision(), before);
    graphs
        .prepared_plans(&runtime, &instance, 2, &provider)
        .unwrap();
    assert_eq!(graphs.preparations(), 2);
    // Unchanged again: the new plans are reused.
    graphs
        .prepared_plans(&runtime, &instance, 2, &provider)
        .unwrap();
    assert_eq!(graphs.preparations(), 2);
}
