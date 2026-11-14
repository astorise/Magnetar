//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_required_path_returns_usage_and_cleans_up() {
    let fixture = e2e_fixture().expect("fixture builds");
    let result = run_success_path(&fixture).expect("success path returns output");
    assert!(result.generation_result.output.usage.generated_tokens > 0);
    assert!(!result.observer.observations().is_empty());
}

/// task group 6 (`make-first-native-cuda-hot-path-device-resident`'s
/// Decision 7) / 6.2: a failure during Kernel Registry/dispatch-plan
/// construction -- here, `attention`'s required workspace failing to admit
/// under a tight Runtime memory budget, the same scenario
/// `check_graph_dispatch_records_memory_feasibility_failure_under_tight_budget`
/// already exercises -- must roll back the output this node had already
/// pre-admitted before that point, not leak it. `attention` is the first
/// node in this fixture's graph whose workspace does not fit the budget, so
/// its own pre-admitted output allocation is still the highest-numbered
/// allocation when the failure fires (workspace itself never became an
/// allocation -- admission failed before one was created), matching
/// `stage_weight_propagates_and_rolls_back_on_provider_write_failure`'s
/// same "highest id -> this attempt's allocation" reasoning.
#[test]
fn e2e_graph_dispatch_rolls_back_pre_admitted_output_on_submit_time_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                max_runtime_bytes: Some(1 << 16),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    let (instance, _memory) =
        load_fixture_instance(&fixture, &mut runtime).expect("instance loads");
    let mut plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2)
        .expect("prepared plans build");
    let graphs =
        first_native_component_graphs_for_prompt(&fixture, 2).expect("component graphs build");
    let ids = HostTensor::new([2], vec![1.0, 2.0]).expect("token id tensor builds");
    let cache_id = KvCacheId::new("test-submit-time-rollback-cache").expect("cache id is valid");

    let result = execute_first_native_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    assert!(
        result.is_err(),
        "expected the tight memory budget to fail graph dispatch"
    );

    let this_attempts_allocation = runtime
        .memory()
        .allocations()
        .max_by_key(|allocation| allocation.id)
        .expect("the failing node's output was pre-admitted before the workspace failed");
    assert_eq!(
        this_attempts_allocation.state,
        MemoryAllocationState::Released,
        "a submit-time (Kernel Registry/dispatch-plan construction) failure must roll back \
         the pre-admitted output allocation, not leak it"
    );

    // audit-complet-cuda-hot-path-2026-09-08 P0-2 / Correctif B's "failure
    // before materialization" case: `attention`'s workspace admission
    // fails before the Provider is ever invoked, so its output was never
    // written to Provider storage in the first place -- rollback's
    // `release_tensor` call is a correctly idempotent no-op here, not a
    // real cleanup. Checked anyway (the completion-time counterpart test
    // covers the genuinely materialized case) so both halves of the
    // audit's test list are directly represented.
    let attention_node = graphs
        .prefill
        .nodes
        .values()
        .find(|node| node.operator.name() == "attention")
        .expect("this fixture's prefill graph has an attention node");
    let attention_output_edge = attention_node
        .outputs
        .first()
        .expect("attention node has an output edge");
    let attention_output_id = TensorResourceId::new(format!("edge.{attention_output_edge}"));
    assert!(
        executor.read_tensor_value(&attention_output_id).is_none(),
        "a submit-time failure must never leave a Provider-side resource behind for the \
         output it pre-admitted but never got to materialize"
    );
}

#[test]
fn e2e_graph_dispatch_does_not_leak_kernel_output_allocations_across_repeated_dispatch() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_does_not_leak_kernel_output_allocations_across_repeated_dispatch(&fixture)
        .expect("repeated graph dispatch does not leak Provider-owned Kernel output allocations");
}

#[test]
fn e2e_generation_loop_decode_positions_follow_generated_tokens() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_generation_loop_decode_positions_follow_generated_tokens(&fixture)
        .expect("generation loop decode positions match the position oracle");
}

#[test]
fn e2e_one_shot_session_uses_normal_model_instance_and_tokenizer_path() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance),
        tokenizer: generation_tokenizer_reference(&fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_one_shot_session(&mut runtime, session_request).expect("creates");
    let status = session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .expect("status is readable");
    assert_eq!(status.lifecycle, SessionLifecycleState::Ready);
    close_inference_session(&mut runtime, &session).expect("closes");
}

#[test]
fn e2e_authoritative_path_collects_correlated_runtime_observations() {
    let fixture = e2e_fixture().expect("fixture builds");
    let outcome = run_success_path(&fixture).expect("success path runs");
    let observations = outcome.observer.observations();
    // ComponentValidated/ComponentInstantiated are only emitted by the real
    // Qwen Component's strict-path preflight (`build_first_native_graphs_
    // from_real_qwen_component`); a test build without a strict Component
    // engine takes the test-oracle fallback instead (task 12.6), which
    // never claims to validate/instantiate a real Component, so asserting
    // this evidence only makes sense when the strict engine is what this
    // build actually exercises.
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    for kind in [
        InferenceApiObservationKind::ComponentValidated,
        InferenceApiObservationKind::ComponentInstantiated,
    ] {
        assert!(
            observations
                .iter()
                .any(|observation| observation.kind == kind),
            "missing authoritative observation {kind:?}"
        );
    }
    for kind in [
        InferenceApiObservationKind::ModelInstanceReady,
        InferenceApiObservationKind::GraphValidationCompleted,
        InferenceApiObservationKind::PlanSelected,
        InferenceApiObservationKind::PlanGuardAccepted,
        InferenceApiObservationKind::KernelResolved,
        InferenceApiObservationKind::KernelPrepared,
        InferenceApiObservationKind::ProviderSubmitted,
        InferenceApiObservationKind::ProviderCompleted,
        InferenceApiObservationKind::KvCacheCommitted,
        InferenceApiObservationKind::LogitsProduced,
        InferenceApiObservationKind::SamplingCompleted,
        InferenceApiObservationKind::TokenCommitted,
        InferenceApiObservationKind::GraphNodeReady,
        InferenceApiObservationKind::PlanBindingResolved,
        InferenceApiObservationKind::PreparedKernelResolved,
        InferenceApiObservationKind::TensorResourceProduced,
        InferenceApiObservationKind::KvUpdatePrepared,
        InferenceApiObservationKind::KvUpdateCommitted,
    ] {
        assert!(
            observations
                .iter()
                .any(|observation| observation.kind == kind),
            "missing authoritative observation {kind:?}"
        );
    }
    // Correctif 17 / task group 17: these six are per-*node*, not global --
    // a real multi-node graph run must produce more than one, each
    // correlated to a different node, not one repeated event.
    let distinct_ready_nodes: std::collections::BTreeSet<&str> = observations
        .iter()
        .filter(|observation| observation.kind == InferenceApiObservationKind::GraphNodeReady)
        .filter_map(|observation| {
            observation
                .message
                .split("node=")
                .nth(1)
                .map(|rest| rest.split(' ').next().unwrap_or(rest))
        })
        .collect();
    assert!(
        distinct_ready_nodes.len() > 1,
        "expected GraphNodeReady for more than one distinct graph node, found: \
         {distinct_ready_nodes:?}"
    );
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::TensorResourceProduced
            && observation.message.contains("node=")
            && observation.message.contains("resource=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::PlanSelected
            && observation.message.contains("request=e2e-success-path")
            && observation.message.contains("plan_generation=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::KernelResolved
            && observation.message.contains("kernel=")
            && observation.message.contains("provider=")
            && observation.message.contains("model_instance=")
    }));
    assert!(observations.iter().any(|observation| {
        observation.kind == InferenceApiObservationKind::PlanSelected
            && observation.message.contains("phase=decode")
            && observation.message.contains("kv_position=")
    }));
    assert!(
        outcome
            .kv_observations
            .iter()
            .any(|observation| observation.kind == KvCacheObservationKind::PrefillCompleted)
    );
    assert!(
        outcome
            .kv_observations
            .iter()
            .any(|observation| observation.kind == KvCacheObservationKind::DecodeAppend)
    );
    assert!(outcome.kv_observations.iter().all(|observation| {
        !observation.raw_prompt_available
            && !observation.raw_cache_available
            && !observation.raw_provider_handle_available
    }));
}

#[test]
fn e2e_chat_session_close_releases_kv_cache_and_model_instance() {
    check_chat_session_close_releases_kv_cache_and_model_instance()
        .expect("closing a chat session releases its KV cache and model instance");
}

#[test]
fn e2e_chat_sessions_are_isolated_from_each_other() {
    check_chat_sessions_are_isolated_from_each_other()
        .expect("two chat sessions for the same model are isolated from each other");
}
