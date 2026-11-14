//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_graph_nodes_transport_stays_tensor_value_typed() {
    check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls().expect(
        "execute_first_native_graph_nodes's per-node transport has no direct HostTensor-typed calls",
    );
}

/// task group 6 (`make-first-native-cuda-hot-path-device-resident`'s
/// Decision 7) / 6.3: a failure in the Kernel's own execution/completion
/// (here, `ctx.provider.complete_kernel` itself, injected via
/// `KernelFailableProviderExecutionApi`) must roll back the output that
/// node had already pre-admitted before dispatch, exactly like a
/// submit-time failure. Targets `embedding`, the first node this fixture's
/// prefill graph dispatches, so zero earlier nodes have succeeded yet --
/// the Runtime's Active allocation count right after model loading (weights
/// staged, nothing dispatched) must be unchanged after the failed attempt,
/// which is a stronger, id-independent way to prove "rolled back, not
/// leaked" than picking out one allocation by id.
#[test]
fn e2e_graph_dispatch_rolls_back_pre_admitted_output_on_kernel_completion_failure() {
    let fixture = e2e_fixture().expect("fixture builds");
    let executor = Arc::new(KernelFailableProviderExecutionApi::new("embedding"));
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(TestFailableKernelProvider {
            executor: executor.clone(),
        }))
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
    let cache_id =
        KvCacheId::new("test-kernel-completion-rollback-cache").expect("cache id is valid");

    let active_before = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();

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
        "expected the injected embedding completion failure to fail graph dispatch"
    );

    let active_after = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(
        active_after,
        active_before + 1,
        "a kernel/completion-time failure on the very first graph node must leave the \
         Runtime's Active allocation count exactly one higher than it was right after model \
         loading -- the one legitimate admission being the graph's own 'input.token_ids' \
         resource (admitted up front, before any node dispatches -- see \
         `execute_first_native_graph_nodes`'s `initial_bindings` loop); `embedding`'s own \
         pre-admitted output must be rolled back, not left behind as a second, leaked \
         allocation"
    );

    // audit-complet-cuda-hot-path-2026-09-08 P0-2 / Correctif B: the
    // allocation/residency check above only proves the *Memory Manager*
    // side was rolled back -- Reference CPU's `submit_kernel_invocation`
    // runs the Kernel and writes its output into Provider storage
    // synchronously, *before* `complete_kernel` is ever called, so by the
    // time this mock's injected `complete_kernel` failure fires, the
    // Provider genuinely already materialized `embedding`'s output. The
    // fix must release it from Provider storage too, not just Memory
    // Manager's ledger -- checked directly against the same `executor`
    // this dispatch actually ran through, not inferred from the
    // allocation count alone.
    let embedding_node = graphs
        .prefill
        .nodes
        .values()
        .find(|node| node.operator.name() == "embedding")
        .expect("this fixture's prefill graph has an embedding node");
    let embedding_output_edge = embedding_node
        .outputs
        .first()
        .expect("embedding node has an output edge");
    let embedding_output_id = TensorResourceId::new(format!("edge.{embedding_output_edge}"));
    assert!(
        executor.read_tensor_value(&embedding_output_id).is_none(),
        "a kernel/completion-time failure must also release the pre-admitted output from \
         Provider storage, not just Memory Manager's residency ledger -- \
         `embedding`'s output was genuinely materialized Provider-side before completion \
         failed, and must not survive rollback as an orphaned Provider-side resource"
    );

    // 6.4/6.5: a retry with the injected failure cleared must succeed
    // cleanly -- proving the rollback left no orphaned residency/allocation
    // under the same resource id that would otherwise collide with, or be
    // silently reused by, the retried dispatch.
    let mut retry_plans = first_native_plans_for_prompt(&runtime, &fixture, &instance, 2)
        .expect("prepared plans build for retry");
    let retry_ids = HostTensor::new([2], vec![1.0, 2.0]).expect("token id tensor builds");
    let retry_cache_id =
        KvCacheId::new("test-kernel-completion-rollback-retry-cache").expect("cache id is valid");
    *executor.fail_operator.lock().unwrap() = None;
    executor.failing_handles.lock().unwrap().clear();
    let retry_result = execute_first_native_graph(
        &mut runtime,
        &fixture,
        &instance,
        &retry_cache_id,
        &graphs.prefill,
        &mut retry_plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), retry_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    assert!(
        retry_result.is_ok(),
        "a retried dispatch after a rolled-back failure must succeed cleanly: {:?}",
        retry_result.err()
    );
}

/// `make-first-native-cuda-hot-path-device-resident` task 5.3: `lm_head`'s
/// tied-embedding weight is transposed exactly once, at Model Load
/// (`bind_qwen_fixture_weights` -> `qwen_weights_with_derived_lm_head`),
/// not recomputed by `resolve_first_native_weight_edge` on every generation step.
/// Proven by running two separate graph dispatches (prefill, then decode)
/// against the same `ModelInstance` and confirming: the `lm_head` resource
/// binding is the same resource id both times, the Provider's stored data
/// for it is byte-identical to `token_embedding`'s transpose computed
/// independently, and that data is unchanged after the second dispatch --
/// nothing re-wrote it in between.
#[test]
fn lm_head_weight_is_transposed_once_at_model_load_not_per_generation_step() {
    let fixture = e2e_fixture().expect("fixture builds");
    assert!(
        fixture.config.tied_embeddings,
        "this test's premise requires a tied-embeddings fixture"
    );
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();

    let weight_bindings = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    let lm_head_id = weight_bindings
        .get("lm_head")
        .expect("tied-embeddings Model Load must stage an lm_head weight")
        .clone();

    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding").unwrap();
    let expected = transpose_rows_cols(token_embedding).unwrap();

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let staged_before = executor.read_tensor(&lm_head_id).unwrap();
    assert_eq!(
        staged_before.data, expected.data,
        "staged lm_head must equal token_embedding's transpose"
    );

    // Two full dispatches (prefill, then decode) through the real graph
    // executor, reusing the same ModelInstance/weight bindings.
    let prompt = [1, 2];
    let mut plans =
        first_native_plans_for_prompt(&runtime, &fixture, &instance, prompt.len() as u64).unwrap();
    let graphs = first_native_component_graphs_for_prompt(&fixture, prompt.len() as u64).unwrap();
    let cache_id = KvCacheId::new("test-lm-head-single-transpose-cache").unwrap();
    let prompt_ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let (_dispatch, _bindings, layer_kv, _provider) = execute_first_native_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), prompt_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .unwrap();
    let admitted_ids = HostTensor::new([1], vec![3.0]).unwrap();
    execute_first_native_graph(
        &mut runtime,
        &fixture,
        &instance,
        &cache_id,
        &graphs.decode,
        &mut plans.decode,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), admitted_ids)]),
        Some(&layer_kv),
        Some(prompt.len() as u64),
        &mut Vec::new(),
    )
    .unwrap();

    // Same resource id, same data, after both dispatches: nothing staged a
    // second (or third) copy or rewrote it per step.
    let weight_bindings_after = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    assert_eq!(weight_bindings_after.get("lm_head"), Some(&lm_head_id));
    let staged_after = executor.read_tensor(&lm_head_id).unwrap();
    assert_eq!(
        staged_after.data, expected.data,
        "lm_head's staged data must still equal the same transpose after two dispatches, not have been recomputed or corrupted"
    );
}

#[test]
fn provider_execution_generic_resolution_reaches_non_reference_cpu_provider() {
    let provider_binding = ProviderBinding::new("magnetar:provider/mock-kernel");
    let runtime = Runtime::builder()
        .register_provider(Arc::new(MockKernelProvider::new()))
        .build()
        .expect("mock provider registers cleanly");

    let api = resolve_kernel_execution_provider(&runtime, &provider_binding).expect(
        "generic resolution finds the registered mock provider without downcasting to a \
         concrete Provider type",
    );

    // Borrow an arbitrary, valid advertisement/operator pair from Reference
    // CPU purely as inert filler: the mock ignores their content entirely,
    // and constructing one from scratch is not the point of this test.
    let reference_cpu = ReferenceCpuProvider::new();
    let advertisement = reference_cpu
        .kernel_advertisements()
        .into_iter()
        .next()
        .expect("Reference CPU advertises at least one Kernel");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let input_id = TensorResourceId::new("mock-input");
    let output_id = TensorResourceId::new("mock-output");
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    api.write_tensor(
        input_id.clone(),
        HostTensor::new([2], vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("mock-invocation"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        provider_binding.clone(),
        affinity.clone(),
    )
    .with_input(KernelResource::new(
        TensorResourceDescriptor::new(input_id, descriptor.clone(), affinity.clone()),
        KernelMemoryClass::Host,
    ))
    .with_output(KernelResource::new(
        TensorResourceDescriptor::new(output_id.clone(), descriptor, affinity),
        KernelMemoryClass::Host,
    ));

    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let handle = api
        .submit_kernel(&advertisement, operator, &invocation, &mut memory)
        .expect("mock Provider implements Kernel-level submission");
    let result = api
        .complete_kernel(&handle)
        .expect("mock Provider implements Kernel-level completion");
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = api
        .read_tensor(&output_id)
        .expect("mock Provider's minimal kernel copied input to output");
    assert_eq!(output.data, vec![1.0, 2.0]);
}

/// Same fix, the other branch: a state with no bound Provider (never
/// touched by `execute_first_native_graph`, e.g. a freshly created prefill state)
/// still falls back to Reference CPU -- preserving today's only real
/// behavior exactly.
#[test]
fn kv_update_transaction_falls_back_to_reference_cpu_when_unbound() {
    let runtime = build_runtime_trusting_fixture(&e2e_fixture().expect("fixture builds"));

    let state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("test-kv-transaction-fallback-cache").unwrap(),
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer").unwrap(),
        ),
        layer_kv: LayerKvMap::new(),
        provider: None,
    };

    let transaction = KvUpdateTransaction::begin(&runtime, &state).expect("reference CPU resolves");
    assert_eq!(
        transaction.provider_binding,
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)
    );
}

#[test]
fn e2e_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage(&fixture)
        .expect("intermediate graph edges resolve from Provider storage, not a private cache");
}

#[test]
fn graph_execution_order_matches_the_previous_quadratic_order() {
    let fixture = e2e_fixture().expect("fixture builds");
    let graphs =
        first_native_component_graphs_for_prompt(&fixture, 5).expect("component graphs build");
    for graph in [&graphs.prefill, &graphs.decode] {
        let order = first_native_graph_execution_order(graph).unwrap();
        assert_eq!(order.len(), graph.nodes.len());
        assert_eq!(Some(order), quadratic_execution_order(graph));
    }

    let mut cyclic = graphs.decode.clone();
    let (first, second) = {
        let mut ids = cyclic.nodes.keys().cloned();
        (ids.next().unwrap(), ids.next().unwrap())
    };
    let second_output = cyclic.nodes[&second].outputs[0].clone();
    let first_output = cyclic.nodes[&first].outputs[0].clone();
    cyclic
        .nodes
        .get_mut(&first)
        .unwrap()
        .inputs
        .push(second_output);
    cyclic
        .nodes
        .get_mut(&second)
        .unwrap()
        .inputs
        .push(first_output);
    assert!(quadratic_execution_order(&cyclic).is_none());
    assert!(matches!(
        first_native_graph_execution_order(&cyclic),
        Err(InferenceApiError::GraphPlanningFailed { .. })
    ));
}
