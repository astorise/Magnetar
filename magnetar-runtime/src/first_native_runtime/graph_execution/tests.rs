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

// ---------------------------------------------------------------------------
// `per-layer-kv-geometry`: decoder layers whose KV geometry differs, and a
// layer that attends over an earlier layer's K/V.
// ---------------------------------------------------------------------------

/// One attention layer of [`layered_graph`].
#[derive(Clone, Copy)]
struct LayeredAttention {
    kv_heads: u64,
    head_dimension: u64,
    window: Option<u64>,
    scale: Option<f64>,
    /// The earlier layer whose K/V this layer attends over.
    kv_source: Option<usize>,
}

const LAYERED_HIDDEN: u64 = 4;
const LAYERED_HEADS: u64 = 2;
const LAYERED_VOCAB: u64 = 6;

/// Sliding window with grouped K/V, then full attention with its own score
/// scale and a wider K/V row, then a layer reusing layer 0's K/V.
const LAYERED_LAYERS: [LayeredAttention; 3] = [
    LayeredAttention {
        kv_heads: 1,
        head_dimension: 4,
        window: Some(2),
        scale: None,
        kv_source: None,
    },
    LayeredAttention {
        kv_heads: 2,
        head_dimension: 3,
        window: None,
        scale: Some(0.25),
        kv_source: None,
    },
    LayeredAttention {
        kv_heads: 1,
        head_dimension: 4,
        window: Some(2),
        scale: None,
        kv_source: Some(0),
    },
];

/// Deterministic weights of [`layered_graph`], keyed by input name.
fn layered_weights() -> BTreeMap<String, HostTensor> {
    let mut seed = 7u32;
    let mut tensor = |rows: u64, cols: u64| {
        let values: Vec<f32> = (0..rows * cols)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 16) % 1000) as f32 / 1000.0 - 0.5
            })
            .collect();
        HostTensor::new([rows, cols], values).unwrap()
    };
    let mut weights = BTreeMap::new();
    for (index, layer) in LAYERED_LAYERS.iter().enumerate() {
        let q_width = LAYERED_HEADS * layer.head_dimension;
        let kv_width = layer.kv_heads * layer.head_dimension;
        weights.insert(format!("w{index}q"), tensor(LAYERED_HIDDEN, q_width));
        if layer.kv_source.is_none() {
            weights.insert(format!("w{index}k"), tensor(LAYERED_HIDDEN, kv_width));
            weights.insert(format!("w{index}v"), tensor(LAYERED_HIDDEN, kv_width));
        }
        weights.insert(format!("w{index}o"), tensor(q_width, LAYERED_HIDDEN));
    }
    weights.insert("wout".into(), tensor(LAYERED_HIDDEN, LAYERED_VOCAB));
    weights
}

/// A graph of `rows` rows over [`LAYERED_LAYERS`], with every weight a
/// graph input: `x -> [q, k, v -> attention -> o -> +x] -> logits`. A
/// decode graph appends its K/V rows to the history.
fn layered_graph(rows: u64, decode: bool) -> ExecutionGraph {
    let phase = if decode {
        ExecutionGraphPhase::Decode
    } else {
        ExecutionGraphPhase::Prefill
    };
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new(format!("layered-{rows}-{decode}")),
        phase,
    );
    let input = |graph: &mut ExecutionGraph, name: &str, shape: Vec<u64>| {
        let id = TensorEdgeId::new(format!("input.{name}"));
        graph.edges.insert(
            id.clone(),
            TensorEdge::new(id.clone(), f32_tensor_descriptor_from_shape(&shape)),
        );
        id
    };
    let node = |graph: &mut ExecutionGraph,
                id: &str,
                operator: &str,
                family: OperatorFamily,
                attributes: Vec<(&str, OperatorAttributeValue)>,
                inputs: Vec<TensorEdgeId>,
                shape: Vec<u64>,
                kv: Option<(usize, &str)>| {
        let node_id = ExecutionNodeId::new(id);
        let output = TensorEdgeId::new(format!("edge.{id}"));
        let mut execution_node =
            ExecutionNode::new(node_id.clone(), OperatorId::magnetar(operator, 1, family));
        for edge in &inputs {
            execution_node = execution_node.with_input(edge.clone());
            graph
                .edges
                .get_mut(edge)
                .unwrap()
                .consumers
                .insert(node_id.clone());
        }
        execution_node = execution_node.with_output(output.clone());
        for (name, value) in attributes {
            execution_node = execution_node.with_attribute(name, value);
        }
        let mut edge = TensorEdge::new(output.clone(), f32_tensor_descriptor_from_shape(&shape))
            .with_producer(node_id.clone());
        if let Some((layer, role)) = kv {
            edge.kv_cache = Some(GraphKvCacheMetadata {
                cache_id: format!("layered.layer{layer}.{role}"),
                behavior: if decode {
                    GraphKvCacheBehavior::Append
                } else {
                    GraphKvCacheBehavior::Output
                },
                paged: false,
                compatibility_key: "layered".into(),
            });
        }
        graph.nodes.insert(node_id, execution_node);
        graph.edges.insert(output.clone(), edge);
        output
    };
    let weights = layered_weights();
    let mut hidden = input(&mut graph, "x", vec![rows, LAYERED_HIDDEN]);
    let mut produced: Vec<(TensorEdgeId, TensorEdgeId)> = Vec::new();
    for (index, layer) in LAYERED_LAYERS.iter().enumerate() {
        let weight = |graph: &mut ExecutionGraph, role: &str| {
            let name = format!("w{index}{role}");
            let shape = weights[&name].shape.clone();
            input(graph, &name, shape)
        };
        let q_width = LAYERED_HEADS * layer.head_dimension;
        let kv_width = layer.kv_heads * layer.head_dimension;
        let wq = weight(&mut graph, "q");
        let q = node(
            &mut graph,
            &format!("layer{index}.q"),
            "matmul",
            OperatorFamily::LinearAlgebra,
            vec![],
            vec![hidden.clone(), wq],
            vec![rows, q_width],
            None,
        );
        let (k, v) = match layer.kv_source {
            Some(source) => produced[source].clone(),
            None => {
                let wk = weight(&mut graph, "k");
                let wv = weight(&mut graph, "v");
                let k = node(
                    &mut graph,
                    &format!("layer{index}.k"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                    vec![],
                    vec![hidden.clone(), wk],
                    vec![rows, kv_width],
                    Some((index, "k")),
                );
                let v = node(
                    &mut graph,
                    &format!("layer{index}.v"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                    vec![],
                    vec![hidden.clone(), wv],
                    vec![rows, kv_width],
                    Some((index, "v")),
                );
                (k, v)
            }
        };
        produced.push((k.clone(), v.clone()));
        let mut attributes = vec![
            ("causal", OperatorAttributeValue::Boolean(true)),
            (
                "head_count",
                OperatorAttributeValue::Integer(LAYERED_HEADS as i64),
            ),
            (
                "kv_head_count",
                OperatorAttributeValue::Integer(layer.kv_heads as i64),
            ),
            (
                "head_dimension",
                OperatorAttributeValue::Integer(layer.head_dimension as i64),
            ),
        ];
        if let Some(window) = layer.window {
            attributes.push((
                "window_size",
                OperatorAttributeValue::Integer(window as i64),
            ));
        }
        if let Some(scale) = layer.scale {
            attributes.push(("scale", OperatorAttributeValue::Float(scale)));
        }
        let attended = node(
            &mut graph,
            &format!("layer{index}.attention"),
            "attention",
            OperatorFamily::Attention,
            attributes,
            vec![q, k, v],
            vec![rows, q_width],
            None,
        );
        let wo = weight(&mut graph, "o");
        let o = node(
            &mut graph,
            &format!("layer{index}.o"),
            "matmul",
            OperatorFamily::LinearAlgebra,
            vec![],
            vec![attended, wo],
            vec![rows, LAYERED_HIDDEN],
            None,
        );
        hidden = node(
            &mut graph,
            &format!("layer{index}.residual"),
            "add",
            OperatorFamily::Tensor,
            vec![],
            vec![hidden, o],
            vec![rows, LAYERED_HIDDEN],
            None,
        );
    }
    let wout = input(&mut graph, "wout", vec![LAYERED_HIDDEN, LAYERED_VOCAB]);
    let logits = node(
        &mut graph,
        "lm_head",
        "matmul",
        OperatorFamily::LinearAlgebra,
        vec![],
        vec![hidden, wout],
        vec![rows, LAYERED_VOCAB],
        None,
    );
    // The graph's output edge is named `logits`, as a Component's is.
    let mut edge = graph.edges.remove(&logits).unwrap();
    edge.id = TensorEdgeId::new("logits");
    graph.edges.insert(edge.id.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("lm_head"))
        .unwrap()
        .outputs = vec![TensorEdgeId::new("logits")];
    graph
}

/// Row `index` of the deterministic input sequence.
fn layered_input_row(index: u64) -> Vec<f32> {
    (0..LAYERED_HIDDEN)
        .map(|column| ((index * 3 + column * 5) % 7) as f32 / 7.0 - 0.4)
        .collect()
}

fn host_matmul(a: &[f32], rows: usize, b: &HostTensor) -> Vec<f32> {
    let (inner, cols) = (b.shape[0] as usize, b.shape[1] as usize);
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = (0..inner)
                .map(|i| a[row * inner + i] * b.data[i * cols + col])
                .sum();
        }
    }
    out
}

/// The logits of every row of `rows` input rows, computed on the host
/// independently of any Kernel.
fn layered_oracle(rows: u64) -> Vec<f32> {
    let weights = layered_weights();
    let n = rows as usize;
    let mut hidden: Vec<f32> = (0..rows).flat_map(layered_input_row).collect();
    let mut produced: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for (index, layer) in LAYERED_LAYERS.iter().enumerate() {
        let d = layer.head_dimension as usize;
        let heads = LAYERED_HEADS as usize;
        let kv_heads = layer.kv_heads as usize;
        let q = host_matmul(&hidden, n, &weights[&format!("w{index}q")]);
        let (k, v) = match layer.kv_source {
            Some(source) => produced[source].clone(),
            None => (
                host_matmul(&hidden, n, &weights[&format!("w{index}k")]),
                host_matmul(&hidden, n, &weights[&format!("w{index}v")]),
            ),
        };
        produced.push((k.clone(), v.clone()));
        let scale = layer.scale.unwrap_or(1.0 / (d as f64).sqrt()) as f32;
        let mut attended = vec![0.0f32; n * heads * d];
        for i in 0..n {
            for h in 0..heads {
                let g = h / (heads / kv_heads);
                let lowest = layer
                    .window
                    .map_or(0, |window| (i + 1).saturating_sub(window as usize));
                let scores: Vec<f32> = (lowest..=i)
                    .map(|j| {
                        (0..d)
                            .map(|e| q[i * heads * d + h * d + e] * k[j * kv_heads * d + g * d + e])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = exps.iter().sum();
                for (offset, weight) in exps.iter().enumerate() {
                    let j = lowest + offset;
                    for e in 0..d {
                        attended[i * heads * d + h * d + e] +=
                            weight / total * v[j * kv_heads * d + g * d + e];
                    }
                }
            }
        }
        let o = host_matmul(&attended, n, &weights[&format!("w{index}o")]);
        for (value, delta) in hidden.iter_mut().zip(o) {
            *value += delta;
        }
    }
    host_matmul(&hidden, n, &weights["wout"])
}

/// Runs [`layered_graph`] on the Reference CPU: a prefill of `prompt` rows,
/// then one decode step over its K/V history. Returns the prefill's logits,
/// the decode step's, and each layer's K store width after the prefill.
fn run_layered(prompt: u64, in_place: bool) -> (Vec<f32>, Vec<f32>, BTreeMap<usize, u64>) {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    // The graphs carry their own weights; execution reads the layer count
    // and layout from the configuration.
    let mut fixture = fixture;
    fixture.config.architecture.layer_count = LAYERED_LAYERS.len() as u64;
    fixture.config.layer_layout = Some(
        LAYERED_LAYERS
            .iter()
            .map(|layer| crate::LayerSpec {
                sliding_window: layer.window.map(|window| window as u32),
                head_dim: layer.head_dimension,
                num_key_value_heads: layer.kv_heads as u32,
                rope_theta: 10_000.0,
                partial_rotary_factor: None,
                value_from_key: false,
                kv_source_layer: layer.kv_source.map(|source| source as u32),
                intermediate_size: 8,
            })
            .collect(),
    );
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let weights = layered_weights();
    let bindings = |rows: std::ops::Range<u64>| {
        let count = rows.end - rows.start;
        let mut bindings: BTreeMap<TensorEdgeId, HostTensor> = weights
            .iter()
            .map(|(name, tensor)| (TensorEdgeId::new(format!("input.{name}")), tensor.clone()))
            .collect();
        bindings.insert(
            TensorEdgeId::new("input.x"),
            HostTensor::new(
                [count, LAYERED_HIDDEN],
                rows.flat_map(layered_input_row).collect::<Vec<f32>>(),
            )
            .unwrap(),
        );
        bindings
    };
    let stores = in_place.then(|| KvStoreSpec::owned(prompt + 4));
    let mut run = |graph: &ExecutionGraph,
                   rows: std::ops::Range<u64>,
                   cache: &str,
                   history: Option<&LayerKvMap>| {
        let mut plan = prepare_first_native_plan_for_graph(
            &runtime,
            graph,
            &instance,
            revision,
            rows.end - rows.start,
            PreparedExecutionPlanGeneration::new(1),
            &provider,
        )
        .unwrap();
        let position = rows.start;
        execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &KvCacheId::new(cache).unwrap(),
            graph,
            &mut plan,
            bindings(rows),
            history,
            Some(position),
            &mut Vec::new(),
            MaterializedEdges::Only(&["logits"]),
            None,
            stores,
        )
        .unwrap()
    };
    let (_, prefill, layer_kv, _) = run(&layered_graph(prompt, false), 0..prompt, "lp", None);
    let (_, decode, _, _) = run(
        &layered_graph(1, true),
        prompt..prompt + 1,
        "ld",
        Some(&layer_kv),
    );
    let widths = layer_kv
        .iter()
        .map(|(layer, state)| {
            let shape = resolve_kernel_execution_provider(&runtime, &provider)
                .unwrap()
                .resident_shape(&state.k)
                .unwrap();
            (*layer, shape[1])
        })
        .collect();
    let logits = |bindings: &BTreeMap<TensorEdgeId, HostTensor>| {
        bindings[&TensorEdgeId::new("logits")].data.clone()
    };
    (logits(&prefill), logits(&decode), widths)
}

fn assert_logits_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: logits count");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4,
            "{what}: logit {index} is {a}, expected {e}"
        );
    }
}

/// Each layer attends with its own head geometry, window and score scale;
/// the shared layer reads layer 0's K/V; only the producing layers keep K/V
/// state, each at its own width; and a decode step over that history
/// matches the full sequence, with the history concatenated or in place.
#[test]
fn layers_with_their_own_kv_geometry_and_a_shared_kv_layer_prefill_and_decode() {
    let prompt = 4;
    let full = layered_oracle(prompt + 1);
    let width = LAYERED_VOCAB as usize;
    for in_place in [false, true] {
        let (prefill, decode, widths) = run_layered(prompt, in_place);
        assert_logits_close(
            &prefill,
            &full[..prompt as usize * width],
            &format!("prefill (in place: {in_place})"),
        );
        assert_logits_close(
            &decode,
            &full[prompt as usize * width..],
            &format!("decode (in place: {in_place})"),
        );
        // Layer 2 reuses layer 0's K/V and keeps none of its own.
        assert_eq!(
            widths,
            BTreeMap::from([(0, 4), (1, 6)]),
            "K widths (in place: {in_place})"
        );
    }
}

/// The Reference CPU's attention Kernels honour `scale`, so the Runtime may
/// send it there; a feature no Kernel of an operator advertises is refused.
#[test]
fn attention_scale_is_sent_only_to_kernels_that_declare_it() {
    let fixture = e2e_fixture().expect("fixture builds");
    let runtime = build_runtime_trusting_fixture(&fixture);
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    assert!(provider_kernels_declare(
        &runtime,
        &provider,
        "attention",
        crate::operator::ATTENTION_SCALE_FEATURE
    ));
    assert!(!provider_kernels_declare(
        &runtime,
        &provider,
        "rope",
        crate::operator::ATTENTION_SCALE_FEATURE
    ));
    assert!(!provider_kernels_declare(
        &runtime,
        &ProviderBinding::new("unregistered"),
        "attention",
        crate::operator::ATTENTION_SCALE_FEATURE
    ));
}
