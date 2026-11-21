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

// ---------------------------------------------------------------------------
// `dispatch-layered-model-forms`: every optional operator form of the
// layered models, through the first-native engine.
// ---------------------------------------------------------------------------

/// A graph under construction, every weight a graph input.
struct FormsGraph {
    graph: ExecutionGraph,
}

impl FormsGraph {
    fn input(&mut self, name: &str, shape: &[u64]) -> TensorEdgeId {
        let id = TensorEdgeId::new(format!("input.{name}"));
        self.graph.edges.insert(
            id.clone(),
            TensorEdge::new(id.clone(), f32_tensor_descriptor_from_shape(shape)),
        );
        id
    }

    fn node(
        &mut self,
        id: &str,
        operator: &str,
        family: OperatorFamily,
        attributes: Vec<(&str, OperatorAttributeValue)>,
        inputs: Vec<TensorEdgeId>,
        shape: &[u64],
    ) -> TensorEdgeId {
        let node_id = ExecutionNodeId::new(id);
        let output = TensorEdgeId::new(format!("edge.{id}"));
        let mut node =
            ExecutionNode::new(node_id.clone(), OperatorId::magnetar(operator, 1, family));
        for edge in &inputs {
            node = node.with_input(edge.clone());
            self.graph
                .edges
                .get_mut(edge)
                .unwrap()
                .consumers
                .insert(node_id.clone());
        }
        node = node.with_output(output.clone());
        for (name, value) in attributes {
            node = node.with_attribute(name, value);
        }
        self.graph.edges.insert(
            output.clone(),
            TensorEdge::new(output.clone(), f32_tensor_descriptor_from_shape(shape))
                .with_producer(node_id.clone()),
        );
        self.graph.nodes.insert(node_id, node);
        output
    }
}

/// Two independent learned residual updates exercise the portable graph
/// path with original layer indices and distinct mixer/MLP weight edges.
#[test]
fn geodesic_residual_graph_preserves_parameters_and_original_layer_indices() {
    let mut builder = FormsGraph {
        graph: ExecutionGraph::new(
            ExecutionGraphId::new("geodesic-residuals"),
            ExecutionGraphPhase::Prefill,
        ),
    };
    let x = builder.input("residual", &[1, 4]);
    let g = builder.input("branch", &[1, 4]);
    let mixer_scale = builder.input("mixer.scale", &[1]);
    let mixer_bias = builder.input("mixer.bias", &[1]);
    let mlp_scale = builder.input("mlp.scale", &[1]);
    let mlp_bias = builder.input("mlp.bias", &[1]);
    let mixer = builder.node(
        "mixer",
        "geodesic-residual",
        OperatorFamily::Tensor,
        vec![("layer_index", OperatorAttributeValue::Integer(0))],
        vec![x.clone(), g.clone(), mixer_scale, mixer_bias],
        &[1, 4],
    );
    let mlp = builder.node(
        "mlp",
        "geodesic-residual",
        OperatorFamily::Tensor,
        vec![("layer_index", OperatorAttributeValue::Integer(11))],
        vec![x, g, mlp_scale, mlp_bias],
        &[1, 4],
    );
    let output = builder.node(
        "sum",
        "add",
        OperatorFamily::Tensor,
        vec![],
        vec![mixer, mlp],
        &[1, 4],
    );
    let mut graph = builder.graph;
    let mut edge = graph.edges.remove(&output).unwrap();
    edge.id = TensorEdgeId::new("logits");
    graph.edges.insert(edge.id.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("sum"))
        .unwrap()
        .outputs = vec![TensorEdgeId::new("logits")];
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &graph,
        &instance,
        revision,
        1,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    for (mlp_parameter, expected) in [
        (1.2, vec![2.9013255, -2.2552404, -0.391_262, 8.194331]),
        (0.0, vec![2.8208141, -2.37731, -0.28452742, 8.144489]),
    ] {
        let tensors = [
            (
                "residual",
                HostTensor::new([1, 4], vec![1.0, -2.0, 0.5, 4.0]).unwrap(),
            ),
            (
                "branch",
                HostTensor::new([1, 4], vec![0.7, 1.2, -1.0, 0.3]).unwrap(),
            ),
            ("mixer.scale", HostTensor::new([1], vec![1.2]).unwrap()),
            ("mixer.bias", HostTensor::new([1], vec![0.04]).unwrap()),
            (
                "mlp.scale",
                HostTensor::new([1], vec![mlp_parameter]).unwrap(),
            ),
            (
                "mlp.bias",
                HostTensor::new([1], vec![if mlp_parameter == 0.0 { 0.0 } else { 0.04 }]).unwrap(),
            ),
        ];
        let bindings = tensors
            .into_iter()
            .map(|(name, value)| (TensorEdgeId::new(format!("input.{name}")), value))
            .collect();
        let (_, outputs, _, _) = execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &KvCacheId::new("geodesic").unwrap(),
            &graph,
            &mut plan,
            bindings,
            None,
            Some(0),
            &mut Vec::new(),
            MaterializedEdges::Only(&["logits"]),
            None,
            None,
        )
        .unwrap();
        assert_logits_close(
            &outputs[&TensorEdgeId::new("logits")].data,
            &expected,
            "geodesic graph",
        );
    }
}

const FORMS_ROWS: u64 = 3;
const FORMS_VOCAB: u64 = 5;
const FORMS_HEADS: u64 = 2;
const FORMS_HEAD_DIM: u64 = 4;
const FORMS_WIDTH: u64 = FORMS_HEADS * FORMS_HEAD_DIM;
const FORMS_EXPERTS: u64 = 3;
const FORMS_EXPERT_WIDTH: u64 = 2;
const FORMS_OUT: u64 = 6;

fn forms_inputs() -> BTreeMap<String, HostTensor> {
    let mut seed = 11u32;
    let mut tensor = |shape: Vec<u64>| {
        let count: u64 = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 16) % 1000) as f32 / 500.0 - 1.0
            })
            .collect();
        HostTensor::new(shape, values).unwrap()
    };
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "token_ids".into(),
        HostTensor::new([FORMS_ROWS], vec![4.0, 1.0, 3.0]).unwrap(),
    );
    inputs.insert("table".into(), tensor(vec![FORMS_VOCAB, FORMS_WIDTH]));
    inputs.insert("k_norm".into(), tensor(vec![FORMS_HEAD_DIM]));
    inputs.insert("layer_scalar".into(), tensor(vec![1]));
    inputs.insert("router".into(), tensor(vec![FORMS_WIDTH, FORMS_EXPERTS]));
    inputs.insert(
        "gate".into(),
        tensor(vec![FORMS_WIDTH, FORMS_EXPERTS * FORMS_EXPERT_WIDTH]),
    );
    inputs.insert(
        "up".into(),
        tensor(vec![FORMS_WIDTH, FORMS_EXPERTS * FORMS_EXPERT_WIDTH]),
    );
    inputs.insert(
        "down".into(),
        tensor(vec![FORMS_EXPERT_WIDTH, FORMS_EXPERTS * FORMS_WIDTH]),
    );
    inputs.insert("per_expert_scale".into(), tensor(vec![FORMS_EXPERTS]));
    inputs.insert("head".into(), tensor(vec![FORMS_WIDTH, FORMS_OUT]));
    inputs
}

/// `embedding(scale) -> rmsnorm(per head, no weight) -> rmsnorm(per head,
/// [head_dimension] weight) -> rope(partial) -> gelu(tanh) -> mul([1]) ->
/// activation(scale) -> moe-experts(gelu_tanh, per_expert_scale) -> matmul
/// -> activation(softcap) = logits`.
fn forms_graph() -> ExecutionGraph {
    let mut g = FormsGraph {
        graph: ExecutionGraph::new(
            ExecutionGraphId::new("layered-forms"),
            ExecutionGraphPhase::Prefill,
        ),
    };
    let rows = FORMS_ROWS;
    let hidden = [rows, FORMS_WIDTH];
    let inputs = forms_inputs();
    let input = |g: &mut FormsGraph, name: &str| g.input(name, &inputs[name].shape.clone());
    let heads = |g: &FormsGraph| {
        let _ = g;
        vec![
            (
                "head_count",
                OperatorAttributeValue::Integer(FORMS_HEADS as i64),
            ),
            (
                "head_dimension",
                OperatorAttributeValue::Integer(FORMS_HEAD_DIM as i64),
            ),
        ]
    };
    let ids = input(&mut g, "token_ids");
    let table = input(&mut g, "table");
    let embedded = g.node(
        "embed",
        "embedding",
        OperatorFamily::Tensor,
        vec![("scale", OperatorAttributeValue::Float(2.0))],
        vec![ids, table],
        &hidden,
    );
    let x = embedded.clone();
    let mut attributes = heads(&g);
    attributes.push(("epsilon", OperatorAttributeValue::Float(1e-6)));
    attributes.push(("with_scale", OperatorAttributeValue::Boolean(false)));
    let x = g.node(
        "v_norm",
        "rmsnorm",
        OperatorFamily::Normalization,
        attributes,
        vec![x],
        &hidden,
    );
    let k_norm = input(&mut g, "k_norm");
    let mut attributes = heads(&g);
    attributes.push(("epsilon", OperatorAttributeValue::Float(1e-6)));
    let x = g.node(
        "k_norm",
        "rmsnorm",
        OperatorFamily::Normalization,
        attributes,
        vec![x, k_norm],
        &hidden,
    );
    let x = g.node(
        "rope",
        "rope",
        OperatorFamily::PositionEncoding,
        vec![
            ("base", OperatorAttributeValue::Float(4.0)),
            (
                "dimension",
                OperatorAttributeValue::Integer(FORMS_HEAD_DIM as i64),
            ),
            (
                "head_count",
                OperatorAttributeValue::Integer(FORMS_HEADS as i64),
            ),
            ("position_offset", OperatorAttributeValue::Integer(0)),
            (
                "rotation_pairing",
                OperatorAttributeValue::String("half_split".into()),
            ),
            ("partial_rotary_factor", OperatorAttributeValue::Float(0.5)),
        ],
        vec![x],
        &hidden,
    );
    let x = g.node(
        "gelu",
        "gelu",
        OperatorFamily::Activation,
        vec![("approximate", OperatorAttributeValue::String("tanh".into()))],
        vec![x],
        &hidden,
    );
    let layer_scalar = input(&mut g, "layer_scalar");
    let x = g.node(
        "layer_scalar",
        "mul",
        OperatorFamily::Tensor,
        vec![],
        vec![x, layer_scalar],
        &hidden,
    );
    let x = g.node(
        "scale",
        "activation",
        OperatorFamily::Activation,
        vec![
            ("kind", OperatorAttributeValue::String("scale".into())),
            ("factor", OperatorAttributeValue::Float(0.5)),
        ],
        vec![x],
        &hidden,
    );
    let router = input(&mut g, "router");
    let logits = g.node(
        "router",
        "matmul",
        OperatorFamily::LinearAlgebra,
        vec![],
        vec![x.clone(), router],
        &[rows, FORMS_EXPERTS],
    );
    let experts = ["gate", "up", "down", "per_expert_scale"].map(|name| input(&mut g, name));
    let mut moe_inputs = vec![x, logits];
    moe_inputs.extend(experts);
    let x = g.node(
        "experts",
        "moe-experts",
        OperatorFamily::LinearAlgebra,
        vec![
            (
                "experts",
                OperatorAttributeValue::Integer(FORMS_EXPERTS as i64),
            ),
            ("top_k", OperatorAttributeValue::Integer(2)),
            ("normalize_top_k", OperatorAttributeValue::Boolean(true)),
            (
                "activation",
                OperatorAttributeValue::String("gelu_tanh".into()),
            ),
        ],
        moe_inputs,
        &hidden,
    );
    // The scaled embeddings feed the head too, where RMSNorm cannot absorb
    // their scale.
    let x = g.node(
        "residual",
        "add",
        OperatorFamily::Tensor,
        vec![],
        vec![x, embedded],
        &hidden,
    );
    let head = input(&mut g, "head");
    let x = g.node(
        "head",
        "matmul",
        OperatorFamily::LinearAlgebra,
        vec![],
        vec![x, head],
        &[rows, FORMS_OUT],
    );
    let softcapped = g.node(
        "softcap",
        "activation",
        OperatorFamily::Activation,
        vec![
            ("kind", OperatorAttributeValue::String("softcap".into())),
            ("cap", OperatorAttributeValue::Float(0.8)),
        ],
        vec![x],
        &[rows, FORMS_OUT],
    );
    let mut graph = g.graph;
    let mut edge = graph.edges.remove(&softcapped).unwrap();
    edge.id = TensorEdgeId::new("logits");
    graph.edges.insert(edge.id.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("softcap"))
        .unwrap()
        .outputs = vec![TensorEdgeId::new("logits")];
    graph
}

/// [`forms_graph`]'s logits computed on the host, by hand except for the
/// routed experts (the shared `operator_semantics` function).
fn forms_oracle() -> Vec<f32> {
    let inputs = forms_inputs();
    let rows = FORMS_ROWS as usize;
    let (width, d, heads) = (
        FORMS_WIDTH as usize,
        FORMS_HEAD_DIM as usize,
        FORMS_HEADS as usize,
    );
    let embedded: Vec<f32> = inputs["token_ids"]
        .data
        .iter()
        .flat_map(|id| {
            let row = *id as usize;
            inputs["table"].data[row * width..(row + 1) * width]
                .iter()
                .map(|value| value * 2.0)
                .collect::<Vec<_>>()
        })
        .collect();
    let mut x = embedded.clone();
    let normalize = |x: &mut [f32], weight: Option<&[f32]>| {
        for block in x.chunks_mut(d) {
            let mean = block.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inverse = 1.0 / (mean + 1e-6).sqrt();
            for (index, value) in block.iter_mut().enumerate() {
                *value *= inverse * weight.map_or(1.0, |weight| weight[index]);
            }
        }
    };
    normalize(&mut x, None);
    normalize(&mut x, Some(&inputs["k_norm"].data));
    // Partial half-split RoPE: floor(0.5 * 4 / 2) = 1 of 2 pairs rotates,
    // with the full width as the frequency denominator.
    for row in 0..rows {
        for head in 0..heads {
            let base = row * width + head * d;
            let frequency = 4f32.powf(-0.0 / d as f32);
            let angle = row as f32 * frequency;
            let (a, b) = (x[base], x[base + d / 2]);
            x[base] = a * angle.cos() - b * angle.sin();
            x[base + d / 2] = a * angle.sin() + b * angle.cos();
        }
    }
    let scalar = inputs["layer_scalar"].data[0];
    for value in &mut x {
        *value = crate::operator_semantics::gelu_tanh(*value) * scalar * 0.5;
    }
    let h = HostTensor::new([FORMS_ROWS, FORMS_WIDTH], x.clone()).unwrap();
    let logits = host_matmul(&x, rows, &inputs["router"]);
    let logits = HostTensor::new([FORMS_ROWS, FORMS_EXPERTS], logits).unwrap();
    let experts = crate::operator_semantics::moe_experts_local_with(
        &h,
        &logits,
        [&inputs["gate"], &inputs["up"], &inputs["down"]],
        crate::operator_semantics::MoeRouting {
            experts: FORMS_EXPERTS,
            top_k: 2,
            normalize: true,
        },
        0,
        FORMS_EXPERTS,
        crate::operator_semantics::MoeExpertOptions {
            activation: crate::operator_semantics::MoeExpertActivation::GeluTanh,
            per_expert_scale: Some(&inputs["per_expert_scale"]),
        },
    )
    .unwrap();
    let residual: Vec<f32> = experts
        .data
        .iter()
        .zip(&embedded)
        .map(|(a, b)| a + b)
        .collect();
    host_matmul(&residual, rows, &inputs["head"])
        .into_iter()
        .map(|value| 0.8 * (value / 0.8).tanh())
        .collect()
}

/// Runs [`forms_graph`] on the Reference CPU and returns its logits, or
/// the dispatch error.
fn run_forms(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
) -> Result<Vec<f32>, InferenceApiError> {
    let fixture = e2e_fixture().expect("fixture builds");
    let graph = forms_graph();
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let revision = require_ready_first_native_instance(runtime, instance)
        .unwrap()
        .status()
        .mutation_version;
    let mut plan = prepare_first_native_plan_for_graph(
        runtime,
        &graph,
        instance,
        revision,
        FORMS_ROWS,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let bindings = forms_inputs()
        .into_iter()
        .map(|(name, tensor)| (TensorEdgeId::new(format!("input.{name}")), tensor))
        .collect();
    let (_, outputs, _, _) = execute_first_native_graph_materializing(
        runtime,
        &fixture,
        instance,
        &KvCacheId::new("forms").unwrap(),
        &graph,
        &mut plan,
        bindings,
        None,
        Some(0),
        &mut Vec::new(),
        MaterializedEdges::Only(&["logits"]),
        None,
        None,
    )?;
    Ok(outputs[&TensorEdgeId::new("logits")].data.clone())
}

/// Every optional operator form reaches the Reference CPU with its
/// attributes and operands, and the graph's logits match the host.
#[test]
fn every_layered_operator_form_reaches_the_kernel() {
    let fixture = e2e_fixture().expect("fixture builds");
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let logits = run_forms(&mut runtime, &instance).unwrap();
    assert_logits_close(&logits, &forms_oracle(), "layered forms");
}

/// The Reference CPU declares every optional form the dispatch gate can
/// require, so a layered graph is never refused there.
#[test]
fn the_reference_cpu_declares_every_layered_form() {
    use crate::operator::*;
    let fixture = e2e_fixture().expect("fixture builds");
    let runtime = build_runtime_trusting_fixture(&fixture);
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    for (operator, feature) in [
        ("attention", ATTENTION_SCALE_FEATURE),
        ("embedding", EMBEDDING_SCALE_FEATURE),
        ("rmsnorm", RMSNORM_PER_HEAD_FEATURE),
        ("rmsnorm", RMSNORM_WITHOUT_WEIGHT_FEATURE),
        ("rope", ROPE_PARTIAL_ROTARY_FEATURE),
        ("moe-experts", MOE_EXPERTS_ACTIVATION_FEATURE),
        ("moe-experts", MOE_PER_EXPERT_SCALE_FEATURE),
        // `express-latent-attention-forms`.
        ("attention", ATTENTION_VALUE_HEAD_DIMENSION_FEATURE),
        ("concat", CONCAT_HEADS_FEATURE),
        ("rope", ROPE_YARN_FREQUENCY_SCALING_FEATURE),
    ] {
        assert!(
            provider_kernels_declare(&runtime, &provider, operator, feature),
            "{operator} declares {feature}"
        );
    }
}

// `express-latent-attention-forms`: one multi-head latent attention layer
// in its expanded form, through the graph executor.

const MLA_HIDDEN: u64 = 6;
const MLA_HEADS: u64 = 2;
const MLA_NOPE: u64 = 3;
const MLA_ROPE: u64 = 4;
const MLA_VALUE: u64 = 4;
const MLA_LATENT: u64 = 5;
const MLA_VOCAB: u64 = 7;
const MLA_SCALE: f64 = 0.45;

fn mla_yarn() -> crate::operator::RopeYarnFrequencyScaling {
    crate::operator::RopeYarnFrequencyScaling {
        factor: 4.0,
        original_context_length: 16,
        beta_fast: 2.0,
        beta_slow: 1.0,
        attention_factor: 1.2,
        truncate: true,
    }
}

fn mla_weights() -> BTreeMap<String, HostTensor> {
    let mut seed = 29u32;
    let mut tensor = |shape: Vec<u64>| {
        let count: u64 = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 16) % 1000) as f32 / 1000.0 - 0.5
            })
            .collect();
        HostTensor::new(shape, values).unwrap()
    };
    let mut weights = BTreeMap::new();
    weights.insert(
        "q_nope".into(),
        tensor(vec![MLA_HIDDEN, MLA_HEADS * MLA_NOPE]),
    );
    weights.insert(
        "q_rope".into(),
        tensor(vec![MLA_HIDDEN, MLA_HEADS * MLA_ROPE]),
    );
    weights.insert("kv_a".into(), tensor(vec![MLA_HIDDEN, MLA_LATENT]));
    weights.insert("kv_a_norm".into(), tensor(vec![MLA_LATENT]));
    weights.insert("k_rope".into(), tensor(vec![MLA_HIDDEN, MLA_ROPE]));
    weights.insert(
        "k_nope".into(),
        tensor(vec![MLA_LATENT, MLA_HEADS * MLA_NOPE]),
    );
    weights.insert("v".into(), tensor(vec![MLA_LATENT, MLA_HEADS * MLA_VALUE]));
    weights.insert("o".into(), tensor(vec![MLA_HEADS * MLA_VALUE, MLA_HIDDEN]));
    weights.insert("out".into(), tensor(vec![MLA_HIDDEN, MLA_VOCAB]));
    weights
}

/// `q = [x Wqn | rope(x Wqr)]` and `k = [c Wkn | rope(x Wkr) for every
/// head]` with `c = rmsnorm(x Wkva)`, `v = c Wv`, attention with Q/K heads
/// of `MLA_NOPE + MLA_ROPE` and V heads of `MLA_VALUE`, then `o`, the
/// residual and the head. K and V are the layer's KV edges.
fn mla_graph(rows: u64, decode: bool) -> ExecutionGraph {
    let phase = if decode {
        ExecutionGraphPhase::Decode
    } else {
        ExecutionGraphPhase::Prefill
    };
    let mut graph =
        ExecutionGraph::new(ExecutionGraphId::new(format!("mla-{rows}-{decode}")), phase);
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
                kv: Option<&str>| {
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
        if let Some(role) = kv {
            edge.kv_cache = Some(GraphKvCacheMetadata {
                cache_id: format!("mla.layer0.{role}"),
                behavior: if decode {
                    GraphKvCacheBehavior::Append
                } else {
                    GraphKvCacheBehavior::Output
                },
                paged: false,
                compatibility_key: "mla".into(),
            });
        }
        graph.nodes.insert(node_id, execution_node);
        graph.edges.insert(output.clone(), edge);
        output
    };
    let weights = mla_weights();
    let weight = |graph: &mut ExecutionGraph, name: &str| {
        let shape = weights[name].shape.clone();
        input(graph, name, shape)
    };
    let matmul =
        |graph: &mut ExecutionGraph, id: &str, a: TensorEdgeId, b: TensorEdgeId, cols: u64, kv| {
            node(
                graph,
                id,
                "matmul",
                OperatorFamily::LinearAlgebra,
                vec![],
                vec![a, b],
                vec![rows, cols],
                kv,
            )
        };
    let rope = |graph: &mut ExecutionGraph, id: &str, a: TensorEdgeId, heads: u64| {
        let mut attributes = BTreeMap::new();
        mla_yarn().insert_attributes(&mut attributes);
        attributes.insert("base".into(), OperatorAttributeValue::Float(100.0));
        attributes.insert(
            "dimension".into(),
            OperatorAttributeValue::Integer(MLA_ROPE as i64),
        );
        attributes.insert(
            "head_count".into(),
            OperatorAttributeValue::Integer(heads as i64),
        );
        attributes.insert("position_offset".into(), OperatorAttributeValue::Integer(0));
        let attributes: Vec<(&str, OperatorAttributeValue)> = attributes
            .into_iter()
            .map(|(name, value)| (&*Box::leak(name.into_boxed_str()), value))
            .collect();
        node(
            graph,
            id,
            "rope",
            OperatorFamily::PositionEncoding,
            attributes,
            vec![a],
            vec![rows, heads * MLA_ROPE],
            None,
        )
    };
    let concat = |graph: &mut ExecutionGraph,
                  id: &str,
                  a: TensorEdgeId,
                  b: TensorEdgeId,
                  second: u64,
                  kv| {
        node(
            graph,
            id,
            "concat",
            OperatorFamily::Tensor,
            vec![
                ("axis", OperatorAttributeValue::String("heads".into())),
                (
                    "head_count",
                    OperatorAttributeValue::Integer(MLA_HEADS as i64),
                ),
                (
                    "second_head_count",
                    OperatorAttributeValue::Integer(second as i64),
                ),
            ],
            vec![a, b],
            vec![rows, MLA_HEADS * (MLA_NOPE + MLA_ROPE)],
            kv,
        )
    };
    let x = input(&mut graph, "x", vec![rows, MLA_HIDDEN]);
    let w = weight(&mut graph, "q_nope");
    let q_nope = matmul(
        &mut graph,
        "q_nope",
        x.clone(),
        w,
        MLA_HEADS * MLA_NOPE,
        None,
    );
    let w = weight(&mut graph, "q_rope");
    let q_pe = matmul(&mut graph, "q_pe", x.clone(), w, MLA_HEADS * MLA_ROPE, None);
    let q_pe = rope(&mut graph, "rope_q", q_pe, MLA_HEADS);
    let w = weight(&mut graph, "kv_a");
    let latent = matmul(&mut graph, "kv_a", x.clone(), w, MLA_LATENT, None);
    let norm = weight(&mut graph, "kv_a_norm");
    let latent = node(
        &mut graph,
        "kv_a_norm",
        "rmsnorm",
        OperatorFamily::Normalization,
        vec![("epsilon", OperatorAttributeValue::Float(1e-6))],
        vec![latent, norm],
        vec![rows, MLA_LATENT],
        None,
    );
    let w = weight(&mut graph, "k_rope");
    let k_pe = matmul(&mut graph, "k_pe", x.clone(), w, MLA_ROPE, None);
    let k_pe = rope(&mut graph, "rope_k", k_pe, 1);
    let w = weight(&mut graph, "k_nope");
    let k_nope = matmul(
        &mut graph,
        "k_nope",
        latent.clone(),
        w,
        MLA_HEADS * MLA_NOPE,
        None,
    );
    let w = weight(&mut graph, "v");
    let v = matmul(&mut graph, "v", latent, w, MLA_HEADS * MLA_VALUE, Some("v"));
    let q = concat(&mut graph, "q", q_nope, q_pe, MLA_HEADS, None);
    let k = concat(&mut graph, "k", k_nope, k_pe, 1, Some("k"));
    let attended = node(
        &mut graph,
        "attention",
        "attention",
        OperatorFamily::Attention,
        vec![
            ("causal", OperatorAttributeValue::Boolean(true)),
            (
                "head_count",
                OperatorAttributeValue::Integer(MLA_HEADS as i64),
            ),
            (
                "kv_head_count",
                OperatorAttributeValue::Integer(MLA_HEADS as i64),
            ),
            (
                "head_dimension",
                OperatorAttributeValue::Integer((MLA_NOPE + MLA_ROPE) as i64),
            ),
            (
                "value_head_dimension",
                OperatorAttributeValue::Integer(MLA_VALUE as i64),
            ),
            ("scale", OperatorAttributeValue::Float(MLA_SCALE)),
        ],
        vec![q, k, v],
        vec![rows, MLA_HEADS * MLA_VALUE],
        None,
    );
    let w = weight(&mut graph, "o");
    let o = matmul(&mut graph, "o", attended, w, MLA_HIDDEN, None);
    let hidden = node(
        &mut graph,
        "residual",
        "add",
        OperatorFamily::Tensor,
        vec![],
        vec![x, o],
        vec![rows, MLA_HIDDEN],
        None,
    );
    let w = weight(&mut graph, "out");
    let logits = matmul(&mut graph, "lm_head", hidden, w, MLA_VOCAB, None);
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

/// Row `index` of the latent-attention test's input sequence.
fn mla_input_row(index: u64) -> Vec<f32> {
    (0..MLA_HIDDEN)
        .map(|column| ((index * 3 + column * 5) % 7) as f32 / 7.0 - 0.4)
        .collect()
}

/// [`mla_graph`]'s logits for `rows` rows of [`mla_input_row`], on the
/// host: interleaved YaRN rotation by hand from the shared frequency rule,
/// the head-wise joins and the attention by direct loops.
fn mla_oracle(rows: u64) -> Vec<f32> {
    let weights = mla_weights();
    let n = rows as usize;
    let x: Vec<f32> = (0..rows).flat_map(mla_input_row).collect();
    let yarn = mla_yarn();
    let range = yarn.correction_range(MLA_ROPE, 100.0);
    let rotate = |values: &mut [f32], heads: usize| {
        let d = MLA_ROPE as usize;
        for row in 0..n {
            for head in 0..heads {
                for pair in 0..d / 2 {
                    let frequency = yarn.scale_frequency(
                        100f32.powf(-2.0 * pair as f32 / d as f32),
                        pair,
                        range,
                    );
                    let angle = row as f32 * frequency;
                    let at = row * heads * d + head * d + 2 * pair;
                    let (a, b) = (values[at], values[at + 1]);
                    let af = yarn.attention_factor as f32;
                    values[at] = (a * angle.cos() - b * angle.sin()) * af;
                    values[at + 1] = (a * angle.sin() + b * angle.cos()) * af;
                }
            }
        }
    };
    let heads = MLA_HEADS as usize;
    let (nope, rope, dv) = (MLA_NOPE as usize, MLA_ROPE as usize, MLA_VALUE as usize);
    let q_nope = host_matmul(&x, n, &weights["q_nope"]);
    let mut q_pe = host_matmul(&x, n, &weights["q_rope"]);
    rotate(&mut q_pe, heads);
    let mut latent = host_matmul(&x, n, &weights["kv_a"]);
    for row in latent.chunks_mut(MLA_LATENT as usize) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inverse = 1.0 / (mean + 1e-6).sqrt();
        for (index, value) in row.iter_mut().enumerate() {
            *value *= inverse * weights["kv_a_norm"].data[index];
        }
    }
    let mut k_pe = host_matmul(&x, n, &weights["k_rope"]);
    rotate(&mut k_pe, 1);
    let k_nope = host_matmul(&latent, n, &weights["k_nope"]);
    let v = host_matmul(&latent, n, &weights["v"]);
    let qk = nope + rope;
    let mut attended = vec![0.0f32; n * heads * dv];
    for i in 0..n {
        for h in 0..heads {
            let query: Vec<f32> = q_nope[i * heads * nope + h * nope..][..nope]
                .iter()
                .chain(&q_pe[i * heads * rope + h * rope..][..rope])
                .copied()
                .collect();
            let scores: Vec<f32> = (0..=i)
                .map(|j| {
                    let key = k_nope[j * heads * nope + h * nope..][..nope]
                        .iter()
                        .chain(&k_pe[j * rope..][..rope]);
                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * MLA_SCALE as f32
                })
                .collect();
            assert_eq!(query.len(), qk);
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            for (j, weight) in exps.iter().enumerate() {
                for e in 0..dv {
                    attended[i * heads * dv + h * dv + e] +=
                        weight / total * v[j * heads * dv + h * dv + e];
                }
            }
        }
    }
    let o = host_matmul(&attended, n, &weights["o"]);
    let hidden: Vec<f32> = x.iter().zip(o).map(|(a, b)| a + b).collect();
    host_matmul(&hidden, n, &weights["out"])
}

/// A prefill of 4 rows, then one decode step over its K/V history, with the
/// history concatenated or in place: the logits match the host, and the K
/// and V stores keep their own widths.
#[test]
fn an_expanded_latent_attention_layer_prefills_and_decodes() {
    prefill_and_decode_an_expanded_latent_attention_layer(false);
}

/// The same layer on a Provider whose values stay resident, as a device
/// Provider's do (`ReplayingExecutor` answers every read with `Opaque`):
/// the head-wise `concat` takes its inputs where they are, instead of
/// asking for host bytes the Provider declines (found running DeepSeek on
/// CUDA).
#[test]
fn an_expanded_latent_attention_layer_runs_on_device_resident_values() {
    prefill_and_decode_an_expanded_latent_attention_layer(true);
}

fn prefill_and_decode_an_expanded_latent_attention_layer(device_resident: bool) {
    let prompt = 4u64;
    let full = mla_oracle(prompt + 1);
    let vocab = MLA_VOCAB as usize;
    for in_place in [false, true] {
        let fixture = e2e_fixture().expect("fixture builds");
        let mut runtime = if device_resident {
            build_runtime_with_model_execution_engine_and_provider(
                &fixture,
                Arc::new(ReplayingProvider {
                    inner: ReferenceCpuProvider::new(),
                    executor: Arc::new(ReplayingExecutor::new()),
                }),
            )
            .expect("the Runtime builds")
        } else {
            build_runtime_trusting_fixture(&fixture)
        };
        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).unwrap();
        let mut fixture = fixture;
        fixture.config.architecture.layer_count = 1;
        let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
        let revision = require_ready_first_native_instance(&runtime, &instance)
            .unwrap()
            .status()
            .mutation_version;
        let weights = mla_weights();
        let bindings = |rows: std::ops::Range<u64>| {
            let count = rows.end - rows.start;
            let mut bindings: BTreeMap<TensorEdgeId, HostTensor> = weights
                .iter()
                .map(|(name, tensor)| (TensorEdgeId::new(format!("input.{name}")), tensor.clone()))
                .collect();
            bindings.insert(
                TensorEdgeId::new("input.x"),
                HostTensor::new(
                    [count, MLA_HIDDEN],
                    rows.flat_map(mla_input_row).collect::<Vec<f32>>(),
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
        let (_, prefill, layer_kv, _) = run(&mla_graph(prompt, false), 0..prompt, "mp", None);
        let (_, decode, _, _) = run(
            &mla_graph(1, true),
            prompt..prompt + 1,
            "md",
            Some(&layer_kv),
        );
        let what = |phase: &str| {
            format!("{phase} (in place: {in_place}, device resident: {device_resident})")
        };
        assert_logits_close(
            &prefill[&TensorEdgeId::new("logits")].data,
            &full[..prompt as usize * vocab],
            &what("prefill"),
        );
        assert_logits_close(
            &decode[&TensorEdgeId::new("logits")].data,
            &full[prompt as usize * vocab..],
            &what("decode"),
        );
        let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
        let state = &layer_kv[&0];
        assert_eq!(
            executor.resident_shape(&state.k).unwrap()[1],
            MLA_HEADS * (MLA_NOPE + MLA_ROPE)
        );
        assert_eq!(
            executor.resident_shape(&state.v).unwrap()[1],
            MLA_HEADS * MLA_VALUE
        );
    }
}
