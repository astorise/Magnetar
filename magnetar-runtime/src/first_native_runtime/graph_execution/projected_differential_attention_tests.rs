//! Projected attention against the pinned upstream Olala layer.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Weights {
    query: Vec<f32>,
    a_key: Vec<f32>,
    b_key: Vec<f32>,
    a_value: Vec<f32>,
    b_value: Vec<f32>,
    shift_key: Vec<f32>,
    shift_value: Vec<f32>,
    lambda: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    softmax_scaler: Vec<f32>,
}
#[derive(Deserialize)]
struct Case {
    start: usize,
    end: usize,
    output: Vec<f32>,
    raw_key: Vec<f32>,
    raw_value: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}
#[derive(Deserialize)]
struct Reference {
    hidden_size: usize,
    query_heads: usize,
    kv_heads: usize,
    signal_heads: usize,
    head_dim: usize,
    rank: usize,
    epsilon: f64,
    window_left: i64,
    softcap: f64,
    scale: f64,
    hidden: Vec<f32>,
    weights: Weights,
    full: Case,
    chunks: Vec<Case>,
}
fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../fixtures/olala-projected-attention-reference.json"
    ))
    .unwrap()
}
fn attributes(c: &Reference, past: usize) -> BTreeMap<String, OperatorAttributeValue> {
    let mut attrs: BTreeMap<_, _> = [
        ("query_heads", c.query_heads),
        ("kv_heads", c.kv_heads),
        ("head_dim", c.head_dim),
        ("rank", c.rank),
        ("past_length", past),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), OperatorAttributeValue::Integer(value as i64)))
    .collect();
    attrs.extend([
        ("epsilon".into(), OperatorAttributeValue::Float(c.epsilon)),
        (
            "zero_centered_norm".into(),
            OperatorAttributeValue::Boolean(true),
        ),
        ("scale".into(), OperatorAttributeValue::Float(c.scale)),
        ("softcap".into(), OperatorAttributeValue::Float(c.softcap)),
        (
            "window_left".into(),
            OperatorAttributeValue::Integer(c.window_left),
        ),
        (
            "round_bfloat16".into(),
            OperatorAttributeValue::Boolean(true),
        ),
    ]);
    attrs
}
fn inputs(
    c: &Reference,
    start: usize,
    end: usize,
    state: Option<&[HostTensor; 4]>,
) -> Vec<HostTensor> {
    let (h, d, q, r, w) = (c.kv_heads, c.head_dim, c.query_heads, c.rank, c.hidden_size);
    let tensor = |shape: Vec<usize>, data: Vec<f32>| {
        HostTensor::new(
            shape.into_iter().map(|n| n as u64).collect::<Vec<_>>(),
            data,
        )
        .unwrap()
    };
    let initial = state.cloned().unwrap_or_else(|| {
        [
            tensor(vec![h, d], vec![0.0; h * d]),
            tensor(vec![h, d], vec![0.0; h * d]),
            tensor(vec![1, h, d], vec![0.0; h * d]),
            tensor(vec![1, h, d], vec![0.0; h * d]),
        ]
    });
    let mut values = vec![tensor(
        vec![end - start, w],
        c.hidden[start * w..end * w].to_vec(),
    )];
    values.extend(initial);
    values.push(tensor(
        vec![end - start],
        (start..end).map(|n| n as f32).collect(),
    ));
    let p = &c.weights;
    values.extend([
        tensor(vec![w, q * d], p.query.clone()),
        tensor(vec![w, h * r], p.a_key.clone()),
        tensor(vec![w, r * d], p.b_key.clone()),
        tensor(vec![w, h * r], p.a_value.clone()),
        tensor(vec![w, r * d], p.b_value.clone()),
        tensor(vec![w, h], p.shift_key.clone()),
        tensor(vec![w, h], p.shift_value.clone()),
        tensor(vec![w, h], p.lambda.clone()),
        tensor(vec![d], p.q_norm.clone()),
        tensor(vec![d], p.k_norm.clone()),
        tensor(vec![q], p.softmax_scaler.clone()),
    ]);
    values
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 4e-6 + 1e-6 * b.abs(),
            "{label}[{index}] {a} != {b}"
        );
    }
}

#[test]
fn position_range_uses_the_graph_offset_or_the_actual_step_position() {
    let input = TensorEdgeId::new("input.hidden");
    let output = TensorEdgeId::new("logits");
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("positions"),
        ExecutionGraphPhase::Prefill,
    )
    .with_edge(TensorEdge::new(
        input.clone(),
        f32_tensor_descriptor_from_shape(&[2, 3]),
    ))
    .with_edge(TensorEdge::new(
        output.clone(),
        f32_tensor_descriptor_from_shape(&[2]),
    ))
    .with_node(
        ExecutionNode::new(
            ExecutionNodeId::new("positions"),
            OperatorId::magnetar("position-range", 1, OperatorFamily::Tensor),
        )
        .with_input(input.clone())
        .with_output(output.clone())
        .with_attribute("position_offset", OperatorAttributeValue::Integer(4)),
    );
    graph.validate(&initial_operator_catalog()).unwrap();
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &graph,
        &instance,
        revision,
        2,
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let bindings = BTreeMap::from([(input, HostTensor::new([2, 3], vec![0.0; 6]).unwrap())]);
    let mut run = |offset| {
        execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &KvCacheId::new("positions").unwrap(),
            &graph,
            &mut plan,
            bindings.clone(),
            None,
            offset,
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            None,
        )
    };
    assert_eq!(run(None).unwrap().1[&output].data, [4.0, 5.0]);
    assert_eq!(run(Some(9)).unwrap().1[&output].data, [9.0, 10.0]);
    assert!(run(Some(16_777_215)).is_err());
}

fn make_graph(
    c: &Reference,
    case: &Case,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("projected-attention"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("projected-attention"),
        OperatorId::magnetar(
            "projected-differential-attention",
            1,
            OperatorFamily::Attention,
        ),
    );
    let mut bindings = BTreeMap::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let id = TensorEdgeId::new(format!("input.{index}"));
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensor.shape),
        ));
        node = node.with_input(id.clone());
        bindings.insert(id, tensor.clone());
    }
    let output = [
        (case.end - case.start) as u64,
        c.signal_heads as u64,
        c.head_dim as u64,
    ];
    let history = [case.end as u64, c.kv_heads as u64, c.head_dim as u64];
    let shapes: [&[u64]; 5] = [
        &output,
        &tensors[1].shape,
        &tensors[2].shape,
        &history,
        &history,
    ];
    for (name, shape) in [
        "output.attention",
        "output.raw-key",
        "output.raw-value",
        "output.keys",
        "output.values",
    ]
    .into_iter()
    .zip(shapes)
    {
        let id = TensorEdgeId::new(name);
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(shape),
        ));
        node = node.with_output(id);
    }
    for (name, value) in attributes(c, case.start) {
        node = node.with_attribute(name, value);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run(c: &Reference, case: &Case, state: Option<&[HostTensor; 4]>) -> [HostTensor; 5] {
    let tensors = inputs(c, case.start, case.end, state);
    let (graph, bindings) = make_graph(c, case, &tensors);
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &graph,
        &instance,
        revision,
        (case.end - case.start) as u64,
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let (_, outputs, _, _) = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("projected-attention").unwrap(),
        &graph,
        &mut plan,
        bindings,
        None,
        Some(0),
        &mut Vec::new(),
        MaterializedEdges::All,
        None,
        None,
    )
    .unwrap();
    [
        "output.attention",
        "output.raw-key",
        "output.raw-value",
        "output.keys",
        "output.values",
    ]
    .map(|name| outputs[&TensorEdgeId::new(name)].clone())
}
fn compare(output: &[HostTensor; 5], case: &Case, c: &Reference) {
    assert_eq!(
        output[0].shape,
        vec![
            (case.end - case.start) as u64,
            c.signal_heads as u64,
            c.head_dim as u64
        ]
    );
    for (tensor, expected, label) in output
        .iter()
        .zip([
            (&case.output, "output"),
            (&case.raw_key, "raw key"),
            (&case.raw_value, "raw value"),
            (&case.keys, "keys"),
            (&case.values, "values"),
        ])
        .map(|(tensor, (values, label))| (tensor, values, label))
    {
        close(&tensor.data, expected, label);
    }
}

#[test]
fn projected_differential_attention_matches_pinned_upstream_layer() {
    let c = reference();
    let full = run(&c, &c.full, None);
    compare(&full, &c.full, &c);
    let mut state = None;
    for case in &c.chunks {
        let output = run(&c, case, state.as_ref());
        compare(&output, case, &c);
        state = Some([
            output[1].clone(),
            output[2].clone(),
            output[3].clone(),
            output[4].clone(),
        ]);
    }
    // Reset at a document boundary, even when the preceding raw state is nonzero.
    let previous = state.unwrap();
    let sentinel = HostTensor::new(
        [1, c.kv_heads as u64, c.head_dim as u64],
        vec![0.0; c.kv_heads * c.head_dim],
    )
    .unwrap();
    let reset_state = [
        previous[0].clone(),
        previous[1].clone(),
        sentinel.clone(),
        sentinel,
    ];
    let reset = run(&c, &c.full, Some(&reset_state));
    compare(&reset, &c.full, &c);
}

/// Olala's raw previous K/V are fixed-size recurrent state, distinct from
/// the growing attention K/V history. Exercise both through the actual graph
/// executor and the Runtime's pending/committed state lifecycle.
#[test]
fn projected_attention_keeps_raw_state_separate_from_kv_history() {
    let c = reference();
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let new_cache = |runtime: &mut Runtime, name: &str| {
        runtime
            .create_kv_cache(KvCache::new(
                KvCacheId::new(name).unwrap(),
                KvCacheScope::Session,
                KvCacheCompatibility::new(
                    GenerationModelReference::ModelInstance(instance.clone()),
                    TokenizerId::new("tokenizer").unwrap(),
                ),
                KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float32),
            ))
            .unwrap()
    };
    let cache = new_cache(&mut runtime, "projected-attention-session");
    let other_cache = new_cache(&mut runtime, "projected-attention-other-session");
    let step = |runtime: &mut Runtime,
                cache: &KvCacheId,
                case: &Case,
                previous: Option<&[HostTensor; 4]>| {
        let tensors = inputs(&c, case.start, case.end, previous);
        let (mut graph, mut bindings) = make_graph(&c, case, &tensors);
        for (slot, input, output) in [
            ("raw_key", "input.1", "output.raw-key"),
            ("raw_value", "input.2", "output.raw-value"),
        ] {
            for (edge, behavior) in [
                (input, GraphRecurrentStateBehavior::Input),
                (output, GraphRecurrentStateBehavior::Output),
            ] {
                graph
                    .edges
                    .get_mut(&TensorEdgeId::new(edge))
                    .unwrap()
                    .recurrent_state = Some(GraphRecurrentStateMetadata {
                    resource_id: format!("projected.layer0.{slot}"),
                    layer: 0,
                    slot: slot.into(),
                    behavior,
                    compatibility_key: "olala-projected-v1".into(),
                });
            }
            // Only the raw state is Runtime-owned. The two variable-length
            // attention-history inputs remain distinct caller bindings.
            bindings.remove(&TensorEdgeId::new(input));
        }
        graph.validate(&initial_operator_catalog()).unwrap();
        let mut plan = prepare_first_native_plan_for_graph(
            runtime,
            &graph,
            &instance,
            revision,
            (case.end - case.start) as u64,
            PreparedExecutionPlanGeneration::new(1),
            &provider,
        )
        .unwrap();
        let (_, outputs, _, _) = execute_first_native_graph_materializing(
            runtime,
            &fixture,
            &instance,
            cache,
            &graph,
            &mut plan,
            bindings,
            None,
            Some(0),
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            None,
        )
        .unwrap();
        [
            "output.attention",
            "output.raw-key",
            "output.raw-value",
            "output.keys",
            "output.values",
        ]
        .map(|name| outputs[&TensorEdgeId::new(name)].clone())
    };
    let history = |output: &[HostTensor; 5]| {
        [
            output[1].clone(),
            output[2].clone(),
            output[3].clone(),
            output[4].clone(),
        ]
    };
    let commit = |runtime: &mut Runtime, cache: &KvCacheId| {
        KvUpdateTransaction {
            provider_binding: provider.clone(),
            executor: executor.clone(),
            promoted: Vec::new(),
        }
        .commit(runtime, cache)
        .unwrap();
    };

    // Prefill two chunks in one attempt, then commit the fixed raw state.
    let first = step(&mut runtime, &cache, &c.chunks[0], None);
    compare(&first, &c.chunks[0], &c);
    let second = step(&mut runtime, &cache, &c.chunks[1], Some(&history(&first)));
    compare(&second, &c.chunks[1], &c);
    commit(&mut runtime, &cache);
    assert_eq!(
        runtime.kv_cache(&cache).unwrap().recurrent_resources.len(),
        2
    );

    // A separate session starts from zero even while this cache has state.
    let isolated = step(&mut runtime, &other_cache, &c.chunks[0], None);
    compare(&isolated, &c.chunks[0], &c);

    // A one-token decode uses the committed raw state and the prior KV
    // history. A failed attempt must leave both reusable for the retry.
    let decoded = step(&mut runtime, &cache, &c.chunks[2], Some(&history(&second)));
    compare(&decoded, &c.chunks[2], &c);
    super::super::recurrent_state::discard_recurrent_state(&mut runtime, &executor, &cache);
    let retry = step(&mut runtime, &cache, &c.chunks[2], Some(&history(&second)));
    compare(&retry, &c.chunks[2], &c);
    commit(&mut runtime, &cache);
    let last = step(&mut runtime, &cache, &c.chunks[3], Some(&history(&retry)));
    compare(&last, &c.chunks[3], &c);

    // Reset the document: neither raw state nor previous KV may carry over.
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    let reset = step(&mut runtime, &cache, &c.full, None);
    compare(&reset, &c.full, &c);
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);

    // Continue token by token after another reset. Compare each token to the
    // upstream full-sequence result and the final raw and KV states to its
    // independently computed final states.
    let mut previous = None;
    let output_width = c.signal_heads * c.head_dim;
    for index in 0..c.full.end {
        let token = Case {
            start: index,
            end: index + 1,
            output: Vec::new(),
            raw_key: Vec::new(),
            raw_value: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
        };
        let output = step(&mut runtime, &cache, &token, previous.as_ref());
        close(
            &output[0].data,
            &c.full.output[index * output_width..(index + 1) * output_width],
            "cached token output",
        );
        commit(&mut runtime, &cache);
        previous = Some(history(&output));
    }
    let final_state = previous.unwrap();
    for (tensor, expected, label) in final_state
        .iter()
        .zip([
            (&c.full.raw_key, "final raw key"),
            (&c.full.raw_value, "final raw value"),
            (&c.full.keys, "final keys"),
            (&c.full.values, "final values"),
        ])
        .map(|(tensor, (expected, label))| (tensor, expected, label))
    {
        close(&tensor.data, expected, label);
    }
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &other_cache);
}

/// Both the fixed raw-token state and the growing K/V history belong to the
/// Runtime. The hand-built graph is experimental and generates positions
/// from the invocation's actual offset instead of a caller-supplied tensor.
#[test]
fn projected_attention_uses_runtime_managed_history_tensors() {
    let c = reference();
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let new_cache = |runtime: &mut Runtime, name: &str| {
        runtime
            .create_kv_cache(KvCache::new(
                KvCacheId::new(name).unwrap(),
                KvCacheScope::Session,
                KvCacheCompatibility::new(
                    GenerationModelReference::ModelInstance(instance.clone()),
                    TokenizerId::new("tokenizer").unwrap(),
                ),
                KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float32),
            ))
            .unwrap()
    };
    let cache = new_cache(&mut runtime, "projected-managed-session");
    let other = new_cache(&mut runtime, "projected-managed-other");
    let step = |runtime: &mut Runtime,
                cache: &KvCacheId,
                case: &Case,
                history: Option<&LayerKvMap>,
                supply_history: bool| {
        let mut tensors = inputs(&c, case.start, case.end, None);
        for tensor in &mut tensors[3..5] {
            *tensor = HostTensor::new(
                [
                    case.start.max(1) as u64,
                    c.kv_heads as u64,
                    c.head_dim as u64,
                ],
                vec![0.0; case.start.max(1) * c.kv_heads * c.head_dim],
            )
            .unwrap();
        }
        let (mut graph, mut bindings) = make_graph(&c, case, &tensors);
        bindings.remove(&TensorEdgeId::new("input.5"));
        graph = graph.with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("positions"),
                OperatorId::magnetar("position-range", 1, OperatorFamily::Tensor),
            )
            .with_input(TensorEdgeId::new("input.0"))
            .with_output(TensorEdgeId::new("input.5"))
            // The Runtime supplies the actual offset on every invocation.
            .with_attribute("position_offset", OperatorAttributeValue::Integer(0)),
        );
        for (slot, input, output) in [
            ("raw_key", "input.1", "output.raw-key"),
            ("raw_value", "input.2", "output.raw-value"),
        ] {
            for (edge, behavior) in [
                (input, GraphRecurrentStateBehavior::Input),
                (output, GraphRecurrentStateBehavior::Output),
            ] {
                graph
                    .edges
                    .get_mut(&TensorEdgeId::new(edge))
                    .unwrap()
                    .recurrent_state = Some(GraphRecurrentStateMetadata {
                    resource_id: format!("managed.layer0.{slot}"),
                    layer: 0,
                    slot: slot.into(),
                    behavior,
                    compatibility_key: "projected-raw-v1".into(),
                });
            }
            bindings.remove(&TensorEdgeId::new(input));
        }
        for (role, input, output) in [
            ("k", "input.3", "output.keys"),
            ("v", "input.4", "output.values"),
        ] {
            for (edge, behavior) in [
                (input, GraphKvCacheBehavior::HistoryInput),
                (output, GraphKvCacheBehavior::Output),
            ] {
                graph
                    .edges
                    .get_mut(&TensorEdgeId::new(edge))
                    .unwrap()
                    .kv_cache = Some(GraphKvCacheMetadata {
                    cache_id: format!("managed.layer0.{role}"),
                    behavior,
                    paged: false,
                    compatibility_key: "projected-history-v1".into(),
                });
            }
            if !supply_history {
                bindings.remove(&TensorEdgeId::new(input));
            }
        }
        graph.validate(&initial_operator_catalog()).unwrap();
        assert_eq!(graph.kv_history_bindings().unwrap().len(), 2);
        assert!(!kv_edges_feed_only_attention(&graph));
        if case.start == 0 {
            let mut mismatched = graph.clone();
            mismatched
                .edges
                .get_mut(&TensorEdgeId::new("output.keys"))
                .unwrap()
                .kv_cache
                .as_mut()
                .unwrap()
                .compatibility_key = "wrong".into();
            assert!(mismatched.validate(&initial_operator_catalog()).is_err());
            let mut unpaired = graph.clone();
            unpaired
                .edges
                .get_mut(&TensorEdgeId::new("output.keys"))
                .unwrap()
                .kv_cache = None;
            assert!(unpaired.kv_history_bindings().is_err());
        }
        let mut plan = prepare_first_native_plan_for_graph(
            runtime,
            &graph,
            &instance,
            revision,
            (case.end - case.start) as u64,
            PreparedExecutionPlanGeneration::new(1),
            &provider,
        )
        .unwrap();
        execute_first_native_graph_materializing(
            runtime,
            &fixture,
            &instance,
            cache,
            &graph,
            &mut plan,
            bindings,
            history,
            Some(case.start as u64),
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            Some(KvStoreSpec::owned(c.full.end as u64)),
        )
        .map(|(_, outputs, pending, _)| {
            let tensors = [
                "output.attention",
                "output.raw-key",
                "output.raw-value",
                "output.keys",
                "output.values",
            ]
            .map(|name| outputs[&TensorEdgeId::new(name)].clone());
            (tensors, pending)
        })
    };
    let commit = |runtime: &mut Runtime, cache: &KvCacheId, pending: &LayerKvMap| {
        let mut transaction = KvUpdateTransaction {
            provider_binding: provider.clone(),
            executor: executor.clone(),
            promoted: Vec::new(),
        };
        for (&layer, state) in pending {
            transaction
                .promote_layer(runtime, cache, layer, state)
                .unwrap();
        }
        transaction.commit(runtime, cache).unwrap()
    };

    let (first, pending) = step(&mut runtime, &cache, &c.chunks[0], None, false).unwrap();
    compare(&first, &c.chunks[0], &c);
    assert!(step(&mut runtime, &cache, &c.chunks[0], None, false).is_err());
    assert!(step(&mut runtime, &other, &c.chunks[1], Some(&pending), false).is_err());
    assert!(step(&mut runtime, &cache, &c.chunks[1], None, false).is_err());
    assert!(step(&mut runtime, &cache, &c.chunks[1], Some(&pending), true).is_err());
    let mut incompatible = pending.clone();
    incompatible
        .get_mut(&0)
        .unwrap()
        .history_schema
        .as_mut()
        .unwrap()[0]
        .1 = "wrong".into();
    assert!(
        step(
            &mut runtime,
            &cache,
            &c.chunks[1],
            Some(&incompatible),
            false
        )
        .is_err()
    );
    let (second, pending) =
        step(&mut runtime, &cache, &c.chunks[1], Some(&pending), false).unwrap();
    compare(&second, &c.chunks[1], &c);
    let mut committed = commit(&mut runtime, &cache, &pending);
    let (isolated, _) = step(&mut runtime, &other, &c.chunks[0], None, false).unwrap();
    compare(&isolated, &c.chunks[0], &c);
    runtime.release_kv_cache(&other).unwrap();
    assert!(step(&mut runtime, &other, &c.chunks[0], None, false).is_err());
    let (decoded, _) = step(&mut runtime, &cache, &c.chunks[2], Some(&committed), false).unwrap();
    compare(&decoded, &c.chunks[2], &c);
    super::super::recurrent_state::discard_recurrent_state(&mut runtime, &executor, &cache);
    let (retry, pending) =
        step(&mut runtime, &cache, &c.chunks[2], Some(&committed), false).unwrap();
    compare(&retry, &c.chunks[2], &c);
    committed = commit(&mut runtime, &cache, &pending);
    let (last, _) = step(&mut runtime, &cache, &c.chunks[3], Some(&committed), false).unwrap();
    compare(&last, &c.chunks[3], &c);
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &other);
}

#[test]
fn projected_differential_attention_rejects_bad_state_and_positions() {
    let c = reference();
    let case = &c.chunks[1];
    let config = crate::operator_semantics::projected_differential_attention_from_attributes(
        &attributes(&c, case.start),
    )
    .unwrap();
    let tensors = inputs(&c, case.start, case.end, None);
    assert!(crate::operator_semantics::projected_differential_attention(&tensors, config).is_err());
    let mut tensors = inputs(&c, 0, c.full.end, None);
    tensors[5].data[1] = 0.0;
    let config = crate::operator_semantics::projected_differential_attention_from_attributes(
        &attributes(&c, 0),
    )
    .unwrap();
    assert!(crate::operator_semantics::projected_differential_attention(&tensors, config).is_err());
    let mut attrs = attributes(&c, 0);
    attrs.insert("kv_heads".into(), OperatorAttributeValue::Integer(0));
    assert!(
        crate::operator_semantics::projected_differential_attention_from_attributes(&attrs)
            .is_err()
    );
}

#[test]
fn projected_differential_attention_failure_rolls_back_outputs() {
    let c = reference();
    let case = &c.full;
    let mut tensors = inputs(&c, 0, case.end, None);
    tensors[5].data[1] = 0.0;
    let (graph, bindings) = make_graph(&c, case, &tensors);
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &graph,
        &instance,
        revision,
        case.end as u64,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let result = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("projected-failure").unwrap(),
        &graph,
        &mut plan,
        bindings,
        None,
        Some(0),
        &mut Vec::new(),
        MaterializedEdges::All,
        None,
        None,
    );
    assert!(result.is_err());
    for name in [
        "output.attention",
        "output.raw-key",
        "output.raw-value",
        "output.keys",
        "output.values",
    ] {
        let id = graph_edge_resource_id(name);
        assert!(runtime.memory().tensor_residency(&id).is_none());
        assert!(executor.read_tensor(&id).is_none());
    }
    for (index, tensor) in tensors.iter().enumerate().take(5).skip(1) {
        assert_eq!(
            executor
                .read_tensor(&TensorResourceId::new(format!("input.{index}")))
                .unwrap(),
            *tensor
        );
    }
}
