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
