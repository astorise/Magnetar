//! Differential attention against the actual pinned Olala eager path.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    query_tokens: usize,
    key_tokens: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    lambda_logits: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    query_heads: usize,
    kv_heads: usize,
    signal_heads: usize,
    head_dim: usize,
    window_left: i64,
    softcap: f64,
    scale: f64,
    full: Case,
    steps: Vec<Case>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../fixtures/olala-differential-attention-reference.json"
    ))
    .unwrap()
}

fn attributes(reference: &Reference) -> BTreeMap<String, OperatorAttributeValue> {
    BTreeMap::from([
        (
            "scale".into(),
            OperatorAttributeValue::Float(reference.scale),
        ),
        (
            "softcap".into(),
            OperatorAttributeValue::Float(reference.softcap),
        ),
        (
            "window_left".into(),
            OperatorAttributeValue::Integer(reference.window_left),
        ),
        (
            "round_bfloat16".into(),
            OperatorAttributeValue::Boolean(true),
        ),
    ])
}

fn inputs(reference: &Reference, case: &Case) -> Vec<HostTensor> {
    let tensor = |shape: Vec<usize>, data: Vec<f32>| {
        HostTensor::new(
            shape
                .into_iter()
                .map(|value| value as u64)
                .collect::<Vec<_>>(),
            data,
        )
        .unwrap()
    };
    vec![
        tensor(
            vec![case.query_tokens, reference.query_heads, reference.head_dim],
            case.query.clone(),
        ),
        tensor(
            vec![case.key_tokens, reference.kv_heads, reference.head_dim],
            case.key.clone(),
        ),
        tensor(
            vec![case.key_tokens, reference.kv_heads, reference.head_dim],
            case.value.clone(),
        ),
        tensor(
            vec![case.query_tokens, reference.kv_heads],
            case.lambda_logits.clone(),
        ),
    ]
}

fn make_graph(
    reference: &Reference,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("differential"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("differential"),
        OperatorId::magnetar("differential-attention", 1, OperatorFamily::Attention),
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
    let shape = [
        tensors[0].shape[0],
        reference.signal_heads as u64,
        reference.head_dim as u64,
    ];
    let output = TensorEdgeId::new("output.attention");
    graph = graph.with_edge(TensorEdge::new(
        output.clone(),
        f32_tensor_descriptor_from_shape(&shape),
    ));
    node = node.with_output(output);
    for (name, value) in attributes(reference) {
        node = node.with_attribute(name, value);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run_graph(reference: &Reference, tensors: Vec<HostTensor>) -> HostTensor {
    let tokens = tensors[0].shape[0];
    let (graph, bindings) = make_graph(reference, &tensors);
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
        tokens,
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let (_, outputs, _, _) = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("differential").unwrap(),
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
    outputs[&TensorEdgeId::new("output.attention")].clone()
}

#[test]
fn differential_attention_matches_pinned_olala_eager_core() {
    let reference = reference();
    for case in std::iter::once(&reference.full).chain(&reference.steps) {
        let output = run_graph(&reference, inputs(&reference, case));
        assert_eq!(
            output.shape,
            vec![
                case.query_tokens as u64,
                reference.signal_heads as u64,
                reference.head_dim as u64
            ]
        );
        for (index, (&actual, &expected)) in output.data.iter().zip(&case.output).enumerate() {
            assert!(
                (actual - expected).abs() <= 3e-6 + 1e-6 * expected.abs(),
                "[{index}] {actual} != {expected}"
            );
        }
    }
}

#[test]
fn differential_attention_rejects_invalid_profile_and_history() {
    let reference = reference();
    let mut attrs = attributes(&reference);
    let config = crate::operator_semantics::differential_attention_from_attributes(&attrs).unwrap();
    let tensors = inputs(&reference, &reference.full);
    assert!(crate::operator_semantics::differential_attention(&tensors[..3], config).is_err());
    let mut bad = tensors.clone();
    bad[2] = HostTensor::new([1, 1, 1], vec![0.0]).unwrap();
    assert!(crate::operator_semantics::differential_attention(&bad, config).is_err());
    attrs.insert("window_left".into(), OperatorAttributeValue::Integer(-2));
    assert!(crate::operator_semantics::differential_attention_from_attributes(&attrs).is_err());
    attrs.insert("window_left".into(), OperatorAttributeValue::Integer(0));
    attrs.insert("softcap".into(), OperatorAttributeValue::Float(-1e-300));
    assert!(crate::operator_semantics::differential_attention_from_attributes(&attrs).is_err());
}

#[test]
fn differential_attention_failure_releases_output() {
    let reference = reference();
    let mut tensors = inputs(&reference, &reference.full);
    tensors[2].data[0] = f32::NAN;
    let (graph, bindings) = make_graph(&reference, &tensors);
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
        tensors[0].shape[0],
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let result = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("diff-failure").unwrap(),
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
    let id = graph_edge_resource_id("output.attention");
    assert!(runtime.memory().tensor_residency(&id).is_none());
    assert!(executor.read_tensor(&id).is_none());
}
