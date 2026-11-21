//! The portable Mamba-3 graph against pinned upstream prefill and step output.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Geometry {
    heads: usize,
    groups: usize,
    rank: usize,
    head_dim: usize,
    state_dim: usize,
    angles: usize,
}
#[derive(Deserialize)]
struct Parameters {
    query_bias: Vec<f32>,
    key_bias: Vec<f32>,
    input_projection: Vec<f32>,
    gate_projection: Vec<f32>,
    output_projection: Vec<f32>,
    skip: Vec<f32>,
}
#[derive(Deserialize)]
struct Token {
    query: Vec<f32>,
    key: Vec<f32>,
    input: Vec<f32>,
    gate: Vec<f32>,
    decay: Vec<f32>,
    dt: Vec<f32>,
    trap: Vec<f32>,
    angle_projection: Vec<f32>,
}
#[derive(Deserialize)]
struct Step {
    output: Vec<f32>,
    angle: Vec<f32>,
    ssm: Vec<f32>,
    key: Vec<f32>,
    input: Vec<f32>,
}
#[derive(Deserialize)]
struct Reference {
    geometry: Geometry,
    parameters: Parameters,
    tokens: Vec<Token>,
    steps: Vec<Step>,
    prefill_output: Vec<f32>,
    prefill_angle: Vec<f32>,
    prefill_ssm: Vec<f32>,
    prefill_key: Vec<f32>,
    prefill_input: Vec<f32>,
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 3e-6 + 1e-6 * b.abs(),
            "{name}[{index}]: {a} != {b}"
        );
    }
}

fn attributes(c: &Geometry) -> BTreeMap<String, OperatorAttributeValue> {
    [
        ("heads", c.heads),
        ("groups", c.groups),
        ("rank", c.rank),
        ("head_dim", c.head_dim),
        ("state_dim", c.state_dim),
        ("angles", c.angles),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), OperatorAttributeValue::Integer(value as i64)))
    .collect()
}

fn inputs(
    reference: &Reference,
    start: usize,
    count: usize,
    state: Option<&[HostTensor; 4]>,
) -> Vec<HostTensor> {
    let c = &reference.geometry;
    let p = &reference.parameters;
    let tensor = |shape: Vec<usize>, data: Vec<f32>| {
        HostTensor::new(
            shape.into_iter().map(|n| n as u64).collect::<Vec<_>>(),
            data,
        )
        .unwrap()
    };
    let gather = |select: fn(&Token) -> &Vec<f32>| {
        reference.tokens[start..start + count]
            .iter()
            .flat_map(|token| select(token).iter().copied())
            .collect::<Vec<_>>()
    };
    let mut result = vec![
        tensor(
            vec![count, c.rank, c.groups, c.state_dim],
            gather(|t| &t.query),
        ),
        tensor(
            vec![count, c.rank, c.groups, c.state_dim],
            gather(|t| &t.key),
        ),
        tensor(vec![count, c.heads, c.head_dim], gather(|t| &t.input)),
        tensor(vec![count, c.heads, c.head_dim], gather(|t| &t.gate)),
        tensor(vec![count, c.heads], gather(|t| &t.decay)),
        tensor(vec![count, c.heads], gather(|t| &t.dt)),
        tensor(vec![count, c.heads], gather(|t| &t.trap)),
        tensor(
            vec![count, c.heads, c.angles],
            gather(|t| &t.angle_projection),
        ),
    ];
    let initial = state.cloned().unwrap_or_else(|| {
        [
            tensor(vec![c.heads, c.angles], vec![0.0; c.heads * c.angles]),
            tensor(
                vec![c.heads, c.head_dim, c.state_dim],
                vec![0.0; c.heads * c.head_dim * c.state_dim],
            ),
            tensor(
                vec![c.rank, c.heads, c.state_dim],
                vec![0.0; c.rank * c.heads * c.state_dim],
            ),
            tensor(vec![c.heads, c.head_dim], vec![0.0; c.heads * c.head_dim]),
        ]
    });
    result.extend(initial);
    result.extend([
        tensor(vec![c.rank, c.heads, c.state_dim], p.query_bias.clone()),
        tensor(vec![c.rank, c.heads, c.state_dim], p.key_bias.clone()),
        tensor(
            vec![c.rank, c.heads, c.head_dim],
            p.input_projection.clone(),
        ),
        tensor(vec![c.rank, c.heads, c.head_dim], p.gate_projection.clone()),
        tensor(
            vec![c.rank, c.heads, c.head_dim],
            p.output_projection.clone(),
        ),
        tensor(vec![c.heads], p.skip.clone()),
    ]);
    result
}

fn make_graph(
    reference: &Reference,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("mamba3"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("mimo"),
        OperatorId::magnetar("mamba3-mimo", 1, OperatorFamily::LinearAlgebra),
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
    for (name, input_index) in [
        "output.activation",
        "output.angle",
        "output.ssm",
        "output.key",
        "output.input",
    ]
    .into_iter()
    .zip([2, 8, 9, 10, 11])
    {
        let id = TensorEdgeId::new(name);
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensors[input_index].shape),
        ));
        node = node.with_output(id);
    }
    for (name, value) in attributes(&reference.geometry) {
        node = node.with_attribute(name, value);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run_graph(reference: &Reference, tensors: Vec<HostTensor>) -> [HostTensor; 5] {
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
        &KvCacheId::new("mamba3").unwrap(),
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
        "output.activation",
        "output.angle",
        "output.ssm",
        "output.key",
        "output.input",
    ]
    .map(|name| outputs[&TensorEdgeId::new(name)].clone())
}

#[test]
fn mamba3_graph_matches_pinned_prefill_chunked_decode_and_reset() {
    let cases: Vec<Reference> = serde_json::from_str(include_str!(
        "../../../fixtures/olala-mamba3-reference.json"
    ))
    .unwrap();
    for reference in cases {
        let full = run_graph(&reference, inputs(&reference, 0, 5, None));
        close(&full[0].data, &reference.prefill_output, "prefill output");
        let wrapped: Vec<_> = full[1]
            .data
            .iter()
            .map(|v| v.rem_euclid(std::f32::consts::TAU))
            .collect();
        close(&wrapped, &reference.prefill_angle, "prefill angle");
        close(&full[2].data, &reference.prefill_ssm, "prefill SSM");
        close(&full[3].data, &reference.prefill_key, "prefill key");
        close(&full[4].data, &reference.prefill_input, "prefill input");
        let prefix = run_graph(&reference, inputs(&reference, 0, 2, None));
        let mut state = [
            prefix[1].clone(),
            prefix[2].clone(),
            prefix[3].clone(),
            prefix[4].clone(),
        ];
        for (index, expected) in reference.steps.iter().enumerate().skip(2) {
            let result = run_graph(&reference, inputs(&reference, index, 1, Some(&state)));
            close(&result[0].data, &expected.output, "decode output");
            close(&result[1].data, &expected.angle, "decode angle");
            close(&result[2].data, &expected.ssm, "decode SSM");
            close(&result[3].data, &expected.key, "decode key");
            close(&result[4].data, &expected.input, "decode input");
            state = [
                result[1].clone(),
                result[2].clone(),
                result[3].clone(),
                result[4].clone(),
            ];
        }
        let reset = run_graph(&reference, inputs(&reference, 0, 1, None));
        close(&reset[0].data, &reference.steps[0].output, "reset output");
    }
}

#[test]
fn mamba3_rejects_invalid_profile_and_state_without_mutation() {
    let reference: Vec<Reference> = serde_json::from_str(include_str!(
        "../../../fixtures/olala-mamba3-reference.json"
    ))
    .unwrap();
    let reference = &reference[0];
    let mut tensors = inputs(reference, 0, 1, None);
    let original = tensors.clone();
    let mut attrs = attributes(&reference.geometry);
    let config = crate::operator_semantics::mamba3_mimo_from_attributes(&attrs).unwrap();
    tensors[6].data[0] = 1.1;
    assert!(crate::operator_semantics::mamba3_mimo(&tensors, config).is_err());
    assert_eq!(&tensors[8..12], &original[8..12]);
    assert!(crate::operator_semantics::mamba3_mimo(&original[..17], config).is_err());
    attrs.insert("groups".into(), OperatorAttributeValue::Integer(3));
    assert!(crate::operator_semantics::mamba3_mimo_from_attributes(&attrs).is_err());
}

#[test]
fn mamba3_graph_failure_rolls_back_all_five_outputs() {
    let cases: Vec<Reference> = serde_json::from_str(include_str!(
        "../../../fixtures/olala-mamba3-reference.json"
    ))
    .unwrap();
    let reference = &cases[0];
    let mut tensors = inputs(reference, 0, 1, None);
    tensors[6].data[0] = 1.1;
    let (graph, bindings) = make_graph(reference, &tensors);
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
        1,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let result = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("mamba3-failure").unwrap(),
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
        "output.activation",
        "output.angle",
        "output.ssm",
        "output.key",
        "output.input",
    ] {
        let id = graph_edge_resource_id(name);
        assert!(
            runtime.memory().tensor_residency(&id).is_none(),
            "leaked admission for {name}"
        );
        assert!(
            executor.read_tensor(&id).is_none(),
            "leaked provider output for {name}"
        );
    }
    for (index, tensor) in tensors.iter().enumerate().take(12).skip(8) {
        assert_eq!(
            executor
                .read_tensor(&TensorResourceId::new(format!("input.{index}")))
                .unwrap(),
            *tensor
        );
    }
}

#[test]
fn mamba3_graph_rejects_a_live_state_resource_collision() {
    let cases: Vec<Reference> = serde_json::from_str(include_str!(
        "../../../fixtures/olala-mamba3-reference.json"
    ))
    .unwrap();
    let reference = &cases[0];
    let tensors = inputs(reference, 0, 1, None);
    let (mut graph, mut bindings) = make_graph(reference, &tensors);
    let previous = TensorEdgeId::new("input.9");
    let colliding = TensorEdgeId::new("edge.output.ssm");
    let mut edge = graph.edges.remove(&previous).unwrap();
    edge.id = colliding.clone();
    graph.edges.insert(colliding.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("mimo"))
        .unwrap()
        .inputs[9] = colliding.clone();
    let state = bindings.remove(&previous).unwrap();
    bindings.insert(colliding.clone(), state);
    graph.validate(&initial_operator_catalog()).unwrap();
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
        1,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let result = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("mamba3-alias").unwrap(),
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
    assert!(
        matches!(&result, Err(InferenceApiError::GraphPlanningFailed { reason })
        if reason.contains("aliases a live tensor resource"))
    );
    assert_eq!(
        executor
            .read_tensor(&TensorResourceId::new(colliding.as_str()))
            .unwrap(),
        tensors[9]
    );
    for name in [
        "output.activation",
        "output.angle",
        "output.key",
        "output.input",
    ] {
        let id = graph_edge_resource_id(name);
        assert!(runtime.memory().tensor_residency(&id).is_none());
        assert!(executor.read_tensor(&id).is_none());
    }
}
