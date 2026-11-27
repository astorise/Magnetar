//! Projected Mamba-3 mixer against the pinned upstream model class.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Geometry {
    hidden: usize,
    heads: usize,
    groups: usize,
    rank: usize,
    head_dim: usize,
    state_dim: usize,
    angles: usize,
    epsilon: f64,
    decay_floor: f64,
    zero_centered_norm: bool,
}
#[derive(Deserialize)]
struct Weights {
    in_proj: Vec<f32>,
    in_proj_dyn: Vec<f32>,
    dt_bias: Vec<f32>,
    b_norm: Vec<f32>,
    c_norm: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    x_proj: Vec<f32>,
    z_proj: Vec<f32>,
    out_proj: Vec<f32>,
    skip: Vec<f32>,
}
#[derive(Deserialize)]
struct State {
    angle: Vec<f32>,
    ssm: Vec<f32>,
    key: Vec<f32>,
    input: Vec<f32>,
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
    weights: Weights,
    inputs: Vec<f32>,
    full_output: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_state: State,
    steps: Vec<Step>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../fixtures/olala-mamba3-mixer-reference.json"
    ))
    .unwrap()
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 4e-6 + 1e-6 * b.abs(),
            "{name}[{index}]: {a} != {b}"
        );
    }
}

fn attributes(c: &Geometry) -> BTreeMap<String, OperatorAttributeValue> {
    let mut result: BTreeMap<_, _> = [
        ("heads", c.heads),
        ("groups", c.groups),
        ("rank", c.rank),
        ("head_dim", c.head_dim),
        ("state_dim", c.state_dim),
        ("angles", c.angles),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), OperatorAttributeValue::Integer(value as i64)))
    .collect();
    result.insert("epsilon".into(), OperatorAttributeValue::Float(c.epsilon));
    result.insert(
        "decay_floor".into(),
        OperatorAttributeValue::Float(c.decay_floor),
    );
    result.insert(
        "zero_centered_norm".into(),
        OperatorAttributeValue::Boolean(c.zero_centered_norm),
    );
    result
}

fn inputs(
    reference: &Reference,
    start: usize,
    count: usize,
    state: Option<&[HostTensor; 4]>,
) -> Vec<HostTensor> {
    let c = &reference.geometry;
    let w = &reference.weights;
    let tensor = |shape: Vec<usize>, values: Vec<f32>| {
        HostTensor::new(
            shape.into_iter().map(|n| n as u64).collect::<Vec<_>>(),
            values,
        )
        .unwrap()
    };
    let projection = c.heads * (2 * c.head_dim + 3);
    let dynamic = 2 * c.groups * c.rank * c.state_dim + c.angles;
    let states = state.cloned().unwrap_or_else(|| {
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
    let mut result = vec![tensor(
        vec![count, c.hidden],
        reference.inputs[start * c.hidden..(start + count) * c.hidden].to_vec(),
    )];
    result.extend(states);
    result.extend([
        tensor(vec![c.hidden, projection], w.in_proj.clone()),
        tensor(vec![c.hidden, dynamic], w.in_proj_dyn.clone()),
        tensor(vec![c.heads], w.dt_bias.clone()),
        tensor(vec![c.state_dim], w.b_norm.clone()),
        tensor(vec![c.state_dim], w.c_norm.clone()),
        tensor(vec![c.rank, c.heads, c.state_dim], w.q_bias.clone()),
        tensor(vec![c.rank, c.heads, c.state_dim], w.k_bias.clone()),
        tensor(vec![c.rank, c.heads, c.head_dim], w.x_proj.clone()),
        tensor(vec![c.rank, c.heads, c.head_dim], w.z_proj.clone()),
        tensor(vec![c.rank, c.heads, c.head_dim], w.out_proj.clone()),
        tensor(vec![c.heads], w.skip.clone()),
    ]);
    result
}

fn state(outputs: &[HostTensor; 5]) -> [HostTensor; 4] {
    [
        outputs[1].clone(),
        outputs[2].clone(),
        outputs[3].clone(),
        outputs[4].clone(),
    ]
}

fn make_graph(
    reference: &Reference,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("mamba3-mixer"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("mixer"),
        OperatorId::magnetar("mamba3-mixer", 1, OperatorFamily::LinearAlgebra),
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
    let activation = [
        tensors[0].shape[0],
        (reference.geometry.heads * reference.geometry.head_dim) as u64,
    ];
    let shapes: [&[u64]; 5] = [
        &activation,
        &tensors[1].shape,
        &tensors[2].shape,
        &tensors[3].shape,
        &tensors[4].shape,
    ];
    for (name, shape) in [
        "output.activation",
        "output.angle",
        "output.ssm",
        "output.key",
        "output.input",
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
    for (name, value) in attributes(&reference.geometry) {
        node = node.with_attribute(name, value);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run_graph(reference: &Reference, tensors: Vec<HostTensor>) -> [HostTensor; 5] {
    run_graph_with_angle_wrap(reference, tensors, false)
}

fn with_prefill_angle_wrap(mut graph: ExecutionGraph) -> ExecutionGraph {
    let final_angle = TensorEdgeId::new("output.angle");
    let unwrapped_angle = TensorEdgeId::new("mixer.unwrapped-angle");
    let descriptor = graph.edges[&final_angle].descriptor.clone();
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("mixer"))
        .unwrap()
        .outputs[1] = unwrapped_angle.clone();
    graph = graph.with_edge(TensorEdge::new(unwrapped_angle.clone(), descriptor));
    graph = graph.with_node(
        ExecutionNode::new(
            ExecutionNodeId::new("wrap-prefill-angle"),
            OperatorId::magnetar("remainder-scalar", 1, OperatorFamily::Tensor),
        )
        .with_input(unwrapped_angle)
        .with_output(final_angle)
        .with_attribute(
            "divisor",
            OperatorAttributeValue::Float(std::f64::consts::TAU),
        ),
    );
    graph
}

fn run_graph_with_angle_wrap(
    reference: &Reference,
    tensors: Vec<HostTensor>,
    wrap_angle: bool,
) -> [HostTensor; 5] {
    let tokens = tensors[0].shape[0];
    let (mut graph, bindings) = make_graph(reference, &tensors);
    if wrap_angle {
        graph = with_prefill_angle_wrap(graph);
    }
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
        tokens,
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let (_, outputs, _, _) = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("mamba3-mixer").unwrap(),
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
fn mamba3_graph_wraps_prefill_angle_before_unwrapped_cached_steps() {
    let reference = reference();
    let prefix = run_graph_with_angle_wrap(&reference, inputs(&reference, 0, 3, None), true);
    close(&prefix[0].data, &reference.prefix_output, "prefix output");
    close(
        &prefix[1].data,
        &reference.prefix_state.angle,
        "wrapped prefill angle",
    );
    close(&prefix[2].data, &reference.prefix_state.ssm, "prefix SSM");
    close(&prefix[3].data, &reference.prefix_state.key, "prefix key");
    close(
        &prefix[4].data,
        &reference.prefix_state.input,
        "prefix input",
    );
    let mut state = state(&prefix);
    for (index, expected) in reference.steps.iter().enumerate() {
        let output = run_graph(&reference, inputs(&reference, 3 + index, 1, Some(&state)));
        close(&output[0].data, &expected.output, "decode output");
        close(&output[1].data, &expected.angle, "unwrapped decode angle");
        close(&output[2].data, &expected.ssm, "decode SSM");
        close(&output[3].data, &expected.key, "decode key");
        close(&output[4].data, &expected.input, "decode input");
        state = self::state(&output);
    }
}

#[test]
fn mamba3_prefill_wrap_is_committed_before_cached_steps() {
    let reference = reference();
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let cache = runtime
        .create_kv_cache(KvCache::new(
            KvCacheId::new("mamba3-managed-session").unwrap(),
            KvCacheScope::Session,
            KvCacheCompatibility::new(
                GenerationModelReference::ModelInstance(instance.clone()),
                TokenizerId::new("tokenizer").unwrap(),
            ),
            KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float32),
        ))
        .unwrap();
    let step = |runtime: &mut Runtime, start: usize, count: usize, wrap: bool| {
        let tensors = inputs(&reference, start, count, None);
        let (mut graph, mut bindings) = make_graph(&reference, &tensors);
        if wrap {
            graph = with_prefill_angle_wrap(graph);
        }
        for (slot, input, output) in [
            ("angle", "input.1", "output.angle"),
            ("ssm", "input.2", "output.ssm"),
            ("key", "input.3", "output.key"),
            ("input", "input.4", "output.input"),
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
                    resource_id: format!("mamba3.layer0.{slot}"),
                    layer: 0,
                    slot: slot.into(),
                    behavior,
                    compatibility_key: "mamba3-mixer-v1".into(),
                });
            }
            bindings.remove(&TensorEdgeId::new(input));
        }
        graph.validate(&initial_operator_catalog()).unwrap();
        let mut plan = prepare_first_native_plan_for_graph(
            runtime,
            &graph,
            &instance,
            revision,
            count as u64,
            PreparedExecutionPlanGeneration::new(1),
            &provider,
        )
        .unwrap();
        let (_, outputs, _, _) = execute_first_native_graph_materializing(
            runtime,
            &fixture,
            &instance,
            &cache,
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
    };
    let commit = |runtime: &mut Runtime| {
        KvUpdateTransaction {
            provider_binding: provider.clone(),
            executor: executor.clone(),
            promoted: Vec::new(),
        }
        .commit(runtime, &cache)
        .unwrap();
    };
    let prefix = step(&mut runtime, 0, 3, true);
    close(
        &prefix[0].data,
        &reference.prefix_output,
        "managed prefill output",
    );
    close(
        &prefix[1].data,
        &reference.prefix_state.angle,
        "managed prefill angle",
    );
    commit(&mut runtime);
    assert_eq!(
        runtime.kv_cache(&cache).unwrap().recurrent_resources.len(),
        4
    );
    for (index, expected) in reference.steps.iter().enumerate() {
        let output = step(&mut runtime, 3 + index, 1, false);
        close(&output[0].data, &expected.output, "managed decode output");
        close(&output[1].data, &expected.angle, "managed decode angle");
        close(&output[2].data, &expected.ssm, "managed decode SSM");
        close(&output[3].data, &expected.key, "managed decode key");
        close(&output[4].data, &expected.input, "managed decode input");
        if index == 0 {
            super::super::recurrent_state::discard_recurrent_state(&mut runtime, &executor, &cache);
            let retry = step(&mut runtime, 3, 1, false);
            close(&retry[0].data, &expected.output, "managed retried decode");
        }
        commit(&mut runtime);
    }
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    let reset = step(&mut runtime, 0, 1, true);
    let width = reference.geometry.heads * reference.geometry.head_dim;
    close(
        &reset[0].data,
        &reference.full_output[..width],
        "managed reset output",
    );
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
}

#[test]
fn mamba3_mixer_matches_pinned_upstream_layer_and_cached_decode() {
    let reference = reference();
    let full = run_graph(&reference, inputs(&reference, 0, 6, None));
    close(&full[0].data, &reference.full_output, "full output");
    let prefix = run_graph(&reference, inputs(&reference, 0, 3, None));
    close(&prefix[0].data, &reference.prefix_output, "prefix output");
    let mut state = state(&prefix);
    for angle in &mut state[0].data {
        *angle = angle.rem_euclid(std::f32::consts::TAU);
    }
    close(
        &state[0].data,
        &reference.prefix_state.angle,
        "prefix angle",
    );
    close(&state[1].data, &reference.prefix_state.ssm, "prefix SSM");
    close(&state[2].data, &reference.prefix_state.key, "prefix key");
    close(
        &state[3].data,
        &reference.prefix_state.input,
        "prefix input",
    );
    for (index, expected) in reference.steps.iter().enumerate() {
        let output = run_graph(&reference, inputs(&reference, 3 + index, 1, Some(&state)));
        close(&output[0].data, &expected.output, "decode output");
        close(&output[1].data, &expected.angle, "decode angle");
        close(&output[2].data, &expected.ssm, "decode SSM");
        close(&output[3].data, &expected.key, "decode key");
        close(&output[4].data, &expected.input, "decode input");
        state = self::state(&output);
    }
    let reset = run_graph(&reference, inputs(&reference, 0, 1, None));
    let width = reference.geometry.heads * reference.geometry.head_dim;
    close(
        &reset[0].data,
        &reference.full_output[..width],
        "reset output",
    );
}

#[test]
fn mamba3_mixer_rejects_invalid_profile_and_preserves_state() {
    let reference = reference();
    let mut tensors = inputs(&reference, 0, 1, None);
    let original = tensors.clone();
    let mut attrs = attributes(&reference.geometry);
    let config = crate::operator_semantics::mamba3_mixer_from_attributes(&attrs).unwrap();
    tensors[0].data.fill(f32::MAX);
    assert!(crate::operator_semantics::mamba3_mixer(&tensors, config).is_err());
    assert_eq!(&tensors[1..5], &original[1..5]);
    assert!(crate::operator_semantics::mamba3_mixer(&original[..15], config).is_err());
    attrs.insert("groups".into(), OperatorAttributeValue::Integer(2));
    assert!(crate::operator_semantics::mamba3_mixer_from_attributes(&attrs).is_err());
    let mut malformed = config;
    malformed.geometry.rank = 0;
    assert!(crate::operator_semantics::mamba3_mixer(&original, malformed).is_err());
}

#[test]
fn mamba3_mixer_graph_failure_rolls_back_all_outputs() {
    let reference = reference();
    let mut tensors = inputs(&reference, 0, 1, None);
    tensors[0].data.fill(f32::MAX);
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
        &KvCacheId::new("mixer-failure").unwrap(),
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
    for (index, tensor) in tensors.iter().enumerate().take(5).skip(1) {
        assert_eq!(
            executor
                .read_tensor(&TensorResourceId::new(format!("input.{index}")))
                .unwrap(),
            *tensor
        );
    }
}

#[test]
fn mamba3_mixer_graph_rejects_live_state_collision() {
    let reference = reference();
    let tensors = inputs(&reference, 0, 1, None);
    let (mut graph, mut bindings) = make_graph(&reference, &tensors);
    let previous = TensorEdgeId::new("input.2");
    let colliding = TensorEdgeId::new("edge.output.ssm");
    let mut edge = graph.edges.remove(&previous).unwrap();
    edge.id = colliding.clone();
    graph.edges.insert(colliding.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("mixer"))
        .unwrap()
        .inputs[2] = colliding.clone();
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
        &KvCacheId::new("mixer-alias").unwrap(),
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
        tensors[2]
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
