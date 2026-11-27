//! End-to-end graph comparison with the pinned upstream NVIDIA mixer.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_dim: usize,
    kernel: usize,
    eps: f64,
    dt_min: f64,
    dt_max: f64,
}

#[derive(Deserialize)]
struct Weights {
    in_proj: Vec<f32>,
    in_bias: Option<Vec<f32>>,
    conv: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
    out_proj: Vec<f32>,
    out_bias: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Step {
    output: Vec<f32>,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    config: Config,
    weights: Weights,
    inputs: Vec<f32>,
    full_output: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_conv_state: Vec<f32>,
    prefix_ssm_state: Vec<f32>,
    steps: Vec<Step>,
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..cols)
        .flat_map(|column| (0..rows).map(move |row| data[row * cols + column]))
        .collect()
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-6, "{label}[{index}]: {a} != {b}");
    }
}

fn inputs(
    reference: &Reference,
    start: usize,
    tokens: usize,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
) -> Vec<HostTensor> {
    let c = &reference.config;
    let w = &reference.weights;
    let inner = c.heads * c.head_dim;
    let conv = inner + 2 * c.groups * c.state_dim;
    let projection = inner + conv + c.heads;
    let tensor = |shape: Vec<u64>, data: Vec<f32>| HostTensor::new(shape, data).unwrap();
    let mut result = vec![
        tensor(
            vec![tokens as u64, c.hidden_size as u64],
            reference.inputs[start * c.hidden_size..(start + tokens) * c.hidden_size].to_vec(),
        ),
        tensor(vec![conv as u64, c.kernel as u64], conv_state),
        tensor(
            vec![c.heads as u64, c.head_dim as u64, c.state_dim as u64],
            ssm_state,
        ),
        tensor(
            vec![c.hidden_size as u64, projection as u64],
            transpose(&w.in_proj, projection, c.hidden_size),
        ),
        tensor(vec![conv as u64, c.kernel as u64], w.conv.clone()),
        tensor(vec![c.heads as u64], w.dt_bias.clone()),
        tensor(vec![c.heads as u64], w.a_log.clone()),
        tensor(vec![c.heads as u64], w.d.clone()),
        tensor(vec![inner as u64], w.norm.clone()),
        tensor(
            vec![inner as u64, c.hidden_size as u64],
            transpose(&w.out_proj, c.hidden_size, inner),
        ),
    ];
    if let Some(bias) = &w.conv_bias {
        result.push(tensor(vec![conv as u64], bias.clone()));
    }
    if let Some(bias) = &w.in_bias {
        result.push(tensor(vec![projection as u64], bias.clone()));
        result.push(tensor(
            vec![c.hidden_size as u64],
            w.out_bias.clone().unwrap(),
        ));
    }
    result
}

fn attributes(reference: &Reference) -> BTreeMap<String, OperatorAttributeValue> {
    let c = &reference.config;
    let mut result = BTreeMap::new();
    for (name, value) in [
        ("heads", c.heads),
        ("head_dim", c.head_dim),
        ("groups", c.groups),
        ("state_dim", c.state_dim),
        ("conv_kernel", c.kernel),
    ] {
        result.insert(name.into(), OperatorAttributeValue::Integer(value as i64));
    }
    for (name, value) in [
        ("epsilon", c.eps),
        ("dt_min", c.dt_min),
        ("dt_max", c.dt_max),
    ] {
        result.insert(name.into(), OperatorAttributeValue::Float(value));
    }
    result.insert(
        "conv_bias".into(),
        OperatorAttributeValue::Boolean(reference.weights.conv_bias.is_some()),
    );
    result.insert(
        "linear_bias".into(),
        OperatorAttributeValue::Boolean(reference.weights.in_bias.is_some()),
    );
    result
}

fn make_graph(
    reference: &Reference,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("mamba2"),
        ExecutionGraphPhase::Prefill,
    );
    let mut bindings = BTreeMap::new();
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("mixer"),
        OperatorId::magnetar("mamba2-mixer", 1, OperatorFamily::LinearAlgebra),
    );
    for (index, tensor) in tensors.iter().enumerate() {
        let id = TensorEdgeId::new(format!("input.{index}"));
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensor.shape),
        ));
        node = node.with_input(id.clone());
        bindings.insert(id, tensor.clone());
    }
    let names = ["output.activation", "output.conv", "output.ssm"];
    for (name, tensor) in names.iter().zip(&tensors[..3]) {
        let id = TensorEdgeId::new(*name);
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensor.shape),
        ));
        node = node.with_output(id);
    }
    for (name, value) in attributes(reference) {
        node = node.with_attribute(name, value);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run_graph(reference: &Reference, tensors: Vec<HostTensor>) -> [HostTensor; 3] {
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
        &KvCacheId::new("mamba2").unwrap(),
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
    ["output.activation", "output.conv", "output.ssm"]
        .map(|name| outputs[&TensorEdgeId::new(name)].clone())
}

#[test]
fn mamba2_graph_matches_nvidia_prefill_and_explicit_state_decode() {
    for json in [
        include_str!("../../../fixtures/nemotron-layer-reference.json"),
        include_str!("../../../fixtures/nemotron-layer-biased-reference.json"),
    ] {
        let reference: Reference = serde_json::from_str(json).unwrap();
        let c = &reference.config;
        let inner = c.heads * c.head_dim;
        let conv = inner + 2 * c.groups * c.state_dim;
        let empty_conv = vec![0.0; conv * c.kernel];
        let empty_ssm = vec![0.0; inner * c.state_dim];
        let full = run_graph(
            &reference,
            inputs(&reference, 0, 8, empty_conv.clone(), empty_ssm.clone()),
        );
        close(&full[0].data, &reference.full_output, "full output");
        let first_chunk = run_graph(
            &reference,
            inputs(&reference, 0, 2, empty_conv.clone(), empty_ssm.clone()),
        );
        let second_chunk = run_graph(
            &reference,
            inputs(
                &reference,
                2,
                2,
                first_chunk[1].data.clone(),
                first_chunk[2].data.clone(),
            ),
        );
        let chunked_output = [
            first_chunk[0].data.as_slice(),
            second_chunk[0].data.as_slice(),
        ]
        .concat();
        close(
            &chunked_output,
            &reference.prefix_output,
            "chunked prefix output",
        );
        close(
            &second_chunk[1].data,
            &reference.prefix_conv_state,
            "chunked convolution state",
        );
        close(
            &second_chunk[2].data,
            &reference.prefix_ssm_state,
            "chunked SSM state",
        );
        let reset = run_graph(&reference, inputs(&reference, 0, 4, empty_conv, empty_ssm));
        let mut prefix = reset;
        close(&prefix[0].data, &reference.prefix_output, "prefix output");
        close(
            &prefix[1].data,
            &reference.prefix_conv_state,
            "prefix convolution state",
        );
        close(
            &prefix[2].data,
            &reference.prefix_ssm_state,
            "prefix SSM state",
        );
        for (index, step) in reference.steps.iter().enumerate() {
            prefix = run_graph(
                &reference,
                inputs(
                    &reference,
                    4 + index,
                    1,
                    prefix[1].data.clone(),
                    prefix[2].data.clone(),
                ),
            );
            close(&prefix[0].data, &step.output, "decode output");
            close(
                &prefix[1].data,
                &step.conv_state,
                "decode convolution state",
            );
            close(&prefix[2].data, &step.ssm_state, "decode SSM state");
        }
        close(&prefix[1].data, &full[1].data, "final convolution state");
        close(&prefix[2].data, &full[2].data, "final SSM state");
    }
}

#[test]
fn mamba2_runtime_state_survives_chunks_retry_and_session_reset() {
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../fixtures/nemotron-layer-reference.json"
    ))
    .unwrap();
    let c = &reference.config;
    let inner = c.heads * c.head_dim;
    let conv = inner + 2 * c.groups * c.state_dim;
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
    let cache = new_cache(&mut runtime, "mamba2-session");
    let other_cache = new_cache(&mut runtime, "mamba2-other-session");
    let step =
        |runtime: &mut Runtime, cache: &KvCacheId, start: usize, count: usize, fail: bool| {
            let mut tensors = inputs(
                &reference,
                start,
                count,
                vec![0.0; conv * c.kernel],
                vec![0.0; inner * c.state_dim],
            );
            if fail {
                tensors[0].data[0] = f32::NAN;
            }
            let (mut graph, mut bindings) = make_graph(&reference, &tensors);
            for (slot, input, output) in [
                ("conv", "input.1", "output.conv"),
                ("ssm", "input.2", "output.ssm"),
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
                        resource_id: format!("mamba2.layer0.{slot}"),
                        layer: 0,
                        slot: slot.into(),
                        behavior,
                        compatibility_key: "nemotron-mamba2-v1".into(),
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
            execute_first_native_graph_materializing(
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
            .map(|(_, outputs, _, _)| {
                ["output.activation", "output.conv", "output.ssm"]
                    .map(|name| outputs[&TensorEdgeId::new(name)].clone())
            })
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

    let first = step(&mut runtime, &cache, 0, 2, false).unwrap();
    let second = step(&mut runtime, &cache, 2, 2, false).unwrap();
    let prefix = [first[0].data.as_slice(), second[0].data.as_slice()].concat();
    close(&prefix, &reference.prefix_output, "runtime chunked prefill");
    close(
        &second[1].data,
        &reference.prefix_conv_state,
        "runtime conv state",
    );
    close(
        &second[2].data,
        &reference.prefix_ssm_state,
        "runtime SSM state",
    );
    commit(&mut runtime, &cache);
    assert_eq!(
        runtime.kv_cache(&cache).unwrap().recurrent_resources.len(),
        2
    );
    let isolated = step(&mut runtime, &other_cache, 0, 2, false).unwrap();
    close(&isolated[0].data, &first[0].data, "isolated session");

    let decoded = step(&mut runtime, &cache, 4, 1, false).unwrap();
    close(
        &decoded[0].data,
        &reference.steps[0].output,
        "runtime decode",
    );
    super::super::recurrent_state::discard_recurrent_state(&mut runtime, &executor, &cache);
    let retry = step(&mut runtime, &cache, 4, 1, false).unwrap();
    close(&retry[0].data, &reference.steps[0].output, "retry output");
    close(
        &retry[1].data,
        &reference.steps[0].conv_state,
        "retry conv state",
    );
    close(
        &retry[2].data,
        &reference.steps[0].ssm_state,
        "retry SSM state",
    );
    commit(&mut runtime, &cache);
    assert!(step(&mut runtime, &cache, 5, 1, true).is_err());
    for (index, expected) in reference.steps.iter().enumerate().skip(1) {
        let output = step(&mut runtime, &cache, 4 + index, 1, false).unwrap();
        close(&output[0].data, &expected.output, "runtime cached output");
        close(
            &output[1].data,
            &expected.conv_state,
            "runtime cached conv state",
        );
        close(
            &output[2].data,
            &expected.ssm_state,
            "runtime cached SSM state",
        );
        commit(&mut runtime, &cache);
    }
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    let reset = step(&mut runtime, &cache, 0, 4, false).unwrap();
    close(&reset[0].data, &reference.prefix_output, "reset output");
    close(
        &reset[1].data,
        &reference.prefix_conv_state,
        "reset conv state",
    );
    close(
        &reset[2].data,
        &reference.prefix_ssm_state,
        "reset SSM state",
    );
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &other_cache);
}

#[test]
fn mamba2_rejects_malformed_inputs_without_mutating_state() {
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../fixtures/nemotron-layer-reference.json"
    ))
    .unwrap();
    let c = &reference.config;
    let inner = c.heads * c.head_dim;
    let conv = inner + 2 * c.groups * c.state_dim;
    let mut tensors = inputs(
        &reference,
        0,
        1,
        vec![0.2; conv * c.kernel],
        vec![0.3; inner * c.state_dim],
    );
    let original = tensors.clone();
    let mut attrs = attributes(&reference);
    let config = crate::operator_semantics::mamba2_mixer_from_attributes(&attrs).unwrap();
    tensors[0].data[0] = f32::NAN;
    assert!(crate::operator_semantics::mamba2_mixer(&tensors, config).is_err());
    assert_eq!(tensors[1].data, original[1].data);
    assert_eq!(tensors[2].data, original[2].data);
    assert!(crate::operator_semantics::mamba2_mixer(&original[..10], config).is_err());
    attrs.remove("groups");
    assert!(crate::operator_semantics::mamba2_mixer_from_attributes(&attrs).is_err());
    attrs.insert("groups".into(), OperatorAttributeValue::Integer(3));
    assert!(crate::operator_semantics::mamba2_mixer_from_attributes(&attrs).is_err());
}

#[test]
fn mamba2_graph_failure_rolls_back_all_three_outputs() {
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../fixtures/nemotron-layer-reference.json"
    ))
    .unwrap();
    let c = &reference.config;
    let inner = c.heads * c.head_dim;
    let conv = inner + 2 * c.groups * c.state_dim;
    let mut tensors = inputs(
        &reference,
        0,
        1,
        vec![0.2; conv * c.kernel],
        vec![0.3; inner * c.state_dim],
    );
    tensors[0].data[0] = f32::NAN;
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
        &KvCacheId::new("mamba-failure").unwrap(),
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
    for name in ["output.activation", "output.conv", "output.ssm"] {
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
    assert_eq!(
        executor
            .read_tensor(&TensorResourceId::new("input.1"))
            .unwrap()
            .data,
        tensors[1].data
    );
    assert_eq!(
        executor
            .read_tensor(&TensorResourceId::new("input.2"))
            .unwrap()
            .data,
        tensors[2].data
    );
}

#[test]
fn mamba2_graph_rejects_resource_name_collision_before_output_admission() {
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../fixtures/nemotron-layer-reference.json"
    ))
    .unwrap();
    let c = &reference.config;
    let inner = c.heads * c.head_dim;
    let conv = inner + 2 * c.groups * c.state_dim;
    let tensors = inputs(
        &reference,
        0,
        1,
        vec![0.2; conv * c.kernel],
        vec![0.3; inner * c.state_dim],
    );
    let (mut graph, mut bindings) = make_graph(&reference, &tensors);
    let previous = TensorEdgeId::new("input.2");
    // An input edge's raw ID can otherwise equal the generated resource ID
    // of a different output edge, despite their graph edge IDs differing.
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
        &KvCacheId::new("mamba-alias").unwrap(),
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
        matches!(&result, Err(InferenceApiError::GraphPlanningFailed { reason }) if reason.contains("aliases a live tensor resource"))
    );
    let state_id = TensorResourceId::new(colliding.as_str());
    assert_eq!(
        executor.read_tensor(&state_id).unwrap().data,
        tensors[2].data
    );
    for name in ["output.activation", "output.conv"] {
        let id = graph_edge_resource_id(name);
        assert!(runtime.memory().tensor_residency(&id).is_none());
        assert!(executor.read_tensor(&id).is_none());
    }
}
