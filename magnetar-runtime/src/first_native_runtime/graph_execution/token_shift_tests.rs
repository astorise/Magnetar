//! Token shifts against the pinned upstream Olala attention implementation.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Chunk {
    start: usize,
    end: usize,
    key: Vec<f32>,
    value: Vec<f32>,
    last_key: Vec<f32>,
    last_value: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    tokens: usize,
    heads: usize,
    head_dim: usize,
    document_start: Vec<f32>,
    raw_key: Vec<f32>,
    raw_value: Vec<f32>,
    key_logits: Vec<f32>,
    value_logits: Vec<f32>,
    full_key: Vec<f32>,
    full_value: Vec<f32>,
    chunks: Vec<Chunk>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../fixtures/olala-token-shift-reference.json"
    ))
    .unwrap()
}

fn inputs(
    reference: &Reference,
    start: usize,
    end: usize,
    key: bool,
    previous: Option<&HostTensor>,
) -> Vec<HostTensor> {
    let (raw, logits) = if key {
        (&reference.raw_key, &reference.key_logits)
    } else {
        (&reference.raw_value, &reference.value_logits)
    };
    let h = reference.heads;
    let d = reference.head_dim;
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
            vec![end - start, h, d],
            raw[start * h * d..end * h * d].to_vec(),
        ),
        tensor(vec![end - start, h], logits[start * h..end * h].to_vec()),
        tensor(
            vec![end - start],
            reference.document_start[start..end].to_vec(),
        ),
        previous
            .cloned()
            .unwrap_or_else(|| tensor(vec![h, d], vec![0.0; h * d])),
    ]
}

fn make_graph(tensors: &[HostTensor]) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("token-shift"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("shift"),
        OperatorId::magnetar("token-shift", 1, OperatorFamily::Tensor),
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
    for (name, index) in [("output.mixed", 0), ("output.raw", 3)] {
        let id = TensorEdgeId::new(name);
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensors[index].shape),
        ));
        node = node.with_output(id);
    }
    graph = graph.with_node(node);
    graph.validate(&initial_operator_catalog()).unwrap();
    (graph, bindings)
}

fn run_graph(tensors: Vec<HostTensor>) -> [HostTensor; 2] {
    let tokens = tensors[0].shape[0];
    let (graph, bindings) = make_graph(&tensors);
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
        &KvCacheId::new("shift").unwrap(),
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
    ["output.mixed", "output.raw"].map(|name| outputs[&TensorEdgeId::new(name)].clone())
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 + 1e-6 * b.abs(),
            "[{index}] {a} != {b}"
        );
    }
}

#[test]
fn token_shift_matches_upstream_for_chunks_decodes_and_document_resets() {
    let reference = reference();
    for key in [true, false] {
        let full = run_graph(inputs(&reference, 0, reference.tokens, key, None));
        close(
            &full[0].data,
            if key {
                &reference.full_key
            } else {
                &reference.full_value
            },
        );
        let raw = if key {
            &reference.raw_key
        } else {
            &reference.raw_value
        };
        close(
            &full[1].data,
            &raw[raw.len() - reference.heads * reference.head_dim..],
        );
        let mut previous = None;
        for chunk in &reference.chunks {
            let output = run_graph(inputs(
                &reference,
                chunk.start,
                chunk.end,
                key,
                previous.as_ref(),
            ));
            close(&output[0].data, if key { &chunk.key } else { &chunk.value });
            close(
                &output[1].data,
                if key {
                    &chunk.last_key
                } else {
                    &chunk.last_value
                },
            );
            previous = Some(output[1].clone());
        }
        // A document may also begin exactly at a chunk boundary, with a
        // nonzero state left by the previous document.
        let before_boundary = run_graph(inputs(&reference, 0, 3, key, None));
        let restarted = run_graph(inputs(&reference, 3, 5, key, Some(&before_boundary[1])));
        let full_output = if key {
            &reference.full_key
        } else {
            &reference.full_value
        };
        let row = reference.heads * reference.head_dim;
        close(&restarted[0].data, &full_output[3 * row..5 * row]);
    }
}

#[test]
fn token_shift_rejects_invalid_flags_and_rolls_back_outputs() {
    let reference = reference();
    let mut tensors = inputs(&reference, 0, 2, true, None);
    tensors[2].data[1] = 0.5;
    let state = tensors[3].clone();
    assert!(crate::operator_semantics::token_shift(&tensors).is_err());
    assert_eq!(tensors[3], state);
    let (graph, bindings) = make_graph(&tensors);
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
        2,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let result = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("shift-failure").unwrap(),
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
    for name in ["output.mixed", "output.raw"] {
        let id = graph_edge_resource_id(name);
        assert!(runtime.memory().tensor_residency(&id).is_none());
        assert!(executor.read_tensor(&id).is_none());
    }
    assert_eq!(
        executor
            .read_tensor(&TensorResourceId::new("input.3"))
            .unwrap(),
        state
    );
}
