//! `support-qwen38-flash-next` lot 3: Qwen4-Exp's operators run as graph
//! nodes of the first-native Runtime, against the same Transformers fixtures
//! as their Reference semantics (`qwen4_exp_semantics_tests`).

use super::*;
use crate::qwen4_exp_semantics_tests::{Flat, assert_matches, fixture};

/// A one-node graph of `name` over `inputs`, run on the Reference CPU: the
/// node's outputs, in order. `shapes` are its declared output edges.
fn run_node(
    name: &str,
    family: OperatorFamily,
    attributes: Vec<(&str, OperatorAttributeValue)>,
    inputs: Vec<HostTensor>,
    shapes: &[Vec<u64>],
) -> Vec<HostTensor> {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new(format!("qwen4-exp-{name}")),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("node"),
        OperatorId::magnetar(name, 1, family),
    );
    let mut bindings = BTreeMap::new();
    for (index, tensor) in inputs.iter().enumerate() {
        let id = TensorEdgeId::new(format!("input.{index}"));
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensor.shape),
        ));
        node = node.with_input(id.clone());
        bindings.insert(id, tensor.clone());
    }
    let outputs: Vec<TensorEdgeId> = (0..shapes.len())
        .map(|index| TensorEdgeId::new(format!("output.{index}")))
        .collect();
    for (id, shape) in outputs.iter().zip(shapes) {
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(shape),
        ));
        node = node.with_output(id.clone());
    }
    for (attribute, value) in attributes {
        node = node.with_attribute(attribute, value);
    }
    let graph = graph.with_node(node);
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
        inputs[0].shape[0],
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let (_, produced, _, _) = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("qwen4-exp-fixture").unwrap(),
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
    outputs.iter().map(|id| produced[id].clone()).collect()
}

fn int(value: usize) -> OperatorAttributeValue {
    OperatorAttributeValue::Integer(value as i64)
}

fn float(value: f32) -> OperatorAttributeValue {
    OperatorAttributeValue::Float(f64::from(value))
}

#[test]
fn the_gated_residual_operators_run_as_graph_nodes() {
    let fixture = fixture();
    let case = &fixture.gated_residual;
    let (tokens, width) = (case.input.shape[0], case.input.shape[1]);
    let hc = fixture.hc_count as u64;
    let hidden = width / hc;
    let attributes = || {
        vec![
            ("hc_count", int(fixture.hc_count)),
            ("epsilon", float(fixture.epsilon)),
            ("weight_offset", float(1.0)),
        ]
    };
    let mut inputs = vec![
        case.input.tensor(),
        case.norm_weight.tensor(),
        case.down.tensor(),
        case.up.tensor(),
    ];
    let [final_mixed] = run_node(
        "gated-residual-final",
        OperatorFamily::LinearAlgebra,
        attributes(),
        inputs.clone(),
        &[vec![tokens, hidden]],
    )
    .try_into()
    .unwrap();
    assert_matches(&final_mixed, &case.final_mixed, "final mixer");

    inputs.push(case.inject.tensor());
    let [mixed, write]: [HostTensor; 2] = run_node(
        "gated-residual",
        OperatorFamily::LinearAlgebra,
        attributes(),
        inputs,
        &[vec![tokens, hidden], vec![tokens, hc]],
    )
    .try_into()
    .unwrap();
    assert_matches(&mixed, &case.mixed, "mixed input");
    assert_matches(&write, &case.injection, "write weights");

    let [after] = run_node(
        "gated-residual-inject",
        OperatorFamily::Tensor,
        Vec::new(),
        vec![case.input.tensor(), case.block_output.tensor(), write],
        &[vec![tokens, width]],
    )
    .try_into()
    .unwrap();
    assert_matches(&after, &case.after_block, "residual after the block");

    let [repeated] = run_node(
        "repeat-features",
        OperatorFamily::Tensor,
        vec![("count", int(fixture.hc_count))],
        vec![case.embedding.tensor()],
        &[vec![tokens, width]],
    )
    .try_into()
    .unwrap();
    assert_matches(&repeated, &case.repeated, "repeated embedding");
}

#[test]
fn the_qsa_selection_runs_as_a_graph_node() {
    let fixture = fixture();
    let case = &fixture.qsa_indexer;
    let tokens = case.keys.shape[0] as usize;
    let slots = case.budget + case.compress_ratio - 1;
    let [selection] = run_node(
        "qsa-select",
        OperatorFamily::Tensor,
        vec![
            ("head_count", int(case.head_count)),
            ("head_dim", int(case.head_dim)),
            ("compress_ratio", int(case.compress_ratio)),
            ("budget", int(case.budget)),
            ("rotary_dim", int(case.rotary_dim)),
            ("epsilon", float(fixture.epsilon)),
            ("weight_offset", float(1.0)),
            ("rope_base", float(case.rope_base)),
        ],
        vec![
            case.queries.tensor(),
            case.keys.tensor(),
            case.key_norm_weight.tensor(),
        ],
        &[vec![tokens as u64, slots as u64]],
    )
    .try_into()
    .unwrap();
    for query in 0..tokens {
        let kept: Vec<usize> = selection.data[query * slots..(query + 1) * slots]
            .iter()
            .filter(|&&index| index >= 0.0)
            .map(|&index| index as usize)
            .collect();
        let expected: Vec<usize> = (0..tokens)
            .filter(|&token| case.selected.data[query * tokens + token] != 0.0)
            .collect();
        assert_eq!(kept, expected, "query {query}");
    }
}

#[test]
fn the_ngram_ple_runs_as_graph_nodes_with_its_states() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let tokens = case.token_ids.len();
    let (width, embed) = (case.hidden.shape[1], case.embeddings.shape[1]);
    let heads = (case.ngram_size - 1) * case.heads_per_ngram;
    let state_len = (case.conv_kernel - 1) * case.conv_dilation;
    let context = case.ngram_size - 1;
    let hash_attributes = || {
        vec![
            ("ngram_size", int(case.ngram_size)),
            ("heads_per_ngram", int(case.heads_per_ngram)),
            ("vocab_size", int(case.vocab_size as usize)),
            ("vocab_base", int(case.vocab_base as usize)),
            ("layer_index", int(case.layer_index as usize)),
            ("seed", int(case.seed as usize)),
            ("eos_token_id", int(case.eos_token_id as usize)),
        ]
    };
    let ple_attributes = || {
        vec![
            ("hc_count", int(fixture.hc_count)),
            ("epsilon", float(fixture.epsilon)),
            ("weight_offset", float(1.0)),
            ("conv_kernel", int(case.conv_kernel)),
            ("conv_dilation", int(case.conv_dilation)),
        ]
    };
    let rows = |tensor: &Flat, from: usize, to: usize| {
        let width = tensor.shape[1] as usize;
        HostTensor::new(
            [(to - from) as u64, width as u64],
            tensor.data[from * width..to * width].to_vec(),
        )
        .unwrap()
    };
    let mut carried_context =
        HostTensor::new([context as u64], vec![case.eos_token_id as f32; context]).unwrap();
    let mut carried_history = HostTensor::new(
        [width, state_len as u64],
        vec![0.0; width as usize * state_len],
    )
    .unwrap();
    let mut produced = Vec::new();
    for (from, to) in [(0, 5), (5, tokens)] {
        let count = (to - from) as u64;
        let ids = HostTensor::new([count], case.token_ids[from..to].to_vec()).unwrap();
        let [embeddings, next_context]: [HostTensor; 2] = run_node(
            "ngram-embedding",
            OperatorFamily::Tensor,
            hash_attributes(),
            vec![ids, carried_context, case.table.tensor()],
            &[
                vec![count, (heads as u64) * case.table.shape[1]],
                vec![context as u64],
            ],
        )
        .try_into()
        .unwrap();
        assert_eq!(embeddings.shape, [count, embed]);
        let [output, next_history]: [HostTensor; 2] = run_node(
            "ngram-ple",
            OperatorFamily::LinearAlgebra,
            ple_attributes(),
            vec![
                rows(&case.hidden, from, to),
                embeddings,
                case.key_proj.tensor(),
                case.value_proj.tensor(),
                case.norm_key.tensor(),
                case.norm_query.tensor(),
                case.norm_conv.tensor(),
                case.conv.tensor(),
                carried_history,
            ],
            &[vec![count, width], vec![width, state_len as u64]],
        )
        .try_into()
        .unwrap();
        produced.extend(output.data);
        carried_context = next_context;
        carried_history = next_history;
    }
    let produced = HostTensor::new(case.output.shape.clone(), produced).unwrap();
    assert_matches(&produced, &case.output, "PLE output across two graph runs");
}
