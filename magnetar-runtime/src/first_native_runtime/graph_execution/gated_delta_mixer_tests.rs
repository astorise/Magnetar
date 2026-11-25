//! Qwen3.5-style portable mixer against independently run PyTorch fixtures.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    hidden_size: usize,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    epsilon: f64,
    hidden: Vec<f32>,
    mask: Vec<f32>,
    weights: BTreeMap<String, Vec<f32>>,
    output: Vec<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_conv_state: Vec<f32>,
    prefix_state: Vec<f32>,
}

fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!(
        "../../../fixtures/qwen35-mixer-reference.json"
    ))
    .unwrap()
}

fn attributes(case: &Case) -> BTreeMap<String, OperatorAttributeValue> {
    let mut attributes: BTreeMap<_, _> = [
        ("key_heads", case.key_heads),
        ("value_heads", case.value_heads),
        ("key_dim", case.key_dim),
        ("value_dim", case.value_dim),
        ("conv_kernel", case.kernel),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), OperatorAttributeValue::Integer(value as i64)))
    .collect();
    attributes.insert(
        "epsilon".into(),
        OperatorAttributeValue::Float(case.epsilon),
    );
    attributes
}

fn inputs(
    case: &Case,
    start: usize,
    count: usize,
    state: Option<&[HostTensor; 2]>,
) -> Vec<HostTensor> {
    let tensor = |shape: Vec<usize>, data: Vec<f32>| {
        HostTensor::new(
            shape.into_iter().map(|v| v as u64).collect::<Vec<_>>(),
            data,
        )
        .unwrap()
    };
    let weight = |name: &str| case.weights[name].clone();
    let hidden = case.hidden_size;
    let key_width = case.key_heads * case.key_dim;
    let value_width = case.value_heads * case.value_dim;
    let channels = 2 * key_width + value_width;
    let states = state.cloned().unwrap_or_else(|| {
        [
            tensor(
                vec![channels, case.kernel],
                vec![0.0; channels * case.kernel],
            ),
            tensor(
                vec![case.value_heads, case.key_dim, case.value_dim],
                vec![0.0; case.value_heads * case.key_dim * case.value_dim],
            ),
        ]
    });
    let mut result = vec![tensor(
        vec![count, hidden],
        case.hidden[start * hidden..(start + count) * hidden].to_vec(),
    )];
    result.extend(states);
    result.extend([
        tensor(vec![channels, hidden], weight("in_proj_qkv.weight")),
        tensor(vec![value_width, hidden], weight("in_proj_z.weight")),
        tensor(vec![case.value_heads, hidden], weight("in_proj_a.weight")),
        tensor(vec![case.value_heads, hidden], weight("in_proj_b.weight")),
        tensor(vec![channels, case.kernel], weight("conv1d.weight")),
        tensor(vec![case.value_heads], weight("A_log")),
        tensor(vec![case.value_heads], weight("dt_bias")),
        tensor(vec![case.value_dim], weight("norm.weight")),
        tensor(vec![hidden, value_width], weight("out_proj.weight")),
        tensor(vec![count], case.mask[start..start + count].to_vec()),
    ]);
    result
}

fn graph(
    case: &Case,
    tensors: &[HostTensor],
    output_gate_type: Option<&str>,
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("gated-delta-mixer"),
        ExecutionGraphPhase::Prefill,
    );
    let mut node = ExecutionNode::new(
        ExecutionNodeId::new("mixer"),
        OperatorId::magnetar("gated-delta-mixer", 1, OperatorFamily::LinearAlgebra),
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
    for (name, tensor) in [
        "output.activation",
        "output.convolution",
        "output.recurrent",
    ]
    .into_iter()
    .zip(&tensors[..3])
    {
        let id = TensorEdgeId::new(name);
        graph = graph.with_edge(TensorEdge::new(
            id.clone(),
            f32_tensor_descriptor_from_shape(&tensor.shape),
        ));
        node = node.with_output(id);
    }
    for (name, value) in attributes(case) {
        node = node.with_attribute(name, value);
    }
    if let Some(kind) = output_gate_type {
        node = node.with_attribute(
            "output_gate_type",
            OperatorAttributeValue::String(kind.into()),
        );
    }
    (graph.with_node(node), bindings)
}

fn run(case: &Case, tensors: Vec<HostTensor>) -> [HostTensor; 3] {
    run_with_gate(case, tensors, None)
}

fn run_with_gate(
    case: &Case,
    tensors: Vec<HostTensor>,
    output_gate_type: Option<&str>,
) -> [HostTensor; 3] {
    let (graph, bindings) = graph(case, &tensors, output_gate_type);
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
        tensors[0].shape[0],
        PreparedExecutionPlanGeneration::new(1),
        &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
    )
    .unwrap();
    let (_, outputs, _, _) = execute_first_native_graph_materializing(
        &mut runtime,
        &fixture,
        &instance,
        &KvCacheId::new("gated-delta-fixture").unwrap(),
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
        "output.convolution",
        "output.recurrent",
    ]
    .map(|name| outputs[&TensorEdgeId::new(name)].clone())
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 4e-6 + 1e-6 * b.abs(),
            "{name}[{index}]: {a} != {b}"
        );
    }
}

#[test]
fn gated_delta_graph_matches_pytorch_for_prefill_chunks_decode_and_reset() {
    for case in cases() {
        let full = run(&case, inputs(&case, 0, case.tokens, None));
        close(&full[0].data, &case.output, "full output");
        close(&full[1].data, &case.conv_state, "full convolution");
        close(&full[2].data, &case.state, "full recurrent");
        let prefix = run(&case, inputs(&case, 0, 2, None));
        close(&prefix[0].data, &case.prefix_output, "prefix output");
        close(
            &prefix[1].data,
            &case.prefix_conv_state,
            "prefix convolution",
        );
        close(&prefix[2].data, &case.prefix_state, "prefix recurrent");
        let state = [prefix[1].clone(), prefix[2].clone()];
        let suffix = run(&case, inputs(&case, 2, case.tokens - 2, Some(&state)));
        close(
            &suffix[0].data,
            &case.output[2 * case.hidden_size..],
            "chunk output",
        );
        close(&suffix[1].data, &case.conv_state, "chunk convolution");
        close(&suffix[2].data, &case.state, "chunk recurrent");
        let mut state = None;
        let mut output = Vec::new();
        for token in 0..case.tokens {
            let step = run(&case, inputs(&case, token, 1, state.as_ref()));
            output.extend_from_slice(&step[0].data);
            state = Some([step[1].clone(), step[2].clone()]);
        }
        close(&output, &case.output, "decode output");
        let final_state = state.unwrap();
        close(&final_state[0].data, &case.conv_state, "decode convolution");
        close(&final_state[1].data, &case.state, "decode recurrent");
        let reset = run(&case, inputs(&case, 0, 1, None));
        close(
            &reset[0].data,
            &case.output[..case.hidden_size],
            "reset output",
        );
    }
}

/// The mixer graph with its two states declared recurrent, as a Component
/// emits it: the Runtime binds and keeps them.
fn recurrent_graph(
    case: &Case,
    tensors: &[HostTensor],
) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let (mut graph, mut bindings) = graph(case, tensors, None);
    for (slot, input, output) in [
        ("conv", "input.1", "output.convolution"),
        ("delta", "input.2", "output.recurrent"),
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
                resource_id: format!("mixer.layer0.{slot}"),
                layer: 0,
                slot: slot.into(),
                behavior,
                compatibility_key: "model-v1".into(),
            });
        }
        bindings.remove(&TensorEdgeId::new(input));
    }
    (graph, bindings)
}

#[test]
fn gated_delta_state_lives_in_the_runtime_across_steps() {
    for case in cases() {
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
                KvCacheId::new("gated-delta-session").unwrap(),
                KvCacheScope::Session,
                KvCacheCompatibility::new(
                    GenerationModelReference::ModelInstance(instance.clone()),
                    TokenizerId::new("tokenizer").unwrap(),
                ),
                KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float32),
            ))
            .unwrap();
        // Runs `count` tokens from `start`, reading and staging the state.
        let step = |runtime: &mut Runtime, start: usize, count: usize| {
            let (graph, bindings) = recurrent_graph(&case, &inputs(&case, start, count, None));
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
                "output.convolution",
                "output.recurrent",
            ]
            .map(|name| outputs[&TensorEdgeId::new(name)].data.clone())
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
        let hidden = case.hidden_size;

        // A prefill in two chunks of one attempt equals the full prefill.
        let prefix = step(&mut runtime, 0, 2);
        close(&prefix[0], &case.prefix_output, "prefix output");
        let rest = step(&mut runtime, 2, case.tokens - 3);
        close(
            &rest[0],
            &case.output[2 * hidden..(case.tokens - 1) * hidden],
            "chunk output",
        );
        commit(&mut runtime);
        // Decode continues from the committed prefill state.
        let decoded = step(&mut runtime, case.tokens - 1, 1);
        close(
            &decoded[0],
            &case.output[(case.tokens - 1) * hidden..],
            "decode output",
        );
        close(&decoded[1], &case.conv_state, "decode convolution");
        close(&decoded[2], &case.state, "decode recurrent");
        // Not accepted: the next attempt decodes the same token again.
        super::super::recurrent_state::discard_recurrent_state(&mut runtime, &executor, &cache);
        let again = step(&mut runtime, case.tokens - 1, 1);
        close(&again[0], &decoded[0], "retried decode output");
        // A new sequence starts from zeros.
        super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
        let reset = step(&mut runtime, 0, 1);
        close(&reset[0], &case.output[..hidden], "reset output");
    }
}

#[test]
fn gated_delta_rejects_bad_shapes_and_mask() {
    let case = cases().remove(0);
    let tensors = inputs(&case, 0, 1, None);
    let (graph, _) = graph(&case, &tensors, None);
    graph.validate(&initial_operator_catalog()).unwrap();
    let mut invalid = graph.clone();
    invalid
        .edges
        .get_mut(&TensorEdgeId::new("input.2"))
        .unwrap()
        .descriptor
        .shape
        .dimensions[0] += 1;
    assert!(invalid.validate(&initial_operator_catalog()).is_err());
    let mut invalid = graph.clone();
    invalid
        .edges
        .get_mut(&TensorEdgeId::new("input.1"))
        .unwrap()
        .descriptor
        .compute_dtype = Some(DTypeDescriptor::portable(ComputeDType::BrainFloat16));
    assert!(invalid.validate(&initial_operator_catalog()).is_err());
    let mut invalid = graph.clone();
    invalid
        .nodes
        .get_mut(&ExecutionNodeId::new("mixer"))
        .unwrap()
        .attributes
        .insert("value_heads".into(), OperatorAttributeValue::Integer(3));
    assert!(invalid.validate(&initial_operator_catalog()).is_err());
    let mut invalid = graph.clone();
    invalid
        .edges
        .get_mut(&TensorEdgeId::new("input.0"))
        .unwrap()
        .descriptor
        .layout = LayoutDescriptor::Strided {
        strides_elements: vec![case.hidden_size as i64, 1],
        offset_elements: 0,
    };
    assert!(invalid.validate(&initial_operator_catalog()).is_err());

    let mut invalid = tensors.clone();
    invalid[12].data[0] = 2.0;
    let config =
        crate::operator_semantics::gated_delta_mixer_from_attributes(&attributes(&case)).unwrap();
    assert!(crate::operator_semantics::gated_delta_mixer(&invalid, config).is_err());
    assert_eq!(&invalid[1..3], &tensors[1..3]);
}

#[test]
fn gated_delta_graph_rejects_live_state_alias() {
    let case = cases().remove(0);
    let tensors = inputs(&case, 0, 1, None);
    let (mut graph, mut bindings) = graph(&case, &tensors, None);
    let original = TensorEdgeId::new("input.2");
    let colliding = TensorEdgeId::new("edge.output.recurrent");
    let mut edge = graph.edges.remove(&original).unwrap();
    edge.id = colliding.clone();
    graph.edges.insert(colliding.clone(), edge);
    graph
        .nodes
        .get_mut(&ExecutionNodeId::new("mixer"))
        .unwrap()
        .inputs[2] = colliding.clone();
    let state = bindings.remove(&original).unwrap();
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
        &KvCacheId::new("gated-delta-alias").unwrap(),
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
    for name in ["output.activation", "output.convolution"] {
        let id = graph_edge_resource_id(name);
        assert!(runtime.memory().tensor_residency(&id).is_none());
        assert!(executor.read_tensor(&id).is_none());
    }
}

#[test]
fn gated_delta_graph_failure_releases_all_three_outputs() {
    let case = cases().remove(0);
    let mut tensors = inputs(&case, 0, 1, None);
    tensors[12].data[0] = 2.0;
    let (graph, bindings) = graph(&case, &tensors, None);
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
        &KvCacheId::new("gated-delta-failure").unwrap(),
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
        "output.convolution",
        "output.recurrent",
    ] {
        let id = graph_edge_resource_id(name);
        assert!(
            runtime.memory().tensor_residency(&id).is_none(),
            "leaked admission for {name}"
        );
        assert!(
            executor.read_tensor(&id).is_none(),
            "leaked output for {name}"
        );
    }
}

/// Reuses the source-hash-checked real-weight reference generated by the
/// Qwen Component. Checkpoint payloads stay outside the repository.
#[test]
#[ignore = "requires verified Qwen3.5 base layer segments and PyTorch reference"]
fn gated_delta_graph_matches_real_checkpoint_mixer() {
    use sha2::{Digest, Sha256};

    #[derive(Deserialize)]
    struct RealReference {
        hidden: Vec<f32>,
        output: Vec<f32>,
        conv_state: Vec<f32>,
        state: Vec<f32>,
        prefix_output: Vec<f32>,
        prefix_conv_state: Vec<f32>,
        prefix_state: Vec<f32>,
    }
    fn checked(path: &std::path::Path, expected: &str) -> Vec<u8> {
        let data = std::fs::read(path).unwrap();
        let digest: String = Sha256::digest(&data)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest, expected, "unexpected digest for {}", path.display());
        data
    }
    let dir = std::path::PathBuf::from(
        std::env::var_os("MAGNETAR_QWEN35_CHECKPOINT_DIR")
            .expect("set MAGNETAR_QWEN35_CHECKPOINT_DIR"),
    );
    let header = checked(
        &dir.join("kev-qwen35-base-header.json"),
        "c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083",
    );
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let scalars = checked(
        &dir.join("qwen35-base-layer0-scalars.bin"),
        "274bf2ffc2f29f3381a158434c3028694c8859a2c1bafe83df567058fa2fafc7",
    );
    let block = checked(
        &dir.join("qwen35-base-layer0-block.bin"),
        "6c68593c6ecdc8ac34a85216d1db652e3fc02899fa6f3bce4a8ecfe227d39eb8",
    );
    let read = |name: &str, dtype: &str, shape: &[usize]| {
        let entry = &header[format!("model.language_model.layers.0.linear_attn.{name}")];
        assert_eq!(entry["dtype"], dtype);
        assert_eq!(entry["shape"], serde_json::json!(shape));
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let raw = if end <= scalars.len() {
            &scalars[start..end]
        } else {
            const BLOCK_START: usize = 508_571_776;
            &block[start - BLOCK_START..end - BLOCK_START]
        };
        let count = shape.iter().product::<usize>();
        match dtype {
            "BF16" => {
                assert_eq!(raw.len(), count * 2);
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))
                    .collect()
            }
            "F32" => {
                assert_eq!(raw.len(), count * 4);
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect()
            }
            _ => panic!("unexpected dtype"),
        }
    };
    let weights = BTreeMap::from([
        (
            "in_proj_qkv.weight".into(),
            read("in_proj_qkv.weight", "BF16", &[6144, 1024]),
        ),
        (
            "in_proj_z.weight".into(),
            read("in_proj_z.weight", "BF16", &[2048, 1024]),
        ),
        (
            "in_proj_a.weight".into(),
            read("in_proj_a.weight", "BF16", &[16, 1024]),
        ),
        (
            "in_proj_b.weight".into(),
            read("in_proj_b.weight", "BF16", &[16, 1024]),
        ),
        (
            "conv1d.weight".into(),
            read("conv1d.weight", "BF16", &[6144, 1, 4]),
        ),
        ("A_log".into(), read("A_log", "F32", &[16])),
        ("dt_bias".into(), read("dt_bias", "BF16", &[16])),
        ("norm.weight".into(), read("norm.weight", "F32", &[128])),
        (
            "out_proj.weight".into(),
            read("out_proj.weight", "BF16", &[1024, 2048]),
        ),
    ]);
    let reference: RealReference = serde_json::from_slice(
        &std::fs::read(dir.join("qwen35-real-mixer-reference.json")).unwrap(),
    )
    .unwrap();
    let case = Case {
        hidden_size: 1024,
        tokens: 3,
        key_heads: 16,
        value_heads: 16,
        key_dim: 128,
        value_dim: 128,
        kernel: 4,
        epsilon: 1e-6,
        hidden: reference.hidden,
        mask: vec![1.0; 3],
        weights,
        output: reference.output,
        conv_state: reference.conv_state,
        state: reference.state,
        prefix_output: reference.prefix_output,
        prefix_conv_state: reference.prefix_conv_state,
        prefix_state: reference.prefix_state,
    };
    let full = run(&case, inputs(&case, 0, 3, None));
    close(&full[0].data, &case.output, "real full output");
    close(&full[1].data, &case.conv_state, "real full convolution");
    close(&full[2].data, &case.state, "real full recurrence");
    let prefix = run(&case, inputs(&case, 0, 2, None));
    close(&prefix[0].data, &case.prefix_output, "real prefix output");
    close(
        &prefix[1].data,
        &case.prefix_conv_state,
        "real prefix convolution",
    );
    close(
        &prefix[2].data,
        &case.prefix_state,
        "real prefix recurrence",
    );
    let state = [prefix[1].clone(), prefix[2].clone()];
    let suffix = run(&case, inputs(&case, 2, 1, Some(&state)));
    close(&suffix[0].data, &case.output[2048..], "real decode output");
    close(&suffix[1].data, &case.conv_state, "real decode convolution");
    close(&suffix[2].data, &case.state, "real decode recurrence");
}

#[test]
fn gated_delta_mask_is_optional() {
    for case in cases() {
        // The fixture's first two tokens are both valid.
        let masked = inputs(&case, 0, 2, None);
        assert!(masked[12].data.iter().all(|&value| value == 1.0));
        let mut unmasked = masked.clone();
        unmasked.pop();
        let (with_mask, without_mask) = (run(&case, masked), run(&case, unmasked));
        for (a, b) in with_mask.iter().zip(&without_mask) {
            assert_eq!(a.data, b.data);
        }
    }
}

/// `output_gate_type` reaches the Reference CPU Provider through the
/// graph: `silu` is the default, `sigmoid` gives the shared semantics'
/// values, and another activation fails the node
/// (`support-qwen38-flash-next`, Qwen4-Exp's linear attention).
#[test]
fn gated_delta_graph_honours_the_output_gate_type() {
    use crate::operator_semantics::{GatedDeltaMixer, MixerOutputGate, gated_delta_mixer};
    let case = cases().into_iter().next().unwrap();
    let tensors = inputs(&case, 0, 2, None);
    let default = run(&case, tensors.clone());
    let silu = run_with_gate(&case, tensors.clone(), Some("silu"));
    for (a, b) in default.iter().zip(&silu) {
        assert_eq!(a.data, b.data, "silu is the default gate");
    }
    let sigmoid = run_with_gate(&case, tensors.clone(), Some("sigmoid"));
    let expected = gated_delta_mixer(
        &tensors,
        GatedDeltaMixer {
            key_heads: case.key_heads,
            value_heads: case.value_heads,
            key_dim: case.key_dim,
            value_dim: case.value_dim,
            conv_kernel: case.kernel,
            epsilon: case.epsilon as f32,
            output_gate: MixerOutputGate::Sigmoid,
        },
    )
    .unwrap();
    for (name, (actual, expected)) in ["activation", "convolution", "recurrent"]
        .into_iter()
        .zip(sigmoid.iter().zip(&expected))
    {
        assert_eq!(actual.data, expected.data, "sigmoid gate, {name}");
    }
    assert_ne!(
        sigmoid[0].data, silu[0].data,
        "the two gates give different activations"
    );
    assert_eq!(sigmoid[1].data, silu[1].data, "the convolution state");
    assert_eq!(sigmoid[2].data, silu[2].data, "the recurrent state");
}
