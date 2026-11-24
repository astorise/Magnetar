//! Numerical graph tests against PyTorch 2.15 layer_norm and exact GELU.

use super::*;

#[test]
fn encoder_affine_gelu_slice_graph_matches_pytorch() {
    // Expected values from torch.nn.functional.layer_norm(..., eps=1e-5),
    // gelu(..., approximate="none") and [:, 1:3], with and without bias.
    let cases = [
        (
            false,
            vec![
                -0.048_225_973,
                -0.033_129_454,
                -0.118_874_565,
                -0.130_225_02,
            ],
        ),
        (
            true,
            vec![-0.085_945_67, 0.463_285_98, -0.140_943_65, -0.161_602_23],
        ),
    ];
    for (with_bias, expected) in cases {
        let input = TensorEdgeId::new("input.values");
        let scale = TensorEdgeId::new("input.scale");
        let bias = TensorEdgeId::new("input.bias");
        let normalized = TensorEdgeId::new("normalized");
        let activated = TensorEdgeId::new("activated");
        let logits = TensorEdgeId::new("logits");
        let mut graph = ExecutionGraph::new(
            ExecutionGraphId::new("encoder-primitives"),
            ExecutionGraphPhase::Prefill,
        )
        .with_edge(TensorEdge::new(
            input.clone(),
            f32_tensor_descriptor_from_shape(&[2, 4]),
        ))
        .with_edge(TensorEdge::new(
            scale.clone(),
            f32_tensor_descriptor_from_shape(&[4]),
        ))
        .with_edge(TensorEdge::new(
            normalized.clone(),
            f32_tensor_descriptor_from_shape(&[2, 4]),
        ))
        .with_edge(TensorEdge::new(
            activated.clone(),
            f32_tensor_descriptor_from_shape(&[2, 4]),
        ))
        .with_edge(TensorEdge::new(
            logits.clone(),
            f32_tensor_descriptor_from_shape(&[2, 2]),
        ));
        let mut norm = ExecutionNode::new(
            ExecutionNodeId::new("norm"),
            OperatorId::magnetar("affine-layernorm", 1, OperatorFamily::Normalization),
        )
        .with_input(input.clone())
        .with_input(scale.clone())
        .with_output(normalized.clone())
        .with_attribute("epsilon", OperatorAttributeValue::Float(1e-5));
        let mut bindings = BTreeMap::from([
            (
                input,
                HostTensor::new([2, 4], vec![-2.0, 0.0, 0.5, 3.0, 1.0, -1.0, 2.0, -2.0]).unwrap(),
            ),
            (
                scale,
                HostTensor::new([4], vec![1.0, 0.5, -1.0, 2.0]).unwrap(),
            ),
        ]);
        if with_bias {
            graph = graph.with_edge(TensorEdge::new(
                bias.clone(),
                f32_tensor_descriptor_from_shape(&[4]),
            ));
            norm = norm.with_input(bias.clone());
            bindings.insert(
                bias,
                HostTensor::new([4], vec![0.25, -0.1, 0.7, 0.9]).unwrap(),
            );
        }
        graph = graph
            .with_node(norm)
            .with_node(
                ExecutionNode::new(
                    ExecutionNodeId::new("activation"),
                    OperatorId::magnetar("gelu-exact", 1, OperatorFamily::Activation),
                )
                .with_input(normalized)
                .with_output(activated.clone()),
            )
            .with_node(
                ExecutionNode::new(
                    ExecutionNodeId::new("slice"),
                    OperatorId::magnetar("slice-last", 1, OperatorFamily::Tensor),
                )
                .with_input(activated)
                .with_output(logits.clone())
                .with_attribute("start", OperatorAttributeValue::Integer(1))
                .with_attribute("width", OperatorAttributeValue::Integer(2)),
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
        let (_, outputs, _, _) = execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &KvCacheId::new("encoder-primitives").unwrap(),
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
        let actual = &outputs[&logits].data;
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 2e-6,
                "bias {with_bias}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn encoder_primitives_reject_invalid_inputs_and_preserve_exact_gelu() {
    let x = HostTensor::new([1, 2], vec![3.0, -2.0]).unwrap();
    let scale = HostTensor::new([2], vec![1.0, 1.0]).unwrap();
    assert!(crate::operator_semantics::affine_layernorm(&x, &scale, None, 0.0).is_err());
    assert!(
        crate::operator_semantics::affine_layernorm(
            &x,
            &HostTensor::new([1], vec![1.0]).unwrap(),
            None,
            1e-5
        )
        .is_err()
    );
    assert!(crate::operator_semantics::slice_last(&x, 1, 2).is_err());
    assert!(crate::operator_semantics::slice_last(&x, 0, 0).is_err());
    let invalid = HostTensor::new([1], vec![f32::NAN]).unwrap();
    assert!(crate::operator_semantics::gelu_exact(&invalid).is_err());
    let exact = crate::operator_semantics::gelu_exact(&x).unwrap();
    let approximation = crate::reference_cpu::gelu(&x);
    assert!((exact.data[0] - 2.995_950_2).abs() < 1e-6);
    assert!((exact.data[0] - approximation.data[0]).abs() > 1e-5);
}

#[test]
fn encoder_erf_gelu_tracks_pytorch_across_input_range() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../fixtures/gelu-exact-reference.json")).unwrap();
    let read = |name: &str| {
        fixture[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let input = read("input");
    let expected = read("output");
    let result = crate::operator_semantics::gelu_exact(
        &HostTensor::new([input.len() as u64], input).unwrap(),
    )
    .unwrap();
    for (index, (&actual, &expected)) in result.data.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 5e-7,
            "GELU sample {index}: {actual} != {expected}"
        );
    }
}
