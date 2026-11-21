//! Full synthetic ModernBERT encoder graph against pinned Transformers layers.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    bias: bool,
    ids: Vec<u32>,
    mask: Vec<u8>,
    positions: Vec<u32>,
    output: Vec<f32>,
}

fn pattern(name: &str, count: usize) -> Vec<f32> {
    let seed = name
        .bytes()
        .enumerate()
        .map(|(index, byte)| (index + 1) * byte as usize)
        .sum::<usize>()
        % 97;
    (0..count)
        .map(|index| {
            let mut value = ((index * 17 + seed) % 101) as f32 - 50.0;
            value /= 200.0;
            if name.contains("norm") && name.ends_with(".weight") {
                value += 1.0;
            }
            value
        })
        .collect()
}

struct Builder {
    graph: ExecutionGraph,
    bindings: BTreeMap<TensorEdgeId, HostTensor>,
    tokens: u64,
    bias: bool,
}

impl Builder {
    fn input(&mut self, name: &str, shape: &[u64], data: Vec<f32>) -> TensorEdgeId {
        let id = TensorEdgeId::new(format!("input.{name}"));
        self.graph.edges.insert(
            id.clone(),
            TensorEdge::new(id.clone(), f32_tensor_descriptor_from_shape(shape)),
        );
        self.bindings
            .insert(id.clone(), HostTensor::new(shape, data).unwrap());
        id
    }

    fn node(
        &mut self,
        name: &str,
        operator: &str,
        family: OperatorFamily,
        inputs: Vec<TensorEdgeId>,
        shape: &[u64],
        attributes: Vec<(&str, OperatorAttributeValue)>,
    ) -> TensorEdgeId {
        let id = TensorEdgeId::new(if name == "logits" {
            name.to_string()
        } else {
            format!("out.{name}")
        });
        self.graph.edges.insert(
            id.clone(),
            TensorEdge::new(id.clone(), f32_tensor_descriptor_from_shape(shape)),
        );
        let mut node = ExecutionNode::new(
            ExecutionNodeId::new(name),
            OperatorId::magnetar(operator, 1, family),
        );
        for input in inputs {
            node = node.with_input(input);
        }
        for (name, value) in attributes {
            node = node.with_attribute(name, value);
        }
        self.graph
            .nodes
            .insert(node.id.clone(), node.with_output(id.clone()));
        id
    }

    fn norm(&mut self, name: &str, input: TensorEdgeId, output_name: &str) -> TensorEdgeId {
        let scale_name = format!("{name}.weight");
        let scale = self.input(&scale_name, &[8], pattern(&scale_name, 8));
        let mut inputs = vec![input, scale];
        if self.bias {
            let bias_name = format!("{name}.bias");
            inputs.push(self.input(&bias_name, &[8], pattern(&bias_name, 8)));
        }
        self.node(
            output_name,
            "affine-layernorm",
            OperatorFamily::Normalization,
            inputs,
            &[self.tokens, 8],
            vec![("epsilon", OperatorAttributeValue::Float(1e-5))],
        )
    }

    fn linear(
        &mut self,
        name: &str,
        input: TensorEdgeId,
        in_width: usize,
        out_width: usize,
    ) -> TensorEdgeId {
        let weight_name = format!("{name}.weight");
        let source = pattern(&weight_name, in_width * out_width);
        let weights = (0..in_width)
            .flat_map(|row| {
                let source = &source;
                (0..out_width).map(move |column| source[column * in_width + row])
            })
            .collect();
        let weight = self.input(&weight_name, &[in_width as u64, out_width as u64], weights);
        let output = self.node(
            &format!("{name}.matmul"),
            "matmul",
            OperatorFamily::LinearAlgebra,
            vec![input, weight],
            &[self.tokens, out_width as u64],
            vec![],
        );
        if !self.bias {
            return output;
        }
        let bias_name = format!("{name}.bias");
        let bias = self.input(
            &bias_name,
            &[out_width as u64],
            pattern(&bias_name, out_width),
        );
        self.node(
            &format!("{name}.add"),
            "add",
            OperatorFamily::Tensor,
            vec![output, bias],
            &[self.tokens, out_width as u64],
            vec![],
        )
    }

    fn slice(&mut self, name: &str, input: TensorEdgeId, start: i64, width: i64) -> TensorEdgeId {
        self.node(
            name,
            "slice-last",
            OperatorFamily::Tensor,
            vec![input],
            &[self.tokens, width as u64],
            vec![
                ("start", OperatorAttributeValue::Integer(start)),
                ("width", OperatorAttributeValue::Integer(width)),
            ],
        )
    }

    fn rope(&mut self, name: &str, input: TensorEdgeId, theta: f64, offset: u32) -> TensorEdgeId {
        self.node(
            name,
            "rope",
            OperatorFamily::PositionEncoding,
            vec![input],
            &[self.tokens, 8],
            vec![
                ("base", OperatorAttributeValue::Float(theta)),
                ("dimension", OperatorAttributeValue::Integer(4)),
                ("head_count", OperatorAttributeValue::Integer(2)),
                (
                    "position_offset",
                    OperatorAttributeValue::Integer(offset as i64),
                ),
                (
                    "rotation_pairing",
                    OperatorAttributeValue::String("half_split".into()),
                ),
            ],
        )
    }
}

fn graph(case: &Case) -> (ExecutionGraph, BTreeMap<TensorEdgeId, HostTensor>) {
    let mut builder = Builder {
        graph: ExecutionGraph::new(
            ExecutionGraphId::new("modernbert-encoder"),
            ExecutionGraphPhase::Prefill,
        ),
        bindings: BTreeMap::new(),
        tokens: case.ids.len() as u64,
        bias: case.bias,
    };
    let ids = builder.input(
        "token_ids",
        &[builder.tokens],
        case.ids.iter().map(|&value| value as f32).collect(),
    );
    let mask = builder.input(
        "mask",
        &[builder.tokens],
        case.mask.iter().map(|&value| value as f32).collect(),
    );
    let embedding = builder.input(
        "embeddings.tok_embeddings.weight",
        &[24, 8],
        pattern("embeddings.tok_embeddings.weight", 24 * 8),
    );
    let embedding = builder.node(
        "embedding",
        "embedding",
        OperatorFamily::Tensor,
        vec![ids, embedding],
        &[builder.tokens, 8],
        vec![],
    );
    let mut state = builder.norm("embeddings.norm", embedding, "embedding_norm");
    for index in 0..3 {
        let prefix = format!("layers.{index}");
        let local = index == 1;
        let input = if index == 0 {
            state.clone()
        } else {
            builder.norm(
                &format!("{prefix}.attn_norm"),
                state.clone(),
                &format!("{prefix}.attn_norm.node"),
            )
        };
        let qkv = builder.linear(&format!("{prefix}.attn.Wqkv"), input, 8, 24);
        let q = builder.slice(&format!("{prefix}.q"), qkv.clone(), 0, 8);
        let k = builder.slice(&format!("{prefix}.k"), qkv.clone(), 8, 8);
        let v = builder.slice(&format!("{prefix}.v"), qkv, 16, 8);
        let theta = if local { 10_000.0 } else { 160_000.0 };
        let q = builder.rope(&format!("{prefix}.rope_q"), q, theta, case.positions[0]);
        let k = builder.rope(&format!("{prefix}.rope_k"), k, theta, case.positions[0]);
        let mut attrs = vec![
            ("head_count", OperatorAttributeValue::Integer(2)),
            ("head_dimension", OperatorAttributeValue::Integer(4)),
        ];
        if local {
            attrs.push(("local_radius", OperatorAttributeValue::Integer(2)));
        }
        let attended = builder.node(
            &format!("{prefix}.attention"),
            "encoder-attention",
            OperatorFamily::Attention,
            vec![q, k, v, mask.clone()],
            &[builder.tokens, 8],
            attrs,
        );
        let attended = builder.linear(&format!("{prefix}.attn.Wo"), attended, 8, 8);
        state = builder.node(
            &format!("{prefix}.attention_residual"),
            "residual-add",
            OperatorFamily::Tensor,
            vec![state, attended],
            &[builder.tokens, 8],
            vec![],
        );
        let normalized = builder.norm(
            &format!("{prefix}.mlp_norm"),
            state.clone(),
            &format!("{prefix}.mlp_norm.node"),
        );
        let projected = builder.linear(&format!("{prefix}.mlp.Wi"), normalized, 8, 24);
        let left = builder.slice(&format!("{prefix}.mlp.input"), projected.clone(), 0, 12);
        let right = builder.slice(&format!("{prefix}.mlp.gate"), projected, 12, 12);
        let activated = builder.node(
            &format!("{prefix}.mlp.gelu"),
            "gelu-exact",
            OperatorFamily::Activation,
            vec![left],
            &[builder.tokens, 12],
            vec![],
        );
        let gated = builder.node(
            &format!("{prefix}.mlp.mul"),
            "mul",
            OperatorFamily::Tensor,
            vec![activated, right],
            &[builder.tokens, 12],
            vec![],
        );
        let output = builder.linear(&format!("{prefix}.mlp.Wo"), gated, 12, 8);
        state = builder.node(
            &format!("{prefix}.mlp.residual"),
            "residual-add",
            OperatorFamily::Tensor,
            vec![state, output],
            &[builder.tokens, 8],
            vec![],
        );
    }
    builder.norm("final_norm", state, "logits");
    (builder.graph, builder.bindings)
}

#[test]
fn modernbert_encoder_graph_matches_pinned_transformers() {
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../fixtures/modernbert-encoder-reference.json"
    ))
    .unwrap();
    for case in cases {
        let (graph, bindings) = graph(&case);
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
            case.ids.len() as u64,
            PreparedExecutionPlanGeneration::new(1),
            &ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME),
        )
        .unwrap();
        let (_, outputs, _, _) = execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &KvCacheId::new("modernbert-graph").unwrap(),
            &graph,
            &mut plan,
            bindings,
            None,
            Some(case.positions[0] as u64),
            &mut Vec::new(),
            MaterializedEdges::Only(&["logits"]),
            None,
            None,
        )
        .unwrap();
        let actual = &outputs[&TensorEdgeId::new("logits")].data;
        assert_eq!(actual.len(), case.output.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&case.output).enumerate() {
            assert!(
                (actual - expected).abs() <= 3e-6 + 1e-6 * expected.abs(),
                "bias {} output[{index}]: {actual} != {expected}",
                case.bias
            );
        }
    }
}
