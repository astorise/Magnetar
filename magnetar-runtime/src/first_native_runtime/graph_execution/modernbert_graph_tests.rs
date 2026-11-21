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

/// Exercises the actual WebAssembly Component, host capabilities, planner and
/// numeric executor. CI can substitute the Component just built from source.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn modernbert_component_graph_matches_pinned_transformers() {
    let bytes = match std::env::var_os("MAGNETAR_MODERNBERT_TEST_COMPONENT") {
        Some(path) => std::fs::read(path).unwrap(),
        None => include_bytes!("../../../fixtures/components/modernbert-encoder.component.wasm")
            .to_vec(),
    };
    let digest = ComponentDigest::sha256(&bytes);
    // Derive a complete local-development manifest from the existing fixture;
    // its digest always identifies the exact Component bytes used by this test.
    let manifest = include_str!(
        "../../../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    )
    .replace(
        "sha256:abfdf515d2d2d52bd0738d52d3628a6077e65d4217c6c19c9598eef7ae0386a1",
        &digest.value,
    )
    .replace("magnetar.qwen.real", "magnetar.modernbert.test")
    .replace("1.8.0", "1.12.0");
    let graph_builder = Arc::new(GraphBuilderCapability::new());
    let model_config = Arc::new(ModelConfigCapability::new());
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().unwrap(),
    ));
    manager.set_resource_limits(inference_component_runtime_limits());
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(&digest.value));
    manager.provide_capability(
        WitInterface::new("magnetar:model-component-graph/graph-builder", "1.12.0"),
        graph_builder.clone(),
    );
    manager.provide_capability(
        WitInterface::new("magnetar:model-component-graph/model-config", "1.12.0"),
        model_config.clone(),
    );
    let definition = manager
        .prepare_pushed_package(ComponentArtifactPackage::new(
            bytes,
            manifest.into_bytes(),
            digest,
            ComponentDistributionSource::new(
                ComponentDistributionSourceKind::DevelopmentFixture,
                "modernbert-test",
            ),
        ))
        .unwrap();
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../fixtures/modernbert-encoder-reference.json"
    ))
    .unwrap();
    let fixture = e2e_fixture().unwrap();
    for case in cases {
        let component = manager.instantiate_prepared_component(definition).unwrap();
        let key = manager.engine_instance_key(component).unwrap().to_string();
        let (_, original_bindings) = graph(&case);
        let mut shapes = BTreeMap::new();
        let mut bindings = BTreeMap::new();
        for (edge, tensor) in original_bindings {
            let name = edge.as_str().strip_prefix("input.").unwrap();
            let target = match name {
                "token_ids" => "input.token_ids".to_string(),
                "mask" => "input.attention_mask".to_string(),
                _ => {
                    shapes.insert(name.to_string(), tensor.shape.clone());
                    format!("weight.{name}")
                }
            };
            bindings.insert(TensorEdgeId::new(target), tensor);
        }
        let mut config = architecture_config_from_first_native_model_config(&fixture.config);
        config.hidden_size = 8;
        config.intermediate_size = 12;
        config.num_hidden_layers = 3;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.head_dim = 4;
        config.vocab_size = 24;
        config.hidden_activation = Some("gelu".into());
        config.attention_bias = case.bias;
        let encoder_config = EncoderGraphConfig {
            norm_epsilon: 1e-5,
            max_positions: 64,
            norm_bias: case.bias,
            attention_bias: case.bias,
            mlp_bias: case.bias,
            layers: (0..3)
                .map(|index| EncoderLayerConfig {
                    local_radius: (index == 1).then_some(2),
                    rope_theta: if index == 1 { 10_000.0 } else { 160_000.0 },
                    attention_normalization: index != 0,
                })
                .collect(),
        };
        model_config.bind_config(&key, config.clone());
        model_config
            .bind_encoder_config(&key, encoder_config.clone())
            .unwrap();
        graph_builder.prepare_session(
            &key,
            SessionContext {
                component_id: "modernbert-test".into(),
                compatibility_key: "modernbert-test".into(),
                kv_namespace: "unused".into(),
                weight_shapes: shapes.clone(),
                output_edge_name: "logits".into(),
            },
        );
        let result = manager
            .invoke(
                ComponentInvocation::new(
                    component,
                    WitInterface::new(
                        "magnetar:model-component-graph/encoder-component-graph-producer",
                        "1.12.0",
                    ),
                    "build-encoder-graph",
                )
                .with_arguments(vec![ComponentValue::S64(case.ids.len() as i64)]),
            )
            .unwrap();
        let [ComponentValue::String(handle)] = result.values.as_slice() else {
            panic!("expected a graph handle, got {:?}", result.values);
        };
        let graph = graph_builder.take_graph(&key, handle).unwrap();
        assert_eq!(graph.phase, ExecutionGraphPhase::Encode);
        assert!(model_config.unread_declared_features(&key).is_empty());
        graph.validate(&initial_operator_catalog()).unwrap();

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
            &KvCacheId::new("modernbert-component").unwrap(),
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
        let actual = &outputs[&TensorEdgeId::new("logits")].data;
        assert_eq!(actual.len(), case.output.len());
        // Uniform RoPE position shifts preserve Q/K dot products; the second
        // independent fixture starts at position 3, while this API starts at 0.
        for (index, (&actual, &expected)) in actual.iter().zip(&case.output).enumerate() {
            assert!(
                (actual - expected).abs() <= 3e-6 + 1e-6 * expected.abs(),
                "bias {} output[{index}]: {actual} != {expected}",
                case.bias
            );
        }
        graph_builder.clear_session(&key);
        model_config.clear_config(&key);
        manager.destroy_instance(component).unwrap();

        // A WebAssembly trap can poison an instance. Isolate each rejection,
        // while preparing a valid builder session so the test cannot pass due
        // merely to a missing session or weight.
        for unsupported in ["length", "activation", "rope-scaling"] {
            let component = manager.instantiate_prepared_component(definition).unwrap();
            let key = manager.engine_instance_key(component).unwrap().to_string();
            let mut unsupported_config = config.clone();
            if unsupported == "activation" {
                unsupported_config.hidden_activation = Some("silu".into());
            } else if unsupported == "rope-scaling" {
                unsupported_config.rope_scaling_factor = Some(2.0);
            }
            model_config.bind_config(&key, unsupported_config);
            model_config
                .bind_encoder_config(&key, encoder_config.clone())
                .unwrap();
            graph_builder.prepare_session(
                &key,
                SessionContext {
                    component_id: "modernbert-test".into(),
                    compatibility_key: "modernbert-test".into(),
                    kv_namespace: "unused".into(),
                    weight_shapes: shapes.clone(),
                    output_edge_name: "logits".into(),
                },
            );
            let tokens = if unsupported == "length" {
                65
            } else {
                case.ids.len() as i64
            };
            let result = manager.invoke(
                ComponentInvocation::new(
                    component,
                    WitInterface::new(
                        "magnetar:model-component-graph/encoder-component-graph-producer",
                        "1.12.0",
                    ),
                    "build-encoder-graph",
                )
                .with_arguments(vec![ComponentValue::S64(tokens)]),
            );
            assert!(result.is_err(), "accepted unsupported {unsupported}");
            graph_builder.clear_session(&key);
            model_config.clear_config(&key);
            manager.destroy_instance(component).unwrap();
        }
    }
}
