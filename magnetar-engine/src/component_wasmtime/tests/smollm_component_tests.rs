//! Runs the independently compiled SmolLM2 graph component in the real
//! Wasmtime engine. Shapes follow the public SmolLM2-135M-Instruct config at
//! revision 12fd25f77366fa6b3b4b768ec3050bf629380bac; no weight data is
//! loaded, so these are graph/ABI checks, not numerical inference tests.

use super::*;

const SMOLLM2_COMPONENT: &[u8] =
    include_bytes!("../../../../magnetar-runtime/fixtures/components/smollm2-graph.component.wasm");
static NEXT_DIRECTORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn smollm2_graph(
    export: &str,
    arguments: Vec<ComponentValue>,
    activation: Option<&str>,
    rope_layers: Option<Vec<bool>>,
) -> Result<crate::ExecutionGraph, String> {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-smollm2-graph-{}-{export}-{}-{}",
        std::process::id(),
        activation.unwrap_or("missing"),
        NEXT_DIRECTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("smollm2.component.wasm");
    let bytes = match std::env::var_os("MAGNETAR_SMOLLM2_TEST_COMPONENT") {
        Some(path) => std::fs::read(path).expect("the supplied SmolLM2 component is readable"),
        None => SMOLLM2_COMPONENT.to_vec(),
    };
    std::fs::write(&artifact, bytes).unwrap();

    let graph_builder = Arc::new(crate::GraphBuilderCapability::new());
    let model_config = Arc::new(crate::ModelConfigCapability::new());
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let gb = WitInterface::new("magnetar:model-component-graph/graph-builder", "1.9.0");
    let mc = WitInterface::new("magnetar:model-component-graph/model-config", "1.9.0");
    let exports = WitInterface::new(
        "magnetar:model-component-graph/model-component-graph-producer",
        "1.9.0",
    );
    engine.register_capability(gb.clone(), graph_builder.clone() as Arc<dyn HostCapability>);
    engine.register_capability(mc.clone(), model_config.clone() as Arc<dyn HostCapability>);
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(301),
        metadata: crate::ComponentMetadata::new("smollm2", "1", "SmolLM2 graph")
            .with_import(gb.clone())
            .with_import(mc.clone())
            .with_export(exports.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: gb });
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: mc });
    let instance = engine.instantiate(&prepared, &link_plan).unwrap();

    let (hidden, intermediate, heads, kv_heads, head_dim, vocab) =
        (576_u64, 1536_u64, 9_u64, 3_u64, 64_u64, 49152_u64);
    let mut weights = BTreeMap::from([
        ("token_embedding".to_string(), vec![vocab, hidden]),
        ("final_norm".to_string(), vec![hidden]),
        ("lm_head".to_string(), vec![hidden, vocab]),
    ]);
    for layer in 0..30 {
        for (name, dimensions) in [
            ("input_norm", vec![hidden]),
            ("post_attn_norm", vec![hidden]),
            ("self_attn.q_proj", vec![hidden, heads * head_dim]),
            ("self_attn.k_proj", vec![hidden, kv_heads * head_dim]),
            ("self_attn.v_proj", vec![hidden, kv_heads * head_dim]),
            ("self_attn.o_proj", vec![heads * head_dim, hidden]),
            ("mlp.gate_proj", vec![hidden, intermediate]),
            ("mlp.up_proj", vec![hidden, intermediate]),
            ("mlp.down_proj", vec![intermediate, hidden]),
        ] {
            weights.insert(format!("layers.{layer}.{name}"), dimensions);
        }
    }
    graph_builder.prepare_session(
        instance.engine_key(),
        crate::graph_builder_capability::SessionContext {
            component_id: "smollm2".into(),
            compatibility_key: "smollm2-135m".into(),
            kv_namespace: "smollm2".into(),
            weight_shapes: weights,
            output_edge_name: "logits".into(),
        },
    );
    model_config.bind_config(
        instance.engine_key(),
        crate::ModelArchitectureConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_hidden_layers: 30,
            num_attention_heads: heads as u32,
            num_key_value_heads: kv_heads as u32,
            head_dim,
            vocab_size: vocab,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000.0,
            hidden_activation: activation.map(str::to_string),
            rope_layers,
            tie_word_embeddings: true,
            attention_bias: false,
            ..Default::default()
        },
    );
    let result = engine.invoke(
        &instance,
        &ComponentInvocation::new(crate::ComponentInstanceId::new(301), exports, export)
            .with_arguments(arguments),
    );
    let graph = match result {
        Ok(result) => {
            let [ComponentValue::String(handle)] = result.values.as_slice() else {
                panic!("expected graph handle, got {:?}", result.values);
            };
            graph_builder
                .take_graph(instance.engine_key(), handle)
                .ok_or_else(|| "the finished graph was not registered".to_string())
        }
        Err(error) => Err(format!("component invocation failed: {error:?}")),
    };
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    graph
}

#[test]
fn smollm2_published_shapes_produce_prefill_and_cached_decode_graphs() {
    for (export, argument, phase) in [
        (
            "build-prefill-graph",
            4,
            crate::ExecutionGraphPhase::Prefill,
        ),
        ("build-decode-graph", 4, crate::ExecutionGraphPhase::Decode),
    ] {
        let graph = smollm2_graph(
            export,
            vec![ComponentValue::S64(argument)],
            Some("silu"),
            None,
        )
        .unwrap();
        assert_eq!(graph.phase, phase);
        assert_eq!(graph.nodes.len(), 483);
        assert!(
            graph
                .edges
                .contains_key(&crate::TensorEdgeId::new("logits"))
        );
        let lm_head = graph
            .edges
            .get(&crate::TensorEdgeId::new("weight.lm_head"))
            .unwrap();
        assert_eq!(
            lm_head.aliasing,
            crate::TensorAliasing::MayAlias(crate::TensorEdgeId::new("weight.token_embedding"))
        );
    }
}

#[test]
fn smollm2_component_refuses_unverified_or_unsupported_activation() {
    for export in ["build-prefill-graph", "build-decode-graph"] {
        for activation in [None, Some("gelu")] {
            assert!(
                smollm2_graph(export, vec![ComponentValue::S64(4)], activation, None).is_err(),
                "{export}: {activation:?}"
            );
        }
    }
}

#[test]
fn smollm3_mixed_rope_and_nope_keep_key_cache_across_segments() {
    let mask = (0..30).map(|layer| layer % 4 != 3).collect::<Vec<_>>();
    for (export, arguments, append) in [
        ("build-prefill-graph", vec![ComponentValue::S64(4)], false),
        ("build-decode-graph", vec![ComponentValue::S64(4)], true),
        (
            "build-prefill-graph-segment",
            vec![
                ComponentValue::S64(4),
                ComponentValue::U32(2),
                ComponentValue::U32(5),
            ],
            false,
        ),
        (
            "build-decode-graph-segment",
            vec![
                ComponentValue::S64(4),
                ComponentValue::U32(2),
                ComponentValue::U32(5),
            ],
            true,
        ),
    ] {
        let graph = smollm2_graph(export, arguments, Some("silu"), Some(mask.clone())).unwrap();
        assert!(
            graph
                .nodes
                .contains_key(&crate::ExecutionNodeId::new("layer2.rope_q"))
        );
        assert!(
            !graph
                .nodes
                .contains_key(&crate::ExecutionNodeId::new("layer3.rope_q"))
        );
        assert!(
            !graph
                .nodes
                .contains_key(&crate::ExecutionNodeId::new("layer3.rope_k"))
        );
        let key = &graph.edges[&crate::TensorEdgeId::new("edge.layer3.k_proj")];
        assert_eq!(
            key.kv_cache.as_ref().unwrap().behavior,
            if append {
                crate::GraphKvCacheBehavior::Append
            } else {
                crate::GraphKvCacheBehavior::Output
            }
        );
    }
    assert!(
        smollm2_graph(
            "build-prefill-graph",
            vec![ComponentValue::S64(4)],
            Some("silu"),
            Some(vec![true])
        )
        .is_err()
    );
}
