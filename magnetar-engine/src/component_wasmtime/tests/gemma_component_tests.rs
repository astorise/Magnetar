//! The real Gemma 4 Model Component (`components/gemma` at `9186a0c`,
//! checked in as `gemma-real.component.wasm`) inside this engine, with the
//! real `GraphBuilderCapability` and `ModelConfigCapability` and a layered
//! `ModelArchitectureConfig` bound. Only weight shapes are offered, no
//! data: these check that the graphs the Component builds pass the
//! graph-builder's validation, for small layouts and for the published
//! 31B, 26B-A4B and E2B shapes. Adopted from the Component's own
//! `docs/engine-smoke-test/`.

use super::*;

const GEMMA_REAL_COMPONENT: &[u8] =
    include_bytes!("../../../../magnetar-runtime/fixtures/components/gemma-real.component.wasm");

/// Distinct scratch directories for tests running in parallel.
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn gemma_graph(
    config: crate::ModelArchitectureConfig,
    weights: Vec<(String, Vec<u64>)>,
    export: &str,
    args: Vec<ComponentValue>,
) -> Result<crate::ExecutionGraph, String> {
    let directory = std::env::temp_dir().join(format!(
        "gemma-e2e-{}-{export}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("gemma.component.wasm");
    let bytes = match std::env::var_os("MAGNETAR_GEMMA_TEST_COMPONENT") {
        Some(path) => std::fs::read(path).expect("fresh Gemma Component is readable"),
        None => GEMMA_REAL_COMPONENT.to_vec(),
    };
    std::fs::write(&artifact, bytes).unwrap();

    let capability = Arc::new(crate::GraphBuilderCapability::new());
    let model_config_capability = Arc::new(crate::ModelConfigCapability::new());
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let gb = WitInterface::new("magnetar:model-component-graph/graph-builder", "1.8.0");
    engine.register_capability(gb.clone(), capability.clone() as Arc<dyn HostCapability>);
    let mc = WitInterface::new("magnetar:model-component-graph/model-config", "1.8.0");
    engine.register_capability(
        mc.clone(),
        model_config_capability.clone() as Arc<dyn HostCapability>,
    );
    let export_interface = WitInterface::new(
        "magnetar:model-component-graph/model-component-graph-producer",
        "1.8.0",
    );
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(300),
        metadata: crate::ComponentMetadata::new("gemma", "1", "gemma 4")
            .with_import(gb.clone())
            .with_import(mc.clone())
            .with_export(export_interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .map_err(|e| format!("prepare: {e:?}"))?;
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: gb });
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: mc });
    let instance = engine
        .instantiate(&prepared, &link_plan)
        .map_err(|e| format!("instantiate: {e:?}"))?;
    capability.prepare_session(
        instance.engine_key(),
        crate::graph_builder_capability::SessionContext {
            component_id: "gemma".into(),
            compatibility_key: "gemma-e2e".into(),
            kv_namespace: "gemma".into(),
            weight_shapes: weights.into_iter().collect(),
            output_edge_name: "logits".into(),
        },
    );
    model_config_capability.bind_config(instance.engine_key(), config);
    let result = engine.invoke(
        &instance,
        &ComponentInvocation::new(
            crate::ComponentInstanceId::new(300),
            export_interface,
            export,
        )
        .with_arguments(args),
    );
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            let _ = engine.destroy(instance);
            let _ = std::fs::remove_dir_all(directory);
            return Err(format!("invoke: {error:?}"));
        }
    };
    let [ComponentValue::String(handle)] = result.values.as_slice() else {
        return Err(format!("unexpected result {:?}", result.values));
    };
    let graph = capability
        .take_graph(instance.engine_key(), handle)
        .ok_or("no graph")?;
    let _ = engine.destroy(instance);
    let _ = std::fs::remove_dir_all(directory);
    Ok(graph)
}

fn spec(
    window: Option<u32>,
    hd: u64,
    kv: u32,
    partial: Option<f64>,
    vfk: bool,
    src: Option<u32>,
    inter: u64,
) -> crate::LayerSpec {
    crate::LayerSpec {
        sliding_window: window,
        head_dim: hd,
        num_key_value_heads: kv,
        rope_theta: if window.is_some() {
            10_000.0
        } else {
            1_000_000.0
        },
        partial_rotary_factor: partial,
        value_from_key: vfk,
        kv_source_layer: src,
        intermediate_size: inter,
    }
}

fn gemma_weights(
    layers: &[crate::LayerSpec],
    heads: u64,
    h: u64,
    vocab: u64,
) -> Vec<(String, Vec<u64>)> {
    let mut w: Vec<(String, Vec<u64>)> = vec![
        ("token_embedding".into(), vec![vocab, h]),
        ("final_norm".into(), vec![h]),
        ("lm_head".into(), vec![h, vocab]),
    ];
    for (i, l) in layers.iter().enumerate() {
        let (hd, kv) = (l.head_dim, l.num_key_value_heads as u64);
        let mut add = |n: &str, d: Vec<u64>| w.push((format!("layers.{i}.{n}"), d));
        for n in [
            "input_norm",
            "post_attn_norm",
            "pre_ffn_norm",
            "post_ffn_norm",
        ] {
            add(n, vec![h]);
        }
        add("layer_scalar", vec![1]);
        add("self_attn.q_proj", vec![h, heads * hd]);
        add("self_attn.q_norm", vec![hd]);
        add("self_attn.o_proj", vec![heads * hd, h]);
        if l.kv_source_layer.is_none() {
            add("self_attn.k_proj", vec![h, kv * hd]);
            add("self_attn.k_norm", vec![hd]);
            if !l.value_from_key {
                add("self_attn.v_proj", vec![h, kv * hd]);
            }
        }
        add("mlp.gate_proj", vec![h, l.intermediate_size]);
        add("mlp.up_proj", vec![h, l.intermediate_size]);
        add("mlp.down_proj", vec![l.intermediate_size, h]);
    }
    w
}

#[test]
fn gemma_e2e_dense_with_shared_kv() {
    let layers = vec![
        spec(Some(4), 4, 1, None, false, None, 16),
        spec(None, 8, 1, Some(0.25), true, None, 16),
        spec(Some(4), 4, 1, None, false, Some(0), 16),
        spec(None, 8, 1, Some(0.25), true, Some(1), 16),
    ];
    let weights = gemma_weights(&layers, 2, 8, 16);
    let config = crate::ModelArchitectureConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_activation: Some("gelu_pytorch_tanh".into()),
        tie_word_embeddings: true,
        layer_layout: Some(layers),
        final_logit_softcapping: Some(30.0),
        ..Default::default()
    };
    let graph = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(3)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("dense graph: {} nodes", graph.nodes.len());
    assert_eq!(graph.nodes.len(), 81);
    assert!(
        graph
            .edges
            .contains_key(&crate::TensorEdgeId::new("logits"))
    );
}

fn base_config(layers: Vec<crate::LayerSpec>) -> crate::ModelArchitectureConfig {
    crate::ModelArchitectureConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: layers.len() as u32,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_activation: Some("gelu_pytorch_tanh".into()),
        tie_word_embeddings: true,
        layer_layout: Some(layers),
        final_logit_softcapping: Some(30.0),
        ..Default::default()
    }
}

#[test]
fn gemma_rejects_unknown_or_incompatible_activations() {
    let layers = vec![spec(Some(4), 4, 1, None, false, None, 16)];
    let weights = gemma_weights(&layers, 2, 8, 16);
    for activation in [None, Some("silu"), Some("gelu")] {
        let mut config = base_config(layers.clone());
        config.hidden_activation = activation.map(str::to_string);
        assert!(
            gemma_graph(
                config,
                weights.clone(),
                "build-prefill-graph",
                vec![ComponentValue::S64(3)]
            )
            .is_err()
        );
    }
}

#[test]
fn gemma_e2e_moe() {
    let layers = vec![
        spec(Some(4), 4, 1, None, false, None, 16),
        spec(None, 8, 1, Some(0.25), true, None, 16),
    ];
    let mut weights = gemma_weights(&layers, 2, 8, 16);
    let (e, i, h) = (4u64, 6u64, 8u64);
    for l in 0..2 {
        for (n, d) in [
            ("moe.router_proj", vec![h, e]),
            ("moe.experts_gate_proj", vec![h, e * i]),
            ("moe.experts_up_proj", vec![h, e * i]),
            ("moe.experts_down_proj", vec![i, e * h]),
            ("moe.per_expert_scale", vec![e]),
            ("pre_ffn_norm_2", vec![h]),
            ("post_ffn_norm_1", vec![h]),
            ("post_ffn_norm_2", vec![h]),
        ] {
            weights.push((format!("layers.{l}.{n}"), d));
        }
    }
    let mut config = base_config(layers);
    config.mixture_of_experts = Some(crate::MixtureOfExpertsConfig {
        experts: 4,
        experts_per_token: 2,
        expert_intermediate_size: 6,
        shared_expert_intermediate_size: None,
        normalize_top_k: true,
    });
    let graph = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(3)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let moe = graph
        .nodes
        .values()
        .filter(|n| format!("{:?}", n.operator).to_lowercase().contains("moe"))
        .count();
    eprintln!(
        "moe graph: {} nodes, {moe} moe-experts nodes",
        graph.nodes.len()
    );
    assert_eq!(moe, 2);
    assert_eq!(graph.nodes.len(), 61);
}

#[test]
fn gemma_e2e_per_layer_inputs() {
    let layers = vec![
        spec(Some(4), 4, 1, None, false, None, 16),
        spec(None, 8, 1, Some(0.25), false, None, 16),
        spec(Some(4), 4, 1, None, false, Some(0), 32),
    ];
    let mut weights = gemma_weights(&layers, 2, 8, 16);
    let (p, h) = (4u64, 8u64);
    weights.push(("per_layer_projection_norm".into(), vec![p]));
    for l in 0..3 {
        for (n, d) in [
            ("per_layer_embedding", vec![16, p]),
            ("per_layer_model_projection", vec![h, p]),
            ("per_layer_input_gate", vec![h, p]),
            ("per_layer_projection", vec![p, h]),
            ("post_per_layer_input_norm", vec![h]),
        ] {
            weights.push((format!("layers.{l}.{n}"), d));
        }
    }
    let mut config = base_config(layers);
    config.per_layer_input_size = Some(4);
    let graph = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(3)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("ple graph: {} nodes", graph.nodes.len());
    assert_eq!(graph.nodes.len(), 101);
}

#[test]
fn gemma_e2e_decode_and_segments() {
    let layers = vec![
        spec(Some(4), 4, 1, None, false, None, 16),
        spec(None, 8, 1, Some(0.25), true, None, 16),
        spec(Some(4), 4, 1, None, false, Some(0), 16),
        spec(None, 8, 1, Some(0.25), true, Some(1), 16),
    ];
    let w = || gemma_weights(&layers, 2, 8, 16);
    let cfg = || base_config(layers.clone());
    let decode = gemma_graph(
        cfg(),
        w(),
        "build-decode-graph",
        vec![ComponentValue::S64(5)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(decode.phase, crate::ExecutionGraphPhase::Decode);
    // Segment [0, 2): embedding in, hidden state out.
    let first = gemma_graph(
        cfg(),
        w(),
        "build-prefill-graph-segment",
        vec![
            ComponentValue::S64(3),
            ComponentValue::U32(0),
            ComponentValue::U32(2),
        ],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("segment 0..2: {} nodes", first.nodes.len());
    assert_eq!(first.nodes.len(), 44);
    // A segment that splits a shared-KV layer from its producer must trap.
    let bad = gemma_graph(
        cfg(),
        w(),
        "build-prefill-graph-segment",
        vec![
            ComponentValue::S64(3),
            ComponentValue::U32(2),
            ComponentValue::U32(4),
        ],
    );
    assert!(
        bad.is_err(),
        "split of a shared-KV layer from its producer must be refused"
    );
    eprintln!("split refused: {}", bad.unwrap_err());
}

/// A published Gemma 4 checkpoint's layer pattern: every `full_every`-th
/// layer attends over the whole context, the others slide.
struct RealLayout {
    layers: usize,
    full_every: usize,
    /// Head width and KV head count of sliding layers.
    sliding: (u64, u32),
    /// Head width and KV head count of full layers.
    full: (u64, u32),
    /// Full layers derive V from K.
    value_from_key: bool,
    window: u32,
    intermediate: u64,
    /// The last `shared` layers reuse the K/V of the last earlier layer of
    /// their kind.
    shared: usize,
    /// Those layers' MLP is twice as wide.
    double_wide: bool,
}

fn real_layers(layout: RealLayout) -> Vec<crate::LayerSpec> {
    let first_shared = layout.layers - layout.shared;
    let full = |i: usize| i % layout.full_every == layout.full_every - 1;
    (0..layout.layers)
        .map(|i| {
            let consumer = layout.shared != 0 && i >= first_shared;
            let src = consumer.then(|| {
                (0..first_shared)
                    .rev()
                    .find(|j| full(*j) == full(i))
                    .unwrap() as u32
            });
            let inter = if consumer && layout.double_wide {
                layout.intermediate * 2
            } else {
                layout.intermediate
            };
            if full(i) {
                let (hd, kv) = layout.full;
                spec(None, hd, kv, Some(0.25), layout.value_from_key, src, inter)
            } else {
                let (hd, kv) = layout.sliding;
                spec(Some(layout.window), hd, kv, None, false, src, inter)
            }
        })
        .collect()
}

fn real_config(
    layers: Vec<crate::LayerSpec>,
    h: u64,
    heads: u32,
    kv: u32,
    inter: u64,
) -> crate::ModelArchitectureConfig {
    crate::ModelArchitectureConfig {
        hidden_size: h,
        intermediate_size: inter,
        num_hidden_layers: layers.len() as u32,
        num_attention_heads: heads,
        num_key_value_heads: kv,
        head_dim: 256,
        vocab_size: 262_144,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_activation: Some("gelu_pytorch_tanh".into()),
        tie_word_embeddings: true,
        layer_layout: Some(layers),
        final_logit_softcapping: Some(30.0),
        ..Default::default()
    }
}

#[test]
fn gemma_e2e_real_31b_shape() {
    let layers = real_layers(RealLayout {
        layers: 60,
        full_every: 6,
        sliding: (256, 16),
        full: (512, 4),
        value_from_key: true,
        window: 1024,
        intermediate: 21504,
        shared: 0,
        double_wide: false,
    });
    let weights = gemma_weights(&layers, 32, 5376, 262_144);
    let config = real_config(layers, 5376, 32, 16, 21504);
    let g = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(16)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("31B prefill: {} nodes", g.nodes.len());
    // 1 + 50*22 + 10*21 + 3
    assert_eq!(g.nodes.len(), 1 + 50 * 22 + 10 * 21 + 3);
}

#[test]
fn gemma_e2e_real_26b_a4b_shape() {
    let layers = real_layers(RealLayout {
        layers: 30,
        full_every: 6,
        sliding: (256, 8),
        full: (512, 2),
        value_from_key: true,
        window: 1024,
        intermediate: 2112,
        shared: 0,
        double_wide: false,
    });
    let mut weights = gemma_weights(&layers, 16, 2816, 262_144);
    let (e, i, h) = (128u64, 704u64, 2816u64);
    for l in 0..30 {
        for (n, d) in [
            ("moe.router_proj", vec![h, e]),
            ("moe.experts_gate_proj", vec![h, e * i]),
            ("moe.experts_up_proj", vec![h, e * i]),
            ("moe.experts_down_proj", vec![i, e * h]),
            ("moe.per_expert_scale", vec![e]),
            ("pre_ffn_norm_2", vec![h]),
            ("post_ffn_norm_1", vec![h]),
            ("post_ffn_norm_2", vec![h]),
        ] {
            weights.push((format!("layers.{l}.{n}"), d));
        }
    }
    let mut config = real_config(layers, h, 16, 8, 2112);
    config.mixture_of_experts = Some(crate::MixtureOfExpertsConfig {
        experts: 128,
        experts_per_token: 8,
        expert_intermediate_size: 704,
        shared_expert_intermediate_size: None,
        normalize_top_k: true,
    });
    let g = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(16)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("26B-A4B prefill: {} nodes", g.nodes.len());
    assert_eq!(g.nodes.len(), 869);
}

#[test]
fn gemma_e2e_real_e2b_shape() {
    // 35 layers (s s s s f)*7, 20 shared, double-wide MLP, per-layer inputs.
    let layers = real_layers(RealLayout {
        layers: 35,
        full_every: 5,
        sliding: (256, 1),
        full: (512, 1),
        value_from_key: false,
        window: 512,
        intermediate: 6144,
        shared: 20,
        double_wide: true,
    });
    let mut weights = gemma_weights(&layers, 8, 1536, 262_144);
    let (p, h) = (256u64, 1536u64);
    weights.push(("per_layer_projection_norm".into(), vec![p]));
    for l in 0..35 {
        for (n, d) in [
            ("per_layer_embedding", vec![262_144, p]),
            ("per_layer_model_projection", vec![h, p]),
            ("per_layer_input_gate", vec![h, p]),
            ("per_layer_projection", vec![p, h]),
            ("post_per_layer_input_norm", vec![h]),
        ] {
            weights.push((format!("layers.{l}.{n}"), d));
        }
    }
    let mut config = real_config(layers, h, 8, 1, 6144);
    config.per_layer_input_size = Some(256);
    let g = gemma_graph(
        config,
        weights,
        "build-prefill-graph",
        vec![ComponentValue::S64(16)],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("E2B prefill: {} nodes", g.nodes.len());
    assert_eq!(g.nodes.len(), 1094);
}
