//! Mixed Nemotron graph production through the real Component engine.

use super::*;
use std::collections::BTreeMap;

#[test]
fn nemotron_component_builds_mixed_prefill_and_decode_graphs() {
    // Built from components/nemotron with cargo build --locked --release
    // --target wasm32-unknown-unknown, followed by wasm-tools component new.
    let bytes = include_bytes!("../../../fixtures/components/nemotron-hybrid.component.wasm");
    let digest = ComponentDigest::sha256(bytes);
    let manifest = include_str!(
        "../../../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    )
    .replace(
        "sha256:321b42b83773544e1ba553cc7196d3d742cd9ef1e2af3f04da72b66e402a730f",
        &digest.value,
    )
    .replace("magnetar.qwen.real", "magnetar.nemotron.hybrid-test")
    .replace("qwen-model-component", "nemotron-model-component")
    .replace("1.14.0", "1.15.0");
    register_inference_component_artifact(
        bytes.to_vec(),
        manifest.into_bytes(),
        &ComponentTrustStore::default().trust_digest(&digest.value),
    )
    .unwrap();

    let fixture = e2e_fixture().unwrap();
    let mut config = architecture_config_from_first_native_model_config(&fixture.config);
    config.hidden_size = 5;
    config.intermediate_size = 3;
    config.num_hidden_layers = 3;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = 5;
    config.vocab_size = 24;
    config.attention_bias = false;
    config.tie_word_embeddings = false;
    config.hidden_activation = Some("relu_squared".into());
    config.rope_layers = Some(vec![false; 3]);
    config.mixture_of_experts = Some(crate::MixtureOfExpertsConfig {
        experts: 4,
        experts_per_token: 2,
        expert_intermediate_size: 3,
        shared_expert_intermediate_size: Some(6),
        normalize_top_k: true,
    });
    config.moe_layers = Some(vec![false, false, true]);
    config.expert_routing = Some(crate::ExpertRoutingConfig {
        scoring: crate::ExpertScoring::Sigmoid,
        groups: Some(crate::ExpertGroups {
            count: 1,
            per_token: 1,
            score: crate::ExpertGroupScore::TopTwoSum,
        }),
        routed_scaling_factor: 2.5,
        selection_bias: true,
        normalization: Some(crate::ExpertNormalization::Always),
        shared_expert_gated: false,
    });
    config.decoder_extensions.mamba2_hybrid = Some(crate::Mamba2HybridConfig {
        layers: vec![
            crate::HybridDecoderBlock::Mamba2,
            crate::HybridDecoderBlock::Attention,
            crate::HybridDecoderBlock::RoutedMoe,
        ],
        heads: 4,
        head_dim: 2,
        groups: 2,
        state_dim: 3,
        conv_kernel: 3,
        epsilon: 1e-5,
        dt_min: 0.0,
        dt_max: None,
        conv_bias: true,
        linear_bias: false,
    });
    config.validate().unwrap();
    let weights: BTreeMap<String, Vec<u64>> = [
        ("token_embedding", vec![24, 5]),
        ("final_norm", vec![5]),
        ("lm_head", vec![5, 24]),
        ("layers.0.norm", vec![5]),
        ("layers.0.mamba2.in_proj", vec![5, 32]),
        ("layers.0.mamba2.conv", vec![20, 3]),
        ("layers.0.mamba2.dt_bias", vec![4]),
        ("layers.0.mamba2.a_log", vec![4]),
        ("layers.0.mamba2.d", vec![4]),
        ("layers.0.mamba2.norm", vec![8]),
        ("layers.0.mamba2.out_proj", vec![8, 5]),
        ("layers.0.mamba2.conv_bias", vec![20]),
        ("layers.1.norm", vec![5]),
        ("layers.1.attention.q_proj", vec![5, 5]),
        ("layers.1.attention.k_proj", vec![5, 5]),
        ("layers.1.attention.v_proj", vec![5, 5]),
        ("layers.1.attention.o_proj", vec![5, 5]),
        ("layers.2.norm", vec![5]),
        ("layers.2.moe.router", vec![5, 4]),
        ("layers.2.moe.experts_in", vec![5, 12]),
        ("layers.2.moe.experts_out", vec![3, 20]),
        ("layers.2.moe.selection_bias", vec![4]),
        ("layers.2.moe.shared.in", vec![5, 6]),
        ("layers.2.moe.shared.out", vec![6, 5]),
    ]
    .into_iter()
    .map(|(name, shape)| (name.to_string(), shape))
    .collect();
    let runtime = named_component_runtime(&digest).unwrap();
    let build = |config: &ModelArchitectureConfig, weights: BTreeMap<String, Vec<u64>>| {
        build_first_native_graphs_with_runtime(
            &runtime.manager,
            &runtime.capability,
            &runtime.model_config_capability,
            runtime.definition,
            config,
            weights,
            &fixture.identity,
            4,
        )
    };
    let (graphs, _, _) = build(&config, weights.clone()).unwrap();
    for graph in [&graphs.prefill, &graphs.decode] {
        graph.validate(&initial_operator_catalog()).unwrap();
        for operator in ["mamba2-mixer", "attention", "moe-experts"] {
            assert!(
                graph
                    .nodes
                    .values()
                    .any(|node| node.operator.name() == operator),
                "{operator}"
            );
        }
        let slots: Vec<_> = graph
            .edges
            .values()
            .filter_map(|edge| edge.recurrent_state.as_ref())
            .map(|state| state.slot.as_str())
            .collect();
        assert_eq!(slots.iter().filter(|slot| **slot == "conv").count(), 2);
        assert_eq!(slots.iter().filter(|slot| **slot == "ssm").count(), 2);
    }
    let mut invalid = config.clone();
    invalid.rope_layers.as_mut().unwrap()[1] = true;
    assert!(build(&invalid, weights.clone()).is_err());
    let mut missing = weights;
    missing.remove("layers.0.mamba2.conv_bias");
    assert!(build(&config, missing).is_err());
}
