#![cfg(not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine")))]

use magnetar_runtime::{
    ComponentDigest, E2eConformanceError, EncoderGraphConfig, FirstNativeEncoderGraphRequest,
    ModelArchitectureConfig, ModelComponentId, ModelComponentIdentity,
    ModelComponentImplementationKind, ModelComponentVersion,
    build_first_native_encoder_graph_from_named_component,
};
use std::collections::BTreeMap;

#[test]
fn encoder_graph_fails_closed_without_a_component_engine() {
    let request = FirstNativeEncoderGraphRequest {
        architecture: ModelArchitectureConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling_factor: None,
            rope_frequency_scaling: None,
            rope_layers: None,
            hidden_activation: Some("gelu".into()),
            tie_word_embeddings: false,
            attention_bias: false,
            mixture_of_experts: None,
            layer_layout: None,
            final_logit_softcapping: None,
            per_layer_input_size: None,
            latent_attention: None,
            moe_layers: None,
            expert_routing: None,
            bos_token_id: None,
            eos_token_id: None,
        },
        encoder: EncoderGraphConfig {
            norm_epsilon: 1e-5,
            max_positions: 16,
            norm_bias: false,
            attention_bias: false,
            mlp_bias: false,
            layers: vec![],
        },
        weight_shapes: BTreeMap::new(),
        identity: ModelComponentIdentity::new(
            ModelComponentId::new("encoder-test").unwrap(),
            ModelComponentVersion::new(1, 0, 0),
            ModelComponentImplementationKind::WebAssemblyComponent,
        ),
        sequence_length: 3,
    };
    let result = build_first_native_encoder_graph_from_named_component(
        &ComponentDigest::sha256(b"unavailable encoder"),
        &request,
    );
    assert!(matches!(
        result,
        Err(E2eConformanceError::ModelComponentFailed { .. })
    ));
}
