//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

fn sample_config() -> ModelArchitectureConfig {
    ModelArchitectureConfig {
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        head_dim: 64,
        vocab_size: 151936,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_scaling_factor: None,
        rope_frequency_scaling: None,
        hidden_activation: None,
        tie_word_embeddings: true,
        attention_bias: true,
        mixture_of_experts: None,
        layer_layout: None,
        final_logit_softcapping: None,
        per_layer_input_size: None,
        latent_attention: None,
        moe_layers: None,
        expert_routing: None,
        bos_token_id: None,
        eos_token_id: Some(151643),
    }
}

#[test]
fn unbound_instance_is_rejected() {
    let capability = ModelConfigCapability::new();
    let error = capability
        .call("instance-1", "model-config", &[])
        .unwrap_err();
    assert!(matches!(
        error,
        ComponentError::CapabilityCallRejected { .. }
    ));
}

#[test]
fn bound_config_round_trips_through_encoding() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("instance-1", sample_config());
    let result = capability.call("instance-1", "model-config", &[]).unwrap();
    let [ComponentValue::Record(fields)] = result.as_slice() else {
        panic!("expected a single record value, got {result:?}");
    };
    let field = |name: &str| {
        fields
            .iter()
            .find(|(field_name, _)| field_name == name)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| panic!("missing field '{name}'"))
    };
    assert_eq!(field("hidden-size"), ComponentValue::S64(896));
    assert_eq!(field("num-hidden-layers"), ComponentValue::U32(24));
    assert_eq!(field("num-attention-heads"), ComponentValue::U32(14));
    assert_eq!(field("num-key-value-heads"), ComponentValue::U32(2));
    assert_eq!(field("tie-word-embeddings"), ComponentValue::Bool(true));
    assert_eq!(field("attention-bias"), ComponentValue::Bool(true));
    assert_eq!(field("rope-scaling-factor"), ComponentValue::Option(None));
}

#[test]
fn cleared_config_is_rejected_again() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("instance-1", sample_config());
    capability.clear_config("instance-1");
    let error = capability
        .call("instance-1", "model-config", &[])
        .unwrap_err();
    assert!(matches!(
        error,
        ComponentError::CapabilityCallRejected { .. }
    ));
}

#[test]
fn rope_scaling_is_none_without_and_a_llama3_variant_with_a_declared_scheme() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("instance-1", sample_config());
    assert_eq!(
        capability.call("instance-1", "rope-scaling", &[]).unwrap(),
        vec![ComponentValue::Option(None)]
    );
    let mut config = sample_config();
    config.rope_frequency_scaling = Some(RopeFrequencyScaling::Banded(
        crate::operator::RopeBandedFrequencyScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_context_length: 8192,
        },
    ));
    capability.bind_config("instance-2", config);
    let result = capability.call("instance-2", "rope-scaling", &[]).unwrap();
    let [ComponentValue::Option(Some(variant))] = result.as_slice() else {
        panic!("expected some(variant), got {result:?}");
    };
    let ComponentValue::Variant(case, Some(payload)) = variant.as_ref() else {
        panic!("expected a variant with a payload, got {variant:?}");
    };
    assert_eq!(case, "banded");
    assert_eq!(
        **payload,
        ComponentValue::Record(vec![
            ("factor".into(), ComponentValue::F64(32.0)),
            ("low-freq-factor".into(), ComponentValue::F64(1.0)),
            ("high-freq-factor".into(), ComponentValue::F64(4.0)),
            (
                "original-max-position-embeddings".into(),
                ComponentValue::S64(8192)
            ),
        ])
    );
    assert!(
        capability
            .call("instance-2", "rope-scaling", &[ComponentValue::U32(1)])
            .is_err()
    );
}

#[test]
fn mixture_of_experts_is_none_for_a_dense_model_and_a_record_for_a_routed_one() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("dense", sample_config());
    assert_eq!(
        capability.call("dense", "mixture-of-experts", &[]).unwrap(),
        vec![ComponentValue::Option(None)]
    );
    let mut config = sample_config();
    config.mixture_of_experts = Some(MixtureOfExpertsConfig {
        experts: 60,
        experts_per_token: 4,
        expert_intermediate_size: 1408,
        shared_expert_intermediate_size: Some(5632),
        normalize_top_k: false,
    });
    capability.bind_config("routed", config);
    assert_eq!(
        capability
            .call("routed", "mixture-of-experts", &[])
            .unwrap(),
        vec![ComponentValue::Option(Some(Box::new(
            ComponentValue::Record(vec![
                ("experts".into(), ComponentValue::U32(60)),
                ("experts-per-token".into(), ComponentValue::U32(4)),
                ("expert-intermediate-size".into(), ComponentValue::S64(1408)),
                (
                    "shared-expert-intermediate-size".into(),
                    ComponentValue::Option(Some(Box::new(ComponentValue::S64(5632))))
                ),
                ("normalize-top-k".into(), ComponentValue::Bool(false)),
            ])
        )))]
    );
}

fn sample_layer(sliding_window: Option<u32>, kv_source_layer: Option<u32>) -> LayerSpec {
    LayerSpec {
        sliding_window,
        head_dim: 64,
        num_key_value_heads: 2,
        rope_theta: 10_000.0,
        partial_rotary_factor: None,
        value_from_key: false,
        kv_source_layer,
        intermediate_size: 4864,
    }
}

#[test]
fn layer_layout_and_options_are_none_for_a_uniform_model() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("uniform", sample_config());
    assert_eq!(
        capability.call("uniform", "layer-layout", &[]).unwrap(),
        vec![ComponentValue::Option(None)]
    );
    assert_eq!(
        capability.call("uniform", "options", &[]).unwrap(),
        vec![ComponentValue::Record(vec![
            (
                "final-logit-softcapping".into(),
                ComponentValue::Option(None)
            ),
            ("per-layer-input-size".into(), ComponentValue::Option(None)),
        ])]
    );
}

#[test]
fn declared_layer_layout_and_options_must_be_read_per_instance() {
    let capability = ModelConfigCapability::new();
    let mut config = sample_config();
    config.layer_layout = Some(vec![
        sample_layer(Some(32), None);
        config.num_hidden_layers as usize
    ]);
    config.final_logit_softcapping = Some(30.0);
    capability.bind_config("first", config.clone());
    capability.bind_config("second", config);
    assert_eq!(
        capability.unread_declared_features("first"),
        vec!["layer-layout", "options"]
    );
    capability.call("first", "layer-layout", &[]).unwrap();
    assert_eq!(
        capability.unread_declared_features("first"),
        vec!["options"]
    );
    assert_eq!(
        capability.unread_declared_features("second"),
        vec!["layer-layout", "options"]
    );
    capability.call("first", "options", &[]).unwrap();
    assert!(capability.unread_declared_features("first").is_empty());
}

#[test]
fn activation_is_exposed_and_non_silu_requires_a_read() {
    let capability = ModelConfigCapability::new();
    let mut config = sample_config();
    config.hidden_activation = Some("silu".into());
    capability.bind_config("legacy", config.clone());
    assert!(capability.unread_declared_features("legacy").is_empty());
    config.hidden_activation = Some("gelu".into());
    capability.bind_config("configured", config);
    assert_eq!(
        capability.unread_declared_features("configured"),
        vec!["hidden-activation"]
    );
    assert_eq!(
        capability
            .call("configured", "hidden-activation", &[])
            .unwrap(),
        vec![ComponentValue::Option(Some(Box::new(
            ComponentValue::String("gelu".into())
        )))]
    );
    assert!(capability.unread_declared_features("configured").is_empty());
}

#[test]
fn layer_layout_encodes_one_record_per_layer_and_options_their_values() {
    let capability = ModelConfigCapability::new();
    let mut config = sample_config();
    let mut global = sample_layer(None, None);
    global.head_dim = 128;
    global.num_key_value_heads = 1;
    global.rope_theta = 1_000_000.0;
    global.partial_rotary_factor = Some(0.25);
    global.value_from_key = true;
    config.layer_layout = Some(vec![sample_layer(Some(512), None), global]);
    config.final_logit_softcapping = Some(30.0);
    config.per_layer_input_size = Some(256);
    capability.bind_config("layered", config);
    let some = |value: ComponentValue| ComponentValue::Option(Some(Box::new(value)));
    let record = |window: ComponentValue,
                  head: i64,
                  kv_heads: u32,
                  theta: f64,
                  partial: ComponentValue,
                  value_from_key: bool| {
        ComponentValue::Record(vec![
            ("sliding-window".into(), window),
            ("head-dimension".into(), ComponentValue::S64(head)),
            ("num-key-value-heads".into(), ComponentValue::U32(kv_heads)),
            ("rope-theta".into(), ComponentValue::F64(theta)),
            ("partial-rotary-factor".into(), partial),
            (
                "value-from-key".into(),
                ComponentValue::Bool(value_from_key),
            ),
            ("kv-source-layer".into(), ComponentValue::Option(None)),
            ("intermediate-size".into(), ComponentValue::S64(4864)),
        ])
    };
    assert_eq!(
        capability.call("layered", "layer-layout", &[]).unwrap(),
        vec![some(ComponentValue::List(vec![
            record(
                some(ComponentValue::U32(512)),
                64,
                2,
                10_000.0,
                ComponentValue::Option(None),
                false
            ),
            record(
                ComponentValue::Option(None),
                128,
                1,
                1_000_000.0,
                some(ComponentValue::F64(0.25)),
                true
            ),
        ]))]
    );
    assert_eq!(
        capability.call("layered", "options", &[]).unwrap(),
        vec![ComponentValue::Record(vec![
            (
                "final-logit-softcapping".into(),
                some(ComponentValue::F64(30.0))
            ),
            (
                "per-layer-input-size".into(),
                some(ComponentValue::U32(256))
            ),
        ])]
    );
}

#[test]
fn layer_layout_rejects_arguments() {
    let capability = ModelConfigCapability::new();
    capability.bind_config("uniform", sample_config());
    assert!(
        capability
            .call("uniform", "layer-layout", &[ComponentValue::U32(0)])
            .is_err()
    );
}

#[test]
fn layer_layout_validation_refuses_inconsistent_layouts() {
    let mut config = sample_config();
    config.num_hidden_layers = 3;
    config.num_attention_heads = 14;
    config.num_key_value_heads = 2;
    let valid = vec![
        sample_layer(Some(512), None),
        sample_layer(None, None),
        sample_layer(Some(512), Some(0)),
    ];
    config.layer_layout = Some(valid);
    config.validate().expect("a consistent layout validates");

    let refused = |edit: &dyn Fn(&mut ModelArchitectureConfig)| {
        let mut config = config.clone();
        edit(&mut config);
        assert!(config.validate().is_err(), "{:?}", config.layer_layout);
    };
    refused(&|config| {
        config.layer_layout.as_mut().unwrap().pop();
    });
    refused(&|config| config.layer_layout.as_mut().unwrap()[1].head_dim = 0);
    refused(&|config| config.layer_layout.as_mut().unwrap()[1].num_key_value_heads = 3);
    refused(&|config| config.layer_layout.as_mut().unwrap()[0].sliding_window = Some(0));
    refused(&|config| config.layer_layout.as_mut().unwrap()[0].rope_theta = f64::NAN);
    refused(&|config| config.layer_layout.as_mut().unwrap()[0].partial_rotary_factor = Some(1.5));
    // A KV source must be earlier, compute its own K/V, and match geometry.
    refused(&|config| config.layer_layout.as_mut().unwrap()[0].kv_source_layer = Some(0));
    refused(&|config| config.layer_layout.as_mut().unwrap()[1].kv_source_layer = Some(2));
    refused(&|config| {
        let layout = config.layer_layout.as_mut().unwrap();
        layout[1].kv_source_layer = Some(0);
        layout[2].kv_source_layer = Some(1);
    });
    refused(&|config| config.layer_layout.as_mut().unwrap()[2].head_dim = 128);
    refused(&|config| config.final_logit_softcapping = Some(0.0));
    refused(&|config| config.per_layer_input_size = Some(0));
}

/// `express-latent-attention-forms`: a declared `1.7.0` feature stays
/// "unread" until the instance calls its function, and a new binding resets
/// what was read.
#[test]
fn unread_declared_features_track_the_1_7_0_functions_each_instance_calls() {
    let capability = ModelConfigCapability::new();
    let mut config = sample_config();
    assert!(capability.unread_declared_features("a").is_empty());
    capability.bind_config("a", config.clone());
    assert!(capability.unread_declared_features("a").is_empty());

    config.mixture_of_experts = Some(MixtureOfExpertsConfig {
        experts: 4,
        experts_per_token: 2,
        expert_intermediate_size: 8,
        shared_expert_intermediate_size: None,
        normalize_top_k: true,
    });
    config.moe_layers = Some(vec![true; config.num_hidden_layers as usize]);
    config.latent_attention = Some(crate::LatentAttentionConfig {
        q_lora_rank: None,
        kv_lora_rank: 32,
        qk_nope_head_dim: 48,
        qk_rope_head_dim: 16,
        v_head_dim: 48,
    });
    capability.bind_config("a", config.clone());
    assert_eq!(
        capability.unread_declared_features("a"),
        vec!["multi-head-latent-attention", "mixture-of-experts-layers"]
    );
    let latent = capability
        .call("a", "multi-head-latent-attention", &[])
        .unwrap();
    assert!(matches!(&latent[..], [ComponentValue::Option(Some(_))]));
    assert_eq!(
        capability.unread_declared_features("a"),
        vec!["mixture-of-experts-layers"]
    );
    capability
        .call("a", "mixture-of-experts-layers", &[])
        .unwrap();
    assert!(capability.unread_declared_features("a").is_empty());
    // Another instance, and a fresh binding of the same one, start over.
    capability.bind_config("b", config.clone());
    assert_eq!(capability.unread_declared_features("b").len(), 2);
    capability.bind_config("a", config);
    assert_eq!(capability.unread_declared_features("a").len(), 2);
    capability.clear_config("a");
    assert!(capability.unread_declared_features("a").is_empty());
}
