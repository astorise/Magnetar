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
        tie_word_embeddings: true,
        attention_bias: true,
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
