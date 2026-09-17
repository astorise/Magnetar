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
