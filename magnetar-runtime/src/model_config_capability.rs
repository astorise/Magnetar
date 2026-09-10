//! Runtime-side implementation of the `magnetar:model-component-graph/
//! model-config` WIT Capability (`model-component-graph-contract`,
//! `implement-production-qwen-model-loading` Decision 6): a
//! [`crate::component::HostCapability`] a configurable Model Component
//! calls into to obtain the Runtime-authorized normalized architecture
//! configuration bound to the active Model Instance/graph-building session,
//! instead of hard-coding fixture dimensions.
//!
//! Deliberately generic, like [`crate::graph_builder_capability::
//! GraphBuilderCapability`]: nothing here names Qwen. A caller binds
//! [`crate::ModelArchitectureConfig`] for a session key before invoking the
//! Component (mirroring `GraphBuilderCapability::prepare_session`), and this
//! capability only ever hands back whatever was bound -- it never derives or
//! infers a configuration on its own.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::ModelArchitectureConfig;
use crate::component::{ComponentError, ComponentValue, HostCapability};

const CAPABILITY_NAME: &str = "magnetar:model-component-graph/model-config";

/// Runtime-side [`HostCapability`] backing `model-config`. One instance is
/// registered once on a `ComponentManager`/engine and shared across every
/// Component instance that imports it; per-instance state (the bound
/// configuration) is keyed by the calling instance's own key, exactly like
/// `GraphBuilderCapability`.
#[derive(Default)]
pub struct ModelConfigCapability {
    bound: Mutex<BTreeMap<String, ModelArchitectureConfig>>,
}

impl ModelConfigCapability {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds `config` for `instance_key` before the caller invokes the
    /// Component's `build-prefill-graph`/`build-decode-graph` export.
    /// Replaces any prior binding for this key.
    pub fn bind_config(&self, instance_key: &str, config: ModelArchitectureConfig) {
        self.bound
            .lock()
            .unwrap()
            .insert(instance_key.to_string(), config);
    }

    /// Drops the bound configuration for `instance_key`, mirroring
    /// `GraphBuilderCapability::clear_session`.
    pub fn clear_config(&self, instance_key: &str) {
        self.bound.lock().unwrap().remove(instance_key);
    }
}

fn encode_architecture_config(config: &ModelArchitectureConfig) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Record(vec![
        field(
            "hidden-size",
            ComponentValue::S64(config.hidden_size as i64),
        ),
        field(
            "intermediate-size",
            ComponentValue::S64(config.intermediate_size as i64),
        ),
        field(
            "num-hidden-layers",
            ComponentValue::U32(config.num_hidden_layers),
        ),
        field(
            "num-attention-heads",
            ComponentValue::U32(config.num_attention_heads),
        ),
        field(
            "num-key-value-heads",
            ComponentValue::U32(config.num_key_value_heads),
        ),
        field(
            "head-dimension",
            ComponentValue::S64(config.head_dim as i64),
        ),
        field(
            "vocabulary-size",
            ComponentValue::S64(config.vocab_size as i64),
        ),
        field(
            "rms-norm-eps",
            ComponentValue::F64(config.rms_norm_eps as f64),
        ),
        field("rope-theta", ComponentValue::F64(config.rope_theta)),
        field(
            "rope-scaling-factor",
            ComponentValue::Option(
                config
                    .rope_scaling_factor
                    .map(|value| Box::new(ComponentValue::F64(value as f64))),
            ),
        ),
        field(
            "tie-word-embeddings",
            ComponentValue::Bool(config.tie_word_embeddings),
        ),
    ])
}

fn rejected(instance_key: &str, message: impl Into<String>) -> ComponentError {
    ComponentError::CapabilityCallRejected {
        capability: CAPABILITY_NAME.to_string(),
        instance_key: instance_key.to_string(),
        message: message.into(),
    }
}

impl HostCapability for ModelConfigCapability {
    fn call(
        &self,
        instance_key: &str,
        operation: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        match operation {
            "model-config" => {
                if !arguments.is_empty() {
                    return Err(rejected(
                        instance_key,
                        format!(
                            "'model-config' expects 0 argument(s), got {}",
                            arguments.len()
                        ),
                    ));
                }
                let bound = self.bound.lock().unwrap();
                let config = bound.get(instance_key).ok_or_else(|| {
                    rejected(
                        instance_key,
                        "no model architecture configuration bound for this instance",
                    )
                })?;
                Ok(vec![encode_architecture_config(config)])
            }
            other => Err(rejected(
                instance_key,
                format!("unknown model-config operation '{other}'"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
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
}
