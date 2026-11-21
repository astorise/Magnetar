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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use crate::component::{ComponentError, ComponentValue, HostCapability};
use crate::{
    ExpertGroupScore, ExpertRoutingConfig, ExpertScoring, LatentAttentionConfig, LayerSpec,
    MixtureOfExpertsConfig, ModelArchitectureConfig, RopeFrequencyScaling,
};

const CAPABILITY_NAME: &str = "magnetar:model-component-graph/model-config";

/// Runtime-side [`HostCapability`] backing `model-config`. One instance is
/// registered once on a `ComponentManager`/engine and shared across every
/// Component instance that imports it; per-instance state (the bound
/// configuration) is keyed by the calling instance's own key, exactly like
/// `GraphBuilderCapability`.
#[derive(Default)]
pub struct ModelConfigCapability {
    bound: Mutex<BTreeMap<String, ModelArchitectureConfig>>,
    /// The operations each bound instance has called since its binding
    /// ([`Self::unread_declared_features`]).
    called: Mutex<BTreeMap<String, BTreeSet<String>>>,
}

/// The functions whose feature a graph does not otherwise show:
/// when the bound configuration declares one, a Component must have read
/// it before its graph is accepted.
const FEATURE_FUNCTIONS: [&str; 6] = [
    "layer-layout",
    "options",
    "multi-head-latent-attention",
    "mixture-of-experts-layers",
    "expert-routing",
    "hidden-activation",
];

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
        self.called.lock().unwrap().remove(instance_key);
    }

    /// Drops the bound configuration for `instance_key`, mirroring
    /// `GraphBuilderCapability::clear_session`.
    pub fn clear_config(&self, instance_key: &str) {
        self.bound.lock().unwrap().remove(instance_key);
        self.called.lock().unwrap().remove(instance_key);
    }

    /// The functions carrying features the configuration bound to
    /// `instance_key` declares that the Component has not called since the binding.
    /// A Component compiled against an earlier version cannot read them, so
    /// a graph it built for such a checkpoint would silently leave the
    /// feature out; the Runtime refuses that graph instead
    /// (`express-latent-attention-forms`).
    pub fn unread_declared_features(&self, instance_key: &str) -> Vec<&'static str> {
        let bound = self.bound.lock().unwrap();
        let Some(config) = bound.get(instance_key) else {
            return Vec::new();
        };
        let called = self.called.lock().unwrap();
        let called = called.get(instance_key);
        FEATURE_FUNCTIONS
            .into_iter()
            .filter(|function| {
                let declared = match *function {
                    "layer-layout" => config.layer_layout.is_some(),
                    "options" => {
                        config.final_logit_softcapping.is_some()
                            || config.per_layer_input_size.is_some()
                    }
                    "multi-head-latent-attention" => config.latent_attention.is_some(),
                    "mixture-of-experts-layers" => config.moe_layers.is_some(),
                    "expert-routing" => config.expert_routing.is_some(),
                    // Preserve compatibility with older fixed-SiLU decoders.
                    // This cannot prove that an older Component actually uses
                    // SiLU; other declared activations must be read explicitly.
                    _ => config
                        .hidden_activation
                        .as_deref()
                        .is_some_and(|value| value != "silu"),
                };
                declared && !called.is_some_and(|called| called.contains(*function))
            })
            .collect()
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
        field(
            "attention-bias",
            ComponentValue::Bool(config.attention_bias),
        ),
    ])
}

/// `option<rope-frequency-scaling>` (`1.4.0`).
fn encode_rope_scaling(scaling: Option<&RopeFrequencyScaling>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(scaling.map(|scaling| {
        Box::new(match scaling {
            RopeFrequencyScaling::Banded(banded) => ComponentValue::Variant(
                "banded".into(),
                Some(Box::new(ComponentValue::Record(vec![
                    field("factor", ComponentValue::F64(banded.factor)),
                    field(
                        "low-freq-factor",
                        ComponentValue::F64(banded.low_freq_factor),
                    ),
                    field(
                        "high-freq-factor",
                        ComponentValue::F64(banded.high_freq_factor),
                    ),
                    field(
                        "original-max-position-embeddings",
                        ComponentValue::S64(banded.original_context_length as i64),
                    ),
                ]))),
            ),
            RopeFrequencyScaling::Yarn {
                scaling,
                mscale_all_dim,
            } => ComponentValue::Variant(
                "yarn".into(),
                Some(Box::new(ComponentValue::Record(vec![
                    field("factor", ComponentValue::F64(scaling.factor)),
                    field(
                        "original-max-position-embeddings",
                        ComponentValue::S64(scaling.original_context_length as i64),
                    ),
                    field("beta-fast", ComponentValue::F64(scaling.beta_fast)),
                    field("beta-slow", ComponentValue::F64(scaling.beta_slow)),
                    field("truncate", ComponentValue::Bool(scaling.truncate)),
                    field(
                        "attention-factor",
                        ComponentValue::F64(scaling.attention_factor),
                    ),
                    field(
                        "mscale-all-dim",
                        ComponentValue::Option(
                            mscale_all_dim.map(|value| Box::new(ComponentValue::F64(value))),
                        ),
                    ),
                ]))),
            ),
        })
    }))
}

/// `option<mla-config>` (`1.7.0`).
fn encode_latent_attention(mla: Option<&LatentAttentionConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(mla.map(|mla| {
        Box::new(ComponentValue::Record(vec![
            field(
                "q-lora-rank",
                ComponentValue::Option(
                    mla.q_lora_rank
                        .map(|rank| Box::new(ComponentValue::U32(rank))),
                ),
            ),
            field("kv-lora-rank", ComponentValue::U32(mla.kv_lora_rank)),
            field(
                "qk-nope-head-dim",
                ComponentValue::S64(mla.qk_nope_head_dim as i64),
            ),
            field(
                "qk-rope-head-dim",
                ComponentValue::S64(mla.qk_rope_head_dim as i64),
            ),
            field("v-head-dim", ComponentValue::S64(mla.v_head_dim as i64)),
        ]))
    }))
}

/// `option<list<bool>>` (`1.7.0`).
fn encode_moe_layers(layers: Option<&[bool]>) -> ComponentValue {
    ComponentValue::Option(layers.map(|layers| {
        Box::new(ComponentValue::List(
            layers
                .iter()
                .map(|&moe| ComponentValue::Bool(moe))
                .collect(),
        ))
    }))
}

/// `option<expert-routing-config>` (`1.7.0`).
fn encode_expert_routing(routing: Option<&ExpertRoutingConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(routing.map(|routing| {
        Box::new(ComponentValue::Record(vec![
            field(
                "scoring",
                ComponentValue::Enum(
                    match routing.scoring {
                        ExpertScoring::Softmax => "softmax",
                        ExpertScoring::Sigmoid => "sigmoid",
                    }
                    .into(),
                ),
            ),
            field(
                "groups",
                ComponentValue::Option(routing.groups.map(|groups| {
                    Box::new(ComponentValue::Record(vec![
                        field("count", ComponentValue::U32(groups.count)),
                        field("per-token", ComponentValue::U32(groups.per_token)),
                        field(
                            "score",
                            ComponentValue::Enum(
                                match groups.score {
                                    ExpertGroupScore::Max => "max",
                                    ExpertGroupScore::TopTwoSum => "top-two-sum",
                                }
                                .into(),
                            ),
                        ),
                    ]))
                })),
            ),
            field(
                "routed-scaling-factor",
                ComponentValue::F64(routing.routed_scaling_factor),
            ),
            field(
                "selection-bias",
                ComponentValue::Bool(routing.selection_bias),
            ),
            field(
                "shared-expert-gated",
                ComponentValue::Bool(routing.shared_expert_gated),
            ),
        ]))
    }))
}

/// `option<moe-config>` (`1.5.0`).
fn encode_mixture_of_experts(moe: Option<&MixtureOfExpertsConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(moe.map(|moe| {
        Box::new(ComponentValue::Record(vec![
            field("experts", ComponentValue::U32(moe.experts)),
            field(
                "experts-per-token",
                ComponentValue::U32(moe.experts_per_token),
            ),
            field(
                "expert-intermediate-size",
                ComponentValue::S64(moe.expert_intermediate_size as i64),
            ),
            field(
                "shared-expert-intermediate-size",
                ComponentValue::Option(
                    moe.shared_expert_intermediate_size
                        .map(|size| Box::new(ComponentValue::S64(size as i64))),
                ),
            ),
            field("normalize-top-k", ComponentValue::Bool(moe.normalize_top_k)),
        ]))
    }))
}

/// `option<list<layer-spec>>` (`1.6.0`).
fn encode_layer_layout(layout: Option<&[LayerSpec]>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(layout.map(|layout| {
        Box::new(ComponentValue::List(
            layout
                .iter()
                .map(|layer| {
                    ComponentValue::Record(vec![
                        field(
                            "sliding-window",
                            ComponentValue::Option(
                                layer
                                    .sliding_window
                                    .map(|window| Box::new(ComponentValue::U32(window))),
                            ),
                        ),
                        field("head-dimension", ComponentValue::S64(layer.head_dim as i64)),
                        field(
                            "num-key-value-heads",
                            ComponentValue::U32(layer.num_key_value_heads),
                        ),
                        field("rope-theta", ComponentValue::F64(layer.rope_theta)),
                        field(
                            "partial-rotary-factor",
                            ComponentValue::Option(
                                layer
                                    .partial_rotary_factor
                                    .map(|factor| Box::new(ComponentValue::F64(factor))),
                            ),
                        ),
                        field("value-from-key", ComponentValue::Bool(layer.value_from_key)),
                        field(
                            "kv-source-layer",
                            ComponentValue::Option(
                                layer
                                    .kv_source_layer
                                    .map(|source| Box::new(ComponentValue::U32(source))),
                            ),
                        ),
                        field(
                            "intermediate-size",
                            ComponentValue::S64(layer.intermediate_size as i64),
                        ),
                    ])
                })
                .collect(),
        ))
    }))
}

/// `architecture-options` (`1.6.0`).
fn encode_options(config: &ModelArchitectureConfig) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Record(vec![
        field(
            "final-logit-softcapping",
            ComponentValue::Option(
                config
                    .final_logit_softcapping
                    .map(|cap| Box::new(ComponentValue::F64(cap))),
            ),
        ),
        field(
            "per-layer-input-size",
            ComponentValue::Option(
                config
                    .per_layer_input_size
                    .map(|size| Box::new(ComponentValue::U32(size))),
            ),
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
            "model-config"
            | "rope-scaling"
            | "mixture-of-experts"
            | "layer-layout"
            | "options"
            | "multi-head-latent-attention"
            | "mixture-of-experts-layers"
            | "expert-routing"
            | "hidden-activation" => {
                if !arguments.is_empty() {
                    return Err(rejected(
                        instance_key,
                        format!(
                            "'{operation}' expects 0 argument(s), got {}",
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
                let encoded = match operation {
                    "model-config" => encode_architecture_config(config),
                    "rope-scaling" => encode_rope_scaling(config.rope_frequency_scaling.as_ref()),
                    "mixture-of-experts" => {
                        encode_mixture_of_experts(config.mixture_of_experts.as_ref())
                    }
                    "layer-layout" => encode_layer_layout(config.layer_layout.as_deref()),
                    "multi-head-latent-attention" => {
                        encode_latent_attention(config.latent_attention.as_ref())
                    }
                    "mixture-of-experts-layers" => encode_moe_layers(config.moe_layers.as_deref()),
                    "expert-routing" => encode_expert_routing(config.expert_routing.as_ref()),
                    "hidden-activation" => ComponentValue::Option(
                        config
                            .hidden_activation
                            .as_ref()
                            .map(|value| Box::new(ComponentValue::String(value.clone()))),
                    ),
                    _ => encode_options(config),
                };
                drop(bound);
                self.called
                    .lock()
                    .unwrap()
                    .entry(instance_key.to_string())
                    .or_default()
                    .insert(operation.to_string());
                Ok(vec![encoded])
            }
            other => Err(rejected(
                instance_key,
                format!("unknown model-config operation '{other}'"),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
