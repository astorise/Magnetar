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
    ExpertGroupScore, ExpertNormalization, ExpertRoutingConfig, ExpertScoring, HybridDecoderBlock,
    HyperConnectionConfig, LatentAttentionConfig, LayerSpec, LinearAttentionConfig,
    Mamba2HybridConfig, MixtureOfExpertsConfig, ModelArchitectureConfig, NgramEmbeddingConfig,
    RopeFrequencyScaling, TokenSelectionConfig,
};

const CAPABILITY_NAME: &str = "magnetar:model-component-graph/model-config";

/// Ordered bidirectional encoder layer configuration supplied by ingestion.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderLayerConfig {
    /// Inclusive symmetric local radius; `None` selects global attention.
    pub local_radius: Option<u32>,
    pub rope_theta: f64,
    pub attention_normalization: bool,
}

/// Encoder-specific configuration for a graph-building session.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderGraphConfig {
    pub norm_epsilon: f64,
    pub max_positions: u64,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub layers: Vec<EncoderLayerConfig>,
}

struct BoundConfig {
    architecture: ModelArchitectureConfig,
    encoder: Option<EncoderGraphConfig>,
}

/// Runtime-side [`HostCapability`] backing `model-config`. One instance is
/// registered once on a `ComponentManager`/engine and shared across every
/// Component instance that imports it; per-instance state (the bound
/// configuration) is keyed by the calling instance's own key, exactly like
/// `GraphBuilderCapability`.
#[derive(Default)]
pub struct ModelConfigCapability {
    bound: Mutex<BTreeMap<String, BoundConfig>>,
    /// The operations each bound instance has called since its binding
    /// ([`Self::unread_declared_features`]).
    called: Mutex<BTreeMap<String, BTreeSet<String>>>,
}

/// The functions whose feature a graph does not otherwise show:
/// when the bound configuration declares one, a Component must have read
/// it before its graph is accepted. Activation must always be read, so a
/// Component can distinguish an unknown activation from a supported one.
const FEATURE_FUNCTIONS: [&str; 17] = [
    "mamba2-hybrid",
    "linear-attention",
    "decoder-variants",
    "hyper-connections",
    "ngram-embeddings",
    "token-selection",
    "linear-attention-gate",
    "layer-layout",
    "options",
    "multi-head-latent-attention",
    "mixture-of-experts-layers",
    "expert-routing",
    "hidden-activation",
    "rope-layers",
    "expert-normalization",
    "latent-norm-epsilon",
    "encoder-config",
];

impl ModelConfigCapability {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds `config` for `instance_key` before the caller invokes the
    /// Component's `build-prefill-graph`/`build-decode-graph` export.
    /// Replaces any prior binding for this key.
    pub fn bind_config(&self, instance_key: &str, config: ModelArchitectureConfig) {
        self.bound.lock().unwrap().insert(
            instance_key.to_string(),
            BoundConfig {
                architecture: config,
                encoder: None,
            },
        );
        self.called.lock().unwrap().remove(instance_key);
    }

    /// Binds an encoder profile to an already-bound architecture. Rebinding
    /// the architecture drops this profile and all feature-read evidence.
    pub fn bind_encoder_config(
        &self,
        instance_key: &str,
        encoder: EncoderGraphConfig,
    ) -> Result<(), ComponentError> {
        let mut bound = self.bound.lock().unwrap();
        let entry = bound.get_mut(instance_key).ok_or_else(|| {
            rejected(
                instance_key,
                "no model architecture configuration bound for this instance",
            )
        })?;
        let architecture = &entry.architecture;
        if let Err(error) = architecture.validate() {
            let message = error.to_string();
            bound.remove(instance_key);
            self.called.lock().unwrap().remove(instance_key);
            return Err(rejected(instance_key, message));
        }
        let epsilon = encoder.norm_epsilon as f32;
        if encoder.layers.len() != architecture.num_hidden_layers as usize
            || encoder.max_positions == 0
            || encoder.max_positions > (1 << 24)
            || architecture.vocab_size > (1 << 24)
            || architecture.num_key_value_heads != architecture.num_attention_heads
            || architecture
                .head_dim
                .checked_mul(architecture.num_attention_heads as u64)
                != Some(architecture.hidden_size)
            || !architecture.head_dim.is_multiple_of(2)
            || !encoder.norm_epsilon.is_finite()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || encoder.attention_bias != architecture.attention_bias
            || encoder.layers.iter().any(|layer| {
                layer.local_radius == Some(0)
                    || !layer.rope_theta.is_finite()
                    || layer.rope_theta <= 0.0
                    || !(layer.rope_theta as f32).is_finite()
                    || (layer.rope_theta as f32) <= 0.0
            })
        {
            bound.remove(instance_key);
            self.called.lock().unwrap().remove(instance_key);
            return Err(rejected(
                instance_key,
                "invalid or incompatible encoder configuration",
            ));
        }
        entry.encoder = Some(encoder);
        if let Some(called) = self.called.lock().unwrap().get_mut(instance_key) {
            called.remove("encoder-config");
        }
        Ok(())
    }

    /// Drops the bound configuration for `instance_key`, mirroring
    /// `GraphBuilderCapability::clear_session`.
    pub fn clear_config(&self, instance_key: &str) {
        self.bound.lock().unwrap().remove(instance_key);
        self.called.lock().unwrap().remove(instance_key);
    }

    /// The functions carrying features the configuration bound to
    /// `instance_key` declares that the Component has not called since the
    /// binding, plus `hidden-activation` even when its value is unknown.
    /// A Component compiled against an earlier version cannot read them, so
    /// a graph it built for such a checkpoint would silently leave the
    /// feature out; the Runtime refuses that graph instead
    /// (`express-latent-attention-forms`).
    pub fn unread_declared_features(&self, instance_key: &str) -> Vec<&'static str> {
        let bound = self.bound.lock().unwrap();
        let Some(entry) = bound.get(instance_key) else {
            return Vec::new();
        };
        let config = &entry.architecture;
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
                    "expert-normalization" => config
                        .expert_routing
                        .is_some_and(|routing| routing.normalization.is_some()),
                    "latent-norm-epsilon" => config
                        .latent_attention
                        .is_some_and(|mla| mla.norm_eps.is_some()),
                    "rope-layers" => config.rope_layers.is_some(),
                    "encoder-config" => entry.encoder.is_some(),
                    "mamba2-hybrid" => config.decoder_extensions.mamba2_hybrid.is_some(),
                    "linear-attention" => config.linear_attention.is_some(),
                    "decoder-variants" => decoder_variants_declared(config),
                    "hyper-connections" => config.decoder_extensions.hyper_connections.is_some(),
                    "ngram-embeddings" => config.decoder_extensions.ngram_embeddings.is_some(),
                    "token-selection" => config.decoder_extensions.token_selection.is_some(),
                    "linear-attention-gate" => config.decoder_extensions.sigmoid_linear_gate,
                    // Even an absent activation is meaningful: a fixed-activation
                    // decoder must see and reject it instead of guessing.
                    "hidden-activation" => true,
                    _ => unreachable!("every tracked model-config function is handled"),
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

/// `option<linear-attention-config>` (`1.13.0`).
fn encode_linear_attention(linear: Option<&LinearAttentionConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(linear.map(|linear| {
        Box::new(ComponentValue::Record(vec![
            field(
                "layers",
                ComponentValue::List(
                    linear
                        .layers
                        .iter()
                        .map(|&l| ComponentValue::Bool(l))
                        .collect(),
                ),
            ),
            field("key-heads", ComponentValue::U32(linear.key_heads)),
            field("value-heads", ComponentValue::U32(linear.value_heads)),
            field(
                "key-head-dimension",
                ComponentValue::S64(linear.key_head_dim as i64),
            ),
            field(
                "value-head-dimension",
                ComponentValue::S64(linear.value_head_dim as i64),
            ),
            field("conv-kernel", ComponentValue::U32(linear.conv_kernel)),
        ]))
    }))
}

/// `option<mamba2-hybrid-config>` (`1.15.0`).
fn encode_mamba2_hybrid(hybrid: Option<&Mamba2HybridConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(hybrid.map(|hybrid| {
        Box::new(ComponentValue::Record(vec![
            field(
                "layers",
                ComponentValue::List(
                    hybrid
                        .layers
                        .iter()
                        .map(|layer| {
                            ComponentValue::Enum(
                                match layer {
                                    HybridDecoderBlock::Mamba2 => "mamba2",
                                    HybridDecoderBlock::Attention => "attention",
                                    HybridDecoderBlock::DenseMlp => "dense-mlp",
                                    HybridDecoderBlock::RoutedMoe => "routed-moe",
                                }
                                .into(),
                            )
                        })
                        .collect(),
                ),
            ),
            field("heads", ComponentValue::U32(hybrid.heads)),
            field(
                "head-dimension",
                ComponentValue::S64(hybrid.head_dim as i64),
            ),
            field("groups", ComponentValue::U32(hybrid.groups)),
            field(
                "state-dimension",
                ComponentValue::S64(hybrid.state_dim as i64),
            ),
            field("conv-kernel", ComponentValue::U32(hybrid.conv_kernel)),
            field("epsilon", ComponentValue::F64(hybrid.epsilon)),
            field("dt-min", ComponentValue::F64(hybrid.dt_min)),
            field(
                "dt-max",
                ComponentValue::Option(hybrid.dt_max.map(|max| Box::new(ComponentValue::F64(max)))),
            ),
            field("conv-bias", ComponentValue::Bool(hybrid.conv_bias)),
            field("linear-bias", ComponentValue::Bool(hybrid.linear_bias)),
        ]))
    }))
}

/// Whether the configuration declares a `decoder-variants` form.
fn decoder_variants_declared(config: &ModelArchitectureConfig) -> bool {
    config
        .norm_weight_offset
        .is_some_and(|offset| offset != 0.0)
        || config.attention_output_gate
}

/// `option<decoder-variants>` (`1.13.0`).
fn encode_decoder_variants(config: &ModelArchitectureConfig) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(decoder_variants_declared(config).then(|| {
        Box::new(ComponentValue::Record(vec![
            field(
                "norm-weight-offset",
                ComponentValue::F64(config.norm_weight_offset.unwrap_or(0.0)),
            ),
            field(
                "attention-output-gate",
                ComponentValue::Bool(config.attention_output_gate),
            ),
        ]))
    }))
}

/// `option<hyper-connection-config>` (`1.14.0`).
fn encode_hyper_connections(connections: Option<HyperConnectionConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(connections.map(|connections| {
        Box::new(ComponentValue::Record(vec![
            field("streams", ComponentValue::U32(connections.streams)),
            field("lowrank", ComponentValue::U32(connections.lowrank)),
        ]))
    }))
}

/// `option<ngram-embedding-config>` (`1.14.0`).
fn encode_ngram_embeddings(ngram: Option<&NgramEmbeddingConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(ngram.map(|ngram| {
        Box::new(ComponentValue::Record(vec![
            field(
                "layers",
                ComponentValue::List(
                    ngram
                        .layers
                        .iter()
                        .map(|&layer| ComponentValue::Bool(layer))
                        .collect(),
                ),
            ),
            field(
                "embedding-dim",
                ComponentValue::S64(ngram.embedding_dim as i64),
            ),
            field("conv-kernel", ComponentValue::U32(ngram.conv_kernel)),
            field("ngram-size", ComponentValue::U32(ngram.ngram_size)),
            field(
                "heads-per-ngram",
                ComponentValue::U32(ngram.heads_per_ngram),
            ),
            field(
                "vocabulary-base",
                ComponentValue::S64(ngram.vocabulary_base as i64),
            ),
            field(
                "vocabulary-multiple",
                ComponentValue::U32(ngram.vocabulary_multiple),
            ),
            field("seed", ComponentValue::S64(ngram.seed as i64)),
            field("eos-token", ComponentValue::U32(ngram.eos_token)),
        ]))
    }))
}

/// `option<token-selection-config>` (`1.14.0`).
fn encode_token_selection(selection: Option<TokenSelectionConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(selection.map(|selection| {
        Box::new(ComponentValue::Record(vec![
            field("heads", ComponentValue::U32(selection.heads)),
            field("key-heads", ComponentValue::U32(selection.key_heads)),
            field(
                "head-dimension",
                ComponentValue::S64(selection.head_dim as i64),
            ),
            field("budget", ComponentValue::U32(selection.budget)),
            field(
                "compress-ratio",
                ComponentValue::U32(selection.compress_ratio),
            ),
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

fn encode_encoder_config(config: Option<&EncoderGraphConfig>) -> ComponentValue {
    let field = |name: &str, value: ComponentValue| (name.to_string(), value);
    ComponentValue::Option(config.map(|config| {
        Box::new(ComponentValue::Record(vec![
            field("norm-epsilon", ComponentValue::F64(config.norm_epsilon)),
            field(
                "max-positions",
                ComponentValue::S64(config.max_positions as i64),
            ),
            field("norm-bias", ComponentValue::Bool(config.norm_bias)),
            field(
                "attention-bias",
                ComponentValue::Bool(config.attention_bias),
            ),
            field("mlp-bias", ComponentValue::Bool(config.mlp_bias)),
            field(
                "layers",
                ComponentValue::List(
                    config
                        .layers
                        .iter()
                        .map(|layer| {
                            ComponentValue::Record(vec![
                                field(
                                    "local-radius",
                                    ComponentValue::Option(
                                        layer
                                            .local_radius
                                            .map(|radius| Box::new(ComponentValue::U32(radius))),
                                    ),
                                ),
                                field("rope-theta", ComponentValue::F64(layer.rope_theta)),
                                field(
                                    "attention-normalization",
                                    ComponentValue::Bool(layer.attention_normalization),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ]))
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
            | "expert-normalization"
            | "latent-norm-epsilon"
            | "encoder-config"
            | "mamba2-hybrid"
            | "linear-attention"
            | "decoder-variants"
            | "hyper-connections"
            | "ngram-embeddings"
            | "token-selection"
            | "linear-attention-gate"
            | "rope-layers"
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
                let entry = bound.get(instance_key).ok_or_else(|| {
                    rejected(
                        instance_key,
                        "no model architecture configuration bound for this instance",
                    )
                })?;
                let config = &entry.architecture;
                let encoded = match operation {
                    "model-config" => encode_architecture_config(config),
                    "rope-scaling" => encode_rope_scaling(config.rope_frequency_scaling.as_ref()),
                    "mixture-of-experts" => {
                        encode_mixture_of_experts(config.mixture_of_experts.as_ref())
                    }
                    "layer-layout" => encode_layer_layout(config.layer_layout.as_deref()),
                    "encoder-config" => encode_encoder_config(entry.encoder.as_ref()),
                    "mamba2-hybrid" => {
                        encode_mamba2_hybrid(config.decoder_extensions.mamba2_hybrid.as_ref())
                    }
                    "linear-attention" => encode_linear_attention(config.linear_attention.as_ref()),
                    "decoder-variants" => encode_decoder_variants(config),
                    "hyper-connections" => {
                        encode_hyper_connections(config.decoder_extensions.hyper_connections)
                    }
                    "ngram-embeddings" => {
                        encode_ngram_embeddings(config.decoder_extensions.ngram_embeddings.as_ref())
                    }
                    "token-selection" => {
                        encode_token_selection(config.decoder_extensions.token_selection)
                    }
                    "linear-attention-gate" => ComponentValue::Option(
                        config
                            .decoder_extensions
                            .sigmoid_linear_gate
                            .then(|| Box::new(ComponentValue::Enum("sigmoid".into()))),
                    ),
                    "multi-head-latent-attention" => {
                        encode_latent_attention(config.latent_attention.as_ref())
                    }
                    "mixture-of-experts-layers" => encode_moe_layers(config.moe_layers.as_deref()),
                    "rope-layers" => encode_moe_layers(config.rope_layers.as_deref()),
                    "expert-routing" => encode_expert_routing(config.expert_routing.as_ref()),
                    "expert-normalization" => ComponentValue::Option(
                        config
                            .expert_routing
                            .and_then(|routing| routing.normalization)
                            .map(|value| {
                                Box::new(ComponentValue::Enum(
                                    match value {
                                        ExpertNormalization::Never => "never",
                                        ExpertNormalization::MultipleExperts => "multiple-experts",
                                        ExpertNormalization::Always => "always",
                                    }
                                    .into(),
                                ))
                            }),
                    ),
                    "latent-norm-epsilon" => ComponentValue::Option(
                        config
                            .latent_attention
                            .and_then(|mla| mla.norm_eps)
                            .map(|value| Box::new(ComponentValue::F64(value))),
                    ),
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
