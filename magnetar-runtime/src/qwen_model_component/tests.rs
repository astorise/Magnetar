//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{
    AdapterSetId, FallbackClass, GenerationModelReference, MemoryManager, ModelArchitecture,
    ModelArtifactId, ModelArtifactKind, ModelArtifactSource, ModelDigest, ModelInstanceDefinition,
    ModelLoadingCoordinator, ModelLoadingRequest, ModelLoadingRequestId, ModelName,
    ModelQuantizationPolicy, ModelRevision, ModelTrustDecision, ModelTrustStatus, ResourceAffinity,
    TokenizerId,
};
use crate::{
    ExecutionGraphId, ExecutionGraphProducer, GraphKvCacheBehavior, GraphKvCacheMetadata,
    GraphModelCompatibility, GraphProductionResult, LayoutDescriptor, ShapeDescriptor,
    TensorAliasing, default_graph_catalog, validate_first_scope_graph,
};

fn f32_edge(id: impl Into<String>, dims: Vec<u64>) -> TensorEdge {
    let id = TensorEdgeId::new(id);
    TensorEdge::new(
        id,
        TensorDescriptor::new(
            ShapeDescriptor::new(dims),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    )
}
fn token_id_edge(id: impl Into<String>, dims: Vec<u64>) -> TensorEdge {
    let id = TensorEdgeId::new(id);
    TensorEdge::new(
        id,
        TensorDescriptor::new(
            ShapeDescriptor::new(dims),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    )
}
/// Build the `weight.lm_head` edge. When the baseline uses tied embeddings,
/// this logical tensor is declared with [`TensorAliasing::MayAlias`] pointing
/// at `weight.token_embedding`, recording that it shares storage rather than
/// silently duplicating it; when untied it is an independent tensor.
fn qwen_lm_head_weight_edge(config: &QwenConfig) -> TensorEdge {
    let a = &config.architecture;
    let mut edge = f32_edge("weight.lm_head", vec![a.hidden_size, a.vocabulary_size]);
    if config.tied_embeddings {
        edge.aliasing = TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"));
    }
    edge
}
/// Build a Qwen prefill or decode Execution Graph: embedding, `layer_count`
/// repeated pre-norm decoder layers (RMSNorm, QKV matmul, RoPE, attention,
/// output projection, residual-add, RMSNorm, gated MLP, residual-add), a
/// final RMSNorm, and an `lm_head` logits projection. Every node uses a
/// required-now Operator.
///
/// Test-oracle only (Correctif 9 / `reach-architecture-freeze-1` task 12.6):
/// production first-native generation never calls this -- the real Qwen
/// WASM Component is the sole production source of graph semantics. This
/// Rust implementation survives only to be compared against the real
/// Component's output in tests, proving the two agree numerically, and as
/// a conformance fixture for this module's own tests.
pub fn qwen_build_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    phase: ExecutionGraphPhase,
    sequence_length: u64,
    kv_cache_enabled: bool,
    position_offset: u64,
) -> Result<ExecutionGraph, QwenComponentError> {
    if sequence_length == 0 {
        return Err(QwenComponentError::GraphProductionFailed {
            reason: "sequence length must be positive".into(),
        });
    }
    let a = &config.architecture;
    let q_dim = a.attention_head_count * a.head_dimension;
    let kv_dim = a.kv_head_count * a.head_dimension;

    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new(format!(
            "qwen-{phase:?}-{}-{}",
            identity.id, identity.version.0
        )),
        phase,
    )
    .with_producer(ExecutionGraphProducer::ModelComponent {
        component_id: identity.id.as_str().into(),
    });
    graph.model = GraphModelCompatibility {
        model_instance_id: None,
        architecture: Some(QWEN_ARCHITECTURE_FAMILY.into()),
        tokenizer_dependency: None,
    };
    graph.fingerprint = Some(qwen_component_compatibility_key(identity));

    graph = graph
        .with_edge(token_id_edge("input.token_ids", vec![sequence_length]))
        .with_edge(f32_edge(
            "weight.token_embedding",
            vec![a.vocabulary_size, a.hidden_size],
        ))
        .with_edge(f32_edge("hidden.0", vec![sequence_length, a.hidden_size]))
        .with_node(
            op_node("embedding", "embedding", OperatorFamily::Tensor)
                .with_input(TensorEdgeId::new("input.token_ids"))
                .with_input(TensorEdgeId::new("weight.token_embedding"))
                .with_output(TensorEdgeId::new("hidden.0")),
        );

    let mut hidden_edge = "hidden.0".to_string();
    let kv_behavior = match phase {
        ExecutionGraphPhase::Decode => GraphKvCacheBehavior::Append,
        _ => GraphKvCacheBehavior::Output,
    };

    for layer in 0..a.layer_count {
        let prefix = format!("layer{layer}");
        let residual_in = hidden_edge.clone();

        let normed = format!("{prefix}.normed");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.input_norm"),
                vec![a.hidden_size],
            ))
            .with_edge(f32_edge(
                normed.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.input_norm"),
                    "rmsnorm",
                    OperatorFamily::Normalization,
                )
                .with_input(TensorEdgeId::new(residual_in.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.input_norm"
                )))
                .with_output(TensorEdgeId::new(normed.clone()))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
            );

        let q = format!("{prefix}.q");
        let k = format!("{prefix}.k");
        let v = format!("{prefix}.v");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.self_attn.q_proj"),
                vec![a.hidden_size, q_dim],
            ))
            .with_edge(f32_edge(q.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.q_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.self_attn.q_proj"
                )))
                .with_output(TensorEdgeId::new(q.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.self_attn.k_proj"),
                vec![a.hidden_size, kv_dim],
            ))
            .with_edge(f32_edge(k.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.k_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.self_attn.k_proj"
                )))
                .with_output(TensorEdgeId::new(k.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.self_attn.v_proj"),
                vec![a.hidden_size, kv_dim],
            ))
            .with_edge(f32_edge(v.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.v_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(normed.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.self_attn.v_proj"
                )))
                .with_output(TensorEdgeId::new(v.clone())),
            );

        let q_rope = format!("{prefix}.q_rope");
        let k_rope = format!("{prefix}.k_rope");
        graph = graph
            .with_edge(f32_edge(q_rope.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.rope_q"),
                    "rope",
                    OperatorFamily::PositionEncoding,
                )
                .with_input(TensorEdgeId::new(q.clone()))
                .with_output(TensorEdgeId::new(q_rope.clone()))
                .with_attribute("base", OperatorAttributeValue::Float(config.rope.base))
                .with_attribute(
                    "dimension",
                    OperatorAttributeValue::Integer(config.rope.dimension as i64),
                )
                .with_attribute(
                    "position_mode",
                    OperatorAttributeValue::String(config.rope.position_mode.as_str().into()),
                )
                .with_attribute(
                    "position_offset",
                    OperatorAttributeValue::Integer(position_offset as i64),
                )
                // `make-first-native-cuda-hot-path-device-resident`:
                // explicit graph data, not inferred by Runtime from this
                // node's id -- Q always rotates `attention_head_count`
                // independent head blocks.
                .with_attribute(
                    "head_count",
                    OperatorAttributeValue::Integer(a.attention_head_count as i64),
                ),
            )
            .with_edge(f32_edge(k_rope.clone(), vec![sequence_length, kv_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.rope_k"),
                    "rope",
                    OperatorFamily::PositionEncoding,
                )
                .with_input(TensorEdgeId::new(k.clone()))
                .with_output(TensorEdgeId::new(k_rope.clone()))
                .with_attribute("base", OperatorAttributeValue::Float(config.rope.base))
                .with_attribute(
                    "dimension",
                    OperatorAttributeValue::Integer(config.rope.dimension as i64),
                )
                .with_attribute(
                    "position_mode",
                    OperatorAttributeValue::String(config.rope.position_mode.as_str().into()),
                )
                .with_attribute(
                    "position_offset",
                    OperatorAttributeValue::Integer(position_offset as i64),
                )
                // K rotates `kv_head_count` independent head blocks --
                // may differ from Q's under grouped-query/multi-query
                // attention (`kv_head_count < attention_head_count`).
                .with_attribute(
                    "head_count",
                    OperatorAttributeValue::Integer(a.kv_head_count as i64),
                ),
            );

        if kv_cache_enabled {
            let cache_metadata = |cache_role: &str| GraphKvCacheMetadata {
                cache_id: format!("qwen.{prefix}.{cache_role}"),
                behavior: kv_behavior.clone(),
                paged: false,
                compatibility_key: qwen_component_compatibility_key(identity),
            };
            if let Some(edge) = graph.edges.get_mut(&TensorEdgeId::new(k_rope.clone())) {
                edge.kv_cache = Some(cache_metadata("k"));
            }
            if let Some(edge) = graph.edges.get_mut(&TensorEdgeId::new(v.clone())) {
                edge.kv_cache = Some(cache_metadata("v"));
            }
        }

        let attn_out = format!("{prefix}.attn_out");
        graph = graph
            .with_edge(f32_edge(attn_out.clone(), vec![sequence_length, q_dim]))
            .with_node(
                op_node(
                    format!("{prefix}.attention"),
                    "attention",
                    OperatorFamily::Attention,
                )
                .with_input(TensorEdgeId::new(q_rope.clone()))
                .with_input(TensorEdgeId::new(k_rope.clone()))
                .with_input(TensorEdgeId::new(v.clone()))
                .with_output(TensorEdgeId::new(attn_out.clone()))
                .with_attribute("causal", OperatorAttributeValue::Boolean(true))
                .with_attribute(
                    "head_count",
                    OperatorAttributeValue::Integer(a.attention_head_count as i64),
                )
                .with_attribute(
                    "kv_head_count",
                    OperatorAttributeValue::Integer(a.kv_head_count as i64),
                )
                .with_attribute(
                    "head_dimension",
                    OperatorAttributeValue::Integer(a.head_dimension as i64),
                )
                .with_attribute(
                    "attention_mask_kind",
                    OperatorAttributeValue::String("causal".into()),
                ),
            );

        let attn_proj = format!("{prefix}.attn_proj");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.self_attn.o_proj"),
                vec![q_dim, a.hidden_size],
            ))
            .with_edge(f32_edge(
                attn_proj.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.o_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(attn_out.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.self_attn.o_proj"
                )))
                .with_output(TensorEdgeId::new(attn_proj.clone())),
            );

        let post_attn = format!("{prefix}.post_attn");
        graph = graph
            .with_edge(f32_edge(
                post_attn.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.residual1"),
                    "residual-add",
                    OperatorFamily::Tensor,
                )
                .with_input(TensorEdgeId::new(residual_in.clone()))
                .with_input(TensorEdgeId::new(attn_proj.clone()))
                .with_output(TensorEdgeId::new(post_attn.clone())),
            );

        let mlp_normed = format!("{prefix}.mlp_normed");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.post_attn_norm"),
                vec![a.hidden_size],
            ))
            .with_edge(f32_edge(
                mlp_normed.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.post_attn_norm"),
                    "rmsnorm",
                    OperatorFamily::Normalization,
                )
                .with_input(TensorEdgeId::new(post_attn.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.post_attn_norm"
                )))
                .with_output(TensorEdgeId::new(mlp_normed.clone()))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
            );

        let gate_up = format!("{prefix}.gate_up");
        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        let activated = format!("{prefix}.activated");
        let mlp_hidden = format!("{prefix}.mlp_hidden");
        let mlp_out = format!("{prefix}.mlp_out");
        graph = graph
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.mlp.gate_up_proj"),
                vec![a.hidden_size, 2 * a.intermediate_size],
            ))
            .with_edge(f32_edge(
                gate_up.clone(),
                vec![sequence_length, 2 * a.intermediate_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.gate_up_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(mlp_normed.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.mlp.gate_up_proj"
                )))
                .with_output(TensorEdgeId::new(gate_up.clone())),
            )
            // Fused gate/up projection, halved by a genuinely two-output
            // "split" node (`define-provider-prepared-kernel-execution-
            // contract` task group 3) rather than two separate matmuls --
            // a real-world LLM-serving optimization (one larger matmul
            // instead of two smaller ones), and this recipe's only node
            // with more than one output edge.
            .with_edge(f32_edge(
                gate.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_edge(f32_edge(
                up.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.split"), "split", OperatorFamily::Tensor)
                    .with_input(TensorEdgeId::new(gate_up.clone()))
                    .with_output(TensorEdgeId::new(gate.clone()))
                    .with_output(TensorEdgeId::new(up.clone())),
            )
            .with_edge(f32_edge(
                activated.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.silu"), "silu", OperatorFamily::Activation)
                    .with_input(TensorEdgeId::new(gate.clone()))
                    .with_output(TensorEdgeId::new(activated.clone())),
            )
            .with_edge(f32_edge(
                mlp_hidden.clone(),
                vec![sequence_length, a.intermediate_size],
            ))
            .with_node(
                op_node(format!("{prefix}.mul"), "mul", OperatorFamily::Tensor)
                    .with_input(TensorEdgeId::new(activated.clone()))
                    .with_input(TensorEdgeId::new(up.clone()))
                    .with_output(TensorEdgeId::new(mlp_hidden.clone())),
            )
            .with_edge(f32_edge(
                format!("weight.layers.{layer}.mlp.down_proj"),
                vec![a.intermediate_size, a.hidden_size],
            ))
            .with_edge(f32_edge(
                mlp_out.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.down_proj"),
                    "matmul",
                    OperatorFamily::LinearAlgebra,
                )
                .with_input(TensorEdgeId::new(mlp_hidden.clone()))
                .with_input(TensorEdgeId::new(format!(
                    "weight.layers.{layer}.mlp.down_proj"
                )))
                .with_output(TensorEdgeId::new(mlp_out.clone())),
            );

        let layer_out = format!("{prefix}.out");
        graph = graph
            .with_edge(f32_edge(
                layer_out.clone(),
                vec![sequence_length, a.hidden_size],
            ))
            .with_node(
                op_node(
                    format!("{prefix}.residual2"),
                    "residual-add",
                    OperatorFamily::Tensor,
                )
                .with_input(TensorEdgeId::new(post_attn.clone()))
                .with_input(TensorEdgeId::new(mlp_out.clone()))
                .with_output(TensorEdgeId::new(layer_out.clone())),
            );

        hidden_edge = layer_out;
    }

    graph = graph
        .with_edge(f32_edge("weight.final_norm", vec![a.hidden_size]))
        .with_edge(f32_edge(
            "hidden.final",
            vec![sequence_length, a.hidden_size],
        ))
        .with_node(
            op_node("final_norm", "rmsnorm", OperatorFamily::Normalization)
                .with_input(TensorEdgeId::new(hidden_edge.clone()))
                .with_input(TensorEdgeId::new("weight.final_norm"))
                .with_output(TensorEdgeId::new("hidden.final"))
                .with_attribute(
                    "epsilon",
                    OperatorAttributeValue::Float(config.rmsnorm_epsilon as f64),
                ),
        )
        .with_edge(qwen_lm_head_weight_edge(config))
        .with_edge(f32_edge("logits", vec![sequence_length, a.vocabulary_size]))
        .with_node(
            op_node("lm_head", "matmul", OperatorFamily::LinearAlgebra)
                .with_input(TensorEdgeId::new("hidden.final"))
                .with_input(TensorEdgeId::new("weight.lm_head"))
                .with_output(TensorEdgeId::new("logits")),
        );

    Ok(graph)
}
/// Produce a validated Qwen prefill Execution Graph for `prompt_length`
/// tokens.
///
/// Test-oracle only -- see [`qwen_build_graph`]'s doc comment.
pub fn qwen_prefill_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    prompt_length: u64,
    kv_cache_enabled: bool,
) -> Result<GraphProductionResult, QwenComponentError> {
    let graph = qwen_build_graph(
        config,
        identity,
        ExecutionGraphPhase::Prefill,
        prompt_length,
        kv_cache_enabled,
        // A prefill starts the sequence, so its first row is position zero.
        0,
    )?;
    validate_first_scope_graph(&graph)?;
    GraphProductionResult::validated(graph, &identity.id, &default_graph_catalog())
        .map_err(QwenComponentError::from)
}

/// Produce a validated Qwen decode Execution Graph for a single new token,
/// consuming prior KV cache.
///
/// `cached_token_count` is how many tokens the KV cache already holds, which
/// is the absolute position of the token being generated. It is required
/// rather than defaulted: a decode graph built as if the new token were at
/// position zero produces wrong rotations for every token after the first, and
/// does so silently.
///
/// Test-oracle only -- see [`qwen_build_graph`]'s doc comment.
pub fn qwen_decode_graph(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    cached_token_count: u64,
) -> Result<GraphProductionResult, QwenComponentError> {
    let graph = qwen_build_graph(
        config,
        identity,
        ExecutionGraphPhase::Decode,
        1,
        true,
        cached_token_count,
    )?;
    validate_first_scope_graph(&graph)?;
    GraphProductionResult::validated(graph, &identity.id, &default_graph_catalog())
        .map_err(QwenComponentError::from)
}
// ---------------------------------------------------------------------------
// Conformance report
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenConformanceCheck {
    pub name: &'static str,
    pub passed: bool,
    pub detail: Option<String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenConformanceReport {
    pub checks: Vec<QwenConformanceCheck>,
}
impl QwenConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}
/// Run the Qwen baseline's runnable conformance fixtures against `config`
/// and `identity`, producing a report. See [`qwen_conformance_fixture_names`]
/// for the full named fixture set this baseline SHALL cover; some fixtures
/// (e.g. authority denial) are covered by dedicated unit tests rather than
/// this data-driven report.
pub fn qwen_conformance_report(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
) -> QwenConformanceReport {
    let mut checks = Vec::new();

    let valid = config.validate(identity);
    checks.push(QwenConformanceCheck {
        name: "valid-minimal-config",
        passed: valid.is_ok(),
        detail: valid.err().map(|error| error.to_string()),
    });

    let mut invalid_family_architecture = config.architecture.clone();
    invalid_family_architecture.family = "not-qwen".into();
    let invalid_family_result =
        QwenConfig::new(invalid_family_architecture, config.rope.clone()).validate(identity);
    checks.push(QwenConformanceCheck {
        name: "invalid-architecture-family",
        passed: matches!(
            invalid_family_result,
            Err(QwenComponentError::ArchitectureUnsupported)
        ),
        detail: invalid_family_result.err().map(|error| error.to_string()),
    });

    let scope_result =
        validate_model_component_first_scope_requirements(&qwen_operator_requirements())
            .map_err(QwenComponentError::from);
    checks.push(QwenConformanceCheck {
        name: "required-operator-scope-validation",
        passed: scope_result.is_ok(),
        detail: scope_result.err().map(|error| error.to_string()),
    });

    let prefill_result = qwen_prefill_graph(config, identity, 4, true);
    checks.push(QwenConformanceCheck {
        name: "prefill-graph-production",
        passed: prefill_result.is_ok(),
        detail: prefill_result.err().map(|error| error.to_string()),
    });

    // Decode the 5th token, i.e. against the 4 tokens the prefill above cached,
    // so the check exercises a non-zero position rather than the degenerate
    // first-token case.
    let decode_result = qwen_decode_graph(config, identity, 4);
    checks.push(QwenConformanceCheck {
        name: "decode-graph-production",
        passed: decode_result.is_ok(),
        detail: decode_result.err().map(|error| error.to_string()),
    });

    checks.push(QwenConformanceCheck {
        name: "unsupported-quantization-rejection",
        passed: qwen_quantization_compatibility()
            .supported_methods
            .is_empty(),
        detail: None,
    });

    checks.push(QwenConformanceCheck {
        name: "authority-denial",
        passed: crate::validate_model_component_authority(["network"]).is_err()
            && !qwen_authority().is_empty(),
        detail: None,
    });

    checks.push(QwenConformanceCheck {
        name: "target-module-exposure",
        passed: qwen_target_modules().len() == QWEN_TARGET_MODULE_ROLES.len(),
        detail: None,
    });

    QwenConformanceReport { checks }
}

use std::collections::BTreeMap;

fn small_architecture() -> ModelComponentArchitectureMetadata {
    qwen_architecture_metadata(8, 2, 2, 2, 4, 16, 32, 64)
}

fn small_config() -> QwenConfig {
    QwenConfig::new(small_architecture(), QwenRopeConfig::standard(4))
}

fn small_identity() -> ModelComponentIdentity {
    qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    )
}

#[test]
fn valid_minimal_config_builds_descriptor() {
    let descriptor = qwen_component_descriptor(small_identity(), &small_config()).unwrap();
    assert_eq!(descriptor.architecture.family, QWEN_ARCHITECTURE_FAMILY);
    assert_eq!(
        descriptor.target_modules.len(),
        QWEN_TARGET_MODULE_ROLES.len()
    );
}

#[test]
fn invalid_architecture_family_is_rejected() {
    let mut architecture = small_architecture();
    architecture.family = "llama".into();
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
    assert_eq!(
        config.validate(&small_identity()),
        Err(QwenComponentError::ArchitectureUnsupported)
    );
}

#[test]
fn invalid_hidden_head_configuration_is_rejected() {
    let mut architecture = small_architecture();
    architecture.hidden_size = 9; // not attention_head_count * head_dimension
    let config = QwenConfig::new(architecture, QwenRopeConfig::standard(4));
    assert!(matches!(
        config.validate(&small_identity()),
        Err(QwenComponentError::ConfigInvalid {
            field: "head dimension",
            ..
        })
    ));
}

#[test]
fn missing_tensor_inventory_is_detected() {
    let config = small_config();
    let tensors = vec![ModelTensorMetadata {
        name: "token_embedding".into(),
        shape: vec![32, 8],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    }];
    assert!(matches!(
        qwen_validate_tensor_inventory(&config, &tensors),
        Err(QwenComponentError::TensorInventoryMissing { .. })
    ));
}

#[test]
fn invalid_tensor_shape_is_detected() {
    let config = small_config();
    let tensor = ModelTensorMetadata {
        name: "lm_head".into(),
        shape: vec![1, 1],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    assert!(matches!(
        qwen_validate_tensor_shapes(&config, std::slice::from_ref(&tensor)),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
}

#[test]
fn target_modules_are_exposed() {
    let modules = qwen_target_modules();
    assert!(
        modules
            .iter()
            .any(|module| module.role == TargetModuleRole::QProj)
    );
    assert!(
        modules
            .iter()
            .any(|module| module.role == TargetModuleRole::LmHead)
    );
}

#[test]
fn prefill_graph_production_succeeds() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
    assert_eq!(result.graph.phase, ExecutionGraphPhase::Prefill);
    assert!(result.validation.is_some());
}

#[test]
fn decode_graph_production_includes_kv_cache_append() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_decode_graph(&config, &identity, 4).unwrap();
    assert_eq!(result.graph.phase, ExecutionGraphPhase::Decode);
    let appended = result
            .graph
            .edges
            .values()
            .any(|edge| matches!(&edge.kv_cache, Some(metadata) if metadata.behavior == GraphKvCacheBehavior::Append));
    assert!(appended);
}

#[test]
fn required_operator_scope_validation_passes_for_baseline_requirements() {
    validate_model_component_first_scope_requirements(&qwen_operator_requirements()).unwrap();
}

#[test]
fn out_of_scope_operator_is_rejected_by_first_scope() {
    let out_of_scope = OperatorRequirement::new(OperatorId::magnetar(
        "flash-attention",
        1,
        OperatorFamily::Attention,
    ));
    assert!(validate_model_component_first_scope_requirements(&[out_of_scope]).is_err());
}

#[test]
fn tokenizer_compatibility_declares_vocabulary_size() {
    let config = small_config();
    let compatibility = qwen_tokenizer_compatibility(&config);
    assert_eq!(
        compatibility.vocabulary_size,
        config.architecture.vocabulary_size
    );
}

#[test]
fn tokenizer_vocabulary_mismatch_is_rejected() {
    let config = small_config();
    let tokenizer = crate::TokenizerMetadata {
        id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
        artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: crate::TokenizerFamily::new("bpe").unwrap(),
        revision: crate::TokenizerRevision::new("rev1").unwrap(),
        vocabulary_size: config.architecture.vocabulary_size as u32 + 1,
        added_token_count: 0,
        token_id_range: crate::TokenIdRange::new(0, config.architecture.vocabulary_size as u32),
        model_max_length: None,
        special_tokens: Vec::new(),
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: false,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    assert_eq!(
        qwen_validate_tokenizer_compatibility(&config, &tokenizer),
        Err(QwenComponentError::TokenizerIncompatible)
    );
}

#[test]
fn generation_defaults_exceeding_context_length_are_rejected() {
    let config = small_config();
    let defaults = crate::ModelGenerationDefaults {
        max_tokens: Some(1_000),
        ..crate::ModelGenerationDefaults::default()
    };
    assert!(matches!(
        qwen_validate_generation_defaults(&config, &defaults, None),
        Err(QwenComponentError::GenerationMetadataInvalid { .. })
    ));
}

#[test]
fn kv_cache_metadata_matches_architecture() {
    let config = small_config();
    let metadata = qwen_kv_cache_metadata(&config);
    assert_eq!(metadata.layer_count, config.architecture.layer_count);
    assert_eq!(metadata.head_dimension, config.architecture.head_dimension);
    assert!(!metadata.paged);
}

#[test]
fn adapter_target_validation_exposes_expected_modules() {
    let config = small_config();
    let compatibility = qwen_adapter_architecture_compatibility(&config, "qwen-baseline");
    assert!(compatibility.target_modules.contains("q_proj"));
    assert_eq!(
        compatibility.hidden_size,
        Some(config.architecture.hidden_size)
    );
}

#[test]
fn unsupported_quantization_is_rejected() {
    let config = small_config();
    let identity = small_identity();
    let descriptor = qwen_component_descriptor(identity, &config).unwrap();
    let mut manifest_architecture = ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
    manifest_architecture.required_component_role = None;
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new("qwen-test").unwrap(),
            ModelRevision::new("rev1").unwrap(),
            ModelDigest::sha256(b"qwen-test"),
        ),
        architecture: manifest_architecture,
        parts: BTreeMap::new(),
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: Some(ModelDType::F32),
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: Some(crate::ModelQuantization {
            format: crate::ModelQuantizationFormat::GgufQ4K,
            group_size: None,
            block_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            per_channel: false,
            workspace_bytes: None,
            required_capabilities: Vec::new(),
        }),
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
        architecture_config: None,
    };
    assert_eq!(
        qwen_validate_model_artifact(&descriptor, &config, &manifest),
        Err(QwenComponentError::QuantizationUnsupported)
    );
}

#[test]
fn authority_denies_forbidden_authorities() {
    let authority = qwen_authority();
    assert!(authority.contains(&ModelComponentAuthority::ModelArtifactRead));
    assert!(crate::validate_model_component_authority(["network"]).is_err());
    assert!(crate::validate_model_component_authority(["filesystem"]).is_err());
}

#[test]
fn no_provider_device_kernel_handle_exposure() {
    assert_eq!(
        crate::provider_handle_access_error(),
        ModelComponentError::ProviderAccessDenied
    );
    assert_eq!(
        crate::device_handle_access_error(),
        ModelComponentError::DeviceAccessDenied
    );
    assert_eq!(
        crate::kernel_handle_access_error(),
        ModelComponentError::KernelAccessDenied
    );
}

#[test]
fn rope_dynamic_scaling_unsupported_is_rejected() {
    let mut rope = QwenRopeConfig::standard(4);
    rope.scale = Some(2.0);
    assert!(matches!(
        rope.validate(4),
        Err(QwenComponentError::RopeUnsupported { .. })
    ));
}

#[test]
fn reference_cpu_covers_required_now_operators() {
    qwen_validate_reference_cpu_coverage().unwrap();
}

#[test]
fn browser_support_matches_implementation_kind() {
    assert!(qwen_browser_supported(ModelComponentImplementationKind::RuntimeNative).is_ok());
}

#[test]
fn contract_versions_accept_self_and_reject_newer_major() {
    qwen_validate_contract_versions(
        QWEN_TENSOR_CONTRACT_VERSION,
        QWEN_TOKENIZER_CONTRACT_VERSION,
        QWEN_KV_CACHE_CONTRACT_VERSION,
        QWEN_ADAPTER_CONTRACT_VERSION,
    )
    .unwrap();
    let newer_tensor = crate::CapabilityVersion::new(QWEN_TENSOR_CONTRACT_VERSION.major + 1, 0, 0);
    assert_eq!(
        qwen_validate_contract_versions(
            newer_tensor,
            QWEN_TOKENIZER_CONTRACT_VERSION,
            QWEN_KV_CACHE_CONTRACT_VERSION,
            QWEN_ADAPTER_CONTRACT_VERSION,
        ),
        Err(QwenComponentError::ComponentUnsupportedVersion)
    );
}

#[test]
fn tied_embedding_shape_must_match_vocabulary_and_hidden_size() {
    let mut config = small_config();
    config.tied_embeddings = true;
    let mut tensors = Vec::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, true) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
        tensors.push(ModelTensorMetadata {
            name,
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        });
    }
    assert!(qwen_validate_tied_embedding_shape(&config, &tensors).is_ok());

    for tensor in &mut tensors {
        if tensor.name == "token_embedding" {
            tensor.shape = vec![1, 1];
        }
    }
    assert!(matches!(
        qwen_validate_tied_embedding_shape(&config, &tensors),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
}

#[test]
fn tied_embeddings_alias_lm_head_weight_in_graph() {
    let mut config = small_config();
    config.tied_embeddings = true;
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
    let lm_head_edge = result
        .graph
        .edges
        .get(&TensorEdgeId::new("weight.lm_head"))
        .unwrap();
    assert_eq!(
        lm_head_edge.aliasing,
        TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"))
    );
}

#[test]
fn untied_lm_head_weight_does_not_alias() {
    let config = small_config();
    let identity = small_identity();
    let result = qwen_prefill_graph(&config, &identity, 4, false).unwrap();
    let lm_head_edge = result
        .graph
        .edges
        .get(&TensorEdgeId::new("weight.lm_head"))
        .unwrap();
    assert_eq!(lm_head_edge.aliasing, TensorAliasing::None);
}

#[test]
fn target_module_details_expose_layer_selector_and_insertion_point() {
    let details = qwen_target_module_details(4);
    assert_eq!(details.len(), QWEN_TARGET_MODULE_ROLES.len());
    let q_proj = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::QProj)
        .unwrap();
    assert_eq!(
        q_proj.layer_selector,
        AdapterLayerSelector::RangeInclusive { start: 0, end: 3 }
    );
    assert_eq!(
        q_proj.graph_insertion_point,
        QwenGraphInsertionPoint::AttentionProjection
    );
    assert!(
        q_proj
            .supported_adapter_methods
            .contains(&AdapterMethod::Lora)
    );
    let embedding = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::Embedding)
        .unwrap();
    assert_eq!(embedding.layer_selector, AdapterLayerSelector::All);
    assert_eq!(
        embedding.graph_insertion_point,
        QwenGraphInsertionPoint::Embedding
    );
    let lm_head = details
        .iter()
        .find(|detail| detail.module.role == TargetModuleRole::LmHead)
        .unwrap();
    assert_eq!(
        lm_head.graph_insertion_point,
        QwenGraphInsertionPoint::LogitsProjection
    );
}

#[test]
fn tokenizer_missing_required_bos_is_rejected() {
    let mut config = small_config();
    config.require_bos = true;
    let tokenizer = crate::TokenizerMetadata {
        id: crate::TokenizerId::new("qwen-tokenizer").unwrap(),
        artifact: crate::TokenizerArtifactId::new("qwen-tokenizer-artifact").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: crate::TokenizerFamily::new("bpe").unwrap(),
        revision: crate::TokenizerRevision::new("rev1").unwrap(),
        vocabulary_size: config.architecture.vocabulary_size as u32,
        added_token_count: 0,
        token_id_range: crate::TokenIdRange::new(0, config.architecture.vocabulary_size as u32 - 1),
        model_max_length: None,
        special_tokens: vec![crate::SpecialToken::new(
            crate::SpecialTokenKind::Eos,
            "<eos>",
            0,
        )],
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: false,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    assert_eq!(
        qwen_validate_tokenizer_compatibility(&config, &tokenizer),
        Err(QwenComponentError::TokenizerIncompatible)
    );
}

#[test]
fn chat_template_required_but_missing_is_rejected() {
    let mut config = small_config();
    config.chat_template_required = true;
    let identity = small_identity();
    let descriptor = qwen_component_descriptor(identity, &config).unwrap();
    let manifest_architecture = ModelArchitecture::new(QWEN_ARCHITECTURE_FAMILY, "qwen-test");
    let mut tensors = Vec::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap_or_default();
        tensors.push(ModelTensorMetadata {
            name,
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: None,
            size_bytes: None,
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        });
    }
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new("qwen-test").unwrap(),
            ModelRevision::new("rev1").unwrap(),
            ModelDigest::sha256(b"qwen-test"),
        ),
        architecture: manifest_architecture,
        parts: BTreeMap::new(),
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: Some(ModelDType::F32),
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: Some(ModelArtifactSource::LocalCache("qwen-test".into())),
        architecture_config: None,
    };
    assert!(matches!(
        qwen_validate_model_artifact(&descriptor, &config, &manifest),
        Err(QwenComponentError::ComponentInvalid { .. })
    ));
}

#[test]
fn generation_defaults_non_authoritative_sampling_values_are_never_rejected() {
    let config = small_config();
    for temperature in [-5.0, 0.0, 1.0, 1_000.0] {
        let defaults = crate::ModelGenerationDefaults {
            temperature: Some(temperature),
            top_p: Some(-1.0),
            top_k: Some(u32::MAX),
            ..crate::ModelGenerationDefaults::default()
        };
        assert!(qwen_validate_generation_defaults(&config, &defaults, None).is_ok());
    }
}

#[test]
fn adapter_activation_rejected_when_baseline_lacks_graph_support() {
    assert_eq!(
        qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport::baseline()),
        Err(QwenComponentError::AdapterUnsupported)
    );
    assert!(
        qwen_validate_adapter_activation_supported(QwenAdapterGraphSupport {
            overlay_supported: true,
            merge_supported: false,
        })
        .is_ok()
    );
}

#[test]
fn adapter_target_shape_mismatch_is_rejected() {
    let config = small_config();
    let target = AdapterTargetModule {
        name: "q_proj".into(),
        role: AdapterTargetModuleRole::QueryProjection,
        layer_selector: None,
        expected_shape: vec![1, 1],
    };
    assert!(matches!(
        qwen_validate_adapter_target_shapes(&config, std::slice::from_ref(&target)),
        Err(QwenComponentError::TensorShapeMismatch { .. })
    ));
    let a = &config.architecture;
    let matching = AdapterTargetModule {
        expected_shape: vec![a.hidden_size, a.attention_head_count * a.head_dimension],
        ..target
    };
    assert!(qwen_validate_adapter_target_shapes(&config, &[matching]).is_ok());
}

#[test]
fn tensor_scope_rejects_placeholder_dtype_and_non_contiguous_layout() {
    let f32_contiguous = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    assert!(qwen_validate_tensor_scope(&f32_contiguous).is_ok());

    let float16 = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float16),
        LayoutDescriptor::Contiguous,
    );
    assert_eq!(
        qwen_validate_tensor_scope(&float16),
        Err(QwenComponentError::DTypeUnsupported)
    );

    let strided = TensorDescriptor::new(
        ShapeDescriptor::new(vec![1, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Strided {
            strides_elements: vec![1, 1],
            offset_elements: 0,
        },
    );
    assert_eq!(
        qwen_validate_tensor_scope(&strided),
        Err(QwenComponentError::LayoutUnsupported)
    );
}

#[test]
fn explicit_dtype_and_layout_conversion_nodes_validate() {
    let raw_input = TensorEdge::new(
        TensorEdgeId::new("raw.input"),
        TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 2]),
            DTypeDescriptor::portable(ComputeDType::Float16),
            LayoutDescriptor::Strided {
                strides_elements: vec![1, 1],
                offset_elements: 0,
            },
        ),
    );
    let converted_dtype = f32_edge("converted.dtype", vec![1, 2]);
    let converted_layout = f32_edge("converted.layout", vec![1, 2]);
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("conversion-test"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(raw_input);
    graph = qwen_insert_dtype_conversion(
        graph,
        "dtype-fix",
        TensorEdgeId::new("raw.input"),
        converted_dtype,
    );
    graph = qwen_insert_layout_conversion(
        graph,
        "layout-fix",
        TensorEdgeId::new("converted.dtype"),
        converted_layout,
    );
    graph.validate(&default_graph_catalog()).unwrap();
}

#[test]
fn reference_cpu_rejects_incompatible_kv_head_count() {
    let q = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    let k = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    let v = crate::HostTensor::new(vec![1, 4], vec![0.0; 4]).unwrap();
    // head_count=2 is not a multiple of kv_head_count=3: unsupported variant.
    assert!(crate::attention(&q, &k, &v, 2, 2, Some(3), None, true).is_err());
    // head_count=2 dividing kv_head_count=2 (standard multi-head) succeeds.
    assert!(crate::attention(&q, &k, &v, 2, 2, Some(2), None, true).is_ok());
}

#[test]
fn kv_cache_compatibility_embeds_component_version() {
    let identity_v1 = small_identity();
    let identity_v2 = qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(2, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let model = GenerationModelReference::LoadedModelContext("ctx".into());
    let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();
    let compatibility_v1 =
        qwen_kv_cache_compatibility(&identity_v1, model.clone(), tokenizer.clone());
    let compatibility_v2 = qwen_kv_cache_compatibility(&identity_v2, model, tokenizer);
    assert_ne!(
        compatibility_v1.model_architecture,
        compatibility_v2.model_architecture
    );
}

#[test]
fn prefix_cache_compatibility_rejects_cross_version_and_cross_adapter_reuse() {
    let identity_v1 = small_identity();
    let identity_v2 = qwen_component_identity(
        ModelComponentId::new("qwen-baseline").unwrap(),
        ModelComponentVersion::new(2, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let config = small_config();
    let model = GenerationModelReference::LoadedModelContext("ctx".into());
    let tokenizer = TokenizerId::new("qwen-tokenizer").unwrap();

    let base = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    let same = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    assert!(base.validate_reuse(&same).is_ok());

    let different_version = qwen_prefix_cache_compatibility(
        &identity_v2,
        &config,
        model.clone(),
        tokenizer.clone(),
        None,
        None,
    );
    assert!(base.validate_reuse(&different_version).is_err());

    let adapter_set = AdapterSetId::empty();
    let with_adapter = qwen_prefix_cache_compatibility(
        &identity_v1,
        &config,
        model,
        tokenizer,
        None,
        Some(&adapter_set),
    );
    assert!(base.validate_reuse(&with_adapter).is_err());
}

#[test]
fn observability_functions_preserve_component_identity_and_tag_events() {
    let component = ModelComponentId::new("qwen-baseline").unwrap();
    let observations = vec![
        qwen_observation_component_resolved(&component),
        qwen_observation_component_validated(&component),
        qwen_observation_component_rejected(&component, "unsupported"),
        qwen_observation_config_validated(&component),
        qwen_observation_tensor_inventory_checked(&component),
        qwen_observation_target_modules_exposed(&component),
        qwen_observation_tokenizer_compatibility_checked(&component),
        qwen_observation_kv_metadata_produced(&component),
        qwen_observation_prefill_graph_produced(&component),
        qwen_observation_decode_graph_produced(&component),
        qwen_observation_graph_validation_failed(&component, "bad graph"),
        qwen_observation_required_operator_missing(&component, "flash-attention"),
        qwen_observation_reference_cpu_coverage_missing(&component),
        qwen_observation_authority_denied(&component, "network"),
        qwen_observation_conformance_result(&component, true),
    ];
    assert_eq!(observations.len(), 15);
    for observation in &observations {
        assert_eq!(observation.component, Some(component.clone()));
        assert!(observation.redacted_metadata.contains_key("qwen-event"));
    }
}

#[test]
fn conformance_report_is_conformant_for_valid_config() {
    let report = qwen_conformance_report(&small_config(), &small_identity());
    assert!(report.is_conformant(), "failing checks: {report:?}");
    let names: BTreeSet<&str> = report.checks.iter().map(|check| check.name).collect();
    assert!(names.contains("valid-minimal-config"));
    assert!(names.contains("prefill-graph-production"));
    assert!(names.contains("decode-graph-production"));
}

fn integration_config() -> QwenConfig {
    QwenConfig::new(
        qwen_architecture_metadata(4, 1, 2, 2, 2, 8, 16, 32),
        QwenRopeConfig::standard(2),
    )
}

fn integration_manifest(architecture: ModelArchitecture) -> ModelManifest {
    let config = integration_config();
    let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001";
    let mut tensor_yaml = String::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, false) {
        let shape = qwen_expected_tensor_shape(&name, &config).unwrap();
        let shape_text = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n"
        ));
    }
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {digest}
model:
  name: qwen-integration-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
artifacts:
  weights:
    kind: model-weights
    digest: {digest}
    size_bytes: 128
  config:
    kind: model-config
    digest: {digest}
    size_bytes: 16
tensors:
{tensor_yaml}"#,
        family = architecture.family,
        identifier = architecture.identifier,
    ))
    .unwrap()
}

/// End-to-end integration: a Qwen Component identity/config drives real
/// Model Loading (trust, precondition validation, memory admission,
/// architecture resolution, residency planning), Qwen artifact/tensor
/// validation against the exact loaded manifest, Model Instance creation
/// referencing the Qwen architecture implementation, and finally Runtime
/// graph planning/execution of the produced prefill graph. This is the
/// concrete mechanism by which Model Loading, Model Instance, and (via
/// the Execution Graph boundary) Generation use the Qwen Component today;
/// `generation::prefill`/`decode_step` remain pre-existing
/// architecture-agnostic placeholders pending a future Runtime inference
/// API change.
#[test]
fn qwen_component_integrates_with_model_loading_and_instance_and_graph_execution() {
    let config = integration_config();
    let identity = qwen_component_identity(
        ModelComponentId::new("qwen-integration").unwrap(),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::RuntimeNative,
    );
    let architecture_implementation = qwen_architecture_implementation(
        &identity,
        crate::ModelArchitectureImplementationKind::ComponentBased,
    );
    let manifest = integration_manifest(architecture_implementation.architecture.clone());

    let descriptor = qwen_component_descriptor(identity.clone(), &config).unwrap();
    qwen_validate_model_artifact(&descriptor, &config, &manifest).unwrap();
    assert_eq!(qwen_target_modules().len(), QWEN_TARGET_MODULE_ROLES.len());

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("qwen-integration-load"),
        manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = coordinator
        .load(
            request,
            &manifest,
            &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture"),
            &mut memory,
        )
        .unwrap();

    let instance = ModelInstanceDefinition::from_loaded_context(
        &loaded,
        architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert_eq!(
        instance.architecture.architecture,
        architecture_implementation.architecture
    );
    assert_eq!(
        instance.architecture.architecture.identifier,
        qwen_component_compatibility_key(&identity)
    );

    let prefill = qwen_prefill_graph(&config, &identity, 4, true).unwrap();
    assert!(prefill.validation.is_some());
    let policy = crate::GraphPlanningPolicy::default();
    let plan = crate::plan_execution_graph(&prefill.graph, &default_graph_catalog(), &policy, None)
        .unwrap();
    assert!(!plan.execution_order.is_empty());
    crate::execute_graph_boundary(&prefill.graph, &default_graph_catalog(), &policy).unwrap();

    let kv_cache_compatibility = qwen_kv_cache_compatibility(
        &identity,
        GenerationModelReference::ModelInstance(
            crate::ModelInstanceId::new("qwen-instance").unwrap(),
        ),
        TokenizerId::new("qwen-tokenizer").unwrap(),
    );
    assert_eq!(
        kv_cache_compatibility.model_architecture,
        Some(qwen_component_compatibility_key(&identity))
    );
}
