//! The Qwen4-Exp Operators' attribute parsing and shared entry point
//! (`support-qwen38-flash-next`): every Provider runs these through
//! [`hyper_stream_operator`], so each reads its attributes the same way.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::{
    GatedResidual, NgramEmbedding, NgramPle, QsaSelect, gated_residual, gated_residual_inject,
    ngram_embedding, ngram_ple, qsa_select, repeat_features,
};

/// The Operators [`hyper_stream_operator`] runs.
pub const HYPER_STREAM_OPERATORS: [&str; 7] = [
    "gated-residual",
    "gated-residual-final",
    "gated-residual-inject",
    "repeat-features",
    "qsa-select",
    "ngram-embedding",
    "ngram-ple",
];

type Attributes = BTreeMap<String, OperatorAttributeValue>;

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
}

fn integer(attributes: &Attributes, name: &str) -> Result<i64, ReferenceCpuError> {
    match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) => Ok(*value),
        _ => Err(invalid(format!("attribute {name} must be an integer"))),
    }
}

fn positive(attributes: &Attributes, name: &str) -> Result<usize, ReferenceCpuError> {
    match integer(attributes, name)? {
        value if value > 0 => {
            usize::try_from(value).map_err(|_| invalid(format!("attribute {name} is too large")))
        }
        _ => Err(invalid(format!("attribute {name} must be positive"))),
    }
}

fn float(attributes: &Attributes, name: &str) -> Result<f32, ReferenceCpuError> {
    match attributes.get(name) {
        Some(OperatorAttributeValue::Float(value)) if (*value as f32).is_finite() => {
            Ok(*value as f32)
        }
        _ => Err(invalid(format!("attribute {name} must be a finite float"))),
    }
}

fn float_or(attributes: &Attributes, name: &str, default: f32) -> Result<f32, ReferenceCpuError> {
    if attributes.contains_key(name) {
        float(attributes, name)
    } else {
        Ok(default)
    }
}

/// `gated-residual`, `gated-residual-final` and `ngram-ple` all name their
/// streams `hc_count`.
pub fn gated_residual_from_attributes(
    attributes: &Attributes,
) -> Result<GatedResidual, ReferenceCpuError> {
    Ok(GatedResidual {
        hc_count: positive(attributes, "hc_count")?,
        epsilon: float(attributes, "epsilon")?,
        weight_offset: float_or(attributes, "weight_offset", 0.0)?,
    })
}

pub fn qsa_select_from_attributes(attributes: &Attributes) -> Result<QsaSelect, ReferenceCpuError> {
    Ok(QsaSelect {
        head_count: positive(attributes, "head_count")?,
        head_dim: positive(attributes, "head_dim")?,
        compress_ratio: positive(attributes, "compress_ratio")?,
        budget: positive(attributes, "budget")?,
        epsilon: float(attributes, "epsilon")?,
        weight_offset: float_or(attributes, "weight_offset", 0.0)?,
        rope_base: float(attributes, "rope_base")?,
        rotary_dim: positive(attributes, "rotary_dim")?,
    })
}

pub fn ngram_embedding_from_attributes(
    attributes: &Attributes,
) -> Result<NgramEmbedding, ReferenceCpuError> {
    let unsigned = |name: &str| {
        u64::try_from(integer(attributes, name)?)
            .map_err(|_| invalid(format!("attribute {name} must not be negative")))
    };
    Ok(NgramEmbedding {
        ngram_size: positive(attributes, "ngram_size")?,
        heads_per_ngram: positive(attributes, "heads_per_ngram")?,
        vocab_size: unsigned("vocab_size")?,
        vocab_base: unsigned("vocab_base")?,
        layer_index: unsigned("layer_index")?,
        seed: unsigned("seed")?,
        eos_token_id: integer(attributes, "eos_token_id")?,
    })
}

pub fn ngram_ple_from_attributes(attributes: &Attributes) -> Result<NgramPle, ReferenceCpuError> {
    Ok(NgramPle {
        hc_count: positive(attributes, "hc_count")?,
        epsilon: float(attributes, "epsilon")?,
        weight_offset: float_or(attributes, "weight_offset", 0.0)?,
        conv_kernel: positive(attributes, "conv_kernel")?,
        conv_dilation: positive(attributes, "conv_dilation")?,
    })
}

/// The shapes of `name`'s outputs, from its inputs' shapes and attributes,
/// without computing them: what a graph executor needs to admit them.
pub fn hyper_stream_output_shapes(
    name: &str,
    input_shapes: &[Vec<u64>],
    attributes: &Attributes,
) -> Result<Vec<Vec<u64>>, ReferenceCpuError> {
    let shape = |index: usize| {
        input_shapes
            .get(index)
            .ok_or_else(|| invalid(format!("{name} is missing input {index}")))
    };
    Ok(match name {
        "gated-residual" | "gated-residual-final" => {
            let config = gated_residual_from_attributes(attributes)?;
            let [tokens, width] = shape(0)?[..] else {
                return Err(invalid("gated-residual input must be rank two"));
            };
            let hc = config.hc_count as u64;
            let mixed = vec![tokens, width / hc];
            if name == "gated-residual" {
                vec![mixed, vec![tokens, hc]]
            } else {
                vec![mixed]
            }
        }
        "gated-residual-inject" => vec![shape(0)?.clone()],
        "repeat-features" => {
            let [tokens, width] = shape(0)?[..] else {
                return Err(invalid("repeat-features input must be rank two"));
            };
            vec![vec![tokens, width * positive(attributes, "count")? as u64]]
        }
        "qsa-select" => {
            let config = qsa_select_from_attributes(attributes)?;
            vec![vec![shape(0)?[0], config.slots() as u64]]
        }
        "ngram-embedding" => {
            let config = ngram_embedding_from_attributes(attributes)?;
            let [_, head_dim] = shape(2)?[..] else {
                return Err(invalid("ngram-embedding table must be rank two"));
            };
            vec![
                vec![shape(0)?[0], config.heads() as u64 * head_dim],
                vec![config.context_len() as u64],
            ]
        }
        "ngram-ple" => {
            let config = ngram_ple_from_attributes(attributes)?;
            let [_, width] = shape(0)?[..] else {
                return Err(invalid("ngram-ple residual must be rank two"));
            };
            vec![shape(0)?.clone(), vec![width, config.state_len() as u64]]
        }
        other => return Err(invalid(format!("{other} is not a Qwen4-Exp Operator"))),
    })
}

/// Runs one Qwen4-Exp Operator on host tensors and returns its outputs in
/// order.
pub fn hyper_stream_operator(
    name: &str,
    inputs: &[HostTensor],
    attributes: &Attributes,
) -> Result<Vec<HostTensor>, ReferenceCpuError> {
    match name {
        "gated-residual" | "gated-residual-final" => {
            let config = gated_residual_from_attributes(attributes)?;
            let expected = if name == "gated-residual" { 5 } else { 4 };
            if inputs.len() != expected {
                return Err(invalid(format!("{name} takes {expected} inputs")));
            }
            gated_residual(inputs, config)
        }
        "gated-residual-inject" => Ok(vec![gated_residual_inject(inputs)?]),
        "repeat-features" => {
            let [input] = inputs else {
                return Err(invalid("repeat-features takes one input"));
            };
            Ok(vec![repeat_features(
                input,
                positive(attributes, "count")?,
            )?])
        }
        "qsa-select" => Ok(vec![qsa_select(
            inputs,
            qsa_select_from_attributes(attributes)?,
        )?]),
        "ngram-embedding" => {
            Ok(ngram_embedding(inputs, ngram_embedding_from_attributes(attributes)?)?.into())
        }
        "ngram-ple" => Ok(ngram_ple(inputs, ngram_ple_from_attributes(attributes)?)?.into()),
        other => Err(invalid(format!("{other} is not a Qwen4-Exp Operator"))),
    }
}
