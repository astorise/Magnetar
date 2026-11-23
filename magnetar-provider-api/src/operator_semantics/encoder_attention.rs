//! Portable single-sequence bidirectional attention with key padding.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

/// Geometry and optional symmetric window for `encoder-attention@1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderAttention {
    pub heads: u64,
    pub head_dim: u64,
    pub local_radius: Option<u64>,
    pub scale: Option<f32>,
}

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Decode the operator's validated attributes without defaulting malformed values.
pub fn encoder_attention_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<EncoderAttention, ReferenceCpuError> {
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => Ok(*value as u64),
        _ => Err(invalid(format!(
            "encoder-attention {name} must be a positive integer"
        ))),
    };
    let local_radius = match attributes.get("local_radius") {
        None => None,
        Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Some(*value as u64),
        _ => {
            return Err(invalid(
                "encoder-attention local_radius must be nonnegative",
            ));
        }
    };
    let scale = match attributes.get("scale") {
        None => None,
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite()
                && *value > 0.0
                && (*value as f32).is_finite()
                && (*value as f32) > 0.0 =>
        {
            Some(*value as f32)
        }
        _ => {
            return Err(invalid(
                "encoder-attention scale must be finite and positive",
            ));
        }
    };
    Ok(EncoderAttention {
        heads: positive("head_count")?,
        head_dim: positive("head_dimension")?,
        local_radius,
        scale,
    })
}

/// Compute f32 attention for one sequence. The padding mask selects valid
/// keys; it does not zero query rows. A local radius is inclusive on both
/// sides. Queries with no eligible keys and non-finite inputs are rejected.
pub fn encoder_attention(
    q: &HostTensor,
    k: &HostTensor,
    v: &HostTensor,
    mask: &HostTensor,
    config: EncoderAttention,
) -> Result<HostTensor, ReferenceCpuError> {
    let [tokens, width] = q.shape.as_slice() else {
        return Err(invalid("encoder-attention q must be [tokens, width]"));
    };
    if *tokens == 0
        || *width == 0
        || k.shape != q.shape
        || v.shape != q.shape
        || mask.shape != [*tokens]
        || mask.data.len() as u64 != *tokens
        || config.heads == 0
        || config.head_dim == 0
        || config.heads.checked_mul(config.head_dim) != Some(*width)
        || config
            .scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        return Err(invalid(
            "encoder-attention tensor shapes or geometry are invalid",
        ));
    }
    let tokens = usize::try_from(*tokens).map_err(|_| invalid("token count overflows usize"))?;
    let width = usize::try_from(*width).map_err(|_| invalid("row width overflows usize"))?;
    let heads = usize::try_from(config.heads).map_err(|_| invalid("head count overflows usize"))?;
    let head_dim =
        usize::try_from(config.head_dim).map_err(|_| invalid("head width overflows usize"))?;
    let count = tokens
        .checked_mul(width)
        .ok_or_else(|| invalid("attention tensor is too large"))?;
    if [q, k, v].iter().any(|tensor| {
        tensor.data.len() != count || tensor.data.iter().any(|value| !value.is_finite())
    }) || mask.data.iter().any(|value| *value != 0.0 && *value != 1.0)
    {
        return Err(invalid(
            "encoder-attention requires finite inputs and a binary mask",
        ));
    }
    let radius = config
        .local_radius
        .map(|value| usize::try_from(value.min(tokens as u64)).unwrap_or(tokens));
    let scale = config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
    let mut output = vec![0.0; count];
    let mut scores = Vec::with_capacity(tokens);
    for row in 0..tokens {
        let lower = radius.map_or(0, |r| row.saturating_sub(r));
        let upper = radius.map_or(tokens, |r| {
            row.saturating_add(r).saturating_add(1).min(tokens)
        });
        if !mask.data[lower..upper].contains(&1.0) {
            return Err(invalid(format!(
                "encoder-attention query {row} has no valid key"
            )));
        }
        for head in 0..heads {
            let offset = head * head_dim;
            scores.clear();
            for key in lower..upper {
                if mask.data[key] == 0.0 {
                    continue;
                }
                let mut dot = 0.0f32;
                for dimension in 0..head_dim {
                    dot += q.data[row * width + offset + dimension]
                        * k.data[key * width + offset + dimension];
                }
                let score = dot * scale;
                if !score.is_finite() {
                    return Err(invalid("non-finite encoder-attention score"));
                }
                scores.push((key, score));
            }
            let max = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|(_, score)| (*score - max).exp()).sum();
            for dimension in 0..head_dim {
                let mut value = 0.0;
                for &(key, score) in &scores {
                    value += ((score - max).exp() / denominator)
                        * v.data[key * width + offset + dimension];
                }
                if !value.is_finite() {
                    return Err(invalid("non-finite encoder-attention output"));
                }
                output[row * width + offset + dimension] = value;
            }
        }
    }
    HostTensor::new(vec![tokens as u64, width as u64], output)
}
