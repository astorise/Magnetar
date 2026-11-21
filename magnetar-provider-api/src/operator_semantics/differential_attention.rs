//! Causal attention with grouped signal and noise heads.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

const MAX_ELEMENTS: usize = 67_108_864;

/// Numerics and causal window for the differential-attention core.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DifferentialAttention {
    pub scale: f32,
    /// Zero disables score capping.
    pub softcap: f32,
    /// Maximum distance to an earlier key; None is an unbounded window.
    pub window_left: Option<usize>,
    /// Round Q/K/V and each attention output to bfloat16, accumulating in f32.
    pub round_bfloat16: bool,
}

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Read the required core attributes without substituting another profile.
pub fn differential_attention_from_attributes(
    attrs: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<DifferentialAttention, ReferenceCpuError> {
    let number = |name: &str, positive: bool| match attrs.get(name) {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite()
                && (*value as f32).is_finite()
                && if positive {
                    *value as f32 > 0.0
                } else {
                    *value == 0.0 || (*value > 0.0 && *value as f32 > 0.0)
                } =>
        {
            Ok(*value as f32)
        }
        _ => Err(invalid(format!("differential-attention invalid {name}"))),
    };
    let window_left = match attrs.get("window_left") {
        Some(OperatorAttributeValue::Integer(-1)) => None,
        Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Some(
            usize::try_from(*value)
                .map_err(|_| invalid("differential-attention window is too large"))?,
        ),
        _ => return Err(invalid("differential-attention invalid window_left")),
    };
    let round_bfloat16 = match attrs.get("round_bfloat16") {
        Some(OperatorAttributeValue::Boolean(value)) => *value,
        _ => return Err(invalid("differential-attention invalid round_bfloat16")),
    };
    Ok(DifferentialAttention {
        scale: number("scale", true)?,
        softcap: number("softcap", false)?,
        window_left,
        round_bfloat16,
    })
}

fn product(dims: &[usize]) -> Result<usize, ReferenceCpuError> {
    dims.iter()
        .try_fold(1usize, |n, &dim| {
            n.checked_mul(dim).filter(|&size| size <= MAX_ELEMENTS)
        })
        .ok_or_else(|| invalid("differential-attention dimensions exceed the reference limit"))
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn bfloat16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// Four inputs: Q, K, V and per-KV-head lambda logits. The result is the
/// signal heads after subtracting their group's gated noise head.
pub fn differential_attention(
    inputs: &[HostTensor],
    config: DifferentialAttention,
) -> Result<HostTensor, ReferenceCpuError> {
    let [q, k, v, lambda] = inputs else {
        return Err(invalid("differential-attention requires four inputs"));
    };
    let [tokens, query_heads, width] = q.shape.as_slice() else {
        return Err(invalid("differential-attention Q must be rank three"));
    };
    let [keys, kv_heads, key_width] = k.shape.as_slice() else {
        return Err(invalid("differential-attention K must be rank three"));
    };
    if *tokens == 0
        || *keys < *tokens
        || *kv_heads == 0
        || *width == 0
        || *key_width != *width
        || *query_heads / *kv_heads < 2
        || !query_heads.is_multiple_of(*kv_heads)
        || !config.scale.is_finite()
        || config.scale <= 0.0
        || !config.softcap.is_finite()
        || config.softcap < 0.0
    {
        return Err(invalid(
            "invalid differential-attention geometry or profile",
        ));
    }
    let convert =
        |n| usize::try_from(n).map_err(|_| invalid("differential-attention dimension overflow"));
    let (t, s, hq, hk, d) = (
        convert(*tokens)?,
        convert(*keys)?,
        convert(*query_heads)?,
        convert(*kv_heads)?,
        convert(*width)?,
    );
    let expected = [
        (
            q,
            vec![*tokens, *query_heads, *width],
            product(&[t, hq, d])?,
        ),
        (k, vec![*keys, *kv_heads, *width], product(&[s, hk, d])?),
        (v, vec![*keys, *kv_heads, *width], product(&[s, hk, d])?),
        (lambda, vec![*tokens, *kv_heads], product(&[t, hk])?),
    ];
    for (tensor, shape, count) in expected {
        if tensor.shape != shape
            || tensor.data.len() != count
            || tensor.data.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "differential-attention input has an invalid shape or value",
            ));
        }
    }
    let per_group = hq / hk;
    let signal_per_group = per_group - 1;
    let signal_heads = hk * signal_per_group;
    let mut output = vec![0.0; product(&[t, signal_heads, d])?];
    let round = |v| {
        if config.round_bfloat16 {
            bfloat16(v)
        } else {
            v
        }
    };
    for token in 0..t {
        let last = s - t + token;
        let first = config.window_left.map_or(0, |w| last.saturating_sub(w));
        for group in 0..hk {
            let mut heads = vec![0.0; product(&[per_group, d])?];
            for head in 0..per_group {
                let q_base = (token * hq + group * per_group + head) * d;
                let mut scores = Vec::with_capacity(last - first + 1);
                for position in first..=last {
                    let k_base = (position * hk + group) * d;
                    let mut score = 0.0;
                    for dim in 0..d {
                        score += round(q.data[q_base + dim]) * round(k.data[k_base + dim]);
                    }
                    score *= config.scale;
                    if config.softcap > 0.0 {
                        score = config.softcap * (score / config.softcap).tanh();
                    }
                    if !score.is_finite() {
                        return Err(invalid("non-finite differential-attention score"));
                    }
                    scores.push(score);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for (index, score) in scores.into_iter().enumerate() {
                    let v_base = ((first + index) * hk + group) * d;
                    for dim in 0..d {
                        heads[head * d + dim] += (score / sum) * round(v.data[v_base + dim]);
                    }
                }
                for dim in 0..d {
                    heads[head * d + dim] = round(heads[head * d + dim]);
                }
            }
            let noise = (per_group - 1) * d;
            let gate = sigmoid(lambda.data[token * hk + group]);
            for signal in 0..signal_per_group {
                let out = (token * signal_heads + group * signal_per_group + signal) * d;
                for dim in 0..d {
                    output[out + dim] = heads[signal * d + dim] - gate * heads[noise + dim];
                }
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite differential-attention output"));
    }
    HostTensor::new([*tokens, signal_heads as u64, *width], output)
}
