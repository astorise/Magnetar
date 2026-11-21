//! Projected no-RoPE attention with explicit raw and attention histories.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::{
    DifferentialAttention, differential_attention, differential_attention_from_attributes,
    token_shift,
};

const MAX_ELEMENTS: usize = 67_108_864;
const MAX_EXACT_POSITION: usize = 16_777_215;

/// The supported biasless, no-RoPE attention profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedDifferentialAttention {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rank: usize,
    pub past_length: usize,
    pub epsilon: f32,
    pub zero_centered_norm: bool,
    pub attention: DifferentialAttention,
}

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Decode the complete profile and validate all required attributes.
pub fn projected_differential_attention_from_attributes(
    attrs: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<ProjectedDifferentialAttention, ReferenceCpuError> {
    let integer = |name: &str, nonzero: bool| match attrs.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value >= 0 && (!nonzero || *value > 0) => {
            usize::try_from(*value).map_err(|_| invalid(format!("invalid {name}")))
        }
        _ => Err(invalid(format!("invalid {name}"))),
    };
    let epsilon = match attrs.get("epsilon") {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite() && (*value as f32).is_finite() && *value as f32 > 0.0 =>
        {
            *value as f32
        }
        _ => return Err(invalid("invalid projected attention epsilon")),
    };
    let zero_centered_norm = match attrs.get("zero_centered_norm") {
        Some(OperatorAttributeValue::Boolean(value)) => *value,
        _ => return Err(invalid("invalid projected attention zero_centered_norm")),
    };
    let config = ProjectedDifferentialAttention {
        query_heads: integer("query_heads", true)?,
        kv_heads: integer("kv_heads", true)?,
        head_dim: integer("head_dim", true)?,
        rank: integer("rank", true)?,
        past_length: integer("past_length", false)?,
        epsilon,
        zero_centered_norm,
        attention: differential_attention_from_attributes(attrs)?,
    };
    validate_config(config)?;
    Ok(config)
}

fn product(dims: &[usize]) -> Result<usize, ReferenceCpuError> {
    dims.iter()
        .try_fold(1usize, |n, &dim| {
            n.checked_mul(dim).filter(|&size| size <= MAX_ELEMENTS)
        })
        .ok_or_else(|| invalid("projected attention dimensions exceed the reference limit"))
}

fn validate_config(config: ProjectedDifferentialAttention) -> Result<(), ReferenceCpuError> {
    if config.kv_heads == 0
        || config.query_heads / config.kv_heads < 2
        || !config.query_heads.is_multiple_of(config.kv_heads)
        || config.head_dim == 0
        || config.rank == 0
        || config.past_length > MAX_EXACT_POSITION
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
    {
        return Err(invalid("invalid projected attention geometry"));
    }
    product(&[config.query_heads, config.head_dim])?;
    product(&[config.kv_heads, config.head_dim])?;
    product(&[config.kv_heads, config.rank])?;
    product(&[config.rank, config.head_dim])?;
    Ok(())
}

fn check(tensor: &HostTensor, shape: &[usize]) -> Result<(), ReferenceCpuError> {
    let count = product(shape)?;
    if tensor.shape != shape.iter().map(|&n| n as u64).collect::<Vec<_>>()
        || tensor.data.len() != count
        || tensor.data.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "projected attention input has an invalid shape or value",
        ));
    }
    Ok(())
}

fn tensor(shape: &[usize], data: Vec<f32>) -> Result<HostTensor, ReferenceCpuError> {
    HostTensor::new(shape.iter().map(|&n| n as u64).collect::<Vec<_>>(), data)
}

fn project(
    hidden: &HostTensor,
    weights: &HostTensor,
    tokens: usize,
    width: usize,
    columns: usize,
) -> Result<Vec<f32>, ReferenceCpuError> {
    let mut result = vec![0.0; product(&[tokens, columns])?];
    for token in 0..tokens {
        for column in 0..columns {
            let mut value = 0.0;
            for dim in 0..width {
                value += hidden.data[token * width + dim] * weights.data[dim * columns + column];
            }
            result[token * columns + column] = value;
        }
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite attention projection"));
    }
    Ok(result)
}

fn normalize(
    values: &mut [f32],
    weights: &[f32],
    epsilon: f32,
    zero_centered: bool,
) -> Result<(), ReferenceCpuError> {
    let width = weights.len();
    for head in values.chunks_exact_mut(width) {
        let variance = head.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let stabilized = variance + epsilon;
        if !stabilized.is_finite() {
            return Err(invalid("non-finite attention normalization"));
        }
        let inverse = stabilized.sqrt().recip();
        for (value, weight) in head.iter_mut().zip(weights) {
            *value *= inverse * if zero_centered { 1.0 + weight } else { *weight };
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite normalized attention"));
    }
    Ok(())
}

/// Inputs: hidden; previous raw K/V; prior K/V attention histories (K is
/// normalized, V is shifted); positions;
/// Q, A_K, B_K, A_V, B_V, shift-K, shift-V and lambda matrices; Q/K RMS weights;
/// and per-query-head position scalers. Outputs: attention, raw K/V and histories.
pub fn projected_differential_attention(
    inputs: &[HostTensor],
    config: ProjectedDifferentialAttention,
) -> Result<[HostTensor; 5], ReferenceCpuError> {
    if inputs.len() != 17 {
        return Err(invalid("projected attention requires 17 inputs"));
    }
    validate_config(config)?;
    let [tokens, hidden_width] = inputs[0].shape.as_slice() else {
        return Err(invalid(
            "projected attention hidden values must be rank two",
        ));
    };
    let t = usize::try_from(*tokens).map_err(|_| invalid("token count overflow"))?;
    let width = usize::try_from(*hidden_width).map_err(|_| invalid("hidden width overflow"))?;
    let ProjectedDifferentialAttention {
        query_heads: hq,
        kv_heads: hk,
        head_dim: d,
        rank: r,
        past_length: past,
        epsilon,
        zero_centered_norm,
        attention,
    } = config;
    if t == 0
        || width == 0
        || past
            .checked_add(t)
            .is_none_or(|n| n - 1 > MAX_EXACT_POSITION)
    {
        return Err(invalid("invalid projected attention sequence"));
    }
    let total = past + t;
    let (q_width, a_width, b_width) = (product(&[hq, d])?, product(&[hk, r])?, product(&[r, d])?);
    let shape = [
        vec![t, width],
        vec![hk, d],
        vec![hk, d],
        vec![past.max(1), hk, d],
        vec![past.max(1), hk, d],
        vec![t],
        vec![width, q_width],
        vec![width, a_width],
        vec![width, b_width],
        vec![width, a_width],
        vec![width, b_width],
        vec![width, hk],
        vec![width, hk],
        vec![width, hk],
        vec![d],
        vec![d],
        vec![hq],
    ];
    for (input, shape) in inputs.iter().zip(&shape) {
        check(input, shape)?;
    }
    for (index, &position) in inputs[5].data.iter().enumerate() {
        if position != (past + index) as f32 {
            return Err(invalid(
                "projected attention positions must be contiguous from past length",
            ));
        }
    }
    let mut query = project(&inputs[0], &inputs[6], t, width, q_width)?;
    let a_key = project(&inputs[0], &inputs[7], t, width, a_width)?;
    let b_key = project(&inputs[0], &inputs[8], t, width, b_width)?;
    let a_value = project(&inputs[0], &inputs[9], t, width, a_width)?;
    let b_value = project(&inputs[0], &inputs[10], t, width, b_width)?;
    let logits_key = project(&inputs[0], &inputs[11], t, width, hk)?;
    let logits_value = project(&inputs[0], &inputs[12], t, width, hk)?;
    let lambda = project(&inputs[0], &inputs[13], t, width, hk)?;
    let mut raw_key = vec![0.0; product(&[t, hk, d])?];
    let mut raw_value = vec![0.0; raw_key.len()];
    for token in 0..t {
        for head in 0..hk {
            for dim in 0..d {
                let mut key = 0.0;
                let mut value = 0.0;
                for rank in 0..r {
                    let a = token * a_width + head * r + rank;
                    let b = token * b_width + rank * d + dim;
                    key += a_key[a] * b_key[b];
                    value += a_value[a] * b_value[b];
                }
                let index = (token * hk + head) * d + dim;
                raw_key[index] = key / r as f32;
                raw_value[index] = value / r as f32;
            }
        }
    }
    let starts: Vec<f32> = (0..t)
        .map(|index| if past + index == 0 { 1.0 } else { 0.0 })
        .collect();
    let shift = |raw: Vec<f32>, logits: Vec<f32>, previous: &HostTensor| {
        token_shift(&[
            tensor(&[t, hk, d], raw)?,
            tensor(&[t, hk], logits)?,
            tensor(&[t], starts.clone())?,
            previous.clone(),
        ])
    };
    let [mut key, raw_key] = shift(raw_key, logits_key, &inputs[1])?;
    let [value, raw_value] = shift(raw_value, logits_value, &inputs[2])?;
    normalize(&mut query, &inputs[14].data, epsilon, zero_centered_norm)?;
    normalize(&mut key.data, &inputs[15].data, epsilon, zero_centered_norm)?;
    for token in 0..t {
        let position = (past + token) as f32 + 1.0;
        let position = match attention.window_left {
            Some(window) if window > 0 => position.min(window as f32),
            _ => position,
        };
        let log_position = position.ln();
        for (head, &head_scale) in inputs[16].data.iter().enumerate() {
            let scale = head_scale * log_position;
            let start = (token * hq + head) * d;
            for value in &mut query[start..start + d] {
                *value *= scale;
            }
        }
    }
    let history = |old: &HostTensor, new: HostTensor| -> Result<HostTensor, ReferenceCpuError> {
        let mut data = Vec::with_capacity(product(&[total, hk, d])?);
        if past > 0 {
            data.extend_from_slice(&old.data);
        }
        data.extend(new.data);
        tensor(&[total, hk, d], data)
    };
    let key_history = history(&inputs[3], key)?;
    let value_history = history(&inputs[4], value)?;
    let result = differential_attention(
        &[
            tensor(&[t, hq, d], query)?,
            key_history.clone(),
            value_history.clone(),
            tensor(&[t, hk], lambda)?,
        ],
        attention,
    )?;
    Ok([result, raw_key, raw_value, key_history, value_history])
}
