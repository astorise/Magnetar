//! Biasless projected Mamba-3 mixer, composed with the explicit recurrence.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::{Mamba3Mimo, mamba3_mimo, mamba3_mimo_from_attributes};

const MAX_ELEMENTS: usize = 67_108_864;

/// Configuration for the single-group, biasless projection profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba3Mixer {
    pub geometry: Mamba3Mimo,
    pub epsilon: f32,
    pub decay_floor: f32,
    pub zero_centered_norm: bool,
}

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Decode all required geometry and normalization attributes.
pub fn mamba3_mixer_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<Mamba3Mixer, ReferenceCpuError> {
    let geometry = mamba3_mimo_from_attributes(attributes)?;
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite() && (*value as f32).is_finite() && (*value as f32) > 0.0 =>
        {
            Ok(*value as f32)
        }
        _ => Err(invalid(format!(
            "mamba3-mixer {name} must be positive and finite"
        ))),
    };
    let zero_centered_norm = match attributes.get("zero_centered_norm") {
        Some(OperatorAttributeValue::Boolean(value)) => *value,
        _ => return Err(invalid("mamba3-mixer zero_centered_norm must be a boolean")),
    };
    if geometry.groups != 1
        || geometry.angles == 0
        || geometry.angles != (geometry.state_dim / 2) / 2
    {
        return Err(invalid(
            "mamba3-mixer requires one group and the supported rotary fraction",
        ));
    }
    Ok(Mamba3Mixer {
        geometry,
        epsilon: positive("epsilon")?,
        decay_floor: positive("decay_floor")?,
        zero_centered_norm,
    })
}

fn product(dimensions: &[usize]) -> Result<usize, ReferenceCpuError> {
    dimensions.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&value| value <= MAX_ELEMENTS)
            .ok_or_else(|| invalid("mamba3-mixer tensor is too large"))
    })
}

fn check(tensor: &HostTensor, shape: &[usize]) -> Result<(), ReferenceCpuError> {
    let count = product(shape)?;
    if tensor.shape.len() != shape.len()
        || tensor
            .shape
            .iter()
            .zip(shape)
            .any(|(&actual, &expected)| actual != expected as u64)
        || tensor.data.len() != count
        || tensor.data.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "mamba3-mixer input has an invalid shape or non-finite value",
        ));
    }
    Ok(())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

fn tensor(shape: &[usize], values: Vec<f32>) -> Result<HostTensor, ReferenceCpuError> {
    HostTensor::new(
        shape.iter().map(|&value| value as u64).collect::<Vec<_>>(),
        values,
    )
}

/// Run projected, normalized Mamba-3 on one sequence. Inputs are hidden,
/// four prior states, the two projection matrices, dt bias, B/C normalization
/// weights, Q/K biases, three MIMO projections and skip weights. Returns
/// flattened activation followed by four next states. Inputs never mutate.
pub fn mamba3_mixer(
    inputs: &[HostTensor],
    config: Mamba3Mixer,
) -> Result<[HostTensor; 5], ReferenceCpuError> {
    if inputs.len() != 16 {
        return Err(invalid("mamba3-mixer requires 16 inputs"));
    }
    let [tokens, hidden] = inputs[0].shape.as_slice() else {
        return Err(invalid("mamba3-mixer hidden input must be rank 2"));
    };
    let tokens =
        usize::try_from(*tokens).map_err(|_| invalid("mamba3-mixer token count is too large"))?;
    let hidden =
        usize::try_from(*hidden).map_err(|_| invalid("mamba3-mixer hidden size is too large"))?;
    let Mamba3Mimo {
        heads: h,
        groups: g,
        rank: r,
        head_dim: p,
        state_dim: n,
        angles: a,
    } = config.geometry;
    config
        .geometry
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if tokens == 0
        || hidden == 0
        || g != 1
        || a == 0
        || a != (n / 2) / 2
        || product(&[h, p])? != product(&[2, hidden])?
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
        || !config.decay_floor.is_finite()
        || config.decay_floor <= 0.0
    {
        return Err(invalid(
            "invalid mamba3-mixer geometry or numerical profile",
        ));
    }
    let inner = product(&[h, p])?;
    let per_head = p
        .checked_mul(2)
        .and_then(|value| value.checked_add(3))
        .ok_or_else(|| invalid("mamba3-mixer projection is too large"))?;
    let projected = product(&[h, per_head])?;
    let rank_state = product(&[r, n])?;
    let dynamic = rank_state
        .checked_mul(2)
        .and_then(|value| value.checked_add(a))
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or_else(|| invalid("mamba3-mixer dynamic projection is too large"))?;
    let shapes = [
        vec![tokens, hidden],
        vec![h, a],
        vec![h, p, n],
        vec![r, h, n],
        vec![h, p],
        vec![hidden, projected],
        vec![hidden, dynamic],
        vec![h],
        vec![n],
        vec![n],
        vec![r, h, n],
        vec![r, h, n],
        vec![r, h, p],
        vec![r, h, p],
        vec![r, h, p],
        vec![h],
    ];
    for (input, shape) in inputs.iter().zip(&shapes) {
        check(input, shape)?;
    }
    let mut projected_values = vec![0.0; product(&[tokens, projected])?];
    let mut dynamic_values = vec![0.0; product(&[tokens, dynamic])?];
    for row in 0..tokens {
        for column in 0..projected {
            let mut value = 0.0;
            for inner_index in 0..hidden {
                value += inputs[0].data[row * hidden + inner_index]
                    * inputs[5].data[inner_index * projected + column];
            }
            projected_values[row * projected + column] = value;
        }
        for column in 0..dynamic {
            let mut value = 0.0;
            for inner_index in 0..hidden {
                value += inputs[0].data[row * hidden + inner_index]
                    * inputs[6].data[inner_index * dynamic + column];
            }
            dynamic_values[row * dynamic + column] = value;
        }
    }
    if projected_values
        .iter()
        .chain(&dynamic_values)
        .any(|value| !value.is_finite())
    {
        return Err(invalid("non-finite mamba3-mixer projection"));
    }
    let mut query = vec![0.0; product(&[tokens, r, g, n])?];
    let mut key = vec![0.0; query.len()];
    let mut value = vec![0.0; product(&[tokens, h, p])?];
    let mut gate = vec![0.0; value.len()];
    let mut decay = vec![0.0; product(&[tokens, h])?];
    let mut dt = vec![0.0; decay.len()];
    let mut trap = vec![0.0; decay.len()];
    let mut angle = vec![0.0; product(&[tokens, h, a])?];
    for row in 0..tokens {
        for (head, &dt_bias) in inputs[7].data.iter().enumerate() {
            let base = row * projected + head * per_head;
            let hp = (row * h + head) * p;
            gate[hp..hp + p].copy_from_slice(&projected_values[base..base + p]);
            value[hp..hp + p].copy_from_slice(&projected_values[base + p..base + 2 * p]);
            let scalar = row * h + head;
            dt[scalar] = softplus(projected_values[base + 2 * p] + dt_bias);
            decay[scalar] = -softplus(projected_values[base + 2 * p + 1]).max(config.decay_floor);
            trap[scalar] = sigmoid(projected_values[base + 2 * p + 2]);
            let offset = (row * h + head) * a;
            angle[offset..offset + a].copy_from_slice(
                &dynamic_values[row * dynamic + 2 * rank_state..row * dynamic + 2 * rank_state + a],
            );
        }
        for rank in 0..r {
            for (target, offset, weight) in [
                (&mut key, 0, &inputs[8].data),
                (&mut query, rank_state, &inputs[9].data),
            ] {
                let start = row * dynamic + offset + rank * n;
                let raw = &dynamic_values[start..start + n];
                let variance = raw.iter().map(|value| value * value).sum::<f32>() / n as f32;
                let stabilized = variance + config.epsilon;
                if !stabilized.is_finite() {
                    return Err(invalid("non-finite mamba3-mixer normalization"));
                }
                let inverse = stabilized.sqrt().recip();
                for dim in 0..n {
                    let scale = if config.zero_centered_norm {
                        1.0 + weight[dim]
                    } else {
                        weight[dim]
                    };
                    target[(row * r + rank) * n + dim] = raw[dim] * inverse * scale;
                }
            }
        }
    }
    let recurrence_inputs = vec![
        tensor(&[tokens, r, g, n], query)?,
        tensor(&[tokens, r, g, n], key)?,
        tensor(&[tokens, h, p], value)?,
        tensor(&[tokens, h, p], gate)?,
        tensor(&[tokens, h], decay)?,
        tensor(&[tokens, h], dt)?,
        tensor(&[tokens, h], trap)?,
        tensor(&[tokens, h, a], angle)?,
        inputs[1].clone(),
        inputs[2].clone(),
        inputs[3].clone(),
        inputs[4].clone(),
        inputs[10].clone(),
        inputs[11].clone(),
        inputs[12].clone(),
        inputs[13].clone(),
        inputs[14].clone(),
        inputs[15].clone(),
    ];
    let [activation, angle, ssm, key, input] = mamba3_mimo(&recurrence_inputs, config.geometry)?;
    Ok([
        tensor(&[tokens, inner], activation.data)?,
        angle,
        ssm,
        key,
        input,
    ])
}
