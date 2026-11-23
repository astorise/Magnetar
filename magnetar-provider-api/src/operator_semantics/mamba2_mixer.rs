//! Pure, single-sequence Mamba-2 mixer with explicit input and output state.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

// Bounds each temporary and tensor before allocation. The published
// Nemotron-H 30B mixer input projection has fewer than 28 million elements.
const MAX_ELEMENTS: usize = 67_108_864;

/// Geometry and numerical parameters of `mamba2-mixer@1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Mixer {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state_dim: usize,
    pub kernel: usize,
    pub epsilon: f32,
    pub dt_min: f32,
    pub dt_max: f32,
    pub conv_bias: bool,
    pub linear_bias: bool,
}

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Decode all attributes explicitly; malformed or missing values are errors.
pub fn mamba2_mixer_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<Mamba2Mixer, ReferenceCpuError> {
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => usize::try_from(*value)
            .map_err(|_| invalid(format!("mamba2-mixer {name} exceeds platform limits"))),
        _ => Err(invalid(format!("mamba2-mixer {name} must be positive"))),
    };
    let float = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite() && (*value as f32).is_finite() =>
        {
            Ok(*value as f32)
        }
        _ => Err(invalid(format!("mamba2-mixer {name} must be finite"))),
    };
    let boolean = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Boolean(value)) => Ok(*value),
        _ => Err(invalid(format!("mamba2-mixer {name} must be a boolean"))),
    };
    let dt_max = if attributes.contains_key("dt_max") {
        float("dt_max")?
    } else {
        f32::INFINITY
    };
    let config = Mamba2Mixer {
        heads: positive("heads")?,
        head_dim: positive("head_dim")?,
        groups: positive("groups")?,
        state_dim: positive("state_dim")?,
        kernel: positive("conv_kernel")?,
        epsilon: float("epsilon")?,
        dt_min: float("dt_min")?,
        dt_max,
        conv_bias: boolean("conv_bias")?,
        linear_bias: boolean("linear_bias")?,
    };
    config.validate()?;
    Ok(config)
}

impl Mamba2Mixer {
    fn validate(self) -> Result<(), ReferenceCpuError> {
        if self.heads == 0
            || self.head_dim == 0
            || self.groups == 0
            || self.state_dim == 0
            || self.kernel == 0
            || !self.heads.is_multiple_of(self.groups)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.dt_min.is_finite()
            || self.dt_min < 0.0
            || self.dt_max.is_nan()
            || self.dt_max < self.dt_min
        {
            return Err(invalid("invalid mamba2-mixer geometry or numerical limits"));
        }
        Ok(())
    }
}

fn product(a: usize, b: usize) -> Result<usize, ReferenceCpuError> {
    a.checked_mul(b)
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or_else(|| invalid("mamba2-mixer tensor is too large"))
}

fn sum(a: usize, b: usize) -> Result<usize, ReferenceCpuError> {
    a.checked_add(b)
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or_else(|| invalid("mamba2-mixer dimension is too large"))
}

fn validate_tensor(tensor: &HostTensor, shape: &[usize]) -> Result<(), ReferenceCpuError> {
    let expected = shape.iter().try_fold(1, |n, &dim| product(n, dim))?;
    if tensor.shape.len() != shape.len()
        || tensor
            .shape
            .iter()
            .zip(shape)
            .any(|(&actual, &dim)| actual != dim as u64)
        || tensor.data.len() != expected
        || tensor.data.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "mamba2-mixer input has an invalid shape or non-finite value",
        ));
    }
    Ok(())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Execute tokens against explicit state, returning activation, convolution
/// history and SSM state. Inputs are never mutated, including on failure.
/// Matrices use Magnetar's `[input, output]` convention. Optional inputs
/// follow the ten mandatory tensors: convolution bias when enabled, then
/// input and output projection biases when linear bias is enabled.
pub fn mamba2_mixer(
    inputs: &[HostTensor],
    config: Mamba2Mixer,
) -> Result<[HostTensor; 3], ReferenceCpuError> {
    config.validate()?;
    let count = 10 + usize::from(config.conv_bias) + 2 * usize::from(config.linear_bias);
    if inputs.len() != count {
        return Err(invalid(
            "mamba2-mixer input count does not match bias attributes",
        ));
    }
    let [tokens, hidden] = inputs[0].shape.as_slice() else {
        return Err(invalid("mamba2-mixer activation must be [tokens, hidden]"));
    };
    let tokens = usize::try_from(*tokens).map_err(|_| invalid("token count is too large"))?;
    let hidden = usize::try_from(*hidden).map_err(|_| invalid("hidden size is too large"))?;
    if tokens == 0 || hidden == 0 {
        return Err(invalid(
            "mamba2-mixer tokens and hidden size must be positive",
        ));
    }
    let inner = product(config.heads, config.head_dim)?;
    let grouped = product(config.groups, config.state_dim)?;
    let conv = sum(inner, product(2, grouped)?)?;
    let projection = sum(sum(inner, conv)?, config.heads)?;
    for (tensor, shape) in inputs[..10].iter().zip([
        vec![tokens, hidden],
        vec![conv, config.kernel],
        vec![config.heads, config.head_dim, config.state_dim],
        vec![hidden, projection],
        vec![conv, config.kernel],
        vec![config.heads],
        vec![config.heads],
        vec![config.heads],
        vec![inner],
        vec![inner, hidden],
    ]) {
        validate_tensor(tensor, &shape)?;
    }
    let mut optional = 10;
    let conv_bias = if config.conv_bias {
        validate_tensor(&inputs[optional], &[conv])?;
        optional += 1;
        Some(&inputs[optional - 1].data)
    } else {
        None
    };
    let (in_bias, out_bias) = if config.linear_bias {
        validate_tensor(&inputs[optional], &[projection])?;
        validate_tensor(&inputs[optional + 1], &[hidden])?;
        (
            Some(&inputs[optional].data),
            Some(&inputs[optional + 1].data),
        )
    } else {
        (None, None)
    };

    let mut history = inputs[1].data.clone();
    let mut state = inputs[2].data.clone();
    let mut output = vec![0.0; product(tokens, hidden)?];
    let mut projected = vec![0.0; projection];
    let mut convolved = vec![0.0; conv];
    let mut gated = vec![0.0; inner];
    for token in 0..tokens {
        for column in 0..projection {
            let mut value = in_bias.map_or(0.0, |bias| bias[column]);
            for row in 0..hidden {
                value += inputs[0].data[token * hidden + row]
                    * inputs[3].data[row * projection + column];
            }
            if !value.is_finite() {
                return Err(invalid("non-finite mamba2-mixer projection"));
            }
            projected[column] = value;
        }
        for channel in 0..conv {
            let start = channel * config.kernel;
            let row = &mut history[start..start + config.kernel];
            row.rotate_left(1);
            row[config.kernel - 1] = projected[inner + channel];
            let mut value = conv_bias.map_or(0.0, |bias| bias[channel]);
            for (&x, &weight) in row
                .iter()
                .zip(&inputs[4].data[start..start + config.kernel])
            {
                value += x * weight;
            }
            if !value.is_finite() {
                return Err(invalid("non-finite mamba2-mixer convolution"));
            }
            convolved[channel] = silu(value);
        }
        for head in 0..config.heads {
            let raw_dt = projected[inner + conv + head] + inputs[5].data[head];
            let a = -inputs[6].data[head].exp();
            if !raw_dt.is_finite() || !a.is_finite() {
                return Err(invalid("non-finite mamba2-mixer time step"));
            }
            let dt = (if raw_dt > 20.0 {
                raw_dt
            } else {
                raw_dt.exp().ln_1p()
            })
            .clamp(config.dt_min, config.dt_max);
            let exponent = dt * a;
            if !exponent.is_finite() {
                return Err(invalid("non-finite mamba2-mixer decay"));
            }
            let decay = exponent.exp();
            let group = head / (config.heads / config.groups);
            for dimension in 0..config.head_dim {
                let x_index = head * config.head_dim + dimension;
                let mut value = 0.0;
                for index in 0..config.state_dim {
                    let state_index = x_index * config.state_dim + index;
                    let bc_index = group * config.state_dim + index;
                    let next = state[state_index] * decay
                        + (dt * convolved[inner + bc_index]) * convolved[x_index];
                    if !next.is_finite() {
                        return Err(invalid("non-finite mamba2-mixer state"));
                    }
                    state[state_index] = next;
                    value += next * convolved[inner + grouped + bc_index];
                }
                value += convolved[x_index] * inputs[7].data[head];
                gated[x_index] = value * silu(projected[x_index]);
                if !gated[x_index].is_finite() {
                    return Err(invalid("non-finite mamba2-mixer gate"));
                }
            }
        }
        let group_size = inner / config.groups;
        for group in 0..config.groups {
            let offset = group * group_size;
            let values = &mut gated[offset..offset + group_size];
            let variance =
                values.iter().map(|value| value * value).sum::<f32>() / group_size as f32;
            if !variance.is_finite() {
                return Err(invalid("non-finite mamba2-mixer variance"));
            }
            let adjusted = variance + config.epsilon;
            if !adjusted.is_finite() {
                return Err(invalid("non-finite mamba2-mixer normalization"));
            }
            let inv = adjusted.sqrt().recip();
            for (index, value) in values.iter_mut().enumerate() {
                *value *= inv * inputs[8].data[offset + index];
            }
        }
        for column in 0..hidden {
            let mut value = out_bias.map_or(0.0, |bias| bias[column]);
            for row in 0..inner {
                value += gated[row] * inputs[9].data[row * hidden + column];
            }
            if !value.is_finite() {
                return Err(invalid("non-finite mamba2-mixer output"));
            }
            output[token * hidden + column] = value;
        }
    }
    Ok([
        HostTensor::new(vec![tokens as u64, hidden as u64], output)?,
        HostTensor::new(vec![conv as u64, config.kernel as u64], history)?,
        HostTensor::new(
            vec![
                config.heads as u64,
                config.head_dim as u64,
                config.state_dim as u64,
            ],
            state,
        )?,
    ])
}
