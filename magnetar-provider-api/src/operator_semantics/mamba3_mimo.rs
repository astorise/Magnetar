//! Portable f32 Mamba-3 recurrence over a single sequence with explicit state.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::mamba3_core::{Geometry, Parameters, State, Token, step};

const MAX_ELEMENTS: usize = 67_108_864;

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Decode the mandatory Mamba-3 geometry. No family-specific defaults are
/// inferred from a missing or malformed graph attribute.
pub fn mamba3_mimo_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<Geometry, ReferenceCpuError> {
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => usize::try_from(*value)
            .map_err(|_| invalid(format!("mamba3-mimo {name} exceeds platform limits"))),
        _ => Err(invalid(format!("mamba3-mimo {name} must be positive"))),
    };
    let geometry = Geometry {
        heads: positive("heads")?,
        groups: positive("groups")?,
        rank: positive("rank")?,
        head_dim: positive("head_dim")?,
        state_dim: positive("state_dim")?,
        angles: positive("angles")?,
    };
    geometry
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(geometry)
}

fn check_tensor(tensor: &HostTensor, shape: &[usize]) -> Result<(), ReferenceCpuError> {
    let count = shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&value| value <= MAX_ELEMENTS)
            .ok_or_else(|| invalid("mamba3-mimo tensor is too large"))
    })?;
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
            "mamba3-mimo input has an invalid shape or non-finite value",
        ));
    }
    Ok(())
}

/// Evaluate preprocessed Q/K, decay, dt, trap, input and gate over one
/// sequence. The four explicit input states are immutable; outputs contain
/// activation and four new states in angle, SSM, key, input order.
/// Tensor shapes and input ordering are defined in the operator contract.
pub fn mamba3_mimo(
    inputs: &[HostTensor],
    geometry: Geometry,
) -> Result<[HostTensor; 5], ReferenceCpuError> {
    geometry
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if geometry.angles == 0 || inputs.len() != 18 {
        return Err(invalid(
            "mamba3-mimo requires positive rotary dimensions and 18 inputs",
        ));
    }
    let Some(&tokens) = inputs[0].shape.first() else {
        return Err(invalid("mamba3-mimo Q requires a token dimension"));
    };
    let tokens =
        usize::try_from(tokens).map_err(|_| invalid("mamba3-mimo token count is too large"))?;
    if tokens == 0 {
        return Err(invalid("mamba3-mimo token count must be positive"));
    }
    let Geometry {
        heads: h,
        groups: g,
        rank: r,
        head_dim: p,
        state_dim: n,
        angles: a,
    } = geometry;
    let shapes = [
        vec![tokens, r, g, n],
        vec![tokens, r, g, n],
        vec![tokens, h, p],
        vec![tokens, h, p],
        vec![tokens, h],
        vec![tokens, h],
        vec![tokens, h],
        vec![tokens, h, a],
        vec![h, a],
        vec![h, p, n],
        vec![r, h, n],
        vec![h, p],
        vec![r, h, n],
        vec![r, h, n],
        vec![r, h, p],
        vec![r, h, p],
        vec![r, h, p],
        vec![h],
    ];
    for (tensor, shape) in inputs.iter().zip(&shapes) {
        check_tensor(tensor, shape)?;
    }
    let parameters = Parameters {
        query_bias: inputs[12].data.clone(),
        key_bias: inputs[13].data.clone(),
        input_projection: inputs[14].data.clone(),
        gate_projection: inputs[15].data.clone(),
        output_projection: inputs[16].data.clone(),
        skip: inputs[17].data.clone(),
    };
    let mut state = State {
        angle: inputs[8].data.clone(),
        ssm: inputs[9].data.clone(),
        key: inputs[10].data.clone(),
        input: inputs[11].data.clone(),
    };
    let mut output = Vec::with_capacity(inputs[2].data.len());
    for index in 0..tokens {
        let slice = |input_index: usize| {
            let width = inputs[input_index].data.len() / tokens;
            &inputs[input_index].data[index * width..(index + 1) * width]
        };
        let token = Token {
            query: slice(0),
            key: slice(1),
            input: slice(2),
            gate: slice(3),
            decay: slice(4),
            dt: slice(5),
            trap: slice(6),
            angle_projection: slice(7),
        };
        let next = step(geometry, &parameters, &state, &token)
            .map_err(|error| invalid(error.to_string()))?;
        output.extend_from_slice(&next.output);
        state = next.state;
    }
    Ok([
        HostTensor::new(vec![tokens as u64, h as u64, p as u64], output)?,
        HostTensor::new(vec![h as u64, a as u64], state.angle)?,
        HostTensor::new(vec![h as u64, p as u64, n as u64], state.ssm)?,
        HostTensor::new(vec![r as u64, h as u64, n as u64], state.key)?,
        HostTensor::new(vec![h as u64, p as u64], state.input)?,
    ])
}
