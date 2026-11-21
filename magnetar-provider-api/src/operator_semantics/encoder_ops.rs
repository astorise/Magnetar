//! Portable f32 primitives for encoder graphs.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Validate and convert the required positive LayerNorm epsilon.
pub fn affine_layernorm_epsilon(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<f32, ReferenceCpuError> {
    match attributes.get("epsilon") {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite()
                && *value > 0.0
                && (*value as f32).is_finite()
                && (*value as f32) > 0.0 =>
        {
            Ok(*value as f32)
        }
        _ => Err(invalid(
            "affine-layernorm epsilon must be finite and positive in f32",
        )),
    }
}

/// Row-wise population LayerNorm, learned scale and optional learned bias.
/// Inputs are `[rows, features]`, `[features]` and optionally `[features]`.
pub fn affine_layernorm(
    input: &HostTensor,
    scale: &HostTensor,
    bias: Option<&HostTensor>,
    epsilon: f32,
) -> Result<HostTensor, ReferenceCpuError> {
    let [rows, width] = input.shape.as_slice() else {
        return Err(invalid("affine-layernorm input must be [rows, features]"));
    };
    let rows = usize::try_from(*rows).map_err(|_| invalid("row count exceeds platform limits"))?;
    let width =
        usize::try_from(*width).map_err(|_| invalid("feature count exceeds platform limits"))?;
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("affine-layernorm shape overflows"))?;
    if rows == 0
        || width == 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.data.len() != count
        || scale.shape != [width as u64]
        || scale.data.len() != width
        || bias.is_some_and(|value| value.shape != [width as u64] || value.data.len() != width)
        || input
            .data
            .iter()
            .chain(&scale.data)
            .chain(bias.into_iter().flat_map(|value| &value.data))
            .any(|value| !value.is_finite())
    {
        return Err(invalid("invalid affine-layernorm inputs"));
    }
    let mut output = Vec::with_capacity(count);
    for row in input.data.chunks_exact(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / width as f32;
        let adjusted = variance + epsilon;
        if !mean.is_finite() || !adjusted.is_finite() {
            return Err(invalid("non-finite affine-layernorm statistics"));
        }
        let inverse = adjusted.sqrt().recip();
        for (index, (&value, &weight)) in row.iter().zip(&scale.data).enumerate() {
            let result =
                (value - mean) * inverse * weight + bias.map_or(0.0, |bias| bias.data[index]);
            if !result.is_finite() {
                return Err(invalid("non-finite affine-layernorm output"));
            }
            output.push(result);
        }
    }
    HostTensor::new(input.shape.clone(), output)
}

/// Error-function GELU (`approximate = none`). The error function uses a
/// five-coefficient approximation in f64 before conversion to f32; this is
/// the erf-defined GELU, distinct from the tanh approximation.
pub fn gelu_exact(input: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    if input.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("gelu-exact requires finite input"));
    }
    let output: Vec<_> = input
        .data
        .iter()
        .map(|&value| {
            let x = value as f64 * std::f64::consts::FRAC_1_SQRT_2;
            let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
            let polynomial = (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t
                - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t)
                * (-x * x).exp();
            let erf = (1.0 - polynomial).copysign(x);
            (0.5 * value as f64 * (1.0 + erf)) as f32
        })
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite gelu-exact output"));
    }
    HostTensor::new(input.shape.clone(), output)
}

/// Decode the required nonnegative start and positive slice width.
pub fn slice_last_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<(u64, u64), ReferenceCpuError> {
    let integer = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Ok(*value as u64),
        _ => Err(invalid(format!(
            "slice-last {name} must be a nonnegative integer"
        ))),
    };
    let start = integer("start")?;
    let width = integer("width")?;
    if width == 0 {
        return Err(invalid("slice-last width must be positive"));
    }
    Ok((start, width))
}

/// Select a contiguous feature range from each row, retaining row order.
pub fn slice_last(
    input: &HostTensor,
    start: u64,
    width: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let [rows, columns] = input.shape.as_slice() else {
        return Err(invalid("slice-last input must be rank two"));
    };
    if *rows == 0
        || *columns == 0
        || width == 0
        || start.checked_add(width).is_none_or(|end| end > *columns)
        || input.data.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid slice-last inputs or range"));
    }
    let rows = usize::try_from(*rows).map_err(|_| invalid("row count exceeds platform limits"))?;
    let columns =
        usize::try_from(*columns).map_err(|_| invalid("column count exceeds platform limits"))?;
    let start =
        usize::try_from(start).map_err(|_| invalid("slice start exceeds platform limits"))?;
    let width =
        usize::try_from(width).map_err(|_| invalid("slice width exceeds platform limits"))?;
    if rows.checked_mul(columns) != Some(input.data.len()) {
        return Err(invalid("invalid slice-last input length"));
    }
    let capacity = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("slice-last shape overflows"))?;
    let mut output = Vec::with_capacity(capacity);
    for row in input.data.chunks_exact(columns) {
        output.extend_from_slice(&row[start..start + width]);
    }
    HostTensor::new(vec![rows as u64, width as u64], output)
}
