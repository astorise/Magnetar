//! F32 elementwise remainder with an explicitly selected positive divisor.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

fn invalid(message: &str) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

/// Read a finite positive divisor that remains finite and positive in f32.
pub fn remainder_scalar_divisor(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<f32, ReferenceCpuError> {
    match attributes.get("divisor") {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite() && (*value as f32).is_finite() && (*value as f32) > 0.0 =>
        {
            Ok(*value as f32)
        }
        _ => Err(invalid(
            "remainder-scalar requires a positive finite f32 divisor",
        )),
    }
}

/// Return the divisor-sign f32 remainder of each finite input element.
/// The input is not mutated; f32 rounding may return the divisor near zero.
pub fn remainder_scalar(input: &HostTensor, divisor: f32) -> Result<HostTensor, ReferenceCpuError> {
    if !divisor.is_finite() || divisor <= 0.0 || input.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("invalid remainder-scalar input or divisor"));
    }
    let output = input
        .data
        .iter()
        .map(|value| value.rem_euclid(divisor))
        .collect::<Vec<_>>();
    HostTensor::new(input.shape.clone(), output)
}

#[cfg(test)]
mod tests;
