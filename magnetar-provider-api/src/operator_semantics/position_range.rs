//! Contiguous absolute token positions represented exactly in f32.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

// The next-position operation used by attention must also remain exact.
const MAX_POSITION: u64 = 16_777_215;

fn invalid() -> ReferenceCpuError {
    ReferenceCpuError::new(
        ReferenceCpuErrorCode::ShapeUnsupported,
        "position-range requires nonempty [tokens, width] and exact f32 positions",
    )
}

/// Parse the required nonnegative starting position.
pub fn position_range_offset(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<u64, ReferenceCpuError> {
    match attributes.get("position_offset") {
        Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Ok(*value as u64),
        _ => Err(invalid()),
    }
}

/// Validate the anchor shape and the complete range before allocating.
pub fn position_range_validate(shape: &[u64], offset: u64) -> Result<u64, ReferenceCpuError> {
    let [tokens, width] = shape else {
        return Err(invalid());
    };
    if *tokens == 0
        || *width == 0
        || offset
            .checked_add(*tokens - 1)
            .is_none_or(|last| last > MAX_POSITION)
    {
        return Err(invalid());
    }
    Ok(*tokens)
}

/// Return `[offset, offset + 1, ...]`; anchor values are not read.
pub fn position_range(input: &HostTensor, offset: u64) -> Result<HostTensor, ReferenceCpuError> {
    let tokens = position_range_validate(&input.shape, offset)?;
    let values: Vec<_> = (0..tokens).map(|row| (offset + row) as f32).collect();
    HostTensor::new([tokens], values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_exact_and_invalid_ranges_are_rejected() {
        let input = HostTensor::new([3, 2], vec![f32::NAN; 6]).unwrap();
        assert_eq!(position_range(&input, 7).unwrap().data, [7.0, 8.0, 9.0]);
        assert_eq!(
            position_range(&input, MAX_POSITION - 2).unwrap().data,
            [16_777_213.0, 16_777_214.0, 16_777_215.0]
        );
        assert!(position_range(&input, MAX_POSITION - 1).is_err());
        assert!(position_range_validate(&[0, 2], 0).is_err());
        assert!(position_range_validate(&[1, 0], 0).is_err());
        assert!(position_range_validate(&[1], 0).is_err());
        assert!(position_range_validate(&[2, 1], u64::MAX).is_err());
        assert!(position_range_offset(&BTreeMap::new()).is_err());
        assert!(
            position_range_offset(&BTreeMap::from([(
                "position_offset".into(),
                OperatorAttributeValue::Integer(-1),
            )]))
            .is_err()
        );
    }
}
