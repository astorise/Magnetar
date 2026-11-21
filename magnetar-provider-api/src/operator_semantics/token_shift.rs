//! Pure token-shift mixing with an explicit previous raw-token state.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

const MAX_ELEMENTS: usize = 67_108_864;

fn invalid(message: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, message)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Mix `[tokens, heads, width]` raw values with the preceding raw value,
/// using `[tokens, heads]` gate logits and `[tokens]` document-start flags.
/// The fourth input is `[heads, width]` previous raw state. A start flag of
/// one suppresses the previous value and the gate for that token. Returns the
/// mixed sequence and the final *raw* token, without mutating any input.
pub fn token_shift(inputs: &[HostTensor]) -> Result<[HostTensor; 2], ReferenceCpuError> {
    let [raw, logits, starts, previous] = inputs else {
        return Err(invalid("token-shift requires four inputs"));
    };
    let [tokens, heads, width] = raw.shape.as_slice() else {
        return Err(invalid("token-shift raw values must be rank three"));
    };
    if *tokens == 0 || *heads == 0 || *width == 0 {
        return Err(invalid("token-shift dimensions must be positive"));
    }
    let tokens = usize::try_from(*tokens).map_err(|_| invalid("token-shift dimension overflow"))?;
    let heads = usize::try_from(*heads).map_err(|_| invalid("token-shift dimension overflow"))?;
    let width = usize::try_from(*width).map_err(|_| invalid("token-shift dimension overflow"))?;
    let product = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, dim| {
                n.checked_mul(*dim).filter(|&size| size <= MAX_ELEMENTS)
            })
            .ok_or_else(|| invalid("token-shift dimensions exceed the reference limit"))
    };
    let row = product(&[heads, width])?;
    let expected = [
        (
            raw,
            vec![tokens as u64, heads as u64, width as u64],
            product(&[tokens, row])?,
        ),
        (
            logits,
            vec![tokens as u64, heads as u64],
            product(&[tokens, heads])?,
        ),
        (starts, vec![tokens as u64], tokens),
        (previous, vec![heads as u64, width as u64], row),
    ];
    for (tensor, shape, len) in expected {
        if tensor.shape != shape
            || tensor.data.len() != len
            || tensor.data.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "token-shift input has an invalid shape or non-finite value",
            ));
        }
    }
    if starts.data.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err(invalid(
            "token-shift document-start flags must be zero or one",
        ));
    }
    let mut mixed = vec![0.0; raw.data.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let alpha = if starts.data[token] == 1.0 {
                0.0
            } else {
                sigmoid(logits.data[token * heads + head])
            };
            for dim in 0..width {
                let index = token * row + head * width + dim;
                let before = if token == 0 {
                    previous.data[head * width + dim]
                } else {
                    raw.data[index - row]
                };
                mixed[index] = alpha * before + (1.0 - alpha) * raw.data[index];
            }
        }
    }
    if mixed.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite token-shift output"));
    }
    Ok([
        HostTensor::new(raw.shape.clone(), mixed)?,
        HostTensor::new(
            previous.shape.clone(),
            raw.data[(tokens - 1) * row..].to_vec(),
        )?,
    ])
}
