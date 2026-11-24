//! Portable f32 interface for an explicit-state gated-delta mixer.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::gated_delta_mixer_core::{MixerConfig, MixerState, MixerWeights, mixer_f32};

const MAX_ELEMENTS: u64 = 67_108_864;

/// Geometry and normalization for the single-sequence Qwen3.5-style profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaMixer {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f32,
}

fn invalid(reason: &str) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
}

pub fn gated_delta_mixer_from_attributes(
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<GatedDeltaMixer, ReferenceCpuError> {
    let positive = |name: &str| match attributes.get(name) {
        Some(OperatorAttributeValue::Integer(value)) if *value > 0 => {
            usize::try_from(*value).map_err(|_| invalid("gated-delta geometry is too large"))
        }
        _ => Err(invalid("gated-delta geometry must be positive")),
    };
    let epsilon = match attributes.get("epsilon") {
        Some(OperatorAttributeValue::Float(value))
            if value.is_finite() && (*value as f32).is_finite() && *value as f32 > 0.0 =>
        {
            *value as f32
        }
        _ => return Err(invalid("gated-delta epsilon must be positive and finite")),
    };
    let config = GatedDeltaMixer {
        key_heads: positive("key_heads")?,
        value_heads: positive("value_heads")?,
        key_dim: positive("key_dim")?,
        value_dim: positive("value_dim")?,
        conv_kernel: positive("conv_kernel")?,
        epsilon,
    };
    if !config.value_heads.is_multiple_of(config.key_heads) {
        return Err(invalid(
            "gated-delta value heads must be grouped over key heads",
        ));
    }
    Ok(config)
}

fn check(tensor: &HostTensor, shape: &[u64]) -> Result<(), ReferenceCpuError> {
    let count = shape.iter().try_fold(1u64, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&value| dim > 0 && value <= MAX_ELEMENTS)
    });
    if tensor.shape != shape
        || count.and_then(|value| usize::try_from(value).ok()) != Some(tensor.data.len())
        || tensor.data.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid gated-delta input shape or values"));
    }
    Ok(())
}

/// Inputs: hidden, prior convolution, prior recurrence, QKV/Z/A/B matrices,
/// depthwise convolution, A_log, dt_bias, normalization, output matrix and
/// a 0/1 attention mask. Projection matrices use checkpoint order [out, in].
/// Outputs: hidden activation and the two next states, in the same shapes as
/// the first three inputs. Initial state is explicit, including at reset.
pub fn gated_delta_mixer(
    inputs: &[HostTensor],
    config: GatedDeltaMixer,
) -> Result<[HostTensor; 3], ReferenceCpuError> {
    // Twelve inputs: no token mask, every token is valid.
    if !(12..=13).contains(&inputs.len()) {
        return Err(invalid(
            "gated-delta mixer requires twelve or thirteen inputs",
        ));
    }
    let [tokens, hidden] = inputs[0].shape.as_slice() else {
        return Err(invalid("gated-delta hidden input must be rank two"));
    };
    let kh = config.key_heads as u64;
    let vh = config.value_heads as u64;
    let kd = config.key_dim as u64;
    let vd = config.value_dim as u64;
    let kernel = config.conv_kernel as u64;
    let key_width = kh
        .checked_mul(kd)
        .ok_or_else(|| invalid("geometry overflow"))?;
    let value_width = vh
        .checked_mul(vd)
        .ok_or_else(|| invalid("geometry overflow"))?;
    let channels = key_width
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_width))
        .ok_or_else(|| invalid("geometry overflow"))?;
    if kh == 0
        || vh == 0
        || !vh.is_multiple_of(kh)
        || kd == 0
        || vd == 0
        || kernel == 0
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
    {
        return Err(invalid("invalid gated-delta geometry"));
    }
    let shapes = [
        vec![*tokens, *hidden],
        vec![channels, kernel],
        vec![vh, kd, vd],
        vec![channels, *hidden],
        vec![value_width, *hidden],
        vec![vh, *hidden],
        vec![vh, *hidden],
        vec![channels, kernel],
        vec![vh],
        vec![vh],
        vec![vd],
        vec![*hidden, value_width],
        vec![*tokens],
    ];
    for (input, shape) in inputs.iter().zip(&shapes) {
        check(input, shape)?;
    }
    let mask = inputs
        .get(12)
        .map(|mask| {
            mask.data
                .iter()
                .map(|value| match *value {
                    0.0 => Ok(0),
                    1.0 => Ok(1),
                    _ => Err(invalid("gated-delta mask must contain only zero or one")),
                })
                .collect::<Result<Vec<u8>, _>>()
        })
        .transpose()?;
    let output = mixer_f32(
        MixerConfig {
            hidden: usize::try_from(*hidden).map_err(|_| invalid("hidden size too large"))?,
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            conv_kernel: config.conv_kernel,
            norm_epsilon: config.epsilon,
        },
        usize::try_from(*tokens).map_err(|_| invalid("token count too large"))?,
        &inputs[0].data,
        mask.as_deref(),
        MixerWeights {
            in_qkv: &inputs[3].data,
            in_z: &inputs[4].data,
            in_a: &inputs[5].data,
            in_b: &inputs[6].data,
            convolution: &inputs[7].data,
            a_log: &inputs[8].data,
            dt_bias: &inputs[9].data,
            norm: &inputs[10].data,
            out: &inputs[11].data,
        },
        Some(MixerState {
            convolution: &inputs[1].data,
            recurrent: &inputs[2].data,
        }),
    )
    .map_err(|error| invalid(&error.to_string()))?;
    Ok([
        HostTensor::new(shapes[0].clone(), output.values)?,
        HostTensor::new(shapes[1].clone(), output.convolution_state)?,
        HostTensor::new(shapes[2].clone(), output.recurrent_state)?,
    ])
}
