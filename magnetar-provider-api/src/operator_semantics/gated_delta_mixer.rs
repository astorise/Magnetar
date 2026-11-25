//! Portable f32 interface for an explicit-state gated-delta mixer.

use std::collections::BTreeMap;

use crate::{
    operator::OperatorAttributeValue,
    tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode},
};

use super::gated_delta_mixer_core::{
    MixerConfig, MixerGateWeights, MixerProjections, MixerState, MixerWeights, mixer_f32,
    mixer_recurrence_f32,
};

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

/// The gated-delta mixer between its input and output projections, for a
/// Provider that computes the projections itself: `projections` are the
/// chunk's `[tokens, width]` QKV, Z, A and B projections (a masked token's
/// all zero), `states` the prior convolution and recurrent states, and
/// `weights` the depthwise convolution, `A_log`, `dt_bias` and the norm
/// weight, in the Operator's shapes. Returns the gated RMSNorm output
/// `[tokens, value_heads * value_dim]`, which the output projection
/// consumes, and the next convolution and recurrent states: the same
/// values [`gated_delta_mixer`] computes from the same projections.
pub fn gated_delta_mixer_from_projections(
    projections: [&[f32]; 4],
    tokens: u64,
    states: [&HostTensor; 2],
    weights: [&HostTensor; 4],
    config: GatedDeltaMixer,
) -> Result<[HostTensor; 3], ReferenceCpuError> {
    let kh = config.key_heads as u64;
    let vh = config.value_heads as u64;
    let kd = config.key_dim as u64;
    let vd = config.value_dim as u64;
    let kernel = config.conv_kernel as u64;
    let value_width = vh
        .checked_mul(vd)
        .ok_or_else(|| invalid("geometry overflow"))?;
    let channels = kh
        .checked_mul(kd)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(value_width))
        .ok_or_else(|| invalid("geometry overflow"))?;
    let shapes = [
        vec![channels, kernel],
        vec![vh, kd, vd],
        vec![channels, kernel],
        vec![vh],
        vec![vh],
        vec![vd],
    ];
    for (input, shape) in states.iter().chain(&weights).zip(&shapes) {
        check(input, shape)?;
    }
    let [qkv, z, a, b] = projections;
    let recurrence = mixer_recurrence_f32(
        MixerConfig {
            hidden: 0,
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            conv_kernel: config.conv_kernel,
            norm_epsilon: config.epsilon,
        },
        usize::try_from(tokens).map_err(|_| invalid("token count too large"))?,
        MixerProjections { qkv, z, a, b },
        MixerGateWeights {
            convolution: &weights[0].data,
            a_log: &weights[1].data,
            dt_bias: &weights[2].data,
            norm: &weights[3].data,
        },
        Some(MixerState {
            convolution: &states[0].data,
            recurrent: &states[1].data,
        }),
    )
    .map_err(|error| invalid(&error.to_string()))?;
    Ok([
        HostTensor::new([tokens, value_width], recurrence.normalized)?,
        HostTensor::new(shapes[0].clone(), recurrence.convolution_state)?,
        HostTensor::new(shapes[1].clone(), recurrence.recurrent_state)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(count: u64, seed: u32, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let mixed = (i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed) % 2001;
                (mixed as f32 - 1000.0) / 1000.0 * scale
            })
            .collect()
    }

    fn tensor(shape: &[u64], seed: u32, scale: f32) -> HostTensor {
        HostTensor::new(shape.to_vec(), values(shape.iter().product(), seed, scale)).unwrap()
    }

    /// `[rows, in] x [out, in]ᵀ`, each sum in input order (as the
    /// Reference's own projections).
    fn project(x: &[f32], weight: &HostTensor) -> Vec<f32> {
        let width = weight.shape[1] as usize;
        x.chunks_exact(width)
            .flat_map(|row| {
                weight
                    .data
                    .chunks_exact(width)
                    .map(|w| w.iter().zip(row).map(|(w, x)| w * x).sum::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// A Provider projecting the inputs itself and passing the projections
    /// to `gated_delta_mixer_from_projections` gets exactly the
    /// Reference's outputs, masked tokens and next states included.
    #[test]
    fn the_mixer_from_its_projections_equals_the_whole_mixer() {
        let config = GatedDeltaMixer {
            key_heads: 2,
            value_heads: 4,
            key_dim: 8,
            value_dim: 4,
            conv_kernel: 4,
            epsilon: 1e-6,
        };
        let (tokens, hidden, channels, value_width) = (3u64, 12u64, 48u64, 16u64);
        let mask = HostTensor::new([tokens], vec![1.0, 0.0, 1.0]).unwrap();
        let inputs = vec![
            tensor(&[tokens, hidden], 1, 1.0),
            tensor(&[channels, 4], 2, 0.5),
            tensor(&[4, 8, 4], 3, 0.2),
            tensor(&[channels, hidden], 4, 0.3),
            tensor(&[value_width, hidden], 5, 0.3),
            tensor(&[4, hidden], 6, 0.3),
            tensor(&[4, hidden], 7, 0.3),
            tensor(&[channels, 4], 8, 0.5),
            tensor(&[4], 9, 0.5),
            tensor(&[4], 10, 0.5),
            tensor(&[4], 11, 0.2),
            tensor(&[hidden, value_width], 12, 0.3),
            mask.clone(),
        ];
        let whole = gated_delta_mixer(&inputs, config).unwrap();

        let mut x = inputs[0].data.clone();
        for (row, &keep) in x.chunks_exact_mut(hidden as usize).zip(&mask.data) {
            if keep == 0.0 {
                row.fill(0.0);
            }
        }
        let projections = [3, 4, 5, 6].map(|index| project(&x, &inputs[index]));
        let [normalized, convolution, recurrent] = gated_delta_mixer_from_projections(
            [
                &projections[0],
                &projections[1],
                &projections[2],
                &projections[3],
            ],
            tokens,
            [&inputs[1], &inputs[2]],
            [&inputs[7], &inputs[8], &inputs[9], &inputs[10]],
            config,
        )
        .unwrap();
        assert_eq!(normalized.shape, [tokens, value_width]);
        assert_eq!(project(&normalized.data, &inputs[11]), whole[0].data);
        assert_eq!(convolution, whole[1]);
        assert_eq!(recurrent, whole[2]);
        // Mismatched projections are refused.
        assert!(
            gated_delta_mixer_from_projections(
                [
                    &projections[0][1..],
                    &projections[1],
                    &projections[2],
                    &projections[3],
                ],
                tokens,
                [&inputs[1], &inputs[2]],
                [&inputs[7], &inputs[8], &inputs[9], &inputs[10]],
                config,
            )
            .is_err()
        );
    }
}
