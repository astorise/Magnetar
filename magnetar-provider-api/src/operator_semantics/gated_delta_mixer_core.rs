//! A single-sequence f32 Qwen3.5 linear-attention mixer reference.
//!
//! This composes supplied projection weights, causal convolution, the gated
//! delta recurrence and the gated normalization/output projection. It owns
//! no Runtime session, does not load or authenticate checkpoint weights,
//! and does not implement a complete Qwen3.5 model or a portable graph.

use super::gated_delta_core::{DeltaDimensions, DeltaInputs, gated_delta_f32};
use std::{error::Error, fmt};

const MAX_ELEMENTS: usize = 67_108_864;

/// The activation of the gate `z` that scales the normalized recurrence
/// output: `silu` for Qwen3.5, and `silu` or `sigmoid` for Qwen4-Exp
/// (`output_gate_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MixerOutputGate {
    #[default]
    Silu,
    Sigmoid,
}

impl MixerOutputGate {
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::Silu => silu(value),
            Self::Sigmoid => sigmoid(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixerConfig {
    pub hidden: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub norm_epsilon: f32,
    pub output_gate: MixerOutputGate,
}

/// All matrices are contiguous checkpoint-order [output, input].
pub struct MixerWeights<'a> {
    pub in_qkv: &'a [f32],
    pub in_z: &'a [f32],
    pub in_a: &'a [f32],
    pub in_b: &'a [f32],
    /// Depthwise convolution, [qkv_channels, conv_kernel].
    pub convolution: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub out: &'a [f32],
}

/// Persistent values from an earlier chunk. Both states must travel together.
pub struct MixerState<'a> {
    /// Raw projected QKV history, [qkv_channels, conv_kernel], oldest first.
    pub convolution: &'a [f32],
    /// [value_heads, key_dim, value_dim].
    pub recurrent: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixerOutput {
    /// [tokens, hidden].
    pub values: Vec<f32>,
    pub convolution_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

/// Compute one chunk. A mask entry of zero clears that input token before
/// its projections, as in the upstream single-sequence implementation.
pub fn mixer_f32(
    config: MixerConfig,
    tokens: usize,
    hidden: &[f32],
    mask: Option<&[u8]>,
    weights: MixerWeights<'_>,
    initial: Option<MixerState<'_>>,
) -> Result<MixerOutput, MixerError> {
    let product = |parts: &[usize]| {
        parts.iter().try_fold(1usize, |value, &part| {
            value
                .checked_mul(part)
                .filter(|&size| part > 0 && size <= MAX_ELEMENTS)
        })
    };
    let key_width = product(&[config.key_heads, config.key_dim]);
    let value_width = product(&[config.value_heads, config.value_dim]);
    let channels = key_width
        .zip(value_width)
        .and_then(|(key, value)| key.checked_mul(2)?.checked_add(value))
        .filter(|&value| value <= MAX_ELEMENTS);
    let invalid = || MixerError::InvalidDimensions;
    key_width.ok_or_else(invalid)?;
    let (value_width, channels) = (
        value_width.ok_or_else(invalid)?,
        channels.ok_or_else(invalid)?,
    );
    let conv_size = product(&[channels, config.conv_kernel]).ok_or_else(invalid)?;
    let recurrent_size =
        product(&[config.value_heads, config.key_dim, config.value_dim]).ok_or_else(invalid)?;
    let valid_len = |parts: &[usize], actual| product(parts) == Some(actual);
    if config.key_heads == 0
        || !config.value_heads.is_multiple_of(config.key_heads)
        || !valid_len(&[tokens, config.hidden], hidden.len())
        || !valid_len(&[channels, config.hidden], weights.in_qkv.len())
        || !valid_len(&[value_width, config.hidden], weights.in_z.len())
        || !valid_len(&[config.value_heads, config.hidden], weights.in_a.len())
        || !valid_len(&[config.value_heads, config.hidden], weights.in_b.len())
        || weights.convolution.len() != conv_size
        || weights.a_log.len() != config.value_heads
        || weights.dt_bias.len() != config.value_heads
        || weights.norm.len() != config.value_dim
        || !valid_len(&[config.hidden, value_width], weights.out.len())
        || product(&[tokens, channels]).is_none()
        || product(&[tokens, value_width]).is_none()
        || product(&[tokens, config.value_heads]).is_none()
        || mask.is_some_and(|mask| mask.len() != tokens || mask.iter().any(|&v| v > 1))
        || initial.as_ref().is_some_and(|state| {
            state.convolution.len() != conv_size || state.recurrent.len() != recurrent_size
        })
    {
        return Err(MixerError::InvalidDimensions);
    }
    if !config.norm_epsilon.is_finite()
        || config.norm_epsilon <= 0.0
        || hidden
            .iter()
            .chain(weights.in_qkv)
            .chain(weights.in_z)
            .chain(weights.in_a)
            .chain(weights.in_b)
            .chain(weights.convolution)
            .chain(weights.a_log)
            .chain(weights.dt_bias)
            .chain(weights.norm)
            .chain(weights.out)
            .chain(initial.as_ref().into_iter().flat_map(|s| s.convolution))
            .chain(initial.as_ref().into_iter().flat_map(|s| s.recurrent))
            .any(|value| !value.is_finite())
    {
        return Err(MixerError::NonFiniteValue);
    }

    // The four input projections of every token; a masked token's input
    // is zero, so are its projections.
    let zero = vec![0.0; config.hidden];
    let mut qkv = Vec::with_capacity(tokens * channels);
    let mut z = Vec::with_capacity(tokens * value_width);
    let mut a = Vec::with_capacity(tokens * config.value_heads);
    let mut b = Vec::with_capacity(tokens * config.value_heads);
    for token in 0..tokens {
        let input = if mask.is_some_and(|mask| mask[token] == 0) {
            &zero[..]
        } else {
            &hidden[token * config.hidden..(token + 1) * config.hidden]
        };
        qkv.extend(linear(input, weights.in_qkv));
        z.extend(linear(input, weights.in_z));
        a.extend(linear(input, weights.in_a));
        b.extend(linear(input, weights.in_b));
    }
    let recurrence = mixer_recurrence_f32(
        config,
        tokens,
        MixerProjections {
            qkv: &qkv,
            z: &z,
            a: &a,
            b: &b,
        },
        MixerGateWeights {
            convolution: weights.convolution,
            a_log: weights.a_log,
            dt_bias: weights.dt_bias,
            norm: weights.norm,
        },
        initial,
    )?;
    let mut output = Vec::with_capacity(tokens * config.hidden);
    for normalized in recurrence.normalized.chunks_exact(value_width) {
        output.extend(linear(normalized, weights.out));
    }
    if output.iter().any(|x| !x.is_finite()) {
        return Err(MixerError::NonFiniteValue);
    }
    Ok(MixerOutput {
        values: output,
        convolution_state: recurrence.convolution_state,
        recurrent_state: recurrence.recurrent_state,
    })
}

/// A chunk's four input projections, `[tokens, width]` each: QKV
/// (`2 * key_width + value_width` channels), Z (`value_width`), A and B
/// (`value_heads`).
pub struct MixerProjections<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
}

/// The mixer's weights besides its projection matrices.
pub struct MixerGateWeights<'a> {
    /// Depthwise convolution, [qkv_channels, conv_kernel].
    pub convolution: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
}

/// The mixer's result before its output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct MixerRecurrence {
    /// The gated RMSNorm of each token's recurrence output, [tokens,
    /// value_width].
    pub normalized: Vec<f32>,
    pub convolution_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

/// Everything between the input projections and the output projection:
/// the causal convolution and its next state, the decay and beta gates, the
/// gated delta recurrence and its next state, and the gated RMSNorm. A
/// Provider computing the projections its own way gets the Reference's
/// values for the rest from here.
pub fn mixer_recurrence_f32(
    config: MixerConfig,
    tokens: usize,
    projections: MixerProjections<'_>,
    weights: MixerGateWeights<'_>,
    initial: Option<MixerState<'_>>,
) -> Result<MixerRecurrence, MixerError> {
    let invalid = || MixerError::InvalidDimensions;
    let key_width = config
        .key_heads
        .checked_mul(config.key_dim)
        .ok_or_else(invalid)?;
    let value_width = config
        .value_heads
        .checked_mul(config.value_dim)
        .ok_or_else(invalid)?;
    let channels = key_width
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_width))
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or_else(invalid)?;
    let conv_size = channels
        .checked_mul(config.conv_kernel)
        .ok_or_else(invalid)?;
    let recurrent_size = config
        .value_heads
        .checked_mul(config.key_dim)
        .and_then(|value| value.checked_mul(config.value_dim))
        .ok_or_else(invalid)?;
    let per_token = |width: usize| tokens.checked_mul(width);
    if config.key_heads == 0
        || config.conv_kernel == 0
        || !config.value_heads.is_multiple_of(config.key_heads)
        || per_token(channels) != Some(projections.qkv.len())
        || per_token(value_width) != Some(projections.z.len())
        || per_token(config.value_heads) != Some(projections.a.len())
        || per_token(config.value_heads) != Some(projections.b.len())
        || weights.convolution.len() != conv_size
        || weights.a_log.len() != config.value_heads
        || weights.dt_bias.len() != config.value_heads
        || weights.norm.len() != config.value_dim
        || initial.as_ref().is_some_and(|state| {
            state.convolution.len() != conv_size || state.recurrent.len() != recurrent_size
        })
    {
        return Err(MixerError::InvalidDimensions);
    }
    if !config.norm_epsilon.is_finite()
        || config.norm_epsilon <= 0.0
        || projections
            .qkv
            .iter()
            .chain(projections.z)
            .chain(projections.a)
            .chain(projections.b)
            .chain(weights.convolution)
            .chain(weights.a_log)
            .chain(weights.dt_bias)
            .chain(weights.norm)
            .chain(initial.as_ref().into_iter().flat_map(|s| s.convolution))
            .chain(initial.as_ref().into_iter().flat_map(|s| s.recurrent))
            .any(|value| !value.is_finite())
    {
        return Err(MixerError::NonFiniteValue);
    }

    let mut conv_state = initial
        .as_ref()
        .map_or_else(|| vec![0.0; conv_size], |state| state.convolution.to_vec());
    let mut query = Vec::with_capacity(tokens * key_width);
    let mut key = Vec::with_capacity(tokens * key_width);
    let mut value = Vec::with_capacity(tokens * value_width);
    let mut decay = Vec::with_capacity(tokens * config.value_heads);
    let mut beta = Vec::with_capacity(tokens * config.value_heads);
    for token in 0..tokens {
        let projected = &projections.qkv[token * channels..(token + 1) * channels];
        let mut convolved = vec![0.0; channels];
        for channel in 0..channels {
            let start = channel * config.conv_kernel;
            let history = &mut conv_state[start..start + config.conv_kernel];
            history.rotate_left(1);
            history[config.conv_kernel - 1] = projected[channel];
            let filter = &weights.convolution[start..start + config.conv_kernel];
            let sum: f32 = history.iter().zip(filter).map(|(x, w)| x * w).sum();
            convolved[channel] = silu(sum);
        }
        query.extend_from_slice(&convolved[..key_width]);
        key.extend_from_slice(&convolved[key_width..key_width * 2]);
        value.extend_from_slice(&convolved[key_width * 2..]);
        let heads = token * config.value_heads..(token + 1) * config.value_heads;
        let (a, b) = (&projections.a[heads.clone()], &projections.b[heads]);
        for head in 0..config.value_heads {
            let time = a[head] + weights.dt_bias[head];
            // Stable softplus avoids overflowing for large positive inputs.
            let softplus = if time > 20.0 {
                time
            } else {
                time.exp().ln_1p()
            };
            decay.push(-weights.a_log[head].exp() * softplus);
            beta.push(sigmoid(b[head]));
        }
    }
    let recurrent = gated_delta_f32(
        DeltaDimensions {
            tokens,
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
        },
        DeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            log_decay: &decay,
            beta: &beta,
            initial_state: initial.as_ref().map(|state| state.recurrent),
        },
    )
    .map_err(|_| MixerError::NonFiniteValue)?;
    let gates = projections.z;
    let mut normalized = Vec::with_capacity(tokens * value_width);
    for token in 0..tokens {
        let start = token * value_width;
        for head in 0..config.value_heads {
            let head_start = start + head * config.value_dim;
            let row = &recurrent.values[head_start..head_start + config.value_dim];
            let square_mean: f32 = row.iter().map(|x| x * x).sum::<f32>() / config.value_dim as f32;
            if !square_mean.is_finite() {
                return Err(MixerError::NonFiniteValue);
            }
            let inverse = (square_mean + config.norm_epsilon).sqrt().recip();
            for column in 0..config.value_dim {
                normalized.push(
                    row[column]
                        * inverse
                        * weights.norm[column]
                        * config.output_gate.apply(gates[head_start + column]),
                );
            }
        }
    }
    if normalized.iter().chain(&conv_state).any(|x| !x.is_finite()) {
        return Err(MixerError::NonFiniteValue);
    }
    Ok(MixerRecurrence {
        normalized,
        convolution_state: conv_state,
        recurrent_state: recurrent.state,
    })
}

fn linear(input: &[f32], weights: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(input.len())
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerError {
    InvalidDimensions,
    NonFiniteValue,
}

impl fmt::Display for MixerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimensions => "invalid gated-delta mixer dimensions",
            Self::NonFiniteValue => "non-finite gated-delta mixer input, weight or result",
        })
    }
}

impl Error for MixerError {}
