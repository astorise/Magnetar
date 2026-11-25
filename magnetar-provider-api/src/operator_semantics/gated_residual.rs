//! Qwen4-Exp's multi-stream gated residual (`support-qwen38-flash-next`).
//!
//! The residual is `hc_count` streams side by side, `[tokens, hc_count *
//! hidden]`. A block reads one mix of them and writes back into each stream
//! with its own weight. The formulas are those of Transformers'
//! `Qwen4ExpTextGatedResidual` and decoder layer, pinned in design D7.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

const MAX_ELEMENTS: u64 = 67_108_864;

/// A `gated-residual` node's attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedResidual {
    pub hc_count: usize,
    pub epsilon: f32,
    /// Added to every norm weight value (1 for the checkpoint's zero-centered
    /// norms).
    pub weight_offset: f32,
}

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
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

fn check(tensor: &HostTensor, shape: &[u64], what: &str) -> Result<(), ReferenceCpuError> {
    let count = shape.iter().try_fold(1u64, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&value| dim > 0 && value <= MAX_ELEMENTS)
    });
    if tensor.shape != shape
        || count.and_then(|value| usize::try_from(value).ok()) != Some(tensor.data.len())
        || tensor.data.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(format!(
            "gated-residual {what} must be {shape:?} and finite, got {:?}",
            tensor.shape
        )));
    }
    Ok(())
}

/// `inputs`: the residual `[T, hc * H]`, the norm weight `[hc * H]`, the
/// down projection `[rank, hc * H]` and the up projection `[hc * H, rank]`
/// (checkpoint `[out, in]` order), and optionally the injection projection
/// `[hc, hc * H]`.
///
/// Returns the mixed block input `[T, H]` and, with the injection projection,
/// the per-stream write weights `[T, hc]`:
/// - `n` is the residual normalized per stream (groups of `H`), scaled by
///   `offset + weight`;
/// - `m = sigmoid(up(silu(down(n) / hc)))`, one weight per stream and
///   feature, and `mixed` the mean over streams of `m * n`;
/// - the write weights are `2 * sigmoid(inject(n) / hc)`.
pub fn gated_residual(
    inputs: &[HostTensor],
    config: GatedResidual,
) -> Result<Vec<HostTensor>, ReferenceCpuError> {
    if !(4..=5).contains(&inputs.len()) {
        return Err(invalid("gated-residual requires four or five inputs"));
    }
    let hc = config.hc_count as u64;
    if hc < 2 || !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(invalid(
            "gated-residual needs hc_count >= 2 and a positive epsilon",
        ));
    }
    let [tokens, width] = inputs[0].shape[..] else {
        return Err(invalid("gated-residual input must be rank two"));
    };
    if width == 0 || !width.is_multiple_of(hc) {
        return Err(invalid(
            "gated-residual width must be a multiple of hc_count",
        ));
    }
    let hidden = width / hc;
    let [rank, _] = inputs[2].shape[..] else {
        return Err(invalid("gated-residual down projection must be rank two"));
    };
    check(&inputs[0], &[tokens, width], "input")?;
    check(&inputs[1], &[width], "norm weight")?;
    check(&inputs[2], &[rank, width], "down projection")?;
    check(&inputs[3], &[width, rank], "up projection")?;
    if let Some(inject) = inputs.get(4) {
        check(inject, &[hc, width], "injection projection")?;
    }
    let (tokens, width, hidden, hc, rank) = (
        tokens as usize,
        width as usize,
        hidden as usize,
        config.hc_count,
        rank as usize,
    );
    let mut mixed = Vec::with_capacity(tokens * hidden);
    let mut write = Vec::new();
    for row in inputs[0].data.chunks_exact(width) {
        // Per-stream RMS norm with the zero-centered weight.
        let mut normed = vec![0.0f32; width];
        for stream in 0..hc {
            let range = stream * hidden..(stream + 1) * hidden;
            let mean: f32 = row[range.clone()].iter().map(|x| x * x).sum::<f32>() / hidden as f32;
            let scale = (mean + config.epsilon).sqrt().recip();
            for index in range {
                normed[index] = row[index] * scale * (config.weight_offset + inputs[1].data[index]);
            }
        }
        let project = |weight: &HostTensor, input: &[f32], out: usize| -> Vec<f32> {
            (0..out)
                .map(|o| {
                    let w = &weight.data[o * input.len()..(o + 1) * input.len()];
                    w.iter().zip(input).map(|(w, x)| w * x).sum()
                })
                .collect()
        };
        let low: Vec<f32> = project(&inputs[2], &normed, rank)
            .into_iter()
            .map(|value| silu(value / hc as f32))
            .collect();
        let weights: Vec<f32> = project(&inputs[3], &low, width)
            .into_iter()
            .map(sigmoid)
            .collect();
        for feature in 0..hidden {
            let sum: f32 = (0..hc)
                .map(|stream| {
                    let index = stream * hidden + feature;
                    weights[index] * normed[index]
                })
                .sum();
            mixed.push(sum / hc as f32);
        }
        if let Some(inject) = inputs.get(4) {
            write.extend(
                project(inject, &normed, hc)
                    .into_iter()
                    .map(|value| 2.0 * sigmoid(value / hc as f32)),
            );
        }
    }
    let mut outputs = vec![HostTensor::new([tokens as u64, hidden as u64], mixed)?];
    if inputs.len() == 5 {
        outputs.push(HostTensor::new([tokens as u64, hc as u64], write)?);
    }
    Ok(outputs)
}

/// `residual + block_output * write_weights` for every stream: the residual
/// `[T, hc * H]`, the block output `[T, H]` and the write weights `[T, hc]`
/// give `[T, hc * H]`.
pub fn gated_residual_inject(inputs: &[HostTensor]) -> Result<HostTensor, ReferenceCpuError> {
    let [residual, block, write] = inputs else {
        return Err(invalid("gated-residual-inject requires three inputs"));
    };
    let [tokens, hidden] = block.shape[..] else {
        return Err(invalid(
            "gated-residual-inject block output must be rank two",
        ));
    };
    let [_, hc] = write.shape[..] else {
        return Err(invalid(
            "gated-residual-inject write weights must be rank two",
        ));
    };
    check(block, &[tokens, hidden], "block output")?;
    check(write, &[tokens, hc], "write weights")?;
    check(residual, &[tokens, hc * hidden], "residual")?;
    let (hidden, hc) = (hidden as usize, hc as usize);
    let mut out = Vec::with_capacity(residual.data.len());
    for ((row, block), write) in residual
        .data
        .chunks_exact(hc * hidden)
        .zip(block.data.chunks_exact(hidden))
        .zip(write.data.chunks_exact(hc))
    {
        for (stream, &weight) in write.iter().enumerate() {
            for (feature, &value) in block.iter().enumerate() {
                out.push(row[stream * hidden + feature] + value * weight);
            }
        }
    }
    HostTensor::new(residual.shape.clone(), out)
}

/// `[T, H]` repeated `count` times along the features: `[T, count * H]`
/// (the embedding becoming the residual's streams).
pub fn repeat_features(input: &HostTensor, count: usize) -> Result<HostTensor, ReferenceCpuError> {
    let [tokens, width] = input.shape[..] else {
        return Err(invalid("repeat-features input must be rank two"));
    };
    check(input, &[tokens, width], "input")?;
    if count == 0 || (count as u64).saturating_mul(input.data.len() as u64) > MAX_ELEMENTS {
        return Err(invalid(
            "repeat-features count must be positive and bounded",
        ));
    }
    let mut out = Vec::with_capacity(input.data.len() * count);
    for row in input.data.chunks_exact(width as usize) {
        for _ in 0..count {
            out.extend_from_slice(row);
        }
    }
    HostTensor::new([tokens, width * count as u64], out)
}
