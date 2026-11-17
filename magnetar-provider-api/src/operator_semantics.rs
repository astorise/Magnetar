//! Operator semantics the Providers reuse.
//!
//! The reference definitions of the row-wise and mixture-of-experts
//! Operators: sigmoid, row scaling, expert routing and the expert
//! combination, with the attribute names that select a Device's share of an
//! expert-split layer. Shared by every Provider so that routing and
//! accumulation order are identical everywhere. Pure functions of host
//! tensors: no Runtime state.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

/// The optimized CPU Provider (`execute-quantized-weights-on-cpu`): like
/// Reference CPU it executes in host memory.
pub const NATIVE_CPU_PROVIDER_NAME: &str = "magnetar:provider/cpu-native";

/// Logistic sigmoid, `1 / (1 + exp(-x))`.
pub fn sigmoid(input: &HostTensor) -> HostTensor {
    let data = input
        .data
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect::<Vec<_>>();
    HostTensor {
        shape: input.shape.clone(),
        data,
    }
}

/// `out[t, :] = x[t, :] * scale[t]`, `scale` of shape `[T, 1]` or `[T]`.
pub fn row_scale(x: &HostTensor, scale: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    let (rows, cols) = rows_cols_of(x, "row-scale's x")?;
    if scale.data.len() as u64 != rows || !(scale.shape == [rows] || scale.shape == [rows, 1]) {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "row-scale expects a [{rows}, 1] scale for x {:?}, got {:?}",
                x.shape, scale.shape
            ),
        ));
    }
    let data = x
        .data
        .iter()
        .enumerate()
        .map(|(index, value)| value * scale.data[index / cols as usize])
        .collect::<Vec<_>>();
    Ok(HostTensor {
        shape: x.shape.clone(),
        data,
    })
}

pub(crate) fn rows_cols_of(
    tensor: &HostTensor,
    what: &str,
) -> Result<(u64, u64), ReferenceCpuError> {
    match tensor.shape[..] {
        [rows, cols] => Ok((rows, cols)),
        _ => Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!("{what} must be 2D, got {:?}", tensor.shape),
        )),
    }
}

/// The experts one token routes to, with their combination weights:
/// `softmax(logits)` in f32, the `top_k` largest probabilities (ties to the
/// lower expert index), divided by their sum when `normalize`. Shared by
/// every Provider's `moe-experts` so routing is identical everywhere.
pub fn moe_route(logits: &[f32], top_k: usize, normalize: bool) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| exps[b].total_cmp(&exps[a]).then(a.cmp(&b)));
    let mut chosen: Vec<(usize, f32)> = order
        .into_iter()
        .take(top_k)
        .map(|expert| (expert, exps[expert] / sum))
        .collect();
    if normalize {
        let total: f32 = chosen.iter().map(|(_, weight)| weight).sum();
        for (_, weight) in &mut chosen {
            *weight /= total;
        }
    }
    chosen
}

/// `moe-experts`' routing attributes: `experts` routed experts, the
/// `top_k` chosen per token, renormalized when `normalize`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeRouting {
    pub experts: u64,
    pub top_k: u64,
    pub normalize: bool,
}

/// The `moe-experts` Operator (`execute-and-cache-moe-experts` F02): for
/// each row of `h [T, H]`, route through [`moe_route`] on `logits [T, E]`
/// and sum `weight * down_e(silu(gate_e(h)) * up_e(h))` over the chosen
/// experts. The stacked weights are `gate`/`up [H, E*I]` (expert `e` owns
/// columns `[e*I, (e+1)*I)`) and `down [I, E*H]` (columns `[e*H,
/// (e+1)*H)`). Each expert's partial sum accumulates in f32 in column
/// order, then the experts add in routing order.
pub fn moe_experts(
    h: &HostTensor,
    logits: &HostTensor,
    weights: [&HostTensor; 3],
    routing: MoeRouting,
) -> Result<HostTensor, ReferenceCpuError> {
    moe_experts_local(h, logits, weights, routing, 0, routing.experts)
}

/// [`moe_experts`] over the `count` experts from `first` only: the stacked
/// weights hold just those experts (gate/up `[H, count*I]`, down `[I,
/// count*H]`), routing still chooses among all `routing.experts` with the
/// full logits, and tokens routed elsewhere contribute nothing -- each
/// Device's share of an expert-split layer, summed across Devices
/// (`enforce-local-multigpu-and-expert-placement` tensor parallelism).
pub fn moe_experts_local(
    h: &HostTensor,
    logits: &HostTensor,
    weights: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    moe_experts_local_with(
        h,
        logits,
        weights,
        routing,
        first,
        count,
        MoeExpertOptions::default(),
    )
}

/// The gate activation of `moe-experts`' experts (`activation`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeExpertActivation {
    /// `silu(gate) * up`, the default.
    #[default]
    Silu,
    /// `gelu_tanh(gate) * up` (`activation = "gelu_tanh"`).
    GeluTanh,
}

impl MoeExpertActivation {
    /// The activation an `activation` attribute names; `None` for an
    /// unknown name.
    pub fn from_attribute(name: &str) -> Option<Self> {
        match name {
            "silu" => Some(Self::Silu),
            "gelu_tanh" => Some(Self::GeluTanh),
            _ => None,
        }
    }

    fn apply(self, gate: f32) -> f32 {
        match self {
            Self::Silu => gate * (1.0 / (1.0 + (-gate).exp())),
            Self::GeluTanh => gelu_tanh(gate),
        }
    }
}

/// `moe-experts`' optional forms (`extend-portable-operator-attributes`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeExpertOptions<'a> {
    pub activation: MoeExpertActivation,
    /// `[experts]` (all routed experts, also on a Device holding a share):
    /// expert `e`'s output is multiplied by `per_expert_scale[e]` before
    /// its routing weight.
    pub per_expert_scale: Option<&'a HostTensor>,
}

/// [`moe_experts_local`] with its optional forms.
pub fn moe_experts_local_with(
    h: &HostTensor,
    logits: &HostTensor,
    [gate, up, down]: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
    options: MoeExpertOptions<'_>,
) -> Result<HostTensor, ReferenceCpuError> {
    let MoeRouting {
        experts,
        top_k,
        normalize,
    } = routing;
    let shape_error =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (tokens, hidden) = rows_cols_of(h, "moe-experts' h")?;
    let (logit_rows, logit_cols) = rows_cols_of(logits, "moe-experts' logits")?;
    let (gate_rows, gate_cols) = rows_cols_of(gate, "moe-experts' gate")?;
    let (down_rows, down_cols) = rows_cols_of(down, "moe-experts' down")?;
    if experts == 0 || top_k == 0 || top_k > experts {
        return Err(shape_error(format!(
            "moe-experts routes top {top_k} of {experts} experts"
        )));
    }
    if count == 0 || first.checked_add(count).is_none_or(|end| end > experts) {
        return Err(shape_error(format!(
            "moe-experts holds experts [{first}, {first} + {count}) of {experts}"
        )));
    }
    if logit_rows != tokens || logit_cols != experts {
        return Err(shape_error(format!(
            "moe-experts expects [{tokens}, {experts}] logits, got {:?}",
            logits.shape
        )));
    }
    if gate_rows != hidden || !gate_cols.is_multiple_of(count) || up.shape != gate.shape {
        return Err(shape_error(format!(
            "moe-experts expects gate and up of [{hidden}, {count} * I], got {:?} and {:?}",
            gate.shape, up.shape
        )));
    }
    let intermediate = gate_cols / count;
    if down_rows != intermediate || down_cols != count * hidden {
        return Err(shape_error(format!(
            "moe-experts expects down of [{intermediate}, {count} * {hidden}], got {:?}",
            down.shape
        )));
    }
    if let Some(scale) = options.per_expert_scale
        && scale.data.len() as u64 != experts
    {
        return Err(shape_error(format!(
            "moe-experts expects a per-expert scale of [{experts}], got {:?}",
            scale.shape
        )));
    }
    let (first, count) = (first as usize, count as usize);
    let (hidden, intermediate) = (hidden as usize, intermediate as usize);
    let (gate_cols, down_cols) = (gate_cols as usize, down_cols as usize);
    let mut out = vec![0.0_f32; tokens as usize * hidden];
    let mut activated = vec![0.0_f32; intermediate];
    for token in 0..tokens as usize {
        let x = &h.data[token * hidden..(token + 1) * hidden];
        let row_logits = &logits.data[token * experts as usize..(token + 1) * experts as usize];
        for (expert, weight) in moe_route(row_logits, top_k as usize, normalize) {
            let weight = weight
                * options
                    .per_expert_scale
                    .map_or(1.0, |scale| scale.data[expert]);
            let Some(expert) = expert.checked_sub(first).filter(|local| *local < count) else {
                continue;
            };
            for (j, value) in activated.iter_mut().enumerate() {
                let column = expert * intermediate + j;
                let (mut g, mut u) = (0.0_f32, 0.0_f32);
                for (k, &xk) in x.iter().enumerate() {
                    g += xk * gate.data[k * gate_cols + column];
                    u += xk * up.data[k * gate_cols + column];
                }
                *value = options.activation.apply(g) * u;
            }
            let row = &mut out[token * hidden..(token + 1) * hidden];
            for (c, slot) in row.iter_mut().enumerate() {
                let column = expert * hidden + c;
                let mut sum = 0.0_f32;
                for (j, &a) in activated.iter().enumerate() {
                    sum += a * down.data[j * down_cols + column];
                }
                *slot += weight * sum;
            }
        }
    }
    Ok(HostTensor {
        shape: vec![tokens, hidden as u64],
        data: out,
    })
}

/// The experts a `moe-experts` invocation holds: `expert_first` and
/// `expert_count` when set (a Device's share of an expert-split layer),
/// else all `experts`.
pub fn moe_local_experts(
    attributes: &std::collections::BTreeMap<String, crate::operator::OperatorAttributeValue>,
    experts: u64,
) -> (u64, u64) {
    let integer = |name: &str| match attributes.get(name) {
        Some(crate::operator::OperatorAttributeValue::Integer(value)) if *value >= 0 => {
            Some(*value as u64)
        }
        _ => None,
    };
    let first = integer(MOE_EXPERT_FIRST_ATTRIBUTE).unwrap_or(0);
    let count = integer(MOE_EXPERT_COUNT_ATTRIBUTE).unwrap_or(experts.saturating_sub(first));
    (first, count)
}

/// `moe-experts`' first held expert (see [`moe_local_experts`]).
pub const MOE_EXPERT_FIRST_ATTRIBUTE: &str = "expert_first";

/// `moe-experts`' held expert count (see [`moe_local_experts`]).
pub const MOE_EXPERT_COUNT_ATTRIBUTE: &str = "expert_count";

/// `gelu` with the tanh approximation: `0.5 x (1 + tanh(sqrt(2/pi) (x +
/// 0.044715 x^3)))`, the portable `gelu`'s definition.
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
}

/// `activation` with `kind = "softcap"`: `cap * tanh(x / cap)`, `cap`
/// finite and positive.
pub fn softcap(input: &HostTensor, cap: f32) -> Result<HostTensor, ReferenceCpuError> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(invalid(format!(
            "softcap cap {cap} must be finite and positive"
        )));
    }
    Ok(HostTensor {
        shape: input.shape.clone(),
        data: input.data.iter().map(|&x| cap * (x / cap).tanh()).collect(),
    })
}

/// `activation` with `kind = "scale"`: `x * factor`, `factor` finite.
pub fn scale(input: &HostTensor, factor: f32) -> Result<HostTensor, ReferenceCpuError> {
    if !factor.is_finite() {
        return Err(invalid(format!("scale factor {factor} must be finite")));
    }
    Ok(HostTensor {
        shape: input.shape.clone(),
        data: input.data.iter().map(|&x| x * factor).collect(),
    })
}

/// `mul`: element-wise, `b` of `a`'s shape or a `[1]` scalar broadcast to
/// every element.
pub fn mul_broadcast(a: &HostTensor, b: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    let data = if a.shape == b.shape {
        a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect()
    } else if b.shape.as_slice() == [1] {
        a.data.iter().map(|x| x * b.data[0]).collect()
    } else {
        return Err(invalid(format!(
            "mul expects b of a's shape {:?} or [1], got {:?}",
            a.shape, b.shape
        )));
    };
    Ok(HostTensor {
        shape: a.shape.clone(),
        data,
    })
}

/// `rmsnorm` with its optional forms: each row splits into `heads` blocks
/// (one block when `heads` is `None`), each normalized by its own mean
/// square, `x * rsqrt(mean(x^2) + epsilon) * weight`. `weight` is `[block]`
/// shared by every block and row, or absent (scale 1). With `heads =
/// Some((h, d))` the row width must be `h * d`.
pub fn rmsnorm_blocks(
    input: &HostTensor,
    weight: Option<&HostTensor>,
    epsilon: f32,
    heads: Option<(u64, u64)>,
) -> Result<HostTensor, ReferenceCpuError> {
    let cols = *input
        .shape
        .last()
        .ok_or_else(|| invalid("rmsnorm expects at least one dimension"))?;
    let block = match heads {
        Some((count, dimension)) if count > 0 && dimension > 0 && count * dimension == cols => {
            dimension
        }
        Some((count, dimension)) => {
            return Err(invalid(format!(
                "rmsnorm head_count {count} x head_dimension {dimension} must equal the row width {cols}"
            )));
        }
        None => cols,
    };
    if block == 0 || !input.data.len().is_multiple_of(block as usize) {
        return Err(invalid(format!("rmsnorm row width {cols} is not usable")));
    }
    if epsilon.is_nan() || epsilon <= 0.0 {
        return Err(invalid("rmsnorm epsilon must be positive"));
    }
    if let Some(weight) = weight
        && weight.data.len() as u64 != block
    {
        return Err(invalid(format!(
            "rmsnorm weight must have {block} values, got {:?}",
            weight.shape
        )));
    }
    let block = block as usize;
    let mut data = Vec::with_capacity(input.data.len());
    for values in input.data.chunks(block) {
        let mean_square = values.iter().map(|value| value * value).sum::<f32>() / block as f32;
        let factor = 1.0 / (mean_square + epsilon).sqrt();
        for (index, value) in values.iter().enumerate() {
            data.push(value * factor * weight.map_or(1.0, |weight| weight.data[index]));
        }
    }
    Ok(HostTensor {
        shape: input.shape.clone(),
        data,
    })
}

/// `batched-matmul` (`cache-latent-attention-history`): `a [T, G*K]` times
/// `b [G*K, N]` group by group (`group_count = G`), giving `[T, G*N]`:
/// output group `g` is `a`'s column group `g` times `b`'s row group `g`.
/// Each product sums in index order, as `matmul` does.
pub fn batched_matmul(
    a: &HostTensor,
    b: &HostTensor,
    group_count: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let invalid =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (rows, a_cols) = rows_cols_of(a, "batched-matmul's input")?;
    let (b_rows, n) = rows_cols_of(b, "batched-matmul's weight")?;
    if group_count == 0 || !a_cols.is_multiple_of(group_count) || a_cols != b_rows {
        return Err(invalid(format!(
            "batched-matmul of {group_count} groups needs [T, G*K] and [G*K, N], got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let k = (a_cols / group_count) as usize;
    let (groups, n, a_cols) = (group_count as usize, n as usize, a_cols as usize);
    let mut data = vec![0.0f32; rows as usize * groups * n];
    for row in 0..rows as usize {
        for group in 0..groups {
            let input = &a.data[row * a_cols + group * k..row * a_cols + (group + 1) * k];
            let out = &mut data[(row * groups + group) * n..(row * groups + group + 1) * n];
            for (column, out) in out.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (i, value) in input.iter().enumerate() {
                    sum += value * b.data[(group * k + i) * n + column];
                }
                *out = sum;
            }
        }
    }
    Ok(HostTensor {
        shape: vec![rows, (groups * n) as u64],
        data,
    })
}

/// The values `attention` reads from `v` under `value_head_stride`
/// (`cache-latent-attention-history`): `v`'s rows hold `kv_head_count`
/// heads of `stride` values, and each head's value is its leading
/// `value_head_dimension`. Returns `[rows, kv_head_count *
/// value_head_dimension]`.
pub fn value_heads(
    v: &HostTensor,
    kv_head_count: u64,
    stride: u64,
    value_head_dimension: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let (rows, cols) = rows_cols_of(v, "attention's V")?;
    if kv_head_count == 0 || value_head_dimension > stride || cols != kv_head_count * stride {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "V rows of {cols} values cannot hold {kv_head_count} heads of {stride} values \
                 with {value_head_dimension}-value values"
            ),
        ));
    }
    let (stride, width) = (stride as usize, value_head_dimension as usize);
    let data = v
        .data
        .chunks_exact(stride)
        .flat_map(|head| head[..width].iter().copied())
        .collect();
    Ok(HostTensor {
        shape: vec![rows, kv_head_count * value_head_dimension],
        data,
    })
}

/// `concat` with `axis = "heads"` (`express-latent-attention-forms`): joins
/// `[T, head_count * a]` and `[T, second_head_count * b]` rows head by head
/// into `[T, head_count * (a + b)]`, output head `h` being the first input's
/// head `h` followed by the second input's head `h` (its only head when
/// `second_head_count` is 1, given to every head).
pub fn concat_heads(
    first: &HostTensor,
    second: &HostTensor,
    head_count: u64,
    second_head_count: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let invalid =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (rows, first_cols) = rows_cols_of(first, "concat's first input")?;
    let (second_rows, second_cols) = rows_cols_of(second, "concat's second input")?;
    if rows != second_rows {
        return Err(invalid(format!(
            "concat across heads needs inputs with the same rows, got {:?} and {:?}",
            first.shape, second.shape
        )));
    }
    let (a, b) = magnetar_core::operator::concat_heads_widths(
        first_cols,
        second_cols,
        head_count,
        second_head_count,
    )
    .map_err(invalid)?;
    let (a, b) = (a as usize, b as usize);
    let (first_cols, second_cols) = (first_cols as usize, second_cols as usize);
    let mut data = Vec::with_capacity(rows as usize * head_count as usize * (a + b));
    for row in 0..rows as usize {
        for head in 0..head_count as usize {
            let start = row * first_cols + head * a;
            data.extend_from_slice(&first.data[start..start + a]);
            let second_head = if second_head_count == 1 { 0 } else { head };
            let start = row * second_cols + second_head * b;
            data.extend_from_slice(&second.data[start..start + b]);
        }
    }
    Ok(HostTensor {
        shape: vec![rows, head_count * (a + b) as u64],
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(rows: u64, cols: u64, seed: u64) -> HostTensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 + seed * 101) % 23) as f32 / 23.0 - 0.5)
            .collect();
        HostTensor::new([rows, cols], data).unwrap()
    }

    /// Each output group is its input group times its weight group.
    #[test]
    fn batched_matmul_multiplies_group_by_group() {
        let (rows, groups, k, n) = (3u64, 4u64, 5u64, 6u64);
        let a = tensor(rows, groups * k, 1);
        let b = tensor(groups * k, n, 2);
        let out = batched_matmul(&a, &b, groups).unwrap();
        assert_eq!(out.shape, vec![rows, groups * n]);
        for group in 0..groups as usize {
            for row in 0..rows as usize {
                for column in 0..n as usize {
                    let expected: f32 = (0..k as usize)
                        .map(|i| {
                            a.data[row * (groups * k) as usize + group * k as usize + i]
                                * b.data[(group * k as usize + i) * n as usize + column]
                        })
                        .sum();
                    let actual =
                        out.data[row * (groups * n) as usize + group * n as usize + column];
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
        assert!(batched_matmul(&a, &tensor(groups * k + 1, n, 2), groups).is_err());
        assert!(batched_matmul(&a, &b, 3).is_err());
    }

    /// The values under a stride are each head's leading values.
    #[test]
    fn value_heads_take_each_heads_leading_values() {
        let v = tensor(2, 2 * 5, 3);
        let values = value_heads(&v, 2, 5, 3).unwrap();
        assert_eq!(values.shape, vec![2, 6]);
        assert_eq!(&values.data[..3], &v.data[..3]);
        assert_eq!(&values.data[3..6], &v.data[5..8]);
        assert_eq!(&values.data[6..9], &v.data[10..13]);
        assert!(value_heads(&v, 2, 4, 3).is_err());
        assert!(value_heads(&v, 2, 5, 6).is_err());
    }
}
