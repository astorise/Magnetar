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
    [gate, up, down]: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
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
    let (first, count) = (first as usize, count as usize);
    let (hidden, intermediate) = (hidden as usize, intermediate as usize);
    let (gate_cols, down_cols) = (gate_cols as usize, down_cols as usize);
    let mut out = vec![0.0_f32; tokens as usize * hidden];
    let mut activated = vec![0.0_f32; intermediate];
    for token in 0..tokens as usize {
        let x = &h.data[token * hidden..(token + 1) * hidden];
        let row_logits = &logits.data[token * experts as usize..(token + 1) * experts as usize];
        for (expert, weight) in moe_route(row_logits, top_k as usize, normalize) {
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
                *value = g * (1.0 / (1.0 + (-g).exp())) * u;
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
