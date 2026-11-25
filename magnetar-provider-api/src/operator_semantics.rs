//! Operator semantics the Providers reuse.
//!
//! The reference definitions of the row-wise and mixture-of-experts
//! Operators: sigmoid, row scaling, expert routing and the expert
//! combination, with the attribute names that select a Device's share of an
//! expert-split layer. Shared by every Provider so that routing and
//! accumulation order are identical everywhere. Pure functions of host
//! tensors: no Runtime state.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

mod differential_attention;
mod encoder_attention;
mod encoder_ops;
mod gated_delta_core;
mod gated_delta_mixer;
mod gated_delta_mixer_core;
mod mamba2_mixer;
mod mamba3_core;
mod mamba3_mimo;
mod mamba3_mixer;
mod projected_differential_attention;
mod sigmoid_routing;
mod token_shift;
pub use differential_attention::{
    DifferentialAttention, differential_attention, differential_attention_from_attributes,
};
pub use encoder_attention::{
    EncoderAttention, encoder_attention, encoder_attention_from_attributes,
};
pub use encoder_ops::{
    affine_layernorm, affine_layernorm_epsilon, gelu_exact, slice_last, slice_last_attributes,
};
pub use gated_delta_mixer::{
    GatedDeltaMixer, gated_delta_mixer, gated_delta_mixer_from_attributes,
    gated_delta_mixer_from_projections,
};
pub use mamba2_mixer::{Mamba2Mixer, mamba2_mixer, mamba2_mixer_from_attributes};
pub use mamba3_core::Geometry as Mamba3Mimo;
pub use mamba3_mimo::{mamba3_mimo, mamba3_mimo_from_attributes};
pub use mamba3_mixer::{Mamba3Mixer, mamba3_mixer, mamba3_mixer_from_attributes};
pub use projected_differential_attention::{
    ProjectedDifferentialAttention, projected_differential_attention,
    projected_differential_attention_from_attributes,
};
pub use sigmoid_routing::{
    ExpertGroupSelection, SigmoidNormalization, SigmoidRouting, sigmoid_moe_route,
    sigmoid_routing_from_attributes,
};
pub use token_shift::token_shift;

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
    /// Ungated `relu(gate)²` (`activation = "relu_squared"`).
    ReluSquared,
}

impl MoeExpertActivation {
    /// The activation an `activation` attribute names; `None` for an
    /// unknown name.
    pub fn from_attribute(name: &str) -> Option<Self> {
        match name {
            "silu" => Some(Self::Silu),
            "gelu_tanh" => Some(Self::GeluTanh),
            "relu_squared" => Some(Self::ReluSquared),
            _ => None,
        }
    }

    fn combine(self, gate: f32, up: f32) -> f32 {
        match self {
            Self::Silu => gate * (1.0 / (1.0 + (-gate).exp())) * up,
            Self::GeluTanh => gelu_tanh(gate) * up,
            Self::ReluSquared => gate.max(0.0).powi(2),
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
    weights: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
    options: MoeExpertOptions<'_>,
) -> Result<HostTensor, ReferenceCpuError> {
    moe_experts_local_impl(h, logits, weights, routing, first, count, options, None)
}

/// The additional operands and options for selection-biased sigmoid routing.
#[derive(Clone, Copy, Debug)]
pub struct MoeSigmoidExpertOptions<'a> {
    pub experts: MoeExpertOptions<'a>,
    pub routing: SigmoidRouting,
    pub selection_bias: &'a HostTensor,
}

/// [`moe_experts_local_with`] with selection-biased sigmoid routing.
pub fn moe_experts_local_sigmoid_with(
    h: &HostTensor,
    logits: &HostTensor,
    weights: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
    options: MoeSigmoidExpertOptions<'_>,
) -> Result<HostTensor, ReferenceCpuError> {
    moe_experts_local_impl(
        h,
        logits,
        weights,
        routing,
        first,
        count,
        options.experts,
        Some((options.routing, options.selection_bias)),
    )
}

#[allow(clippy::too_many_arguments)]
fn moe_experts_local_impl(
    h: &HostTensor,
    logits: &HostTensor,
    [gate, up, down]: [&HostTensor; 3],
    routing: MoeRouting,
    first: u64,
    count: u64,
    options: MoeExpertOptions<'_>,
    sigmoid_routing: Option<(SigmoidRouting, &HostTensor)>,
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
    if let Some((sigmoid, bias)) = sigmoid_routing
        && (sigmoid.top_k as u64 != top_k
            || bias.shape != [experts]
            || bias.data.len() as u64 != experts)
    {
        return Err(shape_error(format!(
            "moe-experts expects a selection bias of [{experts}] and matching top-k, got {:?}",
            bias.shape
        )));
    }
    if options.activation == MoeExpertActivation::ReluSquared
        && h.data
            .iter()
            .chain(gate.data.iter())
            .chain(down.data.iter())
            .any(|value| !value.is_finite())
    {
        return Err(shape_error("non-finite relu_squared expert input".into()));
    }
    let (first, count) = (first as usize, count as usize);
    let (hidden, intermediate) = (hidden as usize, intermediate as usize);
    let (gate_cols, down_cols) = (gate_cols as usize, down_cols as usize);
    let mut out = vec![0.0_f32; tokens as usize * hidden];
    let mut activated = vec![0.0_f32; intermediate];
    for token in 0..tokens as usize {
        let x = &h.data[token * hidden..(token + 1) * hidden];
        let row_logits = &logits.data[token * experts as usize..(token + 1) * experts as usize];
        let selected = match sigmoid_routing {
            Some((sigmoid, bias)) => sigmoid_moe_route(row_logits, &bias.data, sigmoid)?,
            None => moe_route(row_logits, top_k as usize, normalize),
        };
        for (expert, weight) in selected {
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
                    if options.activation != MoeExpertActivation::ReluSquared {
                        u += xk * up.data[k * gate_cols + column];
                    }
                }
                if options.activation == MoeExpertActivation::ReluSquared && !g.is_finite() {
                    return Err(shape_error(
                        "non-finite relu_squared expert projection".into(),
                    ));
                }
                *value = options.activation.combine(g, u);
                if options.activation == MoeExpertActivation::ReluSquared && !value.is_finite() {
                    return Err(shape_error(
                        "non-finite relu_squared expert activation".into(),
                    ));
                }
            }
            let row = &mut out[token * hidden..(token + 1) * hidden];
            for (c, slot) in row.iter_mut().enumerate() {
                let column = expert * hidden + c;
                let mut sum = 0.0_f32;
                for (j, &a) in activated.iter().enumerate() {
                    sum += a * down.data[j * down_cols + column];
                }
                *slot += weight * sum;
                if options.activation == MoeExpertActivation::ReluSquared && !slot.is_finite() {
                    return Err(shape_error("non-finite relu_squared expert output".into()));
                }
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

/// Elementwise squared ReLU; non-finite inputs or outputs are rejected.
pub fn relu_squared(input: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    let mut data = Vec::with_capacity(input.data.len());
    for &x in &input.data {
        let value = x.max(0.0).powi(2);
        if !x.is_finite() || !value.is_finite() {
            return Err(invalid("non-finite relu_squared input or output"));
        }
        data.push(value);
    }
    Ok(HostTensor {
        shape: input.shape.clone(),
        data,
    })
}

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
}

/// `geodesic-residual@1`: apply a learned spherical residual update to each
/// row along the final dimension. Inputs and output are f32, with f32
/// accumulation; scale and bias are scalar tensors. `layer_index` is the
/// original zero-based model layer, including when a graph is segmented.
///
/// For residual `x` and branch `g`, form `t = g - x * dot(x,g) /
/// max(dot(x,x), 1e-12)`, `r = max(norm(x), 1e-6)`, and
/// `n = max(norm(t), 1e-8)`. Set `a = min(n/r, pi/4)` then
/// `theta = min((a*scale+bias)/(layer_index+1), pi/4)`. The result is
/// `x*cos(theta) + (t/n)*r*sin(theta)`. The angle has no lower clamp.
pub fn geodesic_residual(
    residual: &HostTensor,
    branch: &HostTensor,
    scale: &HostTensor,
    bias: &HostTensor,
    layer_index: u32,
) -> Result<HostTensor, ReferenceCpuError> {
    let shape = &residual.shape;
    let width = shape.last().and_then(|width| usize::try_from(*width).ok());
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|d| count.checked_mul(d))
    });
    let scalar = |tensor: &HostTensor| {
        (tensor.shape.is_empty() || tensor.shape == [1]) && tensor.data.len() == 1
    };
    if shape.is_empty()
        || shape.contains(&0)
        || residual.shape != branch.shape
        || elements != Some(residual.data.len())
        || residual.data.len() != branch.data.len()
        || !scalar(scale)
        || !scalar(bias)
        || layer_index == u32::MAX
    {
        return Err(invalid(
            "geodesic-residual has invalid tensor shapes or layer index",
        ));
    }
    let width = width.ok_or_else(|| invalid("geodesic-residual row width is too large"))?;
    let mut result = Vec::with_capacity(residual.data.len());
    if residual
        .data
        .iter()
        .chain(&branch.data)
        .chain(&scale.data)
        .chain(&bias.data)
        .any(|x| !x.is_finite())
    {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ExecutionFailed,
            "non-finite geodesic-residual input",
        ));
    }
    for (x, g) in residual
        .data
        .chunks_exact(width)
        .zip(branch.data.chunks_exact(width))
    {
        let norm_squared: f32 = x.iter().map(|v| v * v).sum();
        let dot: f32 = x.iter().zip(g).map(|(a, b)| a * b).sum();
        let coefficient = dot / norm_squared.max(1e-12);
        let tangent: Vec<f32> = x.iter().zip(g).map(|(a, b)| b - coefficient * a).collect();
        let tangent_norm = tangent.iter().map(|v| v * v).sum::<f32>().sqrt();
        let safe_norm = tangent_norm.max(1e-8);
        let radius = norm_squared.sqrt().max(1e-6);
        let angle = (safe_norm / radius).min(std::f32::consts::FRAC_PI_4);
        let theta = ((angle * scale.data[0] + bias.data[0]) / (layer_index + 1) as f32)
            .min(std::f32::consts::FRAC_PI_4);
        if !norm_squared.is_finite()
            || !dot.is_finite()
            || !coefficient.is_finite()
            || !tangent_norm.is_finite()
            || !theta.is_finite()
        {
            return Err(ReferenceCpuError::new(
                ReferenceCpuErrorCode::ExecutionFailed,
                "non-finite geodesic-residual intermediate",
            ));
        }
        let (sin, cos) = theta.sin_cos();
        for (value, tangent) in x.iter().zip(tangent) {
            let output = value * cos + (tangent / safe_norm) * radius * sin;
            if !output.is_finite() {
                return Err(ReferenceCpuError::new(
                    ReferenceCpuErrorCode::ExecutionFailed,
                    "non-finite geodesic-residual output",
                ));
            }
            result.push(output);
        }
    }
    Ok(HostTensor {
        shape: shape.clone(),
        data: result,
    })
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
/// `b` group by group (`group_count = G`), giving `[T, G*N]`: output group
/// `g` is `a`'s column group `g` times `b`'s row group `g`, which is
/// `[K, N]` (`b [G*K, N]`) or, with `transpose_b`, `[N, K]` used transposed
/// (`b [G*N, K]`). Each product sums in index order, as `matmul` does.
pub fn batched_matmul(
    a: &HostTensor,
    b: &HostTensor,
    group_count: u64,
    transpose_b: bool,
) -> Result<HostTensor, ReferenceCpuError> {
    let invalid =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (rows, a_cols) = rows_cols_of(a, "batched-matmul's input")?;
    let (b_rows, b_cols) = rows_cols_of(b, "batched-matmul's weight")?;
    let shapes_fit = group_count > 0
        && a_cols.is_multiple_of(group_count)
        && b_rows.is_multiple_of(group_count)
        && if transpose_b {
            b_cols == a_cols / group_count
        } else {
            b_rows == a_cols
        };
    if !shapes_fit {
        return Err(invalid(format!(
            "batched-matmul of {group_count} groups (transpose_b {transpose_b}) cannot take \
             {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let groups = group_count as usize;
    let k = (a_cols / group_count) as usize;
    let n = if transpose_b {
        (b_rows / group_count) as usize
    } else {
        b_cols as usize
    };
    let (a_cols, b_cols) = (a_cols as usize, b_cols as usize);
    let weight = |group: usize, i: usize, column: usize| {
        if transpose_b {
            b.data[(group * n + column) * b_cols + i]
        } else {
            b.data[(group * k + i) * b_cols + column]
        }
    };
    let mut data = vec![0.0f32; rows as usize * groups * n];
    for row in 0..rows as usize {
        for group in 0..groups {
            let input = &a.data[row * a_cols + group * k..row * a_cols + (group + 1) * k];
            let out = &mut data[(row * groups + group) * n..(row * groups + group + 1) * n];
            for (column, out) in out.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (i, value) in input.iter().enumerate() {
                    sum += value * weight(group, i, column);
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

/// `latent [rows, latent_width + rope]` times `weight [latent_width, n]`,
/// each row's leading `latent_width` values only: `[rows, n]`, each product
/// summing in index order, as `matmul` does.
fn latent_product(
    latent: &HostTensor,
    weight: &HostTensor,
    latent_width: u64,
    what: &str,
) -> Result<HostTensor, ReferenceCpuError> {
    let invalid =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (rows, width) = rows_cols_of(latent, "attention's latent K or V")?;
    let (weight_rows, n) = rows_cols_of(weight, what)?;
    if latent_width == 0 || latent_width > width || weight_rows != latent_width {
        return Err(invalid(format!(
            "{what} {:?} cannot expand latent rows {:?} of {latent_width} latent values",
            weight.shape, latent.shape
        )));
    }
    let (rows, width, latent_width, n) = (
        rows as usize,
        width as usize,
        latent_width as usize,
        n as usize,
    );
    let mut data = vec![0.0_f32; rows * n];
    for row in 0..rows {
        let out = &mut data[row * n..(row + 1) * n];
        for (inner, &a) in latent.data[row * width..row * width + latent_width]
            .iter()
            .enumerate()
        {
            for (out, &b) in out.iter_mut().zip(&weight.data[inner * n..(inner + 1) * n]) {
                *out += a * b;
            }
        }
    }
    Ok(HostTensor {
        shape: vec![rows as u64, n as u64],
        data,
    })
}

/// `attention`'s keys under `latent_kv_width` (`chunk-latent-prefills`):
/// `latent [rows, latent_width + rope]` holds each key row's latent `c` and
/// one rope key shared by every head, and `w_k [latent_width, head_count *
/// nope]` gives head `h` the key `[c W_k_h | k_rope]`:
/// `[rows, head_count * (nope + rope)]`.
pub fn latent_keys(
    latent: &HostTensor,
    w_k: &HostTensor,
    latent_width: u64,
    head_count: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let invalid =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    let (rows, width) = rows_cols_of(latent, "attention's latent K")?;
    if latent_width >= width {
        return Err(invalid(format!(
            "latent K rows {:?} hold no rope key after {latent_width} latent values",
            latent.shape
        )));
    }
    let nope = latent_product(latent, w_k, latent_width, "attention's key projection")?;
    let rope = HostTensor {
        shape: vec![rows, width - latent_width],
        data: latent
            .data
            .chunks_exact(width as usize)
            .flat_map(|row| row[latent_width as usize..].iter().copied())
            .collect(),
    };
    concat_heads(&nope, &rope, head_count, 1)
}

/// `attention`'s values under `latent_kv_width` (`chunk-latent-prefills`):
/// `w_v [latent_width, head_count * value]` gives head `h` the value
/// `c W_v_h` of each latent row's leading `latent_width` values `c`.
pub fn latent_values(
    latent: &HostTensor,
    w_v: &HostTensor,
    latent_width: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    latent_product(latent, w_v, latent_width, "attention's value projection")
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
mod tests;
