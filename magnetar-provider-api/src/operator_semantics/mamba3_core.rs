//! CPU reference for the preprocessed Mamba-3 MIMO recurrence.
//!
//! The inputs are the *already projected and normalized* Q/K, A, dt, trap,
//! x and z tensors. This implements neither the layer projections nor the
//! Runtime's session lifecycle. Its four explicit states are not KV caches.

use std::{error::Error, fmt};

const MAX_ELEMENTS: usize = 67_108_864;

/// One unbatched recurrence. Q/K have `groups` heads which are repeated
/// across `heads`, while the four state tensors always use all heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub heads: usize,
    pub groups: usize,
    pub rank: usize,
    pub head_dim: usize,
    pub state_dim: usize,
    pub angles: usize,
}

/// Parameters use row-major `[rank, heads, state_dim]` for Q/K biases,
/// `[rank, heads, head_dim]` for the three MIMO projections and `[heads]`
/// for the skip coefficients.
#[derive(Clone, Debug)]
pub struct Parameters {
    pub query_bias: Vec<f32>,
    pub key_bias: Vec<f32>,
    pub input_projection: Vec<f32>,
    pub gate_projection: Vec<f32>,
    pub output_projection: Vec<f32>,
    pub skip: Vec<f32>,
}

/// Preprocessed tensors for one token: Q/K `[rank, groups, state_dim]`,
/// input/gate `[heads, head_dim]`, decay/dt/trap `[heads]`, and angle
/// projection `[heads, angles]`. Trap is already sigmoid-transformed and
/// decay is A (not A times dt).
pub struct Token<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub input: &'a [f32],
    pub gate: &'a [f32],
    pub decay: &'a [f32],
    pub dt: &'a [f32],
    pub trap: &'a [f32],
    pub angle_projection: &'a [f32],
}

/// Angle `[heads, angles]`, SSM `[heads, head_dim, state_dim]`, previous
/// rotated key `[rank, heads, state_dim]`, previous input `[heads, head_dim]`.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub angle: Vec<f32>,
    pub ssm: Vec<f32>,
    pub key: Vec<f32>,
    pub input: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOutput {
    pub output: Vec<f32>,
    pub state: State,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mamba3Error(pub &'static str);

impl fmt::Display for Mamba3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for Mamba3Error {}

fn size(dimensions: &[usize]) -> Result<usize, Mamba3Error> {
    dimensions.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .filter(|&n| n <= MAX_ELEMENTS)
            .ok_or(Mamba3Error("Mamba-3 dimensions exceed the reference limit"))
    })
}

impl Geometry {
    pub(super) fn validate(self) -> Result<(), Mamba3Error> {
        if self.heads == 0
            || self.groups == 0
            || self.rank == 0
            || self.head_dim == 0
            || self.state_dim == 0
            || !self.state_dim.is_multiple_of(2)
            || !self.heads.is_multiple_of(self.groups)
            || self.angles > self.state_dim / 2
        {
            return Err(Mamba3Error(
                "invalid Mamba-3 head, group or rotary geometry",
            ));
        }
        size(&[self.rank, self.groups, self.state_dim])?;
        size(&[self.rank, self.heads, self.state_dim])?;
        size(&[self.rank, self.heads, self.head_dim])?;
        size(&[self.heads, self.head_dim, self.state_dim])?;
        size(&[self.heads, self.angles])?;
        Ok(())
    }
}

impl State {
    /// Reset the four independent tensors to their initial zero state.
    pub fn zero(geometry: Geometry) -> Result<Self, Mamba3Error> {
        geometry.validate()?;
        Ok(Self {
            angle: vec![0.0; size(&[geometry.heads, geometry.angles])?],
            ssm: vec![0.0; size(&[geometry.heads, geometry.head_dim, geometry.state_dim])?],
            key: vec![0.0; size(&[geometry.rank, geometry.heads, geometry.state_dim])?],
            input: vec![0.0; size(&[geometry.heads, geometry.head_dim])?],
        })
    }
}

fn check(values: &[f32], expected: usize) -> Result<(), Mamba3Error> {
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        Err(Mamba3Error("Mamba-3 tensor shape or value is invalid"))
    } else {
        Ok(())
    }
}

/// Evaluate one token and return fresh output/state tensors. The supplied
/// previous state is unchanged even when validation or computation fails.
pub fn step(
    geometry: Geometry,
    parameters: &Parameters,
    previous: &State,
    token: &Token<'_>,
) -> Result<StepOutput, Mamba3Error> {
    geometry.validate()?;
    let Geometry {
        heads: h,
        groups: g,
        rank: r,
        head_dim: p,
        state_dim: n,
        angles: a,
    } = geometry;
    let q_group = size(&[r, g, n])?;
    let q_heads = size(&[r, h, n])?;
    let projection = size(&[r, h, p])?;
    let input = size(&[h, p])?;
    let angles = size(&[h, a])?;
    check(token.query, q_group)?;
    check(token.key, q_group)?;
    check(token.input, input)?;
    check(token.gate, input)?;
    check(token.decay, h)?;
    check(token.dt, h)?;
    check(token.trap, h)?;
    check(token.angle_projection, angles)?;
    check(&parameters.query_bias, q_heads)?;
    check(&parameters.key_bias, q_heads)?;
    check(&parameters.input_projection, projection)?;
    check(&parameters.gate_projection, projection)?;
    check(&parameters.output_projection, projection)?;
    check(&parameters.skip, h)?;
    check(&previous.angle, angles)?;
    check(&previous.ssm, size(&[h, p, n])?)?;
    check(&previous.key, q_heads)?;
    check(&previous.input, input)?;
    if token.dt.iter().any(|&value| value <= 0.0)
        || token.decay.iter().any(|&value| value > 0.0)
        || token
            .trap
            .iter()
            .any(|&value| !(0.0..=1.0).contains(&value))
    {
        return Err(Mamba3Error(
            "Mamba-3 dt, decay or trap is outside the supported profile",
        ));
    }

    let mut next = State::zero(geometry)?;
    let mut query = vec![0.0; q_heads];
    let head_index = |rank: usize, head: usize, dim: usize| (rank * h + head) * n + dim;
    for head in 0..h {
        let group = head / (h / g);
        for angle in 0..a {
            let index = head * a + angle;
            next.angle[index] = previous.angle[index]
                + token.angle_projection[index].tanh() * token.dt[head] * std::f32::consts::PI;
            if !next.angle[index].is_finite() {
                return Err(Mamba3Error("non-finite Mamba-3 angle"));
            }
        }
        for rank in 0..r {
            for dim in 0..n {
                let target = head_index(rank, head, dim);
                let source = (rank * g + group) * n + dim;
                query[target] = token.query[source] + parameters.query_bias[target];
                next.key[target] = token.key[source] + parameters.key_bias[target];
            }
            for angle in 0..a {
                let low = head_index(rank, head, angle);
                let high = head_index(rank, head, n / 2 + angle);
                let (sin, cos) = next.angle[head * a + angle].sin_cos();
                let (q0, q1) = (query[low], query[high]);
                let (k0, k1) = (next.key[low], next.key[high]);
                query[low] = q0 * cos - q1 * sin;
                query[high] = q0 * sin + q1 * cos;
                next.key[low] = k0 * cos - k1 * sin;
                next.key[high] = k0 * sin + k1 * cos;
            }
        }
    }
    next.input.copy_from_slice(token.input);
    let mut output = vec![0.0; input];
    for head in 0..h {
        let alpha = (token.decay[head] * token.dt[head]).exp();
        let beta = (1.0 - token.trap[head]) * token.dt[head] * alpha;
        let gamma = token.trap[head] * token.dt[head];
        for dim in 0..p {
            let hp = head * p + dim;
            for state_dim in 0..n {
                let ssm_index = hp * n + state_dim;
                let mut current = 0.0;
                let mut old = 0.0;
                for rank in 0..r {
                    let proj_index = (rank * h + head) * p + dim;
                    let key_index = head_index(rank, head, state_dim);
                    current += token.input[hp]
                        * parameters.input_projection[proj_index]
                        * next.key[key_index];
                    old += previous.input[hp]
                        * parameters.input_projection[proj_index]
                        * previous.key[key_index];
                }
                next.ssm[ssm_index] =
                    previous.ssm[ssm_index] * alpha + current * gamma + old * beta;
            }
            for rank in 0..r {
                let proj_index = (rank * h + head) * p + dim;
                let mut value = 0.0;
                for state_dim in 0..n {
                    value +=
                        next.ssm[hp * n + state_dim] * query[head_index(rank, head, state_dim)];
                }
                value += token.input[hp]
                    * parameters.input_projection[proj_index]
                    * parameters.skip[head];
                let gate = token.gate[hp] * parameters.gate_projection[proj_index];
                let sigmoid = if gate >= 0.0 {
                    1.0 / (1.0 + (-gate).exp())
                } else {
                    let e = gate.exp();
                    e / (1.0 + e)
                };
                output[hp] += value * gate * sigmoid * parameters.output_projection[proj_index];
            }
        }
    }
    if output
        .iter()
        .chain(&query)
        .chain(&next.angle)
        .chain(&next.ssm)
        .chain(&next.key)
        .any(|value| !value.is_finite())
    {
        return Err(Mamba3Error("non-finite Mamba-3 intermediate or result"));
    }
    Ok(StepOutput {
        output,
        state: next,
    })
}
