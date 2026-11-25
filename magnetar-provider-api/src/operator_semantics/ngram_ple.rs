//! Qwen4-Exp's per-layer embedding from hashed n-grams
//! (`support-qwen38-flash-next`): the hashed lookup and the gate and dilated
//! convolution that add it into the residual streams. The formulas are those
//! of Transformers' `Qwen4ExpTextNGramEmbedding` and `Qwen4ExpTextPLELayer`,
//! pinned in design D7.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

const MAX_ELEMENTS: u64 = 67_108_864;
const MASK64: u64 = u64::MAX;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

/// An `ngram-embedding` node's attributes: the hash of one PLE layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramEmbedding {
    /// Largest n-gram order; `ngram_size - 1` earlier tokens are state.
    pub ngram_size: usize,
    /// Independently hashed rows per n-gram order (orders 2 through
    /// `ngram_size`).
    pub heads_per_ngram: usize,
    /// The model's token vocabulary, which bounds the hash multipliers.
    pub vocab_size: u64,
    /// Lower bound of the first head's prime row count.
    pub vocab_base: u64,
    /// Index of this layer among the PLE layers (not the decoder layer).
    pub layer_index: u64,
    pub seed: u64,
    /// The token that ends an n-gram segment (and fills missing context).
    pub eos_token_id: i64,
}

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
}

fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The `count`-th prime above `start`.
fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

impl NgramEmbedding {
    /// Heads in all: `(ngram_size - 1) * heads_per_ngram`.
    pub fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The previous tokens a position needs: `ngram_size - 1`.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// One multiplier per position of an n-gram, from the seed.
    pub fn multipliers(&self) -> Vec<i64> {
        let max_long = i64::MAX as u64;
        let multiplier_max = max_long / self.vocab_size.max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = self
            .seed
            .wrapping_add(PRIME_1.wrapping_mul(self.layer_index));
        (0..self.ngram_size as u64)
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1)) & MASK64;
                (2 * (splitmix64(value) % half_bound) + 1) as i64
            })
            .collect()
    }

    /// Each head's row count, a distinct prime above `vocab_base`; the
    /// heads' rows follow one another in the table.
    pub fn head_vocab_sizes(&self) -> Vec<u64> {
        let heads = self.heads() as u64;
        (0..heads)
            .map(|head| nth_prime_after(self.vocab_base - 1, self.layer_index * heads + head + 1))
            .collect()
    }

    /// The table row each head reads for every token of `tokens`, given the
    /// `context_len` tokens before it (`eos_token_id` where there are none):
    /// `[tokens.len()][heads]`.
    pub fn row_ids(&self, tokens: &[i64], context: &[i64]) -> Vec<Vec<u64>> {
        let eos = self.eos_token_id;
        let history: Vec<i64> = context.iter().chain(tokens).copied().collect();
        // The last EOS strictly before each position starts its segment.
        let mut segment_start = vec![0usize; history.len()];
        let mut last_eos: Option<usize> = None;
        for (position, &token) in history.iter().enumerate() {
            segment_start[position] = last_eos.map_or(0, |eos_position| eos_position + 1);
            if token == eos {
                last_eos = Some(position);
            }
        }
        let shifted = |position: usize, shift: usize| -> i64 {
            if shift == 0 {
                return history[position];
            }
            let in_segment = position - segment_start[position] >= shift;
            match position.checked_sub(shift) {
                Some(source) if in_segment => history[source],
                _ => eos,
            }
        };
        let multipliers = self.multipliers();
        let sizes = self.head_vocab_sizes();
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut total = 0u64;
        for &size in &sizes {
            offsets.push(total);
            total += size;
        }
        let first = context.len();
        (first..history.len())
            .map(|position| {
                let mut ids = Vec::with_capacity(self.heads());
                for ngram in 2..=self.ngram_size {
                    let mut mixed = shifted(position, 0).wrapping_mul(multipliers[0]);
                    for (shift, multiplier) in multipliers.iter().enumerate().take(ngram).skip(1) {
                        mixed ^= shifted(position, shift).wrapping_mul(*multiplier);
                    }
                    let start = (ngram - 2) * self.heads_per_ngram;
                    for head in start..start + self.heads_per_ngram {
                        let size = sizes[head] as i64;
                        ids.push(mixed.rem_euclid(size) as u64 + offsets[head]);
                    }
                }
                ids
            })
            .collect()
    }
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
            "{what} must be {shape:?} and finite, got {:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn token_ids(tensor: &HostTensor, what: &str) -> Result<Vec<i64>, ReferenceCpuError> {
    tensor
        .data
        .iter()
        .map(|&value| {
            if value.fract() == 0.0 && (0.0..16_777_216.0).contains(&value) {
                Ok(value as i64)
            } else {
                Err(invalid(format!("{what} must hold token ids, got {value}")))
            }
        })
        .collect()
}

/// `inputs`: the token ids `[T]`, the `ngram_size - 1` tokens before them
/// `[context_len]` (the layer's state; `eos_token_id` at a sequence's start)
/// and the table `[rows, head_dim]`. Returns the embeddings `[T, heads *
/// head_dim]`, each head's row concatenated, and the next context.
pub fn ngram_embedding(
    inputs: &[HostTensor],
    config: NgramEmbedding,
) -> Result<[HostTensor; 2], ReferenceCpuError> {
    let [ids, context, table] = inputs else {
        return Err(invalid("ngram-embedding requires three inputs"));
    };
    if config.ngram_size < 2
        || config.heads_per_ngram == 0
        || config.vocab_size == 0
        || config.vocab_base < 2
    {
        return Err(invalid("invalid ngram-embedding geometry"));
    }
    let [tokens] = ids.shape[..] else {
        return Err(invalid("ngram-embedding ids must be rank one"));
    };
    let [rows, head_dim] = table.shape[..] else {
        return Err(invalid("ngram-embedding table must be rank two"));
    };
    let context_len = config.context_len();
    check(ids, &[tokens], "ngram-embedding ids")?;
    check(context, &[context_len as u64], "ngram-embedding context")?;
    check(table, &[rows, head_dim], "ngram-embedding table")?;
    let tokens_ids = token_ids(ids, "ngram-embedding ids")?;
    let context_ids = token_ids(context, "ngram-embedding context")?;
    let heads = config.heads();
    let total: u64 = config.head_vocab_sizes().iter().sum();
    if total > rows {
        return Err(invalid(format!(
            "ngram-embedding table has {rows} rows, its heads need {total}"
        )));
    }
    let head_dim = head_dim as usize;
    let mut out = Vec::with_capacity(tokens_ids.len() * heads * head_dim);
    for row_ids in config.row_ids(&tokens_ids, &context_ids) {
        for row in row_ids {
            let start = row as usize * head_dim;
            out.extend_from_slice(&table.data[start..start + head_dim]);
        }
    }
    let history: Vec<f32> = context_ids
        .iter()
        .chain(&tokens_ids)
        .map(|&token| token as f32)
        .collect();
    let next = history[history.len() - context_len..].to_vec();
    Ok([
        HostTensor::new([tokens, (heads * head_dim) as u64], out)?,
        HostTensor::new([context_len as u64], next)?,
    ])
}

/// An `ngram-ple` node's attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramPle {
    pub hc_count: usize,
    pub epsilon: f32,
    /// Added to every norm weight value (1 for the checkpoint's zero-centered
    /// norms).
    pub weight_offset: f32,
    pub conv_kernel: usize,
    pub conv_dilation: usize,
}

impl NgramPle {
    /// Columns of convolution history kept per channel.
    pub fn state_len(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dilation
    }
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

/// RMS norm per stream (groups of `hidden`) scaled by `offset + weight`.
fn grouped_norm(row: &[f32], weight: &[f32], hidden: usize, config: &NgramPle) -> Vec<f32> {
    let mut out = vec![0.0f32; row.len()];
    for (group, chunk) in row.chunks_exact(hidden).enumerate() {
        let mean = chunk.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
        let scale = (mean + config.epsilon).sqrt().recip();
        for (index, &value) in chunk.iter().enumerate() {
            let at = group * hidden + index;
            out[at] = value * scale * (config.weight_offset + weight[at]);
        }
    }
    out
}

fn project(weight: &HostTensor, input: &[f32]) -> Vec<f32> {
    weight
        .data
        .chunks_exact(input.len())
        .map(|w| w.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

/// What the PLE layer adds to the residual streams. `inputs`: the residual
/// `[T, hc * H]`, the embeddings `[T, E]`, the key projection `[hc * H, E]`
/// and value projection `[H, E]` (checkpoint `[out, in]` order), the three
/// norms' weights `[hc * H]` (key, query, convolution), the depthwise
/// convolution `[hc * H, kernel]` and its history `[hc * H, state_len]`.
///
/// Per token and stream, `gate = sum(norm(key) * norm(residual)) /
/// sqrt(H)`, then `sqrt(max(|gate|, 1e-6)) * sign(gate)`, and `gated =
/// sigmoid(gate) * value`. The result is `gated + silu(conv(norm(gated)))`,
/// the convolution reading every `conv_dilation`-th earlier column.
/// Returns the result `[T, hc * H]` and the next history.
pub fn ngram_ple(
    inputs: &[HostTensor],
    config: NgramPle,
) -> Result<[HostTensor; 2], ReferenceCpuError> {
    let [
        hidden,
        embeddings,
        key_proj,
        value_proj,
        norm_key,
        norm_query,
        norm_conv,
        conv,
        state,
    ] = inputs
    else {
        return Err(invalid("ngram-ple requires nine inputs"));
    };
    let hc = config.hc_count;
    if hc < 2
        || config.conv_kernel == 0
        || config.conv_dilation == 0
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
    {
        return Err(invalid("invalid ngram-ple geometry"));
    }
    let [tokens, width] = hidden.shape[..] else {
        return Err(invalid("ngram-ple residual must be rank two"));
    };
    let [_, embed] = embeddings.shape[..] else {
        return Err(invalid("ngram-ple embeddings must be rank two"));
    };
    if width == 0 || !width.is_multiple_of(hc as u64) {
        return Err(invalid("ngram-ple width must be a multiple of hc_count"));
    }
    let stream = width / hc as u64;
    let state_len = config.state_len() as u64;
    check(hidden, &[tokens, width], "ngram-ple residual")?;
    check(embeddings, &[tokens, embed], "ngram-ple embeddings")?;
    check(key_proj, &[width, embed], "ngram-ple key projection")?;
    check(value_proj, &[stream, embed], "ngram-ple value projection")?;
    for (norm, what) in [
        (norm_key, "key norm"),
        (norm_query, "query norm"),
        (norm_conv, "convolution norm"),
    ] {
        check(norm, &[width], what)?;
    }
    check(
        conv,
        &[width, config.conv_kernel as u64],
        "ngram-ple convolution",
    )?;
    check(state, &[width, state_len], "ngram-ple history")?;
    let (tokens, width, stream, state_len) = (
        tokens as usize,
        width as usize,
        stream as usize,
        state_len as usize,
    );

    let mut gated_rows = Vec::with_capacity(tokens);
    let mut normed_rows = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let e = &embeddings.data[token * embeddings.shape[1] as usize..]
            [..embeddings.shape[1] as usize];
        let key = grouped_norm(&project(key_proj, e), &norm_key.data, stream, &config);
        let value = project(value_proj, e);
        let query = grouped_norm(
            &hidden.data[token * width..(token + 1) * width],
            &norm_query.data,
            stream,
            &config,
        );
        let mut gated = vec![0.0f32; width];
        for s in 0..hc {
            let range = s * stream..(s + 1) * stream;
            let dot: f32 = key[range.clone()]
                .iter()
                .zip(&query[range.clone()])
                .map(|(k, q)| k * q)
                .sum();
            let gate = dot / (stream as f32).sqrt();
            let signed =
                gate.abs().max(1e-6).sqrt() * if gate == 0.0 { 0.0 } else { gate.signum() };
            let weight = sigmoid(signed);
            for (index, &v) in value.iter().enumerate() {
                gated[s * stream + index] = weight * v;
            }
        }
        normed_rows.push(grouped_norm(&gated, &norm_conv.data, stream, &config));
        gated_rows.push(gated);
    }

    // Time axis per channel: the history, then every token's normed value.
    let total = state_len + tokens;
    let at = |channel: usize, time: usize| -> f32 {
        if time < state_len {
            state.data[channel * state_len + time]
        } else {
            normed_rows[time - state_len][channel]
        }
    };
    let kernel = config.conv_kernel;
    let mut out = Vec::with_capacity(tokens * width);
    for (token, gated) in gated_rows.iter().enumerate() {
        for (channel, &value) in gated.iter().enumerate() {
            let taps = &conv.data[channel * kernel..(channel + 1) * kernel];
            let sum: f32 = (0..kernel)
                .map(|tap| taps[tap] * at(channel, token + tap * config.conv_dilation))
                .sum();
            out.push(value + silu(sum));
        }
    }
    let mut next = Vec::with_capacity(width * state_len);
    for channel in 0..width {
        next.extend((total - state_len..total).map(|time| at(channel, time)));
    }
    Ok([
        HostTensor::new([tokens as u64, width as u64], out)?,
        HostTensor::new([width as u64, state_len as u64], next)?,
    ])
}
