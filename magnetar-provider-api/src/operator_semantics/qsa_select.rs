//! Qwen4-Exp's QSA token selection (`support-qwen38-flash-next`): which
//! earlier tokens each query attends. The formulas are those of
//! Transformers' `Qwen4ExpTextQSAIndexer`, pinned in design D7.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

const MAX_ELEMENTS: u64 = 67_108_864;

/// A `qsa-select` node's attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaSelect {
    /// Indexer query heads; the indexer has one key head.
    pub head_count: usize,
    pub head_dim: usize,
    /// Consecutive tokens averaged into one block.
    pub compress_ratio: usize,
    /// Tokens kept from complete blocks (`budget / compress_ratio` blocks).
    pub budget: usize,
    pub epsilon: f32,
    /// Added to the key norm's weight (1 for the checkpoint's zero-centered
    /// norms).
    pub weight_offset: f32,
    pub rope_base: f32,
    /// Leading dimensions of each head the rotary embedding turns.
    pub rotary_dim: usize,
}

impl QsaSelect {
    /// Width of the selection a query gets: the kept blocks' tokens plus the
    /// longest incomplete tail block.
    pub fn slots(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }
}

fn invalid(reason: impl Into<String>) -> ReferenceCpuError {
    ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason)
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
            "qsa-select {what} must be {shape:?} and finite, got {:?}",
            tensor.shape
        )));
    }
    Ok(())
}

/// Rotates the first `rotary_dim` values of `row` for `position`: pairs
/// `(j, j + rotary_dim / 2)`, frequency `base^(-2j / rotary_dim)`.
fn rotate(row: &mut [f32], position: usize, config: &QsaSelect) {
    let half = config.rotary_dim / 2;
    for j in 0..half {
        let inverse = 1.0
            / config
                .rope_base
                .powf((2 * j) as f32 / config.rotary_dim as f32);
        let angle = position as f32 * inverse;
        let (sin, cos) = angle.sin_cos();
        let (first, second) = (row[j], row[j + half]);
        row[j] = first * cos - second * sin;
        row[j + half] = second * cos + first * sin;
    }
}

/// `inputs`: the queries `[T, head_count * head_dim]` (normalized and
/// rotated already), the raw index keys `[K, head_dim]` of every token so
/// far, and the key norm's weight `[head_dim]`. Query `i` sits at position
/// `K - T + i` and sees every position up to its own.
///
/// For each query: the visible tokens form blocks of `compress_ratio`; the
/// complete blocks' keys are averaged, normalized and rotated at the block's
/// first position; a block scores the sum over heads of `relu(q · key)`,
/// divided by `sqrt(head_dim)`. The `budget / compress_ratio` best blocks
/// (the lower block on a tie) are kept, with every token of the incomplete
/// tail block. Returns `[T, slots]`: the kept token positions in ascending
/// order, padded with `-1`.
pub fn qsa_select(
    inputs: &[HostTensor],
    config: QsaSelect,
) -> Result<HostTensor, ReferenceCpuError> {
    let [queries, keys, norm] = inputs else {
        return Err(invalid("qsa-select requires three inputs"));
    };
    let (heads, dim, ratio) = (config.head_count, config.head_dim, config.compress_ratio);
    if heads == 0
        || dim == 0
        || ratio == 0
        || config.budget == 0
        || !config.budget.is_multiple_of(ratio)
        || config.rotary_dim > dim
        || !config.rotary_dim.is_multiple_of(2)
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
        || !(config.rope_base.is_finite() && config.rope_base > 1.0)
    {
        return Err(invalid("invalid qsa-select geometry"));
    }
    let [tokens, _] = queries.shape[..] else {
        return Err(invalid("qsa-select queries must be rank two"));
    };
    let [key_count, _] = keys.shape[..] else {
        return Err(invalid("qsa-select keys must be rank two"));
    };
    check(queries, &[tokens, (heads * dim) as u64], "queries")?;
    check(keys, &[key_count, dim as u64], "keys")?;
    check(norm, &[dim as u64], "key norm weight")?;
    if key_count < tokens {
        return Err(invalid("qsa-select needs a key for every query"));
    }
    let (tokens, key_count) = (tokens as usize, key_count as usize);
    let slots = config.slots();
    let kept_blocks = config.budget / ratio;
    let first_position = key_count - tokens;
    let scale = (dim as f32).sqrt().recip();

    // A complete block's key depends only on its own tokens, so each is
    // computed once, however many queries see it.
    let blocks = key_count / ratio;
    let block_keys: Vec<Vec<f32>> = (0..blocks)
        .map(|block| {
            let mut pooled = vec![0.0f32; dim];
            for token in block * ratio..(block + 1) * ratio {
                for (sum, value) in pooled
                    .iter_mut()
                    .zip(&keys.data[token * dim..(token + 1) * dim])
                {
                    *sum += value;
                }
            }
            for value in &mut pooled {
                *value /= ratio as f32;
            }
            let mean = pooled.iter().map(|x| x * x).sum::<f32>() / dim as f32;
            let inverse = (mean + config.epsilon).sqrt().recip();
            for (value, weight) in pooled.iter_mut().zip(&norm.data) {
                *value *= inverse * (config.weight_offset + weight);
            }
            rotate(&mut pooled, block * ratio, &config);
            pooled
        })
        .collect();

    let mut out = Vec::with_capacity(tokens * slots);
    for query in 0..tokens {
        let visible = first_position + query + 1;
        let complete = visible / ratio;
        let row = &queries.data[query * heads * dim..(query + 1) * heads * dim];
        let mut scored: Vec<(f32, usize)> = (0..complete)
            .map(|block| {
                let key = &block_keys[block];
                let score: f32 = (0..heads)
                    .map(|head| {
                        let q = &row[head * dim..(head + 1) * dim];
                        q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>().max(0.0)
                    })
                    .sum();
                (score * scale, block)
            })
            .collect();
        // Best score first; the lower block wins a tie.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(kept_blocks);
        let mut selected: Vec<usize> = scored
            .iter()
            .flat_map(|&(_, block)| block * ratio..(block + 1) * ratio)
            .chain(complete * ratio..visible)
            .collect();
        selected.sort_unstable();
        out.extend(selected.iter().map(|&token| token as f32));
        out.extend(std::iter::repeat_n(-1.0, slots - selected.len()));
    }
    HostTensor::new([tokens as u64, slots as u64], out)
}
