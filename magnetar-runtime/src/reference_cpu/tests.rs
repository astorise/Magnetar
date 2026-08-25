//! Unit tests for the Reference CPU numeric kernels.
//!
//! Kept in its own file so coverage tooling classifies it as test source
//! rather than Runtime implementation source.
//!
//! `matmul` and `attention` were restructured for the sake of allocation and
//! cache behaviour. This Provider is the oracle other Providers are validated
//! against, so "faster" is only acceptable if it is also *bit-identical*.
//! These tests compare each kernel against the straightforward form it
//! replaced and require exact equality, not approximate.

use super::*;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn value(&mut self) -> f32 {
        (self.next_u64() % 2001) as f32 / 100.0 - 10.0
    }

    fn tensor(&mut self, rows: u64, cols: u64) -> HostTensor {
        let data = (0..rows * cols).map(|_| self.value()).collect::<Vec<_>>();
        HostTensor::new([rows, cols], data).unwrap()
    }
}

/// The triple loop `matmul` replaced: per-element indexing with the transpose
/// branch inside the innermost loop.
fn reference_matmul(
    a: &HostTensor,
    b: &HostTensor,
    transpose_a: bool,
    transpose_b: bool,
) -> Vec<f32> {
    let (a_rows, a_cols) = a.rows_cols().unwrap();
    let (b_rows, b_cols) = b.rows_cols().unwrap();
    let (m, k) = if transpose_a {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };
    let (_, n) = if transpose_b {
        (b_cols, b_rows)
    } else {
        (b_rows, b_cols)
    };
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let a_at = |row: usize, col: usize| -> f32 {
        if transpose_a {
            a.data[col * (a_cols as usize) + row]
        } else {
            a.data[row * (a_cols as usize) + col]
        }
    };
    let b_at = |row: usize, col: usize| -> f32 {
        if transpose_b {
            b.data[col * (b_cols as usize) + row]
        } else {
            b.data[row * (b_cols as usize) + col]
        }
    };
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut accumulator = 0.0_f32;
            for inner in 0..k {
                accumulator += a_at(row, inner) * b_at(inner, col);
            }
            out[row * n + col] = accumulator;
        }
    }
    out
}

#[test]
fn matmul_is_bit_identical_to_the_straightforward_form() {
    let mut rng = Rng(0x513d);
    for transpose_a in [false, true] {
        for transpose_b in [false, true] {
            for trial in 0..40_u64 {
                let m = 1 + trial % 7;
                let k = 1 + (trial / 2) % 6;
                let n = 1 + (trial / 3) % 5;
                let a = if transpose_a {
                    rng.tensor(k, m)
                } else {
                    rng.tensor(m, k)
                };
                let b = if transpose_b {
                    rng.tensor(n, k)
                } else {
                    rng.tensor(k, n)
                };

                let expected = reference_matmul(&a, &b, transpose_a, transpose_b);
                let actual = matmul(&a, &b, transpose_a, transpose_b).unwrap();

                assert_eq!(
                    actual.data, expected,
                    "m {m} k {k} n {n} ta {transpose_a} tb {transpose_b}"
                );
            }
        }
    }
}

#[test]
fn matmul_propagates_non_finite_values_rather_than_skipping_zeros() {
    // 0.0 * NaN is NaN. A zero-skip optimization would silently drop it, so a
    // zero row in `a` against a NaN column in `b` must still yield NaN.
    let a = HostTensor::new([1, 2], [0.0, 0.0]).unwrap();
    let b = HostTensor::new([2, 1], [f32::NAN, 1.0]).unwrap();
    assert!(matmul(&a, &b, false, false).unwrap().data[0].is_nan());

    let b_infinite = HostTensor::new([2, 1], [f32::INFINITY, 1.0]).unwrap();
    assert!(
        matmul(&a, &b_infinite, false, false).unwrap().data[0].is_nan(),
        "0.0 * inf must be NaN"
    );
}

/// The `attention` body before the scratch buffers were hoisted.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &HostTensor,
    k: &HostTensor,
    v: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    kv_head_count: u64,
    window_size: Option<u64>,
    causal: bool,
) -> Vec<f32> {
    let (seq_len, q_model_dim) = q.rows_cols().unwrap();
    let (_, kv_model_dim) = k.rows_cols().unwrap();
    let group_size = (head_count / kv_head_count) as usize;
    let seq_len = seq_len as usize;
    let q_model_dim = q_model_dim as usize;
    let kv_model_dim = kv_model_dim as usize;
    let head_dimension = head_dimension as usize;
    let scale = 1.0 / (head_dimension as f32).sqrt();
    let mut out = vec![0.0_f32; seq_len * q_model_dim];
    for head in 0..head_count as usize {
        let kv_head = head / group_size;
        let q_offset = head * head_dimension;
        let kv_offset = kv_head * head_dimension;
        for query_index in 0..seq_len {
            let key_upper = if causal { query_index + 1 } else { seq_len };
            let key_lower = window_size
                .map(|window| query_index.saturating_sub((window as usize).saturating_sub(1)))
                .unwrap_or(0)
                .min(key_upper);
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for (key_index, score) in scores
                .iter_mut()
                .enumerate()
                .take(key_upper)
                .skip(key_lower)
            {
                let mut dot = 0.0_f32;
                for dim in 0..head_dimension {
                    dot += q.data[query_index * q_model_dim + q_offset + dim]
                        * k.data[key_index * kv_model_dim + kv_offset + dim];
                }
                *score = dot * scale;
            }
            let max = scores[key_lower..key_upper]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let exponentials = scores[key_lower..key_upper]
                .iter()
                .map(|value| (value - max).exp())
                .collect::<Vec<_>>();
            let sum: f32 = exponentials.iter().sum();
            for dim in 0..head_dimension {
                let mut accumulator = 0.0_f32;
                for (offset, weight) in exponentials.iter().enumerate() {
                    let key_index = key_lower + offset;
                    accumulator +=
                        (weight / sum) * v.data[key_index * kv_model_dim + kv_offset + dim];
                }
                out[query_index * q_model_dim + q_offset + dim] = accumulator;
            }
        }
    }
    out
}

#[test]
fn attention_is_bit_identical_to_the_straightforward_form() {
    let mut rng = Rng(0xa77e);
    let shapes = [
        // (head_count, kv_head_count, head_dimension)
        (1_u64, 1_u64, 1_u64),
        (1, 1, 4),
        (2, 1, 3),
        (4, 2, 2),
        (3, 3, 5),
    ];
    for (head_count, kv_head_count, head_dimension) in shapes {
        for seq_len in 1..=6_u64 {
            for causal in [false, true] {
                // A window is only defined for causal attention.
                let windows: &[Option<u64>] = if causal {
                    &[None, Some(1), Some(2), Some(4)]
                } else {
                    &[None]
                };
                for window_size in windows {
                    let q = rng.tensor(seq_len, head_count * head_dimension);
                    let k = rng.tensor(seq_len, kv_head_count * head_dimension);
                    let v = rng.tensor(seq_len, kv_head_count * head_dimension);

                    let expected = reference_attention(
                        &q,
                        &k,
                        &v,
                        head_count,
                        head_dimension,
                        kv_head_count,
                        *window_size,
                        causal,
                    );
                    let actual = attention(
                        &q,
                        &k,
                        &v,
                        head_count,
                        head_dimension,
                        Some(kv_head_count),
                        *window_size,
                        causal,
                    )
                    .unwrap();

                    assert_eq!(
                        actual.data, expected,
                        "heads {head_count}/{kv_head_count} dim {head_dimension} seq {seq_len} causal {causal} window {window_size:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn attention_scratch_reuse_does_not_leak_between_queries() {
    // The scratch buffers are shared across every (head, query) pair now. A
    // stale tail from a wider earlier window would corrupt a later, narrower
    // one, so walk windows from wide to narrow within one call.
    let mut rng = Rng(0x5c47);
    let q = rng.tensor(6, 4);
    let k = rng.tensor(6, 4);
    let v = rng.tensor(6, 4);

    let expected = reference_attention(&q, &k, &v, 2, 2, 2, Some(3), true);
    let actual = attention(&q, &k, &v, 2, 2, Some(2), Some(3), true).unwrap();

    assert_eq!(actual.data, expected);
}
