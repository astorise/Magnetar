//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

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
    let out = batched_matmul(&a, &b, groups, false).unwrap();
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
                let actual = out.data[row * (groups * n) as usize + group * n as usize + column];
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
    // `transpose_b`: each group's block stored `[N, K]`.
    let mut transposed = vec![0.0f32; b.data.len()];
    for group in 0..groups as usize {
        for i in 0..k as usize {
            for column in 0..n as usize {
                transposed[(group * n as usize + column) * k as usize + i] =
                    b.data[(group * k as usize + i) * n as usize + column];
            }
        }
    }
    let b_t = HostTensor::new([groups * n, k], transposed).unwrap();
    assert_eq!(batched_matmul(&a, &b_t, groups, true).unwrap(), out);
    assert!(batched_matmul(&a, &tensor(groups * k + 1, n, 2), groups, false).is_err());
    assert!(batched_matmul(&a, &b, 3, false).is_err());
    assert!(batched_matmul(&a, &b, groups, true).is_err());
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
