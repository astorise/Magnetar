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

use crate::affinity::ProviderPressureLevel;
use crate::affinity::{CapabilityBinding, ExecutionContextId, ProviderBinding};
use crate::capability::CapabilityId;
use crate::compute::{
    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION, ComputeGraphId, DTypeDescriptor,
    ShapeDescriptor, TensorDescriptor, TensorResourceDescriptor, TensorResourceId,
};
use crate::kernel::{
    KernelAdvertisement, KernelError, KernelInvocation, KernelInvocationId, KernelMemoryClass,
    KernelResource, KernelResultStatus,
};
use crate::kernel_registry::validate_kernel_advertisement;
use crate::memory::{MemoryManager, MemoryManagerConfig};
use crate::observability::TraceId;
use crate::operator::TensorLayoutKind;
use crate::operator::{OperatorAttributeValue, TensorRole, initial_operator_catalog};
use crate::planning::{
    ComputeExecutionClassification, ComputeExecutionPlan, ExecutionPlanId, MemoryPlan,
};
use crate::provider::ProviderExecutionApi;
use crate::resolution::ResolutionPolicyId;
use crate::scheduler::{
    ProviderCancellationOutcome, ProviderExecutionHandle, ProviderExecutionRequest,
    ScheduledOperationId, SchedulingState,
};
use crate::tensor::ReferenceCpuErrorCode;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
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

// ---------------------------------------------------------------------------
// Incremental decode
// ---------------------------------------------------------------------------

/// A decode step against a populated cache must produce exactly what the full
/// prefill produces for that position.
///
/// This is the property incremental decoding rests on, and the one the kernels
/// could not express before: `attention` required the query and key sequences
/// to be the same length, so a single new token against N cached keys was
/// rejected outright.
#[test]
fn attention_decode_step_matches_the_corresponding_prefill_row() {
    let mut rng = Rng(0xdec0);
    for (head_count, kv_head_count, head_dimension) in [(1_u64, 1_u64, 2_u64), (4, 2, 3)] {
        for total_len in 1..=6_u64 {
            let k = rng.tensor(total_len, kv_head_count * head_dimension);
            let v = rng.tensor(total_len, kv_head_count * head_dimension);
            let q_full = rng.tensor(total_len, head_count * head_dimension);

            let prefill = attention(
                &q_full,
                &k,
                &v,
                head_count,
                head_dimension,
                Some(kv_head_count),
                None,
                true,
            )
            .unwrap();

            // The last row of the prefill is the same computation a decode
            // step performs for that token.
            let width = (head_count * head_dimension) as usize;
            let last = (total_len as usize - 1) * width;
            let expected = &prefill.data[last..last + width];

            let q_step = HostTensor::new(
                [1, head_count * head_dimension],
                &q_full.data[last..last + width],
            )
            .unwrap();
            let decode = attention(
                &q_step,
                &k,
                &v,
                head_count,
                head_dimension,
                Some(kv_head_count),
                None,
                true,
            )
            .unwrap();

            assert_eq!(
                decode.data, expected,
                "heads {head_count}/{kv_head_count} dim {head_dimension} len {total_len}"
            );
        }
    }
}

/// The sliding window must be measured from the query's absolute position, not
/// from its index within the (length-1) decode query.
#[test]
fn attention_decode_step_windows_from_the_absolute_position() {
    let mut rng = Rng(0x1d05);
    let total_len = 6_u64;
    let k = rng.tensor(total_len, 2);
    let v = rng.tensor(total_len, 2);
    let q_full = rng.tensor(total_len, 2);

    for window in [1_u64, 2, 3] {
        let prefill = attention(&q_full, &k, &v, 1, 2, Some(1), Some(window), true).unwrap();
        let last = (total_len as usize - 1) * 2;
        let expected = &prefill.data[last..last + 2];

        let q_step = HostTensor::new([1, 2], &q_full.data[last..last + 2]).unwrap();
        let decode = attention(&q_step, &k, &v, 1, 2, Some(1), Some(window), true).unwrap();

        assert_eq!(decode.data, expected, "window {window}");
    }
}

#[test]
fn attention_rejects_more_queries_than_cached_keys() {
    let mut rng = Rng(11);
    let q = rng.tensor(3, 2);
    let k = rng.tensor(2, 2);
    let v = rng.tensor(2, 2);
    let error = attention(&q, &k, &v, 1, 2, Some(1), None, true)
        .expect_err("more queries than keys must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

/// RoPE for a decode step must rotate by the token's absolute position, which
/// is what the offset carries. Without it, every generated token would be
/// rotated as if it were the first.
#[test]
fn rope_offset_matches_the_corresponding_prefill_row() {
    let mut rng = Rng(0x0ffe);
    let rows = 5_u64;
    let cols = 4_u64;
    let full = rng.tensor(rows, cols);

    let prefill = rope(&full, 10000.0, 1.0, cols, 0, 1).unwrap();

    for row in 0..rows as usize {
        let start = row * cols as usize;
        let single = HostTensor::new([1, cols], &full.data[start..start + cols as usize]).unwrap();
        let stepped = rope(&single, 10000.0, 1.0, cols, row as u64, 1).unwrap();

        assert_eq!(
            stepped.data,
            &prefill.data[start..start + cols as usize],
            "row {row}"
        );
    }
}

#[test]
fn rope_offset_zero_is_the_previous_behaviour() {
    let mut rng = Rng(0x0ff0);
    let input = rng.tensor(4, 4);
    let rotated = rope(&input, 10000.0, 1.0, 4, 0, 1).unwrap();
    // Position 0 leaves the first row untouched, as before the offset existed.
    assert_eq!(rotated.data[..4], input.data[..4]);
}

/// `make-first-native-cuda-hot-path-device-resident` task 3.5 (in-crate
/// mirror of `providers/cpu`'s own test): a multi-head call (`head_count >
/// 1`, `dimension == head_width`) must rotate each head's column block
/// exactly as if that block had been sliced out and rotated on its own
/// with `head_count = 1`. This is the in-crate `reference_cpu.rs` copy of
/// `rope` -- the one `first_native_runtime.rs`'s own dispatch actually
/// runs through -- independently verified from `providers/cpu`'s copy.
#[test]
fn rope_multi_head_matches_independent_single_head_slices() {
    let mut rng = Rng(0x0fe1);
    let rows = 3_u64;
    let head_count = 4_u64;
    let head_width = 2_u64;
    let cols = head_count * head_width;
    let input = rng.tensor(rows, cols);

    let multi_head = rope(&input, 10000.0, 1.0, head_width, 5, head_count).unwrap();

    for head in 0..head_count as usize {
        let mut slice_data = Vec::with_capacity((rows * head_width) as usize);
        for row in 0..rows as usize {
            let start = row * cols as usize + head * head_width as usize;
            slice_data.extend_from_slice(&input.data[start..start + head_width as usize]);
        }
        let slice = HostTensor::new([rows, head_width], slice_data).unwrap();
        let single_head = rope(&slice, 10000.0, 1.0, head_width, 5, 1).unwrap();

        for row in 0..rows as usize {
            let multi_start = row * cols as usize + head * head_width as usize;
            let single_start = row * head_width as usize;
            assert_eq!(
                multi_head.data[multi_start..multi_start + head_width as usize],
                single_head.data[single_start..single_start + head_width as usize],
                "head {head}, row {row}"
            );
        }
    }
}

/// Partial RoPE (`dimension < head_width`): only the first `dimension`
/// columns of each head's block rotate; the remainder is preserved
/// unchanged, not zeroed.
#[test]
fn rope_partial_rotation_preserves_the_untouched_tail() {
    let mut rng = Rng(0x0fe2);
    let rows = 2_u64;
    let head_count = 2_u64;
    let head_width = 4_u64;
    let dimension = 2_u64;
    let cols = head_count * head_width;
    let input = rng.tensor(rows, cols);

    let rotated = rope(&input, 10000.0, 1.0, dimension, 0, head_count).unwrap();

    for row in 0..rows as usize {
        for head in 0..head_count as usize {
            let block_start = row * cols as usize + head * head_width as usize;
            let tail_start = block_start + dimension as usize;
            let tail_end = block_start + head_width as usize;
            assert_eq!(
                rotated.data[tail_start..tail_end],
                input.data[tail_start..tail_end],
                "head {head}, row {row}: untouched tail must be preserved, not zeroed"
            );
        }
    }
}

/// GQA-shaped independence: two RoPE calls with different `head_count`
/// (as Q and K would have under grouped-query attention) each derive
/// their own `head_width` from their own `cols`/`head_count`.
#[test]
fn rope_head_count_is_independent_per_call_matching_gqa_shapes() {
    let mut rng = Rng(0x0fe3);
    let rows = 2_u64;
    let head_width = 2_u64;
    let q_head_count = 4_u64;
    let kv_head_count = 2_u64;

    for head_count in [q_head_count, kv_head_count] {
        let cols = head_count * head_width;
        let input = rng.tensor(rows, cols);
        let multi_head = rope(&input, 10000.0, 1.0, head_width, 3, head_count).unwrap();
        for head in 0..head_count as usize {
            let mut slice_data = Vec::with_capacity((rows * head_width) as usize);
            for row in 0..rows as usize {
                let start = row * cols as usize + head * head_width as usize;
                slice_data.extend_from_slice(&input.data[start..start + head_width as usize]);
            }
            let slice = HostTensor::new([rows, head_width], slice_data).unwrap();
            let single_head = rope(&slice, 10000.0, 1.0, head_width, 3, 1).unwrap();
            for row in 0..rows as usize {
                let multi_start = row * cols as usize + head * head_width as usize;
                let single_start = row * head_width as usize;
                assert_eq!(
                    multi_head.data[multi_start..multi_start + head_width as usize],
                    single_head.data[single_start..single_start + head_width as usize],
                    "head_count {head_count}, head {head}, row {row}"
                );
            }
        }
    }
}

#[test]
fn rope_rejects_head_count_that_does_not_evenly_divide_cols() {
    let mut rng = Rng(0x0fe4);
    let input = rng.tensor(2, 6);
    let result = rope(&input, 10000.0, 1.0, 2, 0, 4);
    assert!(
        result.is_err(),
        "6 columns does not divide evenly into 4 heads"
    );
}

#[test]
fn rope_rejects_dimension_larger_than_head_width() {
    let mut rng = Rng(0x0fe5);
    let input = rng.tensor(2, 8);
    let result = rope(&input, 10000.0, 1.0, 6, 0, 2);
    assert!(result.is_err(), "dimension must not exceed head_width");
}

#[test]
fn split_last_dim_in_half_splits_each_row_correctly() {
    let input = HostTensor::new([2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let (left, right) = split_last_dim_in_half(&input).unwrap();
    assert_eq!(left.shape, vec![2, 2]);
    assert_eq!(right.shape, vec![2, 2]);
    assert_eq!(left.data, vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(right.data, vec![3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn split_last_dim_in_half_handles_rank_one_input() {
    let input = HostTensor::new([4], [10.0, 20.0, 30.0, 40.0]).unwrap();
    let (left, right) = split_last_dim_in_half(&input).unwrap();
    assert_eq!(left.shape, vec![2]);
    assert_eq!(right.shape, vec![2]);
    assert_eq!(left.data, vec![10.0, 20.0]);
    assert_eq!(right.data, vec![30.0, 40.0]);
}

#[test]
fn split_last_dim_in_half_rejects_odd_last_dimension() {
    let input = HostTensor::new([3], [1.0, 2.0, 3.0]).unwrap();
    assert_eq!(
        split_last_dim_in_half(&input).unwrap_err().code,
        ReferenceCpuErrorCode::ShapeUnsupported
    );
}

#[test]
fn split_last_dim_in_half_rejects_zero_rank() {
    // An empty shape is a scalar (rank 0, exactly one element) -- there is
    // no "last dimension" at all to split.
    let input = HostTensor::new(Vec::<u64>::new(), vec![42.0]).unwrap();
    assert_eq!(
        split_last_dim_in_half(&input).unwrap_err().code,
        ReferenceCpuErrorCode::ShapeUnsupported
    );
}

/// `implement-device-resident-multi-step-cuda-decode` task 2.3: KV-history
/// concatenation's exact contract -- `a`'s rows stacked above `b`'s rows,
/// trailing dimensions preserved.
#[test]
fn concat_stacks_a_rows_above_b_rows() {
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([1, 3], [7.0, 8.0, 9.0]).unwrap();
    let result = concat(&a, &b).unwrap();
    assert_eq!(result.shape, vec![3, 3]);
    assert_eq!(
        result.data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn concat_rejects_a_trailing_dimension_mismatch() {
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([1, 4], [7.0, 8.0, 9.0, 10.0]).unwrap();
    assert_eq!(
        concat(&a, &b).unwrap_err().code,
        ReferenceCpuErrorCode::ShapeUnsupported
    );
}

#[test]
fn concat_handles_rank_one_input() {
    let a = HostTensor::new([2], [1.0, 2.0]).unwrap();
    let b = HostTensor::new([3], [3.0, 4.0, 5.0]).unwrap();
    let result = concat(&a, &b).unwrap();
    assert_eq!(result.shape, vec![5]);
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

use crate::affinity::{FallbackClass, ResourceAffinity};
use crate::compute::ComputeDType;
use crate::device::DeviceType;
use crate::kernel::{KernelImplementationFamily, KernelObservationKind};
use crate::tensor::HostTensor;

fn reference_cpu_host_tensor(shape: impl Into<Vec<u64>>, data: impl Into<Vec<f32>>) -> HostTensor {
    HostTensor::new(shape, data).unwrap()
}

#[test]
fn reference_cpu_provider_identity_and_device_are_stable() {
    let provider = ReferenceCpuProvider::new();
    let metadata = provider.metadata();
    assert_eq!(metadata.name, REFERENCE_CPU_PROVIDER_NAME);
    assert_eq!(metadata.vendor, REFERENCE_CPU_PROVIDER_VENDOR);

    let devices = provider.devices();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), REFERENCE_CPU_DEVICE_ID);
    assert_eq!(devices[0].device_type(), DeviceType::Cpu);

    let (min, max) = REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE;
    assert!(min <= max);
    assert_eq!(
        REFERENCE_CPU_KERNEL_FAMILY,
        KernelImplementationFamily::CpuScalar
    );
}

#[test]
fn reference_cpu_conformance_report_passes_and_is_observed() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let report = executor.run_conformance_checks();
    assert!(
        report.is_conformant(),
        "Reference CPU conformance checks failed: {:?}",
        report.checks
    );
    assert_eq!(report.profile, REFERENCE_CPU_CONFORMANCE_PROFILE);
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelConformanceResult)
    );
}

#[test]
fn reference_cpu_dtype_conversion_rejects_non_f32() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(dtype_conversion(&input, ComputeDType::Float32, ComputeDType::Float32).is_ok());
    assert!(dtype_conversion(&input, ComputeDType::Float16, ComputeDType::Float32).is_err());
}

#[test]
fn reference_cpu_fallback_denied_when_dtype_or_layout_conversion_forbidden() {
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    let dtype_denied = FallbackPolicyContext::new(true).with_dtype_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &dtype_denied).is_err());

    let layout_denied = FallbackPolicyContext::new(true).with_layout_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &layout_denied).is_err());

    let both_allowed = FallbackPolicyContext::new(true)
        .with_dtype_conversion(true, true)
        .with_layout_conversion(true, true);
    assert!(evaluate_fallback(&transparent, &both_allowed).is_ok());
}

#[test]
fn reference_cpu_provider_pressure_is_explicitly_reportable() {
    let provider = ReferenceCpuProvider::new();
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Low
    );
    provider.report_pressure(ProviderPressureLevel::Saturated);
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn reference_cpu_advertises_only_implemented_kernels() {
    let provider = ReferenceCpuProvider::new();
    let advertisements = provider.kernel_advertisements();
    let names = advertisements
        .iter()
        .map(|advertisement| advertisement.id.name.as_str())
        .collect::<BTreeSet<_>>();
    for expected in [
        "matmul",
        "embedding",
        "rmsnorm",
        "rope",
        "attention",
        "softmax",
        "silu",
        "gelu",
        "activation",
        "add",
        "mul",
        "residual-add",
        "dtype-conversion",
        "layout-conversion",
    ] {
        assert!(names.contains(expected), "missing kernel: {expected}");
    }
    assert!(!names.contains("quantize"));
    assert!(!names.contains("dequantize"));
    for advertisement in &advertisements {
        validate_kernel_advertisement(advertisement).unwrap();
    }
}

#[test]
fn reference_cpu_attention_rejects_window_without_causal_mask() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // The window is anchored at the query position, which only fully describes
    // the mask under causal attention.
    let error = attention(&q, &k, &v, 1, 1, None, Some(1), false)
        .expect_err("bidirectional sliding window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_fallback_denied_by_default_allowed_by_policy() {
    let pinned = ResourceAffinity::new(FallbackClass::ProviderPinned);
    assert!(evaluate_fallback(&pinned, &FallbackPolicyContext::new(true)).is_err());

    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(false)).is_err());
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(true)).is_ok());
}

#[test]
fn reference_cpu_fallback_is_observable() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let kernel = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(true))
        .unwrap();
    let observations = executor.observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackConsidered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackUsed)
    );

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(false))
        .unwrap_err();
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackFailed)
    );
}

#[test]
fn reference_cpu_quantize_and_dequantize_placeholders_reject_explicitly() {
    for error in [dequantize_placeholder(), quantize_placeholder()] {
        assert_eq!(error.code, ReferenceCpuErrorCode::DTypeUnsupported);
    }
}

#[test]
fn reference_cpu_matmul_known_output() {
    let a = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let b = reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
    let result = matmul(&a, &b, false, false).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn reference_cpu_matmul_rejects_inner_dimension_mismatch() {
    let a = reference_cpu_host_tensor([2, 3], vec![0.0; 6]);
    let b = reference_cpu_host_tensor([2, 2], vec![0.0; 4]);
    assert!(matmul(&a, &b, false, false).is_err());
}

#[test]
fn reference_cpu_embedding_known_output_and_out_of_range() {
    let table = reference_cpu_host_tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ids = reference_cpu_host_tensor([2], [0.0, 2.0]);
    let result = embedding_lookup(&table, &ids).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![1.0, 2.0, 5.0, 6.0]);

    let out_of_range = reference_cpu_host_tensor([1], [3.0]);
    assert!(embedding_lookup(&table, &out_of_range).is_err());
}

#[test]
fn reference_cpu_rmsnorm_known_output() {
    let input = reference_cpu_host_tensor([1, 4], [1.0, 2.0, 3.0, 4.0]);
    let weight = reference_cpu_host_tensor([4], [1.0, 1.0, 1.0, 1.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let mean_square = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    for (actual, expected) in result.data.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - expected * scale).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_rmsnorm_full_shape_weights_apply_per_row() {
    let input = reference_cpu_host_tensor([2, 2], [3.0, 4.0, 3.0, 4.0]);
    let weight = reference_cpu_host_tensor([2, 2], [1.0, 1.0, 2.0, 3.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let scale = 1.0 / (((9.0_f32 + 16.0) / 2.0) + 1e-6).sqrt();

    assert!((result.data[0] - 3.0 * scale).abs() < 1e-5);
    assert!((result.data[1] - 4.0 * scale).abs() < 1e-5);
    assert!((result.data[2] - 3.0 * scale * 2.0).abs() < 1e-5);
    assert!((result.data[3] - 4.0 * scale * 3.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_rmsnorm_flattens_leading_dimensions() {
    let input = reference_cpu_host_tensor([1, 2, 2], [3.0, 4.0, 5.0, 12.0]);
    let weight = reference_cpu_host_tensor([1, 2, 2], [1.0, 2.0, 3.0, 4.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let row0_scale = 1.0 / (((9.0_f32 + 16.0) / 2.0) + 1e-6).sqrt();
    let row1_scale = 1.0 / (((25.0_f32 + 144.0) / 2.0) + 1e-6).sqrt();

    assert_eq!(result.shape, vec![1, 2, 2]);
    assert!((result.data[0] - 3.0 * row0_scale).abs() < 1e-5);
    assert!((result.data[1] - 4.0 * row0_scale * 2.0).abs() < 1e-5);
    assert!((result.data[2] - 5.0 * row1_scale * 3.0).abs() < 1e-5);
    assert!((result.data[3] - 12.0 * row1_scale * 4.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_rmsnorm_rejects_dtype_shape_mismatch() {
    let input = reference_cpu_host_tensor([1, 4], vec![1.0; 4]);
    let weight = reference_cpu_host_tensor([3], vec![1.0; 3]);
    assert!(rmsnorm(&input, &weight, 1e-6).is_err());
}

#[test]
fn reference_cpu_rope_identity_at_position_zero() {
    let input = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let result = rope(&input, 10000.0, 1.0, 2, 0, 1).unwrap();
    assert!((result.data[0] - 1.0).abs() < 1e-5);
    assert!((result.data[1] - 2.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_softmax_known_output() {
    let input = reference_cpu_host_tensor([1, 3], [1.0, 1.0, 1.0]);
    let result = softmax_rows(&input).unwrap();
    for value in result.data {
        assert!((value - (1.0 / 3.0)).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_softmax_allows_partially_masked_row() {
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
    let result = softmax_rows(&input).unwrap();
    assert!(result.data.iter().all(|value| value.is_finite()));
    assert!((result.data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_silu_known_output() {
    let input = reference_cpu_host_tensor([1], [0.0]);
    let result = silu(&input);
    assert!((result.data[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reference_cpu_elementwise_known_outputs() {
    let a = reference_cpu_host_tensor([2], [1.0, 2.0]);
    let b = reference_cpu_host_tensor([2], [3.0, 4.0]);
    assert_eq!(add(&a, &b).unwrap().data, vec![4.0, 6.0]);
    assert_eq!(mul(&a, &b).unwrap().data, vec![3.0, 8.0]);
    assert_eq!(residual_add(&a, &b).unwrap().data, vec![4.0, 6.0]);

    let mismatched = reference_cpu_host_tensor([3], vec![0.0; 3]);
    assert!(add(&a, &mismatched).is_err());
}

#[test]
fn reference_cpu_attention_causal_masks_future_tokens() {
    let q = reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 2], [10.0, 10.0, 20.0, 20.0]);
    let result = attention(&q, &k, &v, 1, 2, None, None, true).unwrap();
    // Position 0 can only attend to itself, so its output must equal v[0].
    assert!((result.data[0] - 10.0).abs() < 1e-4);
    assert!((result.data[1] - 10.0).abs() < 1e-4);
}

#[test]
fn reference_cpu_attention_grouped_query_shares_kv_heads() {
    // 2 query heads sharing 1 kv head (head_dimension = 2).
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 2], [5.0, 6.0]);
    let v = reference_cpu_host_tensor([1, 2], [7.0, 8.0]);
    let result = attention(&q, &k, &v, 2, 2, Some(1), None, false).unwrap();
    // Single key position: every query head's output must equal v.
    assert_eq!(result.data, vec![7.0, 8.0, 7.0, 8.0]);
}

#[test]
fn reference_cpu_attention_rejects_incompatible_head_grouping() {
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 4], [5.0, 6.0, 7.0, 8.0]);
    let v = k.clone();
    // head_count 2 is not a multiple of kv_head_count 3.
    assert!(attention(&q, &k, &v, 2, 2, Some(3), None, false).is_err());
}

#[test]
fn reference_cpu_attention_window_size_restricts_context() {
    let q = reference_cpu_host_tensor([3, 1], [0.0, 0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([3, 1], [1.0, 2.0, 3.0]);
    // window_size = 1: each position can only see itself.
    let result = attention(&q, &k, &v, 1, 1, None, Some(1), true).unwrap();
    assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn reference_cpu_layout_conversion_rejects_non_contiguous() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Contiguous
        )
        .is_ok()
    );
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Strided
        )
        .is_err()
    );
}

#[test]
fn reference_cpu_quantization_is_explicitly_unsupported() {
    let error = dequantize_placeholder();
    assert_eq!(error.id(), "reference-cpu-dtype-unsupported");
}

fn reference_cpu_resource(
    id: &str,
    shape: impl Into<Vec<u64>>,
) -> (TensorResourceId, KernelResource) {
    let resource_id = TensorResourceId::new(id);
    let descriptor = TensorResourceDescriptor::new(
        resource_id.clone(),
        TensorDescriptor::materialized(
            ShapeDescriptor::new(shape.into()),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    (
        resource_id,
        KernelResource::new(descriptor, KernelMemoryClass::Host),
    )
}

fn reference_cpu_attention_invocation(
    advertisement: &KernelAdvertisement,
    causal: bool,
    mask_kind: Option<&str>,
    q: KernelResource,
    k: KernelResource,
    v: KernelResource,
    out: KernelResource,
) -> KernelInvocation {
    let mut attributes = BTreeMap::new();
    attributes.insert("head_count".to_string(), OperatorAttributeValue::Integer(1));
    attributes.insert(
        "head_dimension".to_string(),
        OperatorAttributeValue::Integer(2),
    );
    attributes.insert(
        "causal".to_string(),
        OperatorAttributeValue::Boolean(causal),
    );
    if let Some(mask_kind) = mask_kind {
        attributes.insert(
            "attention_mask_kind".to_string(),
            OperatorAttributeValue::String(mask_kind.into()),
        );
    }
    KernelInvocation::new(
        KernelInvocationId::new("invocation-attention"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(q)
    .with_input(k)
    .with_input(v)
    .with_output(out)
    .with_attributes(attributes)
}

fn reference_cpu_kernel_by_name<'a>(
    advertisements: &'a [KernelAdvertisement],
    name: &str,
) -> &'a KernelAdvertisement {
    advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == name)
        .unwrap_or_else(|| panic!("no advertisement named {name}"))
}

#[test]
fn reference_cpu_device_advertises_dtype_layout_memory_class_and_limits() {
    let device = reference_cpu_device();
    let metadata = device.metadata();
    assert!(metadata.dtype_support.contains(&ComputeDType::Float32));
    assert!(
        metadata
            .layout_support
            .contains(&TensorLayoutKind::Contiguous)
    );
    assert!(
        metadata
            .memory_class_support
            .contains(&KernelMemoryClass::Host)
    );
    assert!(
        metadata
            .execution_limits
            .max_concurrent_operations
            .is_some()
    );
    assert_eq!(metadata.pressure, ProviderPressureLevel::Low);
}

#[test]
fn reference_cpu_initialize_emits_provider_registered_and_device_detected() {
    let provider = ReferenceCpuProvider::new();
    provider.initialize().unwrap();
    let observations = provider.executor().observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::ProviderRegistered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::DeviceDetected)
    );
}

#[test]
fn reference_cpu_softmax_rejects_invalid_shape() {
    let input = HostTensor {
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
    };
    assert!(softmax_rows(&input).is_err());
}

#[test]
fn reference_cpu_softmax_rejects_fully_masked_row() {
    // Every entry masked out: subtracting the row max would yield NaN for the
    // whole row, so the kernel must reject it rather than return Ok(NaN).
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY; 3]);
    let error = softmax_rows(&input).expect_err("fully masked row must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ExecutionFailed);
}

#[test]
fn reference_cpu_attention_rejects_zero_window() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // A zero window admits no keys at all; it must not be silently widened to 1.
    let error =
        attention(&q, &k, &v, 1, 1, None, Some(0), true).expect_err("zero window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_advertisements_match_numeric_storage_constraints() {
    let advertisements = reference_cpu_kernel_advertisements();
    let rmsnorm = reference_cpu_kernel_by_name(&advertisements, "rmsnorm");
    assert_eq!(rmsnorm.shape.rank, None);

    let embedding = reference_cpu_kernel_by_name(&advertisements, "embedding");
    let input_dtypes = embedding
        .supported_dtypes
        .get(&TensorRole::Input)
        .expect("embedding advertises input dtypes");
    assert!(!input_dtypes.contains(&ComputeDType::SInt32));
    assert!(input_dtypes.contains(&ComputeDType::Float32));
}

#[test]
fn reference_cpu_attention_requires_workspace_from_memory_manager() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "attention");
    assert!(advertisement.workspace.required);
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (q_id, q_resource) = reference_cpu_resource("attn-q", [1, 2]);
    let (k_id, k_resource) = reference_cpu_resource("attn-k", [1, 2]);
    let (v_id, v_resource) = reference_cpu_resource("attn-v", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("attn-out", [1, 2]);
    executor.write_tensor(q_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(k_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(v_id, reference_cpu_host_tensor([1, 2], [5.0, 6.0]));

    // Without a workspace attached, the shared Kernel Contract validation
    // rejects the invocation before Reference CPU ever runs it.
    let invocation_without_workspace = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource.clone(),
        k_resource.clone(),
        v_resource.clone(),
        out_resource.clone(),
    );
    let rejected =
        executor.execute_invocation(advertisement, operator, &invocation_without_workspace);
    assert_eq!(rejected.status, KernelResultStatus::Failed);
    assert_eq!(
        rejected.error,
        Some(KernelError::KernelWorkspaceUnavailable)
    );

    // With a workspace requested through the Memory Manager, execution
    // succeeds.
    let workspace = executor.allocate_workspace(&mut memory, 4096).unwrap();
    let invocation = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource,
        k_resource,
        v_resource,
        out_resource.clone(),
    )
    .with_workspace(workspace);
    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert_eq!(output.data, vec![5.0, 6.0]);
}

#[test]
fn reference_cpu_kernel_submission_is_causal_and_single_consumption() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (a_id, a_resource) = reference_cpu_resource("submit-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("submit-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("submit-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-submit"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    assert!(
        executor.observations().is_empty(),
        "no dispatch should have happened before submission"
    );

    // submit_kernel_invocation is what causally triggers the numerical
    // work; Reference CPU is synchronous, so it has already run by the time
    // this call returns.
    let handle =
        executor.submit_kernel_invocation(advertisement, operator, &invocation, &mut memory);
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelDispatchStarted)
    );

    let result = executor
        .complete_kernel_invocation(&handle)
        .expect("work submitted above is completable exactly once");
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    assert_eq!(result.updated_resources.len(), 1);

    // Single consumption: completing the same handle a second time fails
    // rather than silently re-reporting the same result.
    assert!(executor.complete_kernel_invocation(&handle).is_err());
}

#[test]
fn reference_cpu_kernel_completion_reports_real_failure_not_false_success() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (_a_id, a_resource) = reference_cpu_resource("submit-fail-a", [2, 2]);
    let (_b_id, b_resource) = reference_cpu_resource("submit-fail-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("submit-fail-out", [2, 2]);
    // Inputs are intentionally left unwritten so the Kernel itself fails.

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-submit-fail"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let handle =
        executor.submit_kernel_invocation(advertisement, operator, &invocation, &mut memory);
    let result = executor
        .complete_kernel_invocation(&handle)
        .expect("a submitted invocation is completable even when the Kernel itself failed");
    assert_eq!(
        result.status,
        KernelResultStatus::Failed,
        "completion must report the real Kernel failure, not fabricate a success"
    );
    assert!(result.error.is_some());
}

#[test]
fn reference_cpu_rejects_completion_of_a_handle_that_was_never_submitted() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let fabricated_handle = ProviderExecutionHandle::new(
        ScheduledOperationId::new(0xDEAD_BEEF),
        ExecutionPlanId::new("never-submitted-plan"),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        None,
    );
    assert!(
        executor
            .complete_kernel_invocation(&fabricated_handle)
            .is_err()
    );
    // The generic ProviderExecutionApi surface rejects the same fabricated
    // handle for the same reason: no submission is associated with it.
    assert!(ProviderExecutionApi::complete(executor.as_ref(), &fabricated_handle).is_err());
    assert!(ProviderExecutionApi::status(executor.as_ref(), &fabricated_handle).is_err());
}

#[test]
fn reference_cpu_cancellation_is_explicitly_unsupported_not_silently_ignored() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let handle = ProviderExecutionHandle::new(
        ScheduledOperationId::new(1),
        ExecutionPlanId::new("cancel-probe-plan"),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        None,
    );
    let outcome = ProviderExecutionApi::cancel(executor.as_ref(), &handle).unwrap();
    assert_eq!(outcome, ProviderCancellationOutcome::Unsupported);
}

#[test]
fn reference_cpu_generic_provider_execution_api_completes_exactly_once() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let provider_binding = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
    let plan = ComputeExecutionPlan {
        id: ExecutionPlanId::new("generic-submit-plan"),
        trace_id: TraceId::new("trace-generic-submit"),
        graph: ComputeGraphId::new("generic-submit-graph"),
        provider: provider_binding.clone(),
        device: None,
        capability: CapabilityBinding::new(
            CapabilityId::new(COMPUTE_CAPABILITY_ID),
            COMPUTE_CAPABILITY_VERSION,
        ),
        policy: ResolutionPolicyId::new("generic-submit-policy"),
        classification: ComputeExecutionClassification::Transparent,
        inputs: Vec::new(),
        outputs: Vec::new(),
        constraints: Vec::new(),
        steps: Vec::new(),
        memory_plan: MemoryPlan::new(provider_binding.clone(), None, ExecutionContextId::new(0)),
        diagnostics: Vec::new(),
        validated: true,
    };
    let request = ProviderExecutionRequest {
        operation: ScheduledOperationId::new(1),
        plan,
        provider: provider_binding.clone(),
        device: None,
        affinity: ResourceAffinity::new(FallbackClass::Transparent),
        memory_plan: MemoryPlan::new(provider_binding, None, ExecutionContextId::new(0)),
        steps: Vec::new(),
        constraints: Vec::new(),
    };

    let handle = ProviderExecutionApi::submit(executor.as_ref(), request).unwrap();
    let status = ProviderExecutionApi::status(executor.as_ref(), &handle).unwrap();
    assert_eq!(status.state, SchedulingState::Completed);
    let result = ProviderExecutionApi::complete(executor.as_ref(), &handle).unwrap();
    assert_eq!(result.state, SchedulingState::Completed);
    ProviderExecutionApi::release(executor.as_ref(), handle.clone()).unwrap();

    // submit -> status -> complete -> release is now exhausted: neither
    // status nor a second complete succeeds against the same handle.
    assert!(ProviderExecutionApi::status(executor.as_ref(), &handle).is_err());
    assert!(ProviderExecutionApi::complete(executor.as_ref(), &handle).is_err());
}
