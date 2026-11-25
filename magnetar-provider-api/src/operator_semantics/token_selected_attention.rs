//! Attention over a token selection (`support-qwen38-flash-next`): each
//! query reads only the key positions a selection lists, which is how
//! Qwen4-Exp's QSA narrows the causal attention. The selection is the output
//! of `qsa-select`.

use crate::tensor::{HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};

/// Attention reading a token selection
/// (`attention.token_selection`, `support-qwen38-flash-next`): query row `r`
/// reads only the key positions `selection[r]` lists, in ascending order
/// with `-1` after the last, and only those at or before its own position
/// when `causal` (and inside `window_size` when set). `selection` has shape
/// `[q rows, slots]`; positions are `f32` values holding integers, which is
/// exact below 2^24. A row naming no readable key is an error: the
/// selection always keeps the query's own token.
#[allow(clippy::too_many_arguments)]
pub fn attention_over_selection(
    q: &HostTensor,
    k: &HostTensor,
    v: &HostTensor,
    selection: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    kv_head_count: Option<u64>,
    window_size: Option<u64>,
    causal: bool,
    score_scale: Option<f32>,
) -> Result<HostTensor, ReferenceCpuError> {
    let shape_error =
        |reason: String| ReferenceCpuError::new(ReferenceCpuErrorCode::ShapeUnsupported, reason);
    if score_scale.is_some_and(|scale| !scale.is_finite()) {
        return Err(shape_error("attention scale must be finite".into()));
    }
    if k.shape != v.shape {
        return Err(shape_error(format!(
            "k {:?} and v {:?} must have the same shape",
            k.shape, v.shape
        )));
    }
    let kv_head_count = kv_head_count.unwrap_or(head_count);
    if head_count == 0 || head_dimension == 0 || kv_head_count == 0 {
        return Err(shape_error(
            "head_count, kv_head_count, and head_dimension must all be positive".into(),
        ));
    }
    if !head_count.is_multiple_of(kv_head_count) {
        return Err(shape_error(format!(
            "head_count {head_count} must be an exact multiple of kv_head_count {kv_head_count}"
        )));
    }
    if window_size == Some(0) || (window_size.is_some() && !causal) {
        return Err(shape_error(
            "window_size must be positive and is only defined for causal attention".into(),
        ));
    }
    let (seq_len, q_model_dim) = q.rows_cols()?;
    let (kv_seq_len, kv_model_dim) = k.rows_cols()?;
    if seq_len > kv_seq_len
        || head_count * head_dimension != q_model_dim
        || kv_head_count * head_dimension != kv_model_dim
    {
        return Err(shape_error(format!(
            "q {:?} and k/v {:?} do not fit {head_count} heads of {head_dimension} over \
             {kv_head_count} K/V heads",
            q.shape, k.shape
        )));
    }
    let (selection_rows, slots) = selection.rows_cols()?;
    if selection_rows != seq_len {
        return Err(shape_error(format!(
            "the selection has {selection_rows} rows for {seq_len} query rows"
        )));
    }
    let group_size = (head_count / kv_head_count) as usize;
    let (seq_len, kv_seq_len, slots) = (seq_len as usize, kv_seq_len as usize, slots as usize);
    let (q_model_dim, kv_model_dim) = (q_model_dim as usize, kv_model_dim as usize);
    let head_dimension = head_dimension as usize;
    let scale = score_scale.unwrap_or(1.0 / (head_dimension as f32).sqrt());
    let query_position_offset = kv_seq_len - seq_len;
    // Each row's readable positions, validated once for every head.
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(seq_len);
    for query_index in 0..seq_len {
        let query_position = query_position_offset + query_index;
        let lower = window_size
            .map(|window| query_position.saturating_sub((window as usize).saturating_sub(1)))
            .unwrap_or(0);
        let mut positions: Vec<usize> = Vec::new();
        let mut previous: Option<usize> = None;
        let mut padded = false;
        for slot in 0..slots {
            let value = selection.data[query_index * slots + slot];
            if value == -1.0 {
                padded = true;
                continue;
            }
            if padded || !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return Err(shape_error(format!(
                    "selection row {query_index} must list ascending positions, then -1 \
                     padding, got {value}"
                )));
            }
            let position = value as usize;
            if previous.is_some_and(|before| position <= before) {
                return Err(shape_error(format!(
                    "selection row {query_index} is not strictly ascending at {position}"
                )));
            }
            previous = Some(position);
            if position < kv_seq_len && (!causal || position <= query_position) && position >= lower
            {
                positions.push(position);
            }
        }
        if positions.is_empty() {
            return Err(shape_error(format!(
                "selection row {query_index} names no readable key"
            )));
        }
        rows.push(positions);
    }
    let out_model_dim = head_count as usize * head_dimension;
    let mut out = vec![0.0_f32; seq_len * out_model_dim];
    let mut scores = Vec::new();
    let mut exponentials = Vec::new();
    for head in 0..head_count as usize {
        let kv_head = head / group_size;
        let q_offset = head * head_dimension;
        let kv_offset = kv_head * head_dimension;
        for (query_index, positions) in rows.iter().enumerate() {
            let query_base = query_index * q_model_dim + q_offset;
            scores.clear();
            for &key_index in positions {
                let key_base = key_index * kv_model_dim + kv_offset;
                let mut dot = 0.0_f32;
                for dim in 0..head_dimension {
                    dot += q.data[query_base + dim] * k.data[key_base + dim];
                }
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            exponentials.clear();
            exponentials.extend(scores.iter().map(|value| (value - max).exp()));
            let sum: f32 = exponentials.iter().sum();
            let out_base = query_index * out_model_dim + head * head_dimension;
            for dim in 0..head_dimension {
                let mut accumulator = 0.0_f32;
                for (weight, &key_index) in exponentials.iter().zip(positions) {
                    accumulator +=
                        (weight / sum) * v.data[key_index * kv_model_dim + kv_offset + dim];
                }
                out[out_base + dim] = accumulator;
            }
        }
    }
    HostTensor::new(vec![seq_len as u64, out_model_dim as u64], out)
}
