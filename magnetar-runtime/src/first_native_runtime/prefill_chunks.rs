//! Chunked prefill (`chunk-long-prefills`): a prompt longer than the chunk
//! length runs as consecutive passes of one prefill graph of the chunk
//! length. Each pass continues the KV history the passes before it wrote in
//! place, at its own start position, so the activations a pass holds are
//! bounded by the chunk length instead of the prompt's. The last pass is
//! right-padded to the chunk length like a bucketed prefill
//! (`replay-bucketed-prefill`), so every pass shares one graph and one plan.

use super::*;

/// The chunk length used when `MAGNETAR_PREFILL_CHUNK` is unset.
pub(crate) const DEFAULT_PREFILL_CHUNK: u64 = 2048;

/// The chunk length this process uses: `MAGNETAR_PREFILL_CHUNK` (`0` turns
/// chunking off), else [`DEFAULT_PREFILL_CHUNK`].
pub(crate) fn prefill_chunk_from_env() -> u64 {
    parse_prefill_chunk(std::env::var("MAGNETAR_PREFILL_CHUNK").ok().as_deref())
}

/// Parses a chunk length; a missing or malformed value gives the default.
pub(crate) fn parse_prefill_chunk(configured: Option<&str>) -> u64 {
    configured
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_PREFILL_CHUNK)
}

/// The passes of a chunked prefill of `prompt_tokens` tokens: each pass's
/// first prompt row and its real (unpadded) row count. Empty when the
/// prompt or the chunk is empty.
pub(crate) fn chunk_spans(prompt_tokens: u64, chunk: u64) -> Vec<(u64, u64)> {
    if chunk == 0 {
        return Vec::new();
    }
    (0..prompt_tokens.div_ceil(chunk))
        .map(|pass| {
            let start = pass * chunk;
            (start, chunk.min(prompt_tokens - start))
        })
        .collect()
}

/// The rows a chunked prefill of `prompt_tokens` tokens writes to each KV
/// store: every pass's rows, the last pass's padding included.
pub(crate) fn chunked_rows(prompt_tokens: u64, chunk: u64) -> u64 {
    prompt_tokens.div_ceil(chunk.max(1)) * chunk
}

/// Whether a prefill graph can continue a history its earlier passes wrote:
/// every `attention` node reads its K and V from KV-history edges, and
/// those edges feed nothing but attention. An attention reading K or V
/// computed from the pass's own rows only (an expanded latent attention,
/// whose history holds the latent rows) would not see the earlier passes.
pub(crate) fn prefill_continues_history(graph: &ExecutionGraph) -> bool {
    let is_kv_edge = |edge: &TensorEdgeId| {
        graph
            .edges
            .get(edge)
            .is_some_and(|edge| edge.kv_cache.is_some())
    };
    let attentions: Vec<&ExecutionNode> = graph
        .nodes
        .values()
        .filter(|node| node.operator.name() == "attention")
        .collect();
    !attentions.is_empty()
        && attentions.iter().all(|node| {
            node.inputs.len() >= 3 && is_kv_edge(&node.inputs[1]) && is_kv_edge(&node.inputs[2])
        })
        && super::engine::kv_edges_feed_only_attention(graph)
}

impl E2eRuntimeModelExecutionEngine {
    /// A chunked prefill: the passes of [`chunk_spans`], each running the
    /// `chunk`-row prefill graph over its prompt rows, right-padded, with
    /// the KV history of the passes before it and its first row's position.
    /// Every pass selects its last real row's logits; the last pass's are
    /// the prefill's. The history ends at the prompt's last row: the last
    /// pass's padding rows lie beyond it, and decode writes over them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_chunked_prefill(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        kv_state: &FirstNativeExecutionKvState,
        execution_plan: &mut PreparedExecutionPlan,
        graphs: &CachedGraphs,
        model_instance: &ModelInstanceId,
        chunk: u64,
        node_events: &mut Vec<PerNodeCausalEvent>,
        kv_stores: KvStoreSpec,
    ) -> Result<FirstNativeGraphExecutionOutput, InferenceApiError> {
        let prompt_tokens = request.input_token_ids.len() as u64;
        let ids_edge = TensorEdgeId::new("input.token_ids");
        let row_edge = TensorEdgeId::new(super::prefill_buckets::LOGITS_ROW_INPUT);
        let mut history: Option<LayerKvMap> = None;
        let mut output = None;
        for (start, rows) in chunk_spans(prompt_tokens, chunk) {
            let ids = &request.input_token_ids[start as usize..(start + rows) as usize];
            let ids_tensor = HostTensor::new(
                [chunk],
                super::prefill_buckets::pad_prompt_ids(ids, chunk as usize),
            )
            .map_err(runtime_generation_failed)?;
            let mut inputs = BTreeMap::from([(ids_edge.clone(), ids_tensor)]);
            if graphs.logits_row {
                inputs.insert(
                    row_edge.clone(),
                    HostTensor::new([1], vec![(rows - 1) as f32])
                        .map_err(runtime_generation_failed)?,
                );
            }
            let mut pass = execute_first_native_graph_namespaced(
                runtime,
                &self.fixture,
                model_instance,
                &kv_state.cache,
                &graphs.prefill,
                execution_plan,
                inputs,
                history.as_ref(),
                Some(start),
                node_events,
                if graphs.logits_row {
                    MaterializedEdges::LastRow(&["logits"])
                } else {
                    MaterializedEdges::Row(&["logits"], rows - 1)
                },
                Some(&graphs.prefill_memo),
                Some(kv_stores),
                None,
                true,
            )?;
            // The history ends at this pass's last real row.
            for layer in pass.2.values_mut() {
                if layer.rows.is_some() {
                    layer.rows = Some(start + rows);
                }
            }
            history = Some(pass.2.clone());
            output = Some(pass);
        }
        output.ok_or_else(|| InferenceApiError::GenerationFailed {
            reason: "a chunked prefill needs at least one prompt token".into(),
        })
    }
}

#[cfg(test)]
mod tests;
