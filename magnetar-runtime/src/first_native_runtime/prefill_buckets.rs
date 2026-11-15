//! Bucketed prefill (`replay-bucketed-prefill`): a short prompt runs the
//! prefill graph of a fixed bucket length, right-padded, so prompts of
//! different lengths share one graph (and, later, one recording).

use super::*;

/// The prompt buckets used when `MAGNETAR_PREFILL_BUCKETS` is unset.
pub(crate) const DEFAULT_PREFILL_BUCKETS: [u64; 4] = [16, 32, 64, 128];

/// The Runtime-provided graph input naming the prompt row whose logits are
/// sampled: `[n - 1]` for a prompt of `n` tokens.
pub(crate) const LOGITS_ROW_INPUT: &str = "input.logits_row";

/// The node and edge [`insert_logits_row_gather`] adds.
const LOGITS_ROW_NODE: &str = "logits_row.select";
const LOGITS_ROW_HIDDEN: &str = "logits_row.hidden";

/// The prompt buckets this process uses: `MAGNETAR_PREFILL_BUCKETS` (a
/// comma-separated list of positive lengths), else the defaults. Empty
/// (bucketing off) with `MAGNETAR_PREFILL_REPLAY=0`.
pub(crate) fn prefill_buckets_from_env() -> Vec<u64> {
    if std::env::var("MAGNETAR_PREFILL_REPLAY").as_deref() == Ok("0") {
        return Vec::new();
    }
    let configured = std::env::var("MAGNETAR_PREFILL_BUCKETS").ok();
    parse_prefill_buckets(configured.as_deref())
}

/// Parses a bucket list; a missing or malformed list gives the defaults.
pub(crate) fn parse_prefill_buckets(configured: Option<&str>) -> Vec<u64> {
    let parsed = configured.and_then(|list| {
        list.split(',')
            .map(|length| {
                length
                    .trim()
                    .parse::<u64>()
                    .ok()
                    .filter(|length| *length > 0)
            })
            .collect::<Option<Vec<_>>>()
    });
    let mut buckets = parsed
        .filter(|buckets| !buckets.is_empty())
        .unwrap_or_else(|| DEFAULT_PREFILL_BUCKETS.to_vec());
    buckets.sort_unstable();
    buckets.dedup();
    buckets
}

/// `buckets` sorted, without duplicates, zero, or lengths above the
/// model's `context_length` (when known, nonzero): a padded prompt must
/// fit the positions the model was trained on.
pub(crate) fn buckets_within_context(buckets: Vec<u64>, context_length: u64) -> Vec<u64> {
    let mut buckets: Vec<u64> = buckets
        .into_iter()
        .filter(|length| *length > 0 && (context_length == 0 || *length <= context_length))
        .collect();
    buckets.sort_unstable();
    buckets.dedup();
    buckets
}

/// The smallest bucket holding `prompt_tokens`, or `None` above the
/// largest bucket (and for an empty prompt).
pub(crate) fn prefill_bucket(buckets: &[u64], prompt_tokens: u64) -> Option<u64> {
    if prompt_tokens == 0 {
        return None;
    }
    buckets
        .iter()
        .copied()
        .find(|bucket| *bucket >= prompt_tokens)
}

/// `ids` right-padded to `length` by repeating the last id. Under causal
/// attention no real row reads a padding row.
pub(crate) fn pad_prompt_ids(ids: &[TokenId], length: usize) -> Vec<f32> {
    let last = ids.last().copied().unwrap_or_default();
    ids.iter()
        .copied()
        .chain(std::iter::repeat(last))
        .take(length.max(ids.len()))
        .map(|id| id as f32)
        .collect()
}

/// `graph` with an `embedding` node gathering one row of the LM head's
/// input, at the index given by [`LOGITS_ROW_INPUT`]: the LM head then
/// reads that row only and `"logits"` has one row. `None` when the graph's
/// `"logits"` is not produced by a plain or bias matmul over a rank-2
/// input, or has no `embedding` node to take the Operator from.
pub(crate) fn insert_logits_row_gather(graph: &ExecutionGraph) -> Option<ExecutionGraph> {
    let logits_edge_id = TensorEdgeId::new("logits");
    let row_input_id = TensorEdgeId::new(LOGITS_ROW_INPUT);
    let hidden_row_id = TensorEdgeId::new(LOGITS_ROW_HIDDEN);
    let node_id = ExecutionNodeId::new(LOGITS_ROW_NODE);
    if graph.edges.contains_key(&row_input_id)
        || graph.edges.contains_key(&hidden_row_id)
        || graph.nodes.contains_key(&node_id)
    {
        return None;
    }
    let logits_edge = graph.edges.get(&logits_edge_id)?;
    let head_id = logits_edge.producer.clone()?;
    let head = graph.nodes.get(&head_id)?;
    if !matches!(head.operator.name(), "matmul" | "matmul-bias")
        || head.outputs.as_slice() != [logits_edge_id.clone()]
        || head.inputs.len() < 2
    {
        return None;
    }
    let hidden_id = head.inputs[0].clone();
    let hidden = graph.edges.get(&hidden_id)?;
    let [_, width] = hidden.descriptor.shape.dimensions.as_slice() else {
        return None;
    };
    let width = *width;
    let [_, vocab] = logits_edge.descriptor.shape.dimensions.as_slice() else {
        return None;
    };
    let vocab = *vocab;
    let embedding = graph
        .nodes
        .values()
        .find(|node| node.operator.name() == "embedding")?
        .operator
        .clone();
    let ids = graph.edges.get(&TensorEdgeId::new("input.token_ids"))?;

    let mut rewritten = graph.clone();
    let mut row_input = ids.clone();
    row_input.id = row_input_id.clone();
    row_input.logical_tensor_id = LOGITS_ROW_INPUT.into();
    row_input.descriptor.shape = ShapeDescriptor::new([1]);
    row_input.producer = None;
    row_input.consumers = BTreeSet::from([node_id.clone()]);
    let mut hidden_row = hidden.clone();
    hidden_row.id = hidden_row_id.clone();
    hidden_row.logical_tensor_id = LOGITS_ROW_HIDDEN.into();
    hidden_row.descriptor.shape = ShapeDescriptor::new([1, width]);
    hidden_row.producer = Some(node_id.clone());
    hidden_row.consumers = BTreeSet::from([head_id.clone()]);
    hidden_row.kv_cache = None;
    hidden_row.prefix_cache = None;
    let mut select = ExecutionNode::new(node_id.clone(), embedding)
        .with_input(row_input_id.clone())
        .with_input(hidden_id.clone())
        .with_output(hidden_row_id.clone());
    select.resource_affinity = head.resource_affinity.clone();

    let edge = rewritten.edges.get_mut(&hidden_id)?;
    edge.consumers.remove(&head_id);
    edge.consumers.insert(node_id.clone());
    rewritten.edges.get_mut(&logits_edge_id)?.descriptor.shape = ShapeDescriptor::new([1, vocab]);
    rewritten.nodes.get_mut(&head_id)?.inputs[0] = hidden_row_id;
    rewritten.edges.insert(row_input_id, row_input);
    rewritten.edges.insert(hidden_row.id.clone(), hidden_row);
    rewritten.nodes.insert(node_id, select);
    Some(rewritten)
}

#[cfg(test)]
mod tests;
