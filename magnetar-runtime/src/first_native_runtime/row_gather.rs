//! The rows a step reads from tables kept by row (`support-qwen38-flash-next`,
//! task 4.4, design D8).
//!
//! A Model Instance's row tables (`ModelInstance::row_table`) are never
//! staged on a Provider: Qwen4-Exp's n-gram embedding tables hold hundreds of
//! millions of rows and a step reads a few thousand of them. Before a step's
//! nodes run, the Runtime therefore
//! 1. computes, for every `ngram-embedding` node that reads one, the rows its
//!    tokens name ([`NgramEmbedding::row_ids`], the operator's own hash, from
//!    the step's token ids and the layer's context),
//! 2. hints all of those rows to their mappings at once, so the operating
//!    system queues the reads of every layer together while the first layers
//!    compute, and
//! 3. when the node comes up, copies its rows into the gathered input
//!    `[tokens * heads, head_dim]` and dispatches the node in its gathered
//!    form (`gathered_rows`), which returns exactly what the dense form
//!    returns from the whole table.

use super::*;
use crate::operator_semantics::ngram_embedding_from_attributes;
use crate::{RowTable, RowTableHandle};
use magnetar_provider_api::NgramEmbedding;

/// The attribute that makes an `ngram-embedding` node take gathered rows.
const GATHERED_ROWS: &str = "gathered_rows";

/// What one `ngram-embedding` node reads this step.
#[derive(Debug)]
struct NodeRows {
    table: Arc<RowTable>,
    /// The table rows, `[tokens][heads]` flattened: the gathered input's
    /// rows in order.
    rows: Vec<u64>,
    head_dim: u64,
}

/// The rows of every row-table read of one step, by node.
#[derive(Debug)]
pub(super) struct RowGather {
    nodes: BTreeMap<ExecutionNodeId, NodeRows>,
}

/// A step's tokens and the tokens before them that its n-grams reach back to.
#[derive(Debug)]
pub(super) struct StepTokens {
    /// The step's token ids, in order.
    pub(super) tokens: Vec<i64>,
    /// The `ngram_size - 1` tokens before the first one (the end token where
    /// the sequence starts).
    pub(super) context: Vec<i64>,
}

impl RowGather {
    /// The table a node reads by row, when its table input is one of the
    /// instance's row tables: the weight edge named `weight.{table}`.
    fn table_of<'a>(
        node: &'a ExecutionNode,
        row_tables: &'a BTreeMap<String, RowTableHandle>,
    ) -> Option<(&'a Arc<RowTable>, &'a TensorEdgeId)> {
        if node.operator.name() != "ngram-embedding" {
            return None;
        }
        let table_edge = node.inputs.get(2)?;
        let name = weight_tensor_name_from_edge(table_edge.as_str())?;
        let handle = row_tables.get(&name)?;
        Some((&handle.0, table_edge))
    }

    /// The nodes of `graph` that read a row table, and their rows for the
    /// step `step` describes (asked for once, with the first such node, only
    /// if there is one). `None` when no node reads a row table: the step
    /// is the dense one.
    pub(super) fn plan(
        graph: &ExecutionGraph,
        row_tables: &BTreeMap<String, RowTableHandle>,
        step: impl FnOnce(&ExecutionNode, &NgramEmbedding) -> Result<StepTokens, InferenceApiError>,
    ) -> Result<Option<Self>, InferenceApiError> {
        if row_tables.is_empty() {
            return Ok(None);
        }
        let mut tokens: Option<StepTokens> = None;
        let mut step = Some(step);
        let mut nodes = BTreeMap::new();
        for node in graph.nodes.values() {
            let Some((table, table_edge)) = Self::table_of(node, row_tables) else {
                continue;
            };
            let hash = ngram_embedding_from_attributes(&node.attributes).map_err(|error| {
                InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{}': {error}", node.id),
                }
            })?;
            let head_dim = graph
                .edges
                .get(table_edge)
                .and_then(|edge| edge.descriptor.shape.dimensions.get(1).copied())
                .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                    reason: format!(
                        "graph node '{}': the n-gram table '{table_edge}' is not a matrix",
                        node.id
                    ),
                })?;
            if tokens.is_none() {
                let ask = step.take().expect("the step is described once");
                tokens = Some(ask(node, &hash)?);
            }
            let step_tokens = tokens.as_ref().expect("described just above");
            if step_tokens.context.len() != hash.context_len() {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!(
                        "graph node '{}' reads {} tokens of context, the step has {}",
                        node.id,
                        hash.context_len(),
                        step_tokens.context.len()
                    ),
                });
            }
            let rows = hash
                .row_ids(&step_tokens.tokens, &step_tokens.context)
                .into_iter()
                .flatten()
                .collect();
            nodes.insert(
                node.id.clone(),
                NodeRows {
                    table: Arc::clone(table),
                    rows,
                    head_dim,
                },
            );
        }
        Ok((!nodes.is_empty()).then_some(Self { nodes }))
    }

    /// Hints every node's rows to its table, all of them before any node
    /// runs: later layers' pages are read while the first layers compute.
    pub(super) fn prefetch(&self) {
        for node in self.nodes.values() {
            node.table.prefetch(&node.rows);
        }
    }

    /// Whether `node` takes gathered rows this step.
    pub(super) fn reads(&self, node: &ExecutionNodeId) -> bool {
        self.nodes.contains_key(node)
    }

    /// `node`'s rows as the gathered input `[tokens * heads, head_dim]`.
    pub(super) fn gather(&self, node: &ExecutionNodeId) -> Result<HostTensor, InferenceApiError> {
        let rows = self
            .nodes
            .get(node)
            .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: format!("graph node '{node}' reads no row table this step"),
            })?;
        let values = rows
            .table
            .read_rows(&rows.rows)
            .map_err(runtime_generation_failed)?;
        HostTensor::new([rows.rows.len() as u64, rows.head_dim], values)
            .map_err(runtime_generation_failed)
    }
}

/// The step's tokens and context, read from its graph inputs: the token ids
/// are a host input, and the context is the first reading node's recurrent
/// state, the same one every layer's holds (each layer's context is the
/// tokens before the step). A state a Provider holds is downloaded: a few
/// values, once per step.
pub(super) fn step_tokens(
    node: &ExecutionNode,
    hash: &NgramEmbedding,
    initial: &BTreeMap<TensorEdgeId, HostTensor>,
    resident: &BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
    provider: &Arc<dyn ProviderExecutionApi>,
) -> Result<StepTokens, InferenceApiError> {
    let ids = initial
        .get(&TensorEdgeId::new("input.token_ids"))
        .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
            reason: format!(
                "graph node '{}' reads a row table, which needs the step's token ids on the host",
                node.id
            ),
        })?;
    let context_edge =
        node.inputs
            .get(1)
            .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: format!("graph node '{}' has no context input", node.id),
            })?;
    let state = match (initial.get(context_edge), resident.get(context_edge)) {
        (Some(state), _) => state.clone(),
        (None, Some((resource, _))) => provider.read_tensor(resource).ok_or_else(|| {
            InferenceApiError::GraphPlanningFailed {
                reason: format!("the n-gram context '{resource}' cannot be read"),
            }
        })?,
        (None, None) => {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "graph node '{}' has no value for its context '{context_edge}'",
                    node.id
                ),
            });
        }
    };
    Ok(StepTokens {
        tokens: token_ids(ids)?,
        context: context_ids(&state, hash)?,
    })
}

/// `node` in its gathered form: the same node taking the gathered rows as its
/// table input.
pub(super) fn gathered_form(node: &ExecutionNode) -> ExecutionNode {
    let mut node = node.clone();
    node.attributes.insert(
        GATHERED_ROWS.to_string(),
        OperatorAttributeValue::Boolean(true),
    );
    node
}

/// The token ids of a step's `input.token_ids` tensor.
pub(super) fn token_ids(tensor: &HostTensor) -> Result<Vec<i64>, InferenceApiError> {
    tensor
        .data
        .iter()
        .map(|&value| {
            if value.fract() == 0.0 && (0.0..16_777_216.0).contains(&value) {
                Ok(value as i64)
            } else {
                Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("input.token_ids holds {value}, not a token id"),
                })
            }
        })
        .collect()
}

/// A layer's context state (each token held as its id minus `eos_token_id`)
/// as token ids.
fn context_ids(state: &HostTensor, hash: &NgramEmbedding) -> Result<Vec<i64>, InferenceApiError> {
    state
        .data
        .iter()
        .map(|&value| {
            let id = hash.eos_token_id as f64 + f64::from(value);
            if value.fract() == 0.0 && (0.0..16_777_216.0).contains(&id) {
                Ok(id as i64)
            } else {
                Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("an n-gram context holds {value}, not a token id"),
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
