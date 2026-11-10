//! Runtime-owned tensor-parallel plan (`enforce-local-multigpu-and-expert-
//! placement`, tensor parallelism): which Model Instance weight is split
//! across a Provider's Devices, and how, derived from the parallel
//! semantics the Model Component declared on its graph's nodes
//! ([`crate::operator::TENSOR_PARALLEL_ATTRIBUTE`]). The Runtime hands the
//! plan to the Provider before any weight is written
//! ([`crate::provider::ProviderExecutionApi::apply_tensor_parallel`]); the
//! Provider stores each Device's shard and runs every invocation on every
//! Device over its shard, reducing a `row` matmul's partial sums.
//!
//! Nothing here reads weight names for meaning: a weight is split only
//! because a node consuming it says so.

use std::collections::BTreeMap;

use crate::device::DeviceId;
use crate::execution_graph::ExecutionGraph;
use crate::operator::{OperatorAttributeValue, TENSOR_PARALLEL_ATTRIBUTE};

/// How a weight `[in, out]` (the matmul's `b`) is split across Devices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightShard {
    /// Each Device holds a contiguous range of output columns (a bias, of
    /// its matching range of values).
    Columns,
    /// Each Device holds a contiguous range of input rows.
    Rows,
}

/// Why a plan cannot be built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorParallelError {
    /// The same weight is declared split two different ways.
    Conflicting(String),
}

impl std::fmt::Display for TensorParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflicting(weight) => {
                write!(
                    f,
                    "weight '{weight}' is declared split both by columns and by rows"
                )
            }
        }
    }
}

impl std::error::Error for TensorParallelError {}

/// The Devices a Model Instance is split across, in shard order, and each
/// split weight's [`WeightShard`] by its canonical name. Every weight not
/// named is replicated on every Device.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TensorParallelPlan {
    devices: Vec<DeviceId>,
    shards: BTreeMap<String, WeightShard>,
}

/// The graph edge prefix of a weight (`graph-builder`'s `weight-edge`).
const WEIGHT_EDGE_PREFIX: &str = "weight.";

impl TensorParallelPlan {
    /// Splits, over `devices`, every weight a `column` or `row` matmul-family
    /// node of `graphs` multiplies by (input 1, and input 2 of a
    /// `gated-matmul`; a `column` `matmul-bias`'s bias splits with its
    /// columns).
    pub fn from_graphs<'g>(
        graphs: impl IntoIterator<Item = &'g ExecutionGraph>,
        devices: Vec<DeviceId>,
    ) -> Result<Self, TensorParallelError> {
        let mut shards = BTreeMap::new();
        for graph in graphs {
            for node in graph.nodes.values() {
                let shard = match node.attributes.get(TENSOR_PARALLEL_ATTRIBUTE) {
                    Some(OperatorAttributeValue::String(kind)) if kind == "column" => {
                        WeightShard::Columns
                    }
                    Some(OperatorAttributeValue::String(kind)) if kind == "row" => {
                        WeightShard::Rows
                    }
                    _ => continue,
                };
                let weight_inputs: &[usize] = match (node.operator.name(), shard) {
                    ("matmul" | "matmul-residual", _) => &[1],
                    ("matmul-bias", WeightShard::Columns) => &[1, 2],
                    ("matmul-bias", WeightShard::Rows) => &[1],
                    ("gated-matmul", _) => &[1, 2],
                    _ => continue,
                };
                for &index in weight_inputs {
                    let Some(name) = node
                        .inputs
                        .get(index)
                        .and_then(|edge| edge.as_str().strip_prefix(WEIGHT_EDGE_PREFIX))
                    else {
                        continue;
                    };
                    match shards.insert(name.to_string(), shard) {
                        Some(previous) if previous != shard => {
                            return Err(TensorParallelError::Conflicting(name.to_string()));
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(Self { devices, shards })
    }

    /// A plan splitting each named weight (by canonical name) as given.
    pub fn new(
        devices: Vec<DeviceId>,
        shards: impl IntoIterator<Item = (String, WeightShard)>,
    ) -> Self {
        Self {
            devices,
            shards: shards.into_iter().collect(),
        }
    }

    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Whether anything is split (a plan over one Device, or with no split
    /// weight, changes nothing).
    pub fn splits(&self) -> bool {
        self.devices.len() > 1 && !self.shards.is_empty()
    }

    /// How the weight `name` -- a canonical weight name, or a resource id
    /// ending in `.weight.{canonical name}` -- is split, `None` when it is
    /// replicated.
    pub fn shard_for_weight(&self, name: &str) -> Option<WeightShard> {
        if let Some(shard) = self.shards.get(name) {
            return Some(*shard);
        }
        let canonical = name
            .rsplit_once(".weight.")
            .map(|(_, canonical)| canonical)?;
        self.shards.get(canonical).copied()
    }

    /// The `[start, end)` share of `extent` that Device `index` of
    /// `devices` holds: equal contiguous parts, each a multiple of
    /// `granule` (a quantization block, or 1), the last taking the rest.
    pub fn share(&self, index: usize, extent: u64, granule: u64) -> (u64, u64) {
        let parts = self.devices.len().max(1) as u64;
        let granules = extent / granule.max(1);
        let per = granules / parts;
        let start = index as u64 * per * granule;
        let end = if index as u64 + 1 == parts {
            extent
        } else {
            start + per * granule
        };
        (start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_split_into_equal_granule_aligned_parts() {
        let plan = TensorParallelPlan {
            devices: vec![DeviceId::new("a"), DeviceId::new("b")],
            shards: BTreeMap::new(),
        };
        assert_eq!(plan.share(0, 4864, 1), (0, 2432));
        assert_eq!(plan.share(1, 4864, 1), (2432, 4864));
        assert_eq!(plan.share(0, 1408, 32), (0, 704));
        assert_eq!(plan.share(1, 96, 32), (32, 96));
    }

    #[test]
    fn names_each_split_weight_by_its_canonical_name() {
        let plan = TensorParallelPlan {
            devices: vec![DeviceId::new("a"), DeviceId::new("b")],
            shards: BTreeMap::from([
                ("layers.0.mlp.gate_proj".to_string(), WeightShard::Columns),
                ("layers.0.mlp.down_proj".to_string(), WeightShard::Rows),
            ]),
        };
        assert!(plan.splits());
        assert_eq!(
            plan.shard_for_weight("model.i.weight.layers.0.mlp.down_proj"),
            Some(WeightShard::Rows)
        );
        assert_eq!(
            plan.shard_for_weight("layers.0.mlp.gate_proj"),
            Some(WeightShard::Columns)
        );
        assert_eq!(plan.shard_for_weight("model.i.weight.lm_head"), None);
    }
}
