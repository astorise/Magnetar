//! Semantics-preserving fusion passes over Component-built execution graphs
//! (`cache-and-optimize-prepared-graphs` C03).
//!
//! The first pass folds a row-broadcast bias addition into the matmul that
//! produces its input, so a decode step dispatches one Kernel instead of
//! two for every biased projection (Qwen2's Q, K and V projections: three
//! per layer). The fused `matmul-bias` Operator computes exactly the
//! matmul and then adds the bias row to every output row, the same two
//! operations in the same order, so a Provider implementing it as such
//! produces bit-identical results.
//!
//! A pass only runs when the executing Provider advertises the fused
//! Operator; otherwise the graph is left untouched.

use std::collections::BTreeMap;

use crate::execution_graph::{ExecutionGraph, ExecutionNodeId, TensorEdgeId};
use crate::operator::{OperatorFamily, OperatorId};

/// The fused Operator: `matmul(a, b) + bias`, `bias` one row broadcast to
/// every output row. Takes `matmul`'s attributes.
pub const MATMUL_BIAS_OPERATOR: &str = "matmul-bias";

/// The `matmul-bias` Operator id.
pub fn matmul_bias_operator() -> OperatorId {
    OperatorId::magnetar(MATMUL_BIAS_OPERATOR, 1, OperatorFamily::LinearAlgebra)
}

/// Folds every `add(matmul(a, b), bias)` whose intermediate has no other
/// consumer and no cache semantics, and whose `bias` is a rank-1 weight
/// matching the matmul output's last dimension, into one `matmul-bias`
/// node. The fused node keeps the matmul's id and attributes and the add's
/// output edge, so every downstream node and resource identity is
/// unchanged. Returns the rewritten graph and how many pairs were fused.
pub fn fuse_matmul_bias(
    graph: &ExecutionGraph,
    is_weight: impl Fn(&TensorEdgeId) -> bool,
) -> (ExecutionGraph, usize) {
    let mut consumers: BTreeMap<&TensorEdgeId, Vec<&ExecutionNodeId>> = BTreeMap::new();
    let mut produced: BTreeMap<&TensorEdgeId, &ExecutionNodeId> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        for input in &node.inputs {
            consumers.entry(input).or_default().push(node_id);
        }
        for output in &node.outputs {
            produced.insert(output, node_id);
        }
    }

    let mut fusions = Vec::new();
    for (add_id, add) in &graph.nodes {
        if add.operator.name() != "add" || add.inputs.len() != 2 || add.outputs.len() != 1 {
            continue;
        }
        let (intermediate, bias) = (&add.inputs[0], &add.inputs[1]);
        let Some(matmul_id) = produced.get(intermediate).copied() else {
            continue;
        };
        let matmul = &graph.nodes[matmul_id];
        if matmul.operator.name() != "matmul"
            || matmul.inputs.len() != 2
            || matmul.outputs.len() != 1
            || consumers.get(intermediate).map(Vec::len) != Some(1)
            || produced.contains_key(bias)
            || !is_weight(bias)
        {
            continue;
        }
        let (Some(intermediate_edge), Some(bias_edge)) =
            (graph.edges.get(intermediate), graph.edges.get(bias))
        else {
            continue;
        };
        if intermediate_edge.kv_cache.is_some() || intermediate_edge.prefix_cache.is_some() {
            continue;
        }
        let bias_dimensions = &bias_edge.descriptor.shape.dimensions;
        let output_columns = intermediate_edge.descriptor.shape.dimensions.last();
        if bias_dimensions.len() != 1 || Some(&bias_dimensions[0]) != output_columns {
            continue;
        }
        fusions.push((matmul_id.clone(), add_id.clone(), intermediate.clone()));
    }

    let mut fused = graph.clone();
    for (matmul_id, add_id, intermediate) in &fusions {
        let add = fused
            .nodes
            .remove(add_id)
            .expect("each fused add is a graph node");
        let bias = add.inputs[1].clone();
        let output = add.outputs[0].clone();
        let matmul = fused
            .nodes
            .get_mut(matmul_id)
            .expect("each fused matmul is a graph node");
        matmul.operator = matmul_bias_operator();
        matmul.inputs.push(bias.clone());
        matmul.outputs = vec![output.clone()];
        fused.edges.remove(intermediate);
        if let Some(edge) = fused.edges.get_mut(&bias)
            && edge.consumers.remove(add_id)
        {
            edge.consumers.insert(matmul_id.clone());
        }
        if let Some(edge) = fused.edges.get_mut(&output)
            && edge.producer.as_ref() == Some(add_id)
        {
            edge.producer = Some(matmul_id.clone());
        }
    }
    (fused, fusions.len())
}

#[cfg(test)]
#[path = "graph_fusion_tests.rs"]
mod tests;
