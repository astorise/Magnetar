//! Semantics-preserving fusion passes over Component-built execution graphs
//! (`cache-and-optimize-prepared-graphs` C03).
//!
//! Each pass folds a node whose single output feeds exactly one consumer
//! into that consumer, so a decode step dispatches one Kernel instead of
//! two. The fused node keeps the producer's id and attributes, takes the
//! producer's inputs followed by the consumer's other input, and writes
//! the consumer's output edge, so every downstream node and resource
//! identity is unchanged.
//!
//! - `matmul-bias`: `add(matmul(a, b), bias)`, `bias` a rank-1 weight (the
//!   biased Q/K/V projections of Qwen2: three per layer).
//! - `matmul-residual`: `residual-add(residual, matmul(a, b))` (the output
//!   and down projections feeding each layer's two residual additions).
//! - `silu-mul`: `mul(silu(gate), up)` (each layer's SwiGLU gate).
//!
//! Every fused Operator computes exactly the two operations it replaces,
//! in the same order (floating-point addition and multiplication of two
//! operands being commutative), so a Provider implementing it as such
//! produces bit-identical results. A pass only runs when the executing
//! Provider advertises its fused Operator; otherwise the graph is left
//! untouched.

use std::collections::BTreeMap;

use crate::execution_graph::{ExecutionGraph, ExecutionNodeId, TensorEdgeId};
use crate::operator::{OperatorFamily, OperatorId};

/// `matmul(a, b) + bias`, `bias` one row broadcast to every output row.
/// Takes `matmul`'s attributes.
pub const MATMUL_BIAS_OPERATOR: &str = "matmul-bias";

/// `residual + matmul(a, b)`, `residual` the matmul output's shape. Takes
/// `matmul`'s attributes.
pub const MATMUL_RESIDUAL_OPERATOR: &str = "matmul-residual";

/// `silu(gate) * up`, elementwise.
pub const SILU_MUL_OPERATOR: &str = "silu-mul";

/// The `matmul-bias` Operator id.
pub fn matmul_bias_operator() -> OperatorId {
    OperatorId::magnetar(MATMUL_BIAS_OPERATOR, 1, OperatorFamily::LinearAlgebra)
}

/// The `matmul-residual` Operator id.
pub fn matmul_residual_operator() -> OperatorId {
    OperatorId::magnetar(MATMUL_RESIDUAL_OPERATOR, 1, OperatorFamily::LinearAlgebra)
}

/// The `silu-mul` Operator id.
pub fn silu_mul_operator() -> OperatorId {
    OperatorId::magnetar(SILU_MUL_OPERATOR, 1, OperatorFamily::Activation)
}

/// One fusion found by a pass.
struct Fusion {
    producer: ExecutionNodeId,
    consumer: ExecutionNodeId,
    intermediate: TensorEdgeId,
    /// The consumer's input that is not the intermediate.
    other_input: TensorEdgeId,
}

/// Which node produces each edge, and which nodes read it.
struct EdgeIndex<'g> {
    produced: BTreeMap<&'g TensorEdgeId, &'g ExecutionNodeId>,
    consumers: BTreeMap<&'g TensorEdgeId, usize>,
}

impl<'g> EdgeIndex<'g> {
    fn new(graph: &'g ExecutionGraph) -> Self {
        let mut produced = BTreeMap::new();
        let mut consumers = BTreeMap::new();
        for (node_id, node) in &graph.nodes {
            for input in &node.inputs {
                *consumers.entry(input).or_insert(0) += 1;
            }
            for output in &node.outputs {
                produced.insert(output, node_id);
            }
        }
        Self {
            produced,
            consumers,
        }
    }

    /// The node producing `intermediate`, when that node has the operator
    /// `producer_operator`, `arity` inputs and one output, and
    /// `intermediate` has exactly one consumer and no cache semantics.
    fn sole_producer(
        &self,
        graph: &'g ExecutionGraph,
        intermediate: &TensorEdgeId,
        producer_operator: &str,
        arity: usize,
    ) -> Option<&'g ExecutionNodeId> {
        let producer_id = *self.produced.get(intermediate)?;
        let producer = &graph.nodes[producer_id];
        let edge = graph.edges.get(intermediate)?;
        (producer.operator.name() == producer_operator
            && producer.inputs.len() == arity
            && producer.outputs.len() == 1
            && self.consumers.get(intermediate) == Some(&1)
            && edge.kv_cache.is_none()
            && edge.prefix_cache.is_none())
        .then_some(producer_id)
    }
}

/// Rewrites every fusion: the producer becomes `fused_operator`, appends the
/// consumer's other input and writes the consumer's output; the consumer
/// and the intermediate edge disappear.
fn apply(graph: &ExecutionGraph, fusions: &[Fusion], fused_operator: OperatorId) -> ExecutionGraph {
    let mut fused = graph.clone();
    for fusion in fusions {
        let consumer = fused
            .nodes
            .remove(&fusion.consumer)
            .expect("each fused consumer is a graph node");
        let output = consumer.outputs[0].clone();
        let producer = fused
            .nodes
            .get_mut(&fusion.producer)
            .expect("each fused producer is a graph node");
        producer.operator = fused_operator.clone();
        producer.inputs.push(fusion.other_input.clone());
        producer.outputs = vec![output.clone()];
        fused.edges.remove(&fusion.intermediate);
        if let Some(edge) = fused.edges.get_mut(&fusion.other_input)
            && edge.consumers.remove(&fusion.consumer)
        {
            edge.consumers.insert(fusion.producer.clone());
        }
        if let Some(edge) = fused.edges.get_mut(&output)
            && edge.producer.as_ref() == Some(&fusion.consumer)
        {
            edge.producer = Some(fusion.producer.clone());
        }
    }
    fused
}

/// Folds every `add(matmul(a, b), bias)` whose `bias` is a rank-1 weight
/// matching the matmul output's last dimension into one `matmul-bias`
/// node. Returns the rewritten graph and how many pairs were fused.
pub fn fuse_matmul_bias(
    graph: &ExecutionGraph,
    is_weight: impl Fn(&TensorEdgeId) -> bool,
) -> (ExecutionGraph, usize) {
    let index = EdgeIndex::new(graph);
    let mut fusions = Vec::new();
    for (add_id, add) in &graph.nodes {
        if add.operator.name() != "add" || add.inputs.len() != 2 || add.outputs.len() != 1 {
            continue;
        }
        let (intermediate, bias) = (&add.inputs[0], &add.inputs[1]);
        let Some(matmul_id) = index.sole_producer(graph, intermediate, "matmul", 2) else {
            continue;
        };
        if index.produced.contains_key(bias) || !is_weight(bias) {
            continue;
        }
        let (Some(intermediate_edge), Some(bias_edge)) =
            (graph.edges.get(intermediate), graph.edges.get(bias))
        else {
            continue;
        };
        let bias_dimensions = &bias_edge.descriptor.shape.dimensions;
        let output_columns = intermediate_edge.descriptor.shape.dimensions.last();
        if bias_dimensions.len() != 1 || Some(&bias_dimensions[0]) != output_columns {
            continue;
        }
        fusions.push(Fusion {
            producer: matmul_id.clone(),
            consumer: add_id.clone(),
            intermediate: intermediate.clone(),
            other_input: bias.clone(),
        });
    }
    (
        apply(graph, &fusions, matmul_bias_operator()),
        fusions.len(),
    )
}

/// Folds every `residual-add(residual, matmul(a, b))` (either input order)
/// whose residual has the matmul output's shape into one `matmul-residual`
/// node with inputs `[a, b, residual]`.
pub fn fuse_matmul_residual(graph: &ExecutionGraph) -> (ExecutionGraph, usize) {
    let index = EdgeIndex::new(graph);
    let mut fusions = Vec::new();
    for (add_id, add) in &graph.nodes {
        if add.operator.name() != "residual-add" || add.inputs.len() != 2 || add.outputs.len() != 1
        {
            continue;
        }
        // Prefer the second input as the matmul, the order Qwen2 builds.
        let candidate = [(1, 0), (0, 1)]
            .into_iter()
            .find_map(|(product, residual)| {
                index
                    .sole_producer(graph, &add.inputs[product], "matmul", 2)
                    .map(|matmul_id| (matmul_id, &add.inputs[product], &add.inputs[residual]))
            });
        let Some((matmul_id, intermediate, residual)) = candidate else {
            continue;
        };
        if intermediate == residual {
            continue;
        }
        let (Some(intermediate_edge), Some(residual_edge)) =
            (graph.edges.get(intermediate), graph.edges.get(residual))
        else {
            continue;
        };
        if intermediate_edge.descriptor.shape.dimensions
            != residual_edge.descriptor.shape.dimensions
        {
            continue;
        }
        fusions.push(Fusion {
            producer: matmul_id.clone(),
            consumer: add_id.clone(),
            intermediate: intermediate.clone(),
            other_input: residual.clone(),
        });
    }
    (
        apply(graph, &fusions, matmul_residual_operator()),
        fusions.len(),
    )
}

/// Folds every `mul(silu(gate), up)` (either input order) whose `up` has
/// the activation's shape into one `silu-mul` node with inputs
/// `[gate, up]`.
pub fn fuse_silu_mul(graph: &ExecutionGraph) -> (ExecutionGraph, usize) {
    let index = EdgeIndex::new(graph);
    let mut fusions = Vec::new();
    for (mul_id, mul) in &graph.nodes {
        if mul.operator.name() != "mul" || mul.inputs.len() != 2 || mul.outputs.len() != 1 {
            continue;
        }
        let candidate = [(0, 1), (1, 0)].into_iter().find_map(|(activated, up)| {
            index
                .sole_producer(graph, &mul.inputs[activated], "silu", 1)
                .map(|silu_id| (silu_id, &mul.inputs[activated], &mul.inputs[up]))
        });
        let Some((silu_id, intermediate, up)) = candidate else {
            continue;
        };
        if intermediate == up {
            continue;
        }
        let (Some(intermediate_edge), Some(up_edge)) =
            (graph.edges.get(intermediate), graph.edges.get(up))
        else {
            continue;
        };
        if intermediate_edge.descriptor.shape.dimensions != up_edge.descriptor.shape.dimensions {
            continue;
        }
        fusions.push(Fusion {
            producer: silu_id.clone(),
            consumer: mul_id.clone(),
            intermediate: intermediate.clone(),
            other_input: up.clone(),
        });
    }
    (apply(graph, &fusions, silu_mul_operator()), fusions.len())
}

#[cfg(test)]
#[path = "graph_fusion_tests.rs"]
mod tests;
