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
//! - `gated-matmul`: `silu-mul(matmul(a, gate), matmul(a, up))`, after
//!   the `silu-mul` pass: each layer's whole SwiGLU projection, both
//!   weights read by one Kernel. It folds three nodes, not two.
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

/// `silu(matmul(a, gate)) * matmul(a, up)`, inputs `[a, gate, up]`.
pub const GATED_MATMUL_OPERATOR: &str = "gated-matmul";

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

/// The `gated-matmul` Operator id.
pub fn gated_matmul_operator() -> OperatorId {
    OperatorId::magnetar(GATED_MATMUL_OPERATOR, 1, OperatorFamily::LinearAlgebra)
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

/// Whether `node` is a matmul its Component declared row-parallel: when
/// the model is split across Devices it produces partial sums, so an addend (bias, residual) must be
/// added once after they are reduced, never fused into each partial.
fn is_row_parallel(graph: &ExecutionGraph, node: &crate::execution_graph::ExecutionNodeId) -> bool {
    graph.nodes.get(node).is_some_and(|node| {
        matches!(
            node.attributes.get(crate::operator::TENSOR_PARALLEL_ATTRIBUTE),
            Some(crate::operator::OperatorAttributeValue::String(kind)) if kind == "row"
        )
    })
}

/// Folds every `add(matmul(a, b), bias)` whose `bias` is a rank-1 weight
/// matching the matmul output's last dimension into one `matmul-bias`
/// node. Returns the rewritten graph and how many pairs were fused.
pub fn fuse_matmul_bias(
    graph: &ExecutionGraph,
    is_weight: impl Fn(&TensorEdgeId) -> bool,
) -> (ExecutionGraph, usize) {
    fuse_matmul_bias_with(graph, is_weight, false)
}

/// [`fuse_matmul_bias`] for a model split across Devices by tensor
/// parallelism: row-parallel matmuls keep their bias add unfused.
pub fn fuse_matmul_bias_tensor_parallel(
    graph: &ExecutionGraph,
    is_weight: impl Fn(&TensorEdgeId) -> bool,
) -> (ExecutionGraph, usize) {
    fuse_matmul_bias_with(graph, is_weight, true)
}

fn fuse_matmul_bias_with(
    graph: &ExecutionGraph,
    is_weight: impl Fn(&TensorEdgeId) -> bool,
    tensor_parallel: bool,
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
        if tensor_parallel && is_row_parallel(graph, matmul_id) {
            continue;
        }
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
    fuse_matmul_residual_with(graph, false)
}

/// [`fuse_matmul_residual`] for a model split across Devices by tensor
/// parallelism: row-parallel matmuls keep their residual add unfused.
pub fn fuse_matmul_residual_tensor_parallel(graph: &ExecutionGraph) -> (ExecutionGraph, usize) {
    fuse_matmul_residual_with(graph, true)
}

fn fuse_matmul_residual_with(
    graph: &ExecutionGraph,
    tensor_parallel: bool,
) -> (ExecutionGraph, usize) {
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
        if tensor_parallel && is_row_parallel(graph, matmul_id) {
            continue;
        }
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

/// Folds every `silu-mul(matmul(a, gate), matmul(a, up))` whose two
/// matmuls read the same `a`, have one consumer each and produce the same
/// shape into one `gated-matmul` node with inputs `[a, gate, up]`. The
/// fused node keeps the gate matmul's id and writes the `silu-mul`'s
/// output. Returns the rewritten graph and how many nodes it removed (two
/// per fusion).
pub fn fuse_gated_matmul(graph: &ExecutionGraph) -> (ExecutionGraph, usize) {
    let index = EdgeIndex::new(graph);
    let mut fusions = Vec::new();
    for (silu_mul_id, silu_mul) in &graph.nodes {
        if silu_mul.operator.name() != SILU_MUL_OPERATOR
            || silu_mul.inputs.len() != 2
            || silu_mul.outputs.len() != 1
        {
            continue;
        }
        let (gate_edge, up_edge) = (&silu_mul.inputs[0], &silu_mul.inputs[1]);
        if gate_edge == up_edge {
            continue;
        }
        let (Some(gate_id), Some(up_id)) = (
            index.sole_producer(graph, gate_edge, "matmul", 2),
            index.sole_producer(graph, up_edge, "matmul", 2),
        ) else {
            continue;
        };
        let (gate, up) = (&graph.nodes[gate_id], &graph.nodes[up_id]);
        let (Some(gate_shape), Some(up_shape)) =
            (graph.edges.get(gate_edge), graph.edges.get(up_edge))
        else {
            continue;
        };
        if gate.inputs[0] != up.inputs[0]
            || gate.attributes != up.attributes
            || gate_shape.descriptor.shape != up_shape.descriptor.shape
        {
            continue;
        }
        fusions.push((
            gate_id.clone(),
            up_id.clone(),
            silu_mul_id.clone(),
            gate_edge.clone(),
            up_edge.clone(),
        ));
    }
    let mut fused = graph.clone();
    for (gate_id, up_id, silu_mul_id, gate_edge, up_edge) in &fusions {
        let silu_mul = fused
            .nodes
            .remove(silu_mul_id)
            .expect("each fused silu-mul is a graph node");
        let up = fused
            .nodes
            .remove(up_id)
            .expect("each fused up matmul is a graph node");
        let output = silu_mul.outputs[0].clone();
        let up_weight = up.inputs[1].clone();
        let gate = fused
            .nodes
            .get_mut(gate_id)
            .expect("each fused gate matmul is a graph node");
        gate.operator = gated_matmul_operator();
        gate.inputs.push(up_weight.clone());
        gate.outputs = vec![output.clone()];
        fused.edges.remove(gate_edge);
        fused.edges.remove(up_edge);
        // `a` was read by both matmuls; the up weight now by the gate's id.
        if let Some(edge) = fused.edges.get_mut(&up.inputs[0]) {
            edge.consumers.remove(up_id);
        }
        if let Some(edge) = fused.edges.get_mut(&up_weight)
            && edge.consumers.remove(up_id)
        {
            edge.consumers.insert(gate_id.clone());
        }
        if let Some(edge) = fused.edges.get_mut(&output)
            && edge.producer.as_ref() == Some(silu_mul_id)
        {
            edge.producer = Some(gate_id.clone());
        }
    }
    (fused, 2 * fusions.len())
}

#[cfg(test)]
#[path = "graph_fusion_tests.rs"]
mod tests;
