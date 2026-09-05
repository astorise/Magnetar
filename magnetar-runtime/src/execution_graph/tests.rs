//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{
    DeviceBinding, DeviceId, FallbackClass, OperatorFamily, ShapeDescriptor, TensorDescriptor,
};

fn tensor(shape: [u64; 2]) -> TensorDescriptor {
    TensorDescriptor::materialized(
        ShapeDescriptor::new(shape),
        DTypeDescriptor::portable(crate::ComputeDType::Float32),
    )
}

#[test]
fn graph_validation_accepts_component_produced_portable_graph() {
    let catalog = default_graph_catalog();
    let input = TensorEdgeId::new("input");
    let weight = TensorEdgeId::new("weight");
    let output = TensorEdgeId::new("output");
    let node_id = ExecutionNodeId::new("matmul");
    let graph = ExecutionGraph::new(ExecutionGraphId::new("decode"), ExecutionGraphPhase::Decode)
        .with_producer(ExecutionGraphProducer::ModelComponent {
            component_id: "llama".into(),
        })
        .with_edge(TensorEdge::new(input.clone(), tensor([2, 3])).with_consumer(node_id.clone()))
        .with_edge(TensorEdge::new(weight.clone(), tensor([3, 4])).with_consumer(node_id.clone()))
        .with_edge(TensorEdge::new(output.clone(), tensor([2, 4])).with_producer(node_id.clone()))
        .with_node(
            ExecutionNode::new(
                node_id.clone(),
                OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
            )
            .with_input(input)
            .with_input(weight)
            .with_output(output),
        );

    let report = graph.validate(&catalog).unwrap();
    assert_eq!(report.validated_nodes, 1);
    assert!(graph.identity_key().contains("decode"));
}

#[test]
fn graph_validation_rejects_unknown_operator_and_shape_errors() {
    let catalog = default_graph_catalog();
    let input = TensorEdgeId::new("input");
    let weight = TensorEdgeId::new("weight");
    let output = TensorEdgeId::new("output");
    let node_id = ExecutionNodeId::new("matmul");
    let graph = ExecutionGraph::new(ExecutionGraphId::new("bad"), ExecutionGraphPhase::Test)
        .with_edge(TensorEdge::new(input.clone(), tensor([2, 3])))
        .with_edge(TensorEdge::new(weight.clone(), tensor([5, 4])))
        .with_edge(TensorEdge::new(output.clone(), tensor([2, 4])))
        .with_node(
            ExecutionNode::new(
                node_id,
                OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
            )
            .with_input(input)
            .with_input(weight)
            .with_output(output),
        );

    assert!(matches!(
        graph.validate(&catalog),
        Err(GraphError::Validation(OperatorError::ShapeMismatch { .. }))
    ));
}

#[test]
fn graph_planning_rejects_silent_movement() {
    let catalog = default_graph_catalog();
    let input = TensorEdgeId::new("input");
    let output = TensorEdgeId::new("output");
    let node_id = ExecutionNodeId::new("gelu");
    let edge_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")));
    let planned_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(DeviceId::new("gpu:1")));
    let graph = ExecutionGraph::new(ExecutionGraphId::new("plan"), ExecutionGraphPhase::Prefill)
        .with_edge(
            TensorEdge::new(input.clone(), tensor([2, 3]))
                .with_affinity(edge_affinity)
                .with_consumer(node_id.clone()),
        )
        .with_edge(TensorEdge::new(output.clone(), tensor([2, 3])).with_producer(node_id.clone()))
        .with_node(
            ExecutionNode::new(
                node_id,
                OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
            )
            .with_input(input)
            .with_output(output),
        );

    assert!(matches!(
        plan_execution_graph(
            &graph,
            &catalog,
            &GraphPlanningPolicy::default(),
            Some(&planned_affinity)
        ),
        Err(GraphError::Planning(
            OperatorError::ResourceAffinityConflict { .. }
        ))
    ));
}

#[test]
fn graph_validation_rejects_producer_field_that_disagrees_with_node_outputs() {
    // `node.outputs` is authoritative (Correctif 6); a `TensorEdge::producer`
    // that names a node other than the one whose `outputs` actually lists
    // this edge is a diverging topology, not a harmless annotation.
    let catalog = default_graph_catalog();
    let input = TensorEdgeId::new("input");
    let output = TensorEdgeId::new("output");
    let real_producer = ExecutionNodeId::new("gelu");
    // A real node in the graph, just not the one `node.outputs` actually
    // lists this edge under -- so the earlier "producer node exists" check
    // passes and the producer/outputs consistency check is what fires.
    let wrong_producer = ExecutionNodeId::new("silu");
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("bad-producer"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(TensorEdge::new(input.clone(), tensor([1, 1])))
    .with_edge(
        TensorEdge::new(output.clone(), tensor([1, 1])).with_producer(wrong_producer.clone()),
    )
    .with_node(
        ExecutionNode::new(
            real_producer,
            OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
        )
        .with_input(input)
        .with_output(output),
    )
    .with_node(ExecutionNode::new(
        wrong_producer,
        OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
    ));

    assert!(matches!(
        graph.validate(&catalog),
        Err(GraphError::LifecycleInvalid(_))
    ));
}

#[test]
fn graph_validation_rejects_two_nodes_claiming_the_same_output() {
    let catalog = default_graph_catalog();
    let shared_output = TensorEdgeId::new("shared-output");
    let first = ExecutionNodeId::new("first");
    let second = ExecutionNodeId::new("second");
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("duplicate-producer"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(TensorEdge::new(shared_output.clone(), tensor([1, 1])))
    .with_node(
        ExecutionNode::new(
            first,
            OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
        )
        .with_output(shared_output.clone()),
    )
    .with_node(
        ExecutionNode::new(
            second,
            OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
        )
        .with_output(shared_output),
    );

    assert!(matches!(
        graph.validate(&catalog),
        Err(GraphError::DuplicateProducer(_))
    ));
}

#[test]
fn graph_validation_rejects_cycle() {
    let catalog = default_graph_catalog();
    let a_to_b = TensorEdgeId::new("a-to-b");
    let b_to_a = TensorEdgeId::new("b-to-a");
    let node_a = ExecutionNodeId::new("a");
    let node_b = ExecutionNodeId::new("b");
    let graph = ExecutionGraph::new(ExecutionGraphId::new("cyclic"), ExecutionGraphPhase::Test)
        .with_edge(TensorEdge::new(a_to_b.clone(), tensor([1, 1])))
        .with_edge(TensorEdge::new(b_to_a.clone(), tensor([1, 1])))
        .with_node(
            ExecutionNode::new(
                node_a,
                OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
            )
            .with_input(b_to_a.clone())
            .with_output(a_to_b.clone()),
        )
        .with_node(
            ExecutionNode::new(
                node_b,
                OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
            )
            .with_input(a_to_b)
            .with_output(b_to_a),
        );

    assert!(matches!(
        graph.validate(&catalog),
        Err(GraphError::LifecycleInvalid(_))
    ));
}

#[test]
fn graph_validation_rejects_consumers_field_that_disagrees_with_node_inputs() {
    let catalog = default_graph_catalog();
    let input = TensorEdgeId::new("input");
    let output = TensorEdgeId::new("output");
    let real_consumer = ExecutionNodeId::new("gelu");
    // A real node in the graph, just not the one `node.inputs` actually
    // lists this edge under.
    let wrong_consumer = ExecutionNodeId::new("silu");
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("bad-consumer"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(TensorEdge::new(input.clone(), tensor([1, 1])).with_consumer(wrong_consumer.clone()))
    .with_edge(TensorEdge::new(output.clone(), tensor([1, 1])))
    .with_node(
        ExecutionNode::new(
            real_consumer,
            OperatorId::magnetar("gelu", 1, OperatorFamily::Activation),
        )
        .with_input(input)
        .with_output(output),
    )
    .with_node(ExecutionNode::new(
        wrong_consumer,
        OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
    ));

    assert!(matches!(
        graph.validate(&catalog),
        Err(GraphError::LifecycleInvalid(_))
    ));
}

#[test]
fn kv_prefix_adapter_and_observability_metadata_are_explicit() {
    let edge_id = TensorEdgeId::new("kv");
    let edge = TensorEdge::new(edge_id.clone(), tensor([1, 8]));
    let graph = ExecutionGraph::new(ExecutionGraphId::new("decode"), ExecutionGraphPhase::Decode)
        .with_edge(TensorEdge {
            kv_cache: Some(GraphKvCacheMetadata {
                cache_id: "kv:0".into(),
                behavior: GraphKvCacheBehavior::Append,
                paged: true,
                compatibility_key: "model:a".into(),
            }),
            prefix_cache: Some(GraphPrefixCacheMetadata {
                reused_prefix_length: 4,
                backing_kv_cache: "kv:0".into(),
            }),
            ..edge
        });
    assert!(graph.edges[&edge_id].kv_cache.as_ref().unwrap().paged);
    let observations = execute_graph_boundary(
        &graph,
        &default_graph_catalog(),
        &GraphPlanningPolicy::default(),
    )
    .unwrap();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == GraphObservationKind::GraphExecutionCompleted)
    );
}
