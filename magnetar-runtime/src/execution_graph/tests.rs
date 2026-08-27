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
