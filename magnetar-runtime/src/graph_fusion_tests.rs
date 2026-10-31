use super::*;
use crate::first_native_runtime::{e2e_fixture, first_native_graph_execution_order_for_tests};

fn is_weight(edge: &TensorEdgeId) -> bool {
    edge.as_str().starts_with("weight.")
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn every_biased_projection_of_the_real_qwen_graph_is_fused() {
    let mut fixture = e2e_fixture().expect("fixture builds");
    fixture.config.attention_bias = true;
    let layers = fixture.config.architecture.layer_count as usize;
    let (graphs, _definition, _instance) =
        crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(
            &fixture, 3,
        )
        .expect("the real Qwen Component builds a bias-bearing graph");
    for graph in [&graphs.prefill, &graphs.decode] {
        let (fused, count) = fuse_matmul_bias(graph, is_weight);
        assert_eq!(count, 3 * layers, "one per q/k/v projection per layer");
        assert_eq!(fused.nodes.len(), graph.nodes.len() - count);
        assert_eq!(fused.edges.len(), graph.edges.len() - count);
        let fused_nodes: Vec<_> = fused
            .nodes
            .values()
            .filter(|node| node.operator.name() == MATMUL_BIAS_OPERATOR)
            .collect();
        assert_eq!(fused_nodes.len(), count);
        for node in &fused_nodes {
            assert_eq!(node.inputs.len(), 3);
            assert!(is_weight(&node.inputs[1]) && is_weight(&node.inputs[2]));
            // The fused node writes the edge the add used to write.
            assert!(
                graph
                    .nodes
                    .values()
                    .any(|add| add.operator.name() == "add" && add.outputs == node.outputs)
            );
        }
        // No add is left reading a matmul output and a bias weight.
        assert!(!fused.nodes.values().any(|node| {
            node.operator.name() == "add" && node.inputs.get(1).is_some_and(is_weight)
        }));
        // Every edge a node reads is still produced or is a graph input.
        let produced: std::collections::BTreeSet<_> = fused
            .nodes
            .values()
            .flat_map(|node| &node.outputs)
            .collect();
        for node in fused.nodes.values() {
            for input in &node.inputs {
                assert!(
                    produced.contains(input) || fused.edges.contains_key(input),
                    "{input} has no producer and no edge"
                );
            }
        }
        let order = first_native_graph_execution_order_for_tests(&fused)
            .expect("the fused graph still has a valid execution order");
        assert_eq!(order.len(), fused.nodes.len());
        // Fusing again finds nothing left to fuse.
        assert_eq!(fuse_matmul_bias(&fused, is_weight).1, 0);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn a_graph_without_biases_or_with_a_shared_intermediate_is_left_unchanged() {
    let fixture = e2e_fixture().expect("fixture builds");
    let (graphs, _definition, _instance) =
        crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(
            &fixture, 3,
        )
        .expect("the real Qwen Component builds");
    let (fused, count) = fuse_matmul_bias(&graphs.decode, is_weight);
    assert_eq!(count, 0, "the canonical fixture has no attention bias");
    assert_eq!(fused, graphs.decode);

    let mut biased = e2e_fixture().expect("fixture builds");
    biased.config.attention_bias = true;
    let (graphs, _definition, _instance) =
        crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(&biased, 3)
            .expect("the real Qwen Component builds a bias-bearing graph");
    let mut shared = graphs.decode.clone();
    // Give one biased projection's intermediate a second consumer: that
    // pair must no longer be fused, every other one still is.
    let (add_id, intermediate) = shared
        .nodes
        .iter()
        .find(|(_, node)| node.operator.name() == "add" && is_weight(&node.inputs[1]))
        .map(|(id, node)| (id.clone(), node.inputs[0].clone()))
        .unwrap();
    let other = shared
        .nodes
        .keys()
        .find(|id| **id != add_id)
        .cloned()
        .unwrap();
    shared
        .nodes
        .get_mut(&other)
        .unwrap()
        .inputs
        .push(intermediate.clone());
    let (_, full) = fuse_matmul_bias(&graphs.decode, is_weight);
    let (fused, partial) = fuse_matmul_bias(&shared, is_weight);
    assert_eq!(partial, full - 1);
    assert!(fused.edges.contains_key(&intermediate));
    assert!(fused.nodes.contains_key(&add_id));
}
