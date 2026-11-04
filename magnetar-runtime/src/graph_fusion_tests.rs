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

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn residual_and_swiglu_pairs_of_the_real_qwen_graph_are_fused_after_the_biases() {
    let mut fixture = e2e_fixture().expect("fixture builds");
    fixture.config.attention_bias = true;
    let layers = fixture.config.architecture.layer_count as usize;
    let (graphs, _definition, _instance) =
        crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(
            &fixture, 3,
        )
        .expect("the real Qwen Component builds a bias-bearing graph");
    for graph in [&graphs.prefill, &graphs.decode] {
        let (biased, bias_count) = fuse_matmul_bias(graph, is_weight);
        let (residual, residual_count) = fuse_matmul_residual(&biased);
        let (fused, silu_count) = fuse_silu_mul(&residual);
        assert_eq!(bias_count, 3 * layers);
        assert_eq!(residual_count, 2 * layers, "o_proj and down_proj per layer");
        assert_eq!(silu_count, layers, "one SwiGLU gate per layer");
        assert_eq!(
            fused.nodes.len(),
            graph.nodes.len() - bias_count - residual_count - silu_count
        );
        let count = |name: &str| {
            fused
                .nodes
                .values()
                .filter(|node| node.operator.name() == name)
                .count()
        };
        assert_eq!(count(MATMUL_RESIDUAL_OPERATOR), residual_count);
        assert_eq!(count(SILU_MUL_OPERATOR), silu_count);
        assert_eq!(count("residual-add"), 0);
        assert_eq!(count("silu"), 0);
        assert_eq!(count("mul"), 0);
        for node in fused.nodes.values() {
            if node.operator.name() == MATMUL_RESIDUAL_OPERATOR {
                assert_eq!(node.inputs.len(), 3);
                assert!(is_weight(&node.inputs[1]) && !is_weight(&node.inputs[2]));
            }
            if node.operator.name() == SILU_MUL_OPERATOR {
                assert_eq!(node.inputs.len(), 2);
            }
        }
        let order = first_native_graph_execution_order_for_tests(&fused)
            .expect("the fused graph still has a valid execution order");
        assert_eq!(order.len(), fused.nodes.len());
        assert_eq!(fuse_matmul_residual(&fused).1, 0);
        assert_eq!(fuse_silu_mul(&fused).1, 0);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn each_layers_swiglu_projection_becomes_one_gated_matmul() {
    let mut fixture = e2e_fixture().expect("fixture builds");
    fixture.config.attention_bias = true;
    let layers = fixture.config.architecture.layer_count as usize;
    let (graphs, _definition, _instance) =
        crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(
            &fixture, 3,
        )
        .expect("the real Qwen Component builds a bias-bearing graph");
    for graph in [&graphs.prefill, &graphs.decode] {
        let (biased, _) = fuse_matmul_bias(graph, is_weight);
        let (residual, _) = fuse_matmul_residual(&biased);
        let (swiglu, _) = fuse_silu_mul(&residual);
        // Without the silu-mul pass first, nothing matches.
        assert_eq!(fuse_gated_matmul(&residual).1, 0);
        let (fused, removed) = fuse_gated_matmul(&swiglu);
        assert_eq!(removed, 2 * layers, "the up matmul and silu-mul per layer");
        assert_eq!(fused.nodes.len(), swiglu.nodes.len() - removed);
        assert_eq!(fused.edges.len(), swiglu.edges.len() - removed);
        let gated: Vec<_> = fused
            .nodes
            .values()
            .filter(|node| node.operator.name() == GATED_MATMUL_OPERATOR)
            .collect();
        assert_eq!(gated.len(), layers);
        assert!(
            !fused
                .nodes
                .values()
                .any(|node| node.operator.name() == SILU_MUL_OPERATOR)
        );
        for node in &gated {
            assert_eq!(node.inputs.len(), 3);
            assert!(!is_weight(&node.inputs[0]));
            assert!(is_weight(&node.inputs[1]) && is_weight(&node.inputs[2]));
            // It writes what the silu-mul wrote, and is that edge's producer.
            assert!(swiglu.nodes.values().any(|silu_mul| {
                silu_mul.operator.name() == SILU_MUL_OPERATOR && silu_mul.outputs == node.outputs
            }));
            assert_eq!(
                fused.edges[&node.outputs[0]].producer.as_ref(),
                Some(&node.id)
            );
            for input in &node.inputs {
                if let Some(edge) = fused.edges.get(input) {
                    assert!(edge.consumers.contains(&node.id), "{input}");
                }
            }
        }
        // No edge names a node that is gone.
        for edge in fused.edges.values() {
            for consumer in &edge.consumers {
                assert!(
                    fused.nodes.contains_key(consumer),
                    "{} -> {consumer}",
                    edge.id
                );
            }
            if let Some(producer) = &edge.producer {
                assert!(
                    fused.nodes.contains_key(producer),
                    "{} <- {producer}",
                    edge.id
                );
            }
        }
        let order = first_native_graph_execution_order_for_tests(&fused)
            .expect("the fused graph still has a valid execution order");
        assert_eq!(order.len(), fused.nodes.len());
        assert_eq!(fuse_gated_matmul(&fused).1, 0);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn every_rmsnorm_of_the_real_qwen_graph_folds_into_the_matmuls_reading_it() {
    for attention_bias in [false, true] {
        let mut fixture = e2e_fixture().expect("fixture builds");
        fixture.config.attention_bias = attention_bias;
        let layers = fixture.config.architecture.layer_count as usize;
        let (graphs, _definition, _instance) =
            crate::first_native_runtime::build_first_native_graphs_from_real_qwen_component(
                &fixture, 3,
            )
            .expect("the real Qwen Component builds");
        for graph in [&graphs.prefill, &graphs.decode] {
            let (biased, _) = fuse_matmul_bias(graph, is_weight);
            let (residual, _) = fuse_matmul_residual(&biased);
            let (swiglu, _) = fuse_silu_mul(&residual);
            let (gated, _) = fuse_gated_matmul(&swiglu);
            let count = |graph: &ExecutionGraph, name: &str| {
                graph
                    .nodes
                    .values()
                    .filter(|node| node.operator.name() == name)
                    .count()
            };
            // Without the fused Operators advertised, nothing changes.
            let (unchanged, none) = fuse_rmsnorm_matmuls(&gated, |_| false);
            assert_eq!(none, 0);
            assert_eq!(unchanged, gated);
            let (fused, removed) = fuse_rmsnorm_matmuls(&gated, |_| true);
            // Two per layer (before attention and before the MLP), plus
            // the final norm before the LM head.
            assert_eq!(removed, 2 * layers + 1);
            assert_eq!(count(&fused, "rmsnorm"), 0);
            assert_eq!(fused.nodes.len(), gated.nodes.len() - removed);
            assert_eq!(fused.edges.len(), gated.edges.len() - removed);
            assert_eq!(count(&fused, RMSNORM_GATED_MATMUL_OPERATOR), layers);
            let projections = if attention_bias {
                RMSNORM_MATMUL_BIAS_OPERATOR
            } else {
                RMSNORM_MATMUL_OPERATOR
            };
            let lm_head = usize::from(attention_bias);
            assert_eq!(count(&fused, projections), 3 * layers + 1 - lm_head);
            if attention_bias {
                assert_eq!(count(&fused, RMSNORM_MATMUL_OPERATOR), 1);
            }
            for node in fused.nodes.values() {
                if !node.operator.name().starts_with("rmsnorm-") {
                    continue;
                }
                assert!(!is_weight(&node.inputs[0]), "{}", node.id);
                assert!(is_weight(&node.inputs[1]) && is_weight(&node.inputs[2]));
                assert!(node.attributes.contains_key("epsilon"));
                for input in &node.inputs {
                    if let Some(edge) = fused.edges.get(input) {
                        assert!(edge.consumers.contains(&node.id), "{input}");
                    }
                }
            }
            for edge in fused.edges.values() {
                for consumer in &edge.consumers {
                    assert!(
                        fused.nodes.contains_key(consumer),
                        "{} -> {consumer}",
                        edge.id
                    );
                }
            }
            let order = first_native_graph_execution_order_for_tests(&fused)
                .expect("the fused graph still has a valid execution order");
            assert_eq!(order.len(), fused.nodes.len());
            assert_eq!(fuse_rmsnorm_matmuls(&fused, |_| true).1, 0);
            // A norm one of whose consumers cannot be fused stays.
            let (partial, kept) =
                fuse_rmsnorm_matmuls(&gated, |name| name != RMSNORM_GATED_MATMUL_OPERATOR);
            assert_eq!(kept, layers + 1);
            assert_eq!(count(&partial, "rmsnorm"), layers);
        }
    }
}
