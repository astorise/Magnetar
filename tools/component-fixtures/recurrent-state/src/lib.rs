//! Test-only graph producer for the 1.11 recurrent-state contract.
//!
//! Build for `wasm32-unknown-unknown`, then use `wasm-tools component new`.
//! The resulting graph is valid but deliberately not executable until the
//! Runtime implements transactional recurrent-state lifecycle handling.

wit_bindgen::generate!({
    path: "../../../magnetar-runtime/wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    GraphPhase, TensorShape, add_node, begin_graph, declare_input, finish_graph, recurrent_input,
    recurrent_output, recurrent_resource,
};

fn graph(phase: GraphPhase, position: u64) -> String {
    begin_graph(phase, 1, position);
    let shape = TensorShape {
        dimensions: vec![2, 3],
    };
    let resource = recurrent_resource(2, "conv", &shape);
    let previous = recurrent_input(&resource);
    let next = add_node(
        "update",
        "silu",
        "Activation",
        &[],
        &[previous],
        &shape,
        None,
    );
    recurrent_output(&resource, &next);
    let data = declare_input("data", &shape);
    let output = add_node("output", "add", "Tensor", &[], &[data, next], &shape, None);
    finish_graph(&output)
}

struct Component;
impl Guest for Component {
    fn build_prefill_graph(_: u64) -> String {
        graph(GraphPhase::Prefill, 0)
    }
    fn build_decode_graph(position: u64) -> String {
        graph(GraphPhase::Decode, position)
    }
    fn build_prefill_graph_segment(_: u64, _: u32, _: u32) -> String {
        graph(GraphPhase::Prefill, 0)
    }
    fn build_decode_graph_segment(position: u64, _: u32, _: u32) -> String {
        graph(GraphPhase::Decode, position)
    }
}

export!(Component);
