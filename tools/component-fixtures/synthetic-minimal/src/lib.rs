//! Test-only Component with no decoder layers or hidden activation.
//!
//! Rebuild with `cargo build --locked --release --target wasm32-unknown-unknown`
//! in this directory and `wasm-tools component new` on the resulting wasm.
//! The Gemma submodule supplies the shared 1.8.0 contract.

wit_bindgen::generate!({
    path: "../../../components/gemma/wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, TensorShape, add_node, alias_weight_edge, begin_graph,
    declare_input, finish_graph, weight_edge,
};
use magnetar::model_component_graph::model_config;

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn graph(phase: GraphPhase, length: u64, position: u64) -> String {
    // This graph has no MLP, so neither a known nor an unknown activation
    // changes its semantics. The call still makes that choice explicit.
    let _ = model_config::hidden_activation();
    let config = model_config::model_config();
    begin_graph(phase, length, position);
    let tokens = declare_input("token_ids", &shape(vec![length]));
    let embeddings = weight_edge("token_embedding");
    let hidden = add_node(
        "embedding",
        "embedding",
        "Tensor",
        &[],
        &[tokens, embeddings],
        &shape(vec![length, config.hidden_size]),
        None,
    );
    let norm = weight_edge("final_norm");
    let hidden = add_node(
        "final_norm",
        "rmsnorm",
        "Normalization",
        &[Attribute {
            name: "epsilon".into(),
            value: AttributeValue::FloatValue(config.rms_norm_eps),
        }],
        &[hidden, norm],
        &shape(vec![length, config.hidden_size]),
        None,
    );
    let head = weight_edge("lm_head");
    if config.tie_word_embeddings {
        alias_weight_edge(&head, "token_embedding");
    }
    let logits = add_node(
        "lm_head",
        "matmul",
        "LinearAlgebra",
        &[],
        &[hidden, head],
        &shape(vec![length, config.vocabulary_size]),
        None,
    );
    finish_graph(&logits)
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        graph(GraphPhase::Prefill, sequence_length, 0)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        graph(GraphPhase::Decode, 1, cached_token_count)
    }

    fn build_prefill_graph_segment(sequence_length: u64, _: u32, _: u32) -> String {
        graph(GraphPhase::Prefill, sequence_length, 0)
    }

    fn build_decode_graph_segment(cached_token_count: u64, _: u32, _: u32) -> String {
        graph(GraphPhase::Decode, 1, cached_token_count)
    }
}

export!(Component);
