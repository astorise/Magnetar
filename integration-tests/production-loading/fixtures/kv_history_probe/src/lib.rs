//! Minimal Component for exercising graph-builder 1.16's KV history ABI.
//! It builds a synthetic graph, not an executable language model.

use magnetar_component_api::v1_16::{
    Guest, export,
    magnetar::model_component_graph::graph_builder::{
        self as gb, GraphPhase, KvCacheRole, TensorShape,
    },
};

struct Component;

fn graph(phase: GraphPhase, tokens: u64, position: u64) -> String {
    gb::begin_graph(phase, tokens, position);
    let shape = TensorShape {
        dimensions: vec![position.max(1), 2],
    };
    let mut outputs = Vec::new();
    for (role, name) in [(KvCacheRole::K, "key"), (KvCacheRole::V, "value")] {
        let resource = gb::kv_resource(0, role);
        let input = gb::kv_history_input(&resource, &shape);
        let output = gb::add_node(name, "silu", "Activation", &[], &[input], &shape, None);
        gb::kv_history_output(&resource, &output);
        outputs.push(output);
    }
    let result = gb::add_node("combine", "add", "Tensor", &[], &outputs, &shape, None);
    gb::finish_graph(&result)
}

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
