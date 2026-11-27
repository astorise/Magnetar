//! Synthetic generation probe, not an Olala model.

use magnetar_component_api::v1_16::magnetar::model_component_graph::{
    graph_builder::{self as gb, Attribute, AttributeValue, GraphPhase, KvCacheRole, TensorShape},
    model_config,
};

fn shape(dimensions: &[u64]) -> TensorShape {
    TensorShape {
        dimensions: dimensions.to_vec(),
    }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.into(),
        value,
    }
}

fn integer(name: &str, value: u64) -> Attribute {
    attr(name, AttributeValue::Integer(i64::try_from(value).unwrap()))
}

pub fn graph(phase: GraphPhase, tokens: u64, position: u64) -> String {
    let _activation = model_config::hidden_activation();
    gb::begin_graph(phase, tokens, position);
    let token_ids = gb::declare_input("token_ids", &shape(&[tokens]));
    let embedding = gb::weight_edge("token_embedding");
    let hidden = gb::add_node(
        "embedding",
        "embedding",
        "Tensor",
        &[],
        &[token_ids, embedding],
        &shape(&[tokens, 4]),
        None,
    );
    let positions = gb::add_node(
        "positions",
        "position-range",
        "Tensor",
        &[integer("position_offset", position)],
        &[hidden.clone()],
        &shape(&[tokens]),
        None,
    );
    let raw_shape = shape(&[2, 2]);
    let raw_k = gb::recurrent_resource(0, "raw_k", &raw_shape);
    let raw_v = gb::recurrent_resource(0, "raw_v", &raw_shape);
    let k = gb::kv_resource(0, KvCacheRole::K);
    let v = gb::kv_resource(0, KvCacheRole::V);
    let history_shape = shape(&[position.max(1), 2, 2]);
    let mut inputs = vec![
        hidden.clone(),
        gb::recurrent_input(&raw_k),
        gb::recurrent_input(&raw_v),
        gb::kv_history_input(&k, &history_shape),
        gb::kv_history_input(&v, &history_shape),
        positions,
    ];
    for name in [
        "q",
        "a_key",
        "b_key",
        "a_value",
        "b_value",
        "shift_key",
        "shift_value",
        "lambda",
        "q_norm",
        "k_norm",
        "scaler",
    ] {
        inputs.push(gb::weight_edge(&format!("probe.{name}")));
    }
    let attrs = [
        integer("query_heads", 4),
        integer("kv_heads", 2),
        integer("head_dim", 2),
        integer("rank", 2),
        integer("past_length", position),
        attr("epsilon", AttributeValue::FloatValue(1e-5)),
        attr("zero_centered_norm", AttributeValue::Boolean(false)),
        attr(
            "scale",
            AttributeValue::FloatValue(std::f64::consts::FRAC_1_SQRT_2),
        ),
        attr("softcap", AttributeValue::FloatValue(3.0)),
        integer("window_left", 7),
        attr("round_bfloat16", AttributeValue::Boolean(false)),
    ];
    let updated = shape(&[position + tokens, 2, 2]);
    let outputs = gb::add_node_outputs(
        "projected-attention",
        "projected-differential-attention",
        "Attention",
        &attrs,
        &inputs,
        &[
            shape(&[tokens, 2, 2]),
            raw_shape.clone(),
            raw_shape,
            updated.clone(),
            updated,
        ],
    );
    gb::recurrent_output(&raw_k, &outputs[1]);
    gb::recurrent_output(&raw_v, &outputs[2]);
    gb::kv_history_output(&k, &outputs[3]);
    gb::kv_history_output(&v, &outputs[4]);
    // Logits deliberately remain a simple projection; the attention branch
    // exercises execution and history but is not a qualified model output.
    let lm_head = gb::weight_edge("lm_head");
    let logits = gb::add_node(
        "lm-head",
        "matmul",
        "LinearAlgebra",
        &[],
        &[hidden, lm_head],
        &shape(&[tokens, 258]),
        None,
    );
    gb::finish_graph(&logits)
}
