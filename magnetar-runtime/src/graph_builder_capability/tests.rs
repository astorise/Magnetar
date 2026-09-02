//! Unit tests for the parent module.

use super::*;

fn shape(dimensions: Vec<u64>) -> ComponentValue {
    ComponentValue::Record(vec![(
        "dimensions".to_string(),
        ComponentValue::List(
            dimensions
                .into_iter()
                .map(|d| ComponentValue::S64(d as i64))
                .collect(),
        ),
    )])
}

fn context() -> SessionContext {
    let mut weight_shapes = BTreeMap::new();
    weight_shapes.insert("token_embedding".to_string(), vec![32, 8]);
    weight_shapes.insert("lm_head".to_string(), vec![8, 32]);
    SessionContext {
        component_id: "test-component".to_string(),
        compatibility_key: "test-key".to_string(),
        kv_namespace: "qwen".to_string(),
        weight_shapes,
        output_edge_name: "logits".to_string(),
    }
}

#[test]
fn builds_a_minimal_two_node_graph_end_to_end() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-1", context());

    capability
        .call(
            "instance-1",
            "begin-graph",
            &[
                ComponentValue::Enum("prefill".to_string()),
                ComponentValue::S64(4),
                ComponentValue::S64(0),
            ],
        )
        .expect("begin-graph succeeds");

    let input_edge = capability
        .call(
            "instance-1",
            "declare-input",
            &[
                ComponentValue::String("token_ids".to_string()),
                shape(vec![4]),
            ],
        )
        .expect("declare-input succeeds");
    let ComponentValue::String(input_edge) = &input_edge[0] else {
        panic!("expected a string edge id");
    };

    let weight_edge = capability
        .call(
            "instance-1",
            "weight-edge",
            &[ComponentValue::String("token_embedding".to_string())],
        )
        .expect("weight-edge succeeds");
    let ComponentValue::String(weight_edge) = &weight_edge[0] else {
        panic!("expected a string edge id");
    };

    let node_output = capability
        .call(
            "instance-1",
            "add-node",
            &[
                ComponentValue::String("embedding".to_string()),
                ComponentValue::String("embedding".to_string()),
                ComponentValue::String("Tensor".to_string()),
                ComponentValue::List(vec![]),
                ComponentValue::List(vec![
                    ComponentValue::String(input_edge.clone()),
                    ComponentValue::String(weight_edge.clone()),
                ]),
                shape(vec![4, 8]),
                ComponentValue::Option(None),
            ],
        )
        .expect("add-node succeeds");
    let ComponentValue::String(node_output) = &node_output[0] else {
        panic!("expected a string edge id");
    };

    let handle = capability
        .call(
            "instance-1",
            "finish-graph",
            &[ComponentValue::String(node_output.clone())],
        )
        .expect("finish-graph succeeds");
    let ComponentValue::String(handle) = &handle[0] else {
        panic!("expected a string handle");
    };

    let graph = capability
        .take_graph("instance-1", handle)
        .expect("the finished graph is retrievable by its handle");
    assert_eq!(graph.nodes.len(), 1);
    assert!(graph.edges.contains_key(&TensorEdgeId::new("logits")));
    assert!(
        !graph
            .edges
            .contains_key(&TensorEdgeId::new(node_output.as_str()))
    );
    let embedding_node = graph
        .nodes
        .get(&ExecutionNodeId::new("embedding"))
        .expect("embedding node exists");
    assert_eq!(embedding_node.outputs, vec![TensorEdgeId::new("logits")]);

    // The handle is consumed at most once.
    assert!(capability.take_graph("instance-1", handle).is_none());
}

#[test]
fn add_node_rejects_an_input_edge_that_does_not_exist() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-2", context());
    capability
        .call(
            "instance-2",
            "begin-graph",
            &[
                ComponentValue::Enum("prefill".to_string()),
                ComponentValue::S64(4),
                ComponentValue::S64(0),
            ],
        )
        .expect("begin-graph succeeds");

    let result = capability.call(
        "instance-2",
        "add-node",
        &[
            ComponentValue::String("bad-node".to_string()),
            ComponentValue::String("matmul".to_string()),
            ComponentValue::String("LinearAlgebra".to_string()),
            ComponentValue::List(vec![]),
            ComponentValue::List(vec![ComponentValue::String("does.not.exist".to_string())]),
            shape(vec![4, 8]),
            ComponentValue::Option(None),
        ],
    );
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn weight_edge_rejects_an_unrecognized_logical_name() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-3", context());

    let result = capability.call(
        "instance-3",
        "weight-edge",
        &[ComponentValue::String("not_a_real_weight".to_string())],
    );
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn begin_graph_twice_without_finish_is_rejected() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-4", context());
    let args = [
        ComponentValue::Enum("prefill".to_string()),
        ComponentValue::S64(4),
        ComponentValue::S64(0),
    ];
    capability
        .call("instance-4", "begin-graph", &args)
        .expect("first begin-graph succeeds");
    let result = capability.call("instance-4", "begin-graph", &args);
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn kv_resource_is_namespaced_and_parseable_by_the_existing_qwen_convention() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-5", context());

    let result = capability
        .call(
            "instance-5",
            "kv-resource",
            &[
                ComponentValue::S64(3),
                ComponentValue::Enum("k".to_string()),
            ],
        )
        .expect("kv-resource succeeds");
    assert_eq!(
        result,
        vec![ComponentValue::String("qwen.layer3.k".to_string())]
    );
}

#[test]
fn tied_embeddings_alias_the_lm_head_weight_edge() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-6", context());
    capability
        .call(
            "instance-6",
            "begin-graph",
            &[
                ComponentValue::Enum("prefill".to_string()),
                ComponentValue::S64(4),
                ComponentValue::S64(0),
            ],
        )
        .expect("begin-graph succeeds");
    capability
        .call(
            "instance-6",
            "weight-edge",
            &[ComponentValue::String("token_embedding".to_string())],
        )
        .expect("weight-edge succeeds");
    capability
        .call(
            "instance-6",
            "weight-edge",
            &[ComponentValue::String("lm_head".to_string())],
        )
        .expect("weight-edge succeeds");
    capability
        .call(
            "instance-6",
            "alias-weight-edge",
            &[
                ComponentValue::String("weight.lm_head".to_string()),
                ComponentValue::String("token_embedding".to_string()),
            ],
        )
        .expect("alias-weight-edge succeeds");

    let sessions = capability.sessions.lock().unwrap();
    let graph = sessions
        .get("instance-6")
        .expect("session exists")
        .graph
        .as_ref()
        .expect("graph under construction");
    let lm_head_edge = graph
        .edges
        .get(&TensorEdgeId::new("weight.lm_head"))
        .expect("lm_head edge exists");
    assert_eq!(
        lm_head_edge.aliasing,
        TensorAliasing::MayAlias(TensorEdgeId::new("weight.token_embedding"))
    );
}
