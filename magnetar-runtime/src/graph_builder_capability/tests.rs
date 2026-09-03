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

/// Requirement "Component-Produced Graphs Remain Untrusted Until
/// Validated": a Component naming an Operator the portable catalog does
/// not recognize passes every one of this capability's own per-call
/// structural checks (the edges and node id are all otherwise
/// well-formed) but must still be rejected by `finish-graph`'s Runtime
/// validation pass against `initial_operator_catalog`.
#[test]
fn finish_graph_rejects_a_graph_using_an_unknown_operator() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-7", context());
    capability
        .call(
            "instance-7",
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
            "instance-7",
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

    let node_output = capability
        .call(
            "instance-7",
            "add-node",
            &[
                ComponentValue::String("bogus".to_string()),
                ComponentValue::String("not-a-real-operator".to_string()),
                ComponentValue::String("Tensor".to_string()),
                ComponentValue::List(vec![]),
                ComponentValue::List(vec![ComponentValue::String(input_edge.clone())]),
                shape(vec![4]),
                ComponentValue::Option(None),
            ],
        )
        .expect("add-node itself succeeds -- this capability's own structural checks do not know about the Operator catalog");
    let ComponentValue::String(node_output) = &node_output[0] else {
        panic!("expected a string edge id");
    };

    let result = capability.call(
        "instance-7",
        "finish-graph",
        &[ComponentValue::String(node_output.clone())],
    );
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn parse_operator_family_round_trips_every_declared_family() {
    for family in OperatorFamily::ALL {
        let name = match family {
            OperatorFamily::Tensor => "Tensor",
            OperatorFamily::LinearAlgebra => "LinearAlgebra",
            OperatorFamily::Normalization => "Normalization",
            OperatorFamily::PositionEncoding => "PositionEncoding",
            OperatorFamily::Attention => "Attention",
            OperatorFamily::Activation => "Activation",
            OperatorFamily::Quantization => "Quantization",
            OperatorFamily::Layout => "Layout",
            OperatorFamily::SamplingSupport => "SamplingSupport",
            OperatorFamily::Control => "Control",
        };
        assert_eq!(parse_operator_family(name), Ok(family));
    }
}

#[test]
fn parse_operator_family_rejects_an_unknown_name() {
    assert!(parse_operator_family("NotAFamily").is_err());
}

#[test]
fn expect_arity_rejects_the_wrong_argument_count() {
    assert!(expect_arity("op", 2, &[ComponentValue::S64(1)]).is_err());
    assert!(expect_arity("op", 1, &[ComponentValue::S64(1)]).is_ok());
}

#[test]
fn expect_string_rejects_a_non_string_value() {
    assert!(expect_string(&ComponentValue::S64(1)).is_err());
    assert_eq!(expect_string(&ComponentValue::String("x".into())), Ok("x"));
}

#[test]
fn expect_u64_rejects_non_integers_and_negative_integers() {
    assert!(expect_u64(&ComponentValue::String("x".into())).is_err());
    assert!(expect_u64(&ComponentValue::S64(-1)).is_err());
    assert_eq!(expect_u64(&ComponentValue::S64(7)), Ok(7));
}

#[test]
fn expect_enum_rejects_a_non_enum_value() {
    assert!(expect_enum(&ComponentValue::String("x".into())).is_err());
    assert_eq!(expect_enum(&ComponentValue::Enum("x".into())), Ok("x"));
}

#[test]
fn expect_record_rejects_a_non_record_value() {
    assert!(expect_record(&ComponentValue::S64(1)).is_err());
    assert!(expect_record(&ComponentValue::Record(vec![])).is_ok());
}

#[test]
fn expect_list_rejects_a_non_list_value() {
    assert!(expect_list(&ComponentValue::S64(1)).is_err());
    assert!(expect_list(&ComponentValue::List(vec![])).is_ok());
}

fn attribute_variant(case: &str, payload: Option<ComponentValue>) -> ComponentValue {
    ComponentValue::Variant(case.to_string(), payload.map(Box::new))
}

#[test]
fn decode_attribute_value_decodes_every_declared_case() {
    assert_eq!(
        decode_attribute_value(&attribute_variant(
            "boolean",
            Some(ComponentValue::Bool(true))
        )),
        Ok(OperatorAttributeValue::Boolean(true))
    );
    assert_eq!(
        decode_attribute_value(&attribute_variant("integer", Some(ComponentValue::S64(42)))),
        Ok(OperatorAttributeValue::Integer(42))
    );
    // A negative payload takes the s64-fallback path inside the "integer" case.
    assert_eq!(
        decode_attribute_value(&attribute_variant("integer", Some(ComponentValue::S64(-3)))),
        Ok(OperatorAttributeValue::Integer(-3))
    );
    assert_eq!(
        decode_attribute_value(&attribute_variant(
            "float-value",
            Some(ComponentValue::F64(1.5))
        )),
        Ok(OperatorAttributeValue::Float(1.5))
    );
    assert_eq!(
        decode_attribute_value(&attribute_variant(
            "text",
            Some(ComponentValue::String("hi".into()))
        )),
        Ok(OperatorAttributeValue::String("hi".into()))
    );
}

#[test]
fn decode_attribute_value_rejects_malformed_payloads() {
    assert!(
        decode_attribute_value(&attribute_variant("boolean", Some(ComponentValue::S64(1))))
            .is_err()
    );
    assert!(
        decode_attribute_value(&attribute_variant(
            "integer",
            Some(ComponentValue::Bool(true))
        ))
        .is_err()
    );
    assert!(
        decode_attribute_value(&attribute_variant(
            "float-value",
            Some(ComponentValue::Bool(true))
        ))
        .is_err()
    );
    assert!(decode_attribute_value(&attribute_variant("unknown-case", None)).is_err());
    assert!(decode_attribute_value(&attribute_variant("boolean", None)).is_err());
    assert!(decode_attribute_value(&ComponentValue::S64(1)).is_err());
}

#[test]
fn decode_kv_involvement_decodes_append_and_output_and_rejects_a_non_bool_flag() {
    let append_record = ComponentValue::Record(vec![
        (
            "resource-id".to_string(),
            ComponentValue::String("kv.0.k".into()),
        ),
        ("append".to_string(), ComponentValue::Bool(true)),
    ]);
    assert_eq!(
        decode_kv_involvement(&append_record),
        Ok(("kv.0.k".to_string(), GraphKvCacheBehavior::Append))
    );

    let output_record = ComponentValue::Record(vec![
        (
            "resource-id".to_string(),
            ComponentValue::String("kv.0.k".into()),
        ),
        ("append".to_string(), ComponentValue::Bool(false)),
    ]);
    assert_eq!(
        decode_kv_involvement(&output_record),
        Ok(("kv.0.k".to_string(), GraphKvCacheBehavior::Output))
    );

    let malformed_record = ComponentValue::Record(vec![
        (
            "resource-id".to_string(),
            ComponentValue::String("kv.0.k".into()),
        ),
        ("append".to_string(), ComponentValue::S64(1)),
    ]);
    assert!(decode_kv_involvement(&malformed_record).is_err());
}

#[test]
fn weight_edge_rejects_when_no_prepared_session() {
    let capability = GraphBuilderCapability::new();
    let result = capability.call(
        "no-such-instance",
        "weight-edge",
        &[ComponentValue::String("token_embedding".to_string())],
    );
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn weight_edge_rejects_when_called_before_begin_graph() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-8", context());
    let result = capability.call(
        "instance-8",
        "weight-edge",
        &[ComponentValue::String("token_embedding".to_string())],
    );
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn alias_weight_edge_rejects_a_missing_target_or_source_edge() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-9", context());
    capability
        .call(
            "instance-9",
            "begin-graph",
            &[
                ComponentValue::Enum("prefill".to_string()),
                ComponentValue::S64(4),
                ComponentValue::S64(0),
            ],
        )
        .expect("begin-graph succeeds");

    // The target weight edge ("lm_head") was never created via weight-edge.
    let missing_target = capability.call(
        "instance-9",
        "alias-weight-edge",
        &[
            ComponentValue::String("weight.token_embedding".to_string()),
            ComponentValue::String("lm_head".to_string()),
        ],
    );
    assert!(matches!(
        missing_target,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability
        .call(
            "instance-9",
            "weight-edge",
            &[ComponentValue::String("lm_head".to_string())],
        )
        .expect("weight-edge succeeds");

    // The target now exists, but the source edge ("weight.bogus") does not.
    let missing_source = capability.call(
        "instance-9",
        "alias-weight-edge",
        &[
            ComponentValue::String("weight.bogus".to_string()),
            ComponentValue::String("lm_head".to_string()),
        ],
    );
    assert!(matches!(
        missing_source,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn kv_resource_rejects_when_no_prepared_session_and_an_unknown_role() {
    let capability = GraphBuilderCapability::new();
    let no_session = capability.call(
        "no-such-instance",
        "kv-resource",
        &[
            ComponentValue::S64(0),
            ComponentValue::Enum("k".to_string()),
        ],
    );
    assert!(matches!(
        no_session,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability.prepare_session("instance-10", context());
    let unknown_role = capability.call(
        "instance-10",
        "kv-resource",
        &[
            ComponentValue::S64(0),
            ComponentValue::Enum("bogus".to_string()),
        ],
    );
    assert!(matches!(
        unknown_role,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn add_node_rejects_a_malformed_kv_option_and_a_duplicate_node_id() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-11", context());
    capability
        .call(
            "instance-11",
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
            "instance-11",
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

    // arguments[6] must be an Option, not a bare value.
    let malformed_kv = capability.call(
        "instance-11",
        "add-node",
        &[
            ComponentValue::String("n1".to_string()),
            ComponentValue::String("embedding".to_string()),
            ComponentValue::String("Tensor".to_string()),
            ComponentValue::List(vec![]),
            ComponentValue::List(vec![ComponentValue::String(input_edge.clone())]),
            shape(vec![4, 8]),
            ComponentValue::Bool(false),
        ],
    );
    assert!(matches!(
        malformed_kv,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability
        .call(
            "instance-11",
            "add-node",
            &[
                ComponentValue::String("n1".to_string()),
                ComponentValue::String("embedding".to_string()),
                ComponentValue::String("Tensor".to_string()),
                ComponentValue::List(vec![]),
                ComponentValue::List(vec![ComponentValue::String(input_edge.clone())]),
                shape(vec![4, 8]),
                ComponentValue::Option(None),
            ],
        )
        .expect("first add-node with id 'n1' succeeds");

    let duplicate = capability.call(
        "instance-11",
        "add-node",
        &[
            ComponentValue::String("n1".to_string()),
            ComponentValue::String("embedding".to_string()),
            ComponentValue::String("Tensor".to_string()),
            ComponentValue::List(vec![]),
            ComponentValue::List(vec![ComponentValue::String(input_edge.clone())]),
            shape(vec![4, 8]),
            ComponentValue::Option(None),
        ],
    );
    assert!(matches!(
        duplicate,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn add_node_with_an_append_kv_involvement_records_it_on_the_output_edge() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-12", context());
    capability
        .call(
            "instance-12",
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
            "instance-12",
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

    let kv = ComponentValue::Option(Some(Box::new(ComponentValue::Record(vec![
        (
            "resource-id".to_string(),
            ComponentValue::String("qwen.layer0.k".into()),
        ),
        ("append".to_string(), ComponentValue::Bool(true)),
    ]))));
    let node_output = capability
        .call(
            "instance-12",
            "add-node",
            &[
                ComponentValue::String("n1".to_string()),
                ComponentValue::String("attention".to_string()),
                ComponentValue::String("Attention".to_string()),
                ComponentValue::List(vec![]),
                ComponentValue::List(vec![ComponentValue::String(input_edge.clone())]),
                shape(vec![4, 8]),
                kv,
            ],
        )
        .expect("add-node with an append kv-involvement succeeds");
    let ComponentValue::String(node_output) = &node_output[0] else {
        panic!("expected a string edge id");
    };

    let sessions = capability.sessions.lock().unwrap();
    let graph = sessions
        .get("instance-12")
        .expect("session exists")
        .graph
        .as_ref()
        .expect("graph under construction");
    let output_edge = graph
        .edges
        .get(&TensorEdgeId::new(node_output.as_str()))
        .expect("output edge exists");
    let kv_cache = output_edge
        .kv_cache
        .as_ref()
        .expect("kv_cache metadata is recorded");
    assert_eq!(kv_cache.cache_id, "qwen.layer0.k");
    assert_eq!(kv_cache.behavior, GraphKvCacheBehavior::Append);
}

#[test]
fn finish_graph_rejects_when_no_prepared_session_or_before_begin_graph_or_unknown_output() {
    let capability = GraphBuilderCapability::new();
    let no_session = capability.call(
        "no-such-instance",
        "finish-graph",
        &[ComponentValue::String("logits".to_string())],
    );
    assert!(matches!(
        no_session,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability.prepare_session("instance-13", context());
    let before_begin = capability.call(
        "instance-13",
        "finish-graph",
        &[ComponentValue::String("logits".to_string())],
    );
    assert!(matches!(
        before_begin,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability
        .call(
            "instance-13",
            "begin-graph",
            &[
                ComponentValue::Enum("prefill".to_string()),
                ComponentValue::S64(4),
                ComponentValue::S64(0),
            ],
        )
        .expect("begin-graph succeeds");
    let unknown_output = capability.call(
        "instance-13",
        "finish-graph",
        &[ComponentValue::String("does.not.exist".to_string())],
    );
    assert!(matches!(
        unknown_output,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn declare_input_rejects_when_no_prepared_session_or_before_begin_graph() {
    let capability = GraphBuilderCapability::new();
    let no_session = capability.call(
        "no-such-instance",
        "declare-input",
        &[
            ComponentValue::String("token_ids".to_string()),
            shape(vec![4]),
        ],
    );
    assert!(matches!(
        no_session,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));

    capability.prepare_session("instance-14", context());
    let before_begin = capability.call(
        "instance-14",
        "declare-input",
        &[
            ComponentValue::String("token_ids".to_string()),
            shape(vec![4]),
        ],
    );
    assert!(matches!(
        before_begin,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}

#[test]
fn call_rejects_an_unknown_operation() {
    let capability = GraphBuilderCapability::new();
    capability.prepare_session("instance-15", context());
    let result = capability.call("instance-15", "not-a-real-operation", &[]);
    assert!(matches!(
        result,
        Err(ComponentError::CapabilityCallRejected { .. })
    ));
}
