use super::*;

/// A small executable-shaped graph whose two history tables grow by one row.
fn graph(position: u64) -> ExecutionGraph {
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new(format!("history-decode-{position}")),
        ExecutionGraphPhase::Decode,
    );
    let ids = TensorEdgeId::new("indices");
    graph = graph.with_edge(TensorEdge::new(
        ids.clone(),
        f32_tensor_descriptor_from_shape(&[position + 1]),
    ));
    for role in ["k", "v"] {
        let input = TensorEdgeId::new(format!("history.{role}"));
        let output = TensorEdgeId::new(format!("updated.{role}"));
        let mut input_edge = TensorEdge::new(
            input.clone(),
            f32_tensor_descriptor_from_shape(&[position.max(1), 2]),
        );
        let mut output_edge = TensorEdge::new(
            output.clone(),
            f32_tensor_descriptor_from_shape(&[position + 1, 2]),
        );
        for (edge, behavior) in [
            (&mut input_edge, GraphKvCacheBehavior::HistoryInput),
            (&mut output_edge, GraphKvCacheBehavior::Output),
        ] {
            edge.kv_cache = Some(GraphKvCacheMetadata {
                cache_id: format!("test.layer0.{role}"),
                behavior,
                paged: false,
                compatibility_key: "history-v1".into(),
            });
        }
        graph = graph
            .with_edge(input_edge)
            .with_edge(output_edge)
            .with_node(
                ExecutionNode::new(
                    ExecutionNodeId::new(format!("update-{role}")),
                    OperatorId::magnetar("embedding", 1, OperatorFamily::Tensor),
                )
                .with_input(ids.clone())
                .with_input(input)
                .with_output(output),
            );
    }
    let state_input = TensorEdgeId::new("state.raw");
    let state_output = TensorEdgeId::new("state.next-raw");
    let mut input_edge = TensorEdge::new(
        state_input.clone(),
        f32_tensor_descriptor_from_shape(&[1, 2]),
    );
    let mut output_edge = TensorEdge::new(
        state_output.clone(),
        f32_tensor_descriptor_from_shape(&[1, 2]),
    );
    for (edge, behavior) in [
        (&mut input_edge, GraphRecurrentStateBehavior::Input),
        (&mut output_edge, GraphRecurrentStateBehavior::Output),
    ] {
        edge.recurrent_state = Some(GraphRecurrentStateMetadata {
            resource_id: "test.layer0.raw".into(),
            layer: 0,
            slot: "raw".into(),
            behavior,
            compatibility_key: "raw-v1".into(),
        });
    }
    graph = graph
        .with_edge(input_edge)
        .with_edge(output_edge)
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("update-state"),
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(state_input)
            .with_output(state_output),
        );
    graph.validate(&initial_operator_catalog()).unwrap();
    graph
}

#[test]
fn advancing_history_plans_preserves_guards_and_rejects_invalid_replacements() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let old_graph = graph(2);
    let next_graph = graph(3);
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &old_graph,
        &instance,
        revision,
        1,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let original = plan.clone();

    let mut changed_schema = next_graph.clone();
    for edge in changed_schema.edges.values_mut() {
        if let Some(history) = &mut edge.kv_cache {
            history.compatibility_key = "other".into();
        }
    }
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &changed_schema,
            3,
            &mut plan,
        )
        .is_err()
    );
    assert_eq!(plan, original);
    let mut changed_state = next_graph.clone();
    for edge in changed_state.edges.values_mut() {
        if let Some(state) = &mut edge.recurrent_state {
            state.compatibility_key = "other".into();
        }
    }
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &changed_state,
            3,
            &mut plan,
        )
        .is_err()
    );
    assert_eq!(plan, original);
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &next_graph,
            4,
            &mut plan,
        )
        .is_err()
    );
    assert_eq!(plan, original);

    let mut altered = original.clone();
    altered
        .guards
        .push(PlanGuard::Phase(PreparedExecutionPhase::Prefill));
    let before = altered.clone();
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &next_graph,
            3,
            &mut altered,
        )
        .is_err()
    );
    assert_eq!(altered, before);
    let mut stale_revision = original.clone();
    stale_revision.scope.model_instance_revision = Some(revision + 1);
    let before = stale_revision.clone();
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &next_graph,
            3,
            &mut stale_revision,
        )
        .is_err()
    );
    assert_eq!(stale_revision, before);

    let mut invalidated = original.clone();
    invalidated
        .transition_to(PreparedExecutionPlanState::Invalidated)
        .unwrap();
    let before = invalidated.clone();
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &next_graph,
            3,
            &mut invalidated,
        )
        .is_err()
    );
    assert_eq!(invalidated, before);

    let lease = plan.acquire_lease().unwrap();
    let before = plan.clone();
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &old_graph,
            &next_graph,
            3,
            &mut plan,
        )
        .is_err()
    );
    assert_eq!(plan, before);
    plan.release_lease(lease).unwrap();

    advance_first_native_history_decode_plan(
        &runtime,
        &instance,
        &old_graph,
        &next_graph,
        3,
        &mut plan,
    )
    .unwrap();
    assert_eq!(plan.generation.value(), 2);
    assert_eq!(
        plan.graph_fingerprint,
        semantic_graph_fingerprint(&next_graph)
    );
    assert_eq!(plan.scope, original.scope);
    assert_eq!(plan.guards, original.guards);
    assert!(plan.observations.iter().any(|event| {
        event.kind == PreparedExecutionPlanObservationKind::PlanGenerationPromoted
    }));

    let next_next = graph(4);
    advance_first_native_history_decode_plan(
        &runtime,
        &instance,
        &next_graph,
        &next_next,
        4,
        &mut plan,
    )
    .unwrap();
    assert_eq!(plan.generation.value(), 3);
    let before = plan.clone();
    runtime
        .kernel_registry_mut()
        .invalidate_provider(&provider, "test Provider invalidation");
    assert!(
        advance_first_native_history_decode_plan(
            &runtime,
            &instance,
            &next_next,
            &graph(5),
            5,
            &mut plan,
        )
        .is_err()
    );
    assert_eq!(plan, before);
}

#[test]
fn advanced_plans_execute_with_committed_history_and_recurrent_state() {
    let fixture = e2e_fixture().unwrap();
    let mut runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut runtime).unwrap();
    let revision = require_ready_first_native_instance(&runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let executor = resolve_kernel_execution_provider(&runtime, &provider).unwrap();
    let cache = runtime
        .create_kv_cache(KvCache::new(
            KvCacheId::new("history-plan-session").unwrap(),
            KvCacheScope::Session,
            KvCacheCompatibility::new(
                GenerationModelReference::ModelInstance(instance.clone()),
                TokenizerId::new("tokenizer").unwrap(),
            ),
            KvCacheLayoutMetadata::contiguous(1, 1, 2, 8, ComputeDType::Float32),
        ))
        .unwrap();
    let mut previous_graph = graph(0);
    let mut plan = prepare_first_native_plan_for_graph(
        &runtime,
        &previous_graph,
        &instance,
        revision,
        1,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let mut history = None;
    for position in 0..4 {
        let current = graph(position);
        if position > 0 {
            advance_first_native_history_decode_plan(
                &runtime,
                &instance,
                &previous_graph,
                &current,
                position,
                &mut plan,
            )
            .unwrap();
        }
        let input = HostTensor::new([position + 1], vec![0.0; (position + 1) as usize]).unwrap();
        let (_, _, pending, _) = execute_first_native_graph_materializing(
            &mut runtime,
            &fixture,
            &instance,
            &cache,
            &current,
            &mut plan,
            BTreeMap::from([(TensorEdgeId::new("indices"), input)]),
            history.as_ref(),
            Some(position),
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            None,
        )
        .unwrap();
        let mut transaction = KvUpdateTransaction {
            provider_binding: provider.clone(),
            executor: executor.clone(),
            promoted: Vec::new(),
        };
        for (&layer, state) in &pending {
            transaction
                .promote_layer(&mut runtime, &cache, layer, state)
                .unwrap();
        }
        let committed = transaction.commit(&mut runtime, &cache).unwrap();
        for resource in [&committed[&0].k, &committed[&0].v] {
            assert_eq!(
                executor.read_tensor(resource).unwrap().shape,
                [position + 1, 2]
            );
        }
        assert_eq!(
            runtime.kv_cache(&cache).unwrap().recurrent_resources.len(),
            1
        );
        history = Some(committed);
        previous_graph = current;
    }
    super::super::recurrent_state::release_recurrent_state(&mut runtime, &executor, &cache);
    runtime.release_kv_cache(&cache).unwrap();
}
