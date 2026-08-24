use magnetar_runtime::{
    ComputeDType, DTypeDescriptor, ExecutionGraph, ExecutionGraphId, ExecutionGraphPhase,
    ExecutionNode, ExecutionNodeId, FirstScopeDTypeTier, FirstScopeErrorCode, FirstScopeLayoutTier,
    FirstScopeObservation, FirstScopeObservationKind, FutureOptimizedOperator, KernelMemoryClass,
    KernelResource, KernelSelectionRequest, LayoutDescriptor, OperatorAttributeValue,
    OperatorFamily, OperatorId, OperatorRequirement, OperatorScopeTier, ProviderBinding,
    ResourceAffinity, ShapeDescriptor, TensorDescriptor, TensorEdge, TensorEdgeId,
    TensorLayoutKind, TensorResourceDescriptor, TensorResourceId, first_operator_scope,
    first_scope_dtype_tier, first_scope_layout_tier, first_scope_required_fixture_names,
    future_optimized_operators, operator_scope_entry, reference_cpu_kernel_advertisements,
    validate_first_scope_dtype, validate_first_scope_graph,
    validate_first_scope_kernel_selection_request, validate_first_scope_layout,
    validate_model_component_first_scope_requirements,
    validate_no_placeholder_kernel_advertisements, validate_reference_cpu_required_kernel_coverage,
    validate_required_now_operator,
};
use std::collections::BTreeSet;

#[test]
fn scope_classifies_required_placeholder_unsupported_and_future_optimized() {
    let entries = first_operator_scope();
    assert!(entries.iter().any(|entry| {
        entry.name == "embedding"
            && entry.tier == OperatorScopeTier::RequiredNow
            && entry.requires_reference_cpu_kernel
            && entry.requires_conformance_fixture
    }));
    assert!(entries.iter().any(|entry| {
        entry.name == "paged-attention" && entry.tier == OperatorScopeTier::Placeholder
    }));
    assert!(entries.iter().any(|entry| {
        entry.name == "flash-attention" && entry.tier == OperatorScopeTier::ExplicitlyUnsupported
    }));

    let future = future_optimized_operators()
        .iter()
        .map(|operator| operator.id())
        .collect::<BTreeSet<_>>();
    assert!(future.contains(FutureOptimizedOperator::FlashAttention.id()));
    assert!(future.contains(FutureOptimizedOperator::PagedAttention.id()));
    assert!(future.contains(FutureOptimizedOperator::QuantizedMatmul.id()));
}

#[test]
fn first_decoder_required_now_set_is_minimal_and_platform_neutral() {
    let decoder_required = first_operator_scope()
        .iter()
        .filter(|entry| entry.required_for_first_decoder_model)
        .map(|entry| entry.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        decoder_required,
        BTreeSet::from([
            "add",
            "attention",
            "embedding",
            "matmul",
            "mul",
            "residual-add",
            "rmsnorm",
            "rope",
            "silu",
            "softmax",
        ])
    );
    assert!(
        first_operator_scope()
            .iter()
            .all(|entry| entry.id().namespace() == "magnetar:operator")
    );
}

#[test]
fn scope_validation_reports_placeholder_unsupported_and_out_of_scope_errors() {
    let required = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    assert_eq!(
        validate_required_now_operator(&required).unwrap().tier,
        OperatorScopeTier::RequiredNow
    );

    let placeholder = OperatorId::magnetar("paged-attention", 1, OperatorFamily::Attention);
    assert_eq!(
        validate_required_now_operator(&placeholder)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::OperatorPlaceholderOnly
    );

    let unsupported = OperatorId::magnetar("moe-dispatch", 1, OperatorFamily::Control);
    assert_eq!(
        validate_required_now_operator(&unsupported)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::OperatorExplicitlyUnsupported
    );

    let unknown = OperatorId::magnetar("fft", 1, OperatorFamily::LinearAlgebra);
    assert_eq!(
        validate_required_now_operator(&unknown).unwrap_err().code,
        FirstScopeErrorCode::OperatorOutOfFirstScope
    );
}

#[test]
fn dtype_and_layout_scope_accept_required_baseline_and_reject_silent_conversion() {
    assert_eq!(
        first_scope_dtype_tier(ComputeDType::Float32),
        FirstScopeDTypeTier::Required
    );
    assert_eq!(
        first_scope_dtype_tier(ComputeDType::Float16),
        FirstScopeDTypeTier::Placeholder
    );
    assert_eq!(
        validate_first_scope_dtype(ComputeDType::BrainFloat16)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::DTypeUnsupported
    );

    assert_eq!(
        first_scope_layout_tier(TensorLayoutKind::Contiguous),
        FirstScopeLayoutTier::Required
    );
    assert_eq!(
        first_scope_layout_tier(TensorLayoutKind::Paged),
        FirstScopeLayoutTier::Placeholder
    );
    assert_eq!(
        validate_first_scope_layout(TensorLayoutKind::Paged)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::LayoutUnsupported
    );
}

#[test]
fn reference_cpu_covers_required_now_and_does_not_advertise_placeholders() {
    let advertisements = reference_cpu_kernel_advertisements();
    validate_reference_cpu_required_kernel_coverage(&advertisements).unwrap();
    validate_no_placeholder_kernel_advertisements(&advertisements).unwrap();

    let advertised = advertisements
        .iter()
        .map(|advertisement| advertisement.implemented_operator.name())
        .collect::<BTreeSet<_>>();
    for fixture in first_scope_required_fixture_names() {
        assert!(
            advertised.contains(fixture),
            "required fixture {fixture} must have Reference CPU coverage"
        );
    }
}

#[test]
fn observations_are_redacted_metadata_only() {
    let operator = OperatorId::magnetar("attention", 1, OperatorFamily::Attention);
    let observation = FirstScopeObservation::new(FirstScopeObservationKind::OperatorRejected)
        .with_operator(operator.clone())
        .with_redacted_metadata("reason", "first-scope-attribute-unsupported");

    assert_eq!(observation.operator, Some(operator));
    assert_eq!(observation.kind.id(), "first-scope-operator-rejected");
    assert_eq!(
        observation.redacted_metadata,
        vec![("reason", "first-scope-attribute-unsupported".into())]
    );
}

#[test]
fn gelu_is_available_but_not_required_for_first_decoder_baseline() {
    let gelu = OperatorId::magnetar("gelu", 1, OperatorFamily::Activation);
    let entry = operator_scope_entry(&gelu).expect("gelu scope entry should exist");
    assert_eq!(entry.tier, OperatorScopeTier::RequiredForFirstDecoderModel);
    assert!(!entry.required_for_first_decoder_model);
    assert!(!entry.requires_reference_cpu_kernel);
}

#[test]
fn graph_first_scope_validation_accepts_unfused_mlp_and_rejects_unsupported_variant() {
    let input = TensorEdgeId::new("input");
    let output = TensorEdgeId::new("output");
    let node = ExecutionNodeId::new("silu");
    let tensor = TensorDescriptor::new(
        ShapeDescriptor::new([1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    );
    let graph = ExecutionGraph::new(ExecutionGraphId::new("mlp"), ExecutionGraphPhase::Decode)
        .with_edge(TensorEdge::new(input.clone(), tensor.clone()).with_consumer(node.clone()))
        .with_edge(TensorEdge::new(output.clone(), tensor).with_producer(node.clone()))
        .with_node(
            ExecutionNode::new(
                node,
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(input)
            .with_output(output),
        );
    validate_first_scope_graph(&graph).unwrap();

    let bad = ExecutionGraph::new(ExecutionGraphId::new("flash"), ExecutionGraphPhase::Decode)
        .with_node(ExecutionNode::new(
            ExecutionNodeId::new("flash-attn"),
            OperatorId::magnetar("flash-attention", 1, OperatorFamily::Attention),
        ));
    assert_eq!(
        validate_first_scope_graph(&bad).unwrap_err().code,
        FirstScopeErrorCode::OperatorExplicitlyUnsupported
    );
}

#[test]
fn rope_and_attention_attributes_are_checked_explicitly() {
    let rope = ExecutionGraph::new(ExecutionGraphId::new("rope"), ExecutionGraphPhase::Decode)
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("rope"),
                OperatorId::magnetar("rope", 1, OperatorFamily::PositionEncoding),
            )
            .with_attribute(
                "position_mode",
                OperatorAttributeValue::String("dynamic".into()),
            ),
        );
    assert_eq!(
        validate_first_scope_graph(&rope).unwrap_err().code,
        FirstScopeErrorCode::AttributeUnsupported
    );

    let attention = ExecutionGraph::new(
        ExecutionGraphId::new("attention"),
        ExecutionGraphPhase::Decode,
    )
    .with_node(
        ExecutionNode::new(
            ExecutionNodeId::new("attention"),
            OperatorId::magnetar("attention", 1, OperatorFamily::Attention),
        )
        .with_attribute(
            "attention_mask_kind",
            OperatorAttributeValue::String("block-sparse".into()),
        ),
    );
    assert_eq!(
        validate_first_scope_graph(&attention).unwrap_err().code,
        FirstScopeErrorCode::AttributeUnsupported
    );
}

#[test]
fn model_component_requirements_can_use_scoped_alternatives() {
    let mut requirement = OperatorRequirement::new(OperatorId::magnetar(
        "flash-attention",
        1,
        OperatorFamily::Attention,
    ));
    requirement.alternatives.push(OperatorId::magnetar(
        "attention",
        1,
        OperatorFamily::Attention,
    ));
    validate_model_component_first_scope_requirements(&[requirement]).unwrap();

    let bad = OperatorRequirement::new(OperatorId::magnetar(
        "moe-dispatch",
        1,
        OperatorFamily::Control,
    ));
    assert_eq!(
        validate_model_component_first_scope_requirements(&[bad])
            .unwrap_err()
            .code,
        FirstScopeErrorCode::OperatorExplicitlyUnsupported
    );
}

#[test]
fn kernel_selection_request_first_scope_rejects_placeholder_operator_dtype_and_layout() {
    let provider = ProviderBinding::new("reference-cpu");
    let affinity = ResourceAffinity::new(magnetar_runtime::FallbackClass::Transparent)
        .with_provider(provider.clone());
    let input = KernelResource::new(
        TensorResourceDescriptor::new(
            TensorResourceId::new("input"),
            TensorDescriptor::new(
                ShapeDescriptor::new([1, 4]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
            affinity.clone(),
        ),
        KernelMemoryClass::Host,
    );

    let mut request = KernelSelectionRequest::new(
        "first-scope",
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        affinity,
    )
    .with_input(input);
    request.dtype_requirements.insert(ComputeDType::Float16);
    assert_eq!(
        validate_first_scope_kernel_selection_request(&request)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::DTypeUnsupported
    );

    request.dtype_requirements.clear();
    request.layout_requirements.insert(TensorLayoutKind::Paged);
    assert_eq!(
        validate_first_scope_kernel_selection_request(&request)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::LayoutUnsupported
    );

    request.layout_requirements.clear();
    request.operator = OperatorId::magnetar("paged-attention", 1, OperatorFamily::Attention);
    assert_eq!(
        validate_first_scope_kernel_selection_request(&request)
            .unwrap_err()
            .code,
        FirstScopeErrorCode::OperatorPlaceholderOnly
    );
}
