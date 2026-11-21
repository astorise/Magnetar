//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{ShapeDescriptor, TensorDescriptor};

use crate::compute_types::LayoutDescriptor;

#[test]
fn geodesic_residual_requires_matching_tensors_scalar_parameters_and_layer_index() {
    let catalog = initial_operator_catalog();
    let spec = catalog
        .get(&OperatorId::magnetar(
            "geodesic-residual",
            1,
            OperatorFamily::Tensor,
        ))
        .unwrap();
    let tensor = |shape: &[u64], dtype| {
        TensorDescriptor::materialized(
            ShapeDescriptor::new(shape.to_vec()),
            DTypeDescriptor::portable(dtype),
        )
    };
    let x = tensor(&[2, 4], ComputeDType::Float32);
    let scale = tensor(&[1], ComputeDType::Float32);
    let bias = tensor(&[], ComputeDType::Float32);
    let inputs = vec![x.clone(), x.clone(), scale, bias];
    let attributes = BTreeMap::from([("layer_index".into(), OperatorAttributeValue::Integer(7))]);
    assert!(
        spec.validate_invocation(&inputs, std::slice::from_ref(&x), &attributes)
            .is_ok()
    );
    let mut bad = inputs.clone();
    bad[1] = tensor(&[1, 4], ComputeDType::Float32);
    assert!(
        spec.validate_invocation(&bad, std::slice::from_ref(&x), &attributes)
            .is_err()
    );
    bad = inputs.clone();
    bad[2] = tensor(&[2], ComputeDType::Float32);
    assert!(
        spec.validate_invocation(&bad, std::slice::from_ref(&x), &attributes)
            .is_err()
    );
    bad = inputs.clone();
    bad[2] = tensor(&[1], ComputeDType::Float16);
    assert!(
        spec.validate_invocation(&bad, std::slice::from_ref(&x), &attributes)
            .is_err()
    );
    for index in [-1, u32::MAX as i64] {
        let attributes =
            BTreeMap::from([("layer_index".into(), OperatorAttributeValue::Integer(index))]);
        assert!(
            spec.validate_invocation(&inputs, std::slice::from_ref(&x), &attributes)
                .is_err()
        );
    }
    assert!(
        spec.validate_invocation(&inputs, std::slice::from_ref(&x), &BTreeMap::new())
            .is_err()
    );
}
#[test]
fn operator_catalog_contains_required_families_and_initial_operators() {
    let families = OperatorFamily::ALL
        .into_iter()
        .map(OperatorFamily::id)
        .collect::<BTreeSet<_>>();
    assert!(families.contains("attention"));
    let catalog = initial_operator_catalog();
    for name in ["matmul", "attention", "rmsnorm", "rope", "sampling-helper"] {
        assert!(
            catalog
                .operators
                .keys()
                .any(|operator| operator.name() == name)
        );
    }
}

#[test]
fn operator_attributes_reject_provider_device_and_unknown_selectors() {
    let catalog = initial_operator_catalog();
    let matmul = catalog
        .get(&OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        ))
        .unwrap();
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "provider".into(),
        OperatorAttributeValue::String("cuda".into()),
    );
    assert!(matches!(
        matmul.attributes.validate(&attributes),
        Err(OperatorError::OperatorAttributeInvalid { .. })
    ));
}

/// `make-first-native-cuda-hot-path-device-resident` task 3.2: the `rope`
/// Operator's real `OperatorAttributeSchema` (which rejects any attribute
/// it does not declare) must accept the new `head_count` attribute with
/// its correct kind and reject a wrong one, while still rejecting a
/// genuinely unknown attribute name -- the exact gap the audit found (a
/// dispatch setting `head_count` would have failed validation before ever
/// reaching a Kernel).
#[test]
fn rope_schema_accepts_head_count_with_correct_kind_only() {
    let catalog = initial_operator_catalog();
    let rope = catalog
        .get(&OperatorId::magnetar(
            "rope",
            1,
            OperatorFamily::PositionEncoding,
        ))
        .unwrap();
    let base_attrs = || {
        let mut attributes = BTreeMap::new();
        attributes.insert("base".into(), OperatorAttributeValue::Float(10000.0));
        attributes.insert("dimension".into(), OperatorAttributeValue::Integer(2));
        attributes
    };

    // head_count absent -> accepted (today's exact existing behavior).
    assert!(rope.attributes.validate(&base_attrs()).is_ok());

    // head_count = 1 -> accepted.
    let mut attrs = base_attrs();
    attrs.insert("head_count".into(), OperatorAttributeValue::Integer(1));
    assert!(rope.attributes.validate(&attrs).is_ok());

    // head_count > 1 -> accepted.
    let mut attrs = base_attrs();
    attrs.insert("head_count".into(), OperatorAttributeValue::Integer(8));
    assert!(rope.attributes.validate(&attrs).is_ok());

    // Wrong kind (Float instead of Integer) -> rejected.
    let mut attrs = base_attrs();
    attrs.insert("head_count".into(), OperatorAttributeValue::Float(8.0));
    assert!(matches!(
        rope.attributes.validate(&attrs),
        Err(OperatorError::OperatorAttributeInvalid { .. })
    ));

    // A still-unknown attribute name is still rejected -- this schema
    // change did not accidentally loosen validation generally.
    let mut attrs = base_attrs();
    attrs.insert("totally_unknown".into(), OperatorAttributeValue::Integer(1));
    assert!(matches!(
        rope.attributes.validate(&attrs),
        Err(OperatorError::OperatorAttributeInvalid { .. })
    ));
}

#[test]
fn operator_validation_rejects_shape_dtype_layout_errors() {
    let catalog = initial_operator_catalog();
    let matmul = catalog
        .get(&OperatorId::magnetar(
            "matmul",
            1,
            OperatorFamily::LinearAlgebra,
        ))
        .unwrap();
    let a = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let b = TensorDescriptor::materialized(
        ShapeDescriptor::new([4, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let out = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        matmul.validate_invocation(&[a, b], &[out], &BTreeMap::new()),
        Err(OperatorError::ShapeMismatch { .. })
    ));
}

#[test]
fn opaque_layout_is_not_component_visible() {
    assert!(!TensorLayoutKind::ProviderOpaque.component_visible());
}

/// `implement-device-resident-multi-step-cuda-decode` task 1.3:
/// `ShapeRule::RowConcat` accepts a valid row-concatenation (same trailing
/// dimensions, output row count additive) and rejects a trailing-dimension
/// mismatch -- KV-history concatenation's exact contract, mirroring
/// `ShapeRule::RowBroadcastAdd`'s own precedent for testing a purpose-built
/// shape rule directly against the catalog.
#[test]
fn concat_shape_rule_accepts_additive_rows_and_rejects_column_mismatch() {
    let catalog = initial_operator_catalog();
    let concat = catalog
        .get(&OperatorId::magnetar("concat", 1, OperatorFamily::Tensor))
        .unwrap();

    let a = TensorDescriptor::materialized(
        ShapeDescriptor::new([3, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let b = TensorDescriptor::materialized(
        ShapeDescriptor::new([1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let valid_output = TensorDescriptor::materialized(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(
        concat
            .validate_invocation(&[a.clone(), b.clone()], &[valid_output], &BTreeMap::new())
            .is_ok()
    );

    let wrong_row_count_output = TensorDescriptor::materialized(
        ShapeDescriptor::new([3, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        concat.validate_invocation(
            &[a.clone(), b.clone()],
            &[wrong_row_count_output],
            &BTreeMap::new()
        ),
        Err(OperatorError::ShapeMismatch { .. })
    ));

    let mismatched_columns = TensorDescriptor::materialized(
        ShapeDescriptor::new([1, 5]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let output_for_mismatch = TensorDescriptor::materialized(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        concat.validate_invocation(
            &[a, mismatched_columns],
            &[output_for_mismatch],
            &BTreeMap::new()
        ),
        Err(OperatorError::ShapeMismatch { .. })
    ));
}

#[test]
fn operator_layout_kind_maps_every_layout_descriptor_variant() {
    assert_eq!(
        layout_kind(&LayoutDescriptor::Blocked {
            block_dimensions: vec![4],
        }),
        TensorLayoutKind::Blocked
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: None,
            current_length_elements: None,
            logical_to_physical: None,
            append_behavior: None,
        }),
        TensorLayoutKind::Paged
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        }),
        TensorLayoutKind::QuantizedPacked
    );
}

/// Fused matmul Operators take the matmul they replace's attributes, so a
/// fused node keeping explicit ones still validates.
#[test]
fn fused_matmul_operators_accept_the_matmul_attributes() {
    let catalog = initial_operator_catalog();
    let mut attributes = BTreeMap::new();
    attributes.insert("transpose_a".into(), OperatorAttributeValue::Boolean(false));
    attributes.insert("transpose_b".into(), OperatorAttributeValue::Boolean(false));
    for name in ["matmul", "matmul-bias", "matmul-residual", "gated-matmul"] {
        let spec = catalog
            .get(&OperatorId::magnetar(
                name,
                1,
                OperatorFamily::LinearAlgebra,
            ))
            .unwrap_or_else(|error| panic!("{name} is in the catalog: {error:?}"));
        spec.attributes
            .validate(&attributes)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}

/// `extend-portable-operator-attributes`: optional trailing inputs, the
/// `[1]` scalar operand of `mul`, the per-head RMSNorm weight and the new
/// attributes.
#[test]
fn optional_inputs_scalar_mul_and_the_layered_model_attributes_validate() {
    let catalog = initial_operator_catalog();
    let f32_tensor = |shape: &[u64]| {
        TensorDescriptor::materialized(
            ShapeDescriptor::new(shape.to_vec()),
            DTypeDescriptor::portable(ComputeDType::Float32),
        )
    };
    let attributes = |pairs: &[(&str, OperatorAttributeValue)]| -> BTreeMap<_, _> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.clone()))
            .collect()
    };
    let rmsnorm = catalog
        .get(&OperatorId::magnetar(
            "rmsnorm",
            1,
            OperatorFamily::Normalization,
        ))
        .unwrap();
    let epsilon = attributes(&[("epsilon", OperatorAttributeValue::Float(1e-6))]);
    let x = f32_tensor(&[2, 8]);
    // Scale-less: one input. None at all is refused.
    assert!(
        rmsnorm
            .validate_invocation(std::slice::from_ref(&x), std::slice::from_ref(&x), &epsilon)
            .is_ok()
    );
    assert!(matches!(
        rmsnorm.validate_invocation(&[], std::slice::from_ref(&x), &epsilon),
        Err(OperatorError::InputArityInvalid { expected: 1, .. })
    ));
    // Per head: a [head_dimension] weight dividing the row.
    let heads = attributes(&[
        ("epsilon", OperatorAttributeValue::Float(1e-6)),
        ("head_count", OperatorAttributeValue::Integer(2)),
        ("head_dimension", OperatorAttributeValue::Integer(4)),
    ]);
    assert!(
        rmsnorm
            .validate_invocation(
                &[x.clone(), f32_tensor(&[4])],
                std::slice::from_ref(&x),
                &heads
            )
            .is_ok()
    );
    assert!(
        rmsnorm
            .validate_invocation(
                &[x.clone(), f32_tensor(&[3])],
                std::slice::from_ref(&x),
                &heads
            )
            .is_err()
    );

    let mul = catalog
        .get(&OperatorId::magnetar("mul", 1, OperatorFamily::Tensor))
        .unwrap();
    let none = BTreeMap::new();
    assert!(
        mul.validate_invocation(
            &[x.clone(), f32_tensor(&[1])],
            std::slice::from_ref(&x),
            &none
        )
        .is_ok()
    );
    assert!(
        mul.validate_invocation(&[x.clone(), x.clone()], std::slice::from_ref(&x), &none)
            .is_ok()
    );
    assert!(matches!(
        mul.validate_invocation(
            &[x.clone(), f32_tensor(&[8])],
            std::slice::from_ref(&x),
            &none
        ),
        Err(OperatorError::ShapeMismatch { .. })
    ));
    // The scalar may only be the second operand.
    assert!(
        mul.validate_invocation(
            &[f32_tensor(&[1]), x.clone()],
            std::slice::from_ref(&x),
            &none
        )
        .is_err()
    );

    let moe = catalog
        .get(&OperatorId::magnetar(
            "moe-experts",
            1,
            OperatorFamily::LinearAlgebra,
        ))
        .unwrap();
    assert_eq!((moe.min_inputs, moe.inputs), (5, 6));

    for (operator, family, name, value) in [
        (
            "attention",
            OperatorFamily::Attention,
            "scale",
            OperatorAttributeValue::Float(1.0),
        ),
        (
            "rope",
            OperatorFamily::PositionEncoding,
            "partial_rotary_factor",
            OperatorAttributeValue::Float(0.25),
        ),
        (
            "gelu",
            OperatorFamily::Activation,
            "approximate",
            OperatorAttributeValue::String("tanh".into()),
        ),
        (
            "activation",
            OperatorFamily::Activation,
            "cap",
            OperatorAttributeValue::Float(30.0),
        ),
        (
            "activation",
            OperatorFamily::Activation,
            "factor",
            OperatorAttributeValue::Float(0.5),
        ),
        (
            "embedding",
            OperatorFamily::Tensor,
            "scale",
            OperatorAttributeValue::Float(2.0),
        ),
        (
            "moe-experts",
            OperatorFamily::LinearAlgebra,
            "activation",
            OperatorAttributeValue::String("gelu_tanh".into()),
        ),
    ] {
        let spec = catalog
            .get(&OperatorId::magnetar(operator, 1, family))
            .unwrap();
        let rule = spec
            .attributes
            .rules
            .get(name)
            .unwrap_or_else(|| panic!("{operator} defines {name}"));
        assert!(!rule.required, "{operator}.{name} is optional");
        assert_eq!(rule.kind, value.kind(), "{operator}.{name}");
    }
}

/// `dispatch-layered-model-forms`: only the optional forms a Kernel could
/// ignore need a declared feature; the baseline forms and the attributes'
/// default values need none.
#[test]
fn required_kernel_features_name_only_non_baseline_forms() {
    let attributes = |pairs: &[(&str, OperatorAttributeValue)]| {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    let none = BTreeMap::new();
    assert!(required_kernel_features("attention", &none, 3).is_empty());
    assert!(required_kernel_features("rmsnorm", &none, 2).is_empty());
    assert!(required_kernel_features("rope", &none, 1).is_empty());
    assert!(required_kernel_features("moe-experts", &none, 5).is_empty());
    assert!(required_kernel_features("matmul", &none, 2).is_empty());
    assert_eq!(
        required_kernel_features(
            "attention",
            &attributes(&[("scale", OperatorAttributeValue::Float(1.0))]),
            3
        ),
        vec![ATTENTION_SCALE_FEATURE]
    );
    assert_eq!(
        required_kernel_features(
            "embedding",
            &attributes(&[("scale", OperatorAttributeValue::Float(2.0))]),
            2
        ),
        vec![EMBEDDING_SCALE_FEATURE]
    );
    assert_eq!(
        required_kernel_features(
            "rmsnorm",
            &attributes(&[
                ("head_count", OperatorAttributeValue::Integer(2)),
                ("head_dimension", OperatorAttributeValue::Integer(4)),
            ]),
            1
        ),
        vec![RMSNORM_PER_HEAD_FEATURE, RMSNORM_WITHOUT_WEIGHT_FEATURE]
    );
    assert!(
        required_kernel_features(
            "rope",
            &attributes(&[("partial_rotary_factor", OperatorAttributeValue::Float(1.0))]),
            1
        )
        .is_empty()
    );
    assert_eq!(
        required_kernel_features(
            "rope",
            &attributes(&[("partial_rotary_factor", OperatorAttributeValue::Float(0.25))]),
            1
        ),
        vec![ROPE_PARTIAL_ROTARY_FEATURE]
    );
    assert!(
        required_kernel_features(
            "moe-experts",
            &attributes(&[("activation", OperatorAttributeValue::String("silu".into()))]),
            5
        )
        .is_empty()
    );
    assert_eq!(
        required_kernel_features(
            "moe-experts",
            &attributes(&[(
                "activation",
                OperatorAttributeValue::String("gelu_tanh".into())
            )]),
            6
        ),
        vec![MOE_EXPERTS_ACTIVATION_FEATURE, MOE_PER_EXPERT_SCALE_FEATURE]
    );
    let sigmoid = attributes(&[(
        "routing_scoring",
        OperatorAttributeValue::String("sigmoid".into()),
    )]);
    assert_eq!(
        required_kernel_features("moe-experts", &sigmoid, 6),
        vec![MOE_SIGMOID_ROUTING_FEATURE]
    );
    assert_eq!(
        required_kernel_features("moe-experts", &sigmoid, 7),
        vec![MOE_SIGMOID_ROUTING_FEATURE, MOE_PER_EXPERT_SCALE_FEATURE]
    );
    for name in [
        "routing_scale",
        "routing_group_count",
        "routing_group_selected",
        "routing_normalization",
    ] {
        assert_eq!(
            required_kernel_features(
                "moe-experts",
                &attributes(&[(name, OperatorAttributeValue::Boolean(false))]),
                5
            ),
            vec![MOE_SIGMOID_ROUTING_FEATURE]
        );
    }
}

/// `express-latent-attention-forms`: the value width, head-wise `concat`
/// and YaRN each need their Kernel feature; row `concat` needs none.
#[test]
fn latent_attention_forms_require_their_kernel_features() {
    let attributes = |pairs: &[(&str, OperatorAttributeValue)]| {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        required_kernel_features(
            "attention",
            &attributes(&[("value_head_dimension", OperatorAttributeValue::Integer(128))]),
            3
        ),
        vec![ATTENTION_VALUE_HEAD_DIMENSION_FEATURE]
    );
    assert!(required_kernel_features("concat", &BTreeMap::new(), 2).is_empty());
    assert_eq!(
        required_kernel_features(
            "concat",
            &attributes(&[("axis", OperatorAttributeValue::String("heads".into()))]),
            2
        ),
        vec![CONCAT_HEADS_FEATURE]
    );
    let mut yarn = BTreeMap::new();
    RopeYarnFrequencyScaling {
        factor: 40.0,
        original_context_length: 4096,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 1.0,
        truncate: true,
    }
    .insert_attributes(&mut yarn);
    assert_eq!(
        required_kernel_features("rope", &yarn, 1),
        vec![ROPE_YARN_FREQUENCY_SCALING_FEATURE]
    );
    // YaRN attributes round-trip, and the banded parser refuses them.
    assert!(
        RopeYarnFrequencyScaling::from_attributes(&yarn)
            .unwrap()
            .is_some()
    );
    assert!(RopeBandedFrequencyScaling::from_attributes(&yarn).is_err());
}

#[test]
fn concat_across_heads_validates_its_widths() {
    let catalog = initial_operator_catalog();
    let concat = catalog
        .get(&OperatorId::magnetar("concat", 1, OperatorFamily::Tensor))
        .unwrap();
    let tensor = |dimensions: [u64; 2]| {
        TensorDescriptor::materialized(
            ShapeDescriptor::new(dimensions),
            DTypeDescriptor::portable(ComputeDType::Float32),
        )
    };
    let heads = |second: i64| {
        [
            ("axis", OperatorAttributeValue::String("heads".into())),
            ("head_count", OperatorAttributeValue::Integer(16)),
            ("second_head_count", OperatorAttributeValue::Integer(second)),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect::<BTreeMap<_, _>>()
    };
    // q = [q_nope | q_pe]: 16 heads of 128 and 16 of 64.
    assert!(
        concat
            .validate_invocation(
                &[tensor([5, 2048]), tensor([5, 1024])],
                &[tensor([5, 3072])],
                &heads(16)
            )
            .is_ok()
    );
    // k = [k_nope | k_pe]: one shared 64-wide head for all 16.
    assert!(
        concat
            .validate_invocation(
                &[tensor([5, 2048]), tensor([5, 64])],
                &[tensor([5, 3072])],
                &heads(1)
            )
            .is_ok()
    );
    for (inputs, output, attributes) in [
        // A wrong output width.
        (
            [tensor([5, 2048]), tensor([5, 64])],
            tensor([5, 2112]),
            heads(1),
        ),
        // Different row counts.
        (
            [tensor([5, 2048]), tensor([4, 64])],
            tensor([5, 3072]),
            heads(1),
        ),
        // A second input of neither 1 nor 16 heads.
        (
            [tensor([5, 2048]), tensor([5, 64])],
            tensor([5, 2560]),
            heads(4),
        ),
    ] {
        assert!(matches!(
            concat.validate_invocation(&inputs, &[output], &attributes),
            Err(OperatorError::ShapeMismatch { .. })
        ));
    }
    let unknown_axis = [(
        "axis".to_string(),
        OperatorAttributeValue::String("columns".into()),
    )]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert!(matches!(
        concat.validate_invocation(
            &[tensor([5, 2048]), tensor([5, 2048])],
            &[tensor([10, 2048])],
            &unknown_axis
        ),
        Err(OperatorError::OperatorAttributeInvalid { .. })
    ));
}
