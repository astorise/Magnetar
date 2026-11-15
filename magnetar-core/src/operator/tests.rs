//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{ShapeDescriptor, TensorDescriptor};

use crate::compute_types::LayoutDescriptor;
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
