//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::{ShapeDescriptor, TensorDescriptor};

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
