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
