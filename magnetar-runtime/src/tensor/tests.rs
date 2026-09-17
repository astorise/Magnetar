//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{FallbackClass, ResourceAffinity};
use crate::compute::{
    ComputeDType, DTypeDescriptor, ShapeDescriptor, TensorDescriptor, TensorResourceId,
};
use crate::memory::{MemoryPlacement, TensorResidency};

fn tensor_resource_for_test(id: &str) -> TensorResource {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let residency = TensorResidency::new(
        TensorResourceId::new(id),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    TensorResource::new(TensorResourceId::new(id), descriptor, residency)
}

#[test]
fn tensor_resource_released_is_rejected_for_use() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-3");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.mark_ready().unwrap();
    resource
        .transition_to(TensorLifecycleState::Released)
        .unwrap();
    assert_eq!(
        resource.ensure_usable().unwrap_err(),
        TensorError::ResourceReleased
    );
}

#[test]
fn tensor_mutability_denies_mutation_of_immutable_resource() {
    let error =
        validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, true).unwrap_err();
    assert!(matches!(error, TensorError::MutabilityViolation { .. }));
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Mutable, true).is_ok());
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, false).is_ok());
}

#[test]
fn tensor_aliasing_requires_in_place_kernel_support() {
    let error =
        validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, false).unwrap_err();
    assert!(matches!(error, TensorError::AliasingViolation { .. }));
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, true).is_ok());
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::NoAlias, false).is_ok());
}

#[test]
fn tensor_memory_class_validation_rejects_unsupported_class() {
    let error = validate_memory_class_for_kernel(
        TensorMemoryClass::Device,
        &[TensorMemoryClass::Host, TensorMemoryClass::PinnedHost],
    )
    .unwrap_err();
    assert!(matches!(error, TensorError::MemoryClassUnsupported { .. }));
    assert!(validate_memory_class_for_kernel(TensorMemoryClass::Device, &[]).is_ok());
    assert!(
        validate_memory_class_for_kernel(TensorMemoryClass::Host, &[TensorMemoryClass::Host])
            .is_ok()
    );
}

#[test]
fn tensor_error_and_observation_redact_backend_diagnostics() {
    let error = TensorError::resource_invalid("native handle=0xdeadbeef");
    assert_eq!(
        error.to_string(),
        "tensor resource invalid: [redacted backend diagnostic]"
    );
    let observation = TensorObservation::new(TensorObservationKind::ResourceReady)
        .with_message("C:\\weights\\model.bin");
    assert_eq!(observation.message, "[redacted backend diagnostic]");
}
