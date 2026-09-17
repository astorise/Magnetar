//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
#[test]
fn model_instance_readiness_fails_when_kernel_preparation_failed() {
    let mut checks = ModelInstanceReadinessChecks::default();
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Ready);
    assert!(checks.validate().is_ok());

    checks.kernel_preparation_ready = false;
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Failed);
    assert!(matches!(
        checks.validate(),
        Err(ModelInstanceError::ModelInstanceKernelPreparationFailed)
    ));
}
