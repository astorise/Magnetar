//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::ProviderBinding;
use crate::capability::CapabilityVersion;
use crate::kernel::{KernelId, KernelImplementationFamily, KernelOperatorVersionRange};
use crate::kernel_performance_model::KernelPerformanceFeedbackMode;
use crate::operator::{OperatorFamily, OperatorId};
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

fn conformance_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("kernel-artifact-test-provider"),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::TestFixture,
    )
}

#[test]
fn reproducible_kernel_selection_forces_pinned_performance_feedback() {
    // Implements "Reproducible Mode Prevents Adaptation" (proposal, define-
    // kernel-performance-model-and-adaptive-feedback-contract): a pinned
    // Kernel selection always yields Pinned feedback mode regardless of the
    // Model Instance's requested mode, and dynamic selection leaves the
    // requested mode untouched.
    let policy = ModelInstancePolicy {
        performance_feedback: KernelPerformanceFeedbackMode::Adaptive,
        ..ModelInstancePolicy::default()
    };

    assert_eq!(
        effective_performance_feedback_mode(&policy, &KernelSelectionPolicy::Dynamic),
        KernelPerformanceFeedbackMode::Adaptive
    );

    let pinned = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(
        conformance_kernel_id("attn"),
        "digest-a",
    ));
    assert_eq!(
        effective_performance_feedback_mode(&policy, &pinned),
        KernelPerformanceFeedbackMode::Pinned
    );
}
