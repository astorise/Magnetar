//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
#[test]
fn artifact_trust_is_only_ever_policy_controlled() {
    assert_eq!(
        evaluate_artifact_trust(false),
        KernelArtifactTrust::Untrusted
    );
    assert_eq!(evaluate_artifact_trust(true), KernelArtifactTrust::Trusted);
    assert_eq!(
        KernelArtifactTrust::default(),
        KernelArtifactTrust::Untrusted
    );
}

#[test]
fn hot_path_denies_compilation_cold_path_allows_it() {
    assert!(matches!(
        reject_hot_path_compilation(
            KernelArtifactPath::Hot,
            KernelArtifactColdPathOperation::Compilation
        ),
        Err(KernelArtifactError::HotPathCompilationDenied { .. })
    ));
    assert!(
        reject_hot_path_compilation(
            KernelArtifactPath::Cold,
            KernelArtifactColdPathOperation::Compilation
        )
        .is_ok()
    );
}

#[test]
fn lazy_preparation_requires_explicit_policy_and_admission_state() {
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), false, false).is_ok());
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), true, true),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, false),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, true).is_ok());
}

#[test]
fn inference_requests_reject_kernel_artifact_management_fields() {
    for field in KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS {
        assert!(reject_inference_request_artifact_field(field).is_err());
    }
    assert!(reject_inference_request_artifact_field("prompt").is_ok());
}
