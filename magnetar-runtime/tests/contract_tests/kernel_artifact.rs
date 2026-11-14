use magnetar_runtime::KernelArtifactError;

#[test]
fn kernel_artifact_error_ids_match_proposal_error_model() {
    let cases: &[(KernelArtifactError, &str)] = &[
        (
            KernelArtifactError::ArtifactInvalid { reason: "x".into() },
            "kernel-artifact-invalid",
        ),
        (
            KernelArtifactError::DigestMismatch {
                expected: "a".into(),
                found: "b".into(),
            },
            "kernel-artifact-digest-mismatch",
        ),
        (
            KernelArtifactError::FormatUnsupported { format: "x".into() },
            "kernel-artifact-format-unsupported",
        ),
        (
            KernelArtifactError::Untrusted {
                artifact: "x".into(),
            },
            "kernel-artifact-untrusted",
        ),
        (
            KernelArtifactError::OperatorIncompatible { reason: "x".into() },
            "kernel-artifact-operator-incompatible",
        ),
        (
            KernelArtifactError::DTypeIncompatible { reason: "x".into() },
            "kernel-artifact-dtype-incompatible",
        ),
        (
            KernelArtifactError::LayoutIncompatible { reason: "x".into() },
            "kernel-artifact-layout-incompatible",
        ),
        (
            KernelArtifactError::ShapeIncompatible { reason: "x".into() },
            "kernel-artifact-shape-incompatible",
        ),
        (
            KernelArtifactError::TargetIncompatible { target: "x".into() },
            "kernel-artifact-target-incompatible",
        ),
        (
            KernelArtifactError::ProviderIncompatible {
                provider: "x".into(),
            },
            "kernel-artifact-provider-incompatible",
        ),
        (
            KernelArtifactError::DriverIncompatible { reason: "x".into() },
            "kernel-artifact-driver-incompatible",
        ),
        (
            KernelArtifactError::CompilerIncompatible { reason: "x".into() },
            "kernel-artifact-compiler-incompatible",
        ),
        (
            KernelArtifactError::PreparationUnavailable { reason: "x".into() },
            "kernel-preparation-unavailable",
        ),
        (
            KernelArtifactError::PreparationFailed { reason: "x".into() },
            "kernel-preparation-failed",
        ),
        (
            KernelArtifactError::PreparedHandleInvalid { reason: "x".into() },
            "kernel-prepared-handle-invalid",
        ),
        (
            KernelArtifactError::PreparedGenerationInUse { generation: 1 },
            "kernel-prepared-generation-in-use",
        ),
        (
            KernelArtifactError::PreparedDestroyFailed { reason: "x".into() },
            "kernel-prepared-destroy-failed",
        ),
        (
            KernelArtifactError::PreparedNotReady { kernel: "x".into() },
            "kernel-prepared-not-ready",
        ),
        (
            KernelArtifactError::HotPathCompilationDenied {
                operation: "x".into(),
            },
            "kernel-hot-path-compilation-denied",
        ),
        (
            KernelArtifactError::InternalKernelArtifactError { reason: "x".into() },
            "internal-kernel-artifact-error",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.id(), *expected_id);
        assert!(!error.to_string().is_empty());
    }
}
