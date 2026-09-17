//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::inference_api::InferenceApiError;

#[test]
fn cli_boundary_error_display_is_non_empty_for_every_variant() {
    let variants = vec![
        CliBoundaryError::CliCommandInvalid {
            reason: "bad command".into(),
        },
        CliBoundaryError::CliPromptInputInvalid {
            reason: "bad prompt".into(),
        },
        CliBoundaryError::CliFileReadFailed {
            reason: "file missing".into(),
        },
        CliBoundaryError::CliWorkspaceAccessDenied {
            reason: "policy denied".into(),
        },
        CliBoundaryError::CliGitFailed {
            reason: "git failed".into(),
        },
        CliBoundaryError::CliNetworkDenied {
            reason: "network denied".into(),
        },
        CliBoundaryError::CliSecretUnavailable {
            reason: "secret unavailable".into(),
        },
        CliBoundaryError::CliToolFailed {
            reason: "tool failed".into(),
        },
        CliBoundaryError::CliShellDenied {
            reason: "shell denied".into(),
        },
        CliBoundaryError::CliModelAliasNotFound {
            alias: "my-alias".into(),
        },
        CliBoundaryError::CliModelReferenceInvalid {
            reason: "bad reference".into(),
        },
        CliBoundaryError::CliRuntimeUnavailable {
            reason: "runtime down".into(),
        },
        CliBoundaryError::CliRuntimeRequestFailed(InferenceApiError::ModelLoadingFailed {
            reason: "example".into(),
        }),
        CliBoundaryError::CliStreamInterrupted {
            reason: "stream broke".into(),
        },
        CliBoundaryError::CliCancellationRequested,
        CliBoundaryError::CliDiagnosticsRedacted,
        CliBoundaryError::CliBoundaryViolation {
            capability: "workspace".into(),
        },
        CliBoundaryError::InternalCliError {
            reason: "unexpected".into(),
        },
    ];
    for variant in variants {
        let rendered = variant.to_string();
        assert!(!rendered.is_empty(), "{variant:?} rendered empty");
    }
}

#[test]
fn cli_boundary_rejects_cli_owned_authority_capabilities() {
    for capability in [
        "workspace",
        "filesystem",
        "git",
        "shell",
        "secrets",
        "tool-call",
    ] {
        let error = reject_cli_owned_authority(capability).unwrap_err();
        assert!(matches!(
            error,
            CliBoundaryError::CliBoundaryViolation { .. }
        ));
    }
}

#[test]
fn cli_boundary_allows_inference_scoped_capability() {
    assert!(reject_cli_owned_authority("generation").is_ok());
}

#[test]
fn cli_boundary_error_preserves_wrapped_runtime_error_category() {
    let source = InferenceApiError::SessionNotFound;
    let wrapped = CliBoundaryError::from(source.clone());
    assert_eq!(wrapped.runtime_category(), Some(&source));
}
