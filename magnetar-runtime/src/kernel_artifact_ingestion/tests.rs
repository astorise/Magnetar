//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
#[test]
fn ingestion_state_machine_rejects_illegal_transitions() {
    assert!(IngestionState::Created.can_transition_to(IngestionState::Receiving));
    assert!(!IngestionState::Created.can_transition_to(IngestionState::Committed));
    assert!(!IngestionState::Committed.can_transition_to(IngestionState::Accepted));
    assert!(IngestionState::Committed.is_terminal());
    assert!(!IngestionState::Staged.is_terminal());
}

#[test]
fn ingestion_audit_record_supports_release_evidence_traceability() {
    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    transaction.mark_receiving().unwrap();
    transaction.mark_staged().unwrap();
    transaction.mark_validating().unwrap();
    transaction.mark_policy_evaluating().unwrap();
    transaction.mark_accepted().unwrap();

    let record = KernelIngestionAuditRecord::from_transaction(&transaction)
        .with_manifest_digest("sha256:aaaa")
        .with_redacted_metadata("locator", "https://user:secret@internal/path");
    assert_eq!(
        record.release_evidence_reference(),
        Some(("policy-v1".to_string(), "sha256:aaaa".to_string()))
    );
    assert_ne!(
        record.redacted_metadata.get("locator").map(String::as_str),
        Some("https://user:secret@internal/path")
    );
}

#[test]
fn ingestion_error_ids_are_stable_and_displayable() {
    let cases: &[(IngestionError, &str)] = &[
        (
            IngestionError::StateInvalid { reason: "x".into() },
            "kernel-ingestion-state-invalid",
        ),
        (
            IngestionError::CommitConflict,
            "kernel-ingestion-commit-conflict",
        ),
        (
            IngestionError::ExternalDigestMismatch {
                expected: "a".into(),
                actual: "b".into(),
            },
            "kernel-ingestion-external-digest-mismatch",
        ),
        (
            IngestionError::ManualApprovalCannotBypassIntegrity,
            "kernel-ingestion-manual-approval-cannot-bypass-integrity",
        ),
        (
            IngestionError::ArtifactRevoked {
                digest: "sha256:aaa".into(),
            },
            "kernel-ingestion-artifact-revoked",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.id(), *expected_id);
        assert!(!error.to_string().is_empty());
    }
}
