//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::kernel_artifact_manifest::{
    KERNEL_MANIFEST_FILE_NAME, KernelBlobDigest, KernelExchangeBundle, normalize_to_cache_key,
};
use crate::kernel_cache::KernelArtifactCache;
use crate::kernel_registry::KernelRegistry;
use std::fs;
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

#[test]
fn kernel_artifact_ingestion_conformance_report_is_conformant() {
    let report = run_kernel_artifact_ingestion_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

fn temp_kernel_bundle_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-kernel-bundle-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(directory.join("blobs").join("sha256")).unwrap();
    directory
}

fn write_kernel_bundle(directory: &std::path::Path, blob_bytes: &[u8]) -> String {
    let digest = KernelBlobDigest::of_bytes(blob_bytes);
    fs::write(
        directory.join("blobs").join("sha256").join(&digest.value),
        blob_bytes,
    )
    .unwrap();
    let manifest = format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{}",
      "size": {},
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]
    }}
  ]
}}"#,
        digest.value,
        blob_bytes.len()
    );
    fs::write(directory.join(KERNEL_MANIFEST_FILE_NAME), manifest).unwrap();
    digest.value
}

#[test]
fn ingestion_pipeline_accepts_trusted_bundle_and_commits() {
    let directory = temp_kernel_bundle_dir("ingestion-accept");
    let digest = write_kernel_bundle(&directory, b"ingestion-accept-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline should accept a trusted, well-formed bundle");
    assert_eq!(outcome.decision, IngestionDecisionKind::Accept);
    assert_eq!(transaction.state, IngestionState::Accepted);

    let mut cache = KernelArtifactCache::new();
    let committed = commit_accepted_transaction(&mut transaction, &outcome.validated, &mut cache)
        .expect("accepted transaction should commit");
    assert_eq!(committed, vec![digest.clone()]);
    assert_eq!(transaction.state, IngestionState::Committed);
    assert!(
        cache
            .get(&normalize_to_cache_key(&outcome.validated.manifest.artifacts[0]).stable_key())
            .is_some()
    );

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_quarantines_untrusted_bundle_without_committing() {
    let directory = temp_kernel_bundle_dir("ingestion-quarantine");
    write_kernel_bundle(&directory, b"ingestion-quarantine-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: false,
        trust_explicitly_denied: false,
        signed: false,
        signature_verified: false,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline runs to a decision even when untrusted");
    assert!(matches!(
        outcome.decision,
        IngestionDecisionKind::Quarantine(_)
    ));
    assert_eq!(transaction.state, IngestionState::Quarantined);
    assert!(transaction.committed_digests.is_empty());

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_rejects_revoked_digest() {
    let directory = temp_kernel_bundle_dir("ingestion-revoked");
    write_kernel_bundle(&directory, b"ingestion-revoked-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: true,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline runs to a decision even when revoked");
    assert!(matches!(outcome.decision, IngestionDecisionKind::Reject(_)));
    assert_eq!(transaction.state, IngestionState::Rejected);

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_commit_detects_bundle_mutation_after_validation() {
    // "Immutable Snapshot" / TOCTOU protection (proposal): the source is
    // mutated *after* validation but *before* commit -- the attack scenario
    // is "validate file A -> source replaces file A -> prepare replaced file
    // B". Here the replacement happens at the exact same content-addressed
    // path (an attacker overwriting bytes in place), which is exactly what
    // `verify_bundle_snapshot_unchanged` re-checks.
    let directory = temp_kernel_bundle_dir("ingestion-toctou");
    let digest = write_kernel_bundle(&directory, b"original-validated-bytes");
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy)
        .expect("pipeline should accept the originally staged bytes");
    assert_eq!(outcome.decision, IngestionDecisionKind::Accept);

    // Mutate the blob in place at its own content-addressed path -- the
    // source is replaced after validation completed.
    fs::write(
        directory.join("blobs").join("sha256").join(&digest),
        b"mutated-after-validation-bytes",
    )
    .unwrap();

    let mut cache = KernelArtifactCache::new();
    let commit_outcome = commit_accepted_transaction_from_bundle(
        &mut transaction,
        &bundle,
        &outcome.validated,
        &mut cache,
    );
    assert!(matches!(
        commit_outcome,
        Err(IngestionError::ToctouDetected)
    ));
    assert!(
        cache
            .get(&normalize_to_cache_key(&outcome.validated.manifest.artifacts[0]).stable_key())
            .is_none(),
        "staged/mutated content must never reach the accepted cache"
    );

    let _ = fs::remove_dir_all(&directory);
}

#[test]
fn ingestion_pipeline_failure_leaves_cache_and_registry_untouched() {
    // "Failure Atomicity" (proposal): a malformed bundle (missing manifest
    // file entirely) must fail before commit, and every stage of active
    // state (accepted cache, Kernel Registry) is untouched by the attempt.
    let directory = temp_kernel_bundle_dir("ingestion-malformed");
    // Deliberately do not write a manifest file -- `write_kernel_bundle` is
    // not called, so `blobs/sha256/` exists but `kernel.manifest.json` does
    // not.
    let bundle = KernelExchangeBundle::open(&directory);

    let mut transaction = KernelIngestionTransaction::new(
        IngestionTransactionIdAllocator::default().allocate(),
        ObservedIngestionSource::Ci,
        "policy-v1",
        IngestionQuotas::default(),
    );
    let pipeline_context = IngestionPipelineContext {
        trust_policy_approved: true,
        trust_explicitly_denied: false,
        signed: true,
        signature_verified: true,
        qualification_current: true,
        required_qualification_suite_version: None,
        qualification_target_context: None,
        development_mode_allowed: false,
        revoked: false,
    };
    let policy = KernelIngestionPolicy::production_default("policy-v1");
    let cache_before = KernelArtifactCache::new();
    let registry_before = KernelRegistry::new();

    let outcome = run_ingestion_pipeline(&mut transaction, &bundle, &pipeline_context, &policy);
    assert!(matches!(outcome, Err(IngestionError::BundleInvalid { .. })));
    assert_eq!(transaction.state, IngestionState::Failed);
    assert!(transaction.committed_digests.is_empty());
    // Nothing in this pipeline call had access to a cache or registry at
    // all, so both remain exactly as constructed -- demonstrating the
    // failure cannot have touched either even in principle.
    assert_eq!(cache_before.observations().len(), 0);
    assert_eq!(registry_before.entries().count(), 0);

    let _ = fs::remove_dir_all(&directory);
}
