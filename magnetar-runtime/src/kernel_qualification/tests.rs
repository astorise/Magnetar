//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::kernel_artifact::CompiledKernelArtifactId;

#[test]
fn qualification_status_transitions_reject_skipping_qualifying() {
    let mut record = QualificationRecord::new(
        QualificationIdentity::new(
            CompiledKernelArtifactId::from_digest("digest"),
            1,
            "suite-1",
            "sm90",
            "provider-1",
        ),
        QualificationProfile::baseline_correctness(),
        reference_cpu_oracle("1"),
    );
    assert!(matches!(record.status(), QualificationStatus::Unqualified));
    assert!(record.mark_qualified(None).is_err());
    assert!(record.start_qualifying().is_ok());
    assert!(record.mark_qualified(None).is_ok());
    assert!(record.status().is_eligible());
}

#[test]
fn qualification_profile_does_not_infer_stricter_profile_from_weaker_evidence() {
    let baseline = QualificationProfile::baseline_correctness();
    let strict = QualificationProfile::strict_correctness();
    assert!(!baseline.satisfies(&strict));
    assert!(baseline.satisfies(&baseline.clone()));
}

#[test]
fn oracle_required_when_reference_cpu_does_not_support_operator() {
    assert!(require_oracle(false, None).is_err());
    assert!(require_oracle(true, None).is_ok());
    assert!(require_oracle(false, Some(&reference_cpu_oracle("1"))).is_ok());
}

#[test]
fn kernel_qualification_conformance_report_is_conformant() {
    let report = run_kernel_qualification_conformance();
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
