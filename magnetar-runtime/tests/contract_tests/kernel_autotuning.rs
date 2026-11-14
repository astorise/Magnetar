use magnetar_runtime::QualificationCoverage;
use std::collections::BTreeSet;

#[test]
fn kernel_autotuning_qualification_coverage_never_implicitly_inherits() {
    let exact = QualificationCoverage::ExactInstance {
        fingerprint: "instance-a".into(),
    };
    assert!(exact.covers("instance-a"));
    assert!(!exact.covers("instance-b"));

    let enumerated = QualificationCoverage::EnumeratedInstances {
        fingerprints: BTreeSet::from(["instance-a".to_string(), "instance-b".to_string()]),
    };
    assert!(enumerated.covers("instance-b"));
    assert!(!enumerated.covers("instance-c"));

    let unauthorized_envelope = QualificationCoverage::DeclaredEnvelope { authorized: false };
    assert!(!unauthorized_envelope.covers("any-instance"));
    let authorized_envelope = QualificationCoverage::DeclaredEnvelope { authorized: true };
    assert!(authorized_envelope.covers("any-instance"));

    assert!(!QualificationCoverage::RequiresPerInstanceQualification.covers("any-instance"));
}
