//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
#[test]
fn candidate_state_cannot_skip_from_qualified_directly_to_retired() {
    assert!(!CandidateState::Qualified.can_transition_to(CandidateState::Retired));
    assert!(CandidateState::Qualified.can_transition_to(CandidateState::Candidate));
    assert!(CandidateState::Active.can_transition_to(CandidateState::Retiring));
    assert!(CandidateState::Retiring.can_transition_to(CandidateState::Retired));
}

#[test]
fn kernel_registry_hot_swap_and_retirement_errors_have_expected_ids() {
    let cases = [
        (
            KernelRegistryError::HotSwapFailed { reason: "x".into() },
            "kernel-hot-swap-failed",
        ),
        (
            KernelRegistryError::RetirementInUse { kernel: "x".into() },
            "kernel-retirement-in-use",
        ),
        (
            KernelRegistryError::RetirementFailed { kernel: "x".into() },
            "kernel-retirement-failed",
        ),
    ];
    for (error, expected_id) in cases {
        assert_eq!(error.code(), expected_id);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn automatic_rollback_policy_is_reserved_and_disabled_by_default() {
    assert!(!AutomaticRollbackPolicy::default().enabled);
}
