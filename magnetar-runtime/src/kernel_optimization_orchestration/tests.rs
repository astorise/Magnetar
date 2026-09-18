//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

#[test]
fn conformance_report_is_conformant() {
    let report = run_kernel_optimization_orchestration_conformance();
    for result in &report.results {
        assert!(
            result.passed,
            "requirement '{}' failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn campaign_cannot_be_validated_startable_from_the_hot_path() {
    let campaign = OptimizationCampaign::new(
        OptimizationCampaignId::new("test-campaign"),
        OptimizationTrigger::ManualRequest,
        "v1",
    );
    assert!(campaign.validate_startable(false).is_ok());
    assert!(matches!(
        campaign.validate_startable(true),
        Err(KernelOptimizationError::HotPathDenied)
    ));
}

#[test]
fn campaign_lifecycle_rejects_illegal_transitions() {
    assert!(CampaignLifecycleState::Planned.can_transition_to(CampaignLifecycleState::Queued));
    assert!(!CampaignLifecycleState::Planned.can_transition_to(CampaignLifecycleState::Completed));
    assert!(CampaignLifecycleState::Completed.is_terminal());
    assert!(!CampaignLifecycleState::Running.is_terminal());
}

#[test]
fn budget_exhaustion_is_detected_per_dimension() {
    let budget = CampaignBudget {
        max_candidates: Some(10),
        ..CampaignBudget::default()
    };
    let under = CampaignUsage {
        candidates: 5,
        ..CampaignUsage::default()
    };
    let over = CampaignUsage {
        candidates: 10,
        ..CampaignUsage::default()
    };
    assert_eq!(budget_exceeded(&budget, &under), None);
    assert_eq!(
        budget_exceeded(&budget, &over),
        Some(BudgetDimension::MaxCandidates)
    );
}

#[test]
fn candidate_failure_does_not_force_campaign_abort() {
    assert!(other_candidates_continue(
        CandidateFailureKind::CompilationFailure,
        CandidateFailurePolicy::ContinueRemainingCandidates
    ));
    assert!(!other_candidates_continue(
        CandidateFailureKind::CompilationFailure,
        CandidateFailurePolicy::AbortCampaign
    ));
}

#[test]
fn worker_selection_requires_every_declared_dimension() {
    let profile = WorkerCapabilityProfile {
        provider_implementations: vec![ProviderBinding::new("cuda")],
        device_architecture: Some("sm90".into()),
        compiler_toolchains: vec!["nvcc-12".into()],
        ..WorkerCapabilityProfile::default()
    };
    let compatible = WorkerCapabilityRequirement {
        required_provider: Some(ProviderBinding::new("cuda")),
        required_device_architecture: Some("sm90".into()),
        ..WorkerCapabilityRequirement::default()
    };
    let incompatible = WorkerCapabilityRequirement {
        required_device_architecture: Some("sm80".into()),
        ..WorkerCapabilityRequirement::default()
    };
    assert!(worker_compatible_with_target(&profile, &compatible));
    assert!(!worker_compatible_with_target(&profile, &incompatible));
}

#[test]
fn provider_isolation_detects_shared_instance() {
    let isolated = ProviderIsolation {
        optimization_worker_provider: ProviderBinding::new("cuda-optimization-worker"),
        production_provider: ProviderBinding::new("cuda-production"),
    };
    let shared = ProviderIsolation {
        optimization_worker_provider: ProviderBinding::new("cuda-production"),
        production_provider: ProviderBinding::new("cuda-production"),
    };
    assert!(isolated.is_isolated());
    assert!(!shared.is_isolated());
}
