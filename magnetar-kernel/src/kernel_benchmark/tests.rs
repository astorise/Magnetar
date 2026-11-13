//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
#[test]
fn kernel_benchmark_conformance_report_is_conformant() {
    let report = run_kernel_benchmark_conformance();
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

#[test]
fn regression_policy_correctness_only_never_rejects_on_latency() {
    assert!(evaluate_regression_policy(RegressionPolicy::CorrectnessOnly, 1000.0, 1.0).is_ok());
}
