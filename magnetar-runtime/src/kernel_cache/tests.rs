//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::kernel_artifact::CompiledKernelArtifactId;
use crate::memory::MemoryPressureLevel;

#[test]
fn kernel_cache_conformance_report_is_conformant() {
    let report = run_kernel_cache_conformance();
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
fn kernel_cache_preparation_pressure_is_informational_and_does_not_gate_eligibility() {
    let mut cache = KernelArtifactCache::new();
    assert_eq!(cache.preparation_pressure_hint().level, None);

    cache.set_preparation_pressure_hint(PreparationPressureHint {
        level: Some(MemoryPressureLevel::High),
    });
    assert_eq!(
        cache.preparation_pressure_hint().level,
        Some(MemoryPressureLevel::High)
    );

    // Setting a high pressure hint does not, by itself, change whether an
    // otherwise-eligible entry is usable -- the Memory Manager retains
    // authority over Runtime Tensor allocation, not this cache.
    let key = KernelCacheKey {
        source_digest: None,
        compiled_artifact_digest: "digest-pressure".into(),
        source_format: None,
        compiled_format: "nvidia:cubin".into(),
        compiler_identity: "nvcc".into(),
        compiler_version: "12.0".into(),
        compiler_flags_fingerprint: None,
        provider_version: "1.0.0".into(),
        target_architecture: "sm90".into(),
        driver_runtime_compatibility_class: Default::default(),
        operator_semantics: "magnetar:matmul@1".into(),
        dtype: Default::default(),
        layout: Default::default(),
        shape_specialization: None,
        device_features: Default::default(),
    };
    let mut entry = KernelCacheEntry::new(
        key,
        CompiledKernelArtifactId::from_digest("digest-pressure"),
        "sha256:pressure",
    );
    entry.mark_validating().unwrap();
    entry.mark_ready().unwrap();
    let eligibility = evaluate_cache_eligibility(&entry, true, &CacheEligibilityPolicy::default());
    assert!(eligibility.is_ok());
}

#[test]
fn qualification_cache_key_requires_exact_match_for_reuse() {
    let base = QualificationCacheKey {
        artifact_digest: "digest".into(),
        qualification_suite_version: "1".into(),
        oracle_identity_version: "1".into(),
        qualification_profile: "baseline-correctness@1".into(),
        target_context: "sm90".into(),
        test_matrix_fingerprint: "fp".into(),
        tolerance_profile_fingerprint: "tp".into(),
    };
    let mut different_suite = base.clone();
    different_suite.qualification_suite_version = "2".into();
    assert!(base.is_reusable_for(&base));
    assert!(!base.is_reusable_for(&different_suite));
}
