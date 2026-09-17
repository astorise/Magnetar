//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{DeviceBinding, ProviderBinding};
use crate::affinity::{
    FallbackClass, HealthState, ProviderHealthReport, ProviderStatusSnapshot, ResourceAffinity,
};
use crate::capability::CapabilityVersion;
use crate::device::DeviceId;
use crate::kernel::KernelObservationKind;
use crate::kernel::{KernelId, KernelImplementationFamily, KernelOperatorVersionRange};
use crate::kernel_artifact::{
    CompiledKernelArtifactId, PreparedKernel, PreparedKernelGeneration, PreparedKernelIdAllocator,
};
use crate::kernel_performance_model::KernelPerformanceMetricSummary;
use crate::operator::{OperatorFamily, OperatorId};
use crate::provider::Provider;
use crate::reference_cpu::REFERENCE_CPU_PROVIDER_NAME;
use crate::reference_cpu::ReferenceCpuProvider;
use std::collections::BTreeMap;
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

fn conformance_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("kernel-artifact-test-provider"),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::TestFixture,
    )
}

#[test]
fn reference_cpu_kernel_registry_accepts_provider_advertisements() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    let matmul_id = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    assert!(registry.active_advertisement(&matmul_id).is_some());
}

#[test]
fn prepared_kernel_generations_coexist_and_registry_tracks_readiness() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let kernel = conformance_kernel_id("matmul-generations");
    let device = DeviceBinding::new(DeviceId::new("cuda-0"));
    let provider = ProviderBinding::new("cuda-provider");
    let artifact = CompiledKernelArtifactId::from_digest("digest");

    let mut registry = KernelRegistry::new();
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        artifact.clone(),
        provider.clone(),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().unwrap();
    let generation_one_id = generation_one.id;
    registry.register_prepared_kernel(generation_one);

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        artifact,
        provider,
        device,
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().unwrap();
    registry.register_prepared_kernel(generation_two);

    assert_eq!(registry.prepared_kernels_for(&kernel).count(), 2);
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());
    assert!(!registry.artifact_observations().is_empty());

    registry.retire_prepared_kernel(&generation_one_id).unwrap();
    assert!(registry.destroy_prepared_kernel(&generation_one_id).is_ok());
    assert_eq!(registry.prepared_kernels_for(&kernel).count(), 1);
    assert!(registry.validate_prepared_readiness(&kernel).is_ok());
}

#[test]
fn kernel_registry_lifecycle_conformance_report_is_conformant() {
    let report = run_kernel_registry_lifecycle_conformance();
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
fn registry_performance_evidence_is_keyed_by_generation_and_never_fabricated() {
    // Implements "Registry Preserves Performance Evidence Identity" and
    // "Registry Does Not Generate Performance Evidence" (proposal, define-
    // kernel-performance-model-and-adaptive-feedback-contract): a new
    // Prepared Kernel generation never inherits a prior generation's
    // evidence, and a candidate with no recorded evidence resolves to
    // `None` rather than another candidate's summary.
    let artifact = CompiledKernelArtifactId::from_digest("digest-a");
    let mut allocator = PreparedKernelIdAllocator::default();
    let generation_n = allocator.allocate();
    let generation_n_plus_1 = allocator.allocate();

    let mut evidence = BTreeMap::new();
    evidence.insert(
        performance_evidence_key(&artifact, generation_n),
        KernelPerformanceMetricSummary {
            count: 100,
            ..KernelPerformanceMetricSummary::default()
        },
    );

    assert!(lookup_performance_evidence(&evidence, &artifact, generation_n).is_some());
    assert!(
        lookup_performance_evidence(&evidence, &artifact, generation_n_plus_1).is_none(),
        "a new generation must not silently inherit the prior generation's evidence"
    );
}

#[test]
fn reference_cpu_kernel_registry_selects_registered_candidate() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    registry.set_provider_status(ProviderStatusSnapshot::from_health_report(
        ProviderHealthReport::new(
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            HealthState::Available,
        ),
    ));
    let matmul_operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let request = KernelSelectionRequest::new(
        "select-matmul",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let selection = registry.select(&request).unwrap();
    let selected = selection
        .selected
        .expect("a compatible Reference CPU candidate should be selected");
    assert_eq!(selected.provider.as_str(), REFERENCE_CPU_PROVIDER_NAME);
    assert!(
        selection
            .observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelSelected)
    );
}

#[test]
fn continuous_batching_preserves_in_flight_generation_and_admits_new_work_on_new_generation() {
    let kernel = KernelId::new(
        ProviderBinding::new("conformance-provider"),
        "conformance-kernel",
        crate::CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, crate::OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        crate::KernelImplementationFamily::TestFixture,
    );
    let device = DeviceBinding::new(crate::DeviceId::new("conformance-device"));
    let mut allocator = PreparedKernelIdAllocator::default();
    let mut registry = KernelRegistry::new();

    let mut generation_one = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-batch-v1"),
        ProviderBinding::new("conformance-provider"),
        device.clone(),
        PreparedKernelGeneration::new(1),
    );
    generation_one.mark_ready().unwrap();
    let generation_one_id = generation_one.id;
    registry.register_prepared_kernel(generation_one);
    registry
        .promote_generation(&kernel, generation_one_id)
        .unwrap();

    let slot_binding = registry.bind_batch_slot(&kernel, 7).unwrap();
    assert_eq!(slot_binding.generation, generation_one_id);

    let mut generation_two = PreparedKernel::new(
        allocator.allocate(),
        kernel.clone(),
        CompiledKernelArtifactId::from_digest("digest-batch-v2"),
        ProviderBinding::new("conformance-provider"),
        device,
        PreparedKernelGeneration::new(2),
    );
    generation_two.mark_ready().unwrap();
    let generation_two_id = generation_two.id;
    registry.register_prepared_kernel(generation_two);
    registry
        .promote_generation(&kernel, generation_two_id)
        .unwrap();

    // The existing slot binding is unaffected by promotion...
    assert_eq!(slot_binding.generation, generation_one_id);
    // ...but new batch admissions resolve to the newly promoted generation.
    assert_eq!(
        registry.admit_new_batch_work(&kernel),
        Some(generation_two_id)
    );
}
