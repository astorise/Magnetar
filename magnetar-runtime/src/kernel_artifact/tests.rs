//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{DeviceBinding, ProviderBinding};
use crate::capability::CapabilityVersion;
use crate::device::DeviceId;
use crate::kernel::KernelAdvertisement;
use crate::kernel::{KernelId, KernelImplementationFamily, KernelOperatorVersionRange};
use crate::operator::{OperatorFamily, OperatorId};
#[test]
fn artifact_trust_is_only_ever_policy_controlled() {
    assert_eq!(
        evaluate_artifact_trust(false),
        KernelArtifactTrust::Untrusted
    );
    assert_eq!(evaluate_artifact_trust(true), KernelArtifactTrust::Trusted);
    assert_eq!(
        KernelArtifactTrust::default(),
        KernelArtifactTrust::Untrusted
    );
}

#[test]
fn hot_path_denies_compilation_cold_path_allows_it() {
    assert!(matches!(
        reject_hot_path_compilation(
            KernelArtifactPath::Hot,
            KernelArtifactColdPathOperation::Compilation
        ),
        Err(KernelArtifactError::HotPathCompilationDenied { .. })
    ));
    assert!(
        reject_hot_path_compilation(
            KernelArtifactPath::Cold,
            KernelArtifactColdPathOperation::Compilation
        )
        .is_ok()
    );
}

#[test]
fn lazy_preparation_requires_explicit_policy_and_admission_state() {
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), false, false).is_ok());
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy::disabled(), true, true),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(matches!(
        evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, false),
        Err(KernelArtifactError::PreparationUnavailable { .. })
    ));
    assert!(evaluate_lazy_preparation(LazyPreparationPolicy { enabled: true }, true, true).is_ok());
}

#[test]
fn inference_requests_reject_kernel_artifact_management_fields() {
    for field in KERNEL_ARTIFACT_FORBIDDEN_INFERENCE_FIELDS {
        assert!(reject_inference_request_artifact_field(field).is_err());
    }
    assert!(reject_inference_request_artifact_field("prompt").is_ok());
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
fn kernel_source_format_is_extensible_and_not_provider_binding() {
    let triton = KernelSourceFormat::new("triton", "source").with_version("3");
    assert_eq!(triton.stable_key(), "triton:source@3");
    let custom = KernelSourceFormat::new("vendor", "custom-ir").with_version("2");
    assert!(custom.is_valid());
    assert_eq!(custom.to_string(), "vendor:custom-ir@2");

    let empty = KernelSourceFormat::new("", "name");
    assert!(!empty.is_valid());
}

#[test]
fn prepared_kernel_id_allocator_produces_distinct_opaque_ids() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let first = allocator.allocate();
    let second = allocator.allocate();
    assert_ne!(first, second);
    assert_ne!(first.to_string(), second.to_string());
}

#[test]
fn prepared_kernel_lifecycle_blocks_destruction_while_referenced() {
    let mut allocator = PreparedKernelIdAllocator::default();
    let kernel = conformance_kernel_id("matmul-prepared");
    let mut prepared = PreparedKernel::new(
        allocator.allocate(),
        kernel,
        CompiledKernelArtifactId::from_digest("digest"),
        ProviderBinding::new("cuda-provider"),
        DeviceBinding::new(DeviceId::new("cuda-0")),
        PreparedKernelGeneration::new(1),
    );
    assert_eq!(prepared.active_references(), 0);
    prepared.mark_ready().unwrap();
    assert!(prepared.state.is_dispatchable());

    prepared.add_reference();
    assert!(matches!(
        prepared.destroy(),
        Err(KernelArtifactError::PreparedGenerationInUse { .. })
    ));
    prepared.release_reference();
    prepared.retire().unwrap();
    assert!(prepared.destroy().is_ok());
}

#[test]
fn source_artifact_validation_requires_trust_and_valid_format() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let artifact = KernelSourceArtifact::new(
        KernelSourceArtifactId::from_digest("digest-1"),
        KernelSourceFormat::new("triton", "source").with_version("3"),
        operator,
        KernelArtifactProvenance::AiGenerated,
    );

    let untrusted = validate_source_artifact(&artifact);
    assert!(matches!(
        untrusted,
        Err(KernelArtifactError::Untrusted { .. })
    ));

    let trusted = artifact.with_trust(evaluate_artifact_trust(true));
    assert!(validate_source_artifact(&trusted).is_ok());

    let empty_digest = KernelSourceArtifact::new(
        KernelSourceArtifactId::from_digest(""),
        KernelSourceFormat::new("triton", "source"),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelArtifactProvenance::HumanAuthored,
    )
    .with_trust(evaluate_artifact_trust(true));
    assert!(matches!(
        validate_source_artifact(&empty_digest),
        Err(KernelArtifactError::ArtifactInvalid { .. })
    ));
}

#[test]
fn compiled_artifact_validation_rejects_operator_and_provider_mismatch() {
    let operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let other_operator = OperatorId::magnetar("softmax", 1, OperatorFamily::Activation);
    let provider = ProviderBinding::new("cuda-provider");
    let other_provider = ProviderBinding::new("metal-provider");

    let artifact = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("compiled-digest"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator.clone(),
    )
    .with_trust(evaluate_artifact_trust(true))
    .with_provider_compatibility([provider.clone()]);

    assert!(validate_compiled_artifact(&artifact, &operator, &provider).is_ok());
    assert!(matches!(
        validate_compiled_artifact(&artifact, &other_operator, &provider),
        Err(KernelArtifactError::OperatorIncompatible { .. })
    ));
    assert!(matches!(
        validate_compiled_artifact(&artifact, &operator, &other_provider),
        Err(KernelArtifactError::ProviderIncompatible { .. })
    ));

    let untrusted = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest("compiled-digest-2"),
        "cubin",
        "nvcc",
        "12.4",
        "sm_90",
        operator.clone(),
    );
    assert!(matches!(
        validate_compiled_artifact(&untrusted, &operator, &provider),
        Err(KernelArtifactError::Untrusted { .. })
    ));
}

#[test]
fn kernel_advertisement_may_reference_artifact_metadata() {
    let id = conformance_kernel_id("matmul-advertised");
    let binding = KernelArtifactBinding::new(CompiledKernelArtifactId::from_digest("digest"))
        .with_source_artifact(KernelSourceArtifactId::from_digest("source-digest"));
    let advertisement = KernelAdvertisement::new(id.clone()).with_artifact(binding);
    assert!(advertisement.artifact.is_some());
    assert_eq!(advertisement.id, id);
}

#[test]
fn kernel_artifact_conformance_report_is_conformant() {
    let report = run_kernel_artifact_conformance();
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
