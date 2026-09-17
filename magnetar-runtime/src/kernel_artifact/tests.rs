//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{DeviceBinding, ProviderBinding};
use crate::capability::CapabilityVersion;
use crate::device::DeviceId;
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
