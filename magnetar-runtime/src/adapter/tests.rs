//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::MemoryManagerConfig;

fn digest() -> AdapterDigest {
    AdapterDigest::parse(format!("sha256:{}", "a".repeat(64))).unwrap()
}

fn base() -> AdapterBaseModelCompatibility {
    AdapterBaseModelCompatibility {
        model_name: ModelName::new("qwen").unwrap(),
        model_revision: ModelRevision::new("v1").unwrap(),
        model_artifact: None,
        tokenizer: None,
        architecture: AdapterArchitectureCompatibility {
            family: "qwen".into(),
            implementation: "qwen2".into(),
            hidden_size: Some(4096),
            layer_count: Some(32),
            position_encoding: None,
            target_modules: BTreeSet::from(["q_proj".into()]),
            supported_storage_dtypes: BTreeSet::from([ModelDType::F16]),
            supported_compute_dtypes: BTreeSet::from([ComputeDType::Float16]),
            supported_quantization_formats: BTreeSet::new(),
        },
    }
}

fn artifact() -> AdapterArtifact {
    AdapterArtifact {
        id: AdapterArtifactId::new(
            AdapterName::new("support-lora").unwrap(),
            AdapterRevision::new("r1").unwrap(),
            digest(),
        ),
        method: AdapterMethod::Lora,
        base_model: base(),
        targets: vec![AdapterTargetModule {
            name: "q_proj".into(),
            role: AdapterTargetModuleRole::QueryProjection,
            layer_selector: Some(AdapterLayerSelector::All),
            expected_shape: vec![4096, 8],
        }],
        storage_dtype: ModelDType::F16,
        compute_dtype: Some(ComputeDType::Float16),
        rank: Some(8),
        alpha: Some(16),
        tensors: Vec::new(),
        quantization: None,
        required_capabilities: Vec::new(),
        license: None,
        provenance: None,
        trust: AdapterTrustStatus::Trusted,
    }
}

#[test]
fn digest_identity_distinguishes_same_name() {
    let left = AdapterArtifactId::new(
        AdapterName::new("same").unwrap(),
        AdapterRevision::new("r1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "a".repeat(64))).unwrap(),
    );
    let right = AdapterArtifactId::new(
        AdapterName::new("same").unwrap(),
        AdapterRevision::new("r1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "b".repeat(64))).unwrap(),
    );

    assert_ne!(left, right);
}

#[test]
fn unsupported_method_is_rejected() {
    assert!(matches!(
        AdapterMethod::parse("customer-support-lora"),
        Err(AdapterError::AdapterMethodUnsupported { .. })
    ));
}

#[test]
fn lora_requires_target_modules_and_rank() {
    let mut no_targets = artifact();
    no_targets.targets.clear();
    assert!(matches!(
        no_targets.validate(),
        Err(AdapterError::TargetModuleMissing { .. })
    ));

    let mut no_rank = artifact();
    no_rank.rank = None;
    assert!(matches!(
        no_rank.validate(),
        Err(AdapterError::AdapterRankUnsupported { rank: 0 })
    ));
}

#[test]
fn compatibility_rejects_wrong_base_model() {
    let artifact = artifact();
    let mut incompatible = base();
    incompatible.model_revision = ModelRevision::new("v2").unwrap();

    assert_eq!(
        validate_adapter_compatibility(&artifact, &incompatible, None),
        Err(AdapterError::BaseModelIncompatible)
    );
}

#[test]
fn compatibility_rejects_architecture_dtype_and_target_mismatch() {
    let artifact = artifact();

    let mut incompatible_arch = base();
    incompatible_arch.architecture.hidden_size = Some(2048);
    assert_eq!(
        validate_adapter_compatibility(&artifact, &incompatible_arch, None),
        Err(AdapterError::ArchitectureIncompatible)
    );

    let mut incompatible_dtype = base();
    incompatible_dtype
        .architecture
        .supported_storage_dtypes
        .clear();
    incompatible_dtype
        .architecture
        .supported_storage_dtypes
        .insert(ModelDType::F32);
    assert_eq!(
        validate_adapter_compatibility(&artifact, &incompatible_dtype, None),
        Err(AdapterError::StorageDTypeUnsupported {
            dtype: ModelDType::F16
        })
    );

    let mut missing_target = base();
    missing_target.architecture.target_modules.clear();
    assert!(matches!(
        validate_adapter_compatibility(&artifact, &missing_target, None),
        Err(AdapterError::TargetModuleMissing { .. })
    ));
}

#[test]
fn untrusted_and_revoked_adapters_fail_before_allocation() {
    let mut untrusted = artifact();
    untrusted.trust = AdapterTrustStatus::Untrusted;
    assert_eq!(
        untrusted.validate(),
        Err(AdapterError::AdapterArtifactUntrusted)
    );

    let mut revoked = artifact();
    revoked.trust = AdapterTrustStatus::Revoked;
    assert_eq!(
        revoked.validate(),
        Err(AdapterError::AdapterArtifactRevoked)
    );
}

#[test]
fn lifecycle_allows_ready_activation_but_not_requested_active() {
    assert!(AdapterLifecycleState::Ready.allows_transition_to(AdapterLifecycleState::Active));
    assert!(!AdapterLifecycleState::Requested.allows_transition_to(AdapterLifecycleState::Active));
}

#[test]
fn adapter_memory_uses_memory_manager_admission() {
    let request = AdapterLoadingRequest {
        request_id: AdapterLoadingRequestId::new("load1").unwrap(),
        artifact: artifact(),
        base_model: GenerationModelReference::LoadedModelContext("model1".into()),
        target_usage: AdapterTargetUsage::Generation,
        requested_compute_dtype: Some(ComputeDType::Float16),
        residency_policy: AdapterResidencyPolicy::PreferHost,
        activation_policy: AdapterActivationPolicy::LoadOnly,
        merge_policy: AdapterMergePolicy::Overlay,
        memory_budget_bytes: Some(4096),
        required_capabilities: Vec::new(),
        session: None,
        priority: 1,
        timeout_millis: Some(1000),
        correlation_id: None,
    };
    let estimate = AdapterMemoryEstimate {
        tensor_bytes: 1024,
        quantized_storage_bytes: 0,
        compute_materialization_bytes: 1024,
        transform_workspace_bytes: 0,
        merge_workspace_bytes: 0,
        unmerge_workspace_bytes: 0,
        transfer_staging_bytes: 0,
        pinned_memory_bytes: 0,
        placement: MemoryPlacement::HostOrdinary,
        affinity: None,
        queue_allowed: false,
    };
    let memory = MemoryManager::new(MemoryManagerConfig::default());

    assert!(adapter_memory_feasibility(&request, &estimate, &memory).is_ok());
}

#[test]
fn activation_and_deactivation_follow_policy() {
    let artifact = artifact();
    let mut residency = AdapterResidency {
        id: AdapterResidencyId::new("res1").unwrap(),
        artifact: artifact.id.clone(),
        lifecycle: AdapterLifecycleState::Ready,
        location: AdapterResidencyLocation::Host,
        affinity: None,
        memory_allocation: None,
        provider_resource: None,
    };
    let activation = AdapterActivationRequest {
        residency: residency.id.clone(),
        scope: AdapterActivationScope::Session(InferenceSessionId::new("session1").unwrap()),
        base_model: GenerationModelReference::LoadedModelContext("model1".into()),
        adapter_set: artifact.compatibility_key(),
        policy: AdapterCompositionPolicy::SingleAdapterOnly,
    };
    let policy = AdapterSessionPolicy {
        activation_allowed: true,
        allowed_adapters: Some(BTreeSet::from([artifact.id.clone()])),
        ..AdapterSessionPolicy::default()
    };

    assert!(validate_adapter_activation(&residency, &activation, Some(&policy), None).is_ok());
    residency
        .transition_to(AdapterLifecycleState::Active)
        .unwrap();

    let deactivation = AdapterDeactivationRequest {
        residency: residency.id.clone(),
        scope: activation.scope.clone(),
        release_residency: false,
        invalidate_cache_state: true,
    };
    apply_adapter_deactivation(&mut residency, &deactivation, Some(&policy)).unwrap();
    assert_eq!(residency.lifecycle, AdapterLifecycleState::Inactive);
}

#[test]
fn multiple_adapters_rejected_by_single_policy() {
    let first = artifact().id;
    let second = AdapterArtifactId::new(
        AdapterName::new("other").unwrap(),
        AdapterRevision::new("r1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "b".repeat(64))).unwrap(),
    );
    let set = AdapterSetId::from_adapters([first.clone(), second]);
    let request = AdapterActivationRequest {
        residency: AdapterResidencyId::new("res1").unwrap(),
        scope: AdapterActivationScope::Runtime,
        base_model: GenerationModelReference::LoadedModelContext("model1".into()),
        adapter_set: set.clone(),
        policy: AdapterCompositionPolicy::SingleAdapterOnly,
    };

    assert_eq!(
        activation_uses_adapter(&request, &set),
        Err(AdapterError::MultipleAdaptersUnsupported)
    );
}

#[test]
fn cache_and_batch_reject_incompatible_adapter_sets() {
    let base_model = GenerationModelReference::LoadedModelContext("model1".into());
    let left = AdapterCacheCompatibility {
        base_model: base_model.clone(),
        adapter_set: artifact().compatibility_key(),
        merge_policy: AdapterMergePolicy::Overlay,
    };
    let right = AdapterCacheCompatibility {
        adapter_set: AdapterSetId::empty(),
        ..left.clone()
    };
    assert_eq!(
        left.validate_reuse(&right),
        Err(AdapterError::KvCacheIncompatible)
    );

    let batch_left = AdapterBatchCompatibility {
        base_model: base_model.clone(),
        adapter_set: left.adapter_set,
        execution_strategy: AdapterMergePolicy::Overlay,
        provider: None,
        device: None,
        affinity: None,
    };
    let batch_right = AdapterBatchCompatibility {
        adapter_set: AdapterSetId::empty(),
        ..batch_left.clone()
    };
    assert_eq!(
        batch_left.validate_with(&batch_right),
        Err(AdapterError::ActivationConflict)
    );
}

#[test]
fn generation_context_rejects_silent_activation() {
    let context = AdapterGenerationContext {
        base_model: GenerationModelReference::LoadedModelContext("model1".into()),
        active_adapter_set: artifact().compatibility_key(),
        activation: None,
        implicit_loading_allowed: false,
    };

    assert_eq!(context.validate(), Err(AdapterError::ActivationDenied));
}

#[test]
fn observations_are_redacted() {
    let observation = AdapterObservation::redacted(
        AdapterObservationKind::Activated,
        Some(artifact().id),
        None,
        "adapter activated",
        None,
    );

    assert!(!observation.raw_adapter_tensors_available);
    assert!(!observation.raw_provider_handle_available);
    assert!(!observation.raw_prompt_available);
}
