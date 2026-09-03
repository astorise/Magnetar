use magnetar_runtime::{
    AdapterSetId, BatchCompatibility, ComputeDType, DeviceBinding, DeviceId, FallbackClass,
    GenerationModelReference, KvCache, KvCacheCompatibility, KvCacheId, KvCacheLayoutMetadata,
    KvCacheLifecycleState, KvCacheScope, MemoryAllocationId, MemoryManager, MemoryPressureLevel,
    ModelArchitecture, ModelArchitectureImplementation, ModelArchitectureImplementationKind,
    ModelInstanceCreationChecks, ModelInstanceDefinition, ModelInstanceError, ModelInstanceId,
    ModelInstanceLifecycleState, ModelInstanceManager, ModelInstanceObservationKind,
    ModelInstancePlacement, ModelInstancePolicy, ModelInstanceReadiness,
    ModelInstanceReadinessChecks, ModelInstanceReloadRequest, ModelInstanceSharingContext,
    ModelInstanceSharingPolicy, ModelInstanceUnloadPolicy, ModelInstanceWarmupPlan,
    ModelInstanceWarmupPolicy, ModelInstanceWarmupStep, ModelLoadingCoordinator,
    ModelLoadingRequest, ModelLoadingRequestId, ModelQuantizationPolicy, ModelResidencyId,
    ModelTrustDecision, ModelTrustStatus, PrefixCacheEntryId, ProviderAdmissionDecision,
    ProviderBinding, ProviderHealthState, ProviderModelResource, ProviderPressureLevel,
    ProviderReadinessState, ResourceAffinity, Runtime, RuntimeConfig, TokenizerId,
};

fn digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

fn manifest() -> magnetar_runtime::ModelManifest {
    magnetar_runtime::ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: instance-model
  revision: r1
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: bf16
compute_dtype: bf16
supported_compute_dtypes: [bf16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
    size_bytes: 16
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: bf16
"#,
        digest(),
        digest(),
        digest()
    ))
    .unwrap()
}

fn implementation() -> ModelArchitectureImplementation {
    ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    }
}

fn definition() -> ModelInstanceDefinition {
    let manifest = manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(implementation());
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = coordinator
        .load(
            request,
            &manifest,
            &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture"),
            &mut memory,
        )
        .unwrap();

    ModelInstanceDefinition::from_loaded_context(
        &loaded,
        implementation(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
}

fn loaded_context() -> magnetar_runtime::LoadedModelContext {
    let manifest = manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(implementation());
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("load-runtime"),
        manifest.id.clone(),
    );

    coordinator
        .load(
            request,
            &manifest,
            &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted fixture"),
            &mut memory,
        )
        .unwrap()
}

/// Reaches `Ready` the only way this crate (an external consumer of
/// `magnetar_runtime`'s public API, same as any embedder) now can:
/// `ModelInstance::mark_ready`/`transition_to`/`warmup` and the matching
/// `ModelInstanceManager` wrappers became `pub(crate)` after an external
/// audit of PR #36 demonstrated they were a direct, unverified bypass
/// (`runtime.model_instances_mut().mark_ready(&id)` skipped every
/// readiness check entirely). Reaching `Ready` now requires real evidence
/// the Runtime can verify: a Memory Manager allocation, a real
/// `Provider::write_tensor` call, a matching `TensorResidency` record, and
/// (when the fixture's manifest declares mandatory tensors, as `definition()`
/// here does) binding under the exact required name -- exactly the same
/// evidence a real embedder must supply through `warm_model_instance`. This
/// mirrors the real path deliberately: it is not a workaround, it *is* the
/// contract now.
fn bind_fake_weight(runtime: &mut Runtime, id: &ModelInstanceId) {
    if runtime
        .providers()
        .provider(magnetar_runtime::REFERENCE_CPU_PROVIDER_NAME)
        .is_none()
    {
        runtime
            .register_provider(std::sync::Arc::new(
                magnetar_runtime::ReferenceCpuProvider::new(),
            ))
            .unwrap();
    }
    let allocation = runtime
        .memory_mut()
        .allocate(magnetar_runtime::MemoryAllocationRequest::new(
            magnetar_runtime::MemoryAllocationClass::Tensor,
            1,
            magnetar_runtime::MemoryPlacement::HostOrdinary,
            magnetar_runtime::MemoryAllocationOwner::InferenceArtifact("test".into()),
        ))
        .unwrap();
    let resource_id = magnetar_runtime::TensorResourceId::new(format!("test.weight.{id}"));
    let provider_binding = ProviderBinding::new(magnetar_runtime::REFERENCE_CPU_PROVIDER_NAME);
    runtime
        .providers()
        .provider(magnetar_runtime::REFERENCE_CPU_PROVIDER_NAME)
        .and_then(|provider| provider.execution_api())
        .unwrap()
        .write_tensor(
            resource_id.clone(),
            magnetar_runtime::HostTensor::new([1], [0.0]).unwrap(),
        );
    runtime
        .memory_mut()
        .record_tensor_residency(
            magnetar_runtime::TensorResidency::new(
                resource_id.clone(),
                magnetar_runtime::MemoryPlacement::ProviderOwnedOpaque(provider_binding.clone()),
                ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding),
            )
            .with_allocation(allocation.id),
        )
        .unwrap();
    // Bind under `manifest()`'s own declared tensor name: `required_weight_names`
    // is `pub(crate)` (not readable from this external test crate, by
    // design -- an embedder cannot redeclare it either), populated from
    // this exact fixture's manifest (`tensors: - name: transformer.wte.weight`),
    // so this key is exactly what the inventory-completeness check needs.
    let weight_name = "transformer.wte.weight".to_string();
    runtime
        .model_instances_mut()
        .instance_mut(id)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert(weight_name, resource_id);
    runtime
        .model_instances_mut()
        .instance_mut(id)
        .unwrap()
        .definition
        .resource_bindings
        .memory_allocations
        .insert(allocation.id);
}

fn reach_ready(runtime: &mut Runtime, id: &ModelInstanceId) {
    bind_fake_weight(runtime, id);
    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    magnetar_runtime::warm_model_instance(
        runtime,
        id,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap();
}

#[test]
fn model_instance_id_is_opaque_runtime_owned_and_not_authority() {
    assert!(ModelInstanceId::new("client-instance").is_ok());
    assert!(ModelInstanceId::new("provider:0x1234").is_err());
    assert!(ModelInstanceId::new("device-memory-ptr").is_err());
    assert!(ModelInstanceId::new("raw-weight-ref").is_err());

    let mut manager = ModelInstanceManager::new();
    let id = manager.create(definition()).unwrap();

    assert!(id.as_str().starts_with("model-instance-"));
    assert!(matches!(
        manager.instance(&ModelInstanceId::new("model-instance-999").unwrap()),
        Err(ModelInstanceError::ModelInstanceNotFound)
    ));
}

#[test]
fn model_instance_binds_loaded_context_without_exposing_raw_handles() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    let instance = runtime.model_instance(&id).unwrap();

    assert_eq!(instance.lifecycle(), ModelInstanceLifecycleState::Ready);
    assert_eq!(instance.readiness(), ModelInstanceReadiness::Ready);
    assert_eq!(
        instance.definition.residencies,
        [ModelResidencyId::new(1)].into()
    );
    assert!(!instance.definition.placement.exposes_raw_handles());

    let status = instance.status();
    assert!(!status.raw_weights_available);
    assert!(!status.raw_provider_handle_available);
    assert!(!status.raw_device_handle_available);
    assert!(!status.raw_memory_pointer_available);
}

#[test]
fn lifecycle_and_readiness_are_distinct_and_transitions_are_checked() {
    assert!(
        ModelInstanceLifecycleState::Creating
            .allows_transition_to(ModelInstanceLifecycleState::Loading)
    );
    assert!(
        !ModelInstanceLifecycleState::Creating
            .allows_transition_to(ModelInstanceLifecycleState::Ready)
    );

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    runtime
        .model_instances_mut()
        .instance_mut(&id)
        .unwrap()
        .drain()
        .unwrap();

    let error = runtime
        .model_instances_mut()
        .generation_reference(&id)
        .unwrap_err();
    assert_eq!(error, ModelInstanceError::ModelInstanceDraining);
}

#[test]
fn generation_uses_ready_model_instance_reference_and_usage_lifecycle() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);

    assert_eq!(
        runtime
            .model_instances_mut()
            .generation_reference(&id)
            .unwrap(),
        GenerationModelReference::ModelInstance(id.clone())
    );

    runtime
        .model_instances_mut()
        .acquire_usage(&id, 42)
        .unwrap();
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Active
    );
    assert!(matches!(
        runtime
            .model_instances_mut()
            .unload(&id, ModelInstanceUnloadPolicy::RejectActiveUse),
        Err(ModelInstanceError::ModelInstanceActive)
    ));

    runtime.model_instances_mut().release_usage(&id).unwrap();
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Idle
    );
    runtime
        .model_instances_mut()
        .unload(&id, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

#[test]
fn model_instance_observability_is_redacted() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    runtime
        .model_instances_mut()
        .acquire_usage(&id, 42)
        .unwrap();
    runtime.model_instances_mut().release_usage(&id).unwrap();

    assert!(
        runtime
            .model_instances_mut()
            .observations()
            .iter()
            .any(|observation| observation.kind == ModelInstanceObservationKind::Ready)
    );
    assert!(
        runtime
            .model_instances_mut()
            .observations()
            .iter()
            .all(|observation| {
                !observation.raw_weights_available
                    && !observation.raw_prompt_available
                    && !observation.raw_cache_available
                    && !observation.raw_provider_handle_available
                    && !observation.raw_device_handle_available
            })
    );
}

#[test]
fn memory_pressure_suspends_idle_instance_and_browser_error_is_structured() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    // Reach `Idle` the legitimate way: acquire then release usage
    // (Ready -> Active -> Idle), rather than a raw `transition_to`.
    runtime.model_instances_mut().acquire_usage(&id, 0).unwrap();
    runtime.model_instances_mut().release_usage(&id).unwrap();
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Idle
    );

    runtime
        .model_instances_mut()
        .mark_memory_pressure(&id, MemoryPressureLevel::High)
        .unwrap();

    let instance = runtime.model_instance(&id).unwrap();
    assert_eq!(instance.lifecycle(), ModelInstanceLifecycleState::Suspended);
    assert_eq!(instance.readiness(), ModelInstanceReadiness::Suspended);
    assert_eq!(
        ModelInstanceError::ModelInstanceBrowserFeatureUnsupported.to_string(),
        "model instance browser feature unsupported"
    );
}

#[test]
fn runtime_owns_model_instance_registry_and_usage() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let loaded = loaded_context();
    let id = runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    reach_ready(&mut runtime, &id);

    assert_eq!(
        runtime.model_instance_generation_reference(&id).unwrap(),
        GenerationModelReference::ModelInstance(id.clone())
    );
    runtime.acquire_model_instance_usage(&id, 7).unwrap();
    assert_eq!(
        runtime
            .model_instance_status(&id)
            .unwrap()
            .active_operation_count,
        1
    );
    runtime.release_model_instance_usage(&id).unwrap();
    runtime
        .unload_model_instance(&id, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

#[test]
fn runtime_unload_releases_model_instance_kv_caches() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let loaded = loaded_context();
    let instance = runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    reach_ready(&mut runtime, &instance);
    let cache = KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        KvCacheScope::ModelInstance,
        KvCacheCompatibility::new(
            GenerationModelReference::ModelInstance(instance.clone()),
            TokenizerId::new("instance-tokenizer").unwrap(),
        ),
        KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float16),
    );
    let cache_id = runtime.create_kv_cache(cache).unwrap();
    runtime.prefill_kv_cache_completed(&cache_id, 1).unwrap();

    runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();

    assert_eq!(
        runtime.kv_cache(&cache_id).unwrap().lifecycle,
        KvCacheLifecycleState::Released
    );
}

#[test]
fn runtime_rejected_unload_preserves_model_instance_kv_caches() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let loaded = loaded_context();
    let instance = runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    reach_ready(&mut runtime, &instance);
    let cache = KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        KvCacheScope::ModelInstance,
        KvCacheCompatibility::new(
            GenerationModelReference::ModelInstance(instance.clone()),
            TokenizerId::new("instance-tokenizer").unwrap(),
        ),
        KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float16),
    );
    let cache_id = runtime.create_kv_cache(cache).unwrap();
    runtime.prefill_kv_cache_completed(&cache_id, 1).unwrap();
    runtime.acquire_model_instance_usage(&instance, 0).unwrap();

    assert_eq!(
        runtime.unload_model_instance(&instance, ModelInstanceUnloadPolicy::RejectActiveUse),
        Err(ModelInstanceError::ModelInstanceActive)
    );

    assert_eq!(
        runtime.kv_cache(&cache_id).unwrap().lifecycle,
        KvCacheLifecycleState::Ready
    );
    runtime.release_model_instance_usage(&instance).unwrap();
}

#[test]
fn runtime_close_then_unload_skips_already_released_session_kv_cache_memory() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let loaded = loaded_context();
    let instance = runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    reach_ready(&mut runtime, &instance);
    let session = runtime
        .create_inference_session(magnetar_runtime::SessionCreationRequest {
            model: GenerationModelReference::ModelInstance(instance.clone()),
            tokenizer: magnetar_runtime::GenerationTokenizerReference {
                tokenizer_id: TokenizerId::new("instance-tokenizer").unwrap(),
                metadata: magnetar_runtime::TokenizerMetadata {
                    id: TokenizerId::new("instance-tokenizer").unwrap(),
                    artifact: magnetar_runtime::TokenizerArtifactId::new(
                        "instance-tokenizer-artifact",
                    )
                    .unwrap(),
                    digest: magnetar_runtime::ModelDigest::parse(digest()).unwrap(),
                    revision: magnetar_runtime::TokenizerRevision::new("r1").unwrap(),
                    family: magnetar_runtime::TokenizerFamily::new("fixture").unwrap(),
                    vocabulary_size: 16,
                    added_token_count: 0,
                    token_id_range: magnetar_runtime::TokenIdRange::new(0, 16),
                    model_max_length: Some(16),
                    special_tokens: Vec::new(),
                    additional_special_tokens: Vec::new(),
                    byte_fallback: false,
                    normalization: None,
                    pre_tokenizer: None,
                    supports_offsets: false,
                    supports_token_type_ids: false,
                    supports_browser: true,
                },
            },
            generation_defaults: magnetar_runtime::GenerationParameters::default(),
            policy: magnetar_runtime::SessionPolicy::default(),
            memory: magnetar_runtime::SessionMemoryBudget::default(),
            allowed_capabilities: std::collections::BTreeSet::new(),
            correlation_id: None,
            created_at_millis: 0,
        })
        .unwrap();
    let cache = KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        KvCacheScope::Session,
        KvCacheCompatibility::new(
            GenerationModelReference::ModelInstance(instance.clone()),
            TokenizerId::new("instance-tokenizer").unwrap(),
        ),
        KvCacheLayoutMetadata::contiguous(1, 1, 1, 8, ComputeDType::Float16),
    )
    .with_session(session.clone());
    let cache_id = runtime.create_kv_cache(cache).unwrap();
    let allocation = runtime.allocate_kv_cache_memory(&cache_id).unwrap();
    runtime.prefill_kv_cache_completed(&cache_id, 1).unwrap();

    runtime.close_inference_session(&session).unwrap();
    assert_eq!(
        runtime.kv_cache(&cache_id).unwrap().lifecycle,
        KvCacheLifecycleState::Released
    );

    runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();
    assert_eq!(
        runtime
            .memory()
            .allocations()
            .find(|item| item.id == allocation)
            .unwrap()
            .state,
        magnetar_runtime::MemoryAllocationState::Released
    );
}

#[test]
fn creation_and_readiness_checks_gate_ready_state() {
    let mut manager = ModelInstanceManager::new();
    let denied = ModelInstanceCreationChecks {
        artifact_trusted: false,
        ..ModelInstanceCreationChecks::default()
    };
    assert_eq!(
        manager.create_checked(definition(), &denied),
        Err(ModelInstanceError::ModelInstancePolicyDenied)
    );

    let id = manager
        .create_checked(definition(), &ModelInstanceCreationChecks::default())
        .unwrap();
    let checks = ModelInstanceReadinessChecks {
        provider_ready: false,
        ..ModelInstanceReadinessChecks::default()
    };
    assert_eq!(
        manager
            .instance_mut(&id)
            .unwrap()
            .validate_readiness(&checks),
        Err(ModelInstanceError::ModelInstanceProviderNotReady)
    );
}

#[test]
fn warmup_policy_covers_provider_kernel_shape_metadata_memory_and_adapter_checks() {
    let plan = ModelInstanceWarmupPlan::for_policy(ModelInstanceWarmupPolicy::Full);
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::ProviderInitialization)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::KernelPreparationPlaceholder)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::OperatorGraphPreparationPlaceholder)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::ShapePlanPreparationPlaceholder)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::TokenizerModelMetadataValidation)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::SmallTestExecutionPlaceholder)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::MemoryResidencyVerification)
    );
    assert!(
        plan.steps
            .contains(&ModelInstanceWarmupStep::AdapterReadinessVerification)
    );

    // A freshly created instance already starts in `Loading` (creation no
    // longer auto-readies), so warmup can run directly -- no scaffolding
    // through Reloading/Loading needed once real evidence (a bound,
    // residency-backed weight) is present.
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    bind_fake_weight(&mut runtime, &id);
    magnetar_runtime::warm_model_instance(
        &mut runtime,
        &id,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap();

    let failing = ModelInstanceReadinessChecks {
        adapter_ready: false,
        ..ModelInstanceReadinessChecks::default()
    };
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let failed = runtime.model_instances_mut().create(definition()).unwrap();
    bind_fake_weight(&mut runtime, &failed);
    assert!(matches!(
        magnetar_runtime::warm_model_instance(&mut runtime, &failed, &plan, &failing),
        Err(magnetar_runtime::InferenceApiError::ModelInstanceUnavailable { .. })
    ));
    assert_eq!(
        runtime.model_instance(&failed).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Failed
    );
}

#[test]
fn sharing_policy_considers_tenant_adapter_cache_privacy_and_affinity() {
    let mut instance =
        magnetar_runtime::ModelInstance::new(ModelInstanceId::new("share1").unwrap(), definition());
    instance.definition.policy.sharing = ModelInstanceSharingPolicy::RuntimeLocal;
    let context = ModelInstanceSharingContext::from_definition(&instance.definition);
    assert!(instance.can_share_with(&context));

    let mut private_cache = context.clone();
    private_cache.kv_cache_private = true;
    assert!(!instance.can_share_with(&private_cache));

    instance.definition.policy.sharing = ModelInstanceSharingPolicy::TenantIsolated;
    instance.definition.tenant = Some("tenant-a".into());
    let mut tenant_context = ModelInstanceSharingContext::from_definition(&instance.definition);
    assert!(instance.can_share_with(&tenant_context));
    tenant_context.tenant = Some("tenant-b".into());
    assert!(!instance.can_share_with(&tenant_context));
}

#[test]
fn adapter_activation_records_mutation_and_invalidates_dependent_caches() {
    let mut manager = ModelInstanceManager::new();
    let id = manager.create(definition()).unwrap();
    {
        let instance = manager.instance_mut(&id).unwrap();
        instance
            .definition
            .usage
            .kv_cache_dependencies
            .insert(KvCacheId::new("cache-a").unwrap());
        instance
            .definition
            .usage
            .prefix_cache_dependencies
            .insert(PrefixCacheEntryId::new("prefix-a").unwrap());
    }

    let report = manager
        .activate_adapters(&id, AdapterSetId::empty(), "session:opaque", true)
        .unwrap();

    assert_eq!(report.kv_caches.len(), 1);
    assert_eq!(report.prefix_entries.len(), 1);
    assert_eq!(
        manager.instance(&id).unwrap().definition.mutation_version,
        1
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|event| event.kind == ModelInstanceObservationKind::CacheInvalidation)
    );
}

#[test]
fn batching_compatibility_includes_model_instance_readiness_adapter_and_pressure() {
    let id = ModelInstanceId::new("model-instance-1").unwrap();
    let left = BatchCompatibility {
        model: GenerationModelReference::ModelInstance(id.clone()),
        model_context: None,
        architecture: None,
        compute_dtype: None,
        tokenizer: magnetar_runtime::TokenizerId::new("tok").unwrap(),
        provider: None,
        device: None,
        affinity: None,
        kv_cache_layout: None,
        max_sequence_length: None,
        sampling_policy: None,
        memory_placement: None,
        provider_assisted_sampling: false,
        model_instance_readiness: Some(ModelInstanceReadiness::Ready),
        active_adapter_set: Some(AdapterSetId::empty()),
        provider_pressure: Some(ProviderPressureLevel::Low),
    };
    let mut right = left.clone();
    left.validate_with(&right).unwrap();

    right.model =
        GenerationModelReference::ModelInstance(ModelInstanceId::new("model-instance-2").unwrap());
    assert!(left.validate_with(&right).is_err());
    right = left.clone();
    right.model_instance_readiness = Some(ModelInstanceReadiness::Suspended);
    assert!(left.validate_with(&right).is_err());
    right = left.clone();
    right.active_adapter_set = None;
    assert!(left.validate_with(&right).is_err());
    right = left.clone();
    right.provider_pressure = Some(ProviderPressureLevel::Saturated);
    assert!(left.validate_with(&right).is_err());
}

#[test]
fn provider_and_device_status_drive_instance_lifecycle() {
    // `provider_status_changed`/`device_unavailable` are reactive business
    // methods on an already-Ready `ModelInstance` -- they stay directly
    // callable via `instance_mut()`, unlike the Ready-producing primitives
    // (`transition_to`, `mark_ready`, `warmup`) that are now crate-internal
    // only. A separate `Runtime`-backed instance per scenario (rather than
    // manually resetting `readiness` mid-test, no longer possible on a
    // private field) keeps each scenario's starting state unambiguous.
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id1 = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id1);
    let instance = runtime.model_instances_mut().instance_mut(&id1).unwrap();
    assert_eq!(
        instance.provider_status_changed(
            ProviderHealthState::Healthy,
            ProviderReadinessState::NotReady,
            ProviderPressureLevel::Low,
            ProviderAdmissionDecision::Admit,
        ),
        Err(ModelInstanceError::ModelInstanceProviderNotReady)
    );
    assert_eq!(instance.readiness(), ModelInstanceReadiness::NotReady);

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id2 = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id2);
    let instance = runtime.model_instances_mut().instance_mut(&id2).unwrap();
    assert_eq!(
        instance.provider_status_changed(
            ProviderHealthState::Failed,
            ProviderReadinessState::Ready,
            ProviderPressureLevel::Low,
            ProviderAdmissionDecision::Admit,
        ),
        Err(ModelInstanceError::ModelInstanceProviderFailed)
    );
    assert_eq!(instance.lifecycle(), ModelInstanceLifecycleState::Failed);

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let device_id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &device_id);
    let device = runtime
        .model_instances_mut()
        .instance_mut(&device_id)
        .unwrap();
    assert_eq!(
        device.device_unavailable(true),
        Err(ModelInstanceError::ModelInstanceDeviceLost)
    );
    assert_eq!(device.lifecycle(), ModelInstanceLifecycleState::Suspended);
}

#[test]
fn unload_releases_memory_provider_resources_adapters_and_cache_dependencies() {
    let mut def = definition();
    def.associated_sessions
        .insert(magnetar_runtime::InferenceSessionId::new("session-unload").unwrap());
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-unload").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-unload").unwrap());
    def.usage.adapter_dependencies.insert(AdapterSetId::empty());
    // A deliberately out-of-band id: `reach_ready` below issues its own
    // real allocation from the same `MemoryManager`, whose ids start at 1,
    // so a fixture id of `1` here would silently collide in the
    // `BTreeSet` and undercount `released_memory_allocations`.
    def.resource_bindings.memory_allocations = [MemoryAllocationId::new(999)].into();
    // `placement` stays the fixture's default (no pinned Provider) through
    // `reach_ready`, since `warm_model_instance` now derives `provider_ready`
    // from a real, registered Provider -- "provider-a" below is a fake
    // identity that exists only to prove unload's own
    // `released_provider_resources` counting, not a Provider this test
    // actually registers. Injected directly after reaching Ready.
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(def).unwrap();
    reach_ready(&mut runtime, &id);
    {
        let instance = runtime.model_instances_mut().instance_mut(&id).unwrap();
        instance.definition.placement = ModelInstancePlacement {
            provider: Some(ProviderBinding::new("provider-a")),
            device: Some(DeviceBinding::new(DeviceId::new("device-a"))),
            affinity: ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("provider-a"))
                .with_device(DeviceBinding::new(DeviceId::new("device-a"))),
            provider_resource: Some(ProviderModelResource {
                provider: ProviderBinding::new("provider-a"),
                handle_kind: "opaque-model".into(),
                release_required: true,
            }),
        };
    }

    let report = runtime
        .model_instances_mut()
        .unload(&id, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();

    assert_eq!(report.invalidated.kv_caches.len(), 1);
    assert_eq!(report.invalidated.prefix_entries.len(), 1);
    assert_eq!(report.invalidated.adapters_released.len(), 1);
    // +1 relative to the fixture's single pre-set `memory_allocations`
    // entry: `reach_ready`'s own Memory Manager allocation for the bound
    // weight is released on unload too.
    assert_eq!(report.released_memory_allocations.len(), 2);
    assert_eq!(report.released_provider_resources.len(), 1);
    assert!(!report.dangling_session_references);
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

#[test]
fn reload_creates_validated_replacement_and_blocks_active_semantic_mutation() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    runtime.model_instances_mut().acquire_usage(&id, 1).unwrap();
    let blocked = runtime.model_instances_mut().reload(
        &id,
        ModelInstanceReloadRequest {
            replacement: definition(),
            migrate_sessions: false,
            allow_active_semantic_mutation: false,
        },
    );
    assert_eq!(blocked, Err(ModelInstanceError::ModelInstanceActive));
    runtime.model_instances_mut().release_usage(&id).unwrap();

    let replacement = runtime
        .model_instances_mut()
        .reload(
            &id,
            ModelInstanceReloadRequest {
                replacement: definition(),
                migrate_sessions: true,
                allow_active_semantic_mutation: false,
            },
        )
        .unwrap();
    assert_ne!(id, replacement);
    // `reload`'s replacement is created via the same `create()` this
    // change's fix applies to -- it stays non-Ready until an explicit
    // readiness step, same as any other freshly created instance.
    assert_eq!(
        runtime.model_instance(&replacement).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Loading
    );
    reach_ready(&mut runtime, &replacement);
    assert_eq!(
        runtime.model_instance(&replacement).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

#[test]
fn failure_handling_prevents_failed_and_invalid_instances_from_accepting_work() {
    let mut manager = ModelInstanceManager::new();
    let failed = manager.create(definition()).unwrap();
    manager
        .fail_instance(&failed, ModelInstanceError::ModelInstanceProviderFailed)
        .unwrap();
    assert_eq!(
        manager.generation_reference(&failed),
        Err(ModelInstanceError::ModelInstanceFailed)
    );

    let invalid = manager.create(definition()).unwrap();
    manager
        .invalidate_instance(&invalid, ModelInstanceError::ModelInstanceInvalid)
        .unwrap();
    assert_eq!(
        manager.generation_reference(&invalid),
        Err(ModelInstanceError::ModelInstanceInvalid)
    );
}

#[test]
fn browser_policy_rejects_native_or_oversized_instance_features() {
    let mut def = definition();
    def.policy = ModelInstancePolicy {
        browser_linear_memory_limit_bytes: Some(1),
        ..ModelInstancePolicy::default()
    };
    let mut manager = ModelInstanceManager::new();
    assert_eq!(
        manager.create(def),
        Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported)
    );
}
