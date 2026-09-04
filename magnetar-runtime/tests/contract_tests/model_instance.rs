use magnetar_runtime::{
    AdapterSetId, BatchCompatibility, ComputeDType, FallbackClass, GenerationModelReference,
    KvCache, KvCacheCompatibility, KvCacheId, KvCacheLayoutMetadata, KvCacheLifecycleState,
    KvCacheScope, MemoryAllocationId, MemoryManager, MemoryPressureLevel, ModelArchitecture,
    ModelArchitectureImplementation, ModelArchitectureImplementationKind,
    ModelInstanceCreationChecks, ModelInstanceDefinition, ModelInstanceError, ModelInstanceId,
    ModelInstanceLifecycleState, ModelInstanceManager, ModelInstanceObservationKind,
    ModelInstancePolicy, ModelInstanceReadiness, ModelInstanceReadinessChecks,
    ModelInstanceReloadRequest, ModelInstanceSharingContext, ModelInstanceSharingPolicy,
    ModelInstanceUnloadPolicy, ModelInstanceWarmupPlan, ModelInstanceWarmupPolicy,
    ModelInstanceWarmupStep, ModelLoadingCoordinator, ModelLoadingRequest, ModelLoadingRequestId,
    ModelQuantizationPolicy, ModelResidencyId, ModelTrustStore, PrefixCacheEntryId,
    ProviderAdmissionDecision, ProviderBinding, ProviderHealthState, ProviderModelResource,
    ProviderPressureLevel, ProviderReadinessState, ResourceAffinity, Runtime, RuntimeConfig,
    TokenizerId,
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
            &ModelTrustStore::default()
                .trust_digest(manifest.id.digest.value.clone())
                .evaluate(&manifest),
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
            &ModelTrustStore::default()
                .trust_digest(manifest.id.digest.value.clone())
                .evaluate(&manifest),
            &mut memory,
        )
        .unwrap()
}

/// Reaches `Ready` the only way this crate (an external consumer of
/// `magnetar_runtime`'s public API, same as any embedder) now can:
/// `magnetar_runtime::materialize_model_instance_weights` -- the same
/// Runtime-owned, evidence-minting transaction production code uses
/// (`bind-model-loading-evidence-to-validated-artifact`). This replaced an
/// earlier version of this helper that wrote real Provider bytes and a
/// real residency but then bound the weight and marked the instance Ready
/// by directly poking `resource_bindings`/calling `warm_model_instance` --
/// exactly the hand-assembled forgery an external audit of PR #36
/// demonstrated this crate's own test suite was still capable of, even
/// with every earlier round's check in place. `resource_bindings` is now
/// `pub(crate)` and materialization evidence is Runtime-issued, so this is
/// no longer possible; going through the one real transaction is not a
/// workaround, it *is* the contract now. Binds under `manifest()`'s own
/// declared tensor name (`transformer.wte.weight`), matching
/// `required_weight_names` (itself `pub(crate)`, populated from this exact
/// fixture's manifest) so the inventory-completeness check is satisfied.
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
    let weights = std::collections::BTreeMap::from([(
        "transformer.wte.weight".to_string(),
        magnetar_runtime::HostTensor::new([1], [0.0]).unwrap(),
    )]);
    magnetar_runtime::materialize_model_instance_weights(runtime, id, "test", &weights).unwrap();
}

/// `bind_fake_weight` alone already reaches `Ready` -- its underlying
/// `materialize_model_instance_weights` transaction commits bindings,
/// mints materialization evidence, and marks the instance Ready in one
/// step, the same as the real production path. An explicit follow-up
/// `warm_model_instance` call is not just redundant but would fail (no
/// `Ready -> Ready` lifecycle transition exists).
fn reach_ready(runtime: &mut Runtime, id: &ModelInstanceId) {
    bind_fake_weight(runtime, id);
}

/// A further audit of PR #36 found that `ModelInstanceDefinition`'s
/// `#[derive(Clone)]` copies every field -- including the already-sealed
/// `resource_bindings` -- regardless of the caller's own inability to name
/// those fields directly, and that `ModelInstanceManager::create` accepted
/// a caller-supplied definition as-is. So an external caller with only
/// `pub` access (`ModelInstance::definition()`, `Clone`,
/// `ModelInstanceManager::create`) could clone a `Ready` instance's
/// definition -- carrying its real weight bindings -- into a brand-new
/// instance, aliasing live Provider resources across two distinct
/// `ModelInstanceId`s. Fixed by having `create` unconditionally reset
/// `resource_bindings` regardless of what the supplied definition
/// contained; proven here at the exact public surface the audit
/// described, not only by inspecting the fix's internals.
#[test]
fn cloned_definition_does_not_inherit_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id_a = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id_a);
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let cloned = runtime.model_instance(&id_a).unwrap().definition().clone();
    let id_b = runtime.model_instances_mut().create(cloned).unwrap();

    // B must not be Ready on arrival -- creation alone never implies
    // readiness, cloned definition or not.
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // The real regression: an *empty* materialization attempt on B must
    // not be able to "adopt" A's already-committed bindings by minting
    // fresh evidence over whatever `resource_bindings.weights` happens to
    // contain. If `create` had not reset it, this would have reached
    // `Ready` for B using A's real Provider resource, with zero bytes
    // actually staged by this call.
    magnetar_runtime::materialize_model_instance_weights(
        &mut runtime,
        &id_b,
        "test",
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // A is unaffected throughout -- this is not a mutation of A's own
    // state, only B's (failed) attempt to inherit it.
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// Same root cause as the test above, exercised through `reload` (which
/// also calls `ModelInstanceManager::create` internally with a
/// caller-supplied replacement definition) rather than a direct `create`
/// call, since the audit's fix needed to close the shared chokepoint both
/// paths go through, not just the more obvious one.
#[test]
fn reload_replacement_does_not_inherit_original_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    reach_ready(&mut runtime, &id);
    let existing_definition = runtime.model_instance(&id).unwrap().definition().clone();

    let replacement_id = runtime
        .model_instances_mut()
        .reload(
            &id,
            ModelInstanceReloadRequest {
                replacement: existing_definition,
                migrate_sessions: false,
                allow_active_semantic_mutation: false,
            },
        )
        .unwrap();

    assert_ne!(
        runtime.model_instance(&replacement_id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
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
        instance.definition().residencies,
        [ModelResidencyId::new(1)].into()
    );
    assert!(!instance.definition().placement.exposes_raw_handles());

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
    // longer auto-readies); `bind_fake_weight`'s real materialization
    // transaction alone (evidence-minting commit, then `mark_ready`) is
    // now sufficient to reach `Ready` -- an explicit follow-up
    // `warm_model_instance` call would be redundant and would fail (no
    // `Ready -> Ready` lifecycle transition exists).
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(definition()).unwrap();
    bind_fake_weight(&mut runtime, &id);
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

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
    let mut runtime_local_def = definition();
    runtime_local_def.policy.sharing = ModelInstanceSharingPolicy::RuntimeLocal;
    let instance = magnetar_runtime::ModelInstance::new(
        ModelInstanceId::new("share1").unwrap(),
        runtime_local_def,
    );
    let context = ModelInstanceSharingContext::from_definition(instance.definition());
    assert!(instance.can_share_with(&context));

    let mut private_cache = context.clone();
    private_cache.kv_cache_private = true;
    assert!(!instance.can_share_with(&private_cache));

    let mut tenant_isolated_def = definition();
    tenant_isolated_def.policy.sharing = ModelInstanceSharingPolicy::TenantIsolated;
    tenant_isolated_def.tenant = Some("tenant-a".into());
    let instance = magnetar_runtime::ModelInstance::new(
        ModelInstanceId::new("share2").unwrap(),
        tenant_isolated_def,
    );
    let mut tenant_context = ModelInstanceSharingContext::from_definition(instance.definition());
    assert!(instance.can_share_with(&tenant_context));
    tenant_context.tenant = Some("tenant-b".into());
    assert!(!instance.can_share_with(&tenant_context));
}

#[test]
fn adapter_activation_records_mutation_and_invalidates_dependent_caches() {
    let mut manager = ModelInstanceManager::new();
    let mut def = definition();
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-a").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-a").unwrap());
    let id = manager.create(def).unwrap();

    let report = manager
        .activate_adapters(&id, AdapterSetId::empty(), "session:opaque", true)
        .unwrap();

    assert_eq!(report.kv_caches.len(), 1);
    assert_eq!(report.prefix_entries.len(), 1);
    assert_eq!(
        manager.instance(&id).unwrap().definition().mutation_version,
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
    // `placement` stays the fixture's default (no pinned Provider) through
    // `reach_ready`, since `warm_model_instance` now derives `provider_ready`
    // from a real, registered Provider -- "provider-a" below is a fake
    // identity that exists only to prove unload's own
    // `released_provider_resources` counting, not a Provider this test
    // actually registers. Injected directly after reaching Ready, through
    // `set_provider_resource`/`track_memory_allocation` -- the two narrow,
    // post-creation mutations `ModelInstance` still allows, precisely
    // because neither can affect what artifact/weights/Provider the
    // instance actually executes against (see each method's own doc
    // comment). `ModelInstanceManager::create` resets `resource_bindings`
    // unconditionally, so the out-of-band memory allocation id below can no
    // longer be pre-populated on `def` before `create` the way it could
    // before a further audit of PR #36 found that path let a caller clone
    // another instance's real bindings into a new one.
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(def).unwrap();
    reach_ready(&mut runtime, &id);
    {
        let instance = runtime.model_instances_mut().instance_mut(&id).unwrap();
        instance.set_provider_resource(Some(ProviderModelResource {
            provider: ProviderBinding::new("provider-a"),
            handle_kind: "opaque-model".into(),
            release_required: true,
        }));
        // A deliberately out-of-band id: `reach_ready` above issues its own
        // real allocation from the same `MemoryManager`, whose ids start
        // at 1, so a fixture id of `1` here would silently collide in the
        // `BTreeSet` and undercount `released_memory_allocations`.
        instance.track_memory_allocation(MemoryAllocationId::new(999));
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
