use magnetar_runtime::{
    AdapterSetId, BatchCompatibility, ComputeDType, FallbackClass, GenerationModelReference,
    KvCache, KvCacheCompatibility, KvCacheId, KvCacheLayoutMetadata, KvCacheLifecycleState,
    KvCacheScope, MemoryPressureLevel, ModelArchitecture, ModelArchitectureImplementation,
    ModelArchitectureImplementationKind, ModelInstanceDefinition, ModelInstanceError,
    ModelInstanceId, ModelInstanceLifecycleState, ModelInstanceObservationKind,
    ModelInstanceReadiness, ModelInstanceReadinessChecks, ModelInstanceReloadRequest,
    ModelInstanceSharingContext, ModelInstanceSharingPolicy, ModelInstanceUnloadPolicy,
    ModelInstanceWarmupPlan, ModelInstanceWarmupPolicy, ModelInstanceWarmupStep,
    ModelLoadingApiRequest, ModelLoadingCoordinator, ModelLoadingRequest, ModelLoadingRequestId,
    ModelResidencyId, ModelTrustStore, ProviderAdmissionDecision, ProviderHealthState,
    ProviderPressureLevel, ProviderReadinessState, ResourceAffinity, Runtime, TokenizerId,
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
    storage_dtype: f32
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

/// A `Runtime` whose sealed trust policy trusts `manifest()`'s digest --
/// `ModelLoadingCoordinator::load` and `ModelInstanceDefinition::
/// from_loaded_context`/`ModelInstanceManager::create` became
/// `pub(crate)` (`seal-model-loading-and-instance-creation-primitives`),
/// so this crate, compiled as an external consumer of `magnetar_runtime`'s
/// public API, now reaches loading and instance creation exclusively
/// through `magnetar_runtime::load_model` and
/// `Runtime::create_model_instance`, the same as any real embedder.
fn trusted_runtime() -> Runtime {
    Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest().id.digest.value.clone()))
        .build()
        .unwrap()
}

fn loaded_context(runtime: &mut Runtime) -> magnetar_runtime::LoadedModelContext {
    let manifest = manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(implementation());
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("load-runtime"),
        manifest.id.clone(),
    );
    magnetar_runtime::load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &manifest,
    )
    .unwrap()
}

/// Builds a valid `ModelInstanceDefinition` the only way this crate now
/// can: create a real Model Instance through `Runtime::
/// create_model_instance` and clone its definition back out through the
/// public `ModelInstance::definition()` accessor. Callers that need to
/// mutate the result's still-public fields (`policy`, `tenant`, `usage`,
/// ...) before using it elsewhere may still do so -- only the
/// constructor and the registration sink are sealed, not the type's own
/// field visibility.
fn definition() -> ModelInstanceDefinition {
    let mut runtime = trusted_runtime();
    let loaded = loaded_context(&mut runtime);
    let id = runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
    runtime.model_instance(&id).unwrap().definition().clone()
}

/// Reaches `Ready` the only way this crate (an external consumer of
/// `magnetar_runtime`'s public API, same as any embedder) now can:
/// `magnetar_runtime::materialize_model_instance_weights` -- the same
/// Runtime-owned, evidence-minting transaction production code uses
/// (`bind-model-loading-evidence-to-validated-artifact`). Binds under
/// `manifest()`'s own declared tensor name (`transformer.wte.weight`),
/// matching `required_weight_names` (itself `pub(crate)`, populated from
/// this exact fixture's manifest) so the inventory-completeness check is
/// satisfied.
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
    // Shape must match `manifest()`'s declared `[4, 8]` -- this crate
    // enforces that materialized content agrees with declared shape/dtype
    // independent of digest presence
    // (`seal-runtime-model-trust-and-provenance-authority`).
    let weights = std::collections::BTreeMap::from([(
        "transformer.wte.weight".to_string(),
        magnetar_runtime::HostTensor::new([4, 8], vec![0.0; 32]).unwrap(),
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

/// Creates a Model Instance the one Runtime-sealed way this crate can:
/// `Runtime::create_model_instance`, against a freshly loaded context on
/// `runtime`. Every remaining test in this file that needs an instance
/// uses this instead of the pre-`seal-model-loading-and-instance-creation-
/// primitives` `runtime.model_instances_mut().create(definition())`
/// bypass -- tests that specifically proved that bypass's own sealing
/// behavior (definition cloning, checked-creation validation, pre-create
/// field injection) moved into `magnetar-runtime/src/tests.rs`, where
/// `pub(crate)` access to the now-sealed primitives still applies.
fn create_instance(runtime: &mut Runtime) -> ModelInstanceId {
    let loaded = loaded_context(runtime);
    runtime
        .create_model_instance(
            &loaded,
            implementation(),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap()
}

#[test]
fn model_instance_id_is_opaque_runtime_owned_and_not_authority() {
    assert!(ModelInstanceId::new("client-instance").is_ok());
    assert!(ModelInstanceId::new("provider:0x1234").is_err());
    assert!(ModelInstanceId::new("device-memory-ptr").is_err());
    assert!(ModelInstanceId::new("raw-weight-ref").is_err());

    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);

    assert!(id.as_str().starts_with("model-instance-"));
    assert!(matches!(
        runtime.model_instance(&ModelInstanceId::new("model-instance-999").unwrap()),
        Err(ModelInstanceError::ModelInstanceNotFound)
    ));
}

#[test]
fn model_instance_binds_loaded_context_without_exposing_raw_handles() {
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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

    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let instance = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let instance = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let instance = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
    bind_fake_weight(&mut runtime, &id);
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let failing = ModelInstanceReadinessChecks {
        adapter_ready: false,
        ..ModelInstanceReadinessChecks::default()
    };
    let mut runtime = trusted_runtime();
    let failed = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let id1 = create_instance(&mut runtime);
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

    let mut runtime = trusted_runtime();
    let id2 = create_instance(&mut runtime);
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

    let mut runtime = trusted_runtime();
    let device_id = create_instance(&mut runtime);
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
fn reload_creates_validated_replacement_and_blocks_active_semantic_mutation() {
    let mut runtime = trusted_runtime();
    let id = create_instance(&mut runtime);
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
    let mut runtime = trusted_runtime();
    let failed = create_instance(&mut runtime);
    runtime
        .model_instances_mut()
        .fail_instance(&failed, ModelInstanceError::ModelInstanceProviderFailed)
        .unwrap();
    assert_eq!(
        runtime.model_instances_mut().generation_reference(&failed),
        Err(ModelInstanceError::ModelInstanceFailed)
    );

    let invalid = create_instance(&mut runtime);
    runtime
        .model_instances_mut()
        .invalidate_instance(&invalid, ModelInstanceError::ModelInstanceInvalid)
        .unwrap();
    assert_eq!(
        runtime.model_instances_mut().generation_reference(&invalid),
        Err(ModelInstanceError::ModelInstanceInvalid)
    );
}
