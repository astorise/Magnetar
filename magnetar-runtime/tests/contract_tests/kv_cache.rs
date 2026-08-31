use magnetar_runtime::{
    ComputeDType, DeviceBinding, DeviceId, FallbackClass, GenerationModelReference,
    GenerationParameters, GenerationTokenizerReference, KvCache, KvCacheCompatibility,
    KvCacheError, KvCacheId, KvCacheLayoutMetadata, KvCacheLifecycleState, KvCacheObservationKind,
    KvCachePageMetadata, KvCachePolicy, KvCacheResidency, KvCacheRetentionPolicy, KvCacheScope,
    KvCacheSharingPolicy, MemoryAllocationClass, MemoryAllocationState, MemoryManagerConfig,
    MemoryPlacement, ModelArtifactId, ModelArtifactKind, ModelDigest, ModelInstanceId, ModelName,
    ModelRevision, PrefixCacheBackingKvCache, PrefixCacheCompatibility, PrefixCacheEntry,
    PrefixCacheEntryId, PrefixCacheFingerprint, PrefixCacheLifecycleState, PrefixCachePolicy,
    PrefixCacheSharingPolicy, PrefixFingerprint, ProviderBinding, ResourceAffinity, Runtime,
    RuntimeConfig, SessionCreationRequest, SessionMemoryBudget, SessionPolicy, SpecialToken,
    SpecialTokenKind, TokenIdRange, TokenizerArtifactId, TokenizerFamily, TokenizerId,
    TokenizerMetadata, TokenizerRevision,
};
use std::collections::BTreeSet;

fn model_reference(name: &str) -> GenerationModelReference {
    GenerationModelReference::ModelArtifact(ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new(name).unwrap(),
        ModelRevision::new("r1").unwrap(),
        ModelDigest::parse(
            "sha256:0000000000000000000000000000000000000000000000000000000000000001",
        )
        .unwrap(),
    ))
}

fn compatibility() -> KvCacheCompatibility {
    let tokenizer = TokenizerId::new("kv-tokenizer").unwrap();
    KvCacheCompatibility::new(model_reference("kv-model"), tokenizer.clone())
        .with_prefix_fingerprint(PrefixFingerprint::from_tokens(
            &[1, 2, 3],
            "kv-model-r1",
            &tokenizer,
        ))
}

fn tokenizer_reference() -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: TokenizerId::new("kv-tokenizer").unwrap(),
        metadata: TokenizerMetadata {
            id: TokenizerId::new("kv-tokenizer").unwrap(),
            artifact: TokenizerArtifactId::new("kv-tokenizer-artifact").unwrap(),
            digest: ModelDigest::parse(
                "sha256:0000000000000000000000000000000000000000000000000000000000000001",
            )
            .unwrap(),
            revision: TokenizerRevision::new("r1").unwrap(),
            family: TokenizerFamily::new("fixture").unwrap(),
            vocabulary_size: 512,
            added_token_count: 2,
            token_id_range: TokenIdRange::new(0, 1024),
            model_max_length: Some(256),
            special_tokens: vec![
                SpecialToken::new(SpecialTokenKind::Bos, "<s>", 0),
                SpecialToken::new(SpecialTokenKind::Eos, "</s>", 256),
            ],
            additional_special_tokens: Vec::new(),
            byte_fallback: true,
            normalization: Some("identity".into()),
            pre_tokenizer: Some("bytes".into()),
            supports_offsets: true,
            supports_token_type_ids: true,
            supports_browser: true,
        },
    }
}

fn session_creation_request() -> SessionCreationRequest {
    SessionCreationRequest {
        model: model_reference("kv-model"),
        tokenizer: tokenizer_reference(),
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    }
}

fn cache(scope: KvCacheScope) -> KvCache {
    KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        scope,
        compatibility(),
        KvCacheLayoutMetadata::contiguous(2, 4, 8, 16, ComputeDType::Float16),
    )
}

fn prefix_cache_compatibility() -> PrefixCacheCompatibility {
    PrefixCacheCompatibility::new(
        model_reference("kv-model"),
        TokenizerId::new("kv-tokenizer").unwrap(),
    )
}

fn prefix_cache_fingerprint(tokens: &[u32]) -> PrefixCacheFingerprint {
    PrefixCacheFingerprint::from_validated_tokens(
        tokens,
        "kv-model-r1",
        &TokenizerId::new("kv-tokenizer").unwrap(),
    )
}

fn session_prefix_entry(
    runtime: &Runtime,
    cache: &KvCacheId,
    session: magnetar_runtime::InferenceSessionId,
) -> PrefixCacheEntry {
    let mut entry = PrefixCacheEntry::new(
        PrefixCacheEntryId::new("placeholder").unwrap(),
        prefix_cache_fingerprint(&[1, 2, 3]),
        prefix_cache_compatibility(),
        PrefixCacheBackingKvCache::from_kv_cache(runtime.kv_cache(cache).unwrap()),
    )
    .with_session(session);
    entry.owner = Some("owner-a".into());
    entry.sharing = PrefixCacheSharingPolicy::SessionLocal;
    entry.position_end_exclusive = 3;
    entry
}

#[test]
fn kv_cache_id_is_runtime_owned_opaque_and_not_authority() {
    assert!(KvCacheId::new("client-cache").is_ok());
    assert!(KvCacheId::new("provider:0x1234").is_err());
    assert!(KvCacheId::new("device-cache").is_err());

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .create_kv_cache(cache(KvCacheScope::Operation))
        .unwrap();

    assert!(id.as_str().starts_with("kv-cache-"));
    assert!(matches!(
        runtime.kv_cache(&KvCacheId::new("kv-cache-999").unwrap()),
        Err(KvCacheError::CacheNotFound)
    ));
}

#[test]
fn kv_cache_lifecycle_prefill_decode_seal_and_eviction_are_enforced() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .create_kv_cache(cache(KvCacheScope::Session))
        .unwrap();

    runtime.prefill_kv_cache_completed(&id, 3).unwrap();
    assert_eq!(
        runtime.kv_cache(&id).unwrap().lifecycle,
        KvCacheLifecycleState::Ready
    );

    runtime.append_decode_kv_cache(&id, 1).unwrap();
    assert_eq!(
        runtime.kv_cache(&id).unwrap().layout.current_token_length,
        4
    );

    runtime.seal_kv_cache(&id).unwrap();
    assert!(matches!(
        runtime.append_decode_kv_cache(&id, 1),
        Err(KvCacheError::CacheSealed)
    ));

    runtime.evict_kv_cache(&id).unwrap();
    assert_eq!(
        runtime.kv_cache(&id).unwrap().lifecycle,
        KvCacheLifecycleState::Evicted
    );
}

#[test]
fn kv_cache_reuse_validates_model_tokenizer_prompt_and_affinity() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let mut created = cache(KvCacheScope::RuntimeCache);
    created.policy.sharing = KvCacheSharingPolicy::AllowReadOnlySealed;
    let id = runtime.create_kv_cache(created).unwrap();
    runtime.prefill_kv_cache_completed(&id, 3).unwrap();

    runtime
        .validate_kv_cache_reuse(&id, &compatibility(), None)
        .unwrap();

    let wrong_model = KvCacheCompatibility::new(
        model_reference("other-model"),
        TokenizerId::new("kv-tokenizer").unwrap(),
    );
    assert!(matches!(
        runtime.validate_kv_cache_reuse(&id, &wrong_model, None),
        Err(KvCacheError::CacheModelMismatch)
    ));

    let provider_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"));
    let provider_cache = cache(KvCacheScope::ModelInstance).with_residency(
        KvCacheResidency::provider_owned(ProviderBinding::new("provider-a")),
    );
    let provider_cache_id = runtime.create_kv_cache(provider_cache).unwrap();
    runtime
        .prefill_kv_cache_completed(&provider_cache_id, 1)
        .unwrap();
    runtime
        .validate_kv_cache_reuse(
            &provider_cache_id,
            &compatibility(),
            Some(&provider_affinity),
        )
        .unwrap();

    let other_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-b"));
    assert!(matches!(
        runtime.validate_kv_cache_reuse(
            &provider_cache_id,
            &compatibility(),
            Some(&other_affinity)
        ),
        Err(KvCacheError::CacheProviderMismatch)
    ));
}

#[test]
fn kv_cache_memory_uses_memory_manager_kv_class_and_tracks_allocation() {
    let mut runtime = Runtime::initialize(RuntimeConfig {
        memory: MemoryManagerConfig {
            max_runtime_bytes: Some(4096),
            ..MemoryManagerConfig::default()
        },
        ..RuntimeConfig::default()
    });
    let id = runtime
        .create_kv_cache(cache(KvCacheScope::Operation))
        .unwrap();
    let allocation = runtime.allocate_kv_cache_memory(&id).unwrap();
    let allocated = runtime
        .memory()
        .allocations()
        .find(|item| item.id == allocation)
        .unwrap();

    assert_eq!(allocated.request.class, MemoryAllocationClass::KvCache);
    assert_eq!(
        runtime.kv_cache(&id).unwrap().residency.memory_allocation,
        Some(allocation)
    );

    runtime.prefill_kv_cache_completed(&id, 1).unwrap();
    runtime.evict_kv_cache(&id).unwrap();
    let released = runtime
        .memory()
        .allocations()
        .find(|item| item.id == allocation)
        .unwrap();
    assert_eq!(released.state, MemoryAllocationState::Released);
}

#[test]
fn kv_cache_layout_supports_paged_and_quantized_metadata_without_requiring_pages() {
    let page = KvCachePageMetadata {
        page_size_tokens: 16,
        page_count: 4,
        occupied_pages: 1,
        reusable_free_pages: 3,
        prefix_shared_pages: 0,
    };
    let layout = KvCacheLayoutMetadata::contiguous(2, 4, 8, 64, ComputeDType::Float16)
        .with_paged_metadata(page.clone());

    assert_eq!(layout.page, Some(page));
    assert_eq!(layout.token_capacity, 64);
}

#[test]
fn kv_cache_policy_redacts_observability_and_denies_sharing_by_default() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .create_kv_cache(cache(KvCacheScope::RuntimeCache))
        .unwrap();
    runtime.prefill_kv_cache_completed(&id, 1).unwrap();
    runtime.seal_kv_cache(&id).unwrap();

    assert!(matches!(
        runtime.validate_kv_cache_reuse(&id, &compatibility(), None),
        Err(KvCacheError::CacheSharingDenied)
    ));
    assert!(
        runtime
            .kv_caches()
            .observations()
            .iter()
            .all(|event| !event.raw_prompt_available
                && !event.raw_cache_available
                && !event.raw_provider_handle_available)
    );
    assert!(
        runtime
            .kv_caches()
            .observations()
            .iter()
            .any(|event| event.kind == KvCacheObservationKind::SharingDenied)
    );
}

#[test]
fn kv_cache_session_policy_can_release_or_retain_on_close() {
    let mut cache = cache(KvCacheScope::Session);
    cache.policy = KvCachePolicy {
        retention: KvCacheRetentionPolicy::ReleaseOnSessionClose,
        ..KvCachePolicy::default()
    };
    assert_eq!(
        cache.policy.retention,
        KvCacheRetentionPolicy::ReleaseOnSessionClose
    );
}

#[test]
fn kv_cache_releases_session_state_on_cancel() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let session = runtime
        .create_inference_session(session_creation_request())
        .unwrap();
    let cache_id = runtime
        .create_kv_cache(cache(KvCacheScope::Session).with_session(session.clone()))
        .unwrap();
    runtime.prefill_kv_cache_completed(&cache_id, 1).unwrap();

    runtime.cancel_inference_session(&session).unwrap();

    assert_eq!(
        runtime.kv_cache(&cache_id).unwrap().lifecycle,
        KvCacheLifecycleState::Released
    );
}

#[test]
fn kv_cache_cancel_respects_retention_and_marks_released_prefix_backings() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let session = runtime
        .create_inference_session(session_creation_request())
        .unwrap();
    let mut releasable_cache = cache(KvCacheScope::Session).with_session(session.clone());
    releasable_cache.policy = KvCachePolicy {
        retention: KvCacheRetentionPolicy::ReleaseOnSessionClose,
        sharing: KvCacheSharingPolicy::AllowReadOnlySealed,
        ..KvCachePolicy::default()
    };
    let releasable = runtime.create_kv_cache(releasable_cache).unwrap();
    runtime.prefill_kv_cache_completed(&releasable, 3).unwrap();
    runtime.seal_kv_cache(&releasable).unwrap();
    let releasable_prefix = runtime
        .create_prefix_cache_entry(
            session_prefix_entry(&runtime, &releasable, session.clone()),
            &PrefixCachePolicy::default(),
        )
        .unwrap();
    let mut retained_cache = cache(KvCacheScope::Session).with_session(session.clone());
    retained_cache.policy = KvCachePolicy {
        retention: KvCacheRetentionPolicy::RetainForPrefixReuse,
        sharing: KvCacheSharingPolicy::AllowReadOnlySealed,
        ..KvCachePolicy::default()
    };
    let retained = runtime.create_kv_cache(retained_cache).unwrap();
    runtime.prefill_kv_cache_completed(&retained, 3).unwrap();
    runtime.seal_kv_cache(&retained).unwrap();
    let retained_prefix = runtime
        .create_prefix_cache_entry(
            session_prefix_entry(&runtime, &retained, session.clone()),
            &PrefixCachePolicy::default(),
        )
        .unwrap();

    runtime.cancel_inference_session(&session).unwrap();

    assert_eq!(
        runtime.kv_cache(&releasable).unwrap().lifecycle,
        KvCacheLifecycleState::Released
    );
    assert_eq!(
        runtime
            .prefix_cache_entry(&releasable_prefix)
            .unwrap()
            .backing_kv_cache
            .lifecycle,
        KvCacheLifecycleState::Released
    );
    assert_eq!(
        runtime.kv_cache(&retained).unwrap().lifecycle,
        KvCacheLifecycleState::Sealed
    );
    assert_eq!(
        runtime
            .prefix_cache_entry(&retained_prefix)
            .unwrap()
            .lifecycle,
        PrefixCacheLifecycleState::Ready
    );
    assert_eq!(
        runtime
            .prefix_cache_entry(&retained_prefix)
            .unwrap()
            .backing_kv_cache
            .lifecycle,
        KvCacheLifecycleState::Sealed
    );
}

#[test]
fn kv_cache_releases_model_instance_state_on_unload() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let instance = ModelInstanceId::new("kv-model-instance").unwrap();
    let mut model_cache = cache(KvCacheScope::ModelInstance);
    model_cache.compatibility.model = GenerationModelReference::ModelInstance(instance.clone());
    let cache_id = runtime.create_kv_cache(model_cache).unwrap();
    runtime.prefill_kv_cache_completed(&cache_id, 1).unwrap();

    runtime
        .kv_caches_mut()
        .release_model_instance_caches(&instance)
        .unwrap();

    assert_eq!(
        runtime.kv_cache(&cache_id).unwrap().lifecycle,
        KvCacheLifecycleState::Released
    );
}

#[test]
fn kv_cache_device_binding_requires_matching_affinity() {
    let device = DeviceBinding::new(DeviceId::new("gpu-0"));
    let residency =
        KvCacheResidency::provider_owned(ProviderBinding::new("provider-a")).with_device(device);
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .create_kv_cache(cache(KvCacheScope::ModelInstance).with_residency(residency))
        .unwrap();
    runtime.prefill_kv_cache_completed(&id, 1).unwrap();

    let wrong_device = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu-1")));

    assert!(matches!(
        runtime.validate_kv_cache_reuse(&id, &compatibility(), Some(&wrong_device)),
        Err(KvCacheError::CacheDeviceMismatch)
    ));
}

#[test]
fn browser_compatible_cache_can_use_linear_memory_without_native_handles() {
    let cache = cache(KvCacheScope::Operation).with_residency(KvCacheResidency {
        placement: MemoryPlacement::BrowserLinearMemory,
        ..KvCacheResidency::host()
    });

    assert_eq!(
        cache.residency.placement,
        MemoryPlacement::BrowserLinearMemory
    );
    assert!(cache.residency.provider_resource.is_none());
}
