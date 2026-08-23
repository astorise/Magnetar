use magnetar_runtime::{
    ComputeDType, DeviceBinding, DeviceId, FallbackClass, GenerationModelReference, KvCache,
    KvCacheCompatibility, KvCacheError, KvCacheId, KvCacheLayoutMetadata, KvCacheLifecycleState,
    KvCacheObservationKind, KvCachePageMetadata, KvCachePolicy, KvCacheResidency,
    KvCacheRetentionPolicy, KvCacheScope, KvCacheSharingPolicy, MemoryAllocationClass,
    MemoryAllocationState, MemoryManagerConfig, MemoryPlacement, ModelArtifactId,
    ModelArtifactKind, ModelDigest, ModelName, ModelRevision, PrefixFingerprint, ProviderBinding,
    ResourceAffinity, Runtime, RuntimeConfig, TokenizerId,
};

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

fn cache(scope: KvCacheScope) -> KvCache {
    KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        scope,
        compatibility(),
        KvCacheLayoutMetadata::contiguous(2, 4, 8, 16, ComputeDType::Float16),
    )
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
