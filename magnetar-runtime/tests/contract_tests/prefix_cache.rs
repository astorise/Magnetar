use magnetar_runtime::{
    ComputeDType, FallbackClass, GenerationModelReference, KvCache, KvCacheId,
    KvCacheLayoutMetadata, KvCachePolicy, KvCacheResidency, KvCacheScope, KvCacheSharingPolicy,
    MemoryAllocationClass, MemoryAllocationState, MemoryManagerConfig, ModelArtifactId,
    ModelArtifactKind, ModelDigest, ModelName, ModelRevision, PrefixCacheBackingKvCache,
    PrefixCacheCompatibility, PrefixCacheEntry, PrefixCacheEntryId, PrefixCacheError,
    PrefixCacheFingerprint, PrefixCacheLifecycleState, PrefixCacheLookupRequest,
    PrefixCacheMatchKind, PrefixCacheObservationKind, PrefixCachePolicy, PrefixCacheSharingPolicy,
    PrefixFingerprint, ProviderBinding, ResourceAffinity, Runtime, RuntimeConfig, TokenizerId,
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

fn compatibility() -> PrefixCacheCompatibility {
    let mut compatibility = PrefixCacheCompatibility::new(
        model_reference("prefix-model"),
        TokenizerId::new("prefix-tokenizer").unwrap(),
    );
    compatibility.model_revision = Some("r1".into());
    compatibility.tokenizer_revision = Some("tok-r1".into());
    compatibility.template = Some("chat-v1".into());
    compatibility.position_encoding = Some("rope-v1".into());
    compatibility
}

fn fingerprint(tokens: &[u32]) -> PrefixCacheFingerprint {
    PrefixCacheFingerprint::from_validated_tokens(
        tokens,
        "prefix-model-r1",
        &TokenizerId::new("prefix-tokenizer").unwrap(),
    )
}

fn kv_cache() -> KvCache {
    let tokenizer = TokenizerId::new("prefix-tokenizer").unwrap();
    let mut cache = KvCache::new(
        KvCacheId::new("temporary-cache-id").unwrap(),
        KvCacheScope::RuntimeCache,
        magnetar_runtime::KvCacheCompatibility::new(
            model_reference("prefix-model"),
            tokenizer.clone(),
        )
        .with_prefix_fingerprint(PrefixFingerprint::from_tokens(
            &[1, 2, 3],
            "prefix-model-r1",
            &tokenizer,
        )),
        KvCacheLayoutMetadata::contiguous(2, 4, 8, 16, ComputeDType::Float16),
    );
    cache.policy = KvCachePolicy {
        sharing: KvCacheSharingPolicy::AllowReadOnlySealed,
        ..KvCachePolicy::default()
    };
    cache
}

fn create_backing(runtime: &mut Runtime) -> KvCacheId {
    let kv = runtime.create_kv_cache(kv_cache()).unwrap();
    runtime.prefill_kv_cache_completed(&kv, 3).unwrap();
    runtime.seal_kv_cache(&kv).unwrap();
    kv
}

fn create_provider_backing(runtime: &mut Runtime) -> KvCacheId {
    let kv = runtime
        .create_kv_cache(kv_cache().with_residency(KvCacheResidency::provider_owned(
            ProviderBinding::new("provider-a"),
        )))
        .unwrap();
    runtime.prefill_kv_cache_completed(&kv, 3).unwrap();
    runtime.seal_kv_cache(&kv).unwrap();
    kv
}

fn prefix_entry(runtime: &Runtime, kv: &KvCacheId) -> PrefixCacheEntry {
    let backing = PrefixCacheBackingKvCache::from_kv_cache(runtime.kv_cache(kv).unwrap());
    let mut entry = PrefixCacheEntry::new(
        PrefixCacheEntryId::new("placeholder").unwrap(),
        fingerprint(&[1, 2, 3]),
        compatibility(),
        backing,
    );
    entry.owner = Some("owner-a".into());
    entry.sharing = PrefixCacheSharingPolicy::PrivateOnly;
    entry.position_end_exclusive = 3;
    entry
}

fn lookup_request(tokens: &[u32]) -> PrefixCacheLookupRequest {
    PrefixCacheLookupRequest {
        fingerprint: fingerprint(tokens),
        compatibility: compatibility(),
        requested_prefix_token_length: tokens.len() as u32,
        session: None,
        owner: Some("owner-a".into()),
        tenant: None,
        affinity: None,
        allow_partial: false,
    }
}

#[test]
fn prefix_cache_entry_id_is_runtime_issued_opaque_and_not_authority() {
    assert!(PrefixCacheEntryId::new("client-prefix").is_ok());
    assert!(PrefixCacheEntryId::new("provider:0x1234").is_err());
    assert!(PrefixCacheEntryId::new("device-prefix").is_err());

    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let kv = create_backing(&mut runtime);
    let id = runtime
        .create_prefix_cache_entry(prefix_entry(&runtime, &kv), &PrefixCachePolicy::default())
        .unwrap();

    assert!(id.as_str().starts_with("prefix-cache-"));
    assert!(matches!(
        runtime.prefix_cache_entry(&PrefixCacheEntryId::new("prefix-cache-999").unwrap()),
        Err(PrefixCacheError::PrefixEntryNotFound)
    ));
}

#[test]
fn prefix_cache_exact_hit_miss_and_partial_hit_are_distinct() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let kv = create_backing(&mut runtime);
    runtime
        .create_prefix_cache_entry(prefix_entry(&runtime, &kv), &PrefixCachePolicy::default())
        .unwrap();

    let exact = runtime.lookup_prefix_cache(&lookup_request(&[1, 2, 3]));
    assert_eq!(exact.kind, PrefixCacheMatchKind::ExactPrefixHit);
    assert_eq!(exact.reusable_prefix_token_length, 3);

    let miss = runtime.lookup_prefix_cache(&lookup_request(&[4, 5, 6]));
    assert_eq!(miss.kind, PrefixCacheMatchKind::Miss);

    let mut partial = lookup_request(&[4, 5, 6, 7]);
    partial.allow_partial = true;
    let partial = runtime.lookup_prefix_cache(&partial);
    assert_eq!(partial.kind, PrefixCacheMatchKind::PartialPrefixHit);
    assert_eq!(partial.reusable_prefix_token_length, 3);
}

#[test]
fn prefix_cache_validates_model_tokenizer_template_position_and_affinity() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let kv = create_backing(&mut runtime);
    let id = runtime
        .create_prefix_cache_entry(prefix_entry(&runtime, &kv), &PrefixCachePolicy::default())
        .unwrap();

    runtime
        .validate_prefix_cache_reuse(&id, &lookup_request(&[1, 2, 3]))
        .unwrap();

    let mut wrong_model = lookup_request(&[1, 2, 3]);
    wrong_model.compatibility.model = model_reference("other-model");
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&id, &wrong_model),
        Err(PrefixCacheError::PrefixModelMismatch)
    ));

    let mut wrong_tokenizer = lookup_request(&[1, 2, 3]);
    wrong_tokenizer.compatibility.tokenizer = TokenizerId::new("other-tokenizer").unwrap();
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&id, &wrong_tokenizer),
        Err(PrefixCacheError::PrefixTokenizerMismatch)
    ));

    let mut wrong_template = lookup_request(&[1, 2, 3]);
    wrong_template.compatibility.template = Some("chat-v2".into());
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&id, &wrong_template),
        Err(PrefixCacheError::PrefixTemplateMismatch)
    ));

    let mut wrong_position = lookup_request(&[1, 2, 3]);
    wrong_position.compatibility.position_encoding = Some("rope-v2".into());
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&id, &wrong_position),
        Err(PrefixCacheError::PrefixPositionMismatch)
    ));

    let provider_kv = create_provider_backing(&mut runtime);
    let provider_entry = runtime
        .create_prefix_cache_entry(
            prefix_entry(&runtime, &provider_kv),
            &PrefixCachePolicy::default(),
        )
        .unwrap();
    let mut wrong_affinity = lookup_request(&[1, 2, 3]);
    wrong_affinity.affinity = Some(
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("provider-b")),
    );
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&provider_entry, &wrong_affinity),
        Err(PrefixCacheError::PrefixResourceAffinityConflict)
    ));
}

#[test]
fn prefix_cache_denies_cross_owner_sharing_by_default_and_redacts_observations() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let kv = create_backing(&mut runtime);
    let id = runtime
        .create_prefix_cache_entry(prefix_entry(&runtime, &kv), &PrefixCachePolicy::default())
        .unwrap();

    let mut other_owner = lookup_request(&[1, 2, 3]);
    other_owner.owner = Some("owner-b".into());
    assert!(matches!(
        runtime.validate_prefix_cache_reuse(&id, &other_owner),
        Err(PrefixCacheError::PrefixSharingDenied)
    ));

    assert!(
        runtime
            .prefix_caches()
            .observations()
            .iter()
            .all(|event| !event.raw_prompt_available
                && !event.raw_token_sequence_available
                && !event.raw_kv_cache_available)
    );
    assert!(
        runtime
            .prefix_caches()
            .observations()
            .iter()
            .any(|event| event.kind == PrefixCacheObservationKind::PolicyDeniedHit)
    );
}

#[test]
fn prefix_cache_tracks_memory_and_backing_kv_cache_lifecycle() {
    let mut runtime = Runtime::initialize(RuntimeConfig {
        memory: MemoryManagerConfig {
            max_runtime_bytes: Some(4096),
            ..MemoryManagerConfig::default()
        },
        ..RuntimeConfig::default()
    });
    let kv = create_backing(&mut runtime);
    let id = runtime
        .create_prefix_cache_entry(prefix_entry(&runtime, &kv), &PrefixCachePolicy::default())
        .unwrap();
    let allocation = runtime.allocate_prefix_cache_memory(&id).unwrap();
    let allocated = runtime
        .memory()
        .allocations()
        .find(|item| item.id == allocation)
        .unwrap();

    assert_eq!(allocated.request.class, MemoryAllocationClass::PrefixCache);
    runtime.release_kv_cache(&kv).unwrap();
    assert_eq!(
        runtime.prefix_cache_entry(&id).unwrap().lifecycle,
        PrefixCacheLifecycleState::Released
    );

    let released = runtime
        .memory()
        .allocations()
        .find(|item| item.id == allocation)
        .unwrap();
    assert_eq!(released.state, MemoryAllocationState::Active);
}
