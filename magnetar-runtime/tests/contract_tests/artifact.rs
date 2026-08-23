use magnetar_runtime::ComponentArtifactCache;

#[test]
fn artifact_cache_verifies_digest_keyed_bytes() {
    let bytes = b"component-bytes".to_vec();
    let mut cache = ComponentArtifactCache::default();
    let digest = cache.insert(bytes.clone());

    assert_eq!(cache.get_verified(&digest).unwrap(), Some(bytes.as_slice()));
}
