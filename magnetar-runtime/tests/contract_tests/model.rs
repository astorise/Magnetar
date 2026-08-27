use magnetar_runtime::{
    ComponentDigest, InferenceArtifactKind, InferenceArtifactReference, MemoryManager,
    MemoryManagerConfig, MemoryPlacement, ModelArtifactError, ModelArtifactKind,
    ModelArtifactObserver, ModelDigest, ModelManifest, ModelObservationKind, ModelSignature,
    ModelTrustStatus, ModelTrustStore,
};

fn digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

fn valid_manifest() -> String {
    format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: qwen.example
  revision: r1
  variant: instruct
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: int8
compute_dtype: bf16
supported_compute_dtypes: [bf16, fp16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
  tokenizer:
    kind: tokenizer
    digest: {}
tokenizer: tokenizer
quantization:
  format: q4_k
  group_size: 32
  workspace_bytes: 64
shards:
  - id: shard0
    digest: {}
    size_bytes: 128
    order: 0
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: int8
    shard: shard0
license:
  identifier: apache-2.0
provenance:
  source_repository: example/model
  conversion_tool: fixture
  publisher: example
"#,
        digest(),
        digest(),
        digest(),
        digest(),
        digest()
    )
}

#[test]
fn model_manifest_validates_bundle_identity_parts_and_metadata() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let record = manifest.validate().unwrap();

    assert_eq!(record.id.kind, ModelArtifactKind::ModelBundle);
    assert_eq!(record.license.unwrap().identifier, "apache-2.0");
    assert_eq!(record.provenance.unwrap().publisher.unwrap(), "example");
}

#[test]
fn model_digest_verification_rejects_mismatch() {
    let digest = ModelDigest::parse(digest()).unwrap();

    assert!(matches!(
        digest.verify_bytes(b"different bytes"),
        Err(ModelArtifactError::DigestMismatch { .. })
    ));
}

#[test]
fn model_manifest_rejects_direct_provider_and_device_pinning() {
    let provider_manifest = valid_manifest() + "\nprovider: cuda\n";
    assert!(matches!(
        ModelManifest::from_yaml_str(&provider_manifest),
        Err(ModelArtifactError::ProviderSelectionNotAllowed { .. })
    ));

    let device_manifest = valid_manifest() + "\ndevice: gpu0\n";
    assert!(matches!(
        ModelManifest::from_yaml_str(&device_manifest),
        Err(ModelArtifactError::DeviceSelectionNotAllowed { .. })
    ));
}

#[test]
fn model_manifest_rejects_missing_tokenizer_reference() {
    let manifest = valid_manifest().replace("tokenizer: tokenizer", "tokenizer: missing");
    let manifest = ModelManifest::from_yaml_str(&manifest).unwrap();

    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::TokenizerReferenceMissing { .. })
    ));
}

#[test]
fn model_manifest_rejects_missing_manifest_and_invalid_yaml() {
    assert!(matches!(
        ModelManifest::load_yaml("missing-model-manifest.yaml"),
        Err(ModelArtifactError::SourceUnavailable { .. })
    ));
    assert!(matches!(
        ModelManifest::from_yaml_str("schema: [not valid"),
        Err(ModelArtifactError::InvalidManifest { .. })
    ));
}

#[test]
fn model_manifest_rejects_unsupported_version_and_missing_config() {
    let unsupported = valid_manifest().replace("schema_version: 1", "schema_version: 999");
    let manifest = ModelManifest::from_yaml_str(&unsupported).unwrap();
    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::UnsupportedManifestVersion { found: 999 })
    ));

    let without_config = valid_manifest()
        .replace("  config:\n    kind: model-config\n    digest: sha256:0000000000000000000000000000000000000000000000000000000000000001\n", "");
    let manifest = ModelManifest::from_yaml_str(&without_config).unwrap();
    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::MissingRequiredPart { part }) if part == "model-config"
    ));
}

#[test]
fn model_manifest_rejects_unsupported_architecture_dtype_quantization_and_template() {
    let architecture = valid_manifest().replace("identifier: qwen2", "identifier: qwenprovider");
    assert!(matches!(
        ModelManifest::from_yaml_str(&architecture),
        Err(ModelArtifactError::InvalidManifest { .. })
            | Err(ModelArtifactError::ProviderSelectionNotAllowed { .. })
    ));

    let storage_dtype = valid_manifest().replace("storage_dtype: int8", "storage_dtype: nope");
    assert!(matches!(
        ModelManifest::from_yaml_str(&storage_dtype),
        Err(ModelArtifactError::UnsupportedStorageDType { .. })
    ));

    let compute_dtype = valid_manifest().replace("compute_dtype: bf16", "compute_dtype: nope");
    assert!(matches!(
        ModelManifest::from_yaml_str(&compute_dtype),
        Err(ModelArtifactError::UnsupportedComputeDType { .. })
    ));

    let quantization = valid_manifest().replace("format: q4_k", "format: made-up");
    assert!(matches!(
        ModelManifest::from_yaml_str(&quantization),
        Err(ModelArtifactError::UnsupportedQuantizationFormat { .. })
    ));

    let template = valid_manifest() + "\nchat_template: missing-template\n";
    let manifest = ModelManifest::from_yaml_str(&template).unwrap();
    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::TemplateReferenceMissing { .. })
    ));
}

#[test]
fn model_manifest_rejects_missing_shard_and_invalid_tensor_metadata() {
    let missing_shard = valid_manifest().replace("shard: shard0", "shard: shard1");
    let manifest = ModelManifest::from_yaml_str(&missing_shard).unwrap();
    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::MissingShard { .. })
    ));

    let invalid_tensor = valid_manifest().replace("shape: [4, 8]", "shape: [4, 0]");
    let manifest = ModelManifest::from_yaml_str(&invalid_tensor).unwrap();
    assert!(matches!(
        manifest.validate(),
        Err(ModelArtifactError::InvalidTensorMetadata { .. })
    ));
}

#[test]
fn model_shard_digest_mismatch_and_signature_metadata_are_explicit() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let shard = &manifest.shards[0];

    assert!(matches!(
        shard.verify_bytes(b"different shard bytes"),
        Err(ModelArtifactError::ShardDigestMismatch { .. })
    ));

    let signature = ModelSignature {
        kind: "minisign".into(),
        key_id: Some("key-1".into()),
        digest: manifest.id.digest.clone(),
    };
    assert_eq!(signature.digest, manifest.id.digest);
}

#[test]
fn model_trust_is_policy_owned_not_manifest_owned() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let decision = ModelTrustStore::default()
        .trust_digest(digest())
        .evaluate(&manifest);

    assert_eq!(decision.status, ModelTrustStatus::Trusted);
}

#[test]
fn model_manifest_publisher_metadata_does_not_grant_trust() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let mut store = ModelTrustStore::default();
    store.trusted_publishers.insert("example".into());

    let decision = store.evaluate(&manifest);

    assert_eq!(decision.status, ModelTrustStatus::Unknown);
    assert!(decision.reason.contains("metadata only"));
}

#[test]
fn model_trust_policy_rejects_and_revokes() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let rejected = ModelTrustStore::default()
        .reject_digest(digest())
        .evaluate(&manifest);
    let revoked = ModelTrustStore::default()
        .revoke_digest(digest())
        .evaluate(&manifest);

    assert_eq!(rejected.status, ModelTrustStatus::Rejected);
    assert_eq!(revoked.status, ModelTrustStatus::Revoked);
}

#[test]
fn model_artifact_identity_is_distinct_from_component_artifact_identity() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let component = InferenceArtifactReference::new(
        InferenceArtifactKind::Model,
        "component-style-reference",
        ComponentDigest::sha256(b"component-artifact"),
    )
    .unwrap();

    assert_ne!(manifest.id.digest.value, component.digest.value);
    assert_eq!(manifest.id.kind, ModelArtifactKind::ModelBundle);
}

#[test]
fn model_observer_emits_stable_event_categories() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let artifact = manifest.id.clone();
    let shard = manifest.shards[0].id.clone();
    let mut observer = ModelArtifactObserver::default();

    observer.artifact_discovered(Some(artifact.clone()), "found");
    observer.manifest_loaded(artifact.clone());
    observer.manifest_validation_failed(Some(artifact.clone()), "invalid");
    observer.digest_computed(artifact.clone());
    observer.digest_mismatch(artifact.clone());
    observer.shard_validated(artifact.clone(), &shard);
    observer.artifact_trusted(artifact.clone());
    observer.artifact_rejected(Some(artifact.clone()), "rejected");
    observer.memory_feasibility_checked(artifact.clone(), false);
    observer.residency_planned(artifact.clone());
    observer.artifact_cached(artifact.clone());
    observer.artifact_evicted(artifact);
    observer.source_failure("source failed");

    let kinds = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect::<Vec<_>>();
    assert_eq!(kinds.len(), 13);
    assert!(kinds.contains(&ModelObservationKind::ArtifactDiscovered));
    assert!(kinds.contains(&ModelObservationKind::SourceFailure));
}

#[test]
fn model_residency_uses_memory_manager_feasibility() {
    let manifest = ModelManifest::from_yaml_str(&valid_manifest()).unwrap();
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(16),
        ..MemoryManagerConfig::default()
    });

    let feasibility = manifest
        .memory_feasibility(&manager, MemoryPlacement::HostOrdinary)
        .unwrap();

    assert!(!feasibility.feasible);
}
