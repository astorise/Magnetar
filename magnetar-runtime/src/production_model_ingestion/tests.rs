//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

struct FailingIngestor;

impl ProductionModelArtifactIngestor for FailingIngestor {
    fn ingestor_id(&self) -> &str {
        "test-failing"
    }

    fn ingest(
        &self,
        _source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError> {
        Err(ProductionIngestionError::RequiredPartMissing {
            part: "config.json".into(),
        })
    }
}

struct SucceedingIngestor;

struct EmptyPayloadSource;

impl ProductionArtifactPayloadSource for EmptyPayloadSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        Err(ProductionIngestionError::PayloadUnavailable {
            identity: range.identity.clone(),
        })
    }
}

impl ProductionModelArtifactIngestor for SucceedingIngestor {
    fn ingestor_id(&self) -> &str {
        "test-succeeding"
    }

    fn ingest(
        &self,
        _source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError> {
        Ok(ProductionIngestionResult {
            manifest: untrusted_probe_manifest(),
            payload_source: Arc::new(EmptyPayloadSource),
        })
    }
}

fn untrusted_probe_manifest() -> ModelManifest {
    let digest = crate::ModelDigest::parse(format!("sha256:{}", "2".repeat(64))).unwrap();
    let id = crate::ModelArtifactId::new(
        crate::ModelArtifactKind::ModelBundle,
        crate::ModelName::new("production-ingestion-probe").unwrap(),
        crate::ModelRevision::new("v1").unwrap(),
        digest,
    );
    ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: crate::ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: Default::default(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: Default::default(),
        memory_features: Default::default(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: None,
        artifact_format: ArtifactFormat::HuggingFace,
    }
}

/// "Production Ingestion Does Not Grant Trust": a successful `ingest`
/// call produces a `ModelManifest` that Runtime trust evaluation still
/// treats as unproven -- parsing success never implies trust.
#[test]
fn successful_ingestion_does_not_imply_trust() {
    let mut registry = ProductionIngestionRegistry::new();
    registry.register(Arc::new(SucceedingIngestor));
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(PathBuf::from(".")),
        PathBuf::from("."),
    );
    let result = registry.ingest("test-succeeding", &source).unwrap();
    let trust = crate::ModelTrustStore::default().evaluate(&result.manifest);
    assert_eq!(trust.status(), crate::ModelTrustStatus::Unknown);
}

#[test]
fn unregistered_ingestor_is_implementation_unavailable() {
    let registry = ProductionIngestionRegistry::new();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(PathBuf::from(".")),
        PathBuf::from("."),
    );
    let error = match registry.ingest("does-not-exist", &source) {
        Err(error) => error,
        Ok(_) => panic!("expected an implementation-unavailable error"),
    };
    assert!(matches!(
        error,
        ProductionIngestionError::ImplementationUnavailable { .. }
    ));
}

#[test]
fn registered_ingestor_is_reachable_by_id() {
    let mut registry = ProductionIngestionRegistry::new();
    registry.register(Arc::new(FailingIngestor));
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(PathBuf::from(".")),
        PathBuf::from("."),
    );
    let error = match registry.ingest("test-failing", &source) {
        Err(error) => error,
        Ok(_) => panic!("expected a required-part-missing error"),
    };
    assert!(matches!(
        error,
        ProductionIngestionError::RequiredPartMissing { .. }
    ));
}

#[test]
fn resolve_rejects_parent_traversal() {
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(PathBuf::from(".")),
        std::env::current_dir().unwrap(),
    );
    let error = source.resolve("../outside.json").unwrap_err();
    assert!(matches!(
        error,
        ProductionIngestionError::UnauthorizedSource { .. }
    ));
}

#[test]
fn resolve_rejects_absolute_path() {
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(PathBuf::from(".")),
        std::env::current_dir().unwrap(),
    );
    #[cfg(windows)]
    let absolute = "C:\\Windows\\win.ini";
    #[cfg(not(windows))]
    let absolute = "/etc/passwd";
    let error = source.resolve(absolute).unwrap_err();
    assert!(matches!(
        error,
        ProductionIngestionError::UnauthorizedSource { .. }
    ));
}

#[test]
fn resolve_accepts_file_within_boundary() {
    let dir = std::env::current_dir().unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.clone()),
        dir,
    );
    let resolved = source.resolve("Cargo.toml").unwrap();
    assert!(resolved.ends_with("Cargo.toml"));
}
