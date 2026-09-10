//! Production Model Artifact ingestion contract (`production-model-ingestion`,
//! `implement-production-qwen-model-loading`).
//!
//! Concrete production ingestors (Hugging Face-style bundle normalization,
//! today) understand source/bundle formats and stay external to
//! `magnetar-runtime`, exactly like Model Component, Provider, and Format
//! implementations already do (`project-architecture`'s externalization
//! rule). This module defines the generic, format-neutral boundary an
//! external ingestor implements and `magnetar-runtime` consumes:
//!
//! - [`ProductionModelSource`]: an already-authorized local bundle boundary
//!   (Decision 3 -- the ingestor may only resolve files inside it; nothing
//!   discovers or scans this path on its own).
//! - [`ProductionModelArtifactIngestor`]: the trait a concrete ingestor
//!   implements, turning an authorized source into a normalized
//!   [`crate::ModelManifest`] plus bounded payload access. Its output is
//!   *never* trusted merely because it parsed (Decision 2) -- callers still
//!   run it through [`crate::ModelManifest::validate`] and
//!   [`crate::ModelTrustStore::evaluate`] like any other manifest.
//! - [`ProductionArtifactPayloadSource`]: bounded, validated-range payload
//!   access, so Model Loading can materialize tensors incrementally instead
//!   of requiring a concrete ingestor to hand back one giant byte buffer.
//! - [`ProductionIngestionRegistry`]: where an embedder registers a
//!   concrete ingestor; `magnetar-runtime` only ever holds
//!   `Arc<dyn ProductionModelArtifactIngestor>`, never a concrete type.
//! - [`ProductionIngestionError`]: the structured failure categories this
//!   boundary exposes.

use crate::{ModelArtifactSource, ModelDigest, ModelManifest};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// Stable structured ingestion failure categories ("Production Ingestion
/// Errors Are Structured"). Every concrete ingestor implementation SHALL
/// report failures through this enum rather than an ad hoc string/panic, so
/// a caller can react to a category without parsing prose.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionIngestionError {
    UnsupportedFormat { reason: String },
    UnauthorizedSource { reason: String },
    RequiredPartMissing { part: String },
    MalformedMetadata { reason: String },
    PayloadOutOfBounds { identity: String },
    PayloadUnavailable { identity: String },
    IntegrityMismatch { identity: String },
    ImplementationUnavailable { reason: String },
}

impl ProductionIngestionError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::UnsupportedFormat { .. } => "production-ingestion-unsupported-format",
            Self::UnauthorizedSource { .. } => "production-ingestion-unauthorized-source",
            Self::RequiredPartMissing { .. } => "production-ingestion-required-part-missing",
            Self::MalformedMetadata { .. } => "production-ingestion-malformed-metadata",
            Self::PayloadOutOfBounds { .. } => "production-ingestion-payload-out-of-bounds",
            Self::PayloadUnavailable { .. } => "production-ingestion-payload-unavailable",
            Self::IntegrityMismatch { .. } => "production-ingestion-integrity-mismatch",
            Self::ImplementationUnavailable { .. } => {
                "production-ingestion-implementation-unavailable"
            }
        }
    }
}

impl fmt::Display for ProductionIngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormat { reason }
            | Self::UnauthorizedSource { reason }
            | Self::MalformedMetadata { reason }
            | Self::ImplementationUnavailable { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::RequiredPartMissing { part } => write!(f, "{}: {part}", self.id()),
            Self::PayloadOutOfBounds { identity }
            | Self::PayloadUnavailable { identity }
            | Self::IntegrityMismatch { identity } => {
                write!(f, "{}: {identity}", self.id())
            }
        }
    }
}

impl Error for ProductionIngestionError {}

// ---------------------------------------------------------------------
// Authorized source boundary
// ---------------------------------------------------------------------

/// An already-authorized production model source boundary (Decision 3): a
/// concrete ingestor may resolve bytes only through [`Self::resolve`],
/// which rejects absolute paths, `..` traversal, and symlink escape outside
/// `root`. Nothing in this type performs directory discovery/scanning --
/// `root` itself must already have been supplied by an explicit
/// caller/embedder decision, never inferred from a bare model reference
/// string (mirrors [`crate::model_format_roadmap::validate_local_file_boundary`]'s
/// existing "authorized" precedent, generalized into a reusable boundary
/// every ingestor shares instead of each one reimplementing traversal
/// defense).
#[derive(Clone, Debug)]
pub struct ProductionModelSource {
    kind: ModelArtifactSource,
    root: PathBuf,
}

impl ProductionModelSource {
    /// The only constructor: an embedder/caller explicitly attests `root`
    /// is an authorized local bundle directory for `kind`. `kind` carries
    /// provenance only (local/Hugging Face/Tachyon/...) -- it is never
    /// itself a trust signal (task "Normalize source identity/provenance
    /// without treating local/HuggingFace/Tachyon source kind as trust").
    pub fn authorized_local_bundle(kind: ModelArtifactSource, root: PathBuf) -> Self {
        Self { kind, root }
    }

    pub const fn kind(&self) -> &ModelArtifactSource {
        &self.kind
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves `relative` against `root`, rejecting anything that is not a
    /// plain forward-relative path (no absolute path, no `..`, no `.`
    /// components beyond the trivial empty one) and, once resolved,
    /// rejecting a canonicalized result outside `root`'s own canonical
    /// form (symlink escape). Never reads file contents -- the returned
    /// path is a candidate location a caller may then open under its own
    /// error handling.
    pub fn resolve(&self, relative: &str) -> Result<PathBuf, ProductionIngestionError> {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(ProductionIngestionError::UnauthorizedSource {
                reason: format!("'{relative}' escapes the authorized bundle boundary"),
            });
        }
        let candidate = self.root.join(relative_path);
        let canonical_root = self.root.canonicalize().map_err(|error| {
            ProductionIngestionError::UnauthorizedSource {
                reason: format!("authorized bundle root is unavailable: {error}"),
            }
        })?;
        let canonical_candidate = candidate.canonicalize().map_err(|error| {
            ProductionIngestionError::RequiredPartMissing {
                part: format!("{relative} ({error})"),
            }
        })?;
        if !canonical_candidate.starts_with(&canonical_root) {
            return Err(ProductionIngestionError::UnauthorizedSource {
                reason: format!(
                    "'{relative}' resolves outside the authorized bundle boundary (symlink escape)"
                ),
            });
        }
        Ok(canonical_candidate)
    }
}

// ---------------------------------------------------------------------
// Bounded payload access
// ---------------------------------------------------------------------

/// A validated, bounded reference to one payload range inside an authorized
/// artifact -- never a raw file handle, memory pointer, or unrestricted
/// path. `identity` is the logical tensor or part name Model Loading
/// already recognizes from the normalized [`ModelManifest`], not a
/// filesystem path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionPayloadRange {
    pub identity: String,
    pub offset: u64,
    pub length: u64,
    pub digest: Option<ModelDigest>,
}

/// Bounded artifact/tensor payload access a concrete ingestor exposes to
/// Model Loading, so production loading can read/validate/convert/stage
/// tensors incrementally instead of requiring the whole model to already
/// exist as an in-memory byte buffer.
pub trait ProductionArtifactPayloadSource: Send + Sync {
    /// Reads exactly the bytes for `range`. Implementations SHALL verify
    /// `range.digest` (when declared) before returning success, and SHALL
    /// return [`ProductionIngestionError::PayloadOutOfBounds`] /
    /// [`ProductionIngestionError::PayloadUnavailable`] rather than
    /// panicking or silently truncating on an invalid range.
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError>;
}

// ---------------------------------------------------------------------
// Ingestion contract
// ---------------------------------------------------------------------

/// Normalized ingestion output (Decision 1's "semantic output"): a
/// Runtime-validatable [`ModelManifest`] plus bounded payload access.
/// Producing this value never grants trust -- `manifest` still goes
/// through [`ModelManifest::validate`] and
/// [`crate::ModelTrustStore::evaluate`] exactly like any other manifest
/// before Model Loading may materialize anything from it.
pub struct ProductionIngestionResult {
    pub manifest: ModelManifest,
    pub payload_source: Arc<dyn ProductionArtifactPayloadSource>,
}

/// Format-neutral Runtime contract a concrete production ingestor
/// implements. `magnetar-runtime` never imports a concrete implementation
/// of this trait -- only registers/consumes it through
/// [`ProductionIngestionRegistry`], enforced at CI by the
/// `submodule-integration` job's externalized-module dependency guard.
pub trait ProductionModelArtifactIngestor: Send + Sync {
    /// Stable identity for diagnostics/registry lookup
    /// (e.g. `"huggingface-qwen"`). SHALL NOT be a Provider- or
    /// Device-shaped name.
    fn ingestor_id(&self) -> &str;

    /// Normalizes `source` into a validated Model Artifact plus payload
    /// access. `source` is already an authorized boundary -- the ingestor
    /// SHALL NOT itself decide what is authorized, only operate within it.
    fn ingest(
        &self,
        source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError>;
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

/// Where an embedder registers a concrete ingestor implementation.
/// `magnetar-runtime` holds only `Arc<dyn ProductionModelArtifactIngestor>`
/// values here -- never a concrete type from an externalized crate.
#[derive(Default)]
pub struct ProductionIngestionRegistry {
    ingestors: BTreeMap<String, Arc<dyn ProductionModelArtifactIngestor>>,
}

impl ProductionIngestionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, ingestor: Arc<dyn ProductionModelArtifactIngestor>) {
        self.ingestors
            .insert(ingestor.ingestor_id().to_string(), ingestor);
    }

    pub fn get(&self, ingestor_id: &str) -> Option<&Arc<dyn ProductionModelArtifactIngestor>> {
        self.ingestors.get(ingestor_id)
    }

    /// Ingests `source` through the registered ingestor named
    /// `ingestor_id`, implementing "External Ingestor Registered": Runtime
    /// consumes the normalized output through this generic contract
    /// without learning the concrete implementation's own types.
    pub fn ingest(
        &self,
        ingestor_id: &str,
        source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError> {
        let ingestor = self.get(ingestor_id).ok_or_else(|| {
            ProductionIngestionError::ImplementationUnavailable {
                reason: format!("no production ingestor registered for '{ingestor_id}'"),
            }
        })?;
        ingestor.ingest(source)
    }
}

#[cfg(test)]
mod tests {
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
}
