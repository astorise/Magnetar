use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::Deserialize;
use sha2::{Digest as ShaDigest, Sha256};

use crate::InferenceSessionId;

pub const COMPONENT_ARTIFACT_SCHEMA: &str = "magnetar-component-artifact";
pub const COMPONENT_TRUST_SCHEMA: &str = "magnetar-component-trust";
pub const COMPONENT_ARTIFACT_SCHEMA_VERSION: u64 = 1;
pub const MAGNETAR_RUNTIME_VERSION: &str = "0.1.0";

static NEXT_COMPONENT_DEFINITION_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_COMPONENT_INSTANCE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_DISTRIBUTED_PACKAGE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

fn next_component_definition_id() -> ComponentDefinitionId {
    ComponentDefinitionId(
        NEXT_COMPONENT_DEFINITION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}

fn next_component_instance_id() -> ComponentInstanceId {
    ComponentInstanceId(
        NEXT_COMPONENT_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}

/// A WIT interface identified by its package-qualified name and version.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WitInterface {
    pub name: String,
    pub version: String,
}
impl WitInterface {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Declarative metadata for portable WebAssembly Component contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub imports: BTreeSet<WitInterface>,
    pub exports: BTreeSet<WitInterface>,
}
impl ComponentMetadata {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: description.into(),
            imports: BTreeSet::new(),
            exports: BTreeSet::new(),
        }
    }

    pub fn with_import(mut self, interface: WitInterface) -> Self {
        self.imports.insert(interface);
        self
    }

    pub fn with_export(mut self, interface: WitInterface) -> Self {
        self.exports.insert(interface);
        self
    }
}

/// A discovered Component artifact and its declared metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDescriptor {
    pub metadata: ComponentMetadata,
    pub artifact_path: PathBuf,
    pub manifest_path: Option<PathBuf>,
}

impl ComponentDescriptor {
    pub fn new(metadata: ComponentMetadata, artifact_path: impl Into<PathBuf>) -> Self {
        Self {
            metadata,
            artifact_path: artifact_path.into(),
            manifest_path: None,
        }
    }

    pub fn with_manifest_path(mut self, manifest_path: impl Into<PathBuf>) -> Self {
        self.manifest_path = Some(manifest_path.into());
        self
    }

    pub fn artifact_reference(&self) -> ComponentArtifactReference<'_> {
        ComponentArtifactReference::LocalPath(&self.artifact_path)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentArtifactReference<'a> {
    LocalPath(&'a Path),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentInterfaceShape {
    Function,
    Interface,
    Resource,
    Instance,
    Component,
    Module,
    Type,
    Unknown,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentImportRequirement {
    pub interface: WitInterface,
    pub shape: ComponentInterfaceShape,
}

impl ComponentImportRequirement {
    pub fn new(interface: WitInterface, shape: ComponentInterfaceShape) -> Self {
        Self { interface, shape }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentExportDescription {
    pub interface: WitInterface,
    pub shape: ComponentInterfaceShape,
}

impl ComponentExportDescription {
    pub fn new(interface: WitInterface, shape: ComponentInterfaceShape) -> Self {
        Self { interface, shape }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentContract {
    pub imports: BTreeSet<ComponentImportRequirement>,
    pub exports: BTreeSet<ComponentExportDescription>,
}

impl ComponentContract {
    pub fn from_metadata(metadata: &ComponentMetadata) -> Self {
        Self {
            imports: metadata
                .imports
                .iter()
                .cloned()
                .map(|interface| {
                    ComponentImportRequirement::new(interface, ComponentInterfaceShape::Interface)
                })
                .collect(),
            exports: metadata
                .exports
                .iter()
                .cloned()
                .map(|interface| {
                    ComponentExportDescription::new(interface, ComponentInterfaceShape::Interface)
                })
                .collect(),
        }
    }

    pub fn import_interfaces(&self) -> BTreeSet<WitInterface> {
        self.imports
            .iter()
            .map(|requirement| requirement.interface.clone())
            .collect()
    }

    pub fn export_interfaces(&self) -> BTreeSet<WitInterface> {
        self.exports
            .iter()
            .map(|description| description.interface.clone())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentDefinitionId(u64);
impl ComponentDefinitionId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentInstanceId(u64);
impl ComponentInstanceId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentDefinitionState {
    Registered,
    Validated,
    Prepared,
    Failed,
    Removed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentInstanceState {
    Instantiating,
    Ready,
    Failed,
    Destroyed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDefinition {
    pub id: ComponentDefinitionId,
    pub metadata: ComponentMetadata,
    pub artifact_path: PathBuf,
    pub manifest_path: Option<PathBuf>,
    pub artifact_digest: Option<ComponentDigest>,
    pub trust_decision: Option<ComponentTrustDecision>,
    pub state: ComponentDefinitionState,
}

impl ComponentDefinition {
    fn registered(descriptor: ComponentDescriptor) -> Self {
        Self {
            id: next_component_definition_id(),
            metadata: descriptor.metadata,
            artifact_path: descriptor.artifact_path,
            manifest_path: descriptor.manifest_path,
            artifact_digest: None,
            trust_decision: None,
            state: ComponentDefinitionState::Registered,
        }
    }

    pub fn artifact_reference(&self) -> ComponentArtifactReference<'_> {
        ComponentArtifactReference::LocalPath(&self.artifact_path)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentDigest {
    pub algorithm: String,
    pub value: String,
}

impl ComponentDigest {
    pub fn sha256(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        Self {
            algorithm: "sha256".into(),
            value: format!("sha256:{}", lower_hex(&digest)),
        }
    }

    pub fn parse(algorithm: impl Into<String>, value: impl Into<String>) -> Self {
        let algorithm = algorithm.into().to_ascii_lowercase();
        let value = value.into().to_ascii_lowercase();
        let value = if value.starts_with(&format!("{algorithm}:")) {
            value
        } else {
            format!("{algorithm}:{value}")
        };
        Self { algorithm, value }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentTrustStatus {
    Unknown,
    Trusted,
    Rejected,
    Quarantined,
    Revoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentTrustDecision {
    pub status: ComponentTrustStatus,
    pub reason: String,
}

impl ComponentTrustDecision {
    pub fn new(status: ComponentTrustStatus, reason: impl Into<String>) -> Self {
        Self {
            status,
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentTrustStore {
    pub trusted_digests: BTreeSet<String>,
    pub rejected_digests: BTreeSet<String>,
    pub revoked_digests: BTreeSet<String>,
    pub quarantined_digests: BTreeSet<String>,
    pub trusted_publishers: BTreeSet<String>,
    pub trusted_sources: BTreeSet<String>,
    pub allow_unsigned_local_development: bool,
}

impl ComponentTrustStore {
    pub fn load_yaml(path: impl AsRef<Path>) -> Result<Self, ComponentError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|source| ComponentError::TrustStore {
            path: path.into(),
            message: source.to_string(),
            source: Some(source),
        })?;
        let raw: TrustStoreYaml =
            serde_norway::from_str(&content).map_err(|source| ComponentError::TrustStore {
                path: path.into(),
                message: source.to_string(),
                source: None,
            })?;
        raw.validate(path)
    }

    pub fn trust_digest(mut self, digest: impl Into<String>) -> Self {
        self.trusted_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn reject_digest(mut self, digest: impl Into<String>) -> Self {
        self.rejected_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn revoke_digest(mut self, digest: impl Into<String>) -> Self {
        self.revoked_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn quarantine_digest(mut self, digest: impl Into<String>) -> Self {
        self.quarantined_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn trust_publisher(mut self, publisher: impl Into<String>) -> Self {
        self.trusted_publishers.insert(publisher.into());
        self
    }

    pub fn trust_source(mut self, source: impl Into<String>) -> Self {
        self.trusted_sources.insert(source.into());
        self
    }

    pub fn allow_unsigned_local_development(mut self, allow: bool) -> Self {
        self.allow_unsigned_local_development = allow;
        self
    }

    fn evaluate(
        &self,
        manifest: &ComponentManifest,
        digest: &ComponentDigest,
    ) -> ComponentTrustDecision {
        if self.revoked_digests.contains(&digest.value) {
            return ComponentTrustDecision::new(ComponentTrustStatus::Revoked, "digest revoked");
        }
        if self.quarantined_digests.contains(&digest.value) {
            return ComponentTrustDecision::new(
                ComponentTrustStatus::Quarantined,
                "digest quarantined",
            );
        }
        if self.rejected_digests.contains(&digest.value) {
            return ComponentTrustDecision::new(ComponentTrustStatus::Rejected, "digest rejected");
        }
        if self.trusted_digests.contains(&digest.value) {
            return ComponentTrustDecision::new(ComponentTrustStatus::Trusted, "digest trusted");
        }
        let matched_unauthenticated_metadata = if let Some(publisher) = &manifest.publisher
            && self.trusted_publishers.contains(&publisher.id)
        {
            true
        } else {
            self.trusted_sources.contains(&manifest.source.kind)
        };
        if self.allow_unsigned_local_development && manifest.source.kind == "local" {
            return ComponentTrustDecision::new(
                ComponentTrustStatus::Trusted,
                "explicit development local trust",
            );
        }
        if matched_unauthenticated_metadata {
            return ComponentTrustDecision::new(
                ComponentTrustStatus::Unknown,
                "publisher/source identity is metadata only; no authenticated trust mechanism matched",
            );
        }
        ComponentTrustDecision::new(ComponentTrustStatus::Unknown, "no trust policy matched")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentManifest {
    pub component: ComponentMetadata,
    pub role: String,
    pub digest: ComponentDigest,
    pub runtime_min_version: String,
    pub runtime_max_version: Option<String>,
    pub imports: BTreeSet<WitInterface>,
    pub optional_imports: BTreeSet<WitInterface>,
    pub exports: BTreeSet<WitInterface>,
    pub capabilities: Vec<ComponentCapabilityRequirement>,
    pub engine: ComponentEngineRequirements,
    pub authority: Vec<ComponentAuthorityRequirement>,
    pub publisher: Option<ComponentPublisher>,
    pub source: ComponentSource,
    pub signatures: Vec<ComponentSignature>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentCapabilityRequirement {
    pub id: String,
    pub min_version: String,
    pub max_version: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentEngineRequirements {
    pub profile: Option<ComponentEngineProfile>,
    pub features: BTreeSet<ComponentEngineFeature>,
}

impl ComponentEngineRequirements {
    pub fn require_profile(mut self, profile: ComponentEngineProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    pub fn require_feature(mut self, feature: ComponentEngineFeature) -> Self {
        self.features.insert(feature);
        self
    }

    pub fn validate(
        &self,
        component: &str,
        capabilities: &ComponentEngineCapabilities,
    ) -> Result<(), ComponentError> {
        if let Some(required) = self.profile
            && capabilities.profile != required
        {
            return Err(ComponentError::EngineProfileMismatch {
                component: component.into(),
                required,
                actual: capabilities.profile,
            });
        }
        for feature in &self.features {
            if !capabilities.supports(*feature) {
                return Err(ComponentError::EngineFeatureUnavailable {
                    component: component.into(),
                    feature: *feature,
                    profile: capabilities.profile,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentAuthorityRequirement {
    pub kind: String,
}

impl ComponentAuthorityRequirement {
    pub fn endpoint(&self) -> ComponentAuthorityEndpoint {
        authority_endpoint_for_kind(&self.kind)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InferenceArtifactKind {
    Model,
    Tokenizer,
    PromptTemplate,
    Adapter,
    Quantization,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InferenceCacheKind {
    Kv,
    Prefix,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceArtifactReference {
    pub kind: InferenceArtifactKind,
    pub id: String,
    pub digest: ComponentDigest,
    pub session: Option<InferenceSessionId>,
}

impl InferenceArtifactReference {
    pub fn new(
        kind: InferenceArtifactKind,
        id: impl Into<String>,
        digest: ComponentDigest,
    ) -> Result<Self, ComponentError> {
        let id = id.into();
        validate_runtime_identity(&id).map_err(|message| ComponentError::ArtifactRejected {
            component: id.clone(),
            status: ComponentTrustStatus::Rejected,
            message: message.into(),
        })?;
        Ok(Self {
            kind,
            id,
            digest,
            session: None,
        })
    }

    pub fn with_session(mut self, session: InferenceSessionId) -> Self {
        self.session = Some(session);
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InferenceArtifactRegistry {
    entries: BTreeMap<(InferenceArtifactKind, String), InferenceArtifactReference>,
}

impl InferenceArtifactRegistry {
    pub fn register(&mut self, artifact: InferenceArtifactReference) -> Result<(), ComponentError> {
        self.entries
            .insert((artifact.kind, artifact.id.clone()), artifact);
        Ok(())
    }

    pub fn resolve(
        &self,
        kind: InferenceArtifactKind,
        id: &str,
        session: Option<&InferenceSessionId>,
    ) -> Result<&InferenceArtifactReference, ComponentError> {
        if id.contains('/') || id.contains('\\') || id.contains(':') {
            return Err(ComponentError::ArtifactRejected {
                component: id.into(),
                status: ComponentTrustStatus::Rejected,
                message: "inference artifact access requires a registered artifact identity".into(),
            });
        }
        let artifact = self.entries.get(&(kind, id.into())).ok_or_else(|| {
            ComponentError::ArtifactRejected {
                component: id.into(),
                status: ComponentTrustStatus::Rejected,
                message: "inference artifact is not registered".into(),
            }
        })?;
        if let Some(expected) = &artifact.session
            && Some(expected) != session
        {
            return Err(ComponentError::ArtifactRejected {
                component: id.into(),
                status: ComponentTrustStatus::Rejected,
                message: "inference artifact is not authorized for this session".into(),
            });
        }
        Ok(artifact)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct InferenceCacheScope {
    pub kind: InferenceCacheKind,
    pub session: InferenceSessionId,
    pub model_artifact_id: String,
}

impl InferenceCacheScope {
    pub fn new(
        kind: InferenceCacheKind,
        session: InferenceSessionId,
        model_artifact_id: impl Into<String>,
    ) -> Result<Self, ComponentError> {
        let model_artifact_id = model_artifact_id.into();
        validate_runtime_identity(&model_artifact_id).map_err(|message| {
            ComponentError::ArtifactRejected {
                component: model_artifact_id.clone(),
                status: ComponentTrustStatus::Rejected,
                message: message.into(),
            }
        })?;
        Ok(Self {
            kind,
            session,
            model_artifact_id,
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InferenceCacheRegistry {
    scopes: BTreeSet<InferenceCacheScope>,
}

impl InferenceCacheRegistry {
    pub fn authorize(&mut self, scope: InferenceCacheScope) {
        self.scopes.insert(scope);
    }

    pub fn authorize_access(&self, scope: &InferenceCacheScope) -> Result<(), ComponentError> {
        if self.scopes.contains(scope) {
            Ok(())
        } else {
            Err(ComponentError::ArtifactRejected {
                component: scope.model_artifact_id.clone(),
                status: ComponentTrustStatus::Rejected,
                message: "cache access is not authorized for this session and model".into(),
            })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComponentAuthorityEndpoint {
    Capability { interface: WitInterface },
    RuntimeService { interface: WitInterface },
    InferenceArtifactRegistry { kind: InferenceArtifactKind },
    InferenceCacheService { kind: InferenceCacheKind },
    Observability,
    RuntimeDiagnostics,
    PendingRuntimeService { authority: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentPublisher {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentSource {
    pub kind: String,
    pub uri: String,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentDistributionSourceKind {
    LocalDirectory,
    LocalCache,
    ClientProvided,
    DevelopmentFixture,
    ExternalRegistry,
    Tachyon,
}

impl ComponentDistributionSourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalDirectory => "local-directory",
            Self::LocalCache => "local-cache",
            Self::ClientProvided => "client-provided",
            Self::DevelopmentFixture => "development-fixture",
            Self::ExternalRegistry => "external-registry",
            Self::Tachyon => "tachyon",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDistributionSource {
    pub kind: ComponentDistributionSourceKind,
    pub identity: String,
}

impl ComponentDistributionSource {
    pub fn new(kind: ComponentDistributionSourceKind, identity: impl Into<String>) -> Self {
        Self {
            kind,
            identity: identity.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentProvenance {
    pub builder: Option<String>,
    pub source_repository: Option<String>,
    pub commit_digest: Option<String>,
    pub build_timestamp: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentArtifactPackage {
    pub component_bytes: Vec<u8>,
    pub manifest_bytes: Vec<u8>,
    pub declared_digest: ComponentDigest,
    pub source: ComponentDistributionSource,
    pub publisher: Option<ComponentPublisher>,
    pub signatures: Vec<ComponentSignature>,
    pub provenance: Option<ComponentProvenance>,
}

impl ComponentArtifactPackage {
    pub fn new(
        component_bytes: Vec<u8>,
        manifest_bytes: Vec<u8>,
        declared_digest: ComponentDigest,
        source: ComponentDistributionSource,
    ) -> Self {
        Self {
            component_bytes,
            manifest_bytes,
            declared_digest,
            source,
            publisher: None,
            signatures: Vec::new(),
            provenance: None,
        }
    }

    pub fn with_publisher(mut self, publisher: ComponentPublisher) -> Self {
        self.publisher = Some(publisher);
        self
    }

    pub fn with_signature(mut self, signature: ComponentSignature) -> Self {
        self.signatures.push(signature);
        self
    }

    pub fn with_provenance(mut self, provenance: ComponentProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
}

pub trait ComponentDistributionSourceProvider {
    fn resolve(
        &self,
        component: &str,
        version_requirement: Option<&str>,
    ) -> Result<Vec<ComponentDigest>, ComponentError>;
    fn fetch(&self, digest: &ComponentDigest) -> Result<ComponentArtifactPackage, ComponentError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentSignature {
    pub algorithm: Option<String>,
    pub digest: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentArtifactCache {
    entries: BTreeMap<String, Vec<u8>>,
}

impl ComponentArtifactCache {
    pub fn insert(&mut self, bytes: Vec<u8>) -> ComponentDigest {
        let digest = ComponentDigest::sha256(&bytes);
        self.entries.insert(digest.value.clone(), bytes);
        digest
    }

    pub fn get_verified(&self, digest: &ComponentDigest) -> Result<Option<&[u8]>, ComponentError> {
        let Some(bytes) = self.entries.get(&digest.value) else {
            return Ok(None);
        };
        let computed = ComponentDigest::sha256(bytes);
        if &computed != digest {
            return Err(ComponentError::ArtifactRejected {
                component: "cache".into(),
                status: ComponentTrustStatus::Rejected,
                message: "cached artifact digest does not match cache key".into(),
            });
        }
        Ok(Some(bytes))
    }

    pub fn contains_untrusted(&self, digest: &ComponentDigest) -> bool {
        self.entries.contains_key(&digest.value)
    }

    #[cfg(test)]
    pub(crate) fn insert_unchecked_for_test(&mut self, digest: ComponentDigest, bytes: Vec<u8>) {
        self.entries.insert(digest.value, bytes);
    }
}

impl ComponentManifest {
    pub fn load_yaml(path: impl AsRef<Path>) -> Result<Self, ComponentError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|source| ComponentError::Manifest {
            path: path.into(),
            message: source.to_string(),
            source: Some(source),
        })?;
        let raw: ComponentManifestYaml =
            serde_norway::from_str(&content).map_err(|source| ComponentError::Manifest {
                path: path.into(),
                message: source.to_string(),
                source: None,
            })?;
        raw.validate(path)
    }
}

#[derive(Deserialize)]
struct ComponentManifestYaml {
    schema: String,
    schema_version: u64,
    artifact: ManifestArtifactYaml,
    component: ManifestComponentYaml,
    runtime: ManifestRuntimeYaml,
    wit: ManifestWitYaml,
    capabilities: ManifestCapabilitiesYaml,
    #[serde(default)]
    engine: Option<ManifestEngineYaml>,
    authority: ManifestAuthorityYaml,
    publisher: Option<ComponentPublisherYaml>,
    source: ComponentSourceYaml,
    #[serde(default)]
    signatures: Vec<ComponentSignatureYaml>,
}

#[derive(Deserialize)]
struct ManifestArtifactYaml {
    kind: String,
    digest: ManifestDigestYaml,
}

#[derive(Deserialize)]
struct ManifestDigestYaml {
    algorithm: String,
    value: String,
}

#[derive(Deserialize)]
struct ManifestComponentYaml {
    name: String,
    version: String,
    #[serde(default)]
    description: String,
    role: String,
}

#[derive(Deserialize)]
struct ManifestRuntimeYaml {
    magnetar: ManifestMagnetarRuntimeYaml,
}

#[derive(Deserialize)]
struct ManifestMagnetarRuntimeYaml {
    min_version: String,
    max_version: Option<String>,
}

#[derive(Deserialize, Default)]
struct ManifestWitYaml {
    #[serde(default)]
    imports: Vec<ManifestWitInterfaceYaml>,
    #[serde(default)]
    exports: Vec<ManifestWitInterfaceYaml>,
}

#[derive(Deserialize)]
struct ManifestWitInterfaceYaml {
    package: String,
    interface: String,
    version: String,
    #[serde(default)]
    optional: bool,
}

#[derive(Deserialize, Default)]
struct ManifestCapabilitiesYaml {
    #[serde(default)]
    requires: Vec<ManifestCapabilityYaml>,
}

#[derive(Deserialize)]
struct ManifestEngineYaml {
    profile: Option<String>,
    #[serde(default)]
    features: Vec<String>,
}

#[derive(Deserialize)]
struct ManifestCapabilityYaml {
    id: String,
    version: String,
    max_version: Option<String>,
}

#[derive(Deserialize, Default)]
struct ManifestAuthorityYaml {
    #[serde(default)]
    requires: Vec<ManifestAuthorityRequirementYaml>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ManifestAuthorityRequirementYaml {
    String(String),
    Object { kind: String },
}

#[derive(Deserialize)]
struct ComponentPublisherYaml {
    id: String,
    name: String,
}

#[derive(Deserialize)]
struct ComponentSourceYaml {
    kind: String,
    uri: String,
}

#[derive(Deserialize)]
struct ComponentSignatureYaml {
    algorithm: Option<String>,
    digest: Option<String>,
}

impl ComponentManifestYaml {
    fn validate(self, path: &Path) -> Result<ComponentManifest, ComponentError> {
        if self.schema != COMPONENT_ARTIFACT_SCHEMA {
            return Err(manifest_validation_error(
                path,
                "unsupported manifest schema",
            ));
        }
        if self.schema_version != COMPONENT_ARTIFACT_SCHEMA_VERSION {
            return Err(manifest_validation_error(
                path,
                "unsupported manifest schema version",
            ));
        }
        if self.artifact.kind != "component" {
            return Err(manifest_validation_error(
                path,
                "artifact kind must be component",
            ));
        }
        validate_component_name(&self.component.name)
            .map_err(|message| manifest_validation_error(path, message))?;
        validate_semver(&self.component.version)
            .map_err(|message| manifest_validation_error(path, message))?;
        validate_semver(&self.runtime.magnetar.min_version)
            .map_err(|message| manifest_validation_error(path, message))?;
        if let Some(max_version) = &self.runtime.magnetar.max_version {
            validate_semver(max_version)
                .map_err(|message| manifest_validation_error(path, message))?;
        }
        if self.component.role.trim().is_empty() {
            return Err(manifest_validation_error(
                path,
                "component role must not be empty",
            ));
        }
        if self.source.kind.trim().is_empty() || self.source.uri.trim().is_empty() {
            return Err(manifest_validation_error(
                path,
                "source kind and uri are required",
            ));
        }

        let digest =
            ComponentDigest::parse(self.artifact.digest.algorithm, self.artifact.digest.value);
        if digest.algorithm != "sha256" || !is_sha256_digest(&digest.value) {
            return Err(manifest_validation_error(
                path,
                "artifact digest must be canonical sha256:<64 lowercase hex>",
            ));
        }

        let mut imports = BTreeSet::new();
        let mut optional_imports = BTreeSet::new();
        for interface in self.wit.imports {
            let optional = interface.optional;
            let interface = wit_interface_from_manifest(interface, path)?;
            if optional {
                optional_imports.insert(interface);
            } else {
                imports.insert(interface);
            }
        }
        let exports = self
            .wit
            .exports
            .into_iter()
            .map(|interface| wit_interface_from_manifest(interface, path))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let capabilities = self
            .capabilities
            .requires
            .into_iter()
            .map(|capability| {
                validate_wit_identity(&capability.id)
                    .map_err(|message| manifest_validation_error(path, message))?;
                validate_semver(&capability.version)
                    .map_err(|message| manifest_validation_error(path, message))?;
                if let Some(max_version) = &capability.max_version {
                    validate_semver(max_version)
                        .map_err(|message| manifest_validation_error(path, message))?;
                }
                Ok(ComponentCapabilityRequirement {
                    id: capability.id,
                    min_version: capability.version,
                    max_version: capability.max_version,
                })
            })
            .collect::<Result<Vec<_>, ComponentError>>()?;
        let mut engine = ComponentEngineRequirements::default();
        if let Some(raw_engine) = self.engine {
            if let Some(profile) = raw_engine.profile {
                engine.profile = Some(
                    ComponentEngineProfile::from_manifest_value(&profile).ok_or_else(|| {
                        manifest_validation_error(path, "engine profile is invalid")
                    })?,
                );
            }
            engine.features = raw_engine
                .features
                .into_iter()
                .map(|feature| {
                    ComponentEngineFeature::from_manifest_value(&feature)
                        .ok_or_else(|| manifest_validation_error(path, "engine feature is invalid"))
                })
                .collect::<Result<BTreeSet<_>, ComponentError>>()?;
        }
        let authority = self
            .authority
            .requires
            .into_iter()
            .map(|requirement| {
                let kind = match requirement {
                    ManifestAuthorityRequirementYaml::String(kind) => kind,
                    ManifestAuthorityRequirementYaml::Object { kind } => kind,
                };
                let kind = kind.trim().to_ascii_lowercase();
                validate_authority_kind(&kind)
                    .map_err(|message| manifest_validation_error(path, message))?;
                Ok(ComponentAuthorityRequirement { kind })
            })
            .collect::<Result<Vec<_>, ComponentError>>()?;
        let publisher = self.publisher.map(|publisher| ComponentPublisher {
            id: publisher.id,
            name: publisher.name,
        });
        let signatures = self
            .signatures
            .into_iter()
            .map(|signature| ComponentSignature {
                algorithm: signature.algorithm,
                digest: signature.digest,
            })
            .collect();

        Ok(ComponentManifest {
            component: ComponentMetadata {
                name: self.component.name,
                version: self.component.version,
                description: self.component.description,
                imports: imports.clone(),
                exports: exports.clone(),
            },
            role: self.component.role,
            digest,
            runtime_min_version: self.runtime.magnetar.min_version,
            runtime_max_version: self.runtime.magnetar.max_version,
            imports,
            optional_imports,
            exports,
            capabilities,
            engine,
            authority,
            publisher,
            source: ComponentSource {
                kind: self.source.kind,
                uri: self.source.uri,
            },
            signatures,
        })
    }
}

#[derive(Deserialize)]
struct TrustStoreYaml {
    schema: String,
    schema_version: u64,
    #[serde(default)]
    trusted_digests: BTreeSet<String>,
    #[serde(default)]
    rejected_digests: BTreeSet<String>,
    #[serde(default)]
    revoked_digests: BTreeSet<String>,
    #[serde(default)]
    quarantined_digests: BTreeSet<String>,
    #[serde(default)]
    trusted_publishers: BTreeSet<String>,
    #[serde(default)]
    trusted_sources: BTreeSet<String>,
    development: Option<TrustStoreDevelopmentYaml>,
}

#[derive(Deserialize)]
struct TrustStoreDevelopmentYaml {
    #[serde(default)]
    allow_unsigned_local: bool,
}

impl TrustStoreYaml {
    fn validate(self, path: &Path) -> Result<ComponentTrustStore, ComponentError> {
        if self.schema != COMPONENT_TRUST_SCHEMA {
            return Err(trust_store_error(path, "unsupported trust store schema"));
        }
        if self.schema_version != COMPONENT_ARTIFACT_SCHEMA_VERSION {
            return Err(trust_store_error(
                path,
                "unsupported trust store schema version",
            ));
        }
        for digest in self
            .trusted_digests
            .iter()
            .chain(self.rejected_digests.iter())
            .chain(self.revoked_digests.iter())
            .chain(self.quarantined_digests.iter())
        {
            if !is_sha256_digest(digest) {
                return Err(trust_store_error(
                    path,
                    "trust store digests must be canonical sha256 values",
                ));
            }
        }
        Ok(ComponentTrustStore {
            trusted_digests: lower_set(self.trusted_digests),
            rejected_digests: lower_set(self.rejected_digests),
            revoked_digests: lower_set(self.revoked_digests),
            quarantined_digests: lower_set(self.quarantined_digests),
            trusted_publishers: self.trusted_publishers,
            trusted_sources: self.trusted_sources,
            allow_unsigned_local_development: self
                .development
                .is_some_and(|development| development.allow_unsigned_local),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedComponent {
    definition_id: ComponentDefinitionId,
    engine_key: String,
    contract: ComponentContract,
}

impl PreparedComponent {
    pub fn new(definition_id: ComponentDefinitionId, engine_key: impl Into<String>) -> Self {
        Self::with_contract(definition_id, engine_key, ComponentContract::default())
    }

    pub fn with_contract(
        definition_id: ComponentDefinitionId,
        engine_key: impl Into<String>,
        contract: ComponentContract,
    ) -> Self {
        Self {
            definition_id,
            engine_key: engine_key.into(),
            contract,
        }
    }

    pub const fn definition_id(&self) -> ComponentDefinitionId {
        self.definition_id
    }

    pub fn engine_key(&self) -> &str {
        &self.engine_key
    }

    pub const fn contract(&self) -> &ComponentContract {
        &self.contract
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentEngineInstance {
    definition_id: ComponentDefinitionId,
    engine_key: String,
}

impl ComponentEngineInstance {
    pub fn new(definition_id: ComponentDefinitionId, engine_key: impl Into<String>) -> Self {
        Self {
            definition_id,
            engine_key: engine_key.into(),
        }
    }

    pub const fn definition_id(&self) -> ComponentDefinitionId {
        self.definition_id
    }

    pub fn engine_key(&self) -> &str {
        &self.engine_key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInstance {
    pub id: ComponentInstanceId,
    pub definition_id: ComponentDefinitionId,
    pub state: ComponentInstanceState,
    engine_instance: ComponentEngineInstance,
}

impl ComponentInstance {
    pub const fn engine_instance(&self) -> &ComponentEngineInstance {
        &self.engine_instance
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComponentEndpoint {
    RuntimeService { interface: WitInterface },
    Capability { interface: WitInterface },
}

impl ComponentEndpoint {
    pub const fn interface(&self) -> &WitInterface {
        match self {
            Self::RuntimeService { interface } | Self::Capability { interface } => interface,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentLinkPlan {
    links: BTreeMap<WitInterface, ComponentEndpoint>,
}

impl ComponentLinkPlan {
    pub fn links(&self) -> impl Iterator<Item = (&WitInterface, &ComponentEndpoint)> {
        self.links.iter()
    }

    pub fn endpoint(&self, interface: &WitInterface) -> Option<&ComponentEndpoint> {
        self.links.get(interface)
    }

    #[cfg(all(test, feature = "wasmtime-component-engine"))]
    pub(crate) fn insert_for_test(&mut self, endpoint: ComponentEndpoint) {
        self.links.insert(endpoint.interface().clone(), endpoint);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentResourceLimits {
    pub max_memory_bytes: Option<u64>,
    pub execution_deadline_millis: Option<u64>,
    pub max_concurrent_invocations: Option<u32>,
    pub max_instances: Option<u32>,
    pub engine_execution_budget: Option<u64>,
    pub require_memory_limit: bool,
}

impl Default for ComponentResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: None,
            execution_deadline_millis: None,
            max_concurrent_invocations: Some(1),
            max_instances: None,
            engine_execution_budget: None,
            require_memory_limit: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentObservationKind {
    Distribution,
    EngineSelection,
    EngineRejection,
    PlatformUnsupported,
    Validation,
    Preparation,
    LinkPlan,
    Instantiation,
    Invocation,
    Trap,
    Interruption,
    ResourceLimit,
    Destruction,
    Shutdown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentObservation {
    pub kind: ComponentObservationKind,
    pub component: Option<String>,
    pub instance: Option<ComponentInstanceId>,
    pub message: String,
}

impl ComponentObservation {
    pub fn new(
        kind: ComponentObservationKind,
        component: Option<String>,
        instance: Option<ComponentInstanceId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            component,
            instance,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentEngineProfile {
    #[default]
    Test,
    Native,
    Web,
}

impl ComponentEngineProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "component-engine-native",
            Self::Web => "component-engine-web",
            Self::Test => "component-engine-test",
        }
    }

    fn from_manifest_value(value: &str) -> Option<Self> {
        match value {
            "component-engine-native" | "native" => Some(Self::Native),
            "component-engine-web" | "web" => Some(Self::Web),
            "component-engine-test" | "test" => Some(Self::Test),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentEngineFeature {
    ComponentModel,
    AsyncHostCalls,
    Interruption,
    ResourceLimits,
    BrowserCompatible,
    NativeProviderEndpoints,
    ControlledWasi,
    JsMediatedHostCalls,
    BrowserMemory,
}

impl ComponentEngineFeature {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ComponentModel => "component-model",
            Self::AsyncHostCalls => "async-host-calls",
            Self::Interruption => "interruption",
            Self::ResourceLimits => "resource-limits",
            Self::BrowserCompatible => "browser-compatible",
            Self::NativeProviderEndpoints => "native-provider-endpoints",
            Self::ControlledWasi => "controlled-wasi",
            Self::JsMediatedHostCalls => "js-mediated-host-calls",
            Self::BrowserMemory => "browser-memory",
        }
    }

    fn from_manifest_value(value: &str) -> Option<Self> {
        match value {
            "component-model" => Some(Self::ComponentModel),
            "async-host-calls" => Some(Self::AsyncHostCalls),
            "interruption" => Some(Self::Interruption),
            "resource-limits" => Some(Self::ResourceLimits),
            "browser-compatible" => Some(Self::BrowserCompatible),
            "native-provider-endpoints" => Some(Self::NativeProviderEndpoints),
            "controlled-wasi" => Some(Self::ControlledWasi),
            "js-mediated-host-calls" => Some(Self::JsMediatedHostCalls),
            "browser-memory" => Some(Self::BrowserMemory),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentEngineCapabilities {
    pub profile: ComponentEngineProfile,
    pub component_model: bool,
    pub async_host_calls: bool,
    pub interruption: bool,
    pub resource_limits: bool,
    pub browser_compatible: bool,
    pub native_provider_endpoints: bool,
    pub controlled_wasi: bool,
    pub js_mediated_host_calls: bool,
    pub browser_memory: bool,
}

impl Default for ComponentEngineCapabilities {
    fn default() -> Self {
        Self::test()
    }
}

impl ComponentEngineCapabilities {
    pub const fn native() -> Self {
        Self {
            profile: ComponentEngineProfile::Native,
            component_model: true,
            async_host_calls: true,
            interruption: true,
            resource_limits: true,
            browser_compatible: false,
            native_provider_endpoints: true,
            controlled_wasi: true,
            js_mediated_host_calls: false,
            browser_memory: false,
        }
    }

    pub const fn web() -> Self {
        Self {
            profile: ComponentEngineProfile::Web,
            component_model: true,
            async_host_calls: true,
            interruption: false,
            resource_limits: false,
            browser_compatible: true,
            native_provider_endpoints: false,
            controlled_wasi: false,
            js_mediated_host_calls: true,
            browser_memory: true,
        }
    }

    pub const fn test() -> Self {
        Self {
            profile: ComponentEngineProfile::Test,
            component_model: true,
            async_host_calls: true,
            interruption: true,
            resource_limits: true,
            browser_compatible: true,
            native_provider_endpoints: false,
            controlled_wasi: false,
            js_mediated_host_calls: true,
            browser_memory: true,
        }
    }

    pub const fn supports(&self, feature: ComponentEngineFeature) -> bool {
        match feature {
            ComponentEngineFeature::ComponentModel => self.component_model,
            ComponentEngineFeature::AsyncHostCalls => self.async_host_calls,
            ComponentEngineFeature::Interruption => self.interruption,
            ComponentEngineFeature::ResourceLimits => self.resource_limits,
            ComponentEngineFeature::BrowserCompatible => self.browser_compatible,
            ComponentEngineFeature::NativeProviderEndpoints => self.native_provider_endpoints,
            ComponentEngineFeature::ControlledWasi => self.controlled_wasi,
            ComponentEngineFeature::JsMediatedHostCalls => self.js_mediated_host_calls,
            ComponentEngineFeature::BrowserMemory => self.browser_memory,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentTrapKind {
    Trap,
    Unreachable,
    MemoryFault,
    ResourceLimit,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentInterruptionReason {
    CallerCancelled,
    RuntimeShutdown,
    Deadline,
    ResourcePolicy,
    Administrative,
}

// Not `Eq`: `arguments` can carry `ComponentValue::F64`, and `f64` has no
// total order/equality, so `ComponentValue` (and everything embedding it)
// is `PartialEq`-only from here down.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentInvocation {
    pub instance_id: ComponentInstanceId,
    pub interface: WitInterface,
    pub operation: String,
    pub deadline_millis: Option<u64>,
    pub arguments: Vec<ComponentValue>,
}

impl ComponentInvocation {
    pub fn new(
        instance_id: ComponentInstanceId,
        interface: WitInterface,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            instance_id,
            interface,
            operation: operation.into(),
            deadline_millis: None,
            arguments: Vec::new(),
        }
    }

    /// Attaches call arguments (`model-component-graph-contract`): a caller
    /// invoking a Component export that takes real parameters (not just the
    /// legacy zero-arg/`u32`-result shape) supplies them here instead of
    /// mutating `arguments` directly, matching `WitInterface::new`'s
    /// builder-free-function style.
    pub fn with_arguments(mut self, arguments: Vec<ComponentValue>) -> Self {
        self.arguments = arguments;
        self
    }
}

/// A dynamically-typed WIT value (`model-component-graph-contract`): the
/// value shape [`ComponentInvocation::arguments`]/[`ComponentInvocationResult`]
/// carry across the native Component Model boundary, and what a
/// [`HostCapability`] exchanges with a calling Component. Deliberately not a
/// 1:1 mirror of every `wasmtime::component::Val` case -- only the shapes
/// this repo's WIT interfaces (`compute.wit`, `observability.wit`, and the
/// forthcoming graph-builder interface) actually need: no `map`, `tuple`,
/// `flags`, `resource`, `future`, `stream`, or `error-context` case, since
/// nothing here uses those WIT constructs. `Enum` and `Variant` are kept
/// distinct (rather than collapsing `Enum` into a payload-less `Variant`)
/// because a Wasmtime `Type::Enum` and `Type::Variant` are different target
/// shapes when converting back to `wasmtime::component::Val` -- collapsing
/// them would lose the information needed to pick the right one.
#[derive(Clone, Debug, PartialEq)]
pub enum ComponentValue {
    Bool(bool),
    U32(u32),
    S64(i64),
    F64(f64),
    String(String),
    List(Vec<ComponentValue>),
    /// A WIT `record`: field name/value pairs in declaration order.
    Record(Vec<(String, ComponentValue)>),
    /// A WIT `variant` case: its name and optional payload.
    Variant(String, Option<Box<ComponentValue>>),
    /// A WIT `enum` case: its name (no payload -- an `enum` case never
    /// carries one; a case that does is a `variant`, represented above).
    Enum(String),
    Option(Option<Box<ComponentValue>>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ComponentInvocationResult {
    pub values: Vec<ComponentValue>,
}

impl ComponentInvocationResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn single(value: ComponentValue) -> Self {
        Self {
            values: vec![value],
        }
    }
}

/// A Runtime-provided capability a Component calls into as a host import
/// (`model-component-graph-contract`): the counterpart to a Component
/// *export* the Runtime calls (`ComponentEngine::invoke`). `instance_key` is
/// the calling Component instance's own [`ComponentEngineInstance::engine_key`]
/// -- a single `HostCapability` is registered once and shared across every
/// Component instance that imports its interface, so a capability that needs
/// per-instance state (a graph-builder session under construction, for
/// example) uses this to key it; a capability with no such state can ignore
/// it. `operation` is the WIT function name within the imported interface;
/// `arguments` are already the callee's declared parameters, in order,
/// converted from the Component's own typed call. A capability that rejects
/// a call (invalid arguments, a semantic violation the capability itself is
/// responsible for validating) returns `Err`, which the calling engine
/// surfaces to the Component as a trapped/failed host call -- never a silent
/// default value, matching this crate's fail-closed posture for every other
/// Provider/Kernel boundary.
pub trait HostCapability: Send + Sync {
    fn call(
        &self,
        instance_key: &str,
        operation: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError>;
}

pub trait ComponentEngine: Send {
    fn capabilities(&self) -> ComponentEngineCapabilities;
    fn inspect_contract(
        &mut self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError>;
    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError>;
    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError>;
    fn invoke(
        &mut self,
        instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError>;
    fn interrupt(
        &mut self,
        instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError>;
    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError>;

    /// Registers a real implementation for a host-provided `WitInterface`
    /// (`model-component-graph-contract`), so an engine backend that
    /// actually links Components (`WasmtimeComponentEngine`) can wire a
    /// Component's import of that interface to `capability`'s real behavior
    /// instead of a generic conformance stub. Additive and optional, like
    /// `ProviderExecutionApi`'s defaulted methods: an engine backend that
    /// never links real Components (`MockComponentEngine`, `WebComponentEngine`)
    /// has no meaningful behavior to add here, so the default is a no-op
    /// rather than a required override.
    fn register_capability(
        &mut self,
        interface: WitInterface,
        capability: Arc<dyn HostCapability>,
    ) {
        let _ = (interface, capability);
    }
}

#[derive(Default)]
pub struct MockComponentEngine {
    capabilities: ComponentEngineCapabilities,
    pub prepared: Vec<ComponentDefinitionId>,
    pub instantiated: Vec<ComponentDefinitionId>,
    pub destroyed: Vec<String>,
    pub invoked: Vec<ComponentInvocation>,
    pub interrupted: Vec<ComponentInterruptionReason>,
    pub fail_prepare: Option<String>,
    pub fail_instantiate: Option<String>,
    pub trap_on_invoke: Option<ComponentTrapKind>,
    pub interrupt_on_invoke: Option<ComponentInterruptionReason>,
    pub prepared_contract: Option<ComponentContract>,
}

impl MockComponentEngine {
    pub fn new() -> Self {
        Self {
            capabilities: ComponentEngineCapabilities::test(),
            ..Self::default()
        }
    }

    pub fn without_resource_limits(mut self) -> Self {
        self.capabilities.resource_limits = false;
        self
    }
}

impl ComponentEngine for MockComponentEngine {
    fn capabilities(&self) -> ComponentEngineCapabilities {
        self.capabilities.clone()
    }

    fn inspect_contract(
        &mut self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        Ok(self
            .prepared_contract
            .clone()
            .unwrap_or_else(|| ComponentContract::from_metadata(&definition.metadata)))
    }

    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError> {
        if limits.require_memory_limit && !self.capabilities.resource_limits {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }
        if let Some(message) = &self.fail_prepare {
            return Err(ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: message.clone(),
            });
        }
        self.prepared.push(definition.id);
        Ok(PreparedComponent::with_contract(
            definition.id,
            format!("prepared:{}", definition.metadata.name),
            self.prepared_contract
                .clone()
                .unwrap_or_else(|| ComponentContract::from_metadata(&definition.metadata)),
        ))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        _link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        if let Some(message) = &self.fail_instantiate {
            return Err(ComponentError::InstantiationFailed {
                definition: prepared.definition_id(),
                message: message.clone(),
            });
        }
        self.instantiated.push(prepared.definition_id());
        Ok(ComponentEngineInstance::new(
            prepared.definition_id(),
            format!("instance:{}", prepared.engine_key()),
        ))
    }

    fn invoke(
        &mut self,
        _instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        self.invoked.push(invocation.clone());
        if let Some(reason) = self.interrupt_on_invoke {
            return Err(ComponentError::Interrupted {
                instance: invocation.instance_id,
                reason,
            });
        }
        if let Some(kind) = self.trap_on_invoke {
            return Err(ComponentError::Trap {
                instance: invocation.instance_id,
                kind,
                diagnostic: Some("[redacted component trap]".into()),
            });
        }
        Ok(ComponentInvocationResult::empty())
    }

    fn interrupt(
        &mut self,
        _instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        self.interrupted.push(reason);
        Ok(())
    }

    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        self.destroyed.push(instance.engine_key);
        Ok(())
    }
}

pub struct ComponentManager {
    engine: Box<dyn ComponentEngine>,
    host_interfaces: BTreeSet<WitInterface>,
    authorized_interfaces: BTreeSet<WitInterface>,
    inference_artifacts: InferenceArtifactRegistry,
    inference_caches: InferenceCacheRegistry,
    definitions: BTreeMap<String, ComponentDefinition>,
    prepared: BTreeMap<ComponentDefinitionId, PreparedComponent>,
    instances: BTreeMap<ComponentInstanceId, ComponentInstance>,
    active_invocations: BTreeMap<ComponentInstanceId, u32>,
    observations: Vec<ComponentObservation>,
    limits: ComponentResourceLimits,
    trust_store: ComponentTrustStore,
    owned_distributed_package_dirs: Vec<PathBuf>,
    shutdown: bool,
}

impl Default for ComponentManager {
    fn default() -> Self {
        Self::with_engine(Box::new(MockComponentEngine::new()))
    }
}

impl ComponentManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_engine(engine: Box<dyn ComponentEngine>) -> Self {
        Self {
            engine,
            host_interfaces: BTreeSet::new(),
            authorized_interfaces: BTreeSet::new(),
            inference_artifacts: InferenceArtifactRegistry::default(),
            inference_caches: InferenceCacheRegistry::default(),
            definitions: BTreeMap::new(),
            prepared: BTreeMap::new(),
            instances: BTreeMap::new(),
            active_invocations: BTreeMap::new(),
            observations: Vec::new(),
            limits: ComponentResourceLimits::default(),
            trust_store: ComponentTrustStore::default(),
            owned_distributed_package_dirs: Vec::new(),
            shutdown: false,
        }
    }

    pub fn provide_interface(&mut self, interface: WitInterface) {
        self.host_interfaces.insert(interface.clone());
        self.authorized_interfaces.insert(interface);
    }

    /// Declares `interface` provided (same bookkeeping as
    /// [`Self::provide_interface`]) *and* wires `capability` as its real
    /// implementation on the underlying engine backend, so a Component
    /// importing `interface` calls into real Runtime behavior rather than a
    /// generic conformance stub. See [`HostCapability`] and
    /// [`ComponentEngine::register_capability`].
    pub fn provide_capability(
        &mut self,
        interface: WitInterface,
        capability: Arc<dyn HostCapability>,
    ) {
        self.provide_interface(interface.clone());
        self.engine.register_capability(interface, capability);
    }

    pub fn authorize_interface(&mut self, interface: WitInterface) {
        self.authorized_interfaces.insert(interface);
    }

    pub fn set_resource_limits(&mut self, limits: ComponentResourceLimits) {
        self.limits = limits;
    }

    pub fn set_trust_store(&mut self, trust_store: ComponentTrustStore) {
        self.trust_store = trust_store;
    }

    pub fn register_inference_artifact(
        &mut self,
        artifact: InferenceArtifactReference,
    ) -> Result<(), ComponentError> {
        self.inference_artifacts.register(artifact)
    }

    pub fn resolve_inference_artifact(
        &self,
        kind: InferenceArtifactKind,
        id: &str,
        session: Option<&InferenceSessionId>,
    ) -> Result<&InferenceArtifactReference, ComponentError> {
        self.inference_artifacts.resolve(kind, id, session)
    }

    pub fn authorize_inference_cache(&mut self, scope: InferenceCacheScope) {
        self.inference_caches.authorize(scope);
    }

    pub fn authorize_inference_cache_access(
        &self,
        scope: &InferenceCacheScope,
    ) -> Result<(), ComponentError> {
        self.inference_caches.authorize_access(scope)
    }

    pub fn observations(&self) -> &[ComponentObservation] {
        &self.observations
    }

    pub fn engine_capabilities(&self) -> ComponentEngineCapabilities {
        self.engine.capabilities()
    }

    pub fn register_component(
        &mut self,
        descriptor: ComponentDescriptor,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        if self.definitions.contains_key(&descriptor.metadata.name) {
            return Err(ComponentError::AlreadyRegistered(descriptor.metadata.name));
        }
        let definition = ComponentDefinition::registered(descriptor);
        let id = definition.id;
        self.definitions
            .insert(definition.metadata.name.clone(), definition);
        Ok(id)
    }

    pub fn definition(&self, name: &str) -> Option<&ComponentDefinition> {
        self.definitions.get(name)
    }

    pub fn definition_state(&self, name: &str) -> Option<ComponentDefinitionState> {
        self.definitions
            .get(name)
            .map(|definition| definition.state)
    }

    pub fn instance_state(&self, id: ComponentInstanceId) -> Option<ComponentInstanceState> {
        self.instances.get(&id).map(|instance| instance.state)
    }

    pub fn link_plan(&self, name: &str) -> Result<ComponentLinkPlan, ComponentError> {
        let definition = self
            .definitions
            .get(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))?;
        self.build_link_plan(definition)
    }

    pub fn prepare_component(
        &mut self,
        name: &str,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        let definition = self
            .definitions
            .get_mut(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))?;
        if definition.artifact_path.exists() {
            match validate_component_artifact(self.engine.as_mut(), definition, &self.trust_store) {
                Ok(outcome) => {
                    self.observations.extend(outcome.observations);
                    definition.metadata = outcome.manifest.component;
                    definition.artifact_digest = Some(outcome.digest);
                    definition.trust_decision = Some(outcome.trust_decision);
                }
                Err(error) => {
                    definition.state = ComponentDefinitionState::Failed;
                    self.observe_authority_or_validation_error(name, &error);
                    return Err(error);
                }
            }
        }
        definition.state = ComponentDefinitionState::Validated;
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Validation,
            Some(definition.metadata.name.clone()),
            None,
            "component definition validated",
        ));
        let capabilities = self.engine.capabilities();
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::EngineSelection,
            Some(definition.metadata.name.clone()),
            None,
            format!(
                "selected Component Engine profile '{}'",
                capabilities.profile.as_str()
            ),
        ));
        if let Err(error) = ComponentEngineRequirements::default()
            .validate(&definition.metadata.name, &capabilities)
        {
            definition.state = ComponentDefinitionState::Failed;
            self.observe_error(None, &error);
            return Err(error);
        }
        let prepared = match self.engine.prepare(definition, &self.limits) {
            Ok(prepared) => prepared,
            Err(error) => {
                definition.state = ComponentDefinitionState::Failed;
                self.observe_error(None, &error);
                return Err(error);
            }
        };
        if let Err(error) = validate_prepared_contract(definition, prepared.contract()) {
            definition.state = ComponentDefinitionState::Failed;
            self.observe_error(None, &error);
            return Err(error);
        }
        definition.state = ComponentDefinitionState::Prepared;
        let id = definition.id;
        self.prepared.insert(id, prepared);
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Preparation,
            Some(definition.metadata.name.clone()),
            None,
            "component prepared by engine",
        ));
        Ok(id)
    }

    pub fn prepare_pushed_package(
        &mut self,
        package: ComponentArtifactPackage,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        self.prepare_distributed_package(package, "package received")
    }

    pub fn prepare_pulled_package(
        &mut self,
        source: &dyn ComponentDistributionSourceProvider,
        component: &str,
        version_requirement: Option<&str>,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        let candidates = source.resolve(component, version_requirement)?;
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Distribution,
            Some(component.into()),
            None,
            format!(
                "source resolution returned {} candidate digest(s)",
                candidates.len()
            ),
        ));
        let digest = candidates
            .first()
            .ok_or_else(|| ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ArtifactNotFound,
                message: "source returned no candidate artifact digests".into(),
            })?;
        let package = source.fetch(digest)?;
        self.prepare_distributed_package(package, "fetch success")
    }

    pub fn instantiate_component(
        &mut self,
        name: &str,
    ) -> Result<ComponentInstanceId, ComponentError> {
        let definition_id = self.prepare_component(name)?;
        self.instantiate_prepared_component(definition_id)
    }

    pub fn instantiate_prepared_component(
        &mut self,
        definition_id: ComponentDefinitionId,
    ) -> Result<ComponentInstanceId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        let definition = self
            .definitions
            .values()
            .find(|definition| definition.id == definition_id)
            .ok_or(ComponentError::MissingPreparedDefinition(definition_id))?;
        let name = definition.metadata.name.clone();
        if definition.state != ComponentDefinitionState::Prepared {
            return Err(ComponentError::MissingPreparedDefinition(definition_id));
        }
        if let Some(max_instances) = self.limits.max_instances
            && self.instances.len() >= max_instances as usize
        {
            let error = ComponentError::ResourceLimitExceeded {
                component: name.clone(),
                limit: "instances",
            };
            self.observe_error(None, &error);
            return Err(error);
        }
        let link_plan = self.build_link_plan(definition)?;
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::LinkPlan,
            Some(name.clone()),
            None,
            "runtime-owned link plan built",
        ));
        let prepared = self
            .prepared
            .get(&definition_id)
            .ok_or(ComponentError::MissingPreparedDefinition(definition_id))?;
        let engine_instance = self.engine.instantiate(prepared, &link_plan)?;
        let id = next_component_instance_id();
        self.instances.insert(
            id,
            ComponentInstance {
                id,
                definition_id,
                state: ComponentInstanceState::Ready,
                engine_instance,
            },
        );
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Instantiation,
            Some(name),
            Some(id),
            "component instance ready",
        ));
        Ok(id)
    }

    pub fn invoke(
        &mut self,
        invocation: ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        if invocation.deadline_millis.is_some() && !self.engine.capabilities().interruption {
            let error = ComponentError::ResourceLimitUnsupported {
                component: format!("instance:{}", invocation.instance_id.get()),
                limit: "deadline",
            };
            self.observe_error(Some(invocation.instance_id), &error);
            return Err(error);
        }
        let instance = self
            .instances
            .get(&invocation.instance_id)
            .ok_or(ComponentError::InstanceNotFound(invocation.instance_id))?;
        if instance.state != ComponentInstanceState::Ready {
            return Err(ComponentError::InvalidInstanceTransition {
                instance: invocation.instance_id,
                state: instance.state,
                operation: "invoke",
            });
        }
        let active = self
            .active_invocations
            .entry(invocation.instance_id)
            .or_default();
        if let Some(max) = self.limits.max_concurrent_invocations
            && *active >= max
        {
            let error = ComponentError::ResourceLimitExceeded {
                component: format!("instance:{}", invocation.instance_id.get()),
                limit: "concurrent invocations",
            };
            self.observe_error(Some(invocation.instance_id), &error);
            return Err(error);
        }
        *active += 1;
        let engine_instance = instance.engine_instance.clone();
        let result = self.engine.invoke(&engine_instance, &invocation);
        if let Some(active) = self.active_invocations.get_mut(&invocation.instance_id) {
            *active = active.saturating_sub(1);
            if *active == 0 {
                self.active_invocations.remove(&invocation.instance_id);
            }
        }
        match &result {
            Ok(_) => self.observations.push(ComponentObservation::new(
                ComponentObservationKind::Invocation,
                None,
                Some(invocation.instance_id),
                "component invocation completed",
            )),
            Err(error) => self.observe_error(Some(invocation.instance_id), error),
        }
        result
    }

    pub fn destroy_instance(&mut self, id: ComponentInstanceId) -> Result<(), ComponentError> {
        let mut instance = self
            .instances
            .remove(&id)
            .ok_or(ComponentError::InstanceNotFound(id))?;
        if instance.state == ComponentInstanceState::Destroyed {
            return Err(ComponentError::InvalidInstanceTransition {
                instance: id,
                state: instance.state,
                operation: "destroy",
            });
        }
        instance.state = ComponentInstanceState::Destroyed;
        self.active_invocations.remove(&id);
        let result = self.engine.destroy(instance.engine_instance);
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Destruction,
            None,
            Some(id),
            "component instance destroyed",
        ));
        result
    }

    pub fn shutdown(&mut self) {
        self.shutdown = true;
        let ids = self.instances.keys().copied().collect::<Vec<_>>();
        for id in ids {
            if let Some(instance) = self.instances.get(&id) {
                let _ = self.engine.interrupt(
                    &instance.engine_instance,
                    ComponentInterruptionReason::RuntimeShutdown,
                );
            }
            let _ = self.destroy_instance(id);
        }
        self.prepared.clear();
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Shutdown,
            None,
            None,
            "component manager shutdown",
        ));
    }

    pub fn discover(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<Vec<PathBuf>, ComponentError> {
        let mut found = BTreeSet::new();
        for directory in paths {
            let directory = directory.as_ref();
            for entry in
                std::fs::read_dir(directory).map_err(|source| ComponentError::Discovery {
                    path: directory.into(),
                    source,
                })?
            {
                let path = entry
                    .map_err(|source| ComponentError::Discovery {
                        path: directory.into(),
                        source,
                    })?
                    .path();
                if path.is_file()
                    && path.extension().and_then(|extension| extension.to_str()) == Some("wasm")
                {
                    found.insert(path);
                }
            }
        }
        Ok(found.into_iter().collect())
    }

    fn prepare_distributed_package(
        &mut self,
        package: ComponentArtifactPackage,
        event: &'static str,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        let computed = ComponentDigest::sha256(&package.component_bytes);
        let source_kind = package.source.kind.as_str();
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Distribution,
            None,
            None,
            format!("{event} from {source_kind}"),
        ));
        if computed != package.declared_digest {
            let error = ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::DigestMismatch,
                message: "source-declared digest does not match received bytes".into(),
            };
            self.observe_error(None, &error);
            return Err(error);
        }

        let distribution_dir = std::env::temp_dir().join(format!(
            "magnetar-distributed-component-{}-{}",
            computed.value.replace(':', "-"),
            NEXT_DISTRIBUTED_PACKAGE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&distribution_dir).map_err(|source| ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::PolicyDenied,
            message: source.to_string(),
        })?;
        let artifact_path = distribution_dir.join("component.wasm");
        let manifest_path = distribution_dir.join("component.wasm.magnetar-component.yaml");
        fs::write(&artifact_path, &package.component_bytes).map_err(|source| {
            ComponentError::ComponentLoadFailed {
                path: artifact_path.clone(),
                message: source.to_string(),
                source: Some(source),
            }
        })?;
        fs::write(&manifest_path, &package.manifest_bytes).map_err(|source| {
            ComponentError::Manifest {
                path: manifest_path.clone(),
                message: source.to_string(),
                source: Some(source),
            }
        })?;

        let manifest = ComponentManifest::load_yaml(&manifest_path)?;
        if manifest.digest != computed {
            let error = ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::DigestMismatch,
                message: "manifest-declared digest does not match received bytes".into(),
            };
            self.observe_error(None, &error);
            return Err(error);
        }
        if manifest.component.name.contains("tool")
            || manifest.role.contains("tool")
            || manifest.role.contains("filesystem")
            || manifest.role.contains("shell")
        {
            let error = ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ForbiddenAuthority,
                message: "distributed package is outside Magnetar inference scope".into(),
            };
            self.observe_error(None, &error);
            return Err(error);
        }

        let descriptor = ComponentDescriptor::new(manifest.component.clone(), artifact_path)
            .with_manifest_path(manifest_path);
        if self.definitions.contains_key(&descriptor.metadata.name) {
            let _ = fs::remove_dir_all(&distribution_dir);
            return Err(ComponentError::AlreadyRegistered(descriptor.metadata.name));
        }
        let name = descriptor.metadata.name.clone();
        self.register_component(descriptor)?;
        match self.prepare_component(&name) {
            Ok(id) => {
                self.owned_distributed_package_dirs.push(distribution_dir);
                Ok(id)
            }
            Err(error) => {
                self.definitions.remove(&name);
                let _ = fs::remove_dir_all(&distribution_dir);
                Err(error)
            }
        }
    }

    fn build_link_plan(
        &self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentLinkPlan, ComponentError> {
        let mut links = BTreeMap::new();
        for interface in &definition.metadata.imports {
            if is_forbidden_external_interface(&interface.name) {
                return Err(ComponentError::UnauthorizedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            if !self.authorized_interfaces.contains(interface) {
                return Err(ComponentError::UnauthorizedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            if !self.host_interfaces.contains(interface) {
                return Err(ComponentError::UnresolvedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            if !is_inference_linkable_interface(&interface.name) {
                return Err(ComponentError::UnauthorizedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            links.insert(
                interface.clone(),
                ComponentEndpoint::Capability {
                    interface: interface.clone(),
                },
            );
        }
        Ok(ComponentLinkPlan { links })
    }

    fn observe_error(&mut self, instance: Option<ComponentInstanceId>, error: &ComponentError) {
        let kind = match error {
            ComponentError::Trap { .. } => ComponentObservationKind::Trap,
            ComponentError::Interrupted { .. } => ComponentObservationKind::Interruption,
            ComponentError::ResourceLimitUnsupported { .. }
            | ComponentError::ResourceLimitExceeded { .. } => {
                ComponentObservationKind::ResourceLimit
            }
            ComponentError::NoCompatibleEngine { .. }
            | ComponentError::EngineProfileMismatch { .. }
            | ComponentError::EngineFeatureUnavailable { .. }
            | ComponentError::WasmtimeUnavailable { .. }
            | ComponentError::BrowserEngineUnavailable { .. } => {
                ComponentObservationKind::EngineRejection
            }
            ComponentError::PlatformUnsupported { .. } => {
                ComponentObservationKind::PlatformUnsupported
            }
            ComponentError::HostBindingFailed { .. } => ComponentObservationKind::Instantiation,
            ComponentError::InstantiationFailed { .. } => ComponentObservationKind::Instantiation,
            ComponentError::InvocationFailed { .. }
            | ComponentError::CapabilityCallRejected { .. } => ComponentObservationKind::Invocation,
            ComponentError::PreparationFailed { .. }
            | ComponentError::ComponentLoadFailed { .. } => ComponentObservationKind::Preparation,
            ComponentError::UnauthorizedImport { .. }
            | ComponentError::UnresolvedImport { .. }
            | ComponentError::ContractValidationFailed { .. } => {
                ComponentObservationKind::Validation
            }
            ComponentError::RuntimeShutdown => ComponentObservationKind::Shutdown,
            _ => ComponentObservationKind::Validation,
        };
        self.observations.push(ComponentObservation::new(
            kind,
            None,
            instance,
            redact_component_diagnostic(&error.to_string()),
        ));
    }

    fn observe_authority_or_validation_error(&mut self, component: &str, error: &ComponentError) {
        let message = redact_component_diagnostic(&error.to_string());
        if message.contains("authority kind is outside Magnetar inference scope")
            || message.contains("unsupported authority kind")
        {
            self.observations.push(ComponentObservation::new(
                ComponentObservationKind::Validation,
                Some(component.into()),
                None,
                format!("component authority rejected: {message}"),
            ));
        } else {
            self.observations.push(ComponentObservation::new(
                ComponentObservationKind::Validation,
                Some(component.into()),
                None,
                message,
            ));
        }
    }
}

impl Drop for ComponentManager {
    fn drop(&mut self) {
        for directory in self.owned_distributed_package_dirs.drain(..) {
            let _ = fs::remove_dir_all(directory);
        }
    }
}

struct ComponentArtifactValidationOutcome {
    manifest: ComponentManifest,
    digest: ComponentDigest,
    trust_decision: ComponentTrustDecision,
    observations: Vec<ComponentObservation>,
}

fn validate_component_artifact(
    engine: &mut dyn ComponentEngine,
    definition: &mut ComponentDefinition,
    trust_store: &ComponentTrustStore,
) -> Result<ComponentArtifactValidationOutcome, ComponentError> {
    let mut observations = vec![ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(definition.metadata.name.clone()),
        None,
        format!(
            "component artifact discovered at {}",
            definition.artifact_path.display()
        ),
    )];
    let bytes = fs::read(&definition.artifact_path).map_err(|source| {
        ComponentError::ComponentLoadFailed {
            path: definition.artifact_path.clone(),
            message: source.to_string(),
            source: Some(source),
        }
    })?;
    let digest = ComponentDigest::sha256(&bytes);
    observations.push(ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(definition.metadata.name.clone()),
        None,
        format!("component artifact digest computed: {}", digest.value),
    ));
    let manifest_path = manifest_path_for_definition(definition)?;
    let manifest = ComponentManifest::load_yaml(&manifest_path)?;
    observations.push(ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(definition.metadata.name.clone()),
        None,
        "component artifact manifest loaded and schema validated",
    ));
    let capabilities = engine.capabilities();
    manifest
        .engine
        .validate(&manifest.component.name, &capabilities)?;
    observations.push(ComponentObservation::new(
        ComponentObservationKind::EngineSelection,
        Some(manifest.component.name.clone()),
        None,
        format!(
            "Component artifact compatible with engine profile '{}'",
            capabilities.profile.as_str()
        ),
    ));
    if manifest.digest != digest {
        return Err(ComponentError::ArtifactRejected {
            component: definition.metadata.name.clone(),
            status: ComponentTrustStatus::Rejected,
            message: "manifest digest does not match artifact bytes".into(),
        });
    }
    let actual_contract = engine.inspect_contract(definition)?;
    validate_manifest_wit(definition, &manifest, &actual_contract)?;
    observations.push(ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(definition.metadata.name.clone()),
        None,
        "component artifact WIT declarations match executable contract",
    ));
    validate_runtime_compatibility(definition, &manifest)?;
    validate_capability_compatibility(definition, &manifest)?;
    observations.push(ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(definition.metadata.name.clone()),
        None,
        "component artifact runtime and capability compatibility validated",
    ));
    validate_signature_metadata(definition, &manifest, &digest)?;
    let trust_decision = trust_store.evaluate(&manifest, &digest);
    observations.push(ComponentObservation::new(
        ComponentObservationKind::Validation,
        Some(manifest.component.name.clone()),
        None,
        format!(
            "component artifact trust decision: {:?}",
            trust_decision.status
        ),
    ));
    if trust_decision.status != ComponentTrustStatus::Trusted {
        return Err(ComponentError::ArtifactRejected {
            component: manifest.component.name.clone(),
            status: trust_decision.status,
            message: trust_decision.reason,
        });
    }
    Ok(ComponentArtifactValidationOutcome {
        manifest,
        digest,
        trust_decision,
        observations,
    })
}

fn manifest_path_for_definition(
    definition: &ComponentDefinition,
) -> Result<PathBuf, ComponentError> {
    if let Some(path) = &definition.manifest_path {
        return Ok(path.clone());
    }
    let Some(file_name) = definition
        .artifact_path
        .file_name()
        .and_then(|value| value.to_str())
    else {
        return Err(ComponentError::ManifestMissing {
            artifact: definition.artifact_path.clone(),
        });
    };
    let candidate = definition
        .artifact_path
        .with_file_name(format!("{file_name}.magnetar-component.yaml"));
    if candidate.exists() {
        return Ok(candidate);
    }
    Err(ComponentError::ManifestMissing {
        artifact: definition.artifact_path.clone(),
    })
}

fn validate_manifest_wit(
    definition: &ComponentDefinition,
    manifest: &ComponentManifest,
    actual: &ComponentContract,
) -> Result<(), ComponentError> {
    let actual_imports = actual.import_interfaces();
    if manifest.imports != actual_imports {
        return Err(ComponentError::ContractValidationFailed {
            component: definition.metadata.name.clone(),
            message: "manifest WIT imports do not match executable imports".into(),
        });
    }
    if !manifest.optional_imports.is_disjoint(&actual_imports) {
        return Err(ComponentError::ContractValidationFailed {
            component: definition.metadata.name.clone(),
            message: "manifest optional WIT imports overlap executable required imports".into(),
        });
    }
    let actual_exports = actual.export_interfaces();
    if !manifest.exports.is_subset(&actual_exports) {
        return Err(ComponentError::ContractValidationFailed {
            component: definition.metadata.name.clone(),
            message: "manifest WIT exports are not provided by executable exports".into(),
        });
    }
    Ok(())
}

fn validate_runtime_compatibility(
    definition: &ComponentDefinition,
    manifest: &ComponentManifest,
) -> Result<(), ComponentError> {
    if compare_semver(&manifest.runtime_min_version, MAGNETAR_RUNTIME_VERSION)
        .is_some_and(|ordering| ordering == std::cmp::Ordering::Greater)
    {
        return Err(ComponentError::ArtifactRejected {
            component: definition.metadata.name.clone(),
            status: ComponentTrustStatus::Rejected,
            message: format!(
                "component requires Magnetar Runtime {} but current runtime is {}",
                manifest.runtime_min_version, MAGNETAR_RUNTIME_VERSION
            ),
        });
    }
    if let Some(max_version) = &manifest.runtime_max_version
        && compare_semver(max_version, MAGNETAR_RUNTIME_VERSION)
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
    {
        return Err(ComponentError::ArtifactRejected {
            component: definition.metadata.name.clone(),
            status: ComponentTrustStatus::Rejected,
            message: format!(
                "component supports Magnetar Runtime up to {max_version} but current runtime is {MAGNETAR_RUNTIME_VERSION}"
            ),
        });
    }
    Ok(())
}

fn validate_capability_compatibility(
    definition: &ComponentDefinition,
    manifest: &ComponentManifest,
) -> Result<(), ComponentError> {
    for capability in &manifest.capabilities {
        let Some(import) = manifest
            .imports
            .iter()
            .find(|import| import.name == capability.id)
        else {
            return Err(ComponentError::ArtifactRejected {
                component: definition.metadata.name.clone(),
                status: ComponentTrustStatus::Rejected,
                message: format!(
                    "required capability '{}@{}' is not backed by a WIT import",
                    capability.id, capability.min_version
                ),
            });
        };
        if semver_major(&import.version) != semver_major(&capability.min_version) {
            return Err(ComponentError::ArtifactRejected {
                component: definition.metadata.name.clone(),
                status: ComponentTrustStatus::Rejected,
                message: format!(
                    "required capability '{}' major version is incompatible with WIT import '{}'",
                    capability.id, import.version
                ),
            });
        }
        if compare_semver(&import.version, &capability.min_version)
            .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
        {
            return Err(ComponentError::ArtifactRejected {
                component: definition.metadata.name.clone(),
                status: ComponentTrustStatus::Rejected,
                message: format!(
                    "required capability '{}' needs at least {} but WIT import is {}",
                    capability.id, capability.min_version, import.version
                ),
            });
        }
        if let Some(max_version) = &capability.max_version
            && compare_semver(&import.version, max_version)
                .is_some_and(|ordering| ordering == std::cmp::Ordering::Greater)
        {
            return Err(ComponentError::ArtifactRejected {
                component: definition.metadata.name.clone(),
                status: ComponentTrustStatus::Rejected,
                message: format!(
                    "required capability '{}' allows at most {max_version} but WIT import is {}",
                    capability.id, import.version
                ),
            });
        }
    }
    Ok(())
}

fn validate_signature_metadata(
    definition: &ComponentDefinition,
    manifest: &ComponentManifest,
    digest: &ComponentDigest,
) -> Result<(), ComponentError> {
    for signature in &manifest.signatures {
        if let Some(signature_digest) = &signature.digest
            && signature_digest.to_ascii_lowercase() != digest.value
        {
            return Err(ComponentError::ArtifactRejected {
                component: definition.metadata.name.clone(),
                status: ComponentTrustStatus::Rejected,
                message: "signature metadata digest does not match artifact digest".into(),
            });
        }
    }
    Ok(())
}

fn validate_prepared_contract(
    definition: &ComponentDefinition,
    contract: &ComponentContract,
) -> Result<(), ComponentError> {
    let prepared_imports = contract.import_interfaces();
    for interface in &prepared_imports {
        if !definition.metadata.imports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "prepared Component imports undeclared interface '{}@{}'",
                    interface.name, interface.version
                ),
            });
        }
    }
    for interface in &definition.metadata.imports {
        if !prepared_imports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "declared import '{}@{}' was not found in prepared Component",
                    interface.name, interface.version
                ),
            });
        }
    }

    let prepared_exports = contract.export_interfaces();
    for interface in &definition.metadata.exports {
        if !prepared_exports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "declared export '{}@{}' was not found in prepared Component",
                    interface.name, interface.version
                ),
            });
        }
    }

    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn lower_set(values: BTreeSet<String>) -> BTreeSet<String> {
    values
        .into_iter()
        .map(|value| value.to_ascii_lowercase())
        .collect()
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn wit_interface_from_manifest(
    interface: ManifestWitInterfaceYaml,
    path: &Path,
) -> Result<WitInterface, ComponentError> {
    validate_wit_package(&interface.package)
        .map_err(|message| manifest_validation_error(path, message))?;
    if interface.interface.trim().is_empty() || interface.interface.contains('/') {
        return Err(manifest_validation_error(
            path,
            "WIT interface name is invalid",
        ));
    }
    validate_semver(&interface.version)
        .map_err(|message| manifest_validation_error(path, message))?;
    Ok(WitInterface::new(
        format!("{}/{}", interface.package, interface.interface),
        interface.version,
    ))
}

fn validate_wit_package(value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() || !value.contains(':') {
        return Err("WIT package must include a namespace");
    }
    if value.contains('/') || value.contains('@') || value.contains(' ') {
        return Err("WIT package must not include interface, version, or spaces");
    }
    Ok(())
}

fn validate_component_name(value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        return Err("component name must not be empty");
    }
    if value.starts_with('.') || value.ends_with('.') || value.contains("..") {
        return Err("component name must not be ambiguous");
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || byte == b'.'
            || byte == b'-'
            || byte == b'_'
    }) {
        return Err("component name must use lowercase ASCII segments");
    }
    Ok(())
}

fn validate_wit_identity(value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() || !value.contains(':') || !value.contains('/') {
        return Err("WIT identity must include package namespace and interface");
    }
    if value.contains('@') || value.contains(' ') {
        return Err("WIT identity must not include version or spaces");
    }
    Ok(())
}

fn validate_runtime_identity(value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        return Err("runtime identity must not be empty");
    }
    if value.contains('/') || value.contains('\\') || value.contains(':') {
        return Err("runtime identity must not be a path or URI");
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err("runtime identity must use portable ASCII characters");
    }
    Ok(())
}

fn validate_authority_kind(value: &str) -> Result<(), &'static str> {
    match value {
        "model-artifact-read"
        | "tokenizer-artifact-read"
        | "prompt-template-read"
        | "adapter-artifact-read"
        | "quantization-artifact-read"
        | "inference-session-state"
        | "generation-session-state"
        | "kv-cache-access"
        | "prefix-cache-access"
        | "compute-capability"
        | "generation-capability"
        | "sampling-capability"
        | "observability-emit"
        | "runtime-diagnostics" => Ok(()),
        "filesystem" | "network" | "environment" | "process" | "shell" | "secret" | "secrets"
        | "workspace" | "git" | "source-control" | "tool" | "tool-execution"
        | "external-service" => Err("authority kind is outside Magnetar inference scope"),
        _ => Err("unsupported authority kind"),
    }
}

fn authority_endpoint_for_kind(kind: &str) -> ComponentAuthorityEndpoint {
    match kind {
        "model-artifact-read" => ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Model,
        },
        "tokenizer-artifact-read" => ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Tokenizer,
        },
        "prompt-template-read" => ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::PromptTemplate,
        },
        "adapter-artifact-read" => ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Adapter,
        },
        "quantization-artifact-read" => ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Quantization,
        },
        "kv-cache-access" => ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Kv,
        },
        "prefix-cache-access" => ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Prefix,
        },
        "compute-capability" => ComponentAuthorityEndpoint::Capability {
            interface: WitInterface::new("magnetar:compute/run", "2.0.0"),
        },
        "generation-capability" => ComponentAuthorityEndpoint::PendingRuntimeService {
            authority: kind.into(),
        },
        "sampling-capability" => ComponentAuthorityEndpoint::PendingRuntimeService {
            authority: kind.into(),
        },
        "observability-emit" => ComponentAuthorityEndpoint::Observability,
        "runtime-diagnostics" => ComponentAuthorityEndpoint::RuntimeDiagnostics,
        "inference-session-state" | "generation-session-state" => {
            ComponentAuthorityEndpoint::RuntimeService {
                interface: WitInterface::new("magnetar:runtime/session-state", "1.0.0"),
            }
        }
        _ => ComponentAuthorityEndpoint::PendingRuntimeService {
            authority: kind.into(),
        },
    }
}

fn is_forbidden_external_interface(value: &str) -> bool {
    value.starts_with("wasi:filesystem/")
        || value.starts_with("wasi:sockets/")
        || value.starts_with("wasi:cli/")
        || value.starts_with("magnetar:secrets/")
        || value.starts_with("magnetar:git/")
        || value.starts_with("magnetar:workspace/")
        || value.starts_with("magnetar:process/")
        || value.starts_with("magnetar:shell/")
        || value.starts_with("magnetar:network/")
}

fn is_inference_linkable_interface(value: &str) -> bool {
    value.starts_with("magnetar:")
        && !is_forbidden_external_interface(value)
        && !value.starts_with("magnetar:tool/")
        && !value.starts_with("magnetar:external-service/")
}

fn redact_component_diagnostic(value: &str) -> String {
    let mut redacted = Vec::new();
    for token in value.split_whitespace() {
        let trimmed = token.trim_matches(|ch| matches!(ch, '\'' | '"' | ',' | ':' | ';'));
        if trimmed.contains("\\")
            || trimmed.contains('/')
            || trimmed.to_ascii_lowercase().contains("secret")
        {
            redacted.push("[redacted]");
        } else {
            redacted.push(token);
        }
    }
    redacted.join(" ")
}

fn validate_semver(value: &str) -> Result<(), &'static str> {
    parse_semver(value).map(|_| ())
}

fn parse_semver(value: &str) -> Result<(u64, u64, u64), &'static str> {
    let mut parts = value.split('.');
    let parse = |part: Option<&str>| {
        let part = part.ok_or("version must have major.minor.patch")?;
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("version segments must be numeric");
        }
        part.parse::<u64>()
            .map_err(|_| "version segment is invalid")
    };
    let version = (
        parse(parts.next())?,
        parse(parts.next())?,
        parse(parts.next())?,
    );
    if parts.next().is_some() {
        return Err("version must have exactly three segments");
    }
    Ok(version)
}

fn compare_semver(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    Some(parse_semver(left).ok()?.cmp(&parse_semver(right).ok()?))
}

fn semver_major(value: &str) -> Option<u64> {
    Some(parse_semver(value).ok()?.0)
}

fn manifest_validation_error(path: &Path, message: impl Into<String>) -> ComponentError {
    ComponentError::Manifest {
        path: path.into(),
        message: message.into(),
        source: None,
    }
}

fn trust_store_error(path: &Path, message: impl Into<String>) -> ComponentError {
    ComponentError::TrustStore {
        path: path.into(),
        message: message.into(),
        source: None,
    }
}

#[derive(Debug)]
pub enum ComponentError {
    AlreadyRegistered(String),
    NotFound(String),
    InstanceNotFound(ComponentInstanceId),
    MissingPreparedDefinition(ComponentDefinitionId),
    UnresolvedImport {
        component: String,
        interface: WitInterface,
    },
    UnauthorizedImport {
        component: String,
        interface: WitInterface,
    },
    PreparationFailed {
        component: String,
        message: String,
    },
    ContractValidationFailed {
        component: String,
        message: String,
    },
    InstantiationFailed {
        definition: ComponentDefinitionId,
        message: String,
    },
    InvocationFailed {
        instance: ComponentInstanceId,
        message: String,
    },
    Trap {
        instance: ComponentInstanceId,
        kind: ComponentTrapKind,
        diagnostic: Option<String>,
    },
    Interrupted {
        instance: ComponentInstanceId,
        reason: ComponentInterruptionReason,
    },
    ResourceLimitUnsupported {
        component: String,
        limit: &'static str,
    },
    ResourceLimitExceeded {
        component: String,
        limit: &'static str,
    },
    NoCompatibleEngine {
        component: String,
        target: &'static str,
    },
    EngineProfileMismatch {
        component: String,
        required: ComponentEngineProfile,
        actual: ComponentEngineProfile,
    },
    EngineFeatureUnavailable {
        component: String,
        feature: ComponentEngineFeature,
        profile: ComponentEngineProfile,
    },
    WasmtimeUnavailable {
        component: String,
    },
    BrowserEngineUnavailable {
        component: String,
    },
    PlatformUnsupported {
        component: String,
        target: &'static str,
    },
    HostBindingFailed {
        component: String,
        message: String,
    },
    ComponentLoadFailed {
        path: PathBuf,
        message: String,
        source: Option<std::io::Error>,
    },
    Distribution {
        category: ComponentDistributionErrorCategory,
        message: String,
    },
    ManifestMissing {
        artifact: PathBuf,
    },
    Manifest {
        path: PathBuf,
        message: String,
        source: Option<std::io::Error>,
    },
    TrustStore {
        path: PathBuf,
        message: String,
        source: Option<std::io::Error>,
    },
    ArtifactRejected {
        component: String,
        status: ComponentTrustStatus,
        message: String,
    },
    InvalidInstanceTransition {
        instance: ComponentInstanceId,
        state: ComponentInstanceState,
        operation: &'static str,
    },
    EngineFailure(String),
    RuntimeShutdown,
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A registered [`HostCapability`] rejected a call
    /// (`model-component-graph-contract`). Distinct from `InvocationFailed`
    /// (which needs a [`ComponentInstanceId`] the manager assigns, not
    /// visible to a `HostCapability` -- it only sees the calling engine
    /// instance's own key, see [`HostCapability::call`]'s `instance_key`).
    CapabilityCallRejected {
        capability: String,
        instance_key: String,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentDistributionErrorCategory {
    SourceUnavailable,
    ArtifactNotFound,
    VersionNotFound,
    DigestMismatch,
    ManifestMissing,
    ManifestInvalid,
    WitMismatch,
    CompatibilityFailure,
    ForbiddenAuthority,
    TrustRejected,
    RevokedArtifact,
    CacheIntegrityFailure,
    UnsupportedSignature,
    PolicyDenied,
}

impl fmt::Display for ComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRegistered(name) => write!(f, "component '{name}' is already registered"),
            Self::NotFound(name) => write!(f, "component '{name}' is not registered"),
            Self::InstanceNotFound(id) => {
                write!(f, "component instance '{}' is not registered", id.get())
            }
            Self::MissingPreparedDefinition(id) => {
                write!(
                    f,
                    "component definition '{}' has not been prepared",
                    id.get()
                )
            }
            Self::UnresolvedImport {
                component,
                interface,
            } => write!(
                f,
                "component '{component}' requires unresolved WIT interface '{}@{}'",
                interface.name, interface.version
            ),
            Self::UnauthorizedImport {
                component,
                interface,
            } => write!(
                f,
                "component '{component}' is not authorized to import WIT interface '{}@{}'",
                interface.name, interface.version
            ),
            Self::PreparationFailed { component, message } => {
                write!(f, "component '{component}' preparation failed: {message}")
            }
            Self::ContractValidationFailed { component, message } => {
                write!(
                    f,
                    "component '{component}' contract validation failed: {message}"
                )
            }
            Self::InstantiationFailed {
                definition,
                message,
            } => write!(
                f,
                "component definition '{}' instantiation failed: {message}",
                definition.get()
            ),
            Self::InvocationFailed { instance, message } => write!(
                f,
                "component instance '{}' invocation failed: {message}",
                instance.get()
            ),
            Self::Trap {
                instance,
                kind,
                diagnostic,
            } => {
                write!(
                    f,
                    "component instance '{}' trapped as {kind:?}",
                    instance.get()
                )?;
                if let Some(diagnostic) = diagnostic {
                    write!(f, ": {diagnostic}")?;
                }
                Ok(())
            }
            Self::Interrupted { instance, reason } => write!(
                f,
                "component instance '{}' was interrupted: {reason:?}",
                instance.get()
            ),
            Self::ResourceLimitUnsupported { component, limit } => write!(
                f,
                "component '{component}' requires unsupported resource limit '{limit}'"
            ),
            Self::ResourceLimitExceeded { component, limit } => write!(
                f,
                "component '{component}' exceeded resource limit '{limit}'"
            ),
            Self::NoCompatibleEngine { component, target } => write!(
                f,
                "component '{component}' has no compatible Component Engine for target '{target}'"
            ),
            Self::EngineProfileMismatch {
                component,
                required,
                actual,
            } => write!(
                f,
                "component '{component}' requires Component Engine profile '{}' but selected profile is '{}'",
                required.as_str(),
                actual.as_str()
            ),
            Self::EngineFeatureUnavailable {
                component,
                feature,
                profile,
            } => write!(
                f,
                "component '{component}' requires Component Engine feature '{}' unavailable on profile '{}'",
                feature.as_str(),
                profile.as_str()
            ),
            Self::WasmtimeUnavailable { component } => {
                write!(
                    f,
                    "component '{component}' requires unavailable Wasmtime engine"
                )
            }
            Self::BrowserEngineUnavailable { component } => write!(
                f,
                "component '{component}' requires unavailable browser Component Engine"
            ),
            Self::PlatformUnsupported { component, target } => write!(
                f,
                "component '{component}' is unsupported on target '{target}'"
            ),
            Self::HostBindingFailed { component, message } => {
                write!(f, "component '{component}' host binding failed: {message}")
            }
            Self::ComponentLoadFailed { path, message, .. } => write!(
                f,
                "component artifact '{}' could not be loaded: {message}",
                path.display()
            ),
            Self::Distribution { category, message } => write!(
                f,
                "component distribution failed as {category:?}: {message}"
            ),
            Self::ManifestMissing { artifact } => write!(
                f,
                "component artifact '{}' is missing a sidecar manifest",
                artifact.display()
            ),
            Self::Manifest { path, message, .. } => write!(
                f,
                "component manifest '{}' is invalid: {message}",
                path.display()
            ),
            Self::TrustStore { path, message, .. } => write!(
                f,
                "component trust store '{}' is invalid: {message}",
                path.display()
            ),
            Self::ArtifactRejected {
                component,
                status,
                message,
            } => write!(
                f,
                "component artifact '{component}' is not trusted ({status:?}): {message}"
            ),
            Self::InvalidInstanceTransition {
                instance,
                state,
                operation,
            } => write!(
                f,
                "cannot {operation} component instance '{}' from state {state:?}",
                instance.get()
            ),
            Self::EngineFailure(message) => write!(f, "component engine failed: {message}"),
            Self::RuntimeShutdown => write!(f, "component runtime is shut down"),
            Self::Discovery { path, source } => write!(
                f,
                "could not discover components in '{}': {source}",
                path.display()
            ),
            Self::CapabilityCallRejected {
                capability,
                instance_key,
                message,
            } => write!(
                f,
                "capability '{capability}' rejected a call from instance '{instance_key}': {message}"
            ),
        }
    }
}
impl Error for ComponentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ComponentLoadFailed {
                source: Some(source),
                ..
            } => Some(source),
            Self::Manifest {
                source: Some(source),
                ..
            } => Some(source),
            Self::TrustStore {
                source: Some(source),
                ..
            } => Some(source),
            Self::Discovery { source, .. } => Some(source),
            _ => None,
        }
    }
}
