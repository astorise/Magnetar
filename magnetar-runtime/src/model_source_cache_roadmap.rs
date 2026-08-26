//! Post-baseline model source and artifact cache roadmap contract (see
//! `openspec/changes/define-post-baseline-model-source-and-cache-roadmap`).
//!
//! The baseline Runtime resolves models through a caller-supplied
//! [`crate::inference_api::ModelRef`] and a [`crate::inference_api::ModelRegistry`],
//! and stores artifacts through the existing [`crate::model::ModelArtifactSource`]
//! / [`crate::model::ModelTrustStore`] contracts. This module does not
//! implement real downloads, a `magnetar model pull` UX, a registry
//! protocol, a model hub API, a Tachyon distribution protocol, a cache
//! directory layout, or production credential storage -- the proposal's
//! "Non-Goals" section rules all of that out explicitly. Instead it defines,
//! as executable Rust types and validation functions, the roadmap
//! **contract** any future source/cache implementation must satisfy:
//!
//! - [`ModelSourceKind`]: the seven source kinds from the proposal
//!   (development-fixture, client-provided-artifact, local-cache,
//!   local-directory-source, external-registry-source, model-hub-source,
//!   tachyon-provided-source), with [`ModelSourceKind::from_artifact_source`]
//!   and [`ModelSourceKind::from_resolution_source`] proving they normalize
//!   onto the *existing* [`crate::model::ModelArtifactSource`] and
//!   [`crate::inference_api::ModelResolutionSource`] contracts rather than a
//!   parallel source type. [`ModelSourceKind::grants_trust`] is always
//!   `false` -- "Source kind SHALL not imply trust" made structurally
//!   checkable.
//! - [`validate_development_fixture_source`] /
//!   [`development_fixture_requires_explicit_trust_evaluation`),
//!   [`validate_client_provided_source`], [`validate_local_directory_source`]
//!   (composing [`crate::model_format_roadmap::validate_local_file_boundary`]
//!   rather than duplicating the local-file boundary), and
//!   [`validate_remote_source_policy`] for the external-registry /
//!   model-hub / Tachyon-provided source kinds.
//! - [`CacheKey`]: digest-based cache addressing built on the existing
//!   [`crate::model::ModelDigest`]. [`CacheEntryRef::redacted_path`] never
//!   returns the raw cache path unless a caller explicitly attests
//!   disclosure is policy-allowed.
//! - [`CacheEntryMetadata`]: the cache entry metadata fields from the
//!   proposal, reusing [`crate::model::ModelArtifactId`],
//!   [`crate::model::ModelShardId`], [`crate::tokenizer::TokenizerArtifactId`],
//!   and [`crate::adapter::AdapterArtifactId`] rather than parallel identity
//!   types.
//! - [`CacheLifecycleState`]: the thirteen lifecycle states from the
//!   proposal, with [`reject_non_ready_cache_entry_for_loading`] denying
//!   every non-`Ready` state a structured error.
//! - [`evaluate_cache_trust`] / [`validate_cache_integrity`] /
//!   [`validate_cache_shard_integrity`]: "cache presence SHALL not imply
//!   trust" and "cache integrity SHALL be validated" made checkable --
//!   cached trust is always re-evaluated through
//!   [`crate::model::ModelTrustStore`], never trusted blindly.
//! - [`authorize_cache_mutation`] / [`CacheMutationKind`]: policy-controlled
//!   cache mutation, denying eviction/pruning of entries with active Model
//!   Instance references regardless of policy.
//! - [`EvictionCandidate`] / [`is_evictable`] / [`pin_entry`] /
//!   [`unpin_entry`]: eviction and pinning, with pinning never bypassing
//!   [`reject_non_ready_cache_entry_for_loading`] or [`evaluate_cache_trust`].
//! - [`validate_offline_source`]: "offline mode SHALL use only local cache,
//!   client-provided artifacts, or development fixtures".
//! - [`reject_credential_in_metadata`]: authentication boundary --
//!   credential-shaped cache metadata keys are rejected outright.
//! - [`SourcePolicy`]: allowed source kinds plus the proposal's policy
//!   restriction flags.
//! - [`validate_license_policy`]: "license metadata SHALL not be treated as
//!   verified unless validated by policy".
//! - [`validate_adapter_cache_entry`] / [`validate_tokenizer_cache_entry`]:
//!   adapter/tokenizer cache compatibility, composing
//!   [`crate::adapter::AdapterBaseModelCompatibility`] rather than a parallel
//!   compatibility type.
//! - [`cache_presence_implies_memory_residency`]: always `false` -- cache
//!   storage is bytes and metadata; residency remains owned by Model Loading
//!   and Memory Manager.
//! - [`ModelSourceCacheDiagnostic`]: redacted diagnostics, with
//!   [`ModelSourceCacheDiagnostic::digest_prefix_from`] never exposing a full
//!   digest and no field through which a raw cache path, credential, raw
//!   file content, or raw model weight could be represented.
//! - [`ModelSourceCacheRoadmapError`]: the 22 structured error categories
//!   from the proposal's "Error Model" section.
//! - [`ModelSourceCacheRoadmapObservationKind`] /
//!   [`ModelSourceCacheRoadmapObservation`]: the 18 observation categories,
//!   with redacted metadata only.
//! - [`ModelSourceCacheRoadmapConformanceReport`] /
//!   [`run_model_source_cache_roadmap_conformance`]: a small conformance
//!   report, in the shape of
//!   [`crate::model_format_roadmap::ModelFormatRoadmapConformanceReport`],
//!   asserting the roadmap guarantees above hold.

use crate::compute::redact_backend_diagnostic;
use crate::{
    AdapterArtifactId, AdapterBaseModelCompatibility, ModelArtifactId, ModelArtifactSource,
    ModelDigest, ModelLicenseMetadata, ModelManifest, ModelResolutionSource, ModelShard,
    ModelShardId, ModelTrustDecision, ModelTrustStatus, ModelTrustStore, TokenizerArtifactId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

pub const MODEL_SOURCE_CACHE_ROADMAP_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Source kinds
// ---------------------------------------------------------------------

/// The seven post-baseline model source kinds from the proposal's "Source
/// Kinds" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelSourceKind {
    DevelopmentFixture,
    ClientProvidedArtifact,
    LocalCache,
    LocalDirectorySource,
    ExternalRegistrySource,
    ModelHubSource,
    TachyonProvidedSource,
}

/// The seven source kinds, in the proposal's order.
pub const MODEL_SOURCE_KINDS: &[ModelSourceKind] = &[
    ModelSourceKind::DevelopmentFixture,
    ModelSourceKind::ClientProvidedArtifact,
    ModelSourceKind::LocalCache,
    ModelSourceKind::LocalDirectorySource,
    ModelSourceKind::ExternalRegistrySource,
    ModelSourceKind::ModelHubSource,
    ModelSourceKind::TachyonProvidedSource,
];

impl ModelSourceKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::DevelopmentFixture => "development-fixture",
            Self::ClientProvidedArtifact => "client-provided-artifact",
            Self::LocalCache => "local-cache",
            Self::LocalDirectorySource => "local-directory-source",
            Self::ExternalRegistrySource => "external-registry-source",
            Self::ModelHubSource => "model-hub-source",
            Self::TachyonProvidedSource => "tachyon-provided-source",
        }
    }

    /// Whether this source kind implies trust by itself. Always `false` --
    /// "Source kind SHALL not imply trust" is the roadmap's central
    /// invariant; trust always comes from [`evaluate_cache_trust`] /
    /// [`crate::model::ModelTrustStore`], never from which source produced
    /// the artifact.
    pub const fn grants_trust(self) -> bool {
        false
    }

    /// Whether this source kind is reachable while offline (see
    /// [`validate_offline_source`]).
    pub const fn available_offline(self) -> bool {
        matches!(
            self,
            Self::DevelopmentFixture | Self::ClientProvidedArtifact | Self::LocalCache
        )
    }

    /// Normalizes an existing [`ModelArtifactSource`] into a roadmap source
    /// kind, proving the roadmap's seven kinds map onto the *existing*,
    /// already-closed source enum rather than introducing a parallel one.
    pub const fn from_artifact_source(source: &ModelArtifactSource) -> Self {
        match source {
            ModelArtifactSource::LocalPath(_) => Self::LocalDirectorySource,
            ModelArtifactSource::LocalCache(_) => Self::LocalCache,
            ModelArtifactSource::ClientProvided(_) => Self::ClientProvidedArtifact,
            ModelArtifactSource::Registry(_) | ModelArtifactSource::Oci(_) => {
                Self::ExternalRegistrySource
            }
            ModelArtifactSource::HuggingFace(_) => Self::ModelHubSource,
            ModelArtifactSource::Tachyon(_) => Self::TachyonProvidedSource,
        }
    }

    /// Normalizes an existing [`ModelResolutionSource`] into a roadmap
    /// source kind. Returns `None` for [`ModelResolutionSource::LocalRegistry`],
    /// which names a *lookup mechanism* (the local [`crate::inference_api::ModelRegistry`])
    /// rather than an artifact source kind.
    pub const fn from_resolution_source(source: ModelResolutionSource) -> Option<Self> {
        match source {
            ModelResolutionSource::DevelopmentFixture => Some(Self::DevelopmentFixture),
            ModelResolutionSource::ClientProvidedArtifact => Some(Self::ClientProvidedArtifact),
            ModelResolutionSource::TrustedCache => Some(Self::LocalCache),
            ModelResolutionSource::FutureExternalSource => Some(Self::ExternalRegistrySource),
            ModelResolutionSource::FutureTachyonSource => Some(Self::TachyonProvidedSource),
            ModelResolutionSource::LocalRegistry => None,
        }
    }
}

// ---------------------------------------------------------------------
// Development fixture source
// ---------------------------------------------------------------------

/// "Development fixture source SHALL be denied in production unless policy
/// allows it" -- implementing the proposal's "Development Fixture Source"
/// and "Source Policy" sections' `development fixtures in production`
/// restriction.
pub fn validate_development_fixture_source(
    is_production: bool,
    production_policy_allows_fixtures: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if is_production && !production_policy_allows_fixtures {
        Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
            reason: "development-fixture source is denied by production policy".into(),
        })
    } else {
        Ok(())
    }
}

/// "Fixture artifacts SHALL still pass normal artifact, format, trust, and
/// loading validation, using explicit test trust policy": always
/// re-evaluates trust through the caller-supplied [`ModelTrustStore`] --
/// there is no special-cased "fixtures are always trusted" path.
pub fn development_fixture_requires_explicit_trust_evaluation(
    store: &ModelTrustStore,
    manifest: &ModelManifest,
) -> ModelTrustDecision {
    store.evaluate(manifest)
}

// ---------------------------------------------------------------------
// Client-provided artifact source
// ---------------------------------------------------------------------

/// "Client-provided artifact source ... MAY reference local files or
/// in-memory data through authorized contracts": denies an unauthorized
/// client-provided reference rather than assuming ambient authority.
pub fn validate_client_provided_source(
    authorized: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if authorized {
        Ok(())
    } else {
        Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
            reason: "client-provided artifact source requires an authorized contract".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Local cache source / cache addressing
// ---------------------------------------------------------------------

/// A digest-based cache addressing key, built on the existing
/// [`ModelDigest`] rather than a parallel identity type. "Cache entries
/// SHALL be addressed by digest or normalized artifact identity."
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheKey(String);

impl CacheKey {
    pub fn from_digest(digest: &ModelDigest) -> Self {
        Self(digest.value.clone())
    }

    pub fn from_artifact(id: &ModelArtifactId) -> Self {
        Self::from_digest(&id.digest)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A cache entry's addressable reference: a [`CacheKey`] plus its raw
/// on-disk path. "Public APIs SHALL not expose raw cache paths by default":
/// [`Self::redacted_path`] is the only accessor, and it returns `None`
/// unless the caller explicitly attests disclosure is policy-allowed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheEntryRef {
    key: CacheKey,
    raw_path: std::path::PathBuf,
}

impl CacheEntryRef {
    pub fn new(key: CacheKey, raw_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            key,
            raw_path: raw_path.into(),
        }
    }

    pub fn key(&self) -> &CacheKey {
        &self.key
    }

    /// Returns the raw cache path only when `disclosure_allowed` attests
    /// policy explicitly permits it; otherwise redacts to `None`.
    pub fn redacted_path(&self, disclosure_allowed: bool) -> Option<String> {
        disclosure_allowed.then(|| self.raw_path.display().to_string())
    }
}

// ---------------------------------------------------------------------
// Local directory source
// ---------------------------------------------------------------------

/// "Runtime SHALL not recursively scan arbitrary directories during
/// inference": composes the existing
/// [`crate::model_format_roadmap::validate_local_file_boundary`] rather than
/// duplicating the local-file authorization boundary.
pub fn validate_local_directory_source(
    source: &ModelArtifactSource,
    authorized: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    crate::model_format_roadmap::validate_local_file_boundary(source, authorized).map_err(|error| {
        ModelSourceCacheRoadmapError::ModelSourceInvalid {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// External registry / model hub / Tachyon-provided sources
// ---------------------------------------------------------------------

/// "Registry access SHALL be explicit and policy-controlled" /
/// "Model hub support SHALL remain outside the core inference path" /
/// Tachyon-provided source policy gating -- one shared policy gate for
/// every remote-shaped source kind.
pub fn validate_remote_source_policy(
    kind: ModelSourceKind,
    policy: &SourcePolicy,
) -> Result<(), ModelSourceCacheRoadmapError> {
    policy.validate_kind(kind)
}

// ---------------------------------------------------------------------
// ModelRef resolution
// ---------------------------------------------------------------------

/// What a `ModelRef` resolution attempt produced, mirroring the proposal's
/// "ModelRef Resolution" target list. Reuses [`crate::model_instance::ModelInstanceId`]
/// and [`ModelArtifactId`] rather than introducing parallel identity types.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelRefResolutionOutcome {
    ExistingInstance(crate::model_instance::ModelInstanceId),
    CachedArtifact(ModelArtifactId),
    SourceCandidate {
        kind: ModelSourceKind,
        reference: String,
    },
}

/// "Ambiguous ModelRefs SHALL fail or require policy-defined
/// disambiguation": zero candidates fails not-found, exactly one candidate
/// resolves, more than one fails ambiguous -- resolution can never silently
/// pick one of several matches.
pub fn resolve_model_ref_candidates(
    reference: &str,
    candidates: Vec<ModelRefResolutionOutcome>,
) -> Result<ModelRefResolutionOutcome, ModelSourceCacheRoadmapError> {
    match candidates.len() {
        0 => Err(ModelSourceCacheRoadmapError::ModelSourceNotFound {
            reference: reference.into(),
        }),
        1 => Ok(candidates.into_iter().next().expect("length checked")),
        _ => Err(ModelSourceCacheRoadmapError::ModelSourceAmbiguous {
            reference: reference.into(),
        }),
    }
}

// ---------------------------------------------------------------------
// Model aliases
// ---------------------------------------------------------------------

/// A user-facing model alias, structurally distinct from
/// [`crate::inference_api::ModelRef`] -- resolving an alias always produces
/// a `ModelRef`, never a loaded model.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelAlias(String);

impl ModelAlias {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelSourceCacheRoadmapError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ModelSourceCacheRoadmapError::ModelAliasNotFound { alias: value });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Owned by CLI/source-manager callers where possible; Runtime alias
/// resolution, if used, goes through the same table. "Aliases SHALL not
/// bypass validation": [`Self::resolve`] returns only an owned reference
/// string (the caller still resolves and validates it as a `ModelRef`), not
/// a loaded model.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAliasTable {
    entries: BTreeMap<ModelAlias, Vec<String>>,
}

impl ModelAliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, alias: ModelAlias, reference: impl Into<String>) {
        self.entries
            .entry(alias)
            .or_default()
            .push(reference.into());
    }

    pub fn resolve(&self, alias: &ModelAlias) -> Result<String, ModelSourceCacheRoadmapError> {
        match self.entries.get(alias).map(Vec::as_slice) {
            None | Some([]) => Err(ModelSourceCacheRoadmapError::ModelAliasNotFound {
                alias: alias.as_str().into(),
            }),
            Some([single]) => Ok(single.clone()),
            Some(_) => Err(ModelSourceCacheRoadmapError::ModelAliasAmbiguous {
                alias: alias.as_str().into(),
            }),
        }
    }
}

// ---------------------------------------------------------------------
// Artifact identity
// ---------------------------------------------------------------------

/// Reports which roadmap-named digest identity fields an already-validated
/// [`ModelManifest`] carries, mirroring
/// [`crate::model_format_roadmap::NormalizedManifestCoverage`] -- proving
/// the *existing* [`ModelArtifactId`]/[`ModelManifest`] contract already
/// carries the roadmap's identity fields rather than introducing a parallel
/// identity type.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArtifactIdentityCoverage {
    pub content_digest: bool,
    pub part_digests: bool,
    pub shard_digests: bool,
    pub tokenizer_reference: bool,
    pub source_annotation: bool,
    pub version_metadata: bool,
}

impl ArtifactIdentityCoverage {
    pub fn from_manifest(manifest: &ModelManifest) -> Self {
        Self {
            content_digest: !manifest.id.digest.value.is_empty(),
            part_digests: manifest
                .parts
                .values()
                .any(|part| !part.digest.value.is_empty()),
            shard_digests: manifest
                .shards
                .iter()
                .any(|shard| !shard.digest.value.is_empty()),
            tokenizer_reference: manifest.tokenizer.is_some(),
            source_annotation: manifest.source.is_some(),
            version_metadata: !manifest.id.revision.as_str().is_empty(),
        }
    }

    pub const fn covers_required_fields(&self) -> bool {
        self.content_digest
    }
}

/// "Human-readable names SHALL not be authoritative identity": two
/// artifacts sharing a name but declaring different digests are distinct
/// identities. Holds structurally because [`ModelArtifactId`] derives
/// `PartialEq`/`Ord` over every field, including `digest` -- this function
/// exists as the roadmap's explicit, testable statement of that guarantee.
pub fn artifacts_are_distinct_despite_same_name(
    left: &ModelArtifactId,
    right: &ModelArtifactId,
) -> bool {
    left.name == right.name && left.digest != right.digest && left != right
}

// ---------------------------------------------------------------------
// Cache entry metadata
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheValidationStatus {
    Unvalidated,
    Validating,
    Valid,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheIntegrityStatus {
    Unchecked,
    Valid,
    Corrupt,
}

/// Cache entry lifecycle states from the proposal's "Cache Lifecycle"
/// section, in the proposal's order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheLifecycleState {
    Discovered,
    Resolving,
    Fetching,
    Partial,
    Normalizing,
    Validating,
    Ready,
    Untrusted,
    Revoked,
    Corrupt,
    Evicting,
    Evicted,
    Failed,
}

pub const CACHE_LIFECYCLE_STATES: &[CacheLifecycleState] = &[
    CacheLifecycleState::Discovered,
    CacheLifecycleState::Resolving,
    CacheLifecycleState::Fetching,
    CacheLifecycleState::Partial,
    CacheLifecycleState::Normalizing,
    CacheLifecycleState::Validating,
    CacheLifecycleState::Ready,
    CacheLifecycleState::Untrusted,
    CacheLifecycleState::Revoked,
    CacheLifecycleState::Corrupt,
    CacheLifecycleState::Evicting,
    CacheLifecycleState::Evicted,
    CacheLifecycleState::Failed,
];

impl CacheLifecycleState {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Resolving => "resolving",
            Self::Fetching => "fetching",
            Self::Partial => "partial",
            Self::Normalizing => "normalizing",
            Self::Validating => "validating",
            Self::Ready => "ready",
            Self::Untrusted => "untrusted",
            Self::Revoked => "revoked",
            Self::Corrupt => "corrupt",
            Self::Evicting => "evicting",
            Self::Evicted => "evicted",
            Self::Failed => "failed",
        }
    }

    /// "Partial cache entries SHALL not be used for Model Loading unless
    /// explicitly supported and validated": only `Ready` is loadable.
    pub const fn is_loadable(self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether an entry in this state is in flight (fetching/normalizing/
    /// validating/evicting) and therefore never eviction-eligible regardless
    /// of pin or age.
    pub const fn is_in_flight(self) -> bool {
        matches!(
            self,
            Self::Resolving
                | Self::Fetching
                | Self::Normalizing
                | Self::Validating
                | Self::Evicting
        )
    }
}

/// "Model Loading SHALL reject corrupt, partial, revoked, untrusted, or
/// incompatible cache entries": every non-`Ready` state maps to a
/// structured, state-specific error.
pub fn reject_non_ready_cache_entry_for_loading(
    state: CacheLifecycleState,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if state.is_loadable() {
        return Ok(());
    }
    Err(match state {
        CacheLifecycleState::Partial => ModelSourceCacheRoadmapError::ModelCachePartialEntry {
            reason: "cache entry download or import is incomplete".into(),
        },
        CacheLifecycleState::Corrupt => ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt {
            reason: "cache entry failed integrity validation".into(),
        },
        CacheLifecycleState::Untrusted => ModelSourceCacheRoadmapError::ModelCacheEntryUntrusted {
            reason: "cache entry trust evaluation did not pass policy".into(),
        },
        CacheLifecycleState::Revoked => ModelSourceCacheRoadmapError::ModelCacheEntryRevoked {
            reason: "cache entry trust was revoked".into(),
        },
        other => ModelSourceCacheRoadmapError::ModelCacheEntryInvalid {
            reason: format!("cache entry state '{}' is not loadable", other.id()),
        },
    })
}

/// The cache entry metadata fields from the proposal's "Cache Entry
/// Metadata" section, built on existing identity/reference types
/// ([`ModelArtifactId`], [`ModelShardId`], [`TokenizerArtifactId`],
/// [`AdapterArtifactId`]) rather than parallel ones.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheEntryMetadata {
    pub identity: ModelArtifactId,
    pub normalized_manifest_digest: Option<ModelDigest>,
    pub source_kind: ModelSourceKind,
    pub source_annotations: BTreeMap<String, String>,
    pub trust_status: ModelTrustStatus,
    pub integrity_status: CacheIntegrityStatus,
    pub validation_status: CacheValidationStatus,
    pub format_metadata: Option<String>,
    pub size_estimate_bytes: Option<u64>,
    pub parts: Vec<String>,
    pub shards: Vec<ModelShardId>,
    pub tokenizer: Option<TokenizerArtifactId>,
    pub adapters: Vec<AdapterArtifactId>,
    pub created_unix_seconds: u64,
    pub last_used_unix_seconds: u64,
    pub pinned: bool,
    pub lifecycle: CacheLifecycleState,
}

impl CacheEntryMetadata {
    pub fn new(identity: ModelArtifactId, source_kind: ModelSourceKind) -> Self {
        let now = now_unix_seconds();
        Self {
            identity,
            normalized_manifest_digest: None,
            source_kind,
            source_annotations: BTreeMap::new(),
            trust_status: ModelTrustStatus::Unknown,
            integrity_status: CacheIntegrityStatus::Unchecked,
            validation_status: CacheValidationStatus::Unvalidated,
            format_metadata: None,
            size_estimate_bytes: None,
            parts: Vec::new(),
            shards: Vec::new(),
            tokenizer: None,
            adapters: Vec::new(),
            created_unix_seconds: now,
            last_used_unix_seconds: now,
            pinned: false,
            lifecycle: CacheLifecycleState::Discovered,
        }
    }

    pub fn key(&self) -> CacheKey {
        CacheKey::from_artifact(&self.identity)
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------
// Cache trust model
// ---------------------------------------------------------------------

/// "Cache presence SHALL not imply trust" / "policy SHALL determine whether
/// cached trust is still acceptable" / "revocation checks MAY invalidate
/// cached trust": always re-evaluates trust through [`ModelTrustStore`];
/// `revoked` short-circuits to [`crate::model::ModelTrustStatus::Revoked`]
/// regardless of any previously cached trust status.
pub fn evaluate_cache_trust(
    store: &ModelTrustStore,
    manifest: &ModelManifest,
    revoked: bool,
) -> ModelTrustDecision {
    if revoked {
        return ModelTrustDecision::new(ModelTrustStatus::Revoked, "cached trust was revoked");
    }
    store.evaluate(manifest)
}

// ---------------------------------------------------------------------
// Cache integrity
// ---------------------------------------------------------------------

/// "Cache integrity SHALL be validated before loading": compares a cache
/// entry's declared digest against a recomputed digest.
pub fn validate_cache_integrity(
    declared: &ModelDigest,
    recomputed: &ModelDigest,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if declared == recomputed {
        Ok(())
    } else {
        Err(ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed {
            reason: format!(
                "declared digest '{}' does not match recomputed digest '{}'",
                declared.value, recomputed.value
            ),
        })
    }
}

/// Shard-level cache integrity, composing the existing
/// [`ModelShard::verify_bytes`] rather than duplicating digest comparison.
pub fn validate_cache_shard_integrity(
    shard: &ModelShard,
    bytes: &[u8],
) -> Result<(), ModelSourceCacheRoadmapError> {
    shard.verify_bytes(bytes).map_err(|error| {
        ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Cache mutation
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheMutationKind {
    Insert,
    UpdateMetadata,
    MarkValidated,
    MarkUntrusted,
    MarkRevoked,
    Evict,
    Prune,
    Pin,
    Unpin,
    RepairPlaceholder,
}

/// "Cache mutation SHALL be explicit and policy-controlled" / "Eviction
/// SHALL not remove artifacts required by active Model Instances":
/// eviction/pruning is denied outright when `active_instance_refs > 0`,
/// before the policy flag is even consulted.
pub fn authorize_cache_mutation(
    kind: CacheMutationKind,
    policy_allows: bool,
    active_instance_refs: u32,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if matches!(kind, CacheMutationKind::Evict | CacheMutationKind::Prune)
        && active_instance_refs > 0
    {
        return Err(ModelSourceCacheRoadmapError::ModelCacheActiveReference {
            reason: "cache entry is referenced by an active Model Instance".into(),
        });
    }
    if policy_allows {
        return Ok(());
    }
    Err(match kind {
        CacheMutationKind::Insert => ModelSourceCacheRoadmapError::ModelCacheInsertDenied {
            reason: "cache insertion is denied by policy".into(),
        },
        CacheMutationKind::Evict | CacheMutationKind::Prune => {
            ModelSourceCacheRoadmapError::ModelCacheEvictionDenied {
                reason: "cache eviction is denied by policy".into(),
            }
        }
        _ => ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
            reason: format!("cache mutation {kind:?} is denied by policy"),
        },
    })
}

// ---------------------------------------------------------------------
// Cache eviction and pinning
// ---------------------------------------------------------------------

/// A cache entry plus the eviction inputs from the proposal's "Cache
/// Eviction" section that are not already fields of [`CacheEntryMetadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvictionCandidate {
    pub entry: CacheEntryMetadata,
    pub active_instance_refs: u32,
}

/// "Eviction SHALL not remove artifacts required by active Model Instances"
/// / "Pinned entries SHOULD be protected from automatic eviction": an entry
/// is evictable only when unpinned, unreferenced, and not mid-transition.
pub fn is_evictable(candidate: &EvictionCandidate) -> bool {
    !candidate.entry.pinned
        && candidate.active_instance_refs == 0
        && !candidate.entry.lifecycle.is_in_flight()
}

/// Selects evictable candidates ordered oldest-`last_used`-first, the
/// natural default eviction order combining the proposal's "last used time"
/// and "age" inputs.
pub fn select_eviction_candidates(candidates: &[EvictionCandidate]) -> Vec<&CacheEntryMetadata> {
    let mut evictable: Vec<&CacheEntryMetadata> = candidates
        .iter()
        .filter(|candidate| is_evictable(candidate))
        .map(|candidate| &candidate.entry)
        .collect();
    evictable.sort_by_key(|entry| entry.last_used_unix_seconds);
    evictable
}

/// "Cache entries MAY be pinned." Pinning never bypasses
/// [`reject_non_ready_cache_entry_for_loading`] or [`evaluate_cache_trust`]
/// -- it only affects [`is_evictable`].
pub fn pin_entry(entry: &mut CacheEntryMetadata) {
    entry.pinned = true;
}

pub fn unpin_entry(entry: &mut CacheEntryMetadata) {
    entry.pinned = false;
}

// ---------------------------------------------------------------------
// Offline mode
// ---------------------------------------------------------------------

/// "Offline mode SHALL use only local cache, client-provided artifacts, or
/// development fixtures only."
pub fn validate_offline_source(
    kind: ModelSourceKind,
    offline: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if !offline || kind.available_offline() {
        Ok(())
    } else {
        Err(
            ModelSourceCacheRoadmapError::ModelSourceOfflineUnavailable {
                reason: format!("source kind '{}' requires network access", kind.id()),
            },
        )
    }
}

// ---------------------------------------------------------------------
// Authentication boundary
// ---------------------------------------------------------------------

const CREDENTIAL_SHAPED_KEY_FRAGMENTS: &[&str] = &[
    "token",
    "password",
    "secret",
    "apikey",
    "api_key",
    "credential",
    "bearer",
];

/// "Secrets SHALL not be stored in Runtime cache metadata by default":
/// rejects a cache metadata annotation map whose keys look credential-shaped.
pub fn reject_credential_in_metadata(
    annotations: &BTreeMap<String, String>,
) -> Result<(), ModelSourceCacheRoadmapError> {
    for key in annotations.keys() {
        let lower = key.to_ascii_lowercase();
        if CREDENTIAL_SHAPED_KEY_FRAGMENTS
            .iter()
            .any(|fragment| lower.contains(fragment))
        {
            return Err(
                ModelSourceCacheRoadmapError::ModelSourceAuthenticationFailed {
                    reason: format!("cache metadata key '{key}' appears to carry a credential"),
                },
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Source policy
// ---------------------------------------------------------------------

/// Source policy from the proposal's "Source Policy" section: allowed
/// source kinds plus the named restriction flags.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourcePolicy {
    pub allowed_kinds: BTreeSet<ModelSourceKind>,
    pub allow_unsigned_artifacts: bool,
    pub allow_untrusted_cache_entries: bool,
    pub max_artifact_size_bytes: Option<u64>,
    pub allow_quantized_artifacts: bool,
    pub allow_license_restricted_artifacts: bool,
    pub allow_development_fixtures_in_production: bool,
    pub allow_tachyon_provided_sources: bool,
}

impl Default for SourcePolicy {
    /// Deny-by-default: only the three offline-available source kinds are
    /// allowed, matching [`ModelSourceKind::available_offline`].
    fn default() -> Self {
        Self {
            allowed_kinds: BTreeSet::from([
                ModelSourceKind::DevelopmentFixture,
                ModelSourceKind::ClientProvidedArtifact,
                ModelSourceKind::LocalCache,
            ]),
            allow_unsigned_artifacts: false,
            allow_untrusted_cache_entries: false,
            max_artifact_size_bytes: None,
            allow_quantized_artifacts: true,
            allow_license_restricted_artifacts: false,
            allow_development_fixtures_in_production: false,
            allow_tachyon_provided_sources: false,
        }
    }
}

impl SourcePolicy {
    pub fn allows_kind(&self, kind: ModelSourceKind) -> bool {
        match kind {
            ModelSourceKind::TachyonProvidedSource if !self.allow_tachyon_provided_sources => false,
            _ => self.allowed_kinds.contains(&kind),
        }
    }

    pub fn validate_kind(&self, kind: ModelSourceKind) -> Result<(), ModelSourceCacheRoadmapError> {
        if self.allows_kind(kind) {
            Ok(())
        } else {
            Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
                reason: format!("source kind '{}' is denied by policy", kind.id()),
            })
        }
    }

    pub fn validate_size(&self, size_bytes: u64) -> Result<(), ModelSourceCacheRoadmapError> {
        match self.max_artifact_size_bytes {
            Some(max) if size_bytes > max => {
                Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
                    reason: format!("artifact size {size_bytes} exceeds policy maximum {max}"),
                })
            }
            _ => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------
// License and provenance
// ---------------------------------------------------------------------

/// "License metadata SHALL not be treated as verified unless validated by
/// policy": requires an explicit `policy_validated` attestation before a
/// declared license can gate loading, mirroring
/// [`crate::model_format_roadmap::reject_silent_tokenizer_config_override`]'s
/// shape.
pub fn validate_license_policy(
    license: &ModelLicenseMetadata,
    policy_validated: bool,
    policy_allows: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if !policy_validated {
        return Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
            reason: format!(
                "license '{}' requires explicit policy validation before use",
                license.identifier
            ),
        });
    }
    if policy_allows {
        Ok(())
    } else {
        Err(ModelSourceCacheRoadmapError::ModelSourcePolicyDenied {
            reason: format!("license '{}' is denied by policy", license.identifier),
        })
    }
}

// ---------------------------------------------------------------------
// Model format / adapter / tokenizer cache compatibility
// ---------------------------------------------------------------------

/// "Model source/cache roadmap SHALL integrate with model format
/// normalization": a cache entry is ready for format normalization only
/// once it carries normalized format metadata and has reached `Ready` or
/// `Normalizing`.
pub fn cache_entry_ready_for_format_normalization(entry: &CacheEntryMetadata) -> bool {
    entry.format_metadata.is_some()
        && matches!(
            entry.lifecycle,
            CacheLifecycleState::Ready | CacheLifecycleState::Normalizing
        )
}

/// "Adapter cache entries SHALL preserve base model compatibility
/// metadata": requires the cache entry to carry at least one adapter
/// reference and the compatibility record to name a base model.
pub fn validate_adapter_cache_entry(
    entry: &CacheEntryMetadata,
    base_compatibility: &AdapterBaseModelCompatibility,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if entry.adapters.is_empty() {
        return Err(ModelSourceCacheRoadmapError::ModelCacheEntryInvalid {
            reason: "cache entry carries no adapter reference".into(),
        });
    }
    if base_compatibility.model_name.as_str().is_empty() {
        return Err(ModelSourceCacheRoadmapError::ModelCacheEntryInvalid {
            reason: "adapter cache entry is missing base model compatibility metadata".into(),
        });
    }
    Ok(())
}

/// "Tokenizer cache entries SHALL preserve tokenizer/model compatibility
/// metadata": a cache hit never bypasses the caller's tokenizer/model
/// compatibility check.
pub fn validate_tokenizer_cache_entry(
    entry: &CacheEntryMetadata,
    tokenizer_model_compatible: bool,
) -> Result<(), ModelSourceCacheRoadmapError> {
    if entry.tokenizer.is_none() {
        return Err(ModelSourceCacheRoadmapError::ModelCacheEntryInvalid {
            reason: "cache entry carries no tokenizer reference".into(),
        });
    }
    if tokenizer_model_compatible {
        Ok(())
    } else {
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryInvalid {
            reason: "cached tokenizer is not compatible with the target model".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Memory Manager boundary
// ---------------------------------------------------------------------

/// "Cache presence SHALL not imply artifact is memory-resident": always
/// `false`. [`CacheEntryMetadata`] has no field through which a
/// [`crate::tensor::TensorResource`] or [`crate::memory::MemoryAllocation`]
/// could be represented -- residency is owned exclusively by Model Loading
/// and Memory Manager.
pub const fn cache_presence_implies_memory_residency() -> bool {
    false
}

// ---------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------

/// Redacted source/cache diagnostics from the proposal's "Diagnostics"
/// section. Deliberately has no field through which a credential, raw file
/// content, raw model weight, raw cache path, or Provider/Device/Kernel
/// handle could be represented.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelSourceCacheDiagnostic {
    pub source_kind: Option<ModelSourceKind>,
    pub cache_hit: Option<bool>,
    pub digest_prefix: Option<String>,
    pub validation_status: Option<CacheValidationStatus>,
    pub trust_status: Option<ModelTrustStatus>,
    pub integrity_status: Option<CacheIntegrityStatus>,
    pub size_estimate_bytes: Option<u64>,
    pub missing_parts: Vec<String>,
    pub revoked: bool,
    pub policy_denial_reason: Option<String>,
}

impl ModelSourceCacheDiagnostic {
    /// A short, non-reversible digest prefix -- never the full digest.
    pub fn digest_prefix_from(digest: &ModelDigest) -> String {
        digest.value.chars().take(15).collect()
    }

    /// Redacts a policy denial reason before it reaches observability,
    /// reusing the existing `redact_backend_diagnostic` path rather than a
    /// parallel redaction routine.
    pub fn redact_policy_denial_reason(reason: &str) -> String {
        redact_backend_diagnostic(reason)
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured model source/cache roadmap error, covering every error
/// category from the proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSourceCacheRoadmapError {
    ModelSourceUnsupported { reason: String },
    ModelSourceInvalid { reason: String },
    ModelSourceAmbiguous { reference: String },
    ModelSourcePolicyDenied { reason: String },
    ModelSourceNetworkDenied { reason: String },
    ModelSourceAuthenticationFailed { reason: String },
    ModelSourceNotFound { reference: String },
    ModelSourceOfflineUnavailable { reason: String },
    ModelCacheUnavailable { reason: String },
    ModelCacheMiss { key: String },
    ModelCacheEntryInvalid { reason: String },
    ModelCacheEntryCorrupt { reason: String },
    ModelCacheEntryUntrusted { reason: String },
    ModelCacheEntryRevoked { reason: String },
    ModelCacheIntegrityFailed { reason: String },
    ModelCacheInsertDenied { reason: String },
    ModelCacheEvictionDenied { reason: String },
    ModelCacheActiveReference { reason: String },
    ModelCachePartialEntry { reason: String },
    ModelAliasNotFound { alias: String },
    ModelAliasAmbiguous { alias: String },
    InternalModelSourceCacheError { reason: String },
}

impl ModelSourceCacheRoadmapError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ModelSourceUnsupported { .. } => "model-source-unsupported",
            Self::ModelSourceInvalid { .. } => "model-source-invalid",
            Self::ModelSourceAmbiguous { .. } => "model-source-ambiguous",
            Self::ModelSourcePolicyDenied { .. } => "model-source-policy-denied",
            Self::ModelSourceNetworkDenied { .. } => "model-source-network-denied",
            Self::ModelSourceAuthenticationFailed { .. } => "model-source-authentication-failed",
            Self::ModelSourceNotFound { .. } => "model-source-not-found",
            Self::ModelSourceOfflineUnavailable { .. } => "model-source-offline-unavailable",
            Self::ModelCacheUnavailable { .. } => "model-cache-unavailable",
            Self::ModelCacheMiss { .. } => "model-cache-miss",
            Self::ModelCacheEntryInvalid { .. } => "model-cache-entry-invalid",
            Self::ModelCacheEntryCorrupt { .. } => "model-cache-entry-corrupt",
            Self::ModelCacheEntryUntrusted { .. } => "model-cache-entry-untrusted",
            Self::ModelCacheEntryRevoked { .. } => "model-cache-entry-revoked",
            Self::ModelCacheIntegrityFailed { .. } => "model-cache-integrity-failed",
            Self::ModelCacheInsertDenied { .. } => "model-cache-insert-denied",
            Self::ModelCacheEvictionDenied { .. } => "model-cache-eviction-denied",
            Self::ModelCacheActiveReference { .. } => "model-cache-active-reference",
            Self::ModelCachePartialEntry { .. } => "model-cache-partial-entry",
            Self::ModelAliasNotFound { .. } => "model-alias-not-found",
            Self::ModelAliasAmbiguous { .. } => "model-alias-ambiguous",
            Self::InternalModelSourceCacheError { .. } => "internal-model-source-cache-error",
        }
    }
}

impl fmt::Display for ModelSourceCacheRoadmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelSourceUnsupported { reason }
            | Self::ModelSourceInvalid { reason }
            | Self::ModelSourcePolicyDenied { reason }
            | Self::ModelSourceNetworkDenied { reason }
            | Self::ModelSourceAuthenticationFailed { reason }
            | Self::ModelSourceOfflineUnavailable { reason }
            | Self::ModelCacheUnavailable { reason }
            | Self::ModelCacheEntryInvalid { reason }
            | Self::ModelCacheEntryCorrupt { reason }
            | Self::ModelCacheEntryUntrusted { reason }
            | Self::ModelCacheEntryRevoked { reason }
            | Self::ModelCacheIntegrityFailed { reason }
            | Self::ModelCacheInsertDenied { reason }
            | Self::ModelCacheEvictionDenied { reason }
            | Self::ModelCacheActiveReference { reason }
            | Self::ModelCachePartialEntry { reason }
            | Self::InternalModelSourceCacheError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::ModelSourceAmbiguous { reference } | Self::ModelSourceNotFound { reference } => {
                write!(f, "{}: {reference}", self.id())
            }
            Self::ModelCacheMiss { key } => write!(f, "{}: {key}", self.id()),
            Self::ModelAliasNotFound { alias } | Self::ModelAliasAmbiguous { alias } => {
                write!(f, "{}: {alias}", self.id())
            }
        }
    }
}

impl Error for ModelSourceCacheRoadmapError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Observation categories from the proposal's "Observability" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelSourceCacheRoadmapObservationKind {
    ModelSourceResolved,
    ModelSourceRejected,
    ModelSourceAmbiguous,
    CacheLookupStarted,
    CacheHit,
    CacheMiss,
    CacheEntryValidating,
    CacheEntryReady,
    CacheEntryCorrupt,
    CacheEntryUntrusted,
    CacheEntryRevoked,
    CacheEntryEvicted,
    CacheInsertionStarted,
    CacheInsertionCompleted,
    CachePruningStarted,
    CachePruningCompleted,
    OfflineModeActive,
    SourcePolicyDenied,
}

/// A single model source/cache roadmap observation. Structurally guaranteed
/// to never carry a raw cache path, credential, raw file content, raw model
/// weight, or Provider/Device/Kernel handle by default: the only fields are
/// an enum `kind`, an optional artifact identity, and a `redacted_metadata`
/// string map whose values are always passed through
/// `redact_backend_diagnostic`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSourceCacheRoadmapObservation {
    pub kind: ModelSourceCacheRoadmapObservationKind,
    pub artifact: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ModelSourceCacheRoadmapObservation {
    pub fn new(kind: ModelSourceCacheRoadmapObservationKind) -> Self {
        Self {
            kind,
            artifact: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifact = Some(artifact.into());
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Conformance report
// ---------------------------------------------------------------------

/// A single model source/cache roadmap conformance check result, mirroring
/// [`crate::model_format_roadmap::ModelFormatRoadmapConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSourceCacheRoadmapConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ModelSourceCacheRoadmapConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSourceCacheRoadmapConformanceReport {
    pub results: Vec<ModelSourceCacheRoadmapConformanceResult>,
}

impl ModelSourceCacheRoadmapConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ModelSourceCacheRoadmapConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ModelSourceCacheRoadmapConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the model source/cache roadmap conformance checks described in this
/// module's doc comment: no source kind grants trust; local-directory
/// sources deny unauthorized local file access; ModelRef resolution rejects
/// zero and multiple candidates; aliases reject missing/ambiguous
/// resolution; two same-named artifacts with different digests remain
/// distinct; a non-`Ready` cache entry is never loadable; cached trust is
/// always re-evaluated and revocation always wins; cache integrity mismatch
/// is rejected; eviction/pruning is denied while a Model Instance
/// references the entry; a pinned entry is never evictable; offline mode
/// denies non-offline source kinds; credential-shaped cache metadata is
/// rejected; unvalidated license metadata is denied; and cache presence
/// never implies memory residency.
pub fn run_model_source_cache_roadmap_conformance() -> ModelSourceCacheRoadmapConformanceReport {
    let mut results = Vec::new();

    for kind in MODEL_SOURCE_KINDS {
        record(
            &mut results,
            format!("source kind '{}' does not grant trust", kind.id()),
            !kind.grants_trust(),
            "grants_trust() unexpectedly returned true",
        );
    }

    {
        let outcome = validate_local_directory_source(
            &ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen")),
            false,
        );
        record(
            &mut results,
            "unauthorized local directory source is denied",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let empty = resolve_model_ref_candidates("qwen-test", Vec::new());
        record(
            &mut results,
            "ModelRef resolution with zero candidates fails not-found",
            matches!(
                empty,
                Err(ModelSourceCacheRoadmapError::ModelSourceNotFound { .. })
            ),
            format!("unexpected outcome: {empty:?}"),
        );
        let ambiguous = resolve_model_ref_candidates(
            "qwen-test",
            vec![
                ModelRefResolutionOutcome::SourceCandidate {
                    kind: ModelSourceKind::LocalCache,
                    reference: "a".into(),
                },
                ModelRefResolutionOutcome::SourceCandidate {
                    kind: ModelSourceKind::DevelopmentFixture,
                    reference: "b".into(),
                },
            ],
        );
        record(
            &mut results,
            "ModelRef resolution with multiple candidates fails ambiguous",
            matches!(
                ambiguous,
                Err(ModelSourceCacheRoadmapError::ModelSourceAmbiguous { .. })
            ),
            format!("unexpected outcome: {ambiguous:?}"),
        );
    }

    {
        let alias = ModelAlias::new("qwen").expect("non-empty alias");
        let mut table = ModelAliasTable::new();
        let missing = table.resolve(&alias);
        record(
            &mut results,
            "missing alias fails alias-not-found",
            matches!(
                missing,
                Err(ModelSourceCacheRoadmapError::ModelAliasNotFound { .. })
            ),
            format!("unexpected outcome: {missing:?}"),
        );
        table.register(alias.clone(), "qwen-a");
        table.register(alias.clone(), "qwen-b");
        let ambiguous = table.resolve(&alias);
        record(
            &mut results,
            "ambiguous alias fails alias-ambiguous",
            matches!(
                ambiguous,
                Err(ModelSourceCacheRoadmapError::ModelAliasAmbiguous { .. })
            ),
            format!("unexpected outcome: {ambiguous:?}"),
        );
    }

    {
        let left = probe_artifact_id("qwen-local");
        let mut right = probe_artifact_id("qwen-local");
        right.digest = ModelDigest::sha256(b"different-bytes");
        let distinct = artifacts_are_distinct_despite_same_name(&left, &right);
        record(
            &mut results,
            "same-named artifacts with different digests remain distinct",
            distinct,
            "artifacts with different digests were treated as equal",
        );
    }

    {
        let outcome = reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Partial);
        record(
            &mut results,
            "a partial cache entry is rejected for loading",
            matches!(
                outcome,
                Err(ModelSourceCacheRoadmapError::ModelCachePartialEntry { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let outcome = reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Ready);
        record(
            &mut results,
            "a ready cache entry is accepted for loading",
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let store = ModelTrustStore::default();
        let manifest = probe_manifest();
        let revoked = evaluate_cache_trust(&store, &manifest, true);
        record(
            &mut results,
            "revoked cached trust always wins over stored trust",
            revoked.status == ModelTrustStatus::Revoked,
            format!("unexpected trust decision: {revoked:?}"),
        );
        let unrecognized = evaluate_cache_trust(&store, &manifest, false);
        record(
            &mut results,
            "an unrecognized digest is not trusted merely because it is cached",
            unrecognized.status == ModelTrustStatus::Unknown,
            format!("unexpected trust decision: {unrecognized:?}"),
        );
    }

    {
        let declared = ModelDigest::sha256(b"declared");
        let recomputed = ModelDigest::sha256(b"different");
        let outcome = validate_cache_integrity(&declared, &recomputed);
        record(
            &mut results,
            "a digest mismatch fails cache integrity validation",
            matches!(
                outcome,
                Err(ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = authorize_cache_mutation(CacheMutationKind::Evict, true, 1);
        record(
            &mut results,
            "eviction is denied while a Model Instance references the entry, even with policy allowing it",
            matches!(
                outcome,
                Err(ModelSourceCacheRoadmapError::ModelCacheActiveReference { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let mut entry = CacheEntryMetadata::new(
            probe_artifact_id("qwen-pinned"),
            ModelSourceKind::LocalCache,
        );
        entry.lifecycle = CacheLifecycleState::Ready;
        pin_entry(&mut entry);
        let candidate = EvictionCandidate {
            entry,
            active_instance_refs: 0,
        };
        record(
            &mut results,
            "a pinned cache entry is never evictable",
            !is_evictable(&candidate),
            "pinned entry was reported evictable",
        );
    }

    {
        let outcome = validate_offline_source(ModelSourceKind::ExternalRegistrySource, true);
        record(
            &mut results,
            "offline mode denies a non-offline source kind",
            matches!(
                outcome,
                Err(ModelSourceCacheRoadmapError::ModelSourceOfflineUnavailable { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let outcome = validate_offline_source(ModelSourceKind::LocalCache, true);
        record(
            &mut results,
            "offline mode allows local-cache",
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "registry_token".to_string(),
            "redacted-in-real-use".to_string(),
        );
        let outcome = reject_credential_in_metadata(&annotations);
        record(
            &mut results,
            "credential-shaped cache metadata is rejected",
            matches!(
                outcome,
                Err(ModelSourceCacheRoadmapError::ModelSourceAuthenticationFailed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let license = ModelLicenseMetadata {
            identifier: "apache-2.0".into(),
            url: None,
            usage_restrictions: Vec::new(),
        };
        let outcome = validate_license_policy(&license, false, true);
        record(
            &mut results,
            "unvalidated license metadata is denied even when policy would otherwise allow it",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        record(
            &mut results,
            "cache presence never implies memory residency",
            !cache_presence_implies_memory_residency(),
            "cache_presence_implies_memory_residency() unexpectedly returned true",
        );
    }

    ModelSourceCacheRoadmapConformanceReport { results }
}

/// A minimal, otherwise-unremarkable manifest used only to probe trust
/// evaluation in [`run_model_source_cache_roadmap_conformance`].
fn probe_manifest() -> ModelManifest {
    ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id: probe_artifact_id("source-cache-roadmap-probe"),
        architecture: crate::ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    }
}

fn probe_artifact_id(name: &str) -> ModelArtifactId {
    let digest = ModelDigest::sha256(name.as_bytes());
    ModelArtifactId::new(
        crate::ModelArtifactKind::ModelBundle,
        crate::ModelName::new(name).expect("valid probe name"),
        crate::ModelRevision::new("v1").expect("valid probe revision"),
        digest,
    )
}
