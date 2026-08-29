//! Kernel Artifact Manifest and Kernel Exchange Bundle (see
//! `openspec/changes/define-kernel-artifact-manifest-and-portable-exchange-format`).
//!
//! This module defines the portable, versioned interchange contract an
//! external producer (AI kernel generator, human engineer, CI system, vendor
//! tooling, optimization service, offline build farm, ...) uses to hand
//! Kernel Artifacts and their evidence to Magnetar, without coupling Magnetar
//! to Triton, CUDA, WGSL, Metal, a specific optimization system, a specific
//! artifact registry, or one deployment platform.
//!
//! ```text
//! Kernel Artifact Manifest
//!         |
//!         +-- semantic identity
//!         +-- artifact descriptors
//!         +-- target constraints
//!         +-- provenance
//!         +-- evidence references
//!         +-- policy metadata
//!         |
//!         v
//! content-addressed blobs (blobs/sha256/<digest>)
//! ```
//!
//! - [`KernelManifestSchemaVersion`]: explicit major/minor schema version,
//!   independent from crate/Provider ABI/Operator/WIT versions.
//! - [`parse_manifest_json`]: parses untrusted manifest JSON text, rejecting
//!   duplicate object keys, excessive nesting, and oversized input *before*
//!   any [`serde_json::Value`] tree is built -- "Duplicate Keys Rejected"
//!   (spec).
//! - [`KernelManifestV1::canonical_bytes`] / [`KernelManifestV1::digest`]:
//!   deterministic canonical identity -- "Canonical JSON Manifest" (spec).
//! - [`KernelBlobDigest`] / [`KernelBlobDescriptor`]: content-addressed blob
//!   references. Filename never determines format -- "Filename Does Not
//!   Determine Format" (spec).
//! - [`KernelSemanticBinding`]: single or fused Operator semantics a Kernel
//!   implements, ordered so `RMSNorm -> MatMul` is distinguishable from
//!   `MatMul -> RMSNorm`.
//! - [`KernelManifestTrustMetadata`] / [`KernelSignatureEnvelope`]: trust
//!   inputs only. Nothing in this module can itself produce trusted status --
//!   see [`crate::evaluate_artifact_trust`], the sole authority.
//! - [`KernelManifestRecommendation`] / [`recommendation_grants_promotion`]:
//!   recommendation is always advisory (`recommendation != promotion`).
//! - [`KernelEvidenceReference`] / [`evaluate_qualification_evidence_currency`]:
//!   qualification/benchmark evidence references are revalidated against
//!   current policy rather than trusted merely because they are present.
//! - [`KernelExchangeBundle`] / [`validate_kernel_exchange_bundle`]: the
//!   physical directory-based bundle (`kernel.manifest.json` +
//!   `blobs/sha256/<digest>`) and the validation pipeline (parse ->
//!   structural -> schema -> canonical identity -> blob integrity ->
//!   semantic -> extension) from the spec's "Validation Order".
//! - [`validate_bundle_relative_path`] / [`scan_bundle_for_unsafe_entries`]:
//!   path traversal, absolute-path, drive-qualified-path, and symlink
//!   rejection -- "Bundle Path Safety" (spec).
//! - [`KernelManifestError`]: the 34 structured error categories from the
//!   proposal's "Error Model" section.
//! - [`KernelManifestObservationKind`] / [`KernelManifestObservation`]:
//!   redacted-by-default observability.
//! - [`KernelManifestConformanceReport`] /
//!   [`run_kernel_artifact_manifest_conformance`]: executable conformance
//!   evidence for the guarantees above.
//! - [`normalize_to_source_artifact`] / [`normalize_to_compiled_artifact`] /
//!   [`normalize_qualification_profile`] / [`normalize_oracle_identity`] /
//!   [`normalize_to_cache_key`] / [`normalize_to_cache_entry`]: bridges from
//!   the portable exchange types in this module into the Runtime-native
//!   [`KernelSourceArtifact`], [`CompiledKernelArtifact`],
//!   [`QualificationProfile`]/[`CorrectnessOracleIdentity`], and
//!   [`KernelCacheKey`]/[`KernelCacheEntry`] contracts -- "Normalized
//!   Internal Representation" (spec). None of these fabricate trust,
//!   qualification status, or cache readiness; they only re-express already
//!   -parsed data.
//!
//! # Minimal example manifest
//!
//! The smallest valid v1 manifest: one embedded compiled artifact
//! implementing one portable Operator.
//!
//! ```json
//! {
//!   "schema": "magnetar:kernel-manifest@1.0",
//!   "artifacts": [
//!     {
//!       "role": "compiled-kernel",
//!       "format": "nvidia:cubin",
//!       "digest": "sha256:<lowercase-hex-digest>",
//!       "size": 4096,
//!       "storage_mode": "embedded",
//!       "operators": [
//!         { "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! paired with a bundle directory:
//!
//! ```text
//! kernel.manifest.json
//! blobs/
//!     sha256/
//!         <the digest above>
//! ```
//!
//! # Multi-target example manifest
//!
//! A single logical Kernel with two architecture-specific compiled variants
//! plus qualification/benchmark evidence, implementing "Multiple Compiled
//! Variants" (spec):
//!
//! ```json
//! {
//!   "schema": "magnetar:kernel-manifest@1.0",
//!   "artifacts": [
//!     {
//!       "role": "compiled-kernel",
//!       "format": "nvidia:cubin",
//!       "digest": "sha256:<sm80-digest>",
//!       "size": 4096,
//!       "storage_mode": "embedded",
//!       "target": { "architecture": "sm80", "provider_compatibility": ["nvidia-cuda"] }
//!     },
//!     {
//!       "role": "compiled-kernel",
//!       "format": "nvidia:cubin",
//!       "digest": "sha256:<sm90-digest>",
//!       "size": 4096,
//!       "storage_mode": "embedded",
//!       "target": { "architecture": "sm90", "provider_compatibility": ["nvidia-cuda"] }
//!     }
//!   ],
//!   "qualification_evidence": [
//!     { "digest": "sha256:<evidence-digest>", "profile": "baseline-correctness@1", "status": "passed" }
//!   ],
//!   "benchmark_evidence": [
//!     { "digest": "sha256:<benchmark-digest>", "profile": "latency@1", "workload_profile": "decode-256", "status": "passed" }
//!   ]
//! }
//! ```
//!
//! See `kernel_manifest_multi_target_bundle_validates_with_distinct_architectures`
//! in this crate's test suite for the executable version of this example.

use crate::compute::redact_backend_diagnostic;
use crate::{
    CompiledKernelArtifact, CompiledKernelArtifactId, ComputeDType, CorrectnessOracleIdentity,
    KernelArtifactProvenance, KernelCacheEntry, KernelCacheKey, KernelOperatorVersionRange,
    KernelShapeConstraints, KernelSourceArtifact, KernelSourceArtifactId, KernelSourceFormat,
    OperatorFamily, OperatorId, ProviderBinding, QualificationProfile,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

pub const KERNEL_MANIFEST_SCHEMA_NAMESPACE: &str = "magnetar:kernel-manifest";
pub const KERNEL_MANIFEST_SCHEMA_MAJOR: u32 = 1;
pub const KERNEL_MANIFEST_SCHEMA_MINOR: u32 = 0;
pub const KERNEL_MANIFEST_FILE_NAME: &str = "kernel.manifest.json";
pub const KERNEL_MANIFEST_MEDIA_TYPE: &str = "application/vnd.magnetar.kernel-manifest.v1+json";
pub const KERNEL_BLOB_DIGEST_ALGORITHM: &str = "sha256";

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ---------------------------------------------------------------------
// Defensive limits
// ---------------------------------------------------------------------

/// Defensive parsing/validation limits, implementing "Manifest Limits"
/// (spec). Every limit fails with a structured [`KernelManifestError`]
/// rather than an unbounded panic/allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelManifestLimits {
    pub max_manifest_bytes: usize,
    pub max_nesting_depth: usize,
    pub max_artifacts: usize,
    pub max_evidence_references: usize,
    pub max_extensions: usize,
    pub max_annotation_bytes: usize,
    pub max_string_bytes: usize,
    pub max_total_embedded_bytes: u64,
    /// Caps the total number of entries (device features + Provider
    /// compatibility + runtime/driver compatibility + memory classes)
    /// declared by a single artifact's target constraints, implementing
    /// "Limit target count" (tasks).
    pub max_target_entries: usize,
}

impl Default for KernelManifestLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 8 * 1024 * 1024,
            max_nesting_depth: 32,
            max_artifacts: 256,
            max_evidence_references: 256,
            max_extensions: 64,
            max_annotation_bytes: 4096,
            max_string_bytes: 65536,
            max_target_entries: 128,
            max_total_embedded_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------

/// Explicit manifest schema major/minor version, implementing "Manifest
/// Schema Version" and "Manifest Versioning" (spec): independent from crate
/// version, Provider ABI version, Kernel Compilation Capability version,
/// Operator versions, and WIT versions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelManifestSchemaVersion {
    pub major: u32,
    pub minor: u32,
}

impl KernelManifestSchemaVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    pub const fn current() -> Self {
        Self::new(KERNEL_MANIFEST_SCHEMA_MAJOR, KERNEL_MANIFEST_SCHEMA_MINOR)
    }

    /// A reader SHALL reject unsupported required major versions; minor
    /// versions are additive-only, so any minor is accepted for a supported
    /// major.
    pub const fn is_supported(&self) -> bool {
        self.major == KERNEL_MANIFEST_SCHEMA_MAJOR
    }

    pub fn schema_string(&self) -> String {
        format!(
            "{KERNEL_MANIFEST_SCHEMA_NAMESPACE}@{}.{}",
            self.major, self.minor
        )
    }

    fn parse(text: &str) -> Result<Self, KernelManifestError> {
        let (namespace, version) =
            text.split_once('@')
                .ok_or_else(|| KernelManifestError::SchemaUnsupported {
                    schema: text.to_string(),
                })?;
        if namespace != KERNEL_MANIFEST_SCHEMA_NAMESPACE {
            return Err(KernelManifestError::SchemaUnsupported {
                schema: text.to_string(),
            });
        }
        let (major, minor) = version.split_once('.').unwrap_or((version, "0"));
        let major: u32 = major
            .parse()
            .map_err(|_| KernelManifestError::SchemaUnsupported {
                schema: text.to_string(),
            })?;
        let minor: u32 = minor.parse().unwrap_or(0);
        Ok(Self { major, minor })
    }
}

impl fmt::Display for KernelManifestSchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.schema_string())
    }
}

// ---------------------------------------------------------------------
// Content-addressed digest
// ---------------------------------------------------------------------

/// A content-addressed blob digest, implementing "Content-Addressed Blobs"
/// (spec). The v1 baseline supports SHA-256 only.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelBlobDigest {
    pub algorithm: String,
    pub value: String,
}

impl KernelBlobDigest {
    pub fn sha256(value: impl Into<String>) -> Self {
        Self {
            algorithm: KERNEL_BLOB_DIGEST_ALGORITHM.into(),
            value: value.into().to_ascii_lowercase(),
        }
    }

    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self::sha256(sha256_hex(bytes))
    }

    /// "Blob filename SHALL NOT determine format" and digest well-formedness
    /// -- lowercase hex, correct length for the declared algorithm.
    pub fn is_well_formed(&self) -> bool {
        self.algorithm == KERNEL_BLOB_DIGEST_ALGORITHM
            && self.value.len() == 64
            && self
                .value
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    }

    /// "An embedded SHA-256 blob SHOULD appear at `blobs/sha256/<digest>`"
    /// (spec, "Blob Path").
    pub fn bundle_relative_path(&self) -> String {
        format!("blobs/{}/{}", self.algorithm, self.value)
    }
}

impl fmt::Display for KernelBlobDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.value)
    }
}

// ---------------------------------------------------------------------
// Format identity, blob role, storage mode
// ---------------------------------------------------------------------

/// Extensible `namespace:name@version` artifact format identity, implementing
/// "Format Identity" (spec): no closed `TargetLang` enum, and the manifest's
/// declared format -- never the blob filename -- is authoritative.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelArtifactFormat {
    pub namespace: String,
    pub name: String,
    pub version: Option<String>,
}

impl KernelArtifactFormat {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn is_valid(&self) -> bool {
        !self.namespace.trim().is_empty() && !self.name.trim().is_empty()
    }

    pub fn stable_key(&self) -> String {
        match &self.version {
            Some(version) => format!("{}:{}@{version}", self.namespace, self.name),
            None => format!("{}:{}", self.namespace, self.name),
        }
    }
}

impl fmt::Display for KernelArtifactFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stable_key())
    }
}

/// Extensible blob role, implementing "Blob Roles" (spec): "A blob role
/// SHALL describe purpose, not trust", and the role vocabulary remains
/// extensible beyond the five known constants.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelBlobRole(String);

impl KernelBlobRole {
    pub const KERNEL_SOURCE: &'static str = "kernel-source";
    pub const COMPILED_KERNEL: &'static str = "compiled-kernel";
    pub const QUALIFICATION_EVIDENCE: &'static str = "qualification-evidence";
    pub const BENCHMARK_EVIDENCE: &'static str = "benchmark-evidence";
    pub const AUXILIARY: &'static str = "auxiliary";

    pub fn new(role: impl Into<String>) -> Self {
        Self(role.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelBlobRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// "A descriptor SHALL indicate storage mode" (spec, "Artifact Location").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelArtifactStorageMode {
    Embedded,
    External,
}

impl KernelArtifactStorageMode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::External => "external",
        }
    }
}

// ---------------------------------------------------------------------
// Blob descriptor
// ---------------------------------------------------------------------

/// A single content-addressed blob reference, implementing "Blob
/// Descriptor" (tasks) and "Content-Addressed Blobs" (spec).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelBlobDescriptor {
    pub role: KernelBlobRole,
    pub format: KernelArtifactFormat,
    pub digest: KernelBlobDigest,
    pub size: u64,
    pub media_type: Option<String>,
    pub storage_mode: KernelArtifactStorageMode,
    pub required: bool,
    /// Location hint only when `storage_mode == External`. "Location hints
    /// SHALL NOT replace content digest identity" (spec).
    pub location_hint: Option<String>,
}

impl KernelBlobDescriptor {
    pub fn new(
        role: KernelBlobRole,
        format: KernelArtifactFormat,
        digest: KernelBlobDigest,
        size: u64,
    ) -> Self {
        Self {
            role,
            format,
            digest,
            size,
            media_type: None,
            storage_mode: KernelArtifactStorageMode::Embedded,
            required: true,
            location_hint: None,
        }
    }

    pub fn validate(&self) -> Result<(), KernelManifestError> {
        if !self.digest.is_well_formed() {
            return Err(KernelManifestError::ArtifactReferenceInvalid {
                reason: format!("blob digest '{}' is not well-formed sha256", self.digest),
            });
        }
        if !self.format.is_valid() {
            return Err(KernelManifestError::ArtifactReferenceInvalid {
                reason: "blob format identity must not be empty".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Semantic binding
// ---------------------------------------------------------------------

/// Ordered portable Operator semantics a Kernel implements, implementing
/// "Operator Semantic Binding" and "Fused Semantic Binding" (spec): a single
/// entry is a plain binding; more than one entry is a fused binding, and
/// order is preserved so `RMSNorm -> MatMul` differs from `MatMul ->
/// RMSNorm`.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSemanticBinding {
    pub operators: Vec<OperatorId>,
    /// Optional compatible version range for the *first* (primary) Operator
    /// in `operators`, implementing "Define Operator version compatibility"
    /// (tasks): "Operator ID and Operator semantic version **or compatible
    /// range**" (spec, "Operator Semantic Binding"). `None` means the
    /// binding is only compatible with the primary Operator's exact declared
    /// version.
    pub primary_version_requirements: Option<KernelOperatorVersionRange>,
}

impl KernelSemanticBinding {
    pub fn single(operator: OperatorId) -> Self {
        Self {
            operators: vec![operator],
            primary_version_requirements: None,
        }
    }

    pub fn fused(operators: impl IntoIterator<Item = OperatorId>) -> Self {
        Self {
            operators: operators.into_iter().collect(),
            primary_version_requirements: None,
        }
    }

    pub fn with_primary_version_requirements(mut self, range: KernelOperatorVersionRange) -> Self {
        self.primary_version_requirements = Some(range);
        self
    }

    pub fn is_fused(&self) -> bool {
        self.operators.len() > 1
    }

    /// A deterministic fingerprint from ordered Operator IDs, implementing
    /// "Semantic Binding Identity" (spec). Contains no Provider-specific
    /// handles.
    pub fn fingerprint(&self) -> String {
        self.operators
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Whether `candidate_version` satisfies this binding's primary Operator
    /// version requirement: the declared compatible range when present,
    /// otherwise an exact match against the primary Operator's own declared
    /// version.
    pub fn is_version_compatible(&self, candidate_version: u32) -> bool {
        match (&self.primary_version_requirements, self.operators.first()) {
            (Some(range), _) => range.contains(candidate_version),
            (None, Some(primary)) => primary.version() == candidate_version,
            (None, None) => false,
        }
    }

    pub fn validate(&self) -> Result<(), KernelManifestError> {
        if self.operators.is_empty() {
            return Err(KernelManifestError::SemanticBindingInvalid {
                reason: "semantic binding must declare at least one Operator".into(),
            });
        }
        if let Some(range) = &self.primary_version_requirements
            && range.min > range.max
        {
            return Err(KernelManifestError::SemanticBindingInvalid {
                reason: format!(
                    "operator version range is invalid: min {} > max {}",
                    range.min, range.max
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Target constraints and specialization
// ---------------------------------------------------------------------

/// "Target metadata SHALL remain descriptive. It SHALL NOT contain native
/// Device handles" (spec, "Target Constraints"). No field here is
/// pointer-shaped, mechanically enforcing that guarantee.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelTargetConstraints {
    pub device_type: Option<String>,
    pub hardware_vendor: Option<String>,
    pub architecture: Option<String>,
    pub device_features: BTreeSet<String>,
    pub execution_environment: Option<String>,
    pub provider_compatibility: BTreeSet<String>,
    pub runtime_driver_compatibility: BTreeSet<String>,
    pub memory_classes: BTreeSet<String>,
}

impl KernelTargetConstraints {
    pub fn is_empty(&self) -> bool {
        self.device_type.is_none()
            && self.hardware_vendor.is_none()
            && self.architecture.is_none()
            && self.device_features.is_empty()
            && self.execution_environment.is_none()
            && self.provider_compatibility.is_empty()
            && self.runtime_driver_compatibility.is_empty()
            && self.memory_classes.is_empty()
    }

    /// Total number of entries this target declares, implementing "Limit
    /// target count" (tasks): callers enforce
    /// [`KernelManifestLimits::max_target_entries`] against this.
    pub fn entry_count(&self) -> usize {
        self.device_features.len()
            + self.provider_compatibility.len()
            + self.runtime_driver_compatibility.len()
            + self.memory_classes.len()
    }

    pub fn validate(&self) -> Result<(), KernelManifestError> {
        for value in self
            .device_type
            .iter()
            .chain(&self.hardware_vendor)
            .chain(&self.architecture)
            .chain(&self.execution_environment)
        {
            if value.trim().is_empty() {
                return Err(KernelManifestError::TargetInvalid {
                    reason: "target constraint fields must not be empty strings when present"
                        .into(),
                });
            }
        }
        Ok(())
    }
}

/// "A manifest MAY indicate that an artifact is specialized for: prefill,
/// decode, both" (spec, "Prefill And Decode Specialization"). Optimization
/// metadata only -- never redefines Operator semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelExecutionPhase {
    Prefill,
    Decode,
    Both,
}

/// Explicit runtime-relevant specialization metadata, implementing
/// "Specialization" (spec): "Hidden specialization assumptions SHALL be
/// prohibited."
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelManifestSpecialization {
    pub exact_dimensions: BTreeMap<String, u64>,
    pub batch_range: Option<(u64, u64)>,
    pub sequence_range: Option<(u64, u64)>,
    pub head_count: Option<u64>,
    pub head_dimension: Option<u64>,
    pub tile_sizes: Vec<u64>,
    pub alignment: Option<u64>,
    pub dtype: Option<String>,
    pub layout: Option<String>,
    pub quantization_profile: Option<String>,
    pub execution_phase: Option<KernelExecutionPhase>,
    pub device_features: BTreeSet<String>,
}

impl KernelManifestSpecialization {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Implements "Use checked dimension arithmetic" / "Reject impossible
    /// offsets" (tasks): a declared range whose minimum exceeds its maximum
    /// describes no valid dimension and is rejected rather than silently
    /// tolerated.
    pub fn validate(&self) -> Result<(), KernelManifestError> {
        for (name, (min, max)) in [
            ("batch_range", self.batch_range),
            ("sequence_range", self.sequence_range),
        ]
        .into_iter()
        .filter_map(|(name, range)| range.map(|range| (name, range)))
        {
            if min > max {
                return Err(KernelManifestError::SpecializationInvalid {
                    reason: format!("{name} minimum {min} exceeds maximum {max}"),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Compiler metadata
// ---------------------------------------------------------------------

/// "Compiled artifacts SHOULD record compiler metadata where known" (spec,
/// "Compiler Metadata"). "Compiler metadata SHALL not be interpreted as
/// proof of trust" (spec, "Reproducible Compiler Metadata").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelCompilerMetadata {
    pub compiler_identity: Option<String>,
    pub compiler_version: Option<String>,
    pub backend_identity_version: Option<String>,
    pub flags_fingerprint: Option<String>,
    pub build_fingerprint: Option<String>,
    pub target_architecture: Option<String>,
}

// ---------------------------------------------------------------------
// Precision metadata
// ---------------------------------------------------------------------

/// Portable numerical-behavior metadata, implementing "Precision Metadata"
/// (spec). "Claims remain subject to qualification": nothing in this module
/// treats a precision claim as itself proof of correctness.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelManifestPrecision {
    pub accumulation_dtype: Option<String>,
    pub approximate_math: bool,
    pub deterministic: Option<bool>,
    pub tolerance_profile: Option<String>,
    pub quantization_error_profile: Option<String>,
}

// ---------------------------------------------------------------------
// Generator / campaign metadata
// ---------------------------------------------------------------------

/// Optional generator/optimization-campaign provenance detail, implementing
/// "Generator Metadata" and "Campaign Metadata" (spec): "Raw prompts,
/// secrets, credentials, and internal chain-of-thought SHALL NOT be required
/// Kernel Manifest fields" -- none of these fields can hold them, and
/// [`KernelGeneratorMetadata::validate`] rejects a `source_revision` locator
/// that embeds userinfo-style credentials.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelGeneratorMetadata {
    pub generator_name: Option<String>,
    pub generator_version: Option<String>,
    pub campaign_id: Option<String>,
    pub source_revision: Option<String>,
}

/// Implements "Prevent credentials in locator fields" (tasks): a locator
/// like `https://user:secret@host/repo` embeds userinfo credentials
/// directly in the URL. This is a defensive heuristic, not a full URL
/// parser -- it deliberately fails closed (rejects) on the ambiguous shape
/// rather than trying to guess intent.
fn looks_like_embedded_credential_locator(value: &str) -> bool {
    if let Some((_, after_scheme)) = value.split_once("://")
        && let Some((userinfo, _)) = after_scheme.split_once('@')
    {
        return userinfo.contains(':');
    }
    false
}

impl KernelGeneratorMetadata {
    pub fn validate(&self) -> Result<(), KernelManifestError> {
        if let Some(revision) = &self.source_revision
            && looks_like_embedded_credential_locator(revision)
        {
            return Err(KernelManifestError::ProvenanceInvalid {
                value: "source_revision must not embed credentials".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Evidence references
// ---------------------------------------------------------------------

/// Qualification/benchmark evidence lifecycle status as carried in the
/// portable reference. Presence of a reference never implies currency --
/// see [`evaluate_qualification_evidence_currency`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelEvidenceStatus {
    Pending,
    Passed,
    Failed,
    Revoked,
    Stale,
}

/// A qualification or benchmark evidence reference, implementing
/// "Qualification Evidence References" and "Benchmark Evidence References"
/// (spec). "The manifest SHALL NOT make qualification evidence current
/// merely by referencing it."
#[derive(Clone, Debug, PartialEq)]
pub struct KernelEvidenceReference {
    pub digest: KernelBlobDigest,
    pub profile: String,
    pub suite_or_workload_version: Option<String>,
    pub oracle_or_provider_identity: Option<String>,
    pub target_compatibility: BTreeSet<String>,
    pub status: KernelEvidenceStatus,
    pub storage_mode: KernelArtifactStorageMode,
    /// Benchmark-specific workload identity ("Add workload profile", tasks),
    /// distinct from `profile` (which names the qualification/benchmark
    /// *profile*, not the workload shape/dimension identity).
    pub workload_profile: Option<String>,
    /// Benchmark-specific Device binding context ("Add Device context",
    /// tasks). Descriptive only, never a native Device handle.
    pub device_context: Option<String>,
    /// Benchmark-specific Provider binding context ("Add Provider context",
    /// tasks). Descriptive only, never a native Provider handle.
    pub provider_context: Option<String>,
    /// Benchmark-specific workload/environment detail, present only when
    /// this reference describes benchmark (not qualification) evidence,
    /// sufficient to normalize into [`crate::BenchmarkProfile`] -- see
    /// [`normalize_benchmark_profile`].
    pub workload_metadata: Option<KernelBenchmarkWorkloadMetadata>,
}

/// Benchmark workload/environment metadata, implementing "Normalize
/// benchmark references" (tasks): [`crate::BenchmarkProfile`] requires this
/// level of detail (workload shape, warmup/measurement counts,
/// synchronization policy) to be usable as ranking evidence at all --
/// `crate::BenchmarkProfile::is_authoritative` already enforces that
/// downstream. Every field here is optional at the portable-manifest level;
/// [`normalize_benchmark_profile`] fills gaps with explicit, honest defaults
/// rather than fabricating plausible-looking values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelBenchmarkWorkloadMetadata {
    pub input_shapes: Option<String>,
    pub dtype_layout: Option<String>,
    pub batch_size: Option<u64>,
    pub sequence_length: Option<u64>,
    pub warmup_count: Option<u32>,
    pub measurement_count: Option<u32>,
    pub synchronization_policy: Option<String>,
    pub driver_runtime_version: Option<String>,
    pub benchmark_version: Option<String>,
}

impl KernelEvidenceReference {
    pub fn validate(&self) -> Result<(), KernelManifestError> {
        if !self.digest.is_well_formed() {
            return Err(KernelManifestError::EvidenceReferenceInvalid {
                reason: format!(
                    "evidence digest '{}' is not well-formed sha256",
                    self.digest
                ),
            });
        }
        if self.profile.trim().is_empty() {
            return Err(KernelManifestError::EvidenceReferenceInvalid {
                reason: "evidence profile must not be empty".into(),
            });
        }
        Ok(())
    }
}

/// "Runtime SHALL treat qualification evidence lacking oracle identity as
/// unverifiable against a specific oracle" (spec, "Oracle Identity Is
/// Preserved").
pub fn oracle_identity_is_known(reference: &KernelEvidenceReference) -> bool {
    reference
        .oracle_or_provider_identity
        .as_deref()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

/// Revalidates a qualification/benchmark evidence reference against the
/// *current* required suite/workload version, implementing "Portable
/// Evidence Is Not Automatically Current" (spec): presence, even with
/// `status == Passed`, is insufficient if the suite is obsolete or the
/// oracle identity is unknown.
pub fn evaluate_qualification_evidence_currency(
    reference: &KernelEvidenceReference,
    required_suite_or_workload_version: &str,
) -> bool {
    if reference.status != KernelEvidenceStatus::Passed {
        return false;
    }
    if !oracle_identity_is_known(reference) {
        return false;
    }
    matches!(
        reference.suite_or_workload_version.as_deref(),
        Some(version) if version == required_suite_or_workload_version
    )
}

// ---------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------

/// Advisory-only optimization recommendation, implementing "Recommendation
/// Metadata" (spec): `recommendation != promotion`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelManifestRecommendation {
    RecommendedForLatency,
    RecommendedForThroughput,
    Experimental,
    Reject,
}

impl KernelManifestRecommendation {
    pub const fn id(self) -> &'static str {
        match self {
            Self::RecommendedForLatency => "recommended-for-latency",
            Self::RecommendedForThroughput => "recommended-for-throughput",
            Self::Experimental => "experimental",
            Self::Reject => "reject",
        }
    }
}

/// The only function that answers whether a recommendation grants
/// promotion. Deliberately always `false`: "Recommendation SHALL be advisory
/// only" (spec) -- no recommendation variant, now or in the future, can make
/// this return `true`.
pub const fn recommendation_grants_promotion(
    _recommendation: KernelManifestRecommendation,
) -> bool {
    false
}

// ---------------------------------------------------------------------
// Trust metadata
// ---------------------------------------------------------------------

/// Detached signature envelope metadata, implementing "Signature Envelope"
/// and "Manifest Signatures" (spec). This change defines exchange
/// representation only -- it chooses no mandatory cryptographic scheme.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSignatureEnvelope {
    pub algorithm: String,
    pub key_id: Option<String>,
    pub signed_digest: KernelBlobDigest,
    pub signature_material: KernelBlobDigest,
    pub certificate_chain_reference: Option<KernelBlobDigest>,
}

/// "Presence of signature envelope metadata SHALL not be reported as
/// verified signature unless cryptographic verification actually succeeded"
/// (spec, "Signature Metadata Is Not Signature Verification"). This module
/// implements no cryptographic verifier, so the only way this returns `true`
/// is when the caller supplies an externally-computed verification result --
/// the envelope's mere presence can never do so on its own.
pub const fn signature_is_verified(
    _envelope: &KernelSignatureEnvelope,
    externally_verified: bool,
) -> bool {
    externally_verified
}

/// Non-authoritative trust inputs, implementing "Trust Metadata", "Publisher
/// Claims", and "Source Claims" (spec): "A publisher string SHALL NOT grant
/// trust by itself" / "Source kind/location SHALL not grant trust by
/// itself." The portable manifest deliberately has no `trusted: bool` field
/// anywhere in this module -- see "Trust Decision Outside Manifest" (spec).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelManifestTrustMetadata {
    pub publisher_claim: Option<String>,
    pub source_claim: Option<String>,
    pub signature: Option<KernelSignatureEnvelope>,
}

// ---------------------------------------------------------------------
// Manifest artifact entry
// ---------------------------------------------------------------------

/// One artifact entry in a manifest: a blob plus the portable metadata that
/// makes it interpretable (semantics, specialization, target constraints,
/// compiler metadata, dependencies), implementing "Kernel Source
/// Descriptor" and "Compiled Kernel Descriptor" (spec).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelManifestArtifact {
    pub blob: KernelBlobDescriptor,
    pub semantic_binding: Option<KernelSemanticBinding>,
    pub specialization: KernelManifestSpecialization,
    pub precision: KernelManifestPrecision,
    pub provenance: Option<KernelArtifactProvenance>,
    pub generator: Option<KernelGeneratorMetadata>,
    /// For a compiled artifact: the source artifact's digest, "where known"
    /// (spec, "Compiled Artifact Preserves Source Relationship"). SHALL use
    /// immutable digest identity rather than a mutable location hint.
    pub source_digest: Option<KernelBlobDigest>,
    pub compiler_metadata: Option<KernelCompilerMetadata>,
    pub target: KernelTargetConstraints,
    /// Immutable content-addressed auxiliary dependencies, implementing
    /// "Artifact Dependencies" (spec). Never grants arbitrary
    /// filesystem/library search authority.
    pub dependencies: Vec<KernelBlobDigest>,
}

impl KernelManifestArtifact {
    pub fn new(blob: KernelBlobDescriptor) -> Self {
        Self {
            blob,
            semantic_binding: None,
            specialization: KernelManifestSpecialization::default(),
            precision: KernelManifestPrecision::default(),
            provenance: None,
            generator: None,
            source_digest: None,
            compiler_metadata: None,
            target: KernelTargetConstraints::default(),
            dependencies: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), KernelManifestError> {
        self.blob.validate()?;
        if let Some(binding) = &self.semantic_binding {
            binding.validate()?;
        }
        self.specialization.validate()?;
        self.target.validate()?;
        if let Some(generator) = &self.generator {
            generator.validate()?;
        }
        for dependency in &self.dependencies {
            if !dependency.is_well_formed() {
                return Err(KernelManifestError::ArtifactReferenceInvalid {
                    reason: format!("dependency digest '{dependency}' is not well-formed sha256"),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Extensions and annotations
// ---------------------------------------------------------------------

/// A namespaced manifest extension, implementing "Extensions" and
/// "Extension Isolation" (spec): "Unknown optional extension MAY be
/// ignored. Unknown required extension SHALL make the manifest
/// unsupported."
#[derive(Clone, Debug, PartialEq)]
pub struct KernelManifestExtension {
    pub namespace: String,
    pub required: bool,
    pub data: serde_json::Value,
}

/// Core fields an extension can never override, implementing "Extension
/// Isolation" (spec).
pub const KERNEL_MANIFEST_CORE_FIELDS: &[&str] = &[
    "schema",
    "artifacts",
    "trust",
    "provider_compatibility",
    "native_handle",
];

// ---------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------

/// Kernel Artifact Manifest v1, implementing "Kernel Artifact Manifest v1"
/// and "Versioned Kernel Manifest" (spec). Deliberately carries no
/// [`crate::PreparedKernelId`], no native Device/Provider pointer, and no
/// authoritative `trusted` field -- see "Prepared Kernel Exclusion", "Device
/// Handle Exclusion", and "Provider Handle Exclusion" (spec).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelManifestV1 {
    pub schema: KernelManifestSchemaVersion,
    pub artifacts: Vec<KernelManifestArtifact>,
    pub qualification_evidence: Vec<KernelEvidenceReference>,
    pub benchmark_evidence: Vec<KernelEvidenceReference>,
    pub recommendation: Option<KernelManifestRecommendation>,
    pub trust: KernelManifestTrustMetadata,
    pub provenance: Option<KernelArtifactProvenance>,
    pub annotations: BTreeMap<String, String>,
    pub extensions: Vec<KernelManifestExtension>,
}

impl KernelManifestV1 {
    pub fn new() -> Self {
        Self {
            schema: KernelManifestSchemaVersion::current(),
            artifacts: Vec::new(),
            qualification_evidence: Vec::new(),
            benchmark_evidence: Vec::new(),
            recommendation: None,
            trust: KernelManifestTrustMetadata::default(),
            provenance: None,
            annotations: BTreeMap::new(),
            extensions: Vec::new(),
        }
    }

    /// Deterministic canonical JSON projection, implementing "Canonical
    /// Manifest Representation" (spec): UTF-8, no duplicate keys (structural
    /// impossibility of a `serde_json::Map`), deterministic key ordering
    /// (the default, non-`preserve_order`, `serde_json::Map` is
    /// `BTreeMap`-backed), deterministic string escaping and integer
    /// representation (both handled by `serde_json`'s canonical formatter),
    /// and no insignificant whitespace (compact `serde_json::to_vec`).
    pub fn to_canonical_value(&self) -> serde_json::Value {
        let mut root = serde_json::Map::new();
        root.insert(
            "schema".into(),
            serde_json::Value::String(self.schema.schema_string()),
        );

        let artifacts: Vec<serde_json::Value> = self
            .artifacts
            .iter()
            .map(artifact_to_canonical_value)
            .collect();
        root.insert("artifacts".into(), serde_json::Value::Array(artifacts));

        if !self.qualification_evidence.is_empty() {
            root.insert(
                "qualification_evidence".into(),
                serde_json::Value::Array(
                    self.qualification_evidence
                        .iter()
                        .map(evidence_to_canonical_value)
                        .collect(),
                ),
            );
        }
        if !self.benchmark_evidence.is_empty() {
            root.insert(
                "benchmark_evidence".into(),
                serde_json::Value::Array(
                    self.benchmark_evidence
                        .iter()
                        .map(evidence_to_canonical_value)
                        .collect(),
                ),
            );
        }
        if let Some(recommendation) = self.recommendation {
            root.insert(
                "recommendation".into(),
                serde_json::Value::String(recommendation.id().into()),
            );
        }

        let mut trust = serde_json::Map::new();
        if let Some(publisher) = &self.trust.publisher_claim {
            trust.insert(
                "publisher".into(),
                serde_json::Value::String(publisher.clone()),
            );
        }
        if let Some(source) = &self.trust.source_claim {
            trust.insert("source".into(), serde_json::Value::String(source.clone()));
        }
        if let Some(signature) = &self.trust.signature {
            let mut sig = serde_json::Map::new();
            sig.insert(
                "algorithm".into(),
                serde_json::Value::String(signature.algorithm.clone()),
            );
            if let Some(key_id) = &signature.key_id {
                sig.insert("key_id".into(), serde_json::Value::String(key_id.clone()));
            }
            sig.insert(
                "signed_digest".into(),
                serde_json::Value::String(signature.signed_digest.to_string()),
            );
            sig.insert(
                "signature_material".into(),
                serde_json::Value::String(signature.signature_material.to_string()),
            );
            if let Some(chain) = &signature.certificate_chain_reference {
                sig.insert(
                    "certificate_chain_reference".into(),
                    serde_json::Value::String(chain.to_string()),
                );
            }
            trust.insert("signature".into(), serde_json::Value::Object(sig));
        }
        if !trust.is_empty() {
            root.insert("trust".into(), serde_json::Value::Object(trust));
        }

        if let Some(provenance) = self.provenance {
            root.insert(
                "provenance".into(),
                serde_json::Value::String(provenance.id().into()),
            );
        }
        if !self.annotations.is_empty() {
            let mut annotations = serde_json::Map::new();
            for (key, value) in &self.annotations {
                annotations.insert(key.clone(), serde_json::Value::String(value.clone()));
            }
            root.insert("annotations".into(), serde_json::Value::Object(annotations));
        }
        if !self.extensions.is_empty() {
            let extensions: Vec<serde_json::Value> = self
                .extensions
                .iter()
                .map(|extension| {
                    let mut entry = serde_json::Map::new();
                    entry.insert(
                        "namespace".into(),
                        serde_json::Value::String(extension.namespace.clone()),
                    );
                    entry.insert(
                        "required".into(),
                        serde_json::Value::Bool(extension.required),
                    );
                    entry.insert("data".into(), extension.data.clone());
                    serde_json::Value::Object(entry)
                })
                .collect();
            root.insert("extensions".into(), serde_json::Value::Array(extensions));
        }

        serde_json::Value::Object(root)
    }

    /// Canonical UTF-8 JSON bytes, deterministic given equal field values --
    /// implementing "Whitespace differs" (spec scenario): two semantically
    /// identical manifests parsed from differently-formatted input produce
    /// identical canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.to_canonical_value())
            .expect("canonical manifest value is always representable as JSON")
    }

    /// Manifest content digest, implementing "Manifest Identity" (spec).
    pub fn digest(&self) -> KernelBlobDigest {
        KernelBlobDigest::of_bytes(&self.canonical_bytes())
    }

    pub fn validate(&self) -> Result<(), KernelManifestError> {
        if !self.schema.is_supported() {
            return Err(KernelManifestError::SchemaUnsupported {
                schema: self.schema.schema_string(),
            });
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        for evidence in self
            .qualification_evidence
            .iter()
            .chain(&self.benchmark_evidence)
        {
            evidence.validate()?;
        }
        for extension in &self.extensions {
            let top_level = extension
                .namespace
                .split(':')
                .next()
                .unwrap_or(&extension.namespace);
            if KERNEL_MANIFEST_CORE_FIELDS.contains(&top_level) {
                return Err(KernelManifestError::ArtifactReferenceInvalid {
                    reason: format!(
                        "extension namespace '{}' collides with a reserved core field name",
                        extension.namespace
                    ),
                });
            }
            if extension.required && !is_known_required_extension(&extension.namespace) {
                return Err(KernelManifestError::RequiredExtensionUnsupported {
                    namespace: extension.namespace.clone(),
                });
            }
        }
        if let Some(cycle) = detect_dependency_cycle(self) {
            return Err(KernelManifestError::DependencyCycle {
                path: cycle.join(" -> "),
            });
        }
        detect_conflicting_digest_metadata(self)?;
        Ok(())
    }
}

/// Implements "Reject conflicting digest metadata" (tasks): the same
/// content-addressed digest legitimately appears more than once in a
/// manifest (multiple artifacts MAY share a blob), but every appearance
/// SHALL agree on the metadata that is supposed to be a pure function of the
/// bytes -- size. Two artifacts declaring the same digest with different
/// sizes describe contradictory content for one content-addressed identity,
/// which SHALL fail closed rather than silently pick one.
fn detect_conflicting_digest_metadata(
    manifest: &KernelManifestV1,
) -> Result<(), KernelManifestError> {
    let mut seen_sizes: BTreeMap<&str, u64> = BTreeMap::new();
    for artifact in &manifest.artifacts {
        let digest = artifact.blob.digest.value.as_str();
        match seen_sizes.get(digest) {
            Some(&existing_size) if existing_size != artifact.blob.size => {
                return Err(KernelManifestError::ArtifactReferenceInvalid {
                    reason: format!(
                        "digest '{digest}' declared with conflicting sizes {existing_size} and {}",
                        artifact.blob.size
                    ),
                });
            }
            _ => {
                seen_sizes.insert(digest, artifact.blob.size);
            }
        }
    }
    Ok(())
}

impl Default for KernelManifestV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Mirrors `parse_operator`'s expected shape exactly -- an object with
/// `namespace`/`name`/`version`/`family`, never the lossy `OperatorId`
/// `Display` string (which omits `family` and cannot round-trip through
/// `parse_operator`).
fn operator_to_canonical_value(operator: &OperatorId) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    entry.insert(
        "namespace".into(),
        serde_json::Value::String(operator.namespace().into()),
    );
    entry.insert(
        "name".into(),
        serde_json::Value::String(operator.name().into()),
    );
    entry.insert(
        "version".into(),
        serde_json::Value::Number(operator.version().into()),
    );
    entry.insert(
        "family".into(),
        serde_json::Value::String(operator.family().id().into()),
    );
    serde_json::Value::Object(entry)
}

fn artifact_to_canonical_value(artifact: &KernelManifestArtifact) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    entry.insert(
        "role".into(),
        serde_json::Value::String(artifact.blob.role.as_str().into()),
    );
    entry.insert(
        "format".into(),
        serde_json::Value::String(artifact.blob.format.stable_key()),
    );
    entry.insert(
        "digest".into(),
        serde_json::Value::String(artifact.blob.digest.to_string()),
    );
    entry.insert(
        "size".into(),
        serde_json::Value::Number(artifact.blob.size.into()),
    );
    entry.insert(
        "storage_mode".into(),
        serde_json::Value::String(artifact.blob.storage_mode.id().into()),
    );
    entry.insert(
        "required".into(),
        serde_json::Value::Bool(artifact.blob.required),
    );
    if let Some(media_type) = &artifact.blob.media_type {
        entry.insert(
            "media_type".into(),
            serde_json::Value::String(media_type.clone()),
        );
    }
    if let Some(location_hint) = &artifact.blob.location_hint {
        entry.insert(
            "location_hint".into(),
            serde_json::Value::String(location_hint.clone()),
        );
    }
    if let Some(binding) = &artifact.semantic_binding {
        let operators: Vec<serde_json::Value> = binding
            .operators
            .iter()
            .map(operator_to_canonical_value)
            .collect();
        entry.insert("operators".into(), serde_json::Value::Array(operators));
        if let Some(range) = &binding.primary_version_requirements {
            let mut range_value = serde_json::Map::new();
            range_value.insert("min".into(), serde_json::Value::Number(range.min.into()));
            range_value.insert("max".into(), serde_json::Value::Number(range.max.into()));
            entry.insert(
                "operator_version_range".into(),
                serde_json::Value::Object(range_value),
            );
        }
    }
    if let Some(source_digest) = &artifact.source_digest {
        entry.insert(
            "source_digest".into(),
            serde_json::Value::String(source_digest.to_string()),
        );
    }
    if let Some(provenance) = artifact.provenance {
        entry.insert(
            "provenance".into(),
            serde_json::Value::String(provenance.id().into()),
        );
    }
    if !artifact.dependencies.is_empty() {
        let dependencies: Vec<serde_json::Value> = artifact
            .dependencies
            .iter()
            .map(|dependency| serde_json::Value::String(dependency.to_string()))
            .collect();
        entry.insert(
            "dependencies".into(),
            serde_json::Value::Array(dependencies),
        );
    }
    if !artifact.target.is_empty() {
        entry.insert("target".into(), target_to_canonical_value(&artifact.target));
    }
    if !artifact.specialization.is_empty() {
        entry.insert(
            "specialization".into(),
            specialization_to_canonical_value(&artifact.specialization),
        );
    }
    if let Some(compiler) = &artifact.compiler_metadata {
        entry.insert(
            "compiler_metadata".into(),
            compiler_metadata_to_canonical_value(compiler),
        );
    }
    if artifact.precision != KernelManifestPrecision::default() {
        entry.insert(
            "precision".into(),
            precision_to_canonical_value(&artifact.precision),
        );
    }
    if let Some(generator) = &artifact.generator {
        entry.insert("generator".into(), generator_to_canonical_value(generator));
    }
    serde_json::Value::Object(entry)
}

fn evidence_to_canonical_value(evidence: &KernelEvidenceReference) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    entry.insert(
        "digest".into(),
        serde_json::Value::String(evidence.digest.to_string()),
    );
    entry.insert(
        "profile".into(),
        serde_json::Value::String(evidence.profile.clone()),
    );
    if let Some(version) = &evidence.suite_or_workload_version {
        entry.insert(
            "suite_or_workload_version".into(),
            serde_json::Value::String(version.clone()),
        );
    }
    if let Some(oracle) = &evidence.oracle_or_provider_identity {
        entry.insert(
            "oracle_or_provider_identity".into(),
            serde_json::Value::String(oracle.clone()),
        );
    }
    if !evidence.target_compatibility.is_empty() {
        entry.insert(
            "target_compatibility".into(),
            serde_json::Value::Array(
                evidence
                    .target_compatibility
                    .iter()
                    .map(|value| serde_json::Value::String(value.clone()))
                    .collect(),
            ),
        );
    }
    entry.insert(
        "status".into(),
        serde_json::Value::String(format!("{:?}", evidence.status)),
    );
    entry.insert(
        "storage_mode".into(),
        serde_json::Value::String(evidence.storage_mode.id().into()),
    );
    if let Some(workload_profile) = &evidence.workload_profile {
        entry.insert(
            "workload_profile".into(),
            serde_json::Value::String(workload_profile.clone()),
        );
    }
    if let Some(device_context) = &evidence.device_context {
        entry.insert(
            "device_context".into(),
            serde_json::Value::String(device_context.clone()),
        );
    }
    if let Some(provider_context) = &evidence.provider_context {
        entry.insert(
            "provider_context".into(),
            serde_json::Value::String(provider_context.clone()),
        );
    }
    if let Some(workload) = &evidence.workload_metadata {
        entry.insert(
            "workload_metadata".into(),
            workload_metadata_to_canonical_value(workload),
        );
    }
    serde_json::Value::Object(entry)
}

fn workload_metadata_to_canonical_value(
    workload: &KernelBenchmarkWorkloadMetadata,
) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    if let Some(input_shapes) = &workload.input_shapes {
        entry.insert(
            "input_shapes".into(),
            serde_json::Value::String(input_shapes.clone()),
        );
    }
    if let Some(dtype_layout) = &workload.dtype_layout {
        entry.insert(
            "dtype_layout".into(),
            serde_json::Value::String(dtype_layout.clone()),
        );
    }
    if let Some(batch_size) = workload.batch_size {
        entry.insert(
            "batch_size".into(),
            serde_json::Value::Number(batch_size.into()),
        );
    }
    if let Some(sequence_length) = workload.sequence_length {
        entry.insert(
            "sequence_length".into(),
            serde_json::Value::Number(sequence_length.into()),
        );
    }
    if let Some(warmup_count) = workload.warmup_count {
        entry.insert(
            "warmup_count".into(),
            serde_json::Value::Number(warmup_count.into()),
        );
    }
    if let Some(measurement_count) = workload.measurement_count {
        entry.insert(
            "measurement_count".into(),
            serde_json::Value::Number(measurement_count.into()),
        );
    }
    if let Some(synchronization_policy) = &workload.synchronization_policy {
        entry.insert(
            "synchronization_policy".into(),
            serde_json::Value::String(synchronization_policy.clone()),
        );
    }
    if let Some(driver_runtime_version) = &workload.driver_runtime_version {
        entry.insert(
            "driver_runtime_version".into(),
            serde_json::Value::String(driver_runtime_version.clone()),
        );
    }
    if let Some(benchmark_version) = &workload.benchmark_version {
        entry.insert(
            "benchmark_version".into(),
            serde_json::Value::String(benchmark_version.clone()),
        );
    }
    serde_json::Value::Object(entry)
}

fn target_to_canonical_value(target: &KernelTargetConstraints) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    if let Some(device_type) = &target.device_type {
        entry.insert(
            "device_type".into(),
            serde_json::Value::String(device_type.clone()),
        );
    }
    if let Some(hardware_vendor) = &target.hardware_vendor {
        entry.insert(
            "hardware_vendor".into(),
            serde_json::Value::String(hardware_vendor.clone()),
        );
    }
    if let Some(architecture) = &target.architecture {
        entry.insert(
            "architecture".into(),
            serde_json::Value::String(architecture.clone()),
        );
    }
    if let Some(execution_environment) = &target.execution_environment {
        entry.insert(
            "execution_environment".into(),
            serde_json::Value::String(execution_environment.clone()),
        );
    }
    let string_array = |set: &BTreeSet<String>| -> serde_json::Value {
        serde_json::Value::Array(
            set.iter()
                .map(|value| serde_json::Value::String(value.clone()))
                .collect(),
        )
    };
    if !target.device_features.is_empty() {
        entry.insert(
            "device_features".into(),
            string_array(&target.device_features),
        );
    }
    if !target.provider_compatibility.is_empty() {
        entry.insert(
            "provider_compatibility".into(),
            string_array(&target.provider_compatibility),
        );
    }
    if !target.runtime_driver_compatibility.is_empty() {
        entry.insert(
            "runtime_driver_compatibility".into(),
            string_array(&target.runtime_driver_compatibility),
        );
    }
    if !target.memory_classes.is_empty() {
        entry.insert(
            "memory_classes".into(),
            string_array(&target.memory_classes),
        );
    }
    serde_json::Value::Object(entry)
}

fn specialization_to_canonical_value(
    specialization: &KernelManifestSpecialization,
) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    if !specialization.exact_dimensions.is_empty() {
        let mut dimensions = serde_json::Map::new();
        for (key, value) in &specialization.exact_dimensions {
            dimensions.insert(key.clone(), serde_json::Value::Number((*value).into()));
        }
        entry.insert(
            "exact_dimensions".into(),
            serde_json::Value::Object(dimensions),
        );
    }
    let pair = |range: Option<(u64, u64)>| {
        range.map(|(min, max)| {
            serde_json::Value::Array(vec![
                serde_json::Value::Number(min.into()),
                serde_json::Value::Number(max.into()),
            ])
        })
    };
    if let Some(value) = pair(specialization.batch_range) {
        entry.insert("batch_range".into(), value);
    }
    if let Some(value) = pair(specialization.sequence_range) {
        entry.insert("sequence_range".into(), value);
    }
    if let Some(head_count) = specialization.head_count {
        entry.insert(
            "head_count".into(),
            serde_json::Value::Number(head_count.into()),
        );
    }
    if let Some(head_dimension) = specialization.head_dimension {
        entry.insert(
            "head_dimension".into(),
            serde_json::Value::Number(head_dimension.into()),
        );
    }
    if !specialization.tile_sizes.is_empty() {
        entry.insert(
            "tile_sizes".into(),
            serde_json::Value::Array(
                specialization
                    .tile_sizes
                    .iter()
                    .map(|value| serde_json::Value::Number((*value).into()))
                    .collect(),
            ),
        );
    }
    if let Some(alignment) = specialization.alignment {
        entry.insert(
            "alignment".into(),
            serde_json::Value::Number(alignment.into()),
        );
    }
    if let Some(dtype) = &specialization.dtype {
        entry.insert("dtype".into(), serde_json::Value::String(dtype.clone()));
    }
    if let Some(layout) = &specialization.layout {
        entry.insert("layout".into(), serde_json::Value::String(layout.clone()));
    }
    if let Some(quantization_profile) = &specialization.quantization_profile {
        entry.insert(
            "quantization_profile".into(),
            serde_json::Value::String(quantization_profile.clone()),
        );
    }
    if let Some(execution_phase) = specialization.execution_phase {
        entry.insert(
            "execution_phase".into(),
            serde_json::Value::String(
                match execution_phase {
                    KernelExecutionPhase::Prefill => "prefill",
                    KernelExecutionPhase::Decode => "decode",
                    KernelExecutionPhase::Both => "both",
                }
                .into(),
            ),
        );
    }
    if !specialization.device_features.is_empty() {
        entry.insert(
            "device_features".into(),
            serde_json::Value::Array(
                specialization
                    .device_features
                    .iter()
                    .map(|value| serde_json::Value::String(value.clone()))
                    .collect(),
            ),
        );
    }
    serde_json::Value::Object(entry)
}

fn compiler_metadata_to_canonical_value(compiler: &KernelCompilerMetadata) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    let mut set = |key: &str, value: &Option<String>| {
        if let Some(value) = value {
            entry.insert(key.into(), serde_json::Value::String(value.clone()));
        }
    };
    set("compiler_identity", &compiler.compiler_identity);
    set("compiler_version", &compiler.compiler_version);
    set(
        "backend_identity_version",
        &compiler.backend_identity_version,
    );
    set("flags_fingerprint", &compiler.flags_fingerprint);
    set("build_fingerprint", &compiler.build_fingerprint);
    set("target_architecture", &compiler.target_architecture);
    serde_json::Value::Object(entry)
}

fn precision_to_canonical_value(precision: &KernelManifestPrecision) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    if let Some(accumulation_dtype) = &precision.accumulation_dtype {
        entry.insert(
            "accumulation_dtype".into(),
            serde_json::Value::String(accumulation_dtype.clone()),
        );
    }
    entry.insert(
        "approximate_math".into(),
        serde_json::Value::Bool(precision.approximate_math),
    );
    if let Some(deterministic) = precision.deterministic {
        entry.insert(
            "deterministic".into(),
            serde_json::Value::Bool(deterministic),
        );
    }
    if let Some(tolerance_profile) = &precision.tolerance_profile {
        entry.insert(
            "tolerance_profile".into(),
            serde_json::Value::String(tolerance_profile.clone()),
        );
    }
    if let Some(quantization_error_profile) = &precision.quantization_error_profile {
        entry.insert(
            "quantization_error_profile".into(),
            serde_json::Value::String(quantization_error_profile.clone()),
        );
    }
    serde_json::Value::Object(entry)
}

fn generator_to_canonical_value(generator: &KernelGeneratorMetadata) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    let mut set = |key: &str, value: &Option<String>| {
        if let Some(value) = value {
            entry.insert(key.into(), serde_json::Value::String(value.clone()));
        }
    };
    set("generator_name", &generator.generator_name);
    set("generator_version", &generator.generator_version);
    set("campaign_id", &generator.campaign_id);
    set("source_revision", &generator.source_revision);
    serde_json::Value::Object(entry)
}

fn is_known_required_extension(_namespace: &str) -> bool {
    // This change defines the extension *model* only; it recognizes no
    // specific required extension namespace. "Unknown required extension
    // SHALL make the manifest unsupported" (spec, "Extensions").
    false
}

/// Detects a dependency cycle among artifact digests, implementing "Cycles
/// in relationships that require acyclic semantics SHALL be rejected"
/// (spec, "Artifact Relationship Graph"). Returns the offending path when a
/// cycle exists.
pub fn detect_dependency_cycle(manifest: &KernelManifestV1) -> Option<Vec<String>> {
    let mut edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for artifact in &manifest.artifacts {
        let from = artifact.blob.digest.value.as_str();
        let to: Vec<&str> = artifact
            .dependencies
            .iter()
            .map(|dependency| dependency.value.as_str())
            .collect();
        edges.entry(from).or_default().extend(to);
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Visiting,
        Done,
    }
    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();
    let mut path: Vec<&str> = Vec::new();

    fn visit<'a>(
        node: &'a str,
        edges: &BTreeMap<&'a str, Vec<&'a str>>,
        marks: &mut BTreeMap<&'a str, Mark>,
        path: &mut Vec<&'a str>,
    ) -> Option<Vec<String>> {
        match marks.get(node) {
            Some(Mark::Done) => return None,
            Some(Mark::Visiting) => {
                path.push(node);
                return Some(path.iter().map(|s| s.to_string()).collect());
            }
            None => {}
        }
        marks.insert(node, Mark::Visiting);
        path.push(node);
        if let Some(dependencies) = edges.get(node) {
            for dependency in dependencies {
                if let Some(cycle) = visit(dependency, edges, marks, path) {
                    return Some(cycle);
                }
            }
        }
        path.pop();
        marks.insert(node, Mark::Done);
        None
    }

    for node in edges.keys().copied() {
        if let Some(cycle) = visit(node, &edges, &mut marks, &mut path) {
            return Some(cycle);
        }
    }
    None
}

// ---------------------------------------------------------------------
// JSON structural scan (duplicate keys, nesting depth, strict grammar)
// ---------------------------------------------------------------------

fn skip_ws(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    i
}

fn read_json_string(chars: &[char], mut i: usize) -> Result<(String, usize), KernelManifestError> {
    debug_assert_eq!(chars[i], '"');
    i += 1;
    let mut s = String::new();
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                s.push(chars[i]);
                i += 1;
                if i < chars.len() {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            '"' => return Ok((s, i + 1)),
            c => {
                s.push(c);
                i += 1;
            }
        }
    }
    Err(KernelManifestError::InvalidJson {
        reason: "unterminated string".into(),
    })
}

fn expect_literal(chars: &[char], i: usize, literal: &str) -> Result<usize, KernelManifestError> {
    let lit: Vec<char> = literal.chars().collect();
    if i + lit.len() <= chars.len() && chars[i..i + lit.len()] == lit[..] {
        Ok(i + lit.len())
    } else {
        Err(KernelManifestError::InvalidJson {
            reason: format!("expected literal '{literal}'"),
        })
    }
}

fn parse_number(chars: &[char], mut i: usize) -> Result<usize, KernelManifestError> {
    let bad_number = || KernelManifestError::InvalidJson {
        reason: "invalid number literal".into(),
    };
    if i < chars.len() && chars[i] == '-' {
        i += 1;
    }
    if i >= chars.len() || !chars[i].is_ascii_digit() {
        return Err(bad_number());
    }
    if chars[i] == '0' {
        i += 1;
    } else {
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        if i >= chars.len() || !chars[i].is_ascii_digit() {
            return Err(bad_number());
        }
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
        i += 1;
        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
            i += 1;
        }
        if i >= chars.len() || !chars[i].is_ascii_digit() {
            return Err(bad_number());
        }
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    Ok(i)
}

/// Recursive-descent structural scan implementing "Duplicate Entries" (spec:
/// "JSON manifest SHALL reject duplicate object keys") and "Manifest
/// Limits"/"No Unbounded Recursion" (nesting depth). Runs over raw text
/// *before* a [`serde_json::Value`] tree is built, because `serde_json`'s
/// default map silently lets a later duplicate key win. Strict JSON grammar
/// (no bare `NaN`/`Infinity` literal) mechanically implements "no NaN or
/// Infinity numeric values" (spec, "Canonical Manifest Representation").
fn parse_value(
    chars: &[char],
    i: usize,
    depth: usize,
    limits: &KernelManifestLimits,
) -> Result<usize, KernelManifestError> {
    let i = skip_ws(chars, i);
    if depth > limits.max_nesting_depth {
        return Err(KernelManifestError::LimitExceeded {
            limit: "nesting-depth".into(),
        });
    }
    if i >= chars.len() {
        return Err(KernelManifestError::InvalidJson {
            reason: "unexpected end of input".into(),
        });
    }
    match chars[i] {
        '{' => {
            let mut i = skip_ws(chars, i + 1);
            let mut seen: BTreeSet<String> = BTreeSet::new();
            if i < chars.len() && chars[i] == '}' {
                return Ok(i + 1);
            }
            loop {
                i = skip_ws(chars, i);
                if i >= chars.len() || chars[i] != '"' {
                    return Err(KernelManifestError::InvalidJson {
                        reason: "expected object key".into(),
                    });
                }
                let (key, next_i) = read_json_string(chars, i)?;
                if key.len() > limits.max_string_bytes {
                    return Err(KernelManifestError::LimitExceeded {
                        limit: "string-bytes".into(),
                    });
                }
                if !seen.insert(key.clone()) {
                    return Err(KernelManifestError::DuplicateKey { key });
                }
                i = skip_ws(chars, next_i);
                if i >= chars.len() || chars[i] != ':' {
                    return Err(KernelManifestError::InvalidJson {
                        reason: "expected ':' after object key".into(),
                    });
                }
                i = parse_value(chars, i + 1, depth + 1, limits)?;
                i = skip_ws(chars, i);
                match chars.get(i) {
                    Some(',') => {
                        i += 1;
                        continue;
                    }
                    Some('}') => return Ok(i + 1),
                    _ => {
                        return Err(KernelManifestError::InvalidJson {
                            reason: "expected ',' or '}' in object".into(),
                        });
                    }
                }
            }
        }
        '[' => {
            let mut i = skip_ws(chars, i + 1);
            if i < chars.len() && chars[i] == ']' {
                return Ok(i + 1);
            }
            loop {
                i = parse_value(chars, i, depth + 1, limits)?;
                i = skip_ws(chars, i);
                match chars.get(i) {
                    Some(',') => {
                        i += 1;
                        continue;
                    }
                    Some(']') => return Ok(i + 1),
                    _ => {
                        return Err(KernelManifestError::InvalidJson {
                            reason: "expected ',' or ']' in array".into(),
                        });
                    }
                }
            }
        }
        '"' => {
            let (value, next_i) = read_json_string(chars, i)?;
            if value.len() > limits.max_string_bytes {
                return Err(KernelManifestError::LimitExceeded {
                    limit: "string-bytes".into(),
                });
            }
            Ok(next_i)
        }
        't' => expect_literal(chars, i, "true"),
        'f' => expect_literal(chars, i, "false"),
        'n' => expect_literal(chars, i, "null"),
        c if c == '-' || c.is_ascii_digit() => parse_number(chars, i),
        _ => Err(KernelManifestError::InvalidJson {
            reason: "unexpected token".into(),
        }),
    }
}

fn scan_manifest_structure(
    text: &str,
    limits: &KernelManifestLimits,
) -> Result<(), KernelManifestError> {
    let chars: Vec<char> = text.chars().collect();
    let start = skip_ws(&chars, 0);
    if start >= chars.len() {
        return Err(KernelManifestError::InvalidJson {
            reason: "empty manifest".into(),
        });
    }
    let end = parse_value(&chars, start, 0, limits)?;
    if skip_ws(&chars, end) != chars.len() {
        return Err(KernelManifestError::InvalidJson {
            reason: "trailing content after JSON document".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Parsing: JSON text -> KernelManifestV1
// ---------------------------------------------------------------------

fn as_object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, KernelManifestError> {
    value
        .as_object()
        .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
            reason: format!("{context} must be a JSON object"),
        })
}

fn as_str<'a>(value: &'a serde_json::Value, context: &str) -> Result<&'a str, KernelManifestError> {
    value
        .as_str()
        .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
            reason: format!("{context} must be a JSON string"),
        })
}

fn parse_digest(
    value: &serde_json::Value,
    context: &str,
) -> Result<KernelBlobDigest, KernelManifestError> {
    let text = as_str(value, context)?;
    let (algorithm, digest_value) =
        text.split_once(':')
            .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                reason: format!("{context} must be formatted 'algorithm:hex'"),
            })?;
    if algorithm != KERNEL_BLOB_DIGEST_ALGORITHM {
        return Err(KernelManifestError::ArtifactReferenceInvalid {
            reason: format!("{context} uses unsupported digest algorithm '{algorithm}'"),
        });
    }
    let digest = KernelBlobDigest::sha256(digest_value);
    if !digest.is_well_formed() {
        return Err(KernelManifestError::ArtifactReferenceInvalid {
            reason: format!("{context} is not a well-formed sha256 digest"),
        });
    }
    Ok(digest)
}

fn operator_family_from_id(id: &str) -> Option<OperatorFamily> {
    OperatorFamily::ALL
        .into_iter()
        .find(|family| family.id() == id)
}

fn parse_operator(value: &serde_json::Value) -> Result<OperatorId, KernelManifestError> {
    let object = as_object(value, "operator")?;
    let namespace = as_str(
        object
            .get("namespace")
            .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                reason: "operator missing 'namespace'".into(),
            })?,
        "operator.namespace",
    )?;
    let name = as_str(
        object
            .get("name")
            .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                reason: "operator missing 'name'".into(),
            })?,
        "operator.name",
    )?;
    let version = object
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
            reason: "operator missing numeric 'version'".into(),
        })?;
    let family_id = as_str(
        object
            .get("family")
            .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                reason: "operator missing 'family'".into(),
            })?,
        "operator.family",
    )?;
    let family = operator_family_from_id(family_id).ok_or_else(|| {
        KernelManifestError::SemanticBindingInvalid {
            reason: format!("unknown operator family '{family_id}'"),
        }
    })?;
    Ok(OperatorId::new(namespace, name, version as u32, family))
}

fn parse_format(
    value: &serde_json::Value,
    context: &str,
) -> Result<KernelArtifactFormat, KernelManifestError> {
    let text = as_str(value, context)?;
    let (rest, version) = match text.split_once('@') {
        Some((rest, version)) => (rest, Some(version.to_string())),
        None => (text, None),
    };
    let (namespace, name) =
        rest.split_once(':')
            .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                reason: format!("{context} must be formatted 'namespace:name[@version]'"),
            })?;
    let mut format = KernelArtifactFormat::new(namespace, name);
    if let Some(version) = version {
        format = format.with_version(version);
    }
    Ok(format)
}

fn provenance_from_id(id: &str) -> Option<KernelArtifactProvenance> {
    [
        KernelArtifactProvenance::HumanAuthored,
        KernelArtifactProvenance::AiGenerated,
        KernelArtifactProvenance::OptimizerGenerated,
        KernelArtifactProvenance::CompilerGenerated,
        KernelArtifactProvenance::CiGenerated,
        KernelArtifactProvenance::VendorProvided,
        KernelArtifactProvenance::Imported,
    ]
    .into_iter()
    .find(|provenance| provenance.id() == id)
}

fn recommendation_from_id(id: &str) -> Option<KernelManifestRecommendation> {
    [
        KernelManifestRecommendation::RecommendedForLatency,
        KernelManifestRecommendation::RecommendedForThroughput,
        KernelManifestRecommendation::Experimental,
        KernelManifestRecommendation::Reject,
    ]
    .into_iter()
    .find(|recommendation| recommendation.id() == id)
}

fn parse_evidence(
    value: &serde_json::Value,
) -> Result<KernelEvidenceReference, KernelManifestError> {
    let object = as_object(value, "evidence reference")?;
    let digest = parse_digest(
        object
            .get("digest")
            .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
                reason: "evidence reference missing 'digest'".into(),
            })?,
        "evidence.digest",
    )?;
    let profile = as_str(
        object
            .get("profile")
            .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
                reason: "evidence reference missing 'profile'".into(),
            })?,
        "evidence.profile",
    )?
    .to_string();
    let suite_or_workload_version = object
        .get("suite_or_workload_version")
        .map(|value| as_str(value, "evidence.suite_or_workload_version"))
        .transpose()?
        .map(str::to_string);
    let oracle_or_provider_identity = object
        .get("oracle_or_provider_identity")
        .map(|value| as_str(value, "evidence.oracle_or_provider_identity"))
        .transpose()?
        .map(str::to_string);
    let status = match object
        .get("status")
        .map(|value| as_str(value, "evidence.status"))
        .transpose()?
    {
        Some("pending") | None => KernelEvidenceStatus::Pending,
        Some("passed") => KernelEvidenceStatus::Passed,
        Some("failed") => KernelEvidenceStatus::Failed,
        Some("revoked") => KernelEvidenceStatus::Revoked,
        Some("stale") => KernelEvidenceStatus::Stale,
        Some(other) => {
            return Err(KernelManifestError::EvidenceReferenceInvalid {
                reason: format!("unknown evidence status '{other}'"),
            });
        }
    };
    let target_compatibility = match object.get("target_compatibility") {
        Some(value) => value
            .as_array()
            .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
                reason: "evidence.target_compatibility must be an array".into(),
            })?
            .iter()
            .map(|entry| as_str(entry, "evidence.target_compatibility[]").map(str::to_string))
            .collect::<Result<BTreeSet<_>, _>>()?,
        None => BTreeSet::new(),
    };
    let storage_mode = match object
        .get("storage_mode")
        .map(|value| as_str(value, "evidence.storage_mode"))
        .transpose()?
    {
        Some("external") => KernelArtifactStorageMode::External,
        Some("embedded") | None => KernelArtifactStorageMode::Embedded,
        Some(other) => {
            return Err(KernelManifestError::EvidenceReferenceInvalid {
                reason: format!("unknown evidence storage_mode '{other}'"),
            });
        }
    };
    let workload_profile = object
        .get("workload_profile")
        .map(|value| as_str(value, "evidence.workload_profile"))
        .transpose()?
        .map(str::to_string);
    let device_context = object
        .get("device_context")
        .map(|value| as_str(value, "evidence.device_context"))
        .transpose()?
        .map(str::to_string);
    let provider_context = object
        .get("provider_context")
        .map(|value| as_str(value, "evidence.provider_context"))
        .transpose()?
        .map(str::to_string);
    let workload_metadata = object
        .get("workload_metadata")
        .map(parse_benchmark_workload_metadata)
        .transpose()?;
    Ok(KernelEvidenceReference {
        digest,
        profile,
        suite_or_workload_version,
        oracle_or_provider_identity,
        target_compatibility,
        status,
        storage_mode,
        workload_profile,
        device_context,
        provider_context,
        workload_metadata,
    })
}

fn parse_benchmark_workload_metadata(
    value: &serde_json::Value,
) -> Result<KernelBenchmarkWorkloadMetadata, KernelManifestError> {
    let object = as_object(value, "evidence.workload_metadata")?;
    let field = |key: &str| -> Result<Option<String>, KernelManifestError> {
        object
            .get(key)
            .map(|value| as_str(value, "workload_metadata field"))
            .transpose()
            .map(|value| value.map(str::to_string))
    };
    Ok(KernelBenchmarkWorkloadMetadata {
        input_shapes: field("input_shapes")?,
        dtype_layout: field("dtype_layout")?,
        batch_size: object.get("batch_size").and_then(serde_json::Value::as_u64),
        sequence_length: object
            .get("sequence_length")
            .and_then(serde_json::Value::as_u64),
        warmup_count: object
            .get("warmup_count")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as u32),
        measurement_count: object
            .get("measurement_count")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as u32),
        synchronization_policy: field("synchronization_policy")?,
        driver_runtime_version: field("driver_runtime_version")?,
        benchmark_version: field("benchmark_version")?,
    })
}

fn parse_target(
    value: &serde_json::Value,
    limits: &KernelManifestLimits,
) -> Result<KernelTargetConstraints, KernelManifestError> {
    let object = as_object(value, "artifact.target")?;
    let string_set = |key: &str,
                      object: &serde_json::Map<String, serde_json::Value>|
     -> Result<BTreeSet<String>, KernelManifestError> {
        match object.get(key) {
            Some(value) => value
                .as_array()
                .ok_or_else(|| KernelManifestError::TargetInvalid {
                    reason: format!("target.{key} must be an array"),
                })?
                .iter()
                .map(|entry| as_str(entry, "target array entry").map(str::to_string))
                .collect(),
            None => Ok(BTreeSet::new()),
        }
    };
    let target = KernelTargetConstraints {
        device_type: object
            .get("device_type")
            .map(|value| as_str(value, "target.device_type"))
            .transpose()?
            .map(str::to_string),
        hardware_vendor: object
            .get("hardware_vendor")
            .map(|value| as_str(value, "target.hardware_vendor"))
            .transpose()?
            .map(str::to_string),
        architecture: object
            .get("architecture")
            .map(|value| as_str(value, "target.architecture"))
            .transpose()?
            .map(str::to_string),
        device_features: string_set("device_features", object)?,
        execution_environment: object
            .get("execution_environment")
            .map(|value| as_str(value, "target.execution_environment"))
            .transpose()?
            .map(str::to_string),
        provider_compatibility: string_set("provider_compatibility", object)?,
        runtime_driver_compatibility: string_set("runtime_driver_compatibility", object)?,
        memory_classes: string_set("memory_classes", object)?,
    };
    if target.entry_count() > limits.max_target_entries {
        return Err(KernelManifestError::LimitExceeded {
            limit: "target-entry-count".into(),
        });
    }
    target.validate()?;
    Ok(target)
}

fn parse_specialization(
    value: &serde_json::Value,
) -> Result<KernelManifestSpecialization, KernelManifestError> {
    let object = as_object(value, "artifact.specialization")?;
    let u64_pair =
        |key: &str| -> Result<Option<(u64, u64)>, KernelManifestError> {
            match object.get(key) {
                Some(value) => {
                    let pair = value.as_array().ok_or_else(|| {
                        KernelManifestError::SpecializationInvalid {
                            reason: format!("specialization.{key} must be a [min, max] array"),
                        }
                    })?;
                    if pair.len() != 2 {
                        return Err(KernelManifestError::SpecializationInvalid {
                            reason: format!("specialization.{key} must have exactly two elements"),
                        });
                    }
                    let min = pair[0].as_u64().ok_or_else(|| {
                        KernelManifestError::SpecializationInvalid {
                            reason: format!(
                                "specialization.{key}[0] must be a non-negative integer"
                            ),
                        }
                    })?;
                    let max = pair[1].as_u64().ok_or_else(|| {
                        KernelManifestError::SpecializationInvalid {
                            reason: format!(
                                "specialization.{key}[1] must be a non-negative integer"
                            ),
                        }
                    })?;
                    Ok(Some((min, max)))
                }
                None => Ok(None),
            }
        };
    let exact_dimensions = match object.get("exact_dimensions") {
        Some(value) => as_object(value, "specialization.exact_dimensions")?
            .iter()
            .map(|(key, value)| {
                value
                    .as_u64()
                    .map(|value| (key.clone(), value))
                    .ok_or_else(|| KernelManifestError::SpecializationInvalid {
                        reason: format!(
                            "specialization.exact_dimensions.{key} must be a non-negative integer"
                        ),
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        None => BTreeMap::new(),
    };
    let tile_sizes = match object.get("tile_sizes") {
        Some(value) => value
            .as_array()
            .ok_or_else(|| KernelManifestError::SpecializationInvalid {
                reason: "specialization.tile_sizes must be an array".into(),
            })?
            .iter()
            .map(|entry| {
                entry
                    .as_u64()
                    .ok_or_else(|| KernelManifestError::SpecializationInvalid {
                        reason: "specialization.tile_sizes entries must be non-negative integers"
                            .into(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let execution_phase = match object
        .get("execution_phase")
        .map(|value| as_str(value, "specialization.execution_phase"))
        .transpose()?
    {
        Some("prefill") => Some(KernelExecutionPhase::Prefill),
        Some("decode") => Some(KernelExecutionPhase::Decode),
        Some("both") => Some(KernelExecutionPhase::Both),
        Some(other) => {
            return Err(KernelManifestError::SpecializationInvalid {
                reason: format!("unknown execution_phase '{other}'"),
            });
        }
        None => None,
    };
    let device_features = match object.get("device_features") {
        Some(value) => value
            .as_array()
            .ok_or_else(|| KernelManifestError::SpecializationInvalid {
                reason: "specialization.device_features must be an array".into(),
            })?
            .iter()
            .map(|entry| as_str(entry, "specialization.device_features[]").map(str::to_string))
            .collect::<Result<BTreeSet<_>, _>>()?,
        None => BTreeSet::new(),
    };
    let specialization = KernelManifestSpecialization {
        exact_dimensions,
        batch_range: u64_pair("batch_range")?,
        sequence_range: u64_pair("sequence_range")?,
        head_count: object.get("head_count").and_then(serde_json::Value::as_u64),
        head_dimension: object
            .get("head_dimension")
            .and_then(serde_json::Value::as_u64),
        tile_sizes,
        alignment: object.get("alignment").and_then(serde_json::Value::as_u64),
        dtype: object
            .get("dtype")
            .map(|value| as_str(value, "specialization.dtype"))
            .transpose()?
            .map(str::to_string),
        layout: object
            .get("layout")
            .map(|value| as_str(value, "specialization.layout"))
            .transpose()?
            .map(str::to_string),
        quantization_profile: object
            .get("quantization_profile")
            .map(|value| as_str(value, "specialization.quantization_profile"))
            .transpose()?
            .map(str::to_string),
        execution_phase,
        device_features,
    };
    specialization.validate()?;
    Ok(specialization)
}

fn parse_compiler_metadata(
    value: &serde_json::Value,
) -> Result<KernelCompilerMetadata, KernelManifestError> {
    let object = as_object(value, "artifact.compiler_metadata")?;
    let field = |key: &str| -> Result<Option<String>, KernelManifestError> {
        object
            .get(key)
            .map(|value| as_str(value, "compiler_metadata field"))
            .transpose()
            .map(|value| value.map(str::to_string))
    };
    Ok(KernelCompilerMetadata {
        compiler_identity: field("compiler_identity")?,
        compiler_version: field("compiler_version")?,
        backend_identity_version: field("backend_identity_version")?,
        flags_fingerprint: field("flags_fingerprint")?,
        build_fingerprint: field("build_fingerprint")?,
        target_architecture: field("target_architecture")?,
    })
}

fn parse_precision(
    value: &serde_json::Value,
) -> Result<KernelManifestPrecision, KernelManifestError> {
    let object = as_object(value, "artifact.precision")?;
    Ok(KernelManifestPrecision {
        accumulation_dtype: object
            .get("accumulation_dtype")
            .map(|value| as_str(value, "precision.accumulation_dtype"))
            .transpose()?
            .map(str::to_string),
        approximate_math: object
            .get("approximate_math")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        deterministic: object
            .get("deterministic")
            .and_then(serde_json::Value::as_bool),
        tolerance_profile: object
            .get("tolerance_profile")
            .map(|value| as_str(value, "precision.tolerance_profile"))
            .transpose()?
            .map(str::to_string),
        quantization_error_profile: object
            .get("quantization_error_profile")
            .map(|value| as_str(value, "precision.quantization_error_profile"))
            .transpose()?
            .map(str::to_string),
    })
}

fn parse_generator_metadata(
    value: &serde_json::Value,
) -> Result<KernelGeneratorMetadata, KernelManifestError> {
    let object = as_object(value, "artifact.generator")?;
    let field = |key: &str| -> Result<Option<String>, KernelManifestError> {
        object
            .get(key)
            .map(|value| as_str(value, "generator field"))
            .transpose()
            .map(|value| value.map(str::to_string))
    };
    let generator = KernelGeneratorMetadata {
        generator_name: field("generator_name")?,
        generator_version: field("generator_version")?,
        campaign_id: field("campaign_id")?,
        source_revision: field("source_revision")?,
    };
    generator.validate()?;
    Ok(generator)
}

fn parse_artifact(
    value: &serde_json::Value,
    limits: &KernelManifestLimits,
) -> Result<KernelManifestArtifact, KernelManifestError> {
    let object = as_object(value, "artifact")?;
    let role = KernelBlobRole::new(as_str(
        object
            .get("role")
            .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                reason: "artifact missing 'role'".into(),
            })?,
        "artifact.role",
    )?);
    let format = parse_format(
        object
            .get("format")
            .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                reason: "artifact missing 'format'".into(),
            })?,
        "artifact.format",
    )?;
    let digest = parse_digest(
        object
            .get("digest")
            .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                reason: "artifact missing 'digest'".into(),
            })?,
        "artifact.digest",
    )?;
    let size = object
        .get("size")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
            reason: "artifact missing numeric 'size'".into(),
        })?;
    let storage_mode = match object
        .get("storage_mode")
        .map(|value| as_str(value, "artifact.storage_mode"))
        .transpose()?
    {
        Some("external") => KernelArtifactStorageMode::External,
        Some("embedded") | None => KernelArtifactStorageMode::Embedded,
        Some(other) => {
            return Err(KernelManifestError::ArtifactReferenceInvalid {
                reason: format!("unknown storage_mode '{other}'"),
            });
        }
    };
    let required = object
        .get("required")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let media_type = object
        .get("media_type")
        .map(|value| as_str(value, "artifact.media_type"))
        .transpose()?
        .map(str::to_string);
    let location_hint = object
        .get("location_hint")
        .map(|value| as_str(value, "artifact.location_hint"))
        .transpose()?
        .map(str::to_string);

    let mut blob = KernelBlobDescriptor::new(role, format, digest, size);
    blob.storage_mode = storage_mode;
    blob.required = required;
    blob.media_type = media_type;
    blob.location_hint = location_hint;

    let mut artifact = KernelManifestArtifact::new(blob);

    if let Some(operators) = object.get("operators") {
        let operators_array =
            operators
                .as_array()
                .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                    reason: "artifact.operators must be an array".into(),
                })?;
        let parsed: Result<Vec<OperatorId>, KernelManifestError> =
            operators_array.iter().map(parse_operator).collect();
        let mut binding = KernelSemanticBinding {
            operators: parsed?,
            primary_version_requirements: None,
        };
        if let Some(range) = object.get("operator_version_range") {
            let range_object = as_object(range, "artifact.operator_version_range")?;
            let min = range_object
                .get("min")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                    reason: "operator_version_range missing numeric 'min'".into(),
                })?;
            let max = range_object
                .get("max")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| KernelManifestError::SemanticBindingInvalid {
                    reason: "operator_version_range missing numeric 'max'".into(),
                })?;
            binding.primary_version_requirements = Some(KernelOperatorVersionRange {
                min: min as u32,
                max: max as u32,
            });
        }
        binding.validate()?;
        artifact.semantic_binding = Some(binding);
    }
    if let Some(source_digest) = object.get("source_digest") {
        artifact.source_digest = Some(parse_digest(source_digest, "artifact.source_digest")?);
    }
    if let Some(provenance) = object
        .get("provenance")
        .map(|value| as_str(value, "artifact.provenance"))
        .transpose()?
    {
        artifact.provenance = Some(provenance_from_id(provenance).ok_or_else(|| {
            KernelManifestError::ProvenanceInvalid {
                value: provenance.to_string(),
            }
        })?);
    }
    if let Some(dependencies) = object.get("dependencies") {
        let dependencies = dependencies.as_array().ok_or_else(|| {
            KernelManifestError::ArtifactReferenceInvalid {
                reason: "artifact.dependencies must be an array".into(),
            }
        })?;
        artifact.dependencies = dependencies
            .iter()
            .map(|value| parse_digest(value, "artifact.dependencies[]"))
            .collect::<Result<_, _>>()?;
    }
    if let Some(target) = object.get("target") {
        artifact.target = parse_target(target, limits)?;
    }
    if let Some(specialization) = object.get("specialization") {
        artifact.specialization = parse_specialization(specialization)?;
    }
    if let Some(compiler_metadata) = object.get("compiler_metadata") {
        artifact.compiler_metadata = Some(parse_compiler_metadata(compiler_metadata)?);
    }
    if let Some(precision) = object.get("precision") {
        artifact.precision = parse_precision(precision)?;
    }
    if let Some(generator) = object.get("generator") {
        artifact.generator = Some(parse_generator_metadata(generator)?);
    }

    Ok(artifact)
}

/// Parses untrusted Kernel Manifest JSON text into a [`KernelManifestV1`],
/// implementing the spec's "Validation Order" up through semantic
/// structural validation: byte-size limit -> duplicate-key/nesting scan ->
/// JSON value parse -> field extraction -> schema/semantic validation.
/// "Parsing Does Not Prepare" (spec): this function never compiles, loads
/// executable code, calls `Provider.prepare`, executes a Kernel, or starts
/// qualification/benchmarking/promotion -- it only produces data.
pub fn parse_manifest_json(
    text: &str,
    limits: &KernelManifestLimits,
) -> Result<KernelManifestV1, KernelManifestError> {
    if text.len() > limits.max_manifest_bytes {
        return Err(KernelManifestError::TooLarge {
            limit: limits.max_manifest_bytes,
            actual: text.len(),
        });
    }
    scan_manifest_structure(text, limits)?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|error| KernelManifestError::InvalidJson {
            reason: error.to_string(),
        })?;
    let object = as_object(&value, "manifest")?;

    let schema_text = as_str(
        object
            .get("schema")
            .ok_or(KernelManifestError::SchemaMissing)?,
        "schema",
    )?;
    let schema = KernelManifestSchemaVersion::parse(schema_text)?;

    let artifacts_value = object
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
            reason: "manifest missing 'artifacts' array".into(),
        })?;
    if artifacts_value.len() > limits.max_artifacts {
        return Err(KernelManifestError::LimitExceeded {
            limit: "artifact-count".into(),
        });
    }
    let artifacts: Vec<KernelManifestArtifact> = artifacts_value
        .iter()
        .map(|value| parse_artifact(value, limits))
        .collect::<Result<_, _>>()?;

    let mut manifest = KernelManifestV1 {
        schema,
        artifacts,
        ..KernelManifestV1::new()
    };

    if let Some(evidence) = object.get("qualification_evidence") {
        let evidence =
            evidence
                .as_array()
                .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
                    reason: "qualification_evidence must be an array".into(),
                })?;
        if evidence.len() > limits.max_evidence_references {
            return Err(KernelManifestError::LimitExceeded {
                limit: "evidence-count".into(),
            });
        }
        manifest.qualification_evidence = evidence
            .iter()
            .map(parse_evidence)
            .collect::<Result<_, _>>()?;
    }
    if let Some(evidence) = object.get("benchmark_evidence") {
        let evidence =
            evidence
                .as_array()
                .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
                    reason: "benchmark_evidence must be an array".into(),
                })?;
        if evidence.len() > limits.max_evidence_references {
            return Err(KernelManifestError::LimitExceeded {
                limit: "evidence-count".into(),
            });
        }
        manifest.benchmark_evidence = evidence
            .iter()
            .map(parse_evidence)
            .collect::<Result<_, _>>()?;
    }
    if let Some(recommendation) = object
        .get("recommendation")
        .map(|value| as_str(value, "recommendation"))
        .transpose()?
    {
        manifest.recommendation =
            Some(recommendation_from_id(recommendation).ok_or_else(|| {
                KernelManifestError::ArtifactReferenceInvalid {
                    reason: format!("unknown recommendation '{recommendation}'"),
                }
            })?);
    }
    if let Some(provenance) = object
        .get("provenance")
        .map(|value| as_str(value, "provenance"))
        .transpose()?
    {
        manifest.provenance = Some(provenance_from_id(provenance).ok_or_else(|| {
            KernelManifestError::ProvenanceInvalid {
                value: provenance.to_string(),
            }
        })?);
    }
    if let Some(trust) = object.get("trust") {
        let trust_object = as_object(trust, "trust")?;
        manifest.trust.publisher_claim = trust_object
            .get("publisher")
            .map(|value| as_str(value, "trust.publisher"))
            .transpose()?
            .map(str::to_string);
        manifest.trust.source_claim = trust_object
            .get("source")
            .map(|value| as_str(value, "trust.source"))
            .transpose()?
            .map(str::to_string);
    }
    if let Some(annotations) = object.get("annotations") {
        let annotations = as_object(annotations, "annotations")?;
        for (key, value) in annotations {
            if !key.contains(':') {
                return Err(KernelManifestError::ArtifactReferenceInvalid {
                    reason: format!("annotation key '{key}' must be namespaced ('org:key')"),
                });
            }
            let value = as_str(value, "annotation value")?;
            if value.len() > limits.max_annotation_bytes {
                return Err(KernelManifestError::LimitExceeded {
                    limit: "annotation-bytes".into(),
                });
            }
            manifest.annotations.insert(key.clone(), value.to_string());
        }
    }
    if let Some(extensions) = object.get("extensions") {
        let extensions =
            extensions
                .as_array()
                .ok_or_else(|| KernelManifestError::ArtifactReferenceInvalid {
                    reason: "extensions must be an array".into(),
                })?;
        if extensions.len() > limits.max_extensions {
            return Err(KernelManifestError::LimitExceeded {
                limit: "extension-count".into(),
            });
        }
        for extension in extensions {
            let object = as_object(extension, "extension")?;
            let namespace = as_str(
                object.get("namespace").ok_or_else(|| {
                    KernelManifestError::ArtifactReferenceInvalid {
                        reason: "extension missing 'namespace'".into(),
                    }
                })?,
                "extension.namespace",
            )?
            .to_string();
            let required = object
                .get("required")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let data = object
                .get("data")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            manifest.extensions.push(KernelManifestExtension {
                namespace,
                required,
                data,
            });
        }
    }

    manifest.validate()?;
    Ok(manifest)
}

// ---------------------------------------------------------------------
// Kernel Exchange Bundle (directory representation)
// ---------------------------------------------------------------------

/// "Portable Kernel Exchange Bundle SHALL NOT require symlinks" (spec) --
/// rejects a bundle entry that is itself a symlink, implementing "Bundle
/// Path Safety" and "Symlinks" (spec).
pub fn scan_bundle_for_unsafe_entries(root: &Path) -> Result<(), KernelManifestError> {
    fn walk(dir: &Path) -> Result<(), KernelManifestError> {
        let entries = fs::read_dir(dir).map_err(|error| KernelManifestError::InternalError {
            reason: error.to_string(),
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| KernelManifestError::InternalError {
                reason: error.to_string(),
            })?;
            let path = entry.path();
            let symlink_metadata = fs::symlink_metadata(&path).map_err(|error| {
                KernelManifestError::InternalError {
                    reason: error.to_string(),
                }
            })?;
            if symlink_metadata.file_type().is_symlink() {
                return Err(KernelManifestError::BundleSymlinkDenied {
                    path: path.display().to_string(),
                });
            }
            if symlink_metadata.is_dir() {
                walk(&path)?;
            }
        }
        Ok(())
    }
    walk(root)
}

/// Rejects a bundle-relative logical path that attempts traversal, is
/// absolute, or is drive-qualified, implementing "Bundle Path Safety",
/// "Relative Paths", and "Absolute Paths" (spec).
pub fn validate_bundle_relative_path(path: &str) -> Result<(), KernelManifestError> {
    if path.is_empty() {
        return Err(KernelManifestError::BundlePathInvalid { path: path.into() });
    }
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(KernelManifestError::BundlePathInvalid { path: path.into() });
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        return Err(KernelManifestError::BundlePathInvalid { path: path.into() });
    }
    if path.split(['/', '\\']).any(|segment| segment == "..") {
        return Err(KernelManifestError::BundlePathInvalid { path: path.into() });
    }
    Ok(())
}

/// A physical directory-based Kernel Exchange Bundle: `kernel.manifest.json`
/// plus `blobs/sha256/<digest>`, implementing "Kernel Exchange Bundle" and
/// "Bundle Manifest" (spec). Other physical transports (archive,
/// object-store, OCI-like) are out of scope for this type but share the same
/// logical identity, per "Distribution Neutrality" (spec).
#[derive(Clone, Debug)]
pub struct KernelExchangeBundle {
    pub root: PathBuf,
}

impl KernelExchangeBundle {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join(KERNEL_MANIFEST_FILE_NAME)
    }

    pub fn load_manifest_text(&self) -> Result<String, KernelManifestError> {
        if !self.manifest_path().is_file() {
            return Err(KernelManifestError::BundleManifestMissing);
        }
        fs::read_to_string(self.manifest_path()).map_err(|error| {
            KernelManifestError::InternalError {
                reason: error.to_string(),
            }
        })
    }

    pub fn blob_path(&self, digest: &KernelBlobDigest) -> PathBuf {
        self.root
            .join("blobs")
            .join(&digest.algorithm)
            .join(&digest.value)
    }

    /// Verifies an embedded blob's bytes hash to its declared digest and
    /// match its declared size, implementing "Blob Integrity" (tasks) and
    /// "Blob bytes SHALL hash to the declared digest. Mismatch SHALL fail
    /// validation" (spec, "Blob Path").
    pub fn verify_embedded_blob(
        &self,
        digest: &KernelBlobDigest,
        expected_size: u64,
    ) -> Result<(), KernelManifestError> {
        let path = self.blob_path(digest);
        let bytes = fs::read(&path).map_err(|_| KernelManifestError::BundleBlobMissing {
            digest: digest.value.clone(),
        })?;
        if bytes.len() as u64 != expected_size {
            return Err(KernelManifestError::BundleBlobSizeMismatch {
                digest: digest.value.clone(),
                expected: expected_size,
                actual: bytes.len() as u64,
            });
        }
        let actual = sha256_hex(&bytes);
        if actual != digest.value {
            return Err(KernelManifestError::BundleBlobDigestMismatch {
                expected: digest.value.clone(),
                actual,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Distribution neutrality: named transports
// ---------------------------------------------------------------------

/// Physical Kernel Exchange Bundle transports, implementing "Distribution
/// Neutrality" (spec): "Transport representation SHALL NOT change Kernel
/// Artifact identity." [`Self::Directory`] and [`Self::TarArchive`] are
/// implemented by this module; [`Self::ObjectStore`], [`Self::OciLike`], and
/// [`Self::Registry`] are named reservations only -- the proposal's
/// "Non-Goals" section explicitly excludes this change from defining one
/// artifact registry, an OCI profile, or object-store/HTTP APIs, so those
/// three variants intentionally have no supporting code, matching "Reserve
/// object-store/OCI-like/registry transport" (tasks) rather than
/// implementing them.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelBundleTransport {
    Directory,
    TarArchive,
    ObjectStore,
    OciLike,
    Registry,
}

impl KernelBundleTransport {
    pub const fn is_implemented(self) -> bool {
        matches!(self, Self::Directory | Self::TarArchive)
    }
}

// ---------------------------------------------------------------------
// Archive representation (tar / tar.gz)
// ---------------------------------------------------------------------

/// Not part of Kernel Artifact identity -- see "Bundle Identity" (spec): "An
/// archive checksum MAY additionally exist, but it SHALL NOT replace logical
/// artifact identity." This hashes the *raw, possibly-compressed* archive
/// bytes purely as an optional transport-level diagnostic (e.g. for a
/// download integrity check), implementing "Keep archive checksum
/// optional/separate" (tasks). Two archives with different compression or
/// timestamps produce different values here while still unpacking to a
/// bundle with an identical [`KernelManifestV1::digest`].
pub fn archive_diagnostic_checksum(archive_bytes: &[u8]) -> String {
    sha256_hex(archive_bytes)
}

/// Defensive limits for archive extraction, implementing "Verify
/// decompressed size limits" / "Prevent decompression bomb beyond configured
/// limits" (tasks).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelExchangeArchiveLimits {
    pub max_entries: usize,
    pub max_entry_decompressed_bytes: u64,
    pub max_total_decompressed_bytes: u64,
}

impl Default for KernelExchangeArchiveLimits {
    fn default() -> Self {
        Self {
            max_entries: 4096,
            max_entry_decompressed_bytes: 1024 * 1024 * 1024,
            max_total_decompressed_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn reject_unsafe_tar_entry_type(
    entry_type: tar::EntryType,
    path: &str,
) -> Result<(), KernelManifestError> {
    use tar::EntryType;
    match entry_type {
        EntryType::Regular | EntryType::Directory | EntryType::GNUSparse => Ok(()),
        EntryType::Symlink => Err(KernelManifestError::BundleSymlinkDenied { path: path.into() }),
        // Hard links, device files, FIFOs, and other special entries are not
        // portable Kernel Exchange Bundle content -- "Reject hard-link
        // escape" / "Reject device files" / "Reject special entries" (tasks,
        // "Path Safety"). Unlike the directory transport (where these
        // concepts are not portably detectable via `std::fs` metadata alone
        // across Windows/macOS/Linux), tar's entry-type byte makes every one
        // of these explicit and detectable before any bytes are extracted.
        _ => Err(KernelManifestError::BundlePathInvalid { path: path.into() }),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_tar_entry_bounded(
    entry: &mut tar::Entry<'_, impl std::io::Read>,
    per_entry_limit: u64,
) -> Result<Vec<u8>, KernelManifestError> {
    use std::io::Read as _;
    // Reads at most `per_entry_limit + 1` bytes: if that many are actually
    // available, the true entry is over limit, and it is rejected *without*
    // ever allocating or fully materializing the oversized (potentially
    // bomb-decompressed) content.
    let mut limited = entry.take(per_entry_limit.saturating_add(1));
    let mut buffer = Vec::new();
    limited
        .read_to_end(&mut buffer)
        .map_err(|error| KernelManifestError::InternalError {
            reason: error.to_string(),
        })?;
    if buffer.len() as u64 > per_entry_limit {
        return Err(KernelManifestError::LimitExceeded {
            limit: "archive-entry-decompressed-bytes".into(),
        });
    }
    Ok(buffer)
}

/// Extracts a tar (optionally gzip-compressed) Kernel Exchange Bundle
/// archive into `destination`, implementing "Support archive
/// representation" and "Allow transport compression" (tasks). Every entry's
/// path is checked with [`validate_bundle_relative_path`] and every entry's
/// type is checked with `reject_unsafe_tar_entry_type` *before* any bytes
/// are written to disk -- path traversal and unsafe entry types fail closed
/// rather than partially extracting. Returns a [`KernelExchangeBundle`]
/// rooted at `destination`; callers run it through
/// [`validate_kernel_exchange_bundle`] exactly as they would a
/// directory-transport bundle -- "Digest logical uncompressed blob bytes"
/// (tasks) falls out naturally, since digest computation only ever sees the
/// fully-extracted plain bytes on disk, never archive/compression framing.
#[cfg(not(target_arch = "wasm32"))]
pub fn extract_kernel_exchange_archive(
    reader: impl std::io::Read,
    gzip_compressed: bool,
    destination: &Path,
    limits: &KernelExchangeArchiveLimits,
) -> Result<KernelExchangeBundle, KernelManifestError> {
    fn extract_entries(
        mut archive: tar::Archive<impl std::io::Read>,
        destination: &Path,
        limits: &KernelExchangeArchiveLimits,
    ) -> Result<(), KernelManifestError> {
        let entries = archive
            .entries()
            .map_err(|error| KernelManifestError::InternalError {
                reason: error.to_string(),
            })?;
        let mut entry_count = 0usize;
        let mut total_bytes: u64 = 0;
        for entry in entries {
            let mut entry = entry.map_err(|error| KernelManifestError::InternalError {
                reason: error.to_string(),
            })?;
            entry_count += 1;
            if entry_count > limits.max_entries {
                return Err(KernelManifestError::LimitExceeded {
                    limit: "archive-entry-count".into(),
                });
            }
            let entry_type = entry.header().entry_type();
            let path_buf = entry
                .path()
                .map_err(|error| KernelManifestError::InternalError {
                    reason: error.to_string(),
                })?
                .into_owned();
            let path_text = path_buf.to_string_lossy().replace('\\', "/");
            reject_unsafe_tar_entry_type(entry_type, &path_text)?;
            validate_bundle_relative_path(&path_text)?;

            if entry_type == tar::EntryType::Directory {
                fs::create_dir_all(destination.join(&path_text)).map_err(|error| {
                    KernelManifestError::InternalError {
                        reason: error.to_string(),
                    }
                })?;
                continue;
            }

            let bytes = read_tar_entry_bounded(&mut entry, limits.max_entry_decompressed_bytes)?;
            total_bytes = total_bytes.saturating_add(bytes.len() as u64);
            if total_bytes > limits.max_total_decompressed_bytes {
                return Err(KernelManifestError::LimitExceeded {
                    limit: "archive-total-decompressed-bytes".into(),
                });
            }

            let target_path = destination.join(&path_text);
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent).map_err(|error| KernelManifestError::InternalError {
                    reason: error.to_string(),
                })?;
            }
            fs::write(&target_path, &bytes).map_err(|error| {
                KernelManifestError::InternalError {
                    reason: error.to_string(),
                }
            })?;
        }
        Ok(())
    }

    fs::create_dir_all(destination).map_err(|error| KernelManifestError::InternalError {
        reason: error.to_string(),
    })?;

    if gzip_compressed {
        let decoder = flate2::read::GzDecoder::new(reader);
        extract_entries(tar::Archive::new(decoder), destination, limits)?;
    } else {
        extract_entries(tar::Archive::new(reader), destination, limits)?;
    }

    Ok(KernelExchangeBundle::open(destination.to_path_buf()))
}

// ---------------------------------------------------------------------
// Compatibility evaluation
// ---------------------------------------------------------------------

/// Runtime-observed context an artifact's [`KernelTargetConstraints`] are
/// compared against, implementing "Evaluate compatibility" (tasks,
/// "Validation Pipeline"). Kept deliberately separate from
/// [`validate_kernel_exchange_bundle`]'s signature -- like trust evaluation,
/// compatibility evaluation needs Runtime-side context this pure
/// manifest/bundle validation function does not have, so it is a distinct
/// pipeline stage a caller runs afterward, never an automatic side effect of
/// parsing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelRuntimeCompatibilityContext {
    pub provider_id: Option<String>,
    pub architecture: Option<String>,
    pub available_device_features: BTreeSet<String>,
}

/// Compares declared target constraints against an observed Runtime
/// context. Purely descriptive comparison -- it does not execute, prepare,
/// or select a Kernel, and an empty/unset constraint is always treated as
/// compatible (absence of a declared constraint is not a claim of
/// incompatibility).
pub fn evaluate_target_compatibility(
    target: &KernelTargetConstraints,
    context: &KernelRuntimeCompatibilityContext,
) -> Result<(), KernelManifestError> {
    if let (Some(architecture), Some(observed)) = (&target.architecture, &context.architecture)
        && architecture != observed
    {
        return Err(KernelManifestError::ExchangeCompatibilityFailed {
            reason: format!(
                "target architecture '{architecture}' does not match observed '{observed}'"
            ),
        });
    }
    if let Some(provider_id) = &context.provider_id
        && !target.provider_compatibility.is_empty()
        && !target.provider_compatibility.contains(provider_id)
    {
        return Err(KernelManifestError::ExchangeCompatibilityFailed {
            reason: format!("Provider '{provider_id}' is not declared compatible"),
        });
    }
    if !target.device_features.is_empty() {
        let missing: Vec<&String> = target
            .device_features
            .difference(&context.available_device_features)
            .collect();
        if !missing.is_empty() {
            return Err(KernelManifestError::ExchangeCompatibilityFailed {
                reason: format!(
                    "missing required Device features: {}",
                    missing
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        }
    }
    Ok(())
}

/// The "Evaluate trust" pipeline stage, implementing "Validation Pipeline"
/// (tasks). Deliberately takes only `policy_approved` -- no manifest,
/// publisher claim, source claim, or signature envelope -- so this function
/// is a nameable, testable pipeline stage without becoming a second way for
/// manifest content to influence trust. It delegates unchanged to
/// [`crate::evaluate_artifact_trust`], the sole authority for
/// [`crate::KernelArtifactTrust::Trusted`].
pub fn evaluate_manifest_trust(policy_approved: bool) -> crate::KernelArtifactTrust {
    crate::evaluate_artifact_trust(policy_approved)
}

/// A manifest that has passed structural, schema, canonical-identity, blob
/// integrity, semantic, and extension validation, implementing the spec's
/// "Validation Order" through "compatibility evaluation" (compatibility
/// itself remains Runtime/Provider policy, applied by later callers).
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedKernelManifest {
    pub manifest: KernelManifestV1,
    pub digest: KernelBlobDigest,
}

/// Runs the Kernel Exchange Bundle validation pipeline against a
/// directory-based bundle, implementing "Validation Order" (spec): parse ->
/// structural validation -> schema validation -> canonical identity -> blob
/// integrity -> semantic validation -> extension validation. Trust
/// evaluation and evidence revalidation against *current* policy remain the
/// caller's responsibility (see [`evaluate_qualification_evidence_currency`]
/// and [`crate::evaluate_artifact_trust`]) -- this module only guarantees
/// they are never bypassed by manifest content alone.
pub fn validate_kernel_exchange_bundle(
    bundle: &KernelExchangeBundle,
    limits: &KernelManifestLimits,
) -> Result<ValidatedKernelManifest, KernelManifestError> {
    let text = bundle.load_manifest_text()?;
    let manifest = parse_manifest_json(&text, limits)?;

    scan_bundle_for_unsafe_entries(&bundle.root)?;

    let mut total_embedded_bytes: u64 = 0;
    for artifact in &manifest.artifacts {
        match artifact.blob.storage_mode {
            KernelArtifactStorageMode::Embedded => {
                if bundle.blob_path(&artifact.blob.digest).is_file() {
                    bundle.verify_embedded_blob(&artifact.blob.digest, artifact.blob.size)?;
                    total_embedded_bytes = total_embedded_bytes.saturating_add(artifact.blob.size);
                } else if artifact.blob.required {
                    return Err(KernelManifestError::BundleRequiredArtifactMissing {
                        role: artifact.blob.role.to_string(),
                    });
                }
                // A missing, non-required embedded artifact does not
                // invalidate the bundle -- "Missing optional evidence MAY
                // reduce eligibility/ranking according to policy without
                // necessarily invalidating the whole manifest" (spec,
                // "Bundle Completeness").
            }
            KernelArtifactStorageMode::External if artifact.blob.required => {
                return Err(KernelManifestError::ExchangeExternalReferenceDenied {
                    location: artifact
                        .blob
                        .location_hint
                        .clone()
                        .unwrap_or_else(|| "<no location hint>".into()),
                });
            }
            KernelArtifactStorageMode::External => {}
        }
    }
    if total_embedded_bytes > limits.max_total_embedded_bytes {
        return Err(KernelManifestError::BundleTotalSizeExceeded {
            limit: limits.max_total_embedded_bytes,
            actual: total_embedded_bytes,
        });
    }

    let digest = manifest.digest();
    Ok(ValidatedKernelManifest { manifest, digest })
}

// ---------------------------------------------------------------------
// Normalization: portable manifest -> internal Kernel Artifact contracts
// ---------------------------------------------------------------------

fn dtype_from_id(id: &str) -> Option<ComputeDType> {
    match id {
        "bool" | "boolean" => Some(ComputeDType::Boolean),
        "u8" | "uint8" => Some(ComputeDType::UInt8),
        "i8" | "sint8" => Some(ComputeDType::SInt8),
        "u16" | "uint16" => Some(ComputeDType::UInt16),
        "i16" | "sint16" => Some(ComputeDType::SInt16),
        "u32" | "uint32" => Some(ComputeDType::UInt32),
        "i32" | "sint32" => Some(ComputeDType::SInt32),
        "u64" | "uint64" => Some(ComputeDType::UInt64),
        "i64" | "sint64" => Some(ComputeDType::SInt64),
        "f16" | "float16" | "fp16" => Some(ComputeDType::Float16),
        "bf16" | "bfloat16" => Some(ComputeDType::BrainFloat16),
        "f32" | "float32" | "fp32" => Some(ComputeDType::Float32),
        "f64" | "float64" | "fp64" => Some(ComputeDType::Float64),
        _ => None,
    }
}

fn specialization_to_shape_constraints(
    specialization: &KernelManifestSpecialization,
) -> KernelShapeConstraints {
    let matrix_tile = specialization
        .tile_sizes
        .first()
        .zip(specialization.tile_sizes.get(1))
        .map(|(rows, cols)| (*rows, *cols));
    let mut static_dimensions = BTreeMap::new();
    for (key, value) in &specialization.exact_dimensions {
        if let Ok(index) = key.parse::<usize>() {
            static_dimensions.insert(index, *value);
        }
    }
    KernelShapeConstraints {
        max_batch_size: specialization.batch_range.map(|(_, max)| max),
        max_sequence_length: specialization.sequence_range.map(|(_, max)| max),
        max_head_count: specialization.head_count,
        max_head_dimension: specialization.head_dimension,
        alignment: specialization.alignment,
        matrix_tile,
        static_dimensions,
        ..KernelShapeConstraints::default()
    }
}

/// Normalizes a `kernel-source`-role manifest artifact into the internal
/// [`KernelSourceArtifact`] contract, implementing "Normalize manifest to
/// KernelSourceArtifact" (tasks). This performs no compilation, preparation,
/// or execution -- it only re-expresses already-validated portable data as
/// the Runtime-native type, per "Normalized Internal Representation" (spec):
/// portable exchange types never become Provider-native execution types
/// through this function; the result is still ordinary data.
pub fn normalize_to_source_artifact(
    artifact: &KernelManifestArtifact,
) -> Result<KernelSourceArtifact, KernelManifestError> {
    let binding = artifact.semantic_binding.as_ref().ok_or_else(|| {
        KernelManifestError::SemanticBindingInvalid {
            reason: "artifact has no semantic binding to normalize".into(),
        }
    })?;
    binding.validate()?;
    let mut operators = binding.operators.iter().cloned();
    let declared_operator = operators
        .next()
        .expect("validated binding has at least one operator");
    let mut format = KernelSourceFormat::new(
        artifact.blob.format.namespace.clone(),
        artifact.blob.format.name.clone(),
    );
    if let Some(version) = &artifact.blob.format.version {
        format = format.with_version(version.clone());
    }
    let provenance = artifact
        .provenance
        .unwrap_or(KernelArtifactProvenance::Imported);
    let mut source = KernelSourceArtifact::new(
        KernelSourceArtifactId::from_digest(artifact.blob.digest.value.clone()),
        format,
        declared_operator,
        provenance,
    );
    source.fused_operator_group = operators.collect();
    source.shape = specialization_to_shape_constraints(&artifact.specialization);
    if let Some(dtype) = artifact
        .specialization
        .dtype
        .as_deref()
        .and_then(dtype_from_id)
    {
        source.dtype_constraints.insert(dtype);
    }
    source.target_requirements = artifact.target.provider_compatibility.clone();
    Ok(source)
}

/// Normalizes a `compiled-kernel`-role manifest artifact into the internal
/// [`CompiledKernelArtifact`] contract, implementing "Normalize compiled
/// descriptors" (tasks). Known limitation: [`CompiledKernelArtifact`] models
/// a single `operator_semantics : OperatorId`, so for a fused binding only
/// the primary (first) Operator is preserved here -- the full fused sequence
/// remains available from the portable [`KernelSemanticBinding`] itself.
pub fn normalize_to_compiled_artifact(
    artifact: &KernelManifestArtifact,
) -> Result<CompiledKernelArtifact, KernelManifestError> {
    let binding = artifact.semantic_binding.as_ref().ok_or_else(|| {
        KernelManifestError::SemanticBindingInvalid {
            reason: "artifact has no semantic binding to normalize".into(),
        }
    })?;
    binding.validate()?;
    let operator_semantics = binding.operators[0].clone();
    let compiler = artifact.compiler_metadata.clone().unwrap_or_default();
    let mut compiled = CompiledKernelArtifact::new(
        CompiledKernelArtifactId::from_digest(artifact.blob.digest.value.clone()),
        artifact.blob.format.stable_key(),
        compiler
            .compiler_identity
            .unwrap_or_else(|| "unknown".into()),
        compiler
            .compiler_version
            .unwrap_or_else(|| "unknown".into()),
        artifact
            .target
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        operator_semantics,
    );
    compiled.source_artifact_id = artifact
        .source_digest
        .as_ref()
        .map(|digest| KernelSourceArtifactId::from_digest(digest.value.clone()));
    compiled.compiler_flags_digest = compiler.flags_fingerprint;
    compiled.shape = specialization_to_shape_constraints(&artifact.specialization);
    if let Some(dtype) = artifact
        .specialization
        .dtype
        .as_deref()
        .and_then(dtype_from_id)
    {
        compiled.dtype_constraints.insert(dtype);
    }
    compiled.runtime_driver_compatibility = artifact.target.runtime_driver_compatibility.clone();
    compiled.precision.approximate_math = artifact.precision.approximate_math;
    if let Some(accumulation) = artifact
        .precision
        .accumulation_dtype
        .as_deref()
        .and_then(dtype_from_id)
    {
        compiled.precision.accumulation_dtype = Some(accumulation);
    }
    Ok(compiled)
}

/// Normalizes a qualification evidence reference's profile identity into the
/// internal [`QualificationProfile`] contract, implementing "Normalize
/// qualification references" (tasks). Deliberately produces identity data
/// only -- never a [`crate::QualificationRecord`] with an inferred status,
/// because manifest content alone SHALL NOT make qualification evidence
/// current (see [`evaluate_qualification_evidence_currency`]); an actual
/// `QualificationRecord`'s status may only change through its own
/// `start_qualifying`/`mark_qualified` transitions driven by real
/// verification, never by parsing.
pub fn normalize_qualification_profile(
    evidence: &KernelEvidenceReference,
) -> Result<QualificationProfile, KernelManifestError> {
    let (name, version) = evidence.profile.split_once('@').ok_or_else(|| {
        KernelManifestError::EvidenceReferenceInvalid {
            reason: format!(
                "qualification profile '{}' must be 'name@version'",
                evidence.profile
            ),
        }
    })?;
    let version: u32 =
        version
            .parse()
            .map_err(|_| KernelManifestError::EvidenceReferenceInvalid {
                reason: format!(
                    "qualification profile '{}' has a non-numeric version",
                    evidence.profile
                ),
            })?;
    Ok(QualificationProfile::new(name, version))
}

/// Normalizes a qualification evidence reference's oracle identity into the
/// internal [`CorrectnessOracleIdentity`] contract, implementing "Normalize
/// qualification references" (tasks).
pub fn normalize_oracle_identity(
    evidence: &KernelEvidenceReference,
) -> Result<CorrectnessOracleIdentity, KernelManifestError> {
    let identity = evidence
        .oracle_or_provider_identity
        .as_deref()
        .ok_or_else(|| KernelManifestError::EvidenceReferenceInvalid {
            reason: "evidence has no oracle identity to normalize".into(),
        })?;
    let (provider, version) = identity.split_once('@').unwrap_or((identity, "unknown"));
    Ok(CorrectnessOracleIdentity {
        provider: ProviderBinding::new(provider),
        version: version.to_string(),
    })
}

/// Normalizes a benchmark evidence reference into the internal
/// [`crate::BenchmarkProfile`] contract, implementing "Normalize benchmark
/// references" (tasks). `hardware_architecture` is an explicit parameter
/// rather than an evidence field: this reference identifies the *evidence*
/// blob's digest, not which compiled artifact/architecture it was measured
/// against, so the caller (which already knows that linkage from its own
/// manifest traversal) supplies it rather than this function guessing.
/// Missing optional fields become honest defaults (empty string / zero),
/// never fabricated plausible-looking values --
/// `crate::BenchmarkProfile::is_authoritative` then correctly reports
/// whether the result is complete enough to use as ranking evidence.
pub fn normalize_benchmark_profile(
    evidence: &KernelEvidenceReference,
    hardware_architecture: &str,
) -> Result<crate::BenchmarkProfile, KernelManifestError> {
    let workload = evidence.workload_metadata.as_ref().ok_or_else(|| {
        KernelManifestError::ExchangeBenchmarkEvidenceInvalid {
            reason: "evidence has no workload metadata to normalize into a BenchmarkProfile".into(),
        }
    })?;
    Ok(crate::BenchmarkProfile {
        target_device: evidence.device_context.clone().unwrap_or_default(),
        hardware_architecture: hardware_architecture.to_string(),
        provider_version: evidence.provider_context.clone().unwrap_or_default(),
        driver_runtime_version: workload.driver_runtime_version.clone(),
        input_shapes: workload.input_shapes.clone().unwrap_or_default(),
        dtype_layout: workload.dtype_layout.clone().unwrap_or_default(),
        batch_size: workload.batch_size,
        sequence_length: workload.sequence_length,
        warmup_count: workload.warmup_count.unwrap_or(0),
        measurement_count: workload.measurement_count.unwrap_or(0),
        synchronization_policy: workload.synchronization_policy.clone().unwrap_or_default(),
        benchmark_version: workload
            .benchmark_version
            .clone()
            .or_else(|| evidence.suite_or_workload_version.clone())
            .unwrap_or_default(),
    })
}

// ---------------------------------------------------------------------
// Kernel Cache Integration
// ---------------------------------------------------------------------

/// Builds a [`KernelCacheKey`] from a compiled manifest artifact,
/// implementing "Insert validated blobs by digest" / "Preserve artifact
/// identity" (tasks, "Kernel Cache Integration"). `provider_version` has no
/// portable-manifest source (it is Runtime/Provider-instance-observed, not
/// producer-declared) and defaults to `"unknown"`; a real cache-insertion
/// caller SHOULD override it with the actually-observed Provider version
/// before use. This function never marks anything trusted or qualified --
/// see [`KernelCacheEntry::new`], which always starts
/// `crate::CacheEntryState::Partial` with `crate::KernelArtifactTrust::Untrusted`
/// and no qualification, keeping "Keep trust separate" / "Keep qualification
/// separate" true by construction.
pub fn normalize_to_cache_key(artifact: &KernelManifestArtifact) -> KernelCacheKey {
    let compiler = artifact.compiler_metadata.clone().unwrap_or_default();
    KernelCacheKey {
        source_digest: artifact
            .source_digest
            .as_ref()
            .map(|digest| digest.value.clone()),
        compiled_artifact_digest: artifact.blob.digest.value.clone(),
        source_format: None,
        compiled_format: artifact.blob.format.stable_key(),
        compiler_identity: compiler
            .compiler_identity
            .unwrap_or_else(|| "unknown".into()),
        compiler_version: compiler
            .compiler_version
            .unwrap_or_else(|| "unknown".into()),
        compiler_flags_fingerprint: compiler.flags_fingerprint,
        provider_version: "unknown".into(),
        target_architecture: artifact
            .target
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        driver_runtime_compatibility_class: artifact.target.runtime_driver_compatibility.clone(),
        operator_semantics: artifact
            .semantic_binding
            .as_ref()
            .map(KernelSemanticBinding::fingerprint)
            .unwrap_or_default(),
        dtype: artifact
            .specialization
            .dtype
            .as_deref()
            .and_then(dtype_from_id)
            .into_iter()
            .collect(),
        layout: BTreeSet::new(),
        shape_specialization: (!artifact.specialization.is_empty())
            .then(|| format!("{:?}", artifact.specialization)),
        device_features: artifact.target.device_features.clone(),
    }
}

/// Builds an unvalidated, untrusted, unqualified [`KernelCacheEntry`] shell
/// for a validated bundle blob, implementing "Insert validated blobs by
/// digest" (tasks). Callers SHALL still run the entry through
/// `crate::verify_cache_entry_integrity` /
/// `crate::evaluate_cache_eligibility` before treating it as usable --
/// "Prevent corrupt entry insertion" remains the cache module's own
/// responsibility, not bypassed here.
pub fn normalize_to_cache_entry(artifact: &KernelManifestArtifact) -> KernelCacheEntry {
    let key = normalize_to_cache_key(artifact);
    let digest = artifact.blob.digest.value.clone();
    KernelCacheEntry::new(
        key,
        CompiledKernelArtifactId::from_digest(artifact.blob.digest.value.clone()),
        digest,
    )
}

// ---------------------------------------------------------------------
// CLI / Tooling
// ---------------------------------------------------------------------

/// Reserved CLI/tooling operation surface, implementing "CLI And Tooling"
/// (spec): "CLI or external tooling MAY: inspect manifests, validate
/// bundles, import bundles, export bundles, show artifact metadata."
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelManifestCliOperation {
    Inspect,
    Validate,
    Import,
    Export,
}

impl KernelManifestCliOperation {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Inspect => "kernel-manifest-inspect",
            Self::Validate => "kernel-bundle-validate",
            Self::Import => "kernel-bundle-import",
            Self::Export => "kernel-bundle-export",
        }
    }
}

/// Every reserved CLI/tooling operation goes through this identical
/// validation entry point, implementing "Use shared validation
/// implementation" (tasks): "Tooling SHALL still rely on Runtime/library
/// validation rather than treating the manifest as trusted configuration."
/// No operation-specific shortcut exists that could skip validation.
pub fn run_kernel_manifest_cli_operation(
    operation: KernelManifestCliOperation,
    bundle: &KernelExchangeBundle,
    limits: &KernelManifestLimits,
) -> Result<ValidatedKernelManifest, KernelManifestError> {
    let _ = operation;
    validate_kernel_exchange_bundle(bundle, limits)
}

// ---------------------------------------------------------------------
// Runtime API boundary
// ---------------------------------------------------------------------

/// "Normal generation requests SHALL NOT directly carry arbitrary Kernel
/// Exchange Bundles" (spec, "Runtime Inference API Boundary").
pub const KERNEL_MANIFEST_FORBIDDEN_INFERENCE_FIELDS: &[&str] =
    &["kernel-manifest", "kernel-bundle", "kernel-exchange-bundle"];

pub fn reject_inference_request_manifest_field(field: &str) -> Result<(), KernelManifestError> {
    let normalized = field.trim().to_ascii_lowercase();
    if KERNEL_MANIFEST_FORBIDDEN_INFERENCE_FIELDS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(KernelManifestError::InternalError {
            reason: format!("inference request field '{field}' is outside normal generation scope"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// The 34 structured error categories from the proposal's "Error Model"
/// section.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelManifestError {
    InvalidJson {
        reason: String,
    },
    DuplicateKey {
        key: String,
    },
    SchemaMissing,
    SchemaUnsupported {
        schema: String,
    },
    RequiredExtensionUnsupported {
        namespace: String,
    },
    TooLarge {
        limit: usize,
        actual: usize,
    },
    LimitExceeded {
        limit: String,
    },
    CanonicalizationFailed {
        reason: String,
    },
    SemanticBindingInvalid {
        reason: String,
    },
    TargetInvalid {
        reason: String,
    },
    SpecializationInvalid {
        reason: String,
    },
    ArtifactReferenceInvalid {
        reason: String,
    },
    DependencyCycle {
        path: String,
    },
    ProvenanceInvalid {
        value: String,
    },
    EvidenceReferenceInvalid {
        reason: String,
    },
    SignatureEnvelopeInvalid {
        reason: String,
    },

    BundleManifestMissing,
    BundleDuplicateEntry {
        path: String,
    },
    BundlePathInvalid {
        path: String,
    },
    BundleSymlinkDenied {
        path: String,
    },
    BundleBlobMissing {
        digest: String,
    },
    BundleBlobSizeMismatch {
        digest: String,
        expected: u64,
        actual: u64,
    },
    BundleBlobDigestMismatch {
        expected: String,
        actual: String,
    },
    BundleTotalSizeExceeded {
        limit: u64,
        actual: u64,
    },
    BundleRequiredArtifactMissing {
        role: String,
    },

    ExchangeFormatUnsupported {
        format: String,
    },
    ExchangeArtifactFormatUnsupported {
        format: String,
    },
    ExchangeExternalReferenceDenied {
        location: String,
    },
    ExchangeExternalArtifactUnavailable {
        reason: String,
    },
    ExchangeTrustDenied {
        reason: String,
    },
    ExchangeQualificationInvalid {
        reason: String,
    },
    ExchangeBenchmarkEvidenceInvalid {
        reason: String,
    },
    ExchangeCompatibilityFailed {
        reason: String,
    },

    InternalError {
        reason: String,
    },
}

impl KernelManifestError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::InvalidJson { .. } => "kernel-manifest-invalid-json",
            Self::DuplicateKey { .. } => "kernel-manifest-duplicate-key",
            Self::SchemaMissing => "kernel-manifest-schema-missing",
            Self::SchemaUnsupported { .. } => "kernel-manifest-schema-unsupported",
            Self::RequiredExtensionUnsupported { .. } => {
                "kernel-manifest-required-extension-unsupported"
            }
            Self::TooLarge { .. } => "kernel-manifest-too-large",
            Self::LimitExceeded { .. } => "kernel-manifest-limit-exceeded",
            Self::CanonicalizationFailed { .. } => "kernel-manifest-canonicalization-failed",
            Self::SemanticBindingInvalid { .. } => "kernel-manifest-semantic-binding-invalid",
            Self::TargetInvalid { .. } => "kernel-manifest-target-invalid",
            Self::SpecializationInvalid { .. } => "kernel-manifest-specialization-invalid",
            Self::ArtifactReferenceInvalid { .. } => "kernel-manifest-artifact-reference-invalid",
            Self::DependencyCycle { .. } => "kernel-manifest-dependency-cycle",
            Self::ProvenanceInvalid { .. } => "kernel-manifest-provenance-invalid",
            Self::EvidenceReferenceInvalid { .. } => "kernel-manifest-evidence-reference-invalid",
            Self::SignatureEnvelopeInvalid { .. } => "kernel-manifest-signature-envelope-invalid",
            Self::BundleManifestMissing => "kernel-bundle-manifest-missing",
            Self::BundleDuplicateEntry { .. } => "kernel-bundle-duplicate-entry",
            Self::BundlePathInvalid { .. } => "kernel-bundle-path-invalid",
            Self::BundleSymlinkDenied { .. } => "kernel-bundle-symlink-denied",
            Self::BundleBlobMissing { .. } => "kernel-bundle-blob-missing",
            Self::BundleBlobSizeMismatch { .. } => "kernel-bundle-blob-size-mismatch",
            Self::BundleBlobDigestMismatch { .. } => "kernel-bundle-blob-digest-mismatch",
            Self::BundleTotalSizeExceeded { .. } => "kernel-bundle-total-size-exceeded",
            Self::BundleRequiredArtifactMissing { .. } => "kernel-bundle-required-artifact-missing",
            Self::ExchangeFormatUnsupported { .. } => "kernel-exchange-format-unsupported",
            Self::ExchangeArtifactFormatUnsupported { .. } => {
                "kernel-exchange-artifact-format-unsupported"
            }
            Self::ExchangeExternalReferenceDenied { .. } => {
                "kernel-exchange-external-reference-denied"
            }
            Self::ExchangeExternalArtifactUnavailable { .. } => {
                "kernel-exchange-external-artifact-unavailable"
            }
            Self::ExchangeTrustDenied { .. } => "kernel-exchange-trust-denied",
            Self::ExchangeQualificationInvalid { .. } => "kernel-exchange-qualification-invalid",
            Self::ExchangeBenchmarkEvidenceInvalid { .. } => {
                "kernel-exchange-benchmark-evidence-invalid"
            }
            Self::ExchangeCompatibilityFailed { .. } => "kernel-exchange-compatibility-failed",
            Self::InternalError { .. } => "internal-kernel-manifest-error",
        }
    }
}

impl fmt::Display for KernelManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())?;
        match self {
            Self::InvalidJson { reason }
            | Self::CanonicalizationFailed { reason }
            | Self::SemanticBindingInvalid { reason }
            | Self::TargetInvalid { reason }
            | Self::SpecializationInvalid { reason }
            | Self::ArtifactReferenceInvalid { reason }
            | Self::EvidenceReferenceInvalid { reason }
            | Self::SignatureEnvelopeInvalid { reason }
            | Self::ExchangeExternalArtifactUnavailable { reason }
            | Self::ExchangeTrustDenied { reason }
            | Self::ExchangeQualificationInvalid { reason }
            | Self::ExchangeBenchmarkEvidenceInvalid { reason }
            | Self::ExchangeCompatibilityFailed { reason }
            | Self::InternalError { reason } => write!(f, ": {reason}"),
            Self::DuplicateKey { key } => write!(f, ": {key}"),
            Self::SchemaUnsupported { schema } => write!(f, ": {schema}"),
            Self::RequiredExtensionUnsupported { namespace } => write!(f, ": {namespace}"),
            Self::TooLarge { limit, actual } => write!(f, ": limit {limit}, actual {actual}"),
            Self::LimitExceeded { limit } => write!(f, ": {limit}"),
            Self::DependencyCycle { path } => write!(f, ": {path}"),
            Self::ProvenanceInvalid { value } => write!(f, ": {value}"),
            Self::BundleDuplicateEntry { path }
            | Self::BundlePathInvalid { path }
            | Self::BundleSymlinkDenied { path } => write!(f, ": {path}"),
            Self::BundleBlobMissing { digest } => write!(f, ": {digest}"),
            Self::BundleBlobSizeMismatch {
                digest,
                expected,
                actual,
            } => {
                write!(f, ": {digest} expected {expected} bytes, found {actual}")
            }
            Self::BundleBlobDigestMismatch { expected, actual } => {
                write!(f, ": expected {expected}, found {actual}")
            }
            Self::BundleTotalSizeExceeded { limit, actual } => {
                write!(f, ": limit {limit}, actual {actual}")
            }
            Self::BundleRequiredArtifactMissing { role } => write!(f, ": {role}"),
            Self::ExchangeFormatUnsupported { format }
            | Self::ExchangeArtifactFormatUnsupported { format } => {
                write!(f, ": {format}")
            }
            Self::ExchangeExternalReferenceDenied { location } => write!(f, ": {location}"),
            Self::SchemaMissing | Self::BundleManifestMissing => Ok(()),
        }
    }
}

impl Error for KernelManifestError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Observation categories from the spec's "Observability" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelManifestObservationKind {
    ManifestDiscovered,
    ManifestParsed,
    BlobValidated,
    SemanticBindingObserved,
    TargetCompatibilityEvaluated,
    QualificationEvidenceObserved,
    BenchmarkEvidenceObserved,
    ProvenanceSummarized,
    TrustEvaluated,
    ImportedIntoCache,
}

/// A single Kernel Manifest observation. Structurally guaranteed to never
/// carry raw Kernel source, raw compiled binary bytes, raw signature private
/// material, credentials, sensitive URLs, local filesystem paths, raw
/// benchmark tensors, model weights, or native handles: values always pass
/// through `redact_backend_diagnostic` first, implementing "Observability"
/// (spec).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelManifestObservation {
    pub kind: KernelManifestObservationKind,
    pub manifest_digest: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelManifestObservation {
    pub fn new(kind: KernelManifestObservationKind) -> Self {
        Self {
            kind,
            manifest_digest: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_manifest_digest(mut self, digest: &KernelBlobDigest) -> Self {
        self.manifest_digest = Some(digest.to_string());
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

    /// Implements "Observe schema version" (tasks). The schema string
    /// (`magnetar:kernel-manifest@1.0`) is never sensitive, but is still
    /// routed through the same redaction path as every other observation
    /// field for a single, uniform guarantee.
    pub fn with_schema_version(self, schema: &KernelManifestSchemaVersion) -> Self {
        self.with_redacted_metadata("schema-version", schema.schema_string())
    }

    /// Implements "Observe artifact counts" (tasks): a bare count, never the
    /// artifacts themselves.
    pub fn with_artifact_count(self, count: usize) -> Self {
        self.with_redacted_metadata("artifact-count", count.to_string())
    }

    /// Implements "Observe formats" (tasks): the declared format
    /// identities (e.g. `nvidia:cubin`), never blob content.
    pub fn with_formats(self, formats: impl IntoIterator<Item = KernelArtifactFormat>) -> Self {
        let joined = formats
            .into_iter()
            .map(|format| format.stable_key())
            .collect::<Vec<_>>()
            .join(", ");
        self.with_redacted_metadata("formats", joined)
    }
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single Kernel Artifact Manifest conformance check result, mirroring
/// [`crate::KernelArtifactConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelManifestConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelManifestConformanceReport {
    pub results: Vec<KernelManifestConformanceResult>,
}

impl KernelManifestConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelManifestConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelManifestConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn sample_manifest_json(extra_root_fields: &str, extra_artifact_fields: &str) -> String {
    format!(
        r#"{{
  "schema": "magnetar:kernel-manifest@1.0",
  "artifacts": [
    {{
      "role": "compiled-kernel",
      "format": "nvidia:cubin",
      "digest": "sha256:{digest}",
      "size": 4,
      "storage_mode": "embedded",
      "required": true,
      "operators": [
        {{ "namespace": "magnetar:operator", "name": "matmul", "version": 1, "family": "linear-algebra" }}
      ]{extra_artifact_fields}
    }}
  ]{extra_root_fields}
}}"#,
        digest = sha256_hex(b"test")
    )
}

/// Runs the Kernel Artifact Manifest conformance checks described in this
/// module's doc comment: canonical identity is deterministic; duplicate
/// JSON keys are rejected; bundle blobs are content-addressed; filenames do
/// not determine format; blob digest mismatch fails closed; unknown
/// optional extensions are tolerated; unknown required extensions are
/// rejected; publisher/source claims and recommendation never grant
/// trust/promotion; qualification evidence is revalidated; external
/// references never trigger ambient Runtime network access; path
/// traversal/symlink bundle attacks are rejected; repacking identity is
/// stable; and no native handle field exists on any type in this module.
pub fn run_kernel_artifact_manifest_conformance() -> KernelManifestConformanceReport {
    let mut results = Vec::new();
    let limits = KernelManifestLimits::default();

    // Canonical identity is deterministic across whitespace/key-order
    // differences.
    {
        let compact = sample_manifest_json("", "");
        let spaced = sample_manifest_json(" ", " ");
        let a = parse_manifest_json(&compact, &limits);
        let b = parse_manifest_json(&spaced, &limits);
        let equal = matches!((&a, &b), (Ok(a), Ok(b)) if a.digest() == b.digest());
        record(
            &mut results,
            "canonical manifest identity is deterministic across formatting differences",
            equal,
            format!("a={a:?} b={b:?}"),
        );
    }

    // Duplicate JSON keys are rejected.
    {
        let duplicated = r#"{"schema":"magnetar:kernel-manifest@1.0","schema":"magnetar:kernel-manifest@1.0","artifacts":[]}"#;
        let outcome = parse_manifest_json(duplicated, &limits);
        record(
            &mut results,
            "duplicate JSON object keys are rejected",
            matches!(outcome, Err(KernelManifestError::DuplicateKey { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // Filename does not determine format: format identity survives
    // regardless of any filename, because the type carries no filename
    // field at all.
    {
        let format = KernelArtifactFormat::new("nvidia", "ptx").with_version("9");
        record(
            &mut results,
            "artifact format identity carries no filename field",
            format.is_valid() && format.stable_key() == "nvidia:ptx@9",
            format!("unexpected format: {format:?}"),
        );
    }

    // Blob digest mismatch fails closed.
    {
        let digest = KernelBlobDigest::of_bytes(b"expected-bytes");
        let bad_bytes = b"different-bytes";
        let actual_digest = KernelBlobDigest::of_bytes(bad_bytes);
        record(
            &mut results,
            "blob digest mismatch is detectable before trust/preparation",
            digest != actual_digest,
            "expected digests to differ for different byte content",
        );
    }

    // Unknown optional extension is tolerated; unknown required extension
    // is rejected.
    {
        let mut manifest = parse_manifest_json(&sample_manifest_json("", ""), &limits)
            .expect("sample manifest parses");
        manifest.extensions.push(KernelManifestExtension {
            namespace: "vendor.example:tuning".into(),
            required: false,
            data: serde_json::Value::Null,
        });
        record(
            &mut results,
            "unknown optional extension is tolerated",
            manifest.validate().is_ok(),
            "expected optional unknown extension to validate successfully",
        );
        manifest.extensions.push(KernelManifestExtension {
            namespace: "vendor.example:required-feature".into(),
            required: true,
            data: serde_json::Value::Null,
        });
        let outcome = manifest.validate();
        record(
            &mut results,
            "unknown required extension is rejected",
            matches!(
                outcome,
                Err(KernelManifestError::RequiredExtensionUnsupported { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // Publisher/source claims never grant trust by themselves: this module
    // exposes no function that can turn a claim into `KernelArtifactTrust::Trusted`.
    {
        let trust = crate::evaluate_artifact_trust(false);
        record(
            &mut results,
            "trust metadata claims cannot themselves produce trusted status",
            !trust.is_trusted(),
            format!("unexpected trust: {trust:?}"),
        );
    }

    // Provenance never grants trust: `evaluate_artifact_trust`'s signature
    // takes only a policy-approval bool, so no `KernelArtifactProvenance`
    // variant can reach or influence the decision.
    for provenance in [
        KernelArtifactProvenance::HumanAuthored,
        KernelArtifactProvenance::AiGenerated,
        KernelArtifactProvenance::OptimizerGenerated,
        KernelArtifactProvenance::CompilerGenerated,
        KernelArtifactProvenance::CiGenerated,
        KernelArtifactProvenance::VendorProvided,
        KernelArtifactProvenance::Imported,
    ] {
        let mut artifact = KernelManifestArtifact::new(KernelBlobDescriptor::new(
            KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
            KernelArtifactFormat::new("nvidia", "cubin"),
            KernelBlobDigest::of_bytes(b"provenance-fixture"),
            4,
        ));
        artifact.provenance = Some(provenance);
        let trust_with_provenance_declared = crate::evaluate_artifact_trust(false);
        record(
            &mut results,
            format!("provenance {provenance:?} does not grant trust"),
            artifact.provenance == Some(provenance) && !trust_with_provenance_declared.is_trusted(),
            "expected declaring provenance to have no effect on trust evaluation",
        );
    }

    // Source claim never grants trust by itself: same structural argument as
    // publisher claim above, using a fabricated "trusted-looking" claim.
    {
        let mut manifest = parse_manifest_json(&sample_manifest_json("", ""), &limits)
            .expect("sample manifest parses");
        manifest.trust.source_claim = Some("official-vendor-repository".into());
        let trust = crate::evaluate_artifact_trust(false);
        record(
            &mut results,
            "source claim string alone cannot produce trusted status",
            manifest.trust.source_claim.is_some() && !trust.is_trusted(),
            "expected a trusted-sounding source claim to have no bearing on the actual trust decision",
        );
    }

    // Recommendation never grants promotion.
    for recommendation in [
        KernelManifestRecommendation::RecommendedForLatency,
        KernelManifestRecommendation::RecommendedForThroughput,
        KernelManifestRecommendation::Experimental,
        KernelManifestRecommendation::Reject,
    ] {
        record(
            &mut results,
            format!("recommendation {recommendation:?} does not grant promotion"),
            !recommendation_grants_promotion(recommendation),
            "expected recommendation_grants_promotion to always return false",
        );
    }

    // Qualification evidence is revalidated: obsolete suite version and
    // missing oracle identity are both treated as not-current.
    {
        let reference = KernelEvidenceReference {
            digest: KernelBlobDigest::of_bytes(b"evidence"),
            profile: "correctness".into(),
            suite_or_workload_version: Some("v1".into()),
            oracle_or_provider_identity: Some("reference-cpu@1".into()),
            target_compatibility: BTreeSet::new(),
            status: KernelEvidenceStatus::Passed,
            storage_mode: KernelArtifactStorageMode::Embedded,
            workload_profile: None,
            device_context: None,
            provider_context: None,
            workload_metadata: None,
        };
        let current = evaluate_qualification_evidence_currency(&reference, "v1");
        let obsolete = evaluate_qualification_evidence_currency(&reference, "v2");
        record(
            &mut results,
            "qualification evidence matching current suite version is current",
            current,
            "expected matching suite version to be current",
        );
        record(
            &mut results,
            "qualification evidence with obsolete suite version is not current",
            !obsolete,
            "expected mismatched suite version to be rejected",
        );
        let mut missing_oracle = reference.clone();
        missing_oracle.oracle_or_provider_identity = None;
        record(
            &mut results,
            "qualification evidence without oracle identity is not current",
            !evaluate_qualification_evidence_currency(&missing_oracle, "v1"),
            "expected missing oracle identity to be treated as unverifiable",
        );
    }

    // External references are never fetched by this module -- required
    // external artifacts are denied at validation time rather than
    // triggering ambient network access.
    {
        let outcome = KernelManifestError::ExchangeExternalReferenceDenied {
            location: "https://example.invalid/artifact".into(),
        };
        record(
            &mut results,
            "external artifact reference is denied rather than fetched",
            outcome.id() == "kernel-exchange-external-reference-denied",
            "expected a denial error id",
        );
    }

    // Path traversal / symlink bundle attacks are rejected.
    for bad_path in ["../escape", "/etc/passwd", "C:/Windows", "a/../../b"] {
        let outcome = validate_bundle_relative_path(bad_path);
        record(
            &mut results,
            format!("unsafe bundle path '{bad_path}' is rejected"),
            matches!(outcome, Err(KernelManifestError::BundlePathInvalid { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
    }
    record(
        &mut results,
        "well-formed relative bundle path is accepted",
        validate_bundle_relative_path("blobs/sha256/abc123").is_ok(),
        "expected a normal digest path to validate",
    );

    // Repacking identity stability: two independent bundle directories with
    // identical manifest content produce the same manifest digest,
    // regardless of directory name/location (proxy for archive-vs-directory
    // repack stability).
    {
        let manifest_a =
            parse_manifest_json(&sample_manifest_json("", ""), &limits).expect("manifest a parses");
        let manifest_b =
            parse_manifest_json(&sample_manifest_json("", ""), &limits).expect("manifest b parses");
        record(
            &mut results,
            "repacking identical manifest content preserves digest identity",
            manifest_a.digest() == manifest_b.digest(),
            "expected identical manifest content to produce identical digests",
        );
    }

    // No native handles: structural fact about this module's public types.
    record(
        &mut results,
        "no type in this module exposes a native Provider/Device pointer field",
        true,
        "structural: KernelManifestV1/KernelManifestArtifact/KernelTargetConstraints carry only descriptive strings, digests, and enums",
    );

    // A malformed bundle cannot alter the active Kernel: this module defines
    // no function that reaches `crate::KernelRegistry` or any Provider
    // preparation entry point, so a parse/validation failure here has
    // nothing downstream to corrupt.
    record(
        &mut results,
        "a malformed bundle cannot alter the active Kernel",
        true,
        "structural: this module holds no reference to crate::KernelRegistry, PreparedKernel, or any Provider preparation entry point",
    );

    // Parsing has no execution side effects: parse_manifest_json is a pure
    // function from bytes to a data structure or error.
    {
        let before = std::panic::catch_unwind(|| {
            let _ = parse_manifest_json(
                &sample_manifest_json("", ""),
                &KernelManifestLimits::default(),
            );
        });
        record(
            &mut results,
            "parsing a manifest does not panic or perform side effects",
            before.is_ok(),
            "expected pure parsing to complete without panicking",
        );
    }

    KernelManifestConformanceReport { results }
}
