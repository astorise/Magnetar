//! Content-Addressed Kernel Artifact Cache (see
//! `openspec/changes/define-generated-kernel-qualification-cache-and-hot-swap-contract`).
//!
//! This module does not implement a persistent storage backend or
//! distributed cache replication (proposal's "Non-Goals"). It defines, as
//! executable Rust types and validation functions, the content-addressed
//! cache contract for [`crate::kernel_artifact::CompiledKernelArtifact`]s and
//! their [`crate::kernel_qualification::QualificationRecord`]s:
//!
//! ```text
//! CompiledKernelArtifact + QualificationRecord -> Kernel Artifact Cache -> Provider Preparation
//! ```
//!
//! - [`KernelCacheKey`]: a compatibility-aware content-addressed key,
//!   implementing "Cache Identity": "keys SHOULD include enough
//!   compatibility context to prevent unsafe reuse."
//! - [`QualificationCacheKey`]: implements "Qualification Cache Identity":
//!   "Changing any qualification-relevant dimension SHALL invalidate or
//!   separate the record."
//! - [`CacheEntryState`]: implements "Cache States"' suggested lifecycle
//!   (`partial -> validating -> ready`, plus `untrusted`/`unqualified`/
//!   `qualified`/`rejected`/`revoked`/`corrupt`/`retiring`/`evicting`/
//!   `evicted`) -- "These states describe cache management and SHALL NOT
//!   replace current Runtime eligibility decisions."
//! - [`KernelCacheEntry`] / [`evaluate_cache_eligibility`]: implements "Cache
//!   Hit Does Not Imply Eligibility": `cache hit != trusted`, `!= qualified`,
//!   `!= compatible`, `!= active` -- current Runtime policy always
//!   re-evaluates.
//! - [`verify_cache_entry_integrity`]: implements "Cache Integrity": corrupt
//!   entries are never used, and corruption of one entry SHALL NOT
//!   invalidate unrelated entries.
//! - [`KernelArtifactCache`] / [`KernelArtifactCache::insert`] /
//!   [`KernelArtifactCache::evict`]: implements "Immutable Cache Entries"
//!   (mutation produces a new digest-addressed entry), "Cache Eviction"
//!   (never destroys a still-active Prepared Kernel merely because its
//!   backing artifact is evicted), and "Cache Pinning".
//! - [`KernelCacheError`]: the cache subset of the proposal's "Error Model"
//!   section.
//! - [`CacheObservationKind`] / [`CacheObservation`]: redacted cache
//!   lifecycle observability.
//! - [`KernelCacheConformanceReport`] / [`run_kernel_cache_conformance`]: the
//!   conformance checks from this change's `specs/kernel-cache/spec.md` and
//!   the cache-related requirements of `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::kernel_artifact::{CompiledKernelArtifactId, KernelArtifactTrust};
use crate::kernel_qualification::QualificationStatus;
use crate::{ComputeDType, TensorLayoutKind};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_CACHE_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Cache Identity
// ---------------------------------------------------------------------

/// Compatibility-aware content-addressed cache key, implementing "Cache
/// Identity" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCacheKey {
    pub source_digest: Option<String>,
    pub compiled_artifact_digest: String,
    pub source_format: Option<String>,
    pub compiled_format: String,
    pub compiler_identity: String,
    pub compiler_version: String,
    pub compiler_flags_fingerprint: Option<String>,
    pub provider_version: String,
    pub target_architecture: String,
    pub driver_runtime_compatibility_class: BTreeSet<String>,
    pub operator_semantics: String,
    pub dtype: BTreeSet<ComputeDType>,
    pub layout: BTreeSet<TensorLayoutKind>,
    pub shape_specialization: Option<String>,
    pub device_features: BTreeSet<String>,
}

impl KernelCacheKey {
    /// A stable string suitable for use as the cache map key, mirroring
    /// [`crate::kernel_artifact::KernelArtifactCacheKey::stable_key`].
    pub fn stable_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.compiled_artifact_digest,
            self.compiler_identity,
            self.compiler_version,
            self.provider_version,
            self.target_architecture,
        )
    }
}

/// Qualification-record cache key, implementing "Qualification Cache
/// Identity" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationCacheKey {
    pub artifact_digest: String,
    pub qualification_suite_version: String,
    pub oracle_identity_version: String,
    pub qualification_profile: String,
    pub target_context: String,
    pub test_matrix_fingerprint: String,
    pub tolerance_profile_fingerprint: String,
}

impl QualificationCacheKey {
    /// Implements "Changing any qualification-relevant dimension SHALL
    /// invalidate or separate the record according to policy" (proposal):
    /// exact equality is the only reuse condition.
    pub fn is_reusable_for(&self, requested: &QualificationCacheKey) -> bool {
        self == requested
    }
}

// ---------------------------------------------------------------------
// Cache States
// ---------------------------------------------------------------------

/// Cache entry lifecycle state, implementing "Cache States" (proposal):
/// "These states describe cache management and SHALL NOT replace current
/// Runtime eligibility decisions" -- see [`evaluate_cache_eligibility`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CacheEntryState {
    Partial,
    Validating,
    Ready,
    Untrusted,
    Unqualified,
    Qualified,
    Rejected,
    Revoked,
    Corrupt,
    Retiring,
    Evicting,
    Evicted,
}

impl CacheEntryState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Partial, Self::Validating)
                | (Self::Validating, Self::Ready)
                | (Self::Validating, Self::Corrupt)
                | (Self::Ready, Self::Untrusted)
                | (Self::Ready, Self::Unqualified)
                | (Self::Ready, Self::Qualified)
                | (Self::Ready, Self::Rejected)
                | (Self::Ready, Self::Revoked)
                | (Self::Ready, Self::Corrupt)
                | (Self::Ready, Self::Retiring)
                | (Self::Untrusted, Self::Retiring)
                | (Self::Unqualified, Self::Retiring)
                | (Self::Qualified, Self::Revoked)
                | (Self::Qualified, Self::Retiring)
                | (Self::Rejected, Self::Retiring)
                | (Self::Revoked, Self::Retiring)
                | (Self::Corrupt, Self::Evicting)
                | (Self::Retiring, Self::Evicting)
                | (Self::Evicting, Self::Evicted)
        )
    }

    /// Whether a read of an entry in this state may be used at all,
    /// implementing "Corrupt entries SHALL NOT be used" (proposal).
    pub const fn is_readable(self) -> bool {
        !matches!(self, Self::Corrupt | Self::Evicting | Self::Evicted)
    }
}

// ---------------------------------------------------------------------
// Cache Entry
// ---------------------------------------------------------------------

/// A single Kernel Artifact cache entry, implementing "Kernel Cache
/// Contents" (proposal). Deliberately holds no native handle -- only the
/// [`CompiledKernelArtifactId`] identity, integrity digest, and lifecycle
/// metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCacheEntry {
    pub key: KernelCacheKey,
    pub artifact: CompiledKernelArtifactId,
    pub stored_digest: String,
    pub trust: KernelArtifactTrust,
    pub qualification: Option<QualificationStatus>,
    pub state: CacheEntryState,
    pub pinned: bool,
}

impl KernelCacheEntry {
    pub fn new(
        key: KernelCacheKey,
        artifact: CompiledKernelArtifactId,
        digest: impl Into<String>,
    ) -> Self {
        Self {
            key,
            artifact,
            stored_digest: digest.into(),
            trust: KernelArtifactTrust::Untrusted,
            qualification: None,
            state: CacheEntryState::Partial,
            pinned: false,
        }
    }

    fn transition(&mut self, next: CacheEntryState) -> Result<(), KernelCacheError> {
        if !self.state.can_transition_to(next) {
            return Err(KernelCacheError::EntryInvalid {
                reason: format!("cannot transition from {:?} to {next:?}", self.state),
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn mark_validating(&mut self) -> Result<(), KernelCacheError> {
        self.transition(CacheEntryState::Validating)
    }

    pub fn mark_ready(&mut self) -> Result<(), KernelCacheError> {
        self.transition(CacheEntryState::Ready)
    }

    pub fn mark_corrupt(&mut self) -> Result<(), KernelCacheError> {
        self.transition(CacheEntryState::Corrupt)
    }
}

/// Implements "Cache reads SHALL validate integrity according to policy" and
/// "Corruption of one cache entry SHALL NOT invalidate unrelated entries"
/// (proposal): checks only `entry`'s own digest, never touching or scanning
/// other entries.
pub fn verify_cache_entry_integrity(
    entry: &KernelCacheEntry,
    computed_digest: &str,
) -> Result<(), KernelCacheError> {
    if !entry.state.is_readable() {
        return Err(KernelCacheError::EntryCorrupt {
            artifact: entry.artifact.to_string(),
        });
    }
    if entry.stored_digest != computed_digest {
        return Err(KernelCacheError::EntryCorrupt {
            artifact: entry.artifact.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Cache Hit Does Not Imply Eligibility
// ---------------------------------------------------------------------

/// Runtime-side eligibility policy re-evaluated on every cache hit,
/// implementing "Cache Hit Does Not Imply Eligibility" (proposal): `cache
/// hit != trusted`, `!= qualified`, `!= compatible`, `!= active`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheEligibilityPolicy {
    pub require_trusted: bool,
    pub require_qualified: bool,
    pub require_compatible_target: bool,
}

/// Implements "Current Runtime policy SHALL re-evaluate metadata needed for
/// current execution" (proposal): a cache hit is only ever the starting
/// point for this evaluation, never a substitute for it.
pub fn evaluate_cache_eligibility(
    entry: &KernelCacheEntry,
    target_compatible: bool,
    policy: &CacheEligibilityPolicy,
) -> Result<(), KernelCacheError> {
    if !entry.state.is_readable() {
        return Err(KernelCacheError::EntryCorrupt {
            artifact: entry.artifact.to_string(),
        });
    }
    if matches!(entry.state, CacheEntryState::Revoked) {
        return Err(KernelCacheError::EntryRevoked {
            artifact: entry.artifact.to_string(),
        });
    }
    if policy.require_trusted && !entry.trust.is_trusted() {
        return Err(KernelCacheError::EntryUntrusted {
            artifact: entry.artifact.to_string(),
        });
    }
    if policy.require_qualified
        && !entry
            .qualification
            .map(QualificationStatus::is_eligible)
            .unwrap_or(false)
    {
        return Err(KernelCacheError::EntryUnqualified {
            artifact: entry.artifact.to_string(),
        });
    }
    if policy.require_compatible_target && !target_compatible {
        return Err(KernelCacheError::CompatibilityMismatch {
            artifact: entry.artifact.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Cache Offline Behavior
// ---------------------------------------------------------------------

/// Implements "Cache Offline Behavior" (proposal): "A valid cached Compiled
/// Kernel Artifact MAY enable offline operation. Runtime SHALL NOT require
/// recompilation when a compatible cached artifact is available and policy
/// allows reuse." Reuses [`evaluate_cache_eligibility`]'s readable/revoked
/// checks so offline reuse never bypasses ordinary cache integrity rules --
/// it only adds the "no forced recompilation" and "incompatible target is
/// rejected" behavior specific to offline operation.
pub fn evaluate_offline_reuse(
    entry: &KernelCacheEntry,
    target_compatible: bool,
    policy_allows_reuse: bool,
) -> Result<(), KernelCacheError> {
    if !entry.state.is_readable() {
        return Err(KernelCacheError::EntryCorrupt {
            artifact: entry.artifact.to_string(),
        });
    }
    if !target_compatible {
        return Err(KernelCacheError::CompatibilityMismatch {
            artifact: entry.artifact.to_string(),
        });
    }
    if !policy_allows_reuse {
        return Err(KernelCacheError::Unavailable);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Memory Manager Boundary
// ---------------------------------------------------------------------

/// Optional preparation/executable-memory pressure signal a Kernel cache MAY
/// surface, implementing "Memory Manager" (proposal): "Memory Manager MAY
/// receive pressure information related to preparation or executable-memory
/// use, but SHALL retain authority over Runtime Tensor allocation/
/// residency." This struct carries only a coarse, informational level -- it
/// grants the cache no authority over `crate::memory::MemoryManager`'s
/// tensor allocation decisions, and nothing in this module consumes it to
/// make an eligibility or eviction decision (see [`evaluate_cache_eligibility`],
/// whose signature never takes a pressure level).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparationPressureHint {
    pub level: Option<crate::MemoryPressureLevel>,
}

// ---------------------------------------------------------------------
// Kernel Artifact Cache
// ---------------------------------------------------------------------

/// The content-addressed Kernel Artifact cache, implementing "Qualified
/// Kernel Cache" (proposal): "The cache SHALL remain distinct from Model
/// Artifact cache, Prefix Cache, KV Cache, Runtime Memory Manager
/// residency." This type has no field or dependency on any of those --
/// structurally distinct by construction. Implements "Memory Boundary"
/// (tasks): entries hold identity/digest/state metadata only, never a
/// [`crate::MemoryAllocationId`] or executable pointer, so persistent cache
/// storage and Prepared Kernel executable memory can never alias Runtime
/// Tensor memory through this type.
#[derive(Clone, Debug, Default)]
pub struct KernelArtifactCache {
    entries: BTreeMap<String, KernelCacheEntry>,
    observations: Vec<CacheObservation>,
    preparation_pressure: PreparationPressureHint,
}

impl KernelArtifactCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observations(&self) -> &[CacheObservation] {
        &self.observations
    }

    pub const fn preparation_pressure_hint(&self) -> PreparationPressureHint {
        self.preparation_pressure
    }

    /// Implements "Surface optional preparation pressure" (tasks). Purely
    /// informational: setting this never changes [`Self::insert`],
    /// [`Self::evict`], or [`evaluate_cache_eligibility`] behavior.
    pub fn set_preparation_pressure_hint(&mut self, hint: PreparationPressureHint) {
        self.preparation_pressure = hint;
    }

    pub fn get(&self, key: &str) -> Option<&KernelCacheEntry> {
        self.entries.get(key)
    }

    /// Implements "Immutable Cache Entries" (proposal): "Mutation SHOULD
    /// create a new digest-addressed entry rather than alter an existing
    /// artifact in place." Inserting under a key whose stored digest differs
    /// from the existing entry's digest is rejected -- callers must use a
    /// new key (new digest) for changed content.
    pub fn insert(&mut self, key: String, entry: KernelCacheEntry) -> Result<(), KernelCacheError> {
        if let Some(existing) = self.entries.get(&key)
            && existing.stored_digest != entry.stored_digest
        {
            return Err(KernelCacheError::InsertFailed {
                reason: "mutating an existing digest-addressed entry in place is not allowed"
                    .into(),
            });
        }
        self.entries.insert(key, entry);
        Ok(())
    }

    pub fn record_hit(&mut self, key: &str) {
        let kind = if self.entries.contains_key(key) {
            CacheObservationKind::CacheHit
        } else {
            CacheObservationKind::CacheMiss
        };
        self.observations.push(CacheObservation::new(kind));
    }

    /// Implements "Observe corruption" (tasks): marks the entry at `key`
    /// corrupt and records a distinct [`CacheObservationKind::CacheEntryCorrupt`]
    /// observation, separate from generic invalid-entry handling.
    pub fn record_corruption(&mut self, key: &str) -> Result<(), KernelCacheError> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or_else(|| KernelCacheError::EntryInvalid {
                reason: format!("unknown cache entry {key}"),
            })?;
        entry.mark_corrupt()?;
        self.observations.push(
            CacheObservation::new(CacheObservationKind::CacheEntryCorrupt)
                .with_artifact(entry.artifact.to_string()),
        );
        Ok(())
    }

    pub fn pin(&mut self, key: &str) -> Result<(), KernelCacheError> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or_else(|| KernelCacheError::EntryInvalid {
                reason: format!("unknown cache entry {key}"),
            })?;
        entry.pinned = true;
        Ok(())
    }

    /// Implements "Cache Eviction" (proposal): "Cache eviction SHALL not
    /// destroy a Prepared Kernel that is still active merely because its
    /// source/compiled artifact is evicted" and "Artifacts MAY be pinned...
    /// Pinning policy SHALL be explicit." `prepared_kernel_still_active` is
    /// an explicit caller-supplied fact -- this function never has implicit
    /// access to Prepared Kernel state, so eviction of the persistent
    /// artifact and destruction of native prepared state can never be
    /// conflated.
    pub fn evict(
        &mut self,
        key: &str,
        prepared_kernel_still_active: bool,
    ) -> Result<(), KernelCacheError> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| KernelCacheError::EntryInvalid {
                reason: format!("unknown cache entry {key}"),
            })?;
        if entry.pinned {
            return Err(KernelCacheError::EntryPinned {
                artifact: entry.artifact.to_string(),
            });
        }
        // Evicting the persistent artifact is independent of Prepared Kernel
        // lifetime: `prepared_kernel_still_active` only documents that the
        // caller has already accounted for it elsewhere (see
        // `crate::kernel_registry::KernelRegistry::destroy_prepared_kernel`),
        // it never blocks this cache-level eviction.
        let _ = prepared_kernel_still_active;
        self.entries.remove(key);
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Cache error, covering the cache subset of the
/// proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelCacheError {
    Unavailable,
    Miss,
    EntryInvalid { reason: String },
    EntryCorrupt { artifact: String },
    EntryUntrusted { artifact: String },
    EntryUnqualified { artifact: String },
    EntryRevoked { artifact: String },
    CompatibilityMismatch { artifact: String },
    InsertFailed { reason: String },
    EvictionDenied { reason: String },
    EntryPinned { artifact: String },
}

impl KernelCacheError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Unavailable => "kernel-cache-unavailable",
            Self::Miss => "kernel-cache-miss",
            Self::EntryInvalid { .. } => "kernel-cache-entry-invalid",
            Self::EntryCorrupt { .. } => "kernel-cache-entry-corrupt",
            Self::EntryUntrusted { .. } => "kernel-cache-entry-untrusted",
            Self::EntryUnqualified { .. } => "kernel-cache-entry-unqualified",
            Self::EntryRevoked { .. } => "kernel-cache-entry-revoked",
            Self::CompatibilityMismatch { .. } => "kernel-cache-compatibility-mismatch",
            Self::InsertFailed { .. } => "kernel-cache-insert-failed",
            Self::EvictionDenied { .. } => "kernel-cache-eviction-denied",
            Self::EntryPinned { .. } => "kernel-cache-entry-pinned",
        }
    }
}

impl fmt::Display for KernelCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryInvalid { reason }
            | Self::InsertFailed { reason }
            | Self::EvictionDenied { reason } => write!(f, "{}: {reason}", self.id()),
            Self::EntryCorrupt { artifact }
            | Self::EntryUntrusted { artifact }
            | Self::EntryUnqualified { artifact }
            | Self::EntryRevoked { artifact }
            | Self::CompatibilityMismatch { artifact }
            | Self::EntryPinned { artifact } => write!(f, "{}: {artifact}", self.id()),
            _ => write!(f, "{}", self.id()),
        }
    }
}

impl Error for KernelCacheError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CacheObservationKind {
    CacheHit,
    CacheMiss,
    CacheEntryInvalid,
    CacheEntryCorrupt,
}

/// A single cache observation. Structurally guaranteed to never carry raw
/// binary bytes or native handles, implementing "Observability" (proposal).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheObservation {
    pub kind: CacheObservationKind,
    pub artifact: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl CacheObservation {
    pub fn new(kind: CacheObservationKind) -> Self {
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
// Conformance
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCacheConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCacheConformanceReport {
    pub results: Vec<KernelCacheConformanceResult>,
}

impl KernelCacheConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelCacheConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelCacheConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn conformance_key(digest: &str) -> KernelCacheKey {
    KernelCacheKey {
        source_digest: None,
        compiled_artifact_digest: digest.into(),
        source_format: None,
        compiled_format: "nvidia:cubin".into(),
        compiler_identity: "nvcc".into(),
        compiler_version: "12.0".into(),
        compiler_flags_fingerprint: None,
        provider_version: "1.0.0".into(),
        target_architecture: "sm90".into(),
        driver_runtime_compatibility_class: BTreeSet::new(),
        operator_semantics: "magnetar:matmul@1".into(),
        dtype: BTreeSet::new(),
        layout: BTreeSet::new(),
        shape_specialization: None,
        device_features: BTreeSet::new(),
    }
}

/// Runs the Kernel Cache conformance checks described in this module's doc
/// comment and required by `specs/kernel-cache/spec.md` and the cache
/// portion of `specs/conformance/spec.md`.
pub fn run_kernel_cache_conformance() -> KernelCacheConformanceReport {
    let mut results = Vec::new();

    // Cache hit does not imply eligibility: a revoked entry is rejected even
    // though it is present in the cache.
    let mut revoked_entry = KernelCacheEntry::new(
        conformance_key("digest-a"),
        CompiledKernelArtifactId::from_digest("digest-a"),
        "sha256:aaa",
    );
    revoked_entry.mark_validating().ok();
    revoked_entry.mark_ready().ok();
    revoked_entry.trust = KernelArtifactTrust::Trusted;
    revoked_entry.qualification = Some(QualificationStatus::Qualified);
    revoked_entry.transition(CacheEntryState::Revoked).ok();
    let policy = CacheEligibilityPolicy {
        require_trusted: true,
        require_qualified: true,
        require_compatible_target: true,
    };
    let revoked_result = evaluate_cache_eligibility(&revoked_entry, true, &policy);
    record(
        &mut results,
        "cache hit on revoked entry is rejected",
        matches!(revoked_result, Err(KernelCacheError::EntryRevoked { .. })),
        format!("unexpected outcome: {revoked_result:?}"),
    );

    // Cache corruption fails closed and does not affect unrelated entries.
    let mut corrupt_entry = KernelCacheEntry::new(
        conformance_key("digest-b"),
        CompiledKernelArtifactId::from_digest("digest-b"),
        "sha256:bbb",
    );
    corrupt_entry.mark_validating().ok();
    let integrity = verify_cache_entry_integrity(&corrupt_entry, "sha256:different");
    record(
        &mut results,
        "digest mismatch is rejected as corrupt before use",
        integrity.is_err(),
        format!("unexpected outcome: {integrity:?}"),
    );

    let mut cache = KernelArtifactCache::new();
    let mut healthy_entry = KernelCacheEntry::new(
        conformance_key("digest-c"),
        CompiledKernelArtifactId::from_digest("digest-c"),
        "sha256:ccc",
    );
    healthy_entry.mark_validating().ok();
    healthy_entry.mark_ready().ok();
    cache.insert("digest-c".into(), healthy_entry.clone()).ok();
    corrupt_entry.mark_corrupt().ok();
    cache.insert("digest-b".into(), corrupt_entry).ok();
    record(
        &mut results,
        "corruption of one entry does not remove unrelated entries",
        cache.get("digest-c").is_some() && cache.get("digest-c").unwrap().state.is_readable(),
        "expected unrelated healthy entry to remain readable",
    );

    // Immutable cache entries: inserting a different digest under the same
    // key is rejected rather than mutating in place.
    let mut mutated = healthy_entry.clone();
    mutated.stored_digest = "sha256:mutated".into();
    let mutation_result = cache.insert("digest-c".into(), mutated);
    record(
        &mut results,
        "mutating an existing digest-addressed entry in place is rejected",
        matches!(mutation_result, Err(KernelCacheError::InsertFailed { .. })),
        format!("unexpected outcome: {mutation_result:?}"),
    );

    // Cache pinning protects a rollback candidate from eviction.
    cache.pin("digest-c").ok();
    let eviction = cache.evict("digest-c", false);
    record(
        &mut results,
        "pinned entry is protected from eviction",
        matches!(eviction, Err(KernelCacheError::EntryPinned { .. })),
        format!("unexpected outcome: {eviction:?}"),
    );

    // Eviction of a persistent artifact does not require Prepared Kernel
    // state to be inactive -- the two lifetimes are independent.
    let mut unpinned_entry = KernelCacheEntry::new(
        conformance_key("digest-d"),
        CompiledKernelArtifactId::from_digest("digest-d"),
        "sha256:ddd",
    );
    unpinned_entry.mark_validating().ok();
    unpinned_entry.mark_ready().ok();
    cache.insert("digest-d".into(), unpinned_entry).ok();
    let eviction_while_active = cache.evict("digest-d", true);
    record(
        &mut results,
        "evicting a persistent artifact succeeds independently of active Prepared Kernel state",
        eviction_while_active.is_ok(),
        format!("unexpected outcome: {eviction_while_active:?}"),
    );

    // Cache key identity: differing compatibility-relevant fields produce
    // differing stable keys, and identical fields produce identical keys.
    let key_a = conformance_key("digest-e");
    let mut key_b = conformance_key("digest-e");
    key_b.target_architecture = "sm80".into();
    record(
        &mut results,
        "cache key identity differs when target architecture differs",
        key_a.stable_key() != key_b.stable_key(),
        "expected differing target architecture to change the stable key",
    );
    record(
        &mut results,
        "cache key identity is stable for identical compatibility context",
        key_a.stable_key() == conformance_key("digest-e").stable_key(),
        "expected identical key fields to produce an identical stable key",
    );

    // Offline reuse: a compatible cached artifact avoids recompilation, and
    // an incompatible target is rejected rather than silently reused.
    let mut offline_entry = KernelCacheEntry::new(
        conformance_key("digest-f"),
        CompiledKernelArtifactId::from_digest("digest-f"),
        "sha256:fff",
    );
    offline_entry.mark_validating().ok();
    offline_entry.mark_ready().ok();
    let offline_reuse = evaluate_offline_reuse(&offline_entry, true, true);
    record(
        &mut results,
        "compatible cached artifact is reused offline without recompilation",
        offline_reuse.is_ok(),
        format!("unexpected outcome: {offline_reuse:?}"),
    );
    let offline_incompatible = evaluate_offline_reuse(&offline_entry, false, true);
    record(
        &mut results,
        "incompatible cached target is rejected rather than reused offline",
        matches!(
            offline_incompatible,
            Err(KernelCacheError::CompatibilityMismatch { .. })
        ),
        format!("unexpected outcome: {offline_incompatible:?}"),
    );

    // Corruption is observable as a distinct event, and the corrupted entry
    // becomes unreadable afterward.
    let mut corruption_observable_cache = KernelArtifactCache::new();
    let mut corruptible_entry = KernelCacheEntry::new(
        conformance_key("digest-g"),
        CompiledKernelArtifactId::from_digest("digest-g"),
        "sha256:ggg",
    );
    corruptible_entry.mark_validating().ok();
    corruptible_entry.mark_ready().ok();
    corruption_observable_cache
        .insert("digest-g".into(), corruptible_entry)
        .ok();
    corruption_observable_cache
        .record_corruption("digest-g")
        .ok();
    record(
        &mut results,
        "corruption is recorded as a distinct observation",
        matches!(
            corruption_observable_cache.observations().last(),
            Some(observation) if observation.kind == CacheObservationKind::CacheEntryCorrupt
        ),
        format!(
            "unexpected observations: {:?}",
            corruption_observable_cache.observations()
        ),
    );
    record(
        &mut results,
        "corrupted entry becomes unreadable",
        !corruption_observable_cache
            .get("digest-g")
            .unwrap()
            .state
            .is_readable(),
        "expected corrupted entry to be unreadable",
    );

    KernelCacheConformanceReport { results }
}
