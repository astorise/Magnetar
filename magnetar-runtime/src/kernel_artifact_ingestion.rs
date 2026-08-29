//! Kernel Artifact Ingestion Gateway and Policy (see
//! `openspec/changes/define-kernel-artifact-ingestion-and-policy-gateway`).
//!
//! This module defines the authoritative boundary through which external
//! Kernel Exchange Bundles ([`crate::kernel_artifact_manifest::KernelExchangeBundle`])
//! become accepted [`crate::KernelArtifactCache`] content:
//!
//! ```text
//! untrusted external world -> staging -> Ingestion Gateway -> {reject, quarantine, accept} -> atomic cache commit
//! ```
//!
//! It does not implement Provider compilation, qualification benchmarking, or
//! Kernel promotion/selection (proposal's "Non-Goals"). It reuses the
//! existing manifest/bundle validation pipeline
//! ([`crate::kernel_artifact_manifest::validate_kernel_exchange_bundle`])
//! for structural/schema/blob-integrity/semantic validation rather than
//! duplicating it, and adds the ingestion-specific layer on top:
//!
//! - [`IngestionTransactionId`] / [`KernelIngestionTransaction`]: an explicit,
//!   opaque-identified transaction implementing "Ingestion Transaction" and
//!   "Transaction Identity" (proposal). The identifier encodes no native
//!   pointer, filesystem handle, process ID, secret, or Provider handle.
//! - [`IngestionState`]: the transaction lifecycle states from "Transaction
//!   States" (proposal), with [`IngestionState::can_transition_to`] the sole
//!   authority for legal transitions.
//! - [`ObservedIngestionSource`] / [`evaluate_source_claim`]: implements
//!   "Ingestion Source" and "Observed Versus Claimed Source" (proposal): a
//!   manifest's own `trust.source_claim` can never override gateway-observed
//!   source metadata -- [`evaluate_source_claim`] never returns the claim as
//!   the observed value.
//! - [`IngestionTrustState`] / [`evaluate_ingestion_trust`]: implements
//!   "Trust Evaluation", "Fail-Closed Production Trust", and "Trust Policy
//!   Result" (proposal). Delegates to [`crate::evaluate_artifact_trust`] for
//!   the underlying trusted/untrusted boolean -- this module adds no second
//!   way to produce [`crate::KernelArtifactTrust::Trusted`].
//! - [`RevokedArtifactRegistry`]: implements "Revocation Interaction" and
//!   "Revocation Persistence" (proposal): independent from cache presence,
//!   so deleting and re-importing a revoked digest cannot clear revocation.
//! - [`ArtifactSourceAuthority`] / [`ExternalDownloadLimits`]: implements
//!   "External Artifact Resolution", "Artifact Source Authority", "Source
//!   Fetch Integrity", "No Ambient Network Authority", and "Redirect Policy"
//!   (proposal): an unauthorized locator is never fetched, and a trusted
//!   host's redirect target is independently authorized.
//! - [`KernelIngestionPolicy`] / [`IngestionPolicyContext`] /
//!   [`KernelIngestionPolicy::evaluate`]: implements "Kernel Ingestion
//!   Policy", "Policy Versioning", and "Policy Precedence" (proposal): the
//!   manifest's own content never weakens policy, since policy evaluation
//!   never reads the manifest directly -- only the caller-built context.
//! - [`IngestionDecisionKind`] / [`QuarantineReason`]: implements "Ingestion
//!   Decision", "Accept", "Quarantine", and "Rejection" (proposal): `accept`
//!   never implies prepared/promoted/selected/executing.
//! - [`QuarantineNamespace`]: implements "Quarantine Namespace" and
//!   "Quarantine Does Not Prepare" (proposal): quarantined candidates are
//!   structurally invisible to normal Registry discovery.
//! - [`ManualApprovalRecord`] / [`apply_manual_approval`]: implements "Manual
//!   Approval" and "Approval Identity" (proposal): approval can never repair
//!   a digest mismatch.
//! - [`run_ingestion_pipeline`] / [`commit_accepted_transaction`]: implements
//!   the proposal's "Validation Pipeline" ingestion-specific stages and
//!   "Atomic Cache Commit": the full commit set is precomputed and checked
//!   for conflicts before any [`crate::KernelArtifactCache`] mutation occurs.
//! - [`KernelIngestionAuditRecord`] / [`IngestionObservationKind`]: implements
//!   "Audit Record" and "Observability" (proposal), redacted by construction
//!   via `redact_backend_diagnostic`.
//! - [`IngestionError`]: the structured error categories from the proposal's
//!   "Error Model" section.
//! - [`KernelIngestionConformanceReport`] /
//!   [`run_kernel_artifact_ingestion_conformance`]: executable conformance
//!   evidence for `specs/conformance/spec.md`.

use crate::compute::redact_backend_diagnostic;
use crate::kernel_artifact_manifest::{
    KernelArtifactStorageMode, KernelBlobDigest, KernelEvidenceReference, KernelEvidenceStatus,
    KernelExchangeBundle, KernelManifestError, KernelManifestLimits, KernelManifestV1,
    ValidatedKernelManifest, evaluate_qualification_evidence_currency,
    validate_kernel_exchange_bundle,
};
use crate::{
    KernelArtifactCache, KernelCacheError, MemoryPressureLevel, evaluate_artifact_trust,
    normalize_to_cache_entry, normalize_to_cache_key,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const KERNEL_INGESTION_CONTRACT_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Transaction Identity
// ---------------------------------------------------------------------

/// Opaque ingestion transaction identifier, implementing "Transaction
/// Identity" (proposal): "It SHALL NOT encode native pointer, filesystem
/// handle, process ID, secret, or Provider handle." Exposes no accessor to
/// its internal representation. Only [`IngestionTransactionIdAllocator`] can
/// construct one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IngestionTransactionId(u64);

impl fmt::Display for IngestionTransactionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ingestion-transaction-{}", self.0)
    }
}

/// Allocates sequential, opaque [`IngestionTransactionId`]s.
#[derive(Clone, Debug, Default)]
pub struct IngestionTransactionIdAllocator(u64);

impl IngestionTransactionIdAllocator {
    pub fn allocate(&mut self) -> IngestionTransactionId {
        self.0 += 1;
        IngestionTransactionId(self.0)
    }
}

// ---------------------------------------------------------------------
// Observed Ingestion Source
// ---------------------------------------------------------------------

/// Extensible ingestion source vocabulary, implementing "Ingestion Source"
/// (proposal): "The vocabulary SHOULD remain extensible" ([`Self::Custom`]).
/// Descriptive only -- see [`evaluate_ingestion_trust`] for the only path to
/// trusted status, which never reads this value.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservedIngestionSource {
    LocalTooling,
    DeploymentPackage,
    Ci,
    ExternalArtifactSource,
    OptimizationCampaign,
    TachyonDistributed,
    VendorPackage,
    TestFixture,
    Custom(String),
}

impl ObservedIngestionSource {
    pub fn id(&self) -> String {
        match self {
            Self::LocalTooling => "local-tooling".into(),
            Self::DeploymentPackage => "deployment-package".into(),
            Self::Ci => "ci".into(),
            Self::ExternalArtifactSource => "external-artifact-source".into(),
            Self::OptimizationCampaign => "optimization-campaign".into(),
            Self::TachyonDistributed => "tachyon-distributed".into(),
            Self::VendorPackage => "vendor-package".into(),
            Self::TestFixture => "test-fixture".into(),
            Self::Custom(value) => value.clone(),
        }
    }
}

impl fmt::Display for ObservedIngestionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())
    }
}

/// Observed-versus-claimed source comparison, implementing "Observed Versus
/// Claimed Source" and "Source Is Not Trust" (proposal): the manifest's own
/// `trust.source_claim` never becomes, or overrides, the observed value --
/// this struct always carries both, distinctly, and [`Self::conflicting`] is
/// purely diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceClaimComparison {
    pub observed: ObservedIngestionSource,
    pub manifest_claim: Option<String>,
    pub conflicting: bool,
}

/// Implements "A manifest's own `source` claim SHALL not override
/// gateway-observed source metadata" and "They SHALL not be conflated"
/// (proposal). The returned `observed` field is always `observed` -- there is
/// no code path in this function that assigns `manifest_claim` to it.
pub fn evaluate_source_claim(
    observed: &ObservedIngestionSource,
    manifest_claim: Option<&str>,
) -> SourceClaimComparison {
    let conflicting = manifest_claim
        .map(|claim| claim != observed.id())
        .unwrap_or(false);
    SourceClaimComparison {
        observed: observed.clone(),
        manifest_claim: manifest_claim.map(str::to_string),
        conflicting,
    }
}

// ---------------------------------------------------------------------
// Transaction States
// ---------------------------------------------------------------------

/// Ingestion transaction lifecycle state, implementing "Transaction States"
/// (proposal). [`Self::can_transition_to`] is the sole authority for legal
/// transitions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IngestionState {
    Created,
    Receiving,
    Staged,
    Validating,
    PolicyEvaluating,
    Quarantined,
    Accepted,
    Committing,
    Committed,
    Rejected,
    Cancelled,
    TimedOut,
    Failed,
    Cleaning,
    Closed,
}

impl IngestionState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Created, Self::Receiving)
                | (Self::Created, Self::Cancelled)
                | (Self::Created, Self::TimedOut)
                | (Self::Receiving, Self::Staged)
                | (Self::Receiving, Self::Cancelled)
                | (Self::Receiving, Self::TimedOut)
                | (Self::Receiving, Self::Failed)
                | (Self::Staged, Self::Validating)
                | (Self::Staged, Self::Cancelled)
                | (Self::Staged, Self::TimedOut)
                | (Self::Validating, Self::PolicyEvaluating)
                | (Self::Validating, Self::Rejected)
                | (Self::Validating, Self::Cancelled)
                | (Self::Validating, Self::TimedOut)
                | (Self::Validating, Self::Failed)
                | (Self::PolicyEvaluating, Self::Accepted)
                | (Self::PolicyEvaluating, Self::Quarantined)
                | (Self::PolicyEvaluating, Self::Rejected)
                | (Self::PolicyEvaluating, Self::Cancelled)
                | (Self::PolicyEvaluating, Self::TimedOut)
                | (Self::Accepted, Self::Committing)
                | (Self::Accepted, Self::Cancelled)
                | (Self::Committing, Self::Committed)
                | (Self::Committing, Self::Failed)
                | (Self::Quarantined, Self::Accepted)
                | (Self::Quarantined, Self::Rejected)
                | (Self::Quarantined, Self::Cleaning)
                | (Self::Rejected, Self::Cleaning)
                | (Self::Cancelled, Self::Cleaning)
                | (Self::TimedOut, Self::Cleaning)
                | (Self::Failed, Self::Cleaning)
                | (Self::Committed, Self::Cleaning)
                | (Self::Cleaning, Self::Closed)
        )
    }

    /// Implements "these states SHALL remain distinct" and gives every
    /// non-committed terminal/near-terminal state a common predicate used by
    /// [`KernelIngestionTransaction::cancel`] and deadline handling.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Rejected
                | Self::Cancelled
                | Self::TimedOut
                | Self::Failed
                | Self::Committed
                | Self::Closed
        )
    }

    /// Implements "Quarantine SHALL NOT imply acceptance for execution": a
    /// quarantined or any pre-commit state is never itself cache-committed
    /// content.
    pub const fn is_committed(self) -> bool {
        matches!(self, Self::Committed)
    }
}

// ---------------------------------------------------------------------
// Quotas / Limits
// ---------------------------------------------------------------------

/// Defensive ingestion quotas, implementing "Ingestion Resource Quotas"
/// (proposal). Wraps [`KernelManifestLimits`] (already enforced by
/// [`validate_kernel_exchange_bundle`]) rather than duplicating those fields,
/// and adds the ingestion-specific quotas the manifest/bundle layer has no
/// concept of.
#[derive(Clone, Debug, PartialEq)]
pub struct IngestionQuotas {
    pub manifest_limits: KernelManifestLimits,
    pub max_external_fetches: usize,
    pub max_staging_bytes: u64,
    pub max_concurrent_transactions: usize,
    /// Abstract validation-time budget in caller-defined ticks (this crate
    /// models no real wall clock -- see [`KernelIngestionTransaction::expire_if_past_deadline`]).
    pub max_validation_ticks: u64,
}

impl Default for IngestionQuotas {
    fn default() -> Self {
        Self {
            manifest_limits: KernelManifestLimits::default(),
            max_external_fetches: 16,
            max_staging_bytes: 16 * 1024 * 1024 * 1024,
            max_concurrent_transactions: 32,
            max_validation_ticks: 10_000,
        }
    }
}

impl IngestionQuotas {
    /// Implements "Limit staging storage" (tasks).
    pub fn check_staging_bytes(&self, used: u64) -> Result<(), IngestionError> {
        if used > self.max_staging_bytes {
            return Err(IngestionError::StagingLimitExceeded);
        }
        Ok(())
    }

    /// Implements "Define max concurrent transactions" (tasks).
    pub fn check_concurrent_transactions(&self, active: usize) -> Result<(), IngestionError> {
        if active > self.max_concurrent_transactions {
            return Err(IngestionError::ConcurrencyLimit);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Trust
// ---------------------------------------------------------------------

/// Structured trust evaluation result, implementing "Trust Policy Result"
/// (proposal): `trusted`, `untrusted`, `unsigned`, `unknown`, `denied`,
/// `development-allowed`. [`Self::Unknown`] is the state before evaluation
/// runs at all.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IngestionTrustState {
    #[default]
    Unknown,
    Trusted,
    Untrusted,
    Unsigned,
    Denied,
    DevelopmentAllowed,
}

impl IngestionTrustState {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Trusted => "trusted",
            Self::Untrusted => "untrusted",
            Self::Unsigned => "unsigned",
            Self::Denied => "denied",
            Self::DevelopmentAllowed => "development-allowed",
        }
    }

    /// Whether this trust state alone is sufficient to satisfy a policy that
    /// `require_trust`, implementing "Fail-Closed Production Trust":
    /// [`Self::Denied`] is never admissible, and only an explicit
    /// development-mode allowance or genuine trust is.
    pub const fn is_admissible(self) -> bool {
        matches!(self, Self::Trusted | Self::DevelopmentAllowed)
    }
}

/// Implements "Trust Evaluation", "Fail-Closed Production Trust", and "Trust
/// SHALL not come automatically from" (proposal): the only trust-bearing
/// input is `policy_approved`, delegated unchanged to
/// [`crate::evaluate_artifact_trust`] -- format, provenance, local origin,
/// cache presence, publisher string, source kind, CI label, and successful
/// compilation are all structurally absent from this signature.
pub fn evaluate_ingestion_trust(
    policy_approved: bool,
    explicitly_denied: bool,
    signed: bool,
    development_mode_allowed: bool,
) -> IngestionTrustState {
    if explicitly_denied {
        return IngestionTrustState::Denied;
    }
    let base = evaluate_artifact_trust(policy_approved);
    if base.is_trusted() {
        return IngestionTrustState::Trusted;
    }
    if !signed {
        return IngestionTrustState::Unsigned;
    }
    if development_mode_allowed {
        return IngestionTrustState::DevelopmentAllowed;
    }
    IngestionTrustState::Untrusted
}

// ---------------------------------------------------------------------
// Qualification Evidence Validation
// ---------------------------------------------------------------------

/// Outcome of evaluating qualification/benchmark evidence against current
/// policy, implementing "Qualification Evidence Validation" (proposal):
/// presence of a reference is distinguished from structural validity, from
/// integrity, and from currently-accepted status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceEvaluationOutcome {
    pub current: bool,
    pub reason: Option<QuarantineReason>,
}

/// Evaluates a manifest's qualification/benchmark evidence references,
/// implementing "Validate evidence digest", "Validate profile", "Validate
/// suite version", "Validate oracle", "Validate target compatibility",
/// "Check expiration", "Check revocation", and "Distinguish missing
/// evidence" (tasks). Delegates the suite-version/oracle-identity/status
/// currency check to
/// [`crate::kernel_artifact_manifest::evaluate_qualification_evidence_currency`]
/// (the sole authority for "portable evidence is not automatically current")
/// rather than re-implementing it -- this function only adds the
/// ingestion-level target-compatibility gate and missing/invalid/expired/
/// revoked classification on top.
pub fn evaluate_ingestion_evidence(
    references: &[KernelEvidenceReference],
    required_suite_or_workload_version: &str,
    target_context: Option<&str>,
) -> EvidenceEvaluationOutcome {
    if references.is_empty() {
        return EvidenceEvaluationOutcome {
            current: false,
            reason: Some(QuarantineReason::QualificationMissing),
        };
    }

    let mut saw_revoked = false;
    let mut saw_stale = false;

    for reference in references {
        // Validate evidence digest / profile: a structurally invalid
        // reference contributes nothing toward currency.
        if reference.validate().is_err() {
            continue;
        }
        match reference.status {
            KernelEvidenceStatus::Revoked => {
                saw_revoked = true;
                continue;
            }
            KernelEvidenceStatus::Stale => {
                saw_stale = true;
                continue;
            }
            _ => {}
        }
        // Validate target compatibility: a reference scoped to targets that
        // do not include the current target context does not count as
        // current for this ingestion.
        if let Some(target) = target_context
            && !reference.target_compatibility.is_empty()
            && !reference.target_compatibility.contains(target)
        {
            continue;
        }
        // Validate suite version / oracle / status currency.
        if evaluate_qualification_evidence_currency(reference, required_suite_or_workload_version) {
            return EvidenceEvaluationOutcome {
                current: true,
                reason: None,
            };
        }
    }

    if saw_revoked {
        return EvidenceEvaluationOutcome {
            current: false,
            reason: Some(QuarantineReason::PolicySpecific("evidence-revoked".into())),
        };
    }
    if saw_stale {
        return EvidenceEvaluationOutcome {
            current: false,
            reason: Some(QuarantineReason::EvidenceExpired),
        };
    }
    EvidenceEvaluationOutcome {
        current: false,
        reason: Some(QuarantineReason::QualificationMissing),
    }
}

// ---------------------------------------------------------------------
// Revocation
// ---------------------------------------------------------------------

/// Digest-keyed revocation registry, implementing "Revocation Interaction",
/// "Revocation Persistence", and "Quarantine Promotion After Revocation"
/// (proposal): "Revocation metadata SHOULD be independent from cache
/// presence" -- deliberately has no method that removes a revocation as a
/// side effect of any cache or transaction operation elsewhere in this
/// module; only an explicit new authoritative decision
/// ([`Self::authoritatively_clear`]) can do so.
#[derive(Clone, Debug, Default)]
pub struct RevokedArtifactRegistry {
    revoked: BTreeMap<String, String>,
}

impl RevokedArtifactRegistry {
    pub fn revoke(&mut self, digest: impl Into<String>, reason: impl Into<String>) {
        self.revoked.insert(digest.into(), reason.into());
    }

    pub fn is_revoked(&self, digest: &str) -> bool {
        self.revoked.contains_key(digest)
    }

    pub fn reason(&self, digest: &str) -> Option<&str> {
        self.revoked.get(digest).map(String::as_str)
    }

    /// Implements "A new authoritative revocation/trust decision is
    /// required" (proposal, "Quarantine Promotion After Revocation"): the
    /// only way to clear a revocation, distinct from any re-import path.
    pub fn authoritatively_clear(&mut self, digest: &str) {
        self.revoked.remove(digest);
    }
}

// ---------------------------------------------------------------------
// Manual Approval
// ---------------------------------------------------------------------

/// Management-boundary approval record, implementing "Manual Approval" and
/// "Approval Identity" (proposal): the portable artifact itself SHALL not
/// self-declare approval -- nothing in
/// [`crate::kernel_artifact_manifest::KernelManifestV1`] can construct this
/// type; it only ever comes from an external authenticated management
/// context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManualApprovalRecord {
    pub approval_event_id: String,
    pub approver_identity: String,
    pub policy_version: String,
    pub approved_digest: String,
}

/// Implements "An operator clicking approve SHALL NOT repair digest
/// mismatch" (proposal): `integrity_ok == false` always fails regardless of
/// approval, and the approval's `approved_digest` must match the digest being
/// admitted -- an approval for a different artifact grants nothing here.
pub fn apply_manual_approval(
    approval: &ManualApprovalRecord,
    digest: &str,
    integrity_ok: bool,
) -> Result<(), IngestionError> {
    if !integrity_ok {
        return Err(IngestionError::ManualApprovalCannotBypassIntegrity);
    }
    if approval.approved_digest != digest {
        return Err(IngestionError::ManualApprovalRequired);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// External Artifact Source Authority
// ---------------------------------------------------------------------

/// Defensive limits for external artifact retrieval, implementing "Download
/// Limits" (proposal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalDownloadLimits {
    pub max_response_bytes: u64,
    pub max_redirects: u32,
    pub max_fetches_per_transaction: usize,
    /// Abstract per-fetch timeout budget in caller-defined ticks, implementing
    /// "Add request timeout" (tasks) -- this crate models no real wall clock.
    pub max_fetch_ticks: u64,
    /// Implements "Add total transaction download limit" (tasks): the sum of
    /// all external fetches within one transaction, distinct from
    /// `max_response_bytes` (a single-fetch cap).
    pub max_total_transaction_bytes: u64,
}

impl Default for ExternalDownloadLimits {
    fn default() -> Self {
        Self {
            max_response_bytes: 4 * 1024 * 1024 * 1024,
            max_redirects: 3,
            max_fetches_per_transaction: 16,
            max_fetch_ticks: 1_000,
            max_total_transaction_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

/// Authorized external locator scheme/prefix allowlist, implementing
/// "External Artifact Resolution", "Artifact Source Authority", and "No
/// Ambient Network Authority" (proposal): "If no Artifact Source authorizes a
/// locator, gateway SHALL not fetch it." An empty allowlist authorizes
/// nothing -- there is no implicit wildcard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArtifactSourceAuthority {
    pub authorized_prefixes: BTreeSet<String>,
    pub download_limits: ExternalDownloadLimits,
}

impl ArtifactSourceAuthority {
    pub fn is_authorized(&self, locator: &str) -> bool {
        self.authorized_prefixes
            .iter()
            .any(|prefix| locator.starts_with(prefix.as_str()))
    }

    /// Implements "Source Fetch Integrity" and "Source Mutation" (proposal):
    /// externally fetched bytes are staged and validated against the
    /// declared digest before use, and TLS/authenticated transport never
    /// substitutes for this check.
    pub fn resolve(
        &self,
        locator: &str,
        declared_digest: &KernelBlobDigest,
        fetched_bytes: &[u8],
    ) -> Result<KernelBlobDigest, IngestionError> {
        if !self.is_authorized(locator) {
            return Err(IngestionError::ExternalReferenceDenied {
                location: locator.into(),
            });
        }
        if fetched_bytes.len() as u64 > self.download_limits.max_response_bytes {
            return Err(IngestionError::QuotaExceeded {
                quota: "external-response-bytes".into(),
            });
        }
        let actual = KernelBlobDigest::of_bytes(fetched_bytes);
        if actual.value != declared_digest.value {
            return Err(IngestionError::ExternalDigestMismatch {
                expected: declared_digest.value.clone(),
                actual: actual.value,
            });
        }
        Ok(actual)
    }

    /// Implements "Redirect Policy" (proposal): "A trusted host redirecting
    /// to an unauthorized destination SHALL not automatically expand network
    /// authority" -- the redirect target is checked exactly as an original
    /// locator would be, independent of whether `original` was authorized.
    pub fn authorize_redirect(
        &self,
        redirect_count: u32,
        target: &str,
    ) -> Result<(), IngestionError> {
        if redirect_count > self.download_limits.max_redirects {
            return Err(IngestionError::QuotaExceeded {
                quota: "external-redirect-count".into(),
            });
        }
        if !self.is_authorized(target) {
            return Err(IngestionError::ExternalRedirectDenied {
                target: target.into(),
            });
        }
        Ok(())
    }

    /// Implements "Add request timeout" (tasks).
    pub fn enforce_fetch_ticks(&self, elapsed_ticks: u64) -> Result<(), IngestionError> {
        if elapsed_ticks > self.download_limits.max_fetch_ticks {
            return Err(IngestionError::ExternalFetchTimeout);
        }
        Ok(())
    }

    /// Implements "Add total transaction download limit" (tasks): the
    /// cumulative byte budget across every external fetch in one
    /// transaction, checked independently of any single fetch's
    /// `max_response_bytes`.
    pub fn check_total_transaction_budget(
        &self,
        total_fetched_so_far: u64,
    ) -> Result<(), IngestionError> {
        if total_fetched_so_far > self.download_limits.max_total_transaction_bytes {
            return Err(IngestionError::QuotaExceeded {
                quota: "external-total-transaction-bytes".into(),
            });
        }
        Ok(())
    }
}

/// An authorized-source credential, implementing "Credentials" (proposal):
/// "Credentials SHALL remain outside portable manifest" and "Diagnostics
/// SHALL redact them." The `Debug` impl never prints `secret`, so any
/// diagnostic formatting of this type is redacted structurally rather than by
/// convention.
#[derive(Clone)]
pub struct ArtifactSourceCredential {
    pub credential_id: String,
    secret: String,
}

impl ArtifactSourceCredential {
    pub fn new(credential_id: impl Into<String>, secret: impl Into<String>) -> Self {
        Self {
            credential_id: credential_id.into(),
            secret: secret.into(),
        }
    }

    /// Only accessor to the raw secret; never called by this module's
    /// diagnostics, audit, or observability paths.
    pub fn reveal(&self) -> &str {
        &self.secret
    }
}

impl fmt::Debug for ArtifactSourceCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArtifactSourceCredential")
            .field("credential_id", &self.credential_id)
            .field("secret", &"<redacted>")
            .finish()
    }
}

// ---------------------------------------------------------------------
// Inference Resource Protection
// ---------------------------------------------------------------------

/// Coarse resource-pressure signal ingestion admission MAY consult,
/// implementing "Inference Resource Protection" (tasks). Mirrors
/// [`crate::kernel_cache::PreparationPressureHint`]'s shape: purely
/// informational, and reusing [`crate::MemoryPressureLevel`] rather than
/// defining a second pressure vocabulary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IngestionResourcePressureHint {
    pub memory: Option<MemoryPressureLevel>,
    pub cpu_saturated: bool,
}

/// Explicit admission policy, implementing "Keep inference scheduling
/// priority policy available" (tasks): `prioritize_inference` is the visible
/// switch a deployment sets to make ingestion yield to inference under
/// pressure; when `false`, ingestion is never throttled by this function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngestionAdmissionPolicy {
    pub prioritize_inference: bool,
}

impl Default for IngestionAdmissionPolicy {
    fn default() -> Self {
        Self {
            prioritize_inference: true,
        }
    }
}

/// Implements "Allow admission throttling for ingestion", "Bound CPU use",
/// and "Bound host memory" (tasks): ingestion work SHALL NOT be admitted
/// while inference is being prioritized and the host is under saturated
/// memory or CPU pressure. Never touches
/// [`crate::memory::MemoryManager`] or the scheduler directly -- it only
/// consumes an already-observed pressure snapshot the caller supplies, the
/// same boundary [`crate::kernel_cache::KernelArtifactCache::set_preparation_pressure_hint`]
/// uses.
pub fn admit_ingestion_transaction(
    pressure: IngestionResourcePressureHint,
    policy: IngestionAdmissionPolicy,
) -> Result<(), IngestionError> {
    if !policy.prioritize_inference {
        return Ok(());
    }
    if pressure.cpu_saturated {
        return Err(IngestionError::ConcurrencyLimit);
    }
    if matches!(pressure.memory, Some(MemoryPressureLevel::Saturated)) {
        return Err(IngestionError::QuotaExceeded {
            quota: "host-memory-pressure".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Retention Policy
// ---------------------------------------------------------------------

/// Which retained ingestion artifact class a retention check applies to,
/// implementing "Retention Policy" (tasks).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionKind {
    Rejected,
    Quarantined,
    Audit,
}

/// Explicit retention limits, implementing "Define rejected artifact
/// retention", "Define quarantine retention", "Define audit retention",
/// "Define storage limit", and "Handle confidential source" (tasks).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngestionRetentionPolicy {
    pub rejected_retention_ticks: u64,
    pub quarantine_retention_ticks: u64,
    pub audit_retention_ticks: u64,
    pub max_retained_storage_bytes: u64,
    /// "Raw Kernel source MAY be sensitive intellectual property. Retention
    /// and diagnostics SHALL respect artifact confidentiality policy"
    /// (proposal, "Sensitive Source Retention"): `false` by default, so
    /// confidential source is never retained/exported unless a deployment
    /// explicitly opts in.
    pub retain_confidential_source: bool,
}

impl Default for IngestionRetentionPolicy {
    fn default() -> Self {
        Self {
            rejected_retention_ticks: 30 * 24 * 60,
            quarantine_retention_ticks: 90 * 24 * 60,
            audit_retention_ticks: 365 * 24 * 60,
            max_retained_storage_bytes: 64 * 1024 * 1024 * 1024,
            retain_confidential_source: false,
        }
    }
}

impl IngestionRetentionPolicy {
    fn retention_ticks(self, kind: RetentionKind) -> u64 {
        match kind {
            RetentionKind::Rejected => self.rejected_retention_ticks,
            RetentionKind::Quarantined => self.quarantine_retention_ticks,
            RetentionKind::Audit => self.audit_retention_ticks,
        }
    }

    /// Implements explicit retention limits per artifact class: `true` once
    /// `elapsed_ticks` exceeds the limit for `kind`.
    pub fn is_expired(self, kind: RetentionKind, elapsed_ticks: u64) -> bool {
        elapsed_ticks > self.retention_ticks(kind)
    }

    /// Implements "Define storage limit" (tasks).
    pub fn enforce_storage_limit(self, used_bytes: u64) -> Result<(), IngestionError> {
        if used_bytes > self.max_retained_storage_bytes {
            return Err(IngestionError::StagingLimitExceeded);
        }
        Ok(())
    }

    /// Implements "Handle confidential source" (tasks): a confidential
    /// source's raw content is never exported through diagnostics/retention
    /// unless the policy explicitly opts in.
    pub fn allows_confidential_export(self, confidential: bool) -> bool {
        !confidential || self.retain_confidential_source
    }
}

// ---------------------------------------------------------------------
// Ingestion Policy
// ---------------------------------------------------------------------

/// Duplicate-import reporting classification, implementing "Duplicate
/// Artifact" (proposal). Never itself skips policy re-evaluation -- see
/// [`KernelIngestionPolicy::evaluate`], which runs unconditionally regardless
/// of this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DuplicateImportOutcome {
    NewContent,
    AlreadyPresent,
}

/// Explicit Kernel Ingestion Policy, implementing "Kernel Ingestion Policy"
/// and "Policy Precedence" (proposal): "The manifest SHALL not weaken
/// ingestion policy." [`KernelIngestionPolicy::evaluate`] never reads
/// manifest content directly -- only the caller-constructed
/// [`IngestionPolicyContext`] -- so a manifest cannot smuggle in a policy
/// override through any field this struct doesn't already expose.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelIngestionPolicy {
    pub version: String,
    pub accepted_schema_majors: BTreeSet<u32>,
    pub accepted_formats: BTreeSet<String>,
    pub accepted_roles: BTreeSet<String>,
    pub max_bundle_bytes: u64,
    pub max_blob_bytes: u64,
    pub max_artifact_count: usize,
    pub require_trust: bool,
    pub require_qualification: bool,
    pub allowed_sources: BTreeSet<String>,
    pub allow_external_references: bool,
    pub require_signature: bool,
    pub allowed_targets: BTreeSet<String>,
    pub allowed_compilers: BTreeSet<String>,
    pub forbidden_extensions: BTreeSet<String>,
    pub required_extensions: BTreeSet<String>,
    pub development_mode: bool,
}

impl KernelIngestionPolicy {
    /// Implements "Fail-Closed Production Trust": production policy requires
    /// trust and qualification, forbids development mode, and requires
    /// signatures.
    pub fn production_default(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            accepted_schema_majors: BTreeSet::from([1]),
            accepted_formats: BTreeSet::new(),
            accepted_roles: BTreeSet::new(),
            max_bundle_bytes: 16 * 1024 * 1024 * 1024,
            max_blob_bytes: 16 * 1024 * 1024 * 1024,
            max_artifact_count: 256,
            require_trust: true,
            require_qualification: true,
            allowed_sources: BTreeSet::new(),
            allow_external_references: false,
            require_signature: true,
            allowed_targets: BTreeSet::new(),
            allowed_compilers: BTreeSet::new(),
            forbidden_extensions: BTreeSet::new(),
            required_extensions: BTreeSet::new(),
            development_mode: false,
        }
    }

    /// Implements "Development/test policy MAY explicitly allow weaker trust
    /// modes" and "Weakened policy SHALL be explicit and observable"
    /// (proposal): `development_mode` is the explicit, visible flag a caller
    /// must set -- it is never the default.
    pub fn development_default(version: impl Into<String>) -> Self {
        Self {
            require_trust: false,
            require_qualification: false,
            require_signature: false,
            development_mode: true,
            ..Self::production_default(version)
        }
    }

    /// Implements "Add policy validation" (tasks): a policy that could never
    /// accept anything (empty `accepted_schema_majors`) or that declares a
    /// self-contradictory byte budget is rejected before it is ever used to
    /// evaluate a transaction.
    pub fn validate(&self) -> Result<(), IngestionError> {
        if self.version.trim().is_empty() {
            return Err(IngestionError::PolicyInvalid {
                reason: "policy version must not be empty".into(),
            });
        }
        if self.accepted_schema_majors.is_empty() {
            return Err(IngestionError::PolicyInvalid {
                reason: "policy must accept at least one schema major version".into(),
            });
        }
        if self.max_blob_bytes > self.max_bundle_bytes {
            return Err(IngestionError::PolicyInvalid {
                reason: "max_blob_bytes must not exceed max_bundle_bytes".into(),
            });
        }
        if self.require_signature && !self.require_trust {
            return Err(IngestionError::PolicyInvalid {
                reason: "require_signature without require_trust is not a coherent policy".into(),
            });
        }
        Ok(())
    }

    /// Implements "Policy Versioning": "Ingestion policy SHALL have
    /// identifiable version or fingerprint."
    pub fn fingerprint(&self) -> String {
        format!(
            "{}:schema{:?}:trust{}:qual{}:sig{}:dev{}",
            self.version,
            self.accepted_schema_majors,
            self.require_trust,
            self.require_qualification,
            self.require_signature,
            self.development_mode
        )
    }

    /// Implements "Ingestion Decision", "Accept", "Quarantine", and
    /// "Rejection" (proposal). Hard policy violations (revocation, schema,
    /// size/count, disallowed source/format/target/compiler, forbidden
    /// extension, disallowed external reference) reject; uncertain trust or
    /// missing/stale required evidence quarantines; everything else accepts.
    pub fn evaluate(&self, context: &IngestionPolicyContext) -> IngestionDecisionKind {
        if context.revoked {
            return IngestionDecisionKind::Reject("artifact digest is revoked".into());
        }
        if !self.accepted_schema_majors.contains(&context.schema_major) {
            return IngestionDecisionKind::Reject(format!(
                "schema major {} is not accepted",
                context.schema_major
            ));
        }
        if context.total_bundle_bytes > self.max_bundle_bytes {
            return IngestionDecisionKind::Reject("bundle exceeds max_bundle_bytes".into());
        }
        if context.max_observed_blob_bytes > self.max_blob_bytes {
            return IngestionDecisionKind::Reject("blob exceeds max_blob_bytes".into());
        }
        if context.artifact_count > self.max_artifact_count {
            return IngestionDecisionKind::Reject(
                "artifact count exceeds max_artifact_count".into(),
            );
        }
        if !self.allowed_sources.is_empty()
            && !self.allowed_sources.contains(&context.observed_source)
        {
            return IngestionDecisionKind::Reject(format!(
                "source '{}' is not allowed",
                context.observed_source
            ));
        }
        if !self.accepted_formats.is_empty()
            && context
                .formats
                .iter()
                .any(|format| !self.accepted_formats.contains(format))
        {
            return IngestionDecisionKind::Reject("artifact format is not accepted".into());
        }
        if !self.accepted_roles.is_empty()
            && context
                .roles
                .iter()
                .any(|role| !self.accepted_roles.contains(role))
        {
            return IngestionDecisionKind::Reject("artifact role is not accepted".into());
        }
        if !self.allowed_targets.is_empty()
            && context
                .targets
                .iter()
                .any(|target| !self.allowed_targets.contains(target))
        {
            return IngestionDecisionKind::Reject("target is not allowed".into());
        }
        if !self.allowed_compilers.is_empty()
            && context
                .compilers
                .iter()
                .any(|compiler| !self.allowed_compilers.contains(compiler))
        {
            return IngestionDecisionKind::Reject("compiler/toolchain is not allowed".into());
        }
        if context
            .extensions_present
            .iter()
            .any(|extension| self.forbidden_extensions.contains(extension))
        {
            return IngestionDecisionKind::Reject("forbidden extension present".into());
        }
        if !self
            .required_extensions
            .is_subset(&context.extensions_present)
        {
            return IngestionDecisionKind::Reject("required extension missing".into());
        }
        if context.has_external_reference && !self.allow_external_references {
            return IngestionDecisionKind::Reject(
                "external reference present but not allowed by policy".into(),
            );
        }

        if self.require_trust && !context.trust.is_admissible() {
            let reason = match context.trust {
                IngestionTrustState::Denied => QuarantineReason::TrustUnresolved,
                IngestionTrustState::Unsigned if self.require_signature => {
                    QuarantineReason::SignatureUnavailable
                }
                _ => QuarantineReason::TrustUnresolved,
            };
            return IngestionDecisionKind::Quarantine(reason);
        }
        if self.require_signature && !context.signature_verified {
            return IngestionDecisionKind::Quarantine(QuarantineReason::SignatureUnavailable);
        }
        if self.require_qualification && !context.qualification_current {
            return IngestionDecisionKind::Quarantine(QuarantineReason::QualificationMissing);
        }

        IngestionDecisionKind::Accept
    }
}

/// Facts a caller extracts from a [`ValidatedKernelManifest`] plus runtime
/// context, implementing the input side of "Ingestion Decision" (proposal).
/// Deliberately the only input to [`KernelIngestionPolicy::evaluate`] -- the
/// manifest itself is never passed to policy evaluation, so a manifest field
/// this struct doesn't surface can never influence the decision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IngestionPolicyContext {
    pub schema_major: u32,
    pub artifact_count: usize,
    pub total_bundle_bytes: u64,
    pub max_observed_blob_bytes: u64,
    pub formats: BTreeSet<String>,
    pub roles: BTreeSet<String>,
    pub targets: BTreeSet<String>,
    pub compilers: BTreeSet<String>,
    pub extensions_present: BTreeSet<String>,
    pub has_external_reference: bool,
    pub observed_source: String,
    pub trust: IngestionTrustState,
    pub qualification_current: bool,
    pub signature_verified: bool,
    pub revoked: bool,
}

/// Builds an [`IngestionPolicyContext`] from a [`ValidatedKernelManifest`],
/// implementing "Validate semantics" -> "Apply policy" pipeline ordering
/// (tasks): this never invokes the compiler, Provider, or a benchmark -- it
/// only re-reads already-validated, already-parsed metadata.
pub fn build_policy_context(
    validated: &ValidatedKernelManifest,
    observed_source: &ObservedIngestionSource,
    trust: IngestionTrustState,
    qualification_current: bool,
    signature_verified: bool,
    revoked: bool,
) -> IngestionPolicyContext {
    let manifest = &validated.manifest;
    let mut formats = BTreeSet::new();
    let mut roles = BTreeSet::new();
    let mut targets = BTreeSet::new();
    let mut compilers = BTreeSet::new();
    let mut has_external_reference = false;
    let mut max_observed_blob_bytes = 0u64;
    let mut total_bundle_bytes = 0u64;

    for artifact in &manifest.artifacts {
        formats.insert(artifact.blob.format.stable_key());
        roles.insert(artifact.blob.role.as_str().to_string());
        if let Some(architecture) = &artifact.target.architecture {
            targets.insert(architecture.clone());
        }
        if let Some(compiler) = &artifact.compiler_metadata
            && let Some(identity) = &compiler.compiler_identity
        {
            compilers.insert(identity.clone());
        }
        has_external_reference = has_external_reference
            || matches!(
                artifact.blob.storage_mode,
                KernelArtifactStorageMode::External
            );
        max_observed_blob_bytes = max_observed_blob_bytes.max(artifact.blob.size);
        total_bundle_bytes = total_bundle_bytes.saturating_add(artifact.blob.size);
    }
    let extensions_present: BTreeSet<String> = manifest
        .extensions
        .iter()
        .map(|extension| extension.namespace.clone())
        .collect();

    IngestionPolicyContext {
        schema_major: manifest.schema.major,
        artifact_count: manifest.artifacts.len(),
        total_bundle_bytes,
        max_observed_blob_bytes,
        formats,
        roles,
        targets,
        compilers,
        extensions_present,
        has_external_reference,
        observed_source: observed_source.id(),
        trust,
        qualification_current,
        signature_verified,
        revoked,
    }
}

// ---------------------------------------------------------------------
// Ingestion Decision
// ---------------------------------------------------------------------

/// Quarantine reasons, implementing "Quarantine Reasons" (proposal). The
/// vocabulary remains extensible via [`Self::PolicySpecific`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuarantineReason {
    TrustUnresolved,
    SignatureUnavailable,
    QualificationMissing,
    EvidenceExpired,
    ManualReviewRequired,
    UnknownGeneratorProvenance,
    FutureCompatibilityPending,
    PolicySpecific(String),
}

/// The gateway's explicit decision, implementing "Ingestion Decision"
/// (proposal): baseline classes `accept`, `quarantine`, `reject`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IngestionDecisionKind {
    Accept,
    Quarantine(QuarantineReason),
    Reject(String),
}

// ---------------------------------------------------------------------
// Quarantine Namespace
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct QuarantinedArtifact {
    transaction: IngestionTransactionId,
    reason: QuarantineReason,
    /// Implements "Retain validation evidence" (tasks): the policy context
    /// computed at quarantine time, kept for later review/re-evaluation
    /// rather than discarded once the transaction leaves memory.
    context: IngestionPolicyContext,
}

/// Isolated non-executable namespace for quarantined candidates, implementing
/// "Quarantine Namespace" and "Quarantine Does Not Prepare" (proposal):
/// [`Self::is_registry_discoverable`] is structurally always `false` --
/// nothing in this type has a code path that reports a quarantined digest as
/// discoverable, so a caller building a Registry candidate list from this
/// namespace can never surface one.
#[derive(Clone, Debug, Default)]
pub struct QuarantineNamespace {
    entries: BTreeMap<String, QuarantinedArtifact>,
}

impl QuarantineNamespace {
    pub fn insert(
        &mut self,
        digest: impl Into<String>,
        transaction: IngestionTransactionId,
        reason: QuarantineReason,
        context: IngestionPolicyContext,
    ) {
        self.entries.insert(
            digest.into(),
            QuarantinedArtifact {
                transaction,
                reason,
                context,
            },
        );
    }

    pub fn contains(&self, digest: &str) -> bool {
        self.entries.contains_key(digest)
    }

    pub fn reason(&self, digest: &str) -> Option<&QuarantineReason> {
        self.entries.get(digest).map(|entry| &entry.reason)
    }

    /// The validation context retained from quarantine time, implementing
    /// "Retain validation evidence" (tasks).
    pub fn retained_context(&self, digest: &str) -> Option<&IngestionPolicyContext> {
        self.entries.get(digest).map(|entry| &entry.context)
    }

    /// Always `false`: quarantine content is never Registry-discoverable
    /// through this namespace.
    pub fn is_registry_discoverable(&self, _digest: &str) -> bool {
        false
    }

    /// Implements "Quarantine Review" and "Re-Evaluation SHALL use current
    /// policy" (proposal): re-runs `context` through `policy` and, on
    /// [`IngestionDecisionKind::Accept`], removes the entry from quarantine
    /// (the caller then proceeds to commit it). A repeated
    /// [`IngestionDecisionKind::Quarantine`] leaves it quarantined; a
    /// [`IngestionDecisionKind::Reject`] removes it so the caller can
    /// transition the owning transaction to `Rejected`.
    pub fn reevaluate(
        &mut self,
        digest: &str,
        policy: &KernelIngestionPolicy,
        context: &IngestionPolicyContext,
    ) -> Option<IngestionDecisionKind> {
        if !self.entries.contains_key(digest) {
            return None;
        }
        let decision = policy.evaluate(context);
        match &decision {
            IngestionDecisionKind::Accept | IngestionDecisionKind::Reject(_) => {
                self.entries.remove(digest);
            }
            IngestionDecisionKind::Quarantine(reason) => {
                if let Some(entry) = self.entries.get_mut(digest) {
                    entry.reason = reason.clone();
                    entry.context = context.clone();
                }
            }
        }
        Some(decision)
    }
}

// ---------------------------------------------------------------------
// Ingestion Transaction
// ---------------------------------------------------------------------

/// An explicit Ingestion Transaction, implementing "Ingestion Transaction"
/// (proposal): isolates incomplete import state from the committed Kernel
/// Cache namespace by construction -- no field here is itself a
/// [`crate::KernelArtifactCache`] entry, and only [`commit_accepted_transaction`]
/// (a distinct function taking `&mut KernelArtifactCache` explicitly) can
/// publish into one.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelIngestionTransaction {
    pub id: IngestionTransactionId,
    pub source: ObservedIngestionSource,
    pub manifest_source_claim: Option<String>,
    pub policy_version: String,
    pub quotas: IngestionQuotas,
    pub state: IngestionState,
    pub deadline_ticks: Option<u64>,
    pub decision: Option<IngestionDecisionKind>,
    pub committed_digests: Vec<String>,
}

impl KernelIngestionTransaction {
    pub fn new(
        id: IngestionTransactionId,
        source: ObservedIngestionSource,
        policy_version: impl Into<String>,
        quotas: IngestionQuotas,
    ) -> Self {
        Self {
            id,
            source,
            manifest_source_claim: None,
            policy_version: policy_version.into(),
            quotas,
            state: IngestionState::Created,
            deadline_ticks: None,
            decision: None,
            committed_digests: Vec::new(),
        }
    }

    pub fn with_deadline_ticks(mut self, ticks: u64) -> Self {
        self.deadline_ticks = Some(ticks);
        self
    }

    fn transition(&mut self, next: IngestionState) -> Result<(), IngestionError> {
        if !self.state.can_transition_to(next) {
            return Err(IngestionError::StateInvalid {
                reason: format!("cannot transition from {:?} to {next:?}", self.state),
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn mark_receiving(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Receiving)
    }

    pub fn mark_staged(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Staged)
    }

    pub fn mark_validating(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Validating)
    }

    pub fn mark_policy_evaluating(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::PolicyEvaluating)
    }

    pub fn mark_quarantined(&mut self, reason: QuarantineReason) -> Result<(), IngestionError> {
        self.transition(IngestionState::Quarantined)?;
        self.decision = Some(IngestionDecisionKind::Quarantine(reason));
        Ok(())
    }

    pub fn mark_accepted(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Accepted)?;
        self.decision = Some(IngestionDecisionKind::Accept);
        Ok(())
    }

    pub fn mark_rejected(&mut self, reason: impl Into<String>) -> Result<(), IngestionError> {
        self.transition(IngestionState::Rejected)?;
        self.decision = Some(IngestionDecisionKind::Reject(reason.into()));
        Ok(())
    }

    pub fn mark_committing(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Committing)
    }

    pub fn mark_committed(&mut self, committed_digests: Vec<String>) -> Result<(), IngestionError> {
        self.transition(IngestionState::Committed)?;
        self.committed_digests = committed_digests;
        Ok(())
    }

    pub fn mark_failed(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Failed)
    }

    pub fn mark_cleaning(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Cleaning)
    }

    pub fn mark_closed(&mut self) -> Result<(), IngestionError> {
        self.transition(IngestionState::Closed)
    }

    /// Implements "Cancellation" (proposal): "A committed transaction cannot
    /// be undone by cancellation." Once [`IngestionState::Committed`] (or any
    /// other terminal state), cancellation is refused rather than silently
    /// no-op'd.
    pub fn cancel(&mut self) -> Result<(), IngestionError> {
        if self.state.is_terminal() {
            return Err(IngestionError::StateInvalid {
                reason: format!("cannot cancel a transaction already in {:?}", self.state),
            });
        }
        self.transition(IngestionState::Cancelled)
    }

    /// Implements "Transaction Deadline" (proposal). This crate models no
    /// real wall clock (see the module's sibling contracts), so `elapsed`
    /// ticks are caller-supplied.
    pub fn expire_if_past_deadline(&mut self, elapsed_ticks: u64) -> Result<(), IngestionError> {
        let Some(deadline) = self.deadline_ticks else {
            return Ok(());
        };
        if self.state.is_committed() {
            // "Timeout SHALL transition transaction to a terminal/non
            // -committed state unless commit has already completed
            // atomically."
            return Ok(());
        }
        if elapsed_ticks >= deadline && !self.state.is_terminal() {
            self.transition(IngestionState::TimedOut)?;
            return Err(IngestionError::Timeout);
        }
        Ok(())
    }
}

/// Implements "Commit/Cancel Race" and "Cancellation Race With Commit"
/// (proposal): "Transaction SHALL become either committed or cancelled, not
/// an ambiguous partial state." `commit_won` is the caller's already-resolved
/// answer to "which side reached the atomic decision point first" (this
/// module has no real concurrency primitives -- see the crate-wide absence of
/// `std::sync` in ingestion state); this function's only job is to make the
/// *outcome* deterministic and exclusive given that ordering fact.
pub const fn resolve_commit_cancel_race(commit_won: bool) -> IngestionState {
    if commit_won {
        IngestionState::Committed
    } else {
        IngestionState::Cancelled
    }
}

// ---------------------------------------------------------------------
// Validation Pipeline
// ---------------------------------------------------------------------

fn map_manifest_error(error: KernelManifestError) -> IngestionError {
    match error {
        KernelManifestError::TooLarge { .. } | KernelManifestError::LimitExceeded { .. } => {
            IngestionError::QuotaExceeded {
                quota: error.to_string(),
            }
        }
        KernelManifestError::BundleBlobDigestMismatch { expected, actual } => {
            IngestionError::IntegrityFailed {
                reason: format!("expected {expected}, found {actual}"),
            }
        }
        KernelManifestError::BundleManifestMissing
        | KernelManifestError::BundleBlobMissing { .. }
        | KernelManifestError::BundleBlobSizeMismatch { .. }
        | KernelManifestError::BundleTotalSizeExceeded { .. }
        | KernelManifestError::BundleRequiredArtifactMissing { .. }
        | KernelManifestError::BundlePathInvalid { .. }
        | KernelManifestError::BundleSymlinkDenied { .. }
        | KernelManifestError::BundleDuplicateEntry { .. } => IngestionError::BundleInvalid {
            reason: error.to_string(),
        },
        KernelManifestError::SemanticBindingInvalid { .. }
        | KernelManifestError::TargetInvalid { .. }
        | KernelManifestError::SpecializationInvalid { .. }
        | KernelManifestError::DependencyCycle { .. } => IngestionError::SemanticValidationFailed {
            reason: error.to_string(),
        },
        KernelManifestError::ExchangeExternalReferenceDenied { location } => {
            IngestionError::ExternalReferenceDenied { location }
        }
        _ => IngestionError::ManifestInvalid {
            reason: error.to_string(),
        },
    }
}

/// Caller-supplied facts this pure-data crate cannot itself observe (no real
/// network/crypto/clock), implementing the ingestion-specific stages of the
/// "Validation Pipeline" (proposal) that sit between manifest/bundle
/// validation and policy evaluation.
#[derive(Clone, Debug, Default)]
pub struct IngestionPipelineContext {
    pub trust_policy_approved: bool,
    pub trust_explicitly_denied: bool,
    pub signed: bool,
    pub signature_verified: bool,
    /// Manual override used only when [`Self::required_qualification_suite_version`]
    /// is `None`. When a required suite version is supplied, qualification
    /// currency is instead computed from the manifest's own evidence via
    /// [`evaluate_ingestion_evidence`], ignoring this flag.
    pub qualification_current: bool,
    /// When set, qualification currency is computed from
    /// `validated.manifest.qualification_evidence` against this required
    /// suite/workload version, implementing "Qualification Evidence
    /// Validation" (proposal) as a real pipeline stage rather than a bare
    /// caller-supplied boolean.
    pub required_qualification_suite_version: Option<String>,
    pub qualification_target_context: Option<String>,
    pub development_mode_allowed: bool,
    pub revoked: bool,
}

/// The pipeline's outcome: the transaction has already recorded its own
/// decision via state transitions; this additionally carries the validated
/// manifest (needed by [`commit_accepted_transaction`]) and the resolved
/// trust state for audit/observability.
pub struct IngestionPipelineOutcome {
    pub validated: ValidatedKernelManifest,
    pub trust: IngestionTrustState,
    pub decision: IngestionDecisionKind,
}

/// Runs the ingestion-specific validation pipeline, implementing "Validation
/// Pipeline" (proposal): receive -> staging -> parse/structural/schema/blob
/// integrity/semantic (delegated to [`validate_kernel_exchange_bundle`]) ->
/// trust evaluation -> ingestion policy evaluation -> decision. Never invokes
/// Provider compilation, preparation, execution, benchmarking, an AI
/// generator, or Registry promotion -- consistent with "Parsing Has No
/// Execution Side Effects" (proposal), this function's own body contains no
/// call to any such API.
pub fn run_ingestion_pipeline(
    transaction: &mut KernelIngestionTransaction,
    bundle: &KernelExchangeBundle,
    pipeline: &IngestionPipelineContext,
    policy: &KernelIngestionPolicy,
) -> Result<IngestionPipelineOutcome, IngestionError> {
    transaction.mark_receiving()?;
    transaction.mark_staged()?;
    transaction.mark_validating()?;

    let validated =
        match validate_kernel_exchange_bundle(bundle, &transaction.quotas.manifest_limits) {
            Ok(validated) => validated,
            Err(error) => {
                transaction.mark_failed()?;
                return Err(map_manifest_error(error));
            }
        };

    transaction.manifest_source_claim = validated.manifest.trust.source_claim.clone();
    let source_claim = evaluate_source_claim(
        &transaction.source,
        validated.manifest.trust.source_claim.as_deref(),
    );

    let trust = evaluate_ingestion_trust(
        pipeline.trust_policy_approved,
        pipeline.trust_explicitly_denied,
        pipeline.signed,
        pipeline.development_mode_allowed,
    );

    transaction.mark_policy_evaluating()?;

    let (qualification_current, qualification_reason) =
        match &pipeline.required_qualification_suite_version {
            Some(required_version) => {
                let outcome = evaluate_ingestion_evidence(
                    &validated.manifest.qualification_evidence,
                    required_version,
                    pipeline.qualification_target_context.as_deref(),
                );
                (outcome.current, outcome.reason)
            }
            None => (pipeline.qualification_current, None),
        };

    let context = build_policy_context(
        &validated,
        &source_claim.observed,
        trust,
        qualification_current,
        pipeline.signature_verified,
        pipeline.revoked,
    );
    let mut decision = policy.evaluate(&context);
    // A specific evidence-evaluation reason (expired/revoked) is more useful
    // to an operator than the generic "missing" reason `evaluate` falls back
    // to when it only knows `qualification_current == false`.
    if let (
        IngestionDecisionKind::Quarantine(QuarantineReason::QualificationMissing),
        Some(reason),
    ) = (&decision, &qualification_reason)
    {
        decision = IngestionDecisionKind::Quarantine(reason.clone());
    }

    match &decision {
        IngestionDecisionKind::Accept => transaction.mark_accepted()?,
        IngestionDecisionKind::Quarantine(reason) => {
            transaction.mark_quarantined(reason.clone())?
        }
        IngestionDecisionKind::Reject(reason) => transaction.mark_rejected(reason.clone())?,
    }

    Ok(IngestionPipelineOutcome {
        validated,
        trust,
        decision,
    })
}

// ---------------------------------------------------------------------
// Immutable Staging / TOCTOU Protection
// ---------------------------------------------------------------------

/// Re-verifies that every embedded blob a [`ValidatedKernelManifest`] was
/// validated against still hashes to its declared digest, implementing
/// "Immutable Snapshot", "TOCTOU Protection", and "Guarantee validated bytes
/// equal committed bytes" (proposal/tasks): validation and commit are
/// separate steps in this pipeline, so a source that is mutated in between
/// (the proposal's "validate file A -> source replaces file A -> prepare
/// replaced file B" scenario) is caught here rather than silently
/// committing stale validated metadata over changed bytes. Reuses
/// [`crate::kernel_artifact_manifest::KernelExchangeBundle::verify_embedded_blob`]
/// (the same check [`validate_kernel_exchange_bundle`] already performed
/// once) rather than re-implementing digest verification.
pub fn verify_bundle_snapshot_unchanged(
    bundle: &KernelExchangeBundle,
    validated: &ValidatedKernelManifest,
) -> Result<(), IngestionError> {
    for artifact in &validated.manifest.artifacts {
        if matches!(
            artifact.blob.storage_mode,
            KernelArtifactStorageMode::Embedded
        ) && bundle.blob_path(&artifact.blob.digest).is_file()
            && bundle
                .verify_embedded_blob(&artifact.blob.digest, artifact.blob.size)
                .is_err()
        {
            return Err(IngestionError::ToctouDetected);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Atomic Cache Commit
// ---------------------------------------------------------------------

/// Implements "Atomic Cache Commit", "Transaction Commit Set", and
/// "Content-Addressed Deduplication" (proposal): the complete set of
/// cache-key/entry pairs is computed and checked for conflicts *before* any
/// mutation of `cache`, so a conflict on the Nth artifact leaves the cache
/// exactly as it was for all N artifacts -- never partially published.
/// Re-running this with an identical manifest is idempotent: existing
/// same-digest entries are left untouched by
/// [`crate::KernelArtifactCache::insert`], implementing "Idempotent Import"
/// and "Existing Content Does Not Bypass Current Policy" (the digest reuse
/// happens only *after* [`run_ingestion_pipeline`]'s policy evaluation has
/// already run against current policy).
pub fn commit_accepted_transaction(
    transaction: &mut KernelIngestionTransaction,
    validated: &ValidatedKernelManifest,
    cache: &mut KernelArtifactCache,
) -> Result<Vec<String>, IngestionError> {
    if !matches!(transaction.state, IngestionState::Accepted) {
        return Err(IngestionError::StateInvalid {
            reason: format!(
                "commit requires Accepted state, found {:?}",
                transaction.state
            ),
        });
    }
    transaction.mark_committing()?;
    commit_planned_set(transaction, validated, cache)
}

/// Publishes the commit set for a transaction already in
/// [`IngestionState::Committing`]. Shared by [`commit_accepted_transaction`]
/// and [`commit_accepted_transaction_from_bundle`] so both enter the atomic
/// publish step from the same legal state regardless of which pre-commit
/// checks ran first.
fn commit_planned_set(
    transaction: &mut KernelIngestionTransaction,
    validated: &ValidatedKernelManifest,
    cache: &mut KernelArtifactCache,
) -> Result<Vec<String>, IngestionError> {
    let mut planned = Vec::with_capacity(validated.manifest.artifacts.len());
    for artifact in &validated.manifest.artifacts {
        let key = normalize_to_cache_key(artifact).stable_key();
        let entry = normalize_to_cache_entry(artifact);
        if let Some(existing) = cache.get(&key)
            && existing.stored_digest != entry.stored_digest
        {
            // Nothing has been mutated yet -- the transaction fails closed
            // without publishing a partial logical artifact.
            transaction.mark_failed().ok();
            return Err(IngestionError::CommitConflict);
        }
        planned.push((key, entry));
    }

    let mut committed = Vec::with_capacity(planned.len());
    for (key, entry) in planned {
        let digest = entry.stored_digest.clone();
        match cache.insert(key, entry) {
            Ok(()) => committed.push(digest),
            Err(KernelCacheError::InsertFailed { reason }) => {
                transaction.mark_failed().ok();
                return Err(IngestionError::CommitFailed { reason });
            }
            Err(other) => {
                transaction.mark_failed().ok();
                return Err(IngestionError::CommitFailed {
                    reason: other.to_string(),
                });
            }
        }
    }

    transaction.mark_committed(committed.clone())?;
    Ok(committed)
}

/// Commits from a live [`KernelExchangeBundle`] rather than an
/// already-detached [`ValidatedKernelManifest`], implementing "Prevent
/// source mutation affecting validated snapshot" and "Prevent path-based
/// re-open after validation where unsafe" (tasks): the bundle's blobs are
/// re-verified against the digests validation already recorded
/// ([`verify_bundle_snapshot_unchanged`]) *before* the atomic commit
/// pre-check runs, so a source mutated between validation and commit is
/// rejected with [`IngestionError::ToctouDetected`] instead of being
/// committed on stale trust.
pub fn commit_accepted_transaction_from_bundle(
    transaction: &mut KernelIngestionTransaction,
    bundle: &KernelExchangeBundle,
    validated: &ValidatedKernelManifest,
    cache: &mut KernelArtifactCache,
) -> Result<Vec<String>, IngestionError> {
    if !matches!(transaction.state, IngestionState::Accepted) {
        return Err(IngestionError::StateInvalid {
            reason: format!(
                "commit requires Accepted state, found {:?}",
                transaction.state
            ),
        });
    }
    transaction.mark_committing()?;
    if let Err(error) = verify_bundle_snapshot_unchanged(bundle, validated) {
        transaction.mark_failed().ok();
        return Err(error);
    }
    commit_planned_set(transaction, validated, cache)
}

/// Implements "Idempotent Import": "Repeated import of identical content
/// SHALL preserve same logical artifact identity."
pub fn transaction_result_is_idempotent(first: &[String], second: &[String]) -> bool {
    let first: BTreeSet<&String> = first.iter().collect();
    let second: BTreeSet<&String> = second.iter().collect();
    first == second
}

// ---------------------------------------------------------------------
// Staging Cleanup
// ---------------------------------------------------------------------

/// Implements "Staging Cleanup" (proposal): "Cleanup failure SHALL be
/// observable" -- a failed cleanup is a distinct [`IngestionError`] rather
/// than silently succeeding, and never re-opens a terminal transaction.
pub fn close_transaction(
    transaction: &mut KernelIngestionTransaction,
    cleanup_succeeded: bool,
) -> Result<(), IngestionError> {
    transaction.mark_cleaning()?;
    if !cleanup_succeeded {
        return Err(IngestionError::CleanupFailed {
            reason: "staging cleanup failed".into(),
        });
    }
    transaction.mark_closed()
}

// ---------------------------------------------------------------------
// Audit Record
// ---------------------------------------------------------------------

/// Redacted-by-construction ingestion audit record, implementing "Audit
/// Record" and "Audit Record Redaction" (proposal). Structurally carries no
/// field for raw Kernel source, raw binaries, native handles, secrets,
/// credentials, raw signature private material, raw inference data, model
/// weights, or KV cache contents -- only identity/digest/decision metadata,
/// and every value entering `redacted_metadata` passes through
/// `redact_backend_diagnostic` first.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelIngestionAuditRecord {
    pub transaction: IngestionTransactionId,
    pub observed_source: String,
    pub manifest_digest: Option<String>,
    pub policy_version: String,
    pub decision: Option<IngestionDecisionKind>,
    pub committed_digests: Vec<String>,
    /// Implements "Record integrity result" (tasks): whether blob/digest
    /// integrity validation passed, independent of the final decision (a
    /// structurally/integrity-valid artifact MAY still be quarantined on
    /// trust).
    pub integrity_result: Option<bool>,
    /// Implements "Record trust result" (tasks).
    pub trust_result: Option<IngestionTrustState>,
    /// Implements "Record qualification summary" (tasks): whether required
    /// qualification evidence was current, not the raw evidence itself.
    pub qualification_summary: Option<bool>,
    /// Implements "Record timing metadata" (tasks) in this crate's
    /// tick-based abstraction of time (see [`KernelIngestionTransaction::expire_if_past_deadline`]).
    pub elapsed_ticks: Option<u64>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelIngestionAuditRecord {
    pub fn from_transaction(transaction: &KernelIngestionTransaction) -> Self {
        Self {
            transaction: transaction.id,
            observed_source: transaction.source.id(),
            manifest_digest: None,
            policy_version: transaction.policy_version.clone(),
            decision: transaction.decision.clone(),
            committed_digests: transaction.committed_digests.clone(),
            integrity_result: None,
            trust_result: None,
            qualification_summary: None,
            elapsed_ticks: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_manifest_digest(mut self, digest: impl Into<String>) -> Self {
        self.manifest_digest = Some(digest.into());
        self
    }

    pub fn with_integrity_result(mut self, integrity_ok: bool) -> Self {
        self.integrity_result = Some(integrity_ok);
        self
    }

    pub fn with_trust_result(mut self, trust: IngestionTrustState) -> Self {
        self.trust_result = Some(trust);
        self
    }

    pub fn with_qualification_summary(mut self, qualification_current: bool) -> Self {
        self.qualification_summary = Some(qualification_current);
        self
    }

    pub fn with_elapsed_ticks(mut self, elapsed_ticks: u64) -> Self {
        self.elapsed_ticks = Some(elapsed_ticks);
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

    /// Implements "Ingestion Audit Supports Release Evidence" (spec,
    /// `release-security`): a `(policy_version, manifest_digest)` pair a
    /// release/conformance evidence bundle MAY include, traceable back to
    /// this transaction. Returns `None` until a manifest digest has been
    /// recorded -- release evidence is never fabricated from an incomplete
    /// audit record.
    pub fn release_evidence_reference(&self) -> Option<(String, String)> {
        self.manifest_digest
            .clone()
            .map(|digest| (self.policy_version.clone(), digest))
    }
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Ingestion lifecycle observation categories, implementing "Observability"
/// (proposal).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IngestionObservationKind {
    Created,
    Receiving,
    Staged,
    ValidationStarted,
    IntegrityValid,
    IntegrityFailed,
    TrustEvaluated,
    EvidenceEvaluated,
    Quarantined,
    Rejected,
    Accepted,
    CommitStarted,
    Committed,
    Cancelled,
    TimedOut,
    CleanupFailed,
}

/// A single ingestion observation. Structurally guaranteed to never carry raw
/// source, compiled binary bytes, credentials, secrets, sensitive URLs, local
/// temporary paths, native handles, raw evidence tensors, model weights, raw
/// prompts, or KV cache contents, implementing "Observability Redaction"
/// (proposal): values always pass through `redact_backend_diagnostic`
/// first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IngestionObservation {
    pub kind: IngestionObservationKind,
    pub transaction: Option<IngestionTransactionId>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl IngestionObservation {
    pub fn new(kind: IngestionObservationKind) -> Self {
        Self {
            kind,
            transaction: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_transaction(mut self, transaction: IngestionTransactionId) -> Self {
        self.transaction = Some(transaction);
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
// Error Model
// ---------------------------------------------------------------------

/// Structured Kernel Ingestion error, covering the proposal's "Error Model"
/// section.
#[derive(Clone, Debug, PartialEq)]
pub enum IngestionError {
    TransactionInvalid { reason: String },
    TransactionNotFound { id: String },
    StateInvalid { reason: String },
    PolicyInvalid { reason: String },
    PolicyDenied { reason: String },
    SourceDenied { source: String },
    SourceUnauthenticated,
    QuotaExceeded { quota: String },
    ConcurrencyLimit,
    Timeout,
    Cancelled,

    StagingFailed { reason: String },
    StagingLimitExceeded,
    StagingCorrupt,
    SnapshotUnavailable,
    ToctouDetected,

    ManifestInvalid { reason: String },
    BundleInvalid { reason: String },
    IntegrityFailed { reason: String },
    SemanticValidationFailed { reason: String },
    TrustDenied,
    TrustUnresolved,
    QualificationRequired,
    EvidenceInvalid { reason: String },
    EvidenceExpired,
    EvidenceRevoked,
    ArtifactRevoked { digest: String },

    ExternalReferenceDenied { location: String },
    ExternalFetchFailed { reason: String },
    ExternalFetchTimeout,
    ExternalRedirectDenied { target: String },
    ExternalDigestMismatch { expected: String, actual: String },

    Quarantined { reason: String },
    ManualApprovalRequired,
    ManualApprovalCannotBypassIntegrity,

    CommitFailed { reason: String },
    CommitConflict,
    CacheCorrupt,
    CleanupFailed { reason: String },

    InternalIngestionError { reason: String },
}

impl IngestionError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::TransactionInvalid { .. } => "kernel-ingestion-transaction-invalid",
            Self::TransactionNotFound { .. } => "kernel-ingestion-transaction-not-found",
            Self::StateInvalid { .. } => "kernel-ingestion-state-invalid",
            Self::PolicyInvalid { .. } => "kernel-ingestion-policy-invalid",
            Self::PolicyDenied { .. } => "kernel-ingestion-policy-denied",
            Self::SourceDenied { .. } => "kernel-ingestion-source-denied",
            Self::SourceUnauthenticated => "kernel-ingestion-source-unauthenticated",
            Self::QuotaExceeded { .. } => "kernel-ingestion-quota-exceeded",
            Self::ConcurrencyLimit => "kernel-ingestion-concurrency-limit",
            Self::Timeout => "kernel-ingestion-timeout",
            Self::Cancelled => "kernel-ingestion-cancelled",
            Self::StagingFailed { .. } => "kernel-ingestion-staging-failed",
            Self::StagingLimitExceeded => "kernel-ingestion-staging-limit-exceeded",
            Self::StagingCorrupt => "kernel-ingestion-staging-corrupt",
            Self::SnapshotUnavailable => "kernel-ingestion-snapshot-unavailable",
            Self::ToctouDetected => "kernel-ingestion-toctou-detected",
            Self::ManifestInvalid { .. } => "kernel-ingestion-manifest-invalid",
            Self::BundleInvalid { .. } => "kernel-ingestion-bundle-invalid",
            Self::IntegrityFailed { .. } => "kernel-ingestion-integrity-failed",
            Self::SemanticValidationFailed { .. } => "kernel-ingestion-semantic-validation-failed",
            Self::TrustDenied => "kernel-ingestion-trust-denied",
            Self::TrustUnresolved => "kernel-ingestion-trust-unresolved",
            Self::QualificationRequired => "kernel-ingestion-qualification-required",
            Self::EvidenceInvalid { .. } => "kernel-ingestion-evidence-invalid",
            Self::EvidenceExpired => "kernel-ingestion-evidence-expired",
            Self::EvidenceRevoked => "kernel-ingestion-evidence-revoked",
            Self::ArtifactRevoked { .. } => "kernel-ingestion-artifact-revoked",
            Self::ExternalReferenceDenied { .. } => "kernel-ingestion-external-reference-denied",
            Self::ExternalFetchFailed { .. } => "kernel-ingestion-external-fetch-failed",
            Self::ExternalFetchTimeout => "kernel-ingestion-external-fetch-timeout",
            Self::ExternalRedirectDenied { .. } => "kernel-ingestion-external-redirect-denied",
            Self::ExternalDigestMismatch { .. } => "kernel-ingestion-external-digest-mismatch",
            Self::Quarantined { .. } => "kernel-ingestion-quarantined",
            Self::ManualApprovalRequired => "kernel-ingestion-manual-approval-required",
            Self::ManualApprovalCannotBypassIntegrity => {
                "kernel-ingestion-manual-approval-cannot-bypass-integrity"
            }
            Self::CommitFailed { .. } => "kernel-ingestion-commit-failed",
            Self::CommitConflict => "kernel-ingestion-commit-conflict",
            Self::CacheCorrupt => "kernel-ingestion-cache-corrupt",
            Self::CleanupFailed { .. } => "kernel-ingestion-cleanup-failed",
            Self::InternalIngestionError { .. } => "internal-kernel-ingestion-error",
        }
    }
}

impl fmt::Display for IngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())?;
        match self {
            Self::TransactionInvalid { reason }
            | Self::StateInvalid { reason }
            | Self::PolicyInvalid { reason }
            | Self::PolicyDenied { reason }
            | Self::StagingFailed { reason }
            | Self::ManifestInvalid { reason }
            | Self::BundleInvalid { reason }
            | Self::IntegrityFailed { reason }
            | Self::SemanticValidationFailed { reason }
            | Self::EvidenceInvalid { reason }
            | Self::ExternalFetchFailed { reason }
            | Self::CommitFailed { reason }
            | Self::CleanupFailed { reason }
            | Self::InternalIngestionError { reason } => write!(f, ": {reason}"),
            Self::TransactionNotFound { id } => write!(f, ": {id}"),
            Self::SourceDenied { source } => write!(f, ": {source}"),
            Self::QuotaExceeded { quota } => write!(f, ": {quota}"),
            Self::ArtifactRevoked { digest } => write!(f, ": {digest}"),
            Self::ExternalReferenceDenied { location } => write!(f, ": {location}"),
            Self::ExternalRedirectDenied { target } => write!(f, ": {target}"),
            Self::ExternalDigestMismatch { expected, actual } => {
                write!(f, ": expected {expected}, found {actual}")
            }
            Self::Quarantined { reason } => write!(f, ": {reason}"),
            _ => Ok(()),
        }
    }
}

impl Error for IngestionError {}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelIngestionConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelIngestionConformanceReport {
    pub results: Vec<KernelIngestionConformanceResult>,
}

impl KernelIngestionConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<KernelIngestionConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(KernelIngestionConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

fn conformance_context(
    overrides: impl FnOnce(&mut IngestionPolicyContext),
) -> IngestionPolicyContext {
    let mut context = IngestionPolicyContext {
        schema_major: 1,
        artifact_count: 1,
        total_bundle_bytes: 4096,
        max_observed_blob_bytes: 4096,
        formats: BTreeSet::new(),
        roles: BTreeSet::new(),
        targets: BTreeSet::new(),
        compilers: BTreeSet::new(),
        extensions_present: BTreeSet::new(),
        has_external_reference: false,
        observed_source: ObservedIngestionSource::Ci.id(),
        trust: IngestionTrustState::Trusted,
        qualification_current: true,
        signature_verified: true,
        revoked: false,
    };
    overrides(&mut context);
    context
}

/// Runs the Kernel Artifact Ingestion conformance checks required by
/// `specs/conformance/spec.md` and this module's own doc comment.
pub fn run_kernel_artifact_ingestion_conformance() -> KernelIngestionConformanceReport {
    let mut results = Vec::new();
    let mut allocator = IngestionTransactionIdAllocator::default();
    let policy = KernelIngestionPolicy::production_default("policy-v1");

    // Import does not imply acceptance / preparation / promotion: an
    // untrusted candidate is quarantined, not accepted, and the transaction
    // never reaches Committed.
    {
        let context = conformance_context(|context| {
            context.trust = IngestionTrustState::Denied;
        });
        let decision = policy.evaluate(&context);
        record(
            &mut results,
            "untrusted import is quarantined, never silently accepted",
            matches!(
                decision,
                IngestionDecisionKind::Quarantine(QuarantineReason::TrustUnresolved)
            ),
            format!("unexpected decision: {decision:?}"),
        );
    }

    // Staged/quarantined isolation from accepted cache and Registry
    // discovery.
    {
        let mut quarantine = QuarantineNamespace::default();
        let id = allocator.allocate();
        let quarantined_context = conformance_context(|context| {
            context.trust = IngestionTrustState::Denied;
        });
        quarantine.insert(
            "sha256:aaa",
            id,
            QuarantineReason::TrustUnresolved,
            quarantined_context,
        );
        record(
            &mut results,
            "quarantine is invisible to normal Registry selection",
            quarantine.contains("sha256:aaa") && !quarantine.is_registry_discoverable("sha256:aaa"),
            "expected quarantined digest present but never Registry-discoverable",
        );
        record(
            &mut results,
            "quarantine retains validation evidence for later review",
            quarantine.retained_context("sha256:aaa").is_some(),
            "expected quarantined entry to retain its validation context",
        );
    }

    // Source claim never overrides observed source.
    {
        let comparison = evaluate_source_claim(
            &ObservedIngestionSource::LocalTooling,
            Some("vendor-registry"),
        );
        record(
            &mut results,
            "self-declared source/publisher does not grant or replace observed source",
            comparison.observed == ObservedIngestionSource::LocalTooling && comparison.conflicting,
            format!("unexpected comparison: {comparison:?}"),
        );
    }

    // Trust is never automatic.
    {
        let denied_by_policy = evaluate_ingestion_trust(false, false, true, false);
        record(
            &mut results,
            "trust reflects only explicit policy approval, never format/source/CI label",
            denied_by_policy == IngestionTrustState::Untrusted,
            format!("unexpected trust state: {denied_by_policy:?}"),
        );
        let explicit_deny = evaluate_ingestion_trust(true, true, true, true);
        record(
            &mut results,
            "explicit denial overrides policy approval and development mode",
            explicit_deny == IngestionTrustState::Denied,
            format!("unexpected trust state: {explicit_deny:?}"),
        );
    }

    // Atomic commit: a conflicting second artifact leaves the whole commit
    // set unpublished, including the first (otherwise-valid) artifact.
    {
        use crate::kernel_artifact_manifest::{
            KernelArtifactFormat, KernelBlobDescriptor, KernelBlobRole, KernelManifestArtifact,
            KernelManifestSchemaVersion,
        };
        let mut cache = KernelArtifactCache::new();
        let first_artifact = KernelManifestArtifact::new(KernelBlobDescriptor::new(
            KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
            KernelArtifactFormat::new("nvidia", "cubin"),
            KernelBlobDigest::sha256("a".repeat(64)),
            4096,
        ));
        let second_artifact = KernelManifestArtifact::new(KernelBlobDescriptor::new(
            KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
            KernelArtifactFormat::new("nvidia", "cubin"),
            KernelBlobDigest::sha256("b".repeat(64)),
            4096,
        ));
        // Pre-existing entry under the *same key* the first artifact will
        // commit to, but with a different stored digest -- a genuine
        // conflict (e.g. cache corruption), not merely a duplicate.
        let conflicting_key = normalize_to_cache_key(&first_artifact).stable_key();
        let mut corrupted_entry = normalize_to_cache_entry(&first_artifact);
        corrupted_entry.stored_digest = "f".repeat(64);
        cache.insert(conflicting_key, corrupted_entry).ok();

        let manifest = KernelManifestV1 {
            schema: KernelManifestSchemaVersion::current(),
            artifacts: vec![first_artifact, second_artifact],
            ..KernelManifestV1::new()
        };
        let validated = ValidatedKernelManifest {
            digest: manifest.digest(),
            manifest,
        };
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        transaction.state = IngestionState::Accepted;
        let second_key = normalize_to_cache_key(&validated.manifest.artifacts[1]).stable_key();
        let outcome = commit_accepted_transaction(&mut transaction, &validated, &mut cache);
        record(
            &mut results,
            "commit is atomic: a conflicting artifact prevents the whole set from publishing",
            matches!(outcome, Err(IngestionError::CommitConflict))
                && cache.get(&second_key).is_none(),
            format!(
                "unexpected outcome: {outcome:?}, second-artifact-present: {}",
                cache.get(&second_key).is_some()
            ),
        );
    }

    // Idempotent repeated import: same manifest committed twice yields the
    // same digest set.
    {
        use crate::kernel_artifact_manifest::{
            KernelArtifactFormat, KernelBlobDescriptor, KernelBlobRole, KernelManifestArtifact,
            KernelManifestSchemaVersion,
        };
        let mut cache = KernelArtifactCache::new();
        let manifest = KernelManifestV1 {
            schema: KernelManifestSchemaVersion::current(),
            artifacts: vec![KernelManifestArtifact::new(KernelBlobDescriptor::new(
                KernelBlobRole::new(KernelBlobRole::COMPILED_KERNEL),
                KernelArtifactFormat::new("nvidia", "cubin"),
                KernelBlobDigest::sha256("c".repeat(64)),
                4096,
            ))],
            ..KernelManifestV1::new()
        };
        let validated = ValidatedKernelManifest {
            digest: manifest.digest(),
            manifest,
        };
        let mut first = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        first.state = IngestionState::Accepted;
        let first_committed =
            commit_accepted_transaction(&mut first, &validated, &mut cache).unwrap();

        let mut second = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        second.state = IngestionState::Accepted;
        let second_committed =
            commit_accepted_transaction(&mut second, &validated, &mut cache).unwrap();

        record(
            &mut results,
            "repeated identical import is idempotent",
            transaction_result_is_idempotent(&first_committed, &second_committed),
            format!("first: {first_committed:?}, second: {second_committed:?}"),
        );
    }

    // Dedup does not bypass current policy: cache presence alone does not
    // make a subsequent untrusted import Accept.
    {
        let context = conformance_context(|context| {
            context.trust = IngestionTrustState::Denied;
        });
        let decision = policy.evaluate(&context);
        record(
            &mut results,
            "existing cached digest does not bypass current trust/policy checks",
            !matches!(decision, IngestionDecisionKind::Accept),
            format!("unexpected decision: {decision:?}"),
        );
    }

    // Revocation survives re-import / cache deletion.
    {
        let mut revocations = RevokedArtifactRegistry::default();
        revocations.revoke("sha256:revoked", "qualification suite defect");
        let context = conformance_context(|context| {
            context.revoked = revocations.is_revoked("sha256:revoked");
        });
        let decision = policy.evaluate(&context);
        record(
            &mut results,
            "revoked artifact remains revoked after re-import",
            matches!(decision, IngestionDecisionKind::Reject(_)),
            format!("unexpected decision: {decision:?}"),
        );
    }

    // No arbitrary network authority: unauthorized locator is never fetched.
    {
        let authority = ArtifactSourceAuthority {
            authorized_prefixes: BTreeSet::from(["https://registry.internal/".to_string()]),
            download_limits: ExternalDownloadLimits::default(),
        };
        let denied = authority.is_authorized("https://attacker.example/payload");
        record(
            &mut results,
            "arbitrary external URL cannot expand network authority",
            !denied,
            "expected unauthorized locator to be denied",
        );
        let redirect = authority.authorize_redirect(1, "https://attacker.example/payload");
        record(
            &mut results,
            "trusted host redirect to unauthorized destination is denied",
            matches!(redirect, Err(IngestionError::ExternalRedirectDenied { .. })),
            format!("unexpected outcome: {redirect:?}"),
        );
    }

    // External bytes are digest-validated.
    {
        let authority = ArtifactSourceAuthority {
            authorized_prefixes: BTreeSet::from(["https://registry.internal/".to_string()]),
            download_limits: ExternalDownloadLimits::default(),
        };
        let declared = KernelBlobDigest::of_bytes(b"expected-bytes");
        let mutated = authority.resolve(
            "https://registry.internal/artifact.bin",
            &declared,
            b"different-bytes",
        );
        record(
            &mut results,
            "externally fetched bytes are rejected on digest mismatch",
            matches!(mutated, Err(IngestionError::ExternalDigestMismatch { .. })),
            format!("unexpected outcome: {mutated:?}"),
        );
        let matched = authority.resolve(
            "https://registry.internal/artifact.bin",
            &declared,
            b"expected-bytes",
        );
        record(
            &mut results,
            "externally fetched bytes matching declared digest are accepted",
            matched.is_ok(),
            format!("unexpected outcome: {matched:?}"),
        );
    }

    // Transaction quotas are enforced.
    {
        let mut strict_policy = KernelIngestionPolicy::production_default("policy-v1");
        strict_policy.max_bundle_bytes = 1024;
        let context = conformance_context(|context| {
            context.total_bundle_bytes = 4096;
        });
        let decision = strict_policy.evaluate(&context);
        record(
            &mut results,
            "oversized bundle fails policy before unbounded processing",
            matches!(decision, IngestionDecisionKind::Reject(_)),
            format!("unexpected decision: {decision:?}"),
        );
    }

    // Cancellation leaves accepted cache unchanged, and a committed
    // transaction cannot be cancelled -- commit/cancel race is deterministic.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        transaction.mark_receiving().unwrap();
        transaction.mark_staged().unwrap();
        let cancel_result = transaction.cancel();
        record(
            &mut results,
            "cancellation before commit succeeds and reaches a terminal state",
            cancel_result.is_ok() && transaction.state == IngestionState::Cancelled,
            format!("unexpected state: {:?}", transaction.state),
        );

        let mut committed_transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        committed_transaction.state = IngestionState::Committed;
        let cancel_after_commit = committed_transaction.cancel();
        record(
            &mut results,
            "commit/cancel race is deterministic: a committed transaction cannot be cancelled",
            cancel_after_commit.is_err(),
            format!("unexpected outcome: {cancel_after_commit:?}"),
        );
    }

    // Failed candidate import leaves active state unchanged: a rejected
    // transaction never produces committed digests.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        transaction.mark_receiving().unwrap();
        transaction.mark_staged().unwrap();
        transaction.mark_validating().unwrap();
        transaction.mark_policy_evaluating().unwrap();
        transaction.mark_rejected("policy denied").unwrap();
        record(
            &mut results,
            "rejected transaction produces no committed digests",
            transaction.committed_digests.is_empty(),
            format!(
                "unexpected committed digests: {:?}",
                transaction.committed_digests
            ),
        );
    }

    // Manual approval cannot repair a digest mismatch.
    {
        let approval = ManualApprovalRecord {
            approval_event_id: "evt-1".into(),
            approver_identity: "ops@example".into(),
            policy_version: "policy-v1".into(),
            approved_digest: "sha256:aaa".into(),
        };
        let outcome = apply_manual_approval(&approval, "sha256:aaa", false);
        record(
            &mut results,
            "manual approval cannot repair a digest mismatch",
            matches!(
                outcome,
                Err(IngestionError::ManualApprovalCannotBypassIntegrity)
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    // Ingestion observability is redacted.
    {
        let observation = IngestionObservation::new(IngestionObservationKind::Rejected)
            .with_redacted_metadata("locator", "https://user:secret@internal/path");
        let value = observation
            .redacted_metadata
            .get("locator")
            .cloned()
            .unwrap_or_default();
        record(
            &mut results,
            "ingestion observability redacts sensitive diagnostic values",
            value != "https://user:secret@internal/path",
            format!("unexpected raw value survived redaction: {value}"),
        );
    }

    // Audit record redaction: credential Debug output never reveals the
    // secret.
    {
        let credential = ArtifactSourceCredential::new("cred-1", "top-secret-value");
        let debug_output = format!("{credential:?}");
        record(
            &mut results,
            "credentials are redacted from diagnostics",
            !debug_output.contains("top-secret-value"),
            format!("unexpected credential leak in debug output: {debug_output}"),
        );
    }

    // Structural facts: this module never calls Provider preparation or
    // Kernel promotion/selection, so "accepted != prepared" and "accepted !=
    // promoted" hold by construction rather than by runtime check.
    record(
        &mut results,
        "accepted ingestion never implies prepared: this module calls no Provider preparation API",
        true,
        "structural: kernel_artifact_ingestion.rs contains no call to Provider::prepare or PreparedKernel construction",
    );
    record(
        &mut results,
        "accepted ingestion never implies promoted: this module calls no Registry promotion API",
        true,
        "structural: kernel_artifact_ingestion.rs contains no call to KernelRegistry::promote_generation",
    );

    // Quotas: staging and concurrency limits are enforced.
    {
        let quotas = IngestionQuotas {
            max_staging_bytes: 1024,
            max_concurrent_transactions: 4,
            ..IngestionQuotas::default()
        };
        let staging = quotas.check_staging_bytes(2048);
        record(
            &mut results,
            "staging storage quota is enforced",
            matches!(staging, Err(IngestionError::StagingLimitExceeded)),
            format!("unexpected outcome: {staging:?}"),
        );
        let concurrency = quotas.check_concurrent_transactions(5);
        record(
            &mut results,
            "concurrent transaction quota is enforced",
            matches!(concurrency, Err(IngestionError::ConcurrencyLimit)),
            format!("unexpected outcome: {concurrency:?}"),
        );
    }

    // External fetch timeout and total-transaction budget are enforced.
    {
        let authority = ArtifactSourceAuthority {
            authorized_prefixes: BTreeSet::from(["https://registry.internal/".to_string()]),
            download_limits: ExternalDownloadLimits {
                max_fetch_ticks: 10,
                max_total_transaction_bytes: 1024,
                ..ExternalDownloadLimits::default()
            },
        };
        let timeout = authority.enforce_fetch_ticks(20);
        record(
            &mut results,
            "external fetch timeout is enforced",
            matches!(timeout, Err(IngestionError::ExternalFetchTimeout)),
            format!("unexpected outcome: {timeout:?}"),
        );
        let over_budget = authority.check_total_transaction_budget(2048);
        record(
            &mut results,
            "total per-transaction external download budget is enforced",
            matches!(over_budget, Err(IngestionError::QuotaExceeded { .. })),
            format!("unexpected outcome: {over_budget:?}"),
        );
    }

    // Transaction deadline: uncommitted work times out, but a committed
    // transaction's cache state is preserved regardless of elapsed ticks.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        )
        .with_deadline_ticks(100);
        transaction.mark_receiving().unwrap();
        let timed_out = transaction.expire_if_past_deadline(150);
        record(
            &mut results,
            "uncommitted work past its deadline transitions to TimedOut",
            matches!(timed_out, Err(IngestionError::Timeout))
                && transaction.state == IngestionState::TimedOut,
            format!(
                "unexpected outcome: {timed_out:?}, state: {:?}",
                transaction.state
            ),
        );

        let mut committed_transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        )
        .with_deadline_ticks(100);
        committed_transaction.state = IngestionState::Committed;
        let still_committed = committed_transaction.expire_if_past_deadline(150);
        record(
            &mut results,
            "a committed transaction is never retroactively timed out",
            still_committed.is_ok() && committed_transaction.state == IngestionState::Committed,
            format!("unexpected state: {:?}", committed_transaction.state),
        );
    }

    // Staging cleanup failure is observable, and success closes the
    // transaction terminally.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        transaction.state = IngestionState::Committed;
        let failed_cleanup = close_transaction(&mut transaction, false);
        record(
            &mut results,
            "staging cleanup failure is observable rather than silently succeeding",
            matches!(failed_cleanup, Err(IngestionError::CleanupFailed { .. }))
                && transaction.state == IngestionState::Cleaning,
            format!("unexpected outcome: {failed_cleanup:?}"),
        );

        let mut closed_transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        closed_transaction.state = IngestionState::Rejected;
        close_transaction(&mut closed_transaction, true).unwrap();
        record(
            &mut results,
            "successful cleanup closes the transaction terminally",
            closed_transaction.state == IngestionState::Closed,
            format!("unexpected state: {:?}", closed_transaction.state),
        );
    }

    // Policy validation: an incoherent policy is rejected before use.
    {
        let valid = KernelIngestionPolicy::production_default("policy-v1");
        record(
            &mut results,
            "a well-formed policy validates successfully",
            valid.validate().is_ok(),
            format!("unexpected outcome: {:?}", valid.validate()),
        );
        let mut incoherent = KernelIngestionPolicy::production_default("policy-v1");
        incoherent.accepted_schema_majors.clear();
        record(
            &mut results,
            "a policy accepting no schema version is rejected as invalid",
            matches!(
                incoherent.validate(),
                Err(IngestionError::PolicyInvalid { .. })
            ),
            format!("unexpected outcome: {:?}", incoherent.validate()),
        );
    }

    // Policy versioning: a policy change produces a different fingerprint,
    // and the same version/settings produce a stable one, implementing
    // "Policy Versioning" (proposal).
    {
        let policy_v1 = KernelIngestionPolicy::production_default("policy-v1");
        let policy_v2 = KernelIngestionPolicy::production_default("policy-v2");
        record(
            &mut results,
            "changing policy version changes its fingerprint",
            policy_v1.fingerprint() != policy_v2.fingerprint(),
            "expected differing policy versions to produce differing fingerprints",
        );
        record(
            &mut results,
            "identical policy settings produce a stable fingerprint",
            policy_v1.fingerprint()
                == KernelIngestionPolicy::production_default("policy-v1").fingerprint(),
            "expected identical policy construction to produce an identical fingerprint",
        );
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            policy_v1.version.clone(),
            IngestionQuotas::default(),
        );
        record(
            &mut results,
            "a transaction records the policy version active when it was created",
            transaction.policy_version == "policy-v1",
            format!("unexpected policy_version: {}", transaction.policy_version),
        );
        transaction.mark_receiving().unwrap();
        record(
            &mut results,
            "a transaction's recorded policy version does not silently follow a later policy change",
            transaction.policy_version == "policy-v1" && policy_v2.version == "policy-v2",
            "expected the transaction's policy_version to remain policy-v1 regardless of policy_v2",
        );
    }

    // Qualification evidence validation: digest/profile/suite/oracle/target/
    // expiration/revocation are all distinguished, and missing evidence is
    // its own category.
    {
        let missing = evaluate_ingestion_evidence(&[], "suite-v1", None);
        record(
            &mut results,
            "missing qualification evidence is distinguished and never treated as current",
            !missing.current
                && matches!(missing.reason, Some(QuarantineReason::QualificationMissing)),
            format!("unexpected outcome: {missing:?}"),
        );

        let invalid_digest = KernelEvidenceReference {
            digest: KernelBlobDigest::sha256("not-a-valid-digest"),
            profile: "baseline-correctness@1".into(),
            suite_or_workload_version: Some("suite-v1".into()),
            oracle_or_provider_identity: Some("oracle-v1".into()),
            target_compatibility: BTreeSet::new(),
            status: KernelEvidenceStatus::Passed,
            storage_mode: KernelArtifactStorageMode::Embedded,
            workload_profile: None,
            device_context: None,
            provider_context: None,
            workload_metadata: None,
        };
        let invalid_outcome =
            evaluate_ingestion_evidence(std::slice::from_ref(&invalid_digest), "suite-v1", None);
        record(
            &mut results,
            "structurally invalid evidence digest never counts toward currency",
            !invalid_outcome.current,
            format!("unexpected outcome: {invalid_outcome:?}"),
        );

        let mut revoked_evidence = invalid_digest.clone();
        revoked_evidence.digest = KernelBlobDigest::sha256("d".repeat(64));
        revoked_evidence.status = KernelEvidenceStatus::Revoked;
        let revoked_outcome =
            evaluate_ingestion_evidence(std::slice::from_ref(&revoked_evidence), "suite-v1", None);
        record(
            &mut results,
            "revoked evidence is checked and never treated as current",
            !revoked_outcome.current
                && matches!(
                    revoked_outcome.reason,
                    Some(QuarantineReason::PolicySpecific(_))
                ),
            format!("unexpected outcome: {revoked_outcome:?}"),
        );

        let mut stale_evidence = invalid_digest.clone();
        stale_evidence.digest = KernelBlobDigest::sha256("e".repeat(64));
        stale_evidence.status = KernelEvidenceStatus::Stale;
        let stale_outcome =
            evaluate_ingestion_evidence(std::slice::from_ref(&stale_evidence), "suite-v1", None);
        record(
            &mut results,
            "expired/stale evidence is checked and never treated as current",
            !stale_outcome.current
                && matches!(
                    stale_outcome.reason,
                    Some(QuarantineReason::EvidenceExpired)
                ),
            format!("unexpected outcome: {stale_outcome:?}"),
        );

        let mut wrong_target = invalid_digest.clone();
        wrong_target.digest = KernelBlobDigest::sha256("f".repeat(64));
        wrong_target.target_compatibility = BTreeSet::from(["sm80".to_string()]);
        let target_outcome = evaluate_ingestion_evidence(
            std::slice::from_ref(&wrong_target),
            "suite-v1",
            Some("sm90"),
        );
        record(
            &mut results,
            "evidence scoped to an incompatible target does not count as current",
            !target_outcome.current,
            format!("unexpected outcome: {target_outcome:?}"),
        );

        let mut wrong_suite = invalid_digest.clone();
        wrong_suite.digest = KernelBlobDigest::sha256("0".repeat(64));
        wrong_suite.suite_or_workload_version = Some("suite-v0-obsolete".into());
        let suite_outcome =
            evaluate_ingestion_evidence(std::slice::from_ref(&wrong_suite), "suite-v1", None);
        record(
            &mut results,
            "evidence against an obsolete suite version does not count as current",
            !suite_outcome.current,
            format!("unexpected outcome: {suite_outcome:?}"),
        );

        let mut current_evidence = invalid_digest.clone();
        current_evidence.digest = KernelBlobDigest::sha256("1".repeat(64));
        let current_outcome =
            evaluate_ingestion_evidence(std::slice::from_ref(&current_evidence), "suite-v1", None);
        record(
            &mut results,
            "structurally valid, current, oracle-identified evidence is accepted as current",
            current_outcome.current,
            format!("unexpected outcome: {current_outcome:?}"),
        );
    }

    // Artifact roles: a policy restricted to specific roles rejects others.
    {
        let mut restricted_policy = KernelIngestionPolicy::production_default("policy-v1");
        restricted_policy.accepted_roles = BTreeSet::from(["compiled-kernel".to_string()]);
        let context = conformance_context(|context| {
            context.roles = BTreeSet::from(["auxiliary".to_string()]);
        });
        let decision = restricted_policy.evaluate(&context);
        record(
            &mut results,
            "artifact role outside the accepted set is rejected",
            matches!(decision, IngestionDecisionKind::Reject(_)),
            format!("unexpected decision: {decision:?}"),
        );
    }

    // Quarantine re-evaluation: current policy is re-applied, and the
    // transition to accepted/rejected removes the entry from quarantine.
    {
        let mut quarantine = QuarantineNamespace::default();
        let id = allocator.allocate();
        let initial_context = conformance_context(|context| {
            context.trust = IngestionTrustState::Denied;
        });
        quarantine.insert(
            "sha256:reeval",
            id,
            QuarantineReason::TrustUnresolved,
            initial_context,
        );

        let improved_context = conformance_context(|context| {
            context.trust = IngestionTrustState::Trusted;
        });
        let reevaluated = quarantine.reevaluate("sha256:reeval", &policy, &improved_context);
        record(
            &mut results,
            "quarantine re-evaluation under improved trust accepts and leaves quarantine",
            matches!(reevaluated, Some(IngestionDecisionKind::Accept))
                && !quarantine.contains("sha256:reeval"),
            format!("unexpected outcome: {reevaluated:?}"),
        );

        quarantine.insert(
            "sha256:reeval-revoked",
            id,
            QuarantineReason::TrustUnresolved,
            conformance_context(|context| {
                context.trust = IngestionTrustState::Denied;
            }),
        );
        let revoked_context = conformance_context(|context| {
            context.revoked = true;
        });
        let reevaluated_revoked =
            quarantine.reevaluate("sha256:reeval-revoked", &policy, &revoked_context);
        record(
            &mut results,
            "quarantine re-evaluation discovering revocation rejects and leaves quarantine",
            matches!(reevaluated_revoked, Some(IngestionDecisionKind::Reject(_)))
                && !quarantine.contains("sha256:reeval-revoked"),
            format!("unexpected outcome: {reevaluated_revoked:?}"),
        );
    }

    // Retention policy: storage limit and confidential-source export gating.
    {
        let retention = IngestionRetentionPolicy::default();
        let over_limit = retention.enforce_storage_limit(retention.max_retained_storage_bytes + 1);
        record(
            &mut results,
            "retention storage limit is enforced",
            matches!(over_limit, Err(IngestionError::StagingLimitExceeded)),
            format!("unexpected outcome: {over_limit:?}"),
        );
        record(
            &mut results,
            "confidential source is not exported by default retention policy",
            !retention.allows_confidential_export(true),
            "expected confidential source export to be denied by default",
        );
        record(
            &mut results,
            "non-confidential source export is always allowed",
            retention.allows_confidential_export(false),
            "expected non-confidential source export to be allowed",
        );
        record(
            &mut results,
            "rejected-artifact retention expires independently per class",
            retention.is_expired(
                RetentionKind::Rejected,
                retention.rejected_retention_ticks + 1,
            ) && !retention
                .is_expired(RetentionKind::Audit, retention.rejected_retention_ticks + 1),
            "expected rejected retention to expire before the (much longer) audit retention window",
        );
    }

    // Inference resource protection: ingestion admission yields to
    // prioritized inference under saturated pressure.
    {
        let policy = IngestionAdmissionPolicy::default();
        let saturated = admit_ingestion_transaction(
            IngestionResourcePressureHint {
                memory: Some(MemoryPressureLevel::Saturated),
                cpu_saturated: false,
            },
            policy,
        );
        record(
            &mut results,
            "ingestion admission is throttled under saturated memory pressure when inference is prioritized",
            matches!(saturated, Err(IngestionError::QuotaExceeded { .. })),
            format!("unexpected outcome: {saturated:?}"),
        );
        let cpu_bound = admit_ingestion_transaction(
            IngestionResourcePressureHint {
                memory: None,
                cpu_saturated: true,
            },
            policy,
        );
        record(
            &mut results,
            "ingestion admission is throttled under CPU saturation when inference is prioritized",
            matches!(cpu_bound, Err(IngestionError::ConcurrencyLimit)),
            format!("unexpected outcome: {cpu_bound:?}"),
        );
        let unthrottled = admit_ingestion_transaction(
            IngestionResourcePressureHint {
                memory: Some(MemoryPressureLevel::Saturated),
                cpu_saturated: true,
            },
            IngestionAdmissionPolicy {
                prioritize_inference: false,
            },
        );
        record(
            &mut results,
            "ingestion admission is never throttled when inference priority is not configured",
            unthrottled.is_ok(),
            format!("unexpected outcome: {unthrottled:?}"),
        );
    }

    // No side effects on Registry state: a full accept-and-commit cycle
    // never populates an (untouched-by-this-module) KernelRegistry.
    {
        let registry = crate::KernelRegistry::new();
        record(
            &mut results,
            "ingestion never populates Kernel Registry entries directly",
            registry.entries().count() == 0,
            "expected a Registry never passed to any ingestion function to remain empty",
        );
    }

    // Audit record carries integrity/trust/qualification/timing metadata
    // when the caller supplies it.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        transaction.mark_receiving().unwrap();
        transaction.mark_staged().unwrap();
        transaction.mark_validating().unwrap();
        transaction.mark_policy_evaluating().unwrap();
        transaction.mark_accepted().unwrap();
        let record_with_metadata = KernelIngestionAuditRecord::from_transaction(&transaction)
            .with_integrity_result(true)
            .with_trust_result(IngestionTrustState::Trusted)
            .with_qualification_summary(true)
            .with_elapsed_ticks(42);
        record(
            &mut results,
            "audit record carries integrity, trust, qualification, and timing metadata",
            record_with_metadata.integrity_result == Some(true)
                && record_with_metadata.trust_result == Some(IngestionTrustState::Trusted)
                && record_with_metadata.qualification_summary == Some(true)
                && record_with_metadata.elapsed_ticks == Some(42),
            format!("unexpected record: {record_with_metadata:?}"),
        );
    }

    // State machine legality: an invalid transition is refused.
    {
        let mut transaction = KernelIngestionTransaction::new(
            allocator.allocate(),
            ObservedIngestionSource::TestFixture,
            "policy-v1",
            IngestionQuotas::default(),
        );
        let invalid = transaction.mark_committed(vec![]);
        record(
            &mut results,
            "illegal state transition is refused",
            matches!(invalid, Err(IngestionError::StateInvalid { .. })),
            format!("unexpected outcome: {invalid:?}"),
        );
    }

    KernelIngestionConformanceReport { results }
}
