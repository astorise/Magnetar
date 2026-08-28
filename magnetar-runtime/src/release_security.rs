//! Release security and supply-chain hardening policy contract (see
//! `openspec/changes/define-release-security-and-supply-chain-hardening`).
//!
//! This module does not implement cryptographic signing, SLSA compliance,
//! production sandboxing, remote registry authentication, model hub
//! security, server authentication, or a legal license-approval process --
//! the proposal's "Non-Goals" rule all of that out explicitly, and it does
//! not make native Providers untrusted sandboxed plugins or guarantee all
//! future Providers are secure. Instead it defines, as executable Rust types
//! and validation functions, the release **security** policy for `v0.1`,
//! composing existing crate contracts (Model/Component/Adapter trust,
//! `cli_boundary`, `inference_api`, `provider_roadmap` handle scopes,
//! `compute::redact_backend_diagnostic`, `release_packaging`) rather than
//! duplicating them:
//!
//! - [`RELEASE_SECURITY_SCOPE_INCLUDED`] / [`RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS`]
//!   / [`reject_hardened_security_claim_for_excluded_feature`]: the `v0.1`
//!   CPU-local security scope and the "Security Scope For v0.1" boundary.
//! - [`DependencyAdvisorySeverity`] / [`DependencyAdvisory`] /
//!   [`DependencyAuditReport`]: the "Dependency Audit" policy -- a critical,
//!   unmitigated advisory in a required dependency blocks stable release.
//! - [`LicenseAuditStatus`] / [`DependencyLicense`] / [`LicenseAuditReport`]:
//!   the "License Audit" policy -- an unapproved incompatible or unknown
//!   license blocks stable release.
//! - [`SbomEntry`] / [`SbomAvailability`] / [`SbomManifest`]: the "SBOM"
//!   policy -- generated SBOM or a documented limitation, never silence.
//! - [`verify_checksum_matches_final_artifact`]: the "Checksums" policy,
//!   reusing [`crate::release_packaging::ArtifactChecksum`] rather than a
//!   parallel checksum type.
//! - [`SignatureStatus`] / [`validate_signature_status`]: the "Signatures"
//!   policy -- signature absence SHALL not be hidden.
//! - [`ReleaseProvenance`]: the "Provenance" policy, reusing
//!   [`crate::release_packaging::redact_build_metadata`] to guarantee no
//!   provenance field is secret- or local-path-shaped.
//! - [`ReproducibilityStatus`] / [`ReproducibilityReport`]: the
//!   "Reproducibility" policy.
//! - [`LockfileState`] / [`reject_unreviewed_lockfile_drift`]: the "Lockfile
//!   Policy".
//! - [`BuildScriptReview`] / [`flag_unexpected_build_script`]: the "Build
//!   Script Policy".
//! - [`SecretScanTarget`] / [`SECRET_SCAN_TARGETS`] / [`SecretScanReport`]:
//!   the "Secret Scanning" policy -- any detected secret blocks stable
//!   release.
//! - [`ArtifactIntegrityStatus`]: the "Artifact Integrity" policy.
//! - [`validate_redaction_gate`] / [`record_release_security_observation`]:
//!   the "Redaction Gates" and "Observability" policies, composing
//!   `crate::compute::redact_backend_diagnostic` with the additional
//!   secret/credential/prompt-shaped content this change's redaction gate
//!   covers.
//! - [`ProviderTrustModel`] / [`DynamicProviderLoadingStatus`] /
//!   [`validate_dynamic_provider_loading_status`] /
//!   [`ProviderTrustSignalSource`] /
//!   [`reject_provider_registration_implies_trust`]: the "Provider Trust
//!   Boundary" policy, composing [`crate::provider::ProviderLoadingMode`].
//! - [`reject_release_native_handle_exposure`][]: the "Native Handle Boundary"
//!   policy, composing
//!   [`crate::release_packaging::reject_release_public_api_handle_exposure`]
//!   and [`crate::provider_roadmap::reject_provider_specific_handle_capability`]
//!   instead of a third forbidden-fragment list.
//! - [`validate_component_release_execution_trust`] /
//!   [`reject_component_release_authority_expansion`]: the "Component
//!   Artifact Trust Boundary" policy, composing
//!   [`crate::component::ComponentTrustDecision`] and
//!   [`crate::inference_api::validate_inference_scope`].
//! - [`validate_model_artifact_release_trust`] / [`FixtureModelTrustPolicy`]
//!   / [`validate_fixture_model_trust`]: the "Model Artifact Trust Boundary"
//!   policy, composing [`crate::model::ModelTrustDecision`].
//! - [`validate_source_cache_release_trust`] / [`NonTrustCacheSignal`] /
//!   [`reject_cache_signal_alone_as_trust`]: the "Source Cache Trust
//!   Boundary" policy, composing
//!   [`crate::model_source_cache_roadmap::CacheEntryMetadata`].
//! - [`validate_cli_authority_not_delegated_to_runtime`]: the "CLI Boundary
//!   Security" policy, composing
//!   [`crate::cli_boundary::reject_cli_owned_authority`].
//! - [`validate_runtime_inference_api_security`]: the "Runtime Inference API
//!   Security" policy, composing
//!   [`crate::inference_api::validate_inference_scope`].
//! - [`UnsafeCodeReview`] / [`UnsafeCodePolicy`] /
//!   [`magnetar_runtime_unsafe_code_inventory`]: the "Unsafe Code Policy",
//!   including the concrete real inventory of every `unsafe` fn in this
//!   crate's required baseline (`crate::provider::ProviderLoader`'s dynamic
//!   loading functions).
//! - [`DependencyFeatureCapability`] / [`DependencyFeatureReview`] /
//!   [`reject_unexpected_capability_expanding_feature`]: the "Dependency
//!   Feature Policy".
//! - [`VulnerabilityHandlingPolicy`]: the "Vulnerability Handling" policy.
//! - [`SecurityReleaseNotes`]: the "Security Notes" policy. Named
//!   `SecurityReleaseNotes` (not `ReleaseSecurityNotes`) because
//!   [`crate::release_packaging::ReleaseSecurityNotes`] already occupies
//!   that name as a deliberately shallow placeholder that defers hardening
//!   detail to this change; this type is that detail.
//! - [`ReleaseSecurityGateInputs`] / [`evaluate_release_security_blocking`]:
//!   the "Release Blocking Criteria" policy.
//! - [`SecurityException`] / [`reject_undocumented_security_exception`]: the
//!   "Security Exceptions" policy.
//! - [`ReleaseSecurityObservationKind`] / [`ReleaseSecurityObservation`] /
//!   [`record_release_security_observation`]: the "Observability" policy.
//! - [`ReleaseSecurityError`]: structured error categories covering every
//!   failure category above.
//! - [`ReleaseSecurityConformanceReport`] / [`run_release_security_conformance`]:
//!   a conformance report, in the shape of
//!   [`crate::ReleasePackagingConformanceReport`], asserting the guarantees
//!   above hold.

use std::{error::Error, fmt};

use crate::{
    ArtifactChecksum, CacheEntryMetadata, ComponentTrustDecision, ComponentTrustStatus,
    ModelTrustDecision, ModelTrustStatus, ProviderLoadingMode,
    cli_boundary::reject_cli_owned_authority,
    compute::redact_backend_diagnostic,
    inference_api::validate_inference_scope,
    provider_roadmap::reject_provider_specific_handle_capability,
    release_packaging::{redact_build_metadata, reject_release_public_api_handle_exposure},
};

pub const RELEASE_SECURITY_POLICY_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Security Scope
// ---------------------------------------------------------------------

/// The `v0.1` CPU-local security scope, implementing "Security Scope For
/// v0.1" -- "It SHALL include" (`proposal.md`).
pub const RELEASE_SECURITY_SCOPE_INCLUDED: &[&str] = &[
    "rust-source",
    "workspace-dependencies",
    "release-binaries",
    "release-reports",
    "openspec-baseline",
    "wit-packages",
    "reference-cpu-provider",
    "fixture-model-artifacts",
    "fixture-tokenizer-artifacts",
    "runtime-inference-api",
    "cli-boundary-harness",
    "e2e-local-conformance",
];

/// Features `v0.1` security notes SHALL not claim hardened production
/// security for, implementing "Security Scope For v0.1" -- "It SHALL not
/// claim hardened production security for" (`proposal.md`). Deliberately
/// separate from [`crate::release_packaging`]'s deferred-roadmap-feature
/// list: that list governs whether a feature may be presented as *included*
/// baseline, this one governs whether a feature may be claimed *hardened*.
pub const RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS: &[&str] = &[
    "cuda",
    "metal",
    "openvino",
    "qnn",
    "webgpu",
    "server-api",
    "model-hub-downloads",
    "remote-registry-authentication",
    "production-sandboxing",
    "agent-tool-runtime",
    "large-third-party-model-execution",
    // Kernel Optimization Orchestration boundary: Magnetar's security
    // boundary does not claim control over arbitrary external optimization
    // infrastructure (see "Optimization Authority Is Outside Runtime
    // Security Claim", `specs/release-security/spec.md`).
    "kernel-optimization-orchestration",
];

/// "CUDA is not claimed as hardened" (`specs/release-security/spec.md`,
/// "v0.1 Security Scope"): rejects claiming hardened security for a feature
/// in [`RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS`].
pub fn reject_hardened_security_claim_for_excluded_feature(
    feature: &str,
    claimed_hardened: bool,
) -> Result<(), ReleaseSecurityError> {
    let normalized = feature.trim().to_ascii_lowercase().replace(' ', "-");
    if claimed_hardened
        && RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS.contains(&normalized.as_str())
    {
        return Err(ReleaseSecurityError::HardenedClaimForExcludedFeature {
            feature: feature.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Dependency Audit
// ---------------------------------------------------------------------

/// Advisory severity, implementing "Dependency Audit" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DependencyAdvisorySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// A single known advisory against a dependency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyAdvisory {
    pub crate_name: String,
    pub advisory_id: String,
    pub severity: DependencyAdvisorySeverity,
    pub mitigated: bool,
    pub mitigation: Option<String>,
}

/// "Dependency audit SHOULD check": known advisories, yanked crates,
/// duplicate high-risk dependencies, unexpected build scripts, unexpected
/// native dependencies, dependency tree drift -- license metadata is
/// intentionally not duplicated here, see [`LicenseAuditReport`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DependencyAuditReport {
    pub advisories: Vec<DependencyAdvisory>,
    pub yanked_crates: Vec<String>,
    pub duplicate_high_risk_dependencies: Vec<String>,
    pub unexpected_build_scripts: Vec<String>,
    pub unexpected_native_dependencies: Vec<String>,
    pub dependency_tree_drift_detected: bool,
}

impl DependencyAuditReport {
    /// "A known critical advisory in a required release dependency SHALL
    /// block stable release unless explicitly accepted with documented
    /// mitigation": the SHALL-strength check this report enforces; the
    /// remaining `SHOULD`-strength fields are recorded but not blocking.
    pub fn validate_for_stable_release(&self) -> Result<(), ReleaseSecurityError> {
        for advisory in &self.advisories {
            if advisory.severity == DependencyAdvisorySeverity::Critical && !advisory.mitigated {
                return Err(ReleaseSecurityError::CriticalAdvisoryUnmitigated {
                    crate_name: advisory.crate_name.clone(),
                    advisory_id: advisory.advisory_id.clone(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// License Audit
// ---------------------------------------------------------------------

/// License audit status, implementing "License Audit" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LicenseAuditStatus {
    Compatible,
    Incompatible,
    Unknown,
    MissingMetadata,
}

/// A single dependency's license audit entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyLicense {
    pub crate_name: String,
    pub spdx: Option<String>,
    pub status: LicenseAuditStatus,
    pub exception_approved: bool,
}

/// "License audit SHOULD identify" dependency licenses, incompatible
/// licenses, unknown licenses, missing license metadata, license
/// exceptions, bundled third-party notices.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LicenseAuditReport {
    pub licenses: Vec<DependencyLicense>,
    pub third_party_notices_generated: bool,
}

impl LicenseAuditReport {
    /// "Unknown or incompatible licenses in required release dependencies
    /// SHALL block stable release unless explicitly approved": missing
    /// metadata alone is recorded but is `SHOULD`-strength, not blocking.
    pub fn validate_for_stable_release(&self) -> Result<(), ReleaseSecurityError> {
        for license in &self.licenses {
            let blocking = matches!(
                license.status,
                LicenseAuditStatus::Incompatible | LicenseAuditStatus::Unknown
            );
            if blocking && !license.exception_approved {
                return Err(ReleaseSecurityError::IncompatibleLicenseUnapproved {
                    crate_name: license.crate_name.clone(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// SBOM
// ---------------------------------------------------------------------

/// A single SBOM entry, implementing "SBOM SHOULD include" (`proposal.md`):
/// package name, package version, dependency license metadata, and source
/// repository metadata where available. The dependency list itself is
/// [`SbomManifest::entries`] -- each dependency is its own [`SbomEntry`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SbomEntry {
    pub package_name: String,
    pub package_version: String,
    pub licenses: Vec<String>,
    pub source_repository: Option<String>,
}

/// Whether an SBOM was generated, implementing "Release SHOULD produce an
/// SBOM or SBOM placeholder".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SbomAvailability {
    Generated,
    PlaceholderDocumented,
    #[default]
    Missing,
}

/// "If full SBOM generation is not implemented for `v0.1`, release notes
/// SHALL state the limitation": [`SbomManifest::validate`] requires either a
/// non-empty generated SBOM or a documented limitation note, never silent
/// absence. `build_target` and `feature_flags` carry the two manifest-level
/// fields from "SBOM SHOULD include" that apply to the whole release rather
/// than to a single dependency entry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SbomManifest {
    pub availability: SbomAvailability,
    pub limitation_note: Option<String>,
    pub entries: Vec<SbomEntry>,
    pub build_target: Option<String>,
    pub feature_flags: Vec<String>,
}

impl SbomManifest {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        match self.availability {
            SbomAvailability::Generated if !self.entries.is_empty() => Ok(()),
            SbomAvailability::PlaceholderDocumented
                if self
                    .limitation_note
                    .as_deref()
                    .is_some_and(|note| !note.trim().is_empty()) =>
            {
                Ok(())
            }
            _ => Err(ReleaseSecurityError::SbomMissingOrUndocumented),
        }
    }
}

// ---------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------

/// "Checksums SHALL be generated from final release artifacts": compares a
/// declared [`ArtifactChecksum`] against a digest recomputed from the final
/// artifact, reusing [`ArtifactChecksum`] rather than a parallel type.
pub fn verify_checksum_matches_final_artifact(
    checksum: &ArtifactChecksum,
    recomputed_digest: &str,
) -> Result<(), ReleaseSecurityError> {
    if checksum.digest == recomputed_digest {
        Ok(())
    } else {
        Err(ReleaseSecurityError::ChecksumMismatch {
            artifact: checksum.artifact.clone(),
        })
    }
}

// ---------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------

/// Signature availability, implementing "Signatures" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureStatus {
    Implemented,
    NotImplementedDocumented,
    NotImplementedUndocumented,
}

/// "Signature absence SHALL not be hidden": rejects
/// [`SignatureStatus::NotImplementedUndocumented`].
pub fn validate_signature_status(status: SignatureStatus) -> Result<(), ReleaseSecurityError> {
    if matches!(status, SignatureStatus::NotImplementedUndocumented) {
        Err(ReleaseSecurityError::SignatureAbsenceUndocumented)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------

/// Provenance metadata fields from "Provenance MAY include" (`proposal.md`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseProvenance {
    pub source_commit: Option<String>,
    pub release_tag: Option<String>,
    pub ci_run_id: Option<String>,
    pub build_target: Option<String>,
    pub build_profile: Option<String>,
    pub rustc_version: Option<String>,
    pub lockfile_digest: Option<String>,
    pub openspec_baseline_digest: Option<String>,
    pub wit_package_digest: Option<String>,
    pub conformance_report_digest: Option<String>,
}

impl ReleaseProvenance {
    fn fields(&self) -> [(&'static str, &Option<String>); 10] {
        [
            ("source_commit", &self.source_commit),
            ("release_tag", &self.release_tag),
            ("ci_run_id", &self.ci_run_id),
            ("build_target", &self.build_target),
            ("build_profile", &self.build_profile),
            ("rustc_version", &self.rustc_version),
            ("lockfile_digest", &self.lockfile_digest),
            ("openspec_baseline_digest", &self.openspec_baseline_digest),
            ("wit_package_digest", &self.wit_package_digest),
            ("conformance_report_digest", &self.conformance_report_digest),
        ]
    }

    /// "Provenance SHALL not include secrets or local developer paths by
    /// default": reuses [`redact_build_metadata`] (the same helper
    /// [`crate::release_packaging::redact_build_metadata`] uses for build
    /// metadata) rather than a parallel redaction rule, and rejects any
    /// populated field that redaction would have changed.
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        for (key, value) in self.fields() {
            if let Some(value) = value {
                let sanitized = redact_build_metadata(key, value);
                if &sanitized != value {
                    return Err(ReleaseSecurityError::ProvenanceContainsSecretOrLocalPath {
                        field: key.to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------

/// Reproducibility status, implementing "Reproducibility" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReproducibilityStatus {
    FullyReproducible,
    PartiallyReproducible,
    NotDocumented,
}

/// "If builds are not fully reproducible, release notes SHALL state
/// limitations": [`ReproducibilityReport::validate`] requires either a fully
/// reproducible status or documented limitations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReproducibilityReport {
    pub status: ReproducibilityStatus,
    pub limitations: Vec<String>,
}

impl Default for ReproducibilityReport {
    fn default() -> Self {
        Self {
            status: ReproducibilityStatus::NotDocumented,
            limitations: Vec::new(),
        }
    }
}

impl ReproducibilityReport {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        match self.status {
            ReproducibilityStatus::NotDocumented => {
                Err(ReleaseSecurityError::ReproducibilityUndocumented)
            }
            ReproducibilityStatus::PartiallyReproducible if self.limitations.is_empty() => {
                Err(ReleaseSecurityError::ReproducibilityUndocumented)
            }
            _ => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------
// Lockfile Policy
// ---------------------------------------------------------------------

/// Lockfile state, implementing "Lockfile Policy" (`proposal.md`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LockfileState {
    pub checked_in: bool,
    pub digest: Option<String>,
    pub drift_detected: bool,
    pub drift_reviewed: bool,
}

/// "Release SHALL use a checked-in dependency lockfile where appropriate"
/// and "Unreviewed lockfile drift SHOULD block release candidates".
pub fn reject_unreviewed_lockfile_drift(state: &LockfileState) -> Result<(), ReleaseSecurityError> {
    if !state.checked_in {
        return Err(ReleaseSecurityError::LockfileNotCheckedIn);
    }
    if state.drift_detected && !state.drift_reviewed {
        return Err(ReleaseSecurityError::LockfileDriftUnreviewed);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Build Script Policy
// ---------------------------------------------------------------------

/// A single dependency's build script review, implementing "Build Script
/// Policy" (`proposal.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildScriptReview {
    pub crate_name: String,
    pub has_build_script: bool,
    pub unexpected: bool,
    pub reviewed: bool,
    pub native_build_documented: bool,
}

/// "Unexpected build scripts in new dependencies SHOULD be flagged":
/// rejects an unexpected, unreviewed build script.
pub fn flag_unexpected_build_script(
    review: &BuildScriptReview,
) -> Result<(), ReleaseSecurityError> {
    if review.has_build_script && review.unexpected && !review.reviewed {
        return Err(ReleaseSecurityError::UnexpectedBuildScriptUnreviewed {
            crate_name: review.crate_name.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Secret Scanning
// ---------------------------------------------------------------------

/// Secret scan targets from "Secret scanning SHOULD check" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SecretScanTarget {
    SourceFiles,
    GeneratedDocs,
    ReleaseNotes,
    ConformanceReports,
    E2eReports,
    Logs,
    BuildMetadata,
    PackagedArtifacts,
}

pub const SECRET_SCAN_TARGETS: &[SecretScanTarget] = &[
    SecretScanTarget::SourceFiles,
    SecretScanTarget::GeneratedDocs,
    SecretScanTarget::ReleaseNotes,
    SecretScanTarget::ConformanceReports,
    SecretScanTarget::E2eReports,
    SecretScanTarget::Logs,
    SecretScanTarget::BuildMetadata,
    SecretScanTarget::PackagedArtifacts,
];

pub const fn secret_scan_target_id(target: SecretScanTarget) -> &'static str {
    match target {
        SecretScanTarget::SourceFiles => "source-files",
        SecretScanTarget::GeneratedDocs => "generated-docs",
        SecretScanTarget::ReleaseNotes => "release-notes",
        SecretScanTarget::ConformanceReports => "conformance-reports",
        SecretScanTarget::E2eReports => "e2e-reports",
        SecretScanTarget::Logs => "logs",
        SecretScanTarget::BuildMetadata => "build-metadata",
        SecretScanTarget::PackagedArtifacts => "packaged-artifacts",
    }
}

/// A single secret scan finding for one [`SecretScanTarget`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecretScanFinding {
    pub target: SecretScanTarget,
    pub detected: bool,
    pub location: Option<String>,
}

/// "Detected secrets SHALL block stable release."
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SecretScanReport {
    pub findings: Vec<SecretScanFinding>,
}

impl SecretScanReport {
    pub fn validate_for_stable_release(&self) -> Result<(), ReleaseSecurityError> {
        for finding in &self.findings {
            if finding.detected {
                return Err(ReleaseSecurityError::SecretDetected {
                    target: secret_scan_target_id(finding.target).to_string(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Artifact Integrity
// ---------------------------------------------------------------------

/// "Artifact integrity SHOULD include" the five checks below, implementing
/// "Artifact Integrity" (`proposal.md`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArtifactIntegrityStatus {
    pub source_state_clean_or_ci_controlled: bool,
    pub release_tag_matches_source: bool,
    pub openspec_report_matches_baseline: bool,
    pub conformance_reports_match_commit: bool,
    pub checksums_match_final_artifacts: bool,
}

impl ArtifactIntegrityStatus {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        let checks: [(&str, bool); 5] = [
            (
                "source-state-clean-or-ci-controlled",
                self.source_state_clean_or_ci_controlled,
            ),
            (
                "release-tag-matches-source",
                self.release_tag_matches_source,
            ),
            (
                "openspec-report-matches-baseline",
                self.openspec_report_matches_baseline,
            ),
            (
                "conformance-reports-match-commit",
                self.conformance_reports_match_commit,
            ),
            (
                "checksums-match-final-artifacts",
                self.checksums_match_final_artifacts,
            ),
        ];
        for (check, ok) in checks {
            if !ok {
                return Err(ReleaseSecurityError::ArtifactIntegrityFailed {
                    check: check.to_string(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Redaction Gates & Sensitive Content
// ---------------------------------------------------------------------

/// The thirteen categories a redaction gate SHALL verify are absent by
/// default, implementing "Redaction Gates" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RedactionCategory {
    RawPrompt,
    Secret,
    Credential,
    RawFileContents,
    RawModelWeights,
    RawTensorValues,
    RawKvCacheContents,
    RawProviderHandle,
    RawDeviceHandle,
    RawKernelHandle,
    RawMemoryPointer,
    LocalFilesystemPath,
    RawCachePath,
}

pub const REDACTION_CATEGORIES: &[RedactionCategory] = &[
    RedactionCategory::RawPrompt,
    RedactionCategory::Secret,
    RedactionCategory::Credential,
    RedactionCategory::RawFileContents,
    RedactionCategory::RawModelWeights,
    RedactionCategory::RawTensorValues,
    RedactionCategory::RawKvCacheContents,
    RedactionCategory::RawProviderHandle,
    RedactionCategory::RawDeviceHandle,
    RedactionCategory::RawKernelHandle,
    RedactionCategory::RawMemoryPointer,
    RedactionCategory::LocalFilesystemPath,
    RedactionCategory::RawCachePath,
];

/// Textual fragments `redact_backend_diagnostic` does not already cover
/// (it only recognizes `0x`/`handle=`-shaped native handles and `\`/`/`-
/// shaped paths): prompt-, secret-, credential-, weight-, tensor-, KV-cache-,
/// and file-content-shaped diagnostic text.
const RELEASE_SECURITY_SENSITIVE_CONTENT_FRAGMENTS: &[&str] = &[
    "raw prompt",
    "prompt=",
    "secret",
    "credential",
    "password",
    "api_key",
    "apikey",
    "token=",
    "raw model weight",
    "raw tensor",
    "raw kv cache",
    "raw file content",
];

/// Redacts `raw` for release security observability/diagnostics: first
/// applies `redact_backend_diagnostic` (native handles, local paths), then
/// checks the additional sensitive-content fragments this change's
/// redaction gate covers.
fn redact_release_security_detail(raw: &str) -> String {
    let backend_redacted = redact_backend_diagnostic(raw);
    if backend_redacted != raw {
        return backend_redacted;
    }
    let normalized = raw.to_ascii_lowercase();
    if RELEASE_SECURITY_SENSITIVE_CONTENT_FRAGMENTS
        .iter()
        .any(|fragment| normalized.contains(fragment))
    {
        "[redacted release security detail]".into()
    } else {
        raw.to_string()
    }
}

/// "Stable release SHALL pass redaction gates": rejects a diagnostic that
/// `redact_release_security_detail` would have had to redact, implementing
/// every scenario shaped like "Given diagnostics include raw prompt by
/// default; When release gate runs; Then release is blocked."
pub fn validate_redaction_gate(diagnostic: &str) -> Result<(), ReleaseSecurityError> {
    let redacted = redact_release_security_detail(diagnostic);
    if redacted == diagnostic {
        Ok(())
    } else {
        Err(ReleaseSecurityError::RedactionGateFailed {
            diagnostic: redacted,
        })
    }
}

// ---------------------------------------------------------------------
// Provider Trust Boundary
// ---------------------------------------------------------------------

/// "Release security notes SHALL state that Providers are trusted native
/// code": a documentation-shaped record, not a negotiable field -- Providers
/// are always trusted native code in this policy, implementing "Provider
/// Trust Boundary" (`proposal.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderTrustModel {
    pub native_code_trusted: bool,
    pub reference_cpu_provider_included: bool,
}

impl Default for ProviderTrustModel {
    fn default() -> Self {
        Self {
            native_code_trusted: true,
            reference_cpu_provider_included: true,
        }
    }
}

/// Dynamic Provider loading status, implementing "Dynamic Provider loading,
/// if present, SHALL be disabled, experimental, or clearly marked unstable
/// unless security reviewed."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicProviderLoadingStatus {
    Disabled,
    Experimental,
    MarkedUnstable,
    SecurityReviewed,
    StableUnreviewed,
}

/// Rejects [`DynamicProviderLoadingStatus::StableUnreviewed`] for a
/// dynamically loaded [`ProviderLoadingMode`] -- every other status is an
/// accepted way to present dynamic Provider loading.
pub fn validate_dynamic_provider_loading_status(
    mode: ProviderLoadingMode,
    status: DynamicProviderLoadingStatus,
) -> Result<(), ReleaseSecurityError> {
    if mode.is_dynamic() && matches!(status, DynamicProviderLoadingStatus::StableUnreviewed) {
        return Err(ReleaseSecurityError::DynamicProviderLoadingUnreviewed);
    }
    Ok(())
}

/// Where a Provider's trust decision came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderTrustSignalSource {
    ConfiguredPolicy,
    RegistrationOnly,
}

/// "Provider registration SHALL not imply Provider trust beyond configured
/// policy": rejects a trust decision derived purely from
/// [`ProviderTrustSignalSource::RegistrationOnly`].
pub fn reject_provider_registration_implies_trust(
    source: ProviderTrustSignalSource,
) -> Result<(), ReleaseSecurityError> {
    if matches!(source, ProviderTrustSignalSource::RegistrationOnly) {
        return Err(ReleaseSecurityError::ProviderRegistrationTrustImplied);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Native Handle Boundary
// ---------------------------------------------------------------------

/// Additional native-handle-shaped fragments not already covered by
/// [`reject_release_public_api_handle_exposure`] or
/// [`reject_provider_specific_handle_capability`] -- "raw CPU allocation
/// pointer" from "Native handle examples" (`proposal.md`).
const RELEASE_SECURITY_ADDITIONAL_HANDLE_FRAGMENTS: &[&str] = &["raw-cpu-allocation-pointer"];

/// "Release public APIs, diagnostics, and reports SHALL not expose native
/// Provider, Device, Kernel, tensor, or memory handles": composes
/// [`reject_release_public_api_handle_exposure`] (generic Provider/Device/
/// Kernel/tensor/memory fragments) and
/// [`reject_provider_specific_handle_capability`] (CUDA/Metal/OpenVINO/QNN
/// fragments) instead of a third forbidden-fragment list.
pub fn reject_release_native_handle_exposure(surface: &str) -> Result<(), ReleaseSecurityError> {
    if let Err(error) = reject_release_public_api_handle_exposure(surface) {
        return Err(ReleaseSecurityError::NativeHandleExposureDenied {
            surface: surface.to_string(),
            reason: error.to_string(),
        });
    }
    if let Err(error) = reject_provider_specific_handle_capability(surface) {
        return Err(ReleaseSecurityError::NativeHandleExposureDenied {
            surface: surface.to_string(),
            reason: error.to_string(),
        });
    }
    let normalized = surface.trim().to_ascii_lowercase();
    if RELEASE_SECURITY_ADDITIONAL_HANDLE_FRAGMENTS
        .iter()
        .any(|fragment| normalized.contains(fragment))
    {
        return Err(ReleaseSecurityError::NativeHandleExposureDenied {
            surface: surface.to_string(),
            reason: "raw CPU allocation pointer exposure denied".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Component Artifact Trust Boundary
// ---------------------------------------------------------------------

/// "Component Artifacts SHALL be validated before execution" and "Unsigned
/// Component Artifacts SHALL be denied in production policy unless
/// explicitly allowed for development/test": composes
/// [`ComponentTrustDecision`] rather than a parallel trust type.
pub fn validate_component_release_execution_trust(
    decision: &ComponentTrustDecision,
    signed: bool,
    production_policy: bool,
    explicitly_allowed_unsigned: bool,
) -> Result<(), ReleaseSecurityError> {
    if decision.status != ComponentTrustStatus::Trusted {
        return Err(ReleaseSecurityError::ComponentArtifactUntrusted {
            reason: decision.reason.clone(),
        });
    }
    if production_policy && !signed && !explicitly_allowed_unsigned {
        return Err(ReleaseSecurityError::UnsignedComponentDeniedInProduction);
    }
    Ok(())
}

/// "Component execution in release builds SHALL not gain filesystem,
/// network, secret, shell, process, Git, tool, Provider handle, Device
/// handle, Kernel handle, or raw tensor pointer authority unless explicitly
/// authorized by inference-scoped contracts": composes
/// [`validate_inference_scope`] (OS-capability authority) and
/// [`reject_release_native_handle_exposure`] (native handle authority)
/// against a [`crate::component::ComponentAuthorityRequirement`]'s `kind`
/// string, rather than a parallel Component-specific forbidden list.
pub fn reject_component_release_authority_expansion(
    capability: &str,
) -> Result<(), ReleaseSecurityError> {
    if let Err(error) = validate_inference_scope(capability) {
        return Err(ReleaseSecurityError::ComponentAuthorityExpansionDenied {
            capability: capability.to_string(),
            reason: error.to_string(),
        });
    }
    reject_release_native_handle_exposure(capability).map_err(|error| {
        ReleaseSecurityError::ComponentAuthorityExpansionDenied {
            capability: capability.to_string(),
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Model Artifact Trust Boundary
// ---------------------------------------------------------------------

/// "Model Artifacts SHALL pass trust and integrity validation before
/// loading" and "Recognized format SHALL not imply trust": `recognized_format`
/// is accepted only to make that non-influence explicit at call sites -- it
/// never changes the outcome, which depends solely on `decision.status`.
pub fn validate_model_artifact_release_trust(
    decision: &ModelTrustDecision,
    recognized_format: bool,
) -> Result<(), ReleaseSecurityError> {
    let _ = recognized_format;
    if decision.status != ModelTrustStatus::Trusted {
        return Err(ReleaseSecurityError::ModelArtifactUntrusted {
            reason: decision.reason.clone(),
        });
    }
    Ok(())
}

/// "Fixture artifacts SHALL have explicit test trust policy."
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FixtureModelTrustPolicy {
    pub explicit_test_policy_documented: bool,
}

/// Requires [`FixtureModelTrustPolicy::explicit_test_policy_documented`]
/// before delegating to [`validate_model_artifact_release_trust`] --
/// fixture status alone is never sufficient.
pub fn validate_fixture_model_trust(
    decision: &ModelTrustDecision,
    fixture_policy: &FixtureModelTrustPolicy,
) -> Result<(), ReleaseSecurityError> {
    if !fixture_policy.explicit_test_policy_documented {
        return Err(ReleaseSecurityError::FixtureTrustPolicyUndocumented);
    }
    validate_model_artifact_release_trust(decision, true)
}

// ---------------------------------------------------------------------
// Source Cache Trust Boundary
// ---------------------------------------------------------------------

/// "Cache presence SHALL not imply trust": requires
/// [`CacheEntryMetadata::trust_status`] to be
/// [`ModelTrustStatus::Trusted`] regardless of source kind, alias, lifecycle,
/// or pin state.
pub fn validate_source_cache_release_trust(
    entry: &CacheEntryMetadata,
) -> Result<(), ReleaseSecurityError> {
    if entry.trust_status != ModelTrustStatus::Trusted {
        return Err(ReleaseSecurityError::CacheEntryTrustNotEstablished {
            artifact: entry.identity.digest.value.clone(),
        });
    }
    Ok(())
}

/// Signals that SHALL NOT by themselves imply trust, implementing "Source
/// Cache Trust Boundary" -- "cache hit != trusted", "source kind !=
/// trusted", "alias != trusted", "local file != trusted", "fixture !=
/// trusted unless test policy" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NonTrustCacheSignal {
    CacheHit,
    SourceKind,
    Alias,
    LocalFile,
    FixtureStatus,
}

/// Asserts that `signal` being present on `entry` does not, by itself,
/// satisfy [`validate_source_cache_release_trust`] -- an explicit
/// [`ModelTrustStatus::Trusted`] decision is still required.
pub fn reject_cache_signal_alone_as_trust(
    signal: NonTrustCacheSignal,
    entry: &CacheEntryMetadata,
) -> Result<(), ReleaseSecurityError> {
    validate_source_cache_release_trust(entry).map_err(|_| {
        ReleaseSecurityError::CacheSignalDoesNotImplyTrust {
            signal: format!("{signal:?}"),
        }
    })
}

// ---------------------------------------------------------------------
// CLI Boundary Security
// ---------------------------------------------------------------------

/// "CLI authority SHALL not become Runtime ambient authority": composes
/// [`reject_cli_owned_authority`] rather than a parallel capability list.
pub fn validate_cli_authority_not_delegated_to_runtime(
    capability: &str,
) -> Result<(), ReleaseSecurityError> {
    reject_cli_owned_authority(capability).map_err(|error| {
        ReleaseSecurityError::CliAuthorityDelegatedToRuntime {
            capability: capability.to_string(),
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Runtime Inference API Security
// ---------------------------------------------------------------------

/// "Runtime Inference API SHALL remain inference-only": composes
/// [`validate_inference_scope`] rather than a parallel capability list.
pub fn validate_runtime_inference_api_security(
    capability: &str,
) -> Result<(), ReleaseSecurityError> {
    validate_inference_scope(capability).map_err(|error| {
        ReleaseSecurityError::RuntimeInferenceApiAuthorityExpansionDenied {
            capability: capability.to_string(),
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Unsafe Code Policy
// ---------------------------------------------------------------------

/// A single reviewed (or unreviewed) `unsafe` usage site, implementing
/// "Unsafe Code Policy" (`proposal.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsafeCodeReview {
    pub location: String,
    pub justified: bool,
    pub reviewed: bool,
}

/// "Unsafe code MAY be denied in release gates unless explicitly allowed":
/// `deny_unreviewed` toggles whether [`UnsafeCodePolicy::validate`] enforces
/// that every review is reviewed and justified.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnsafeCodePolicy {
    pub reviews: Vec<UnsafeCodeReview>,
    pub deny_unreviewed: bool,
}

impl UnsafeCodePolicy {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        if !self.deny_unreviewed {
            return Ok(());
        }
        for review in &self.reviews {
            if !review.reviewed || !review.justified {
                return Err(ReleaseSecurityError::UnsafeCodeUnreviewed {
                    location: review.location.clone(),
                });
            }
        }
        Ok(())
    }
}

/// The concrete `unsafe` inventory for the `v0.1` required baseline,
/// implementing "Detect unsafe Rust usage" / "Review unsafe blocks where
/// present" / "Document unsafe rationale" (`tasks.md`). As of this change,
/// the only `unsafe` surface in `magnetar-runtime`'s required baseline is
/// `ProviderLoader::load_dynamic`, `ProviderLoader::load_dynamic_with_policy`,
/// and `ProviderLoader::discover_and_load` in
/// [`crate::provider`] -- each already carries a `# Safety` doc comment
/// justifying it (dynamic Provider loading is inherently an FFI/native-code
/// boundary) and is call-site-gated by
/// [`crate::provider::ProviderLoadingPolicy::allows`].
pub fn magnetar_runtime_unsafe_code_inventory() -> UnsafeCodePolicy {
    UnsafeCodePolicy {
        reviews: vec![
            UnsafeCodeReview {
                location: "magnetar-runtime/src/provider.rs: ProviderLoader::load_dynamic".into(),
                justified: true,
                reviewed: true,
            },
            UnsafeCodeReview {
                location:
                    "magnetar-runtime/src/provider.rs: ProviderLoader::load_dynamic_with_policy"
                        .into(),
                justified: true,
                reviewed: true,
            },
            UnsafeCodeReview {
                location: "magnetar-runtime/src/provider.rs: ProviderLoader::discover_and_load"
                    .into(),
                justified: true,
                reviewed: true,
            },
        ],
        deny_unreviewed: true,
    }
}

// ---------------------------------------------------------------------
// Dependency Feature Policy
// ---------------------------------------------------------------------

/// Capability classes a dependency feature may expand, implementing
/// "Dependency Feature Policy" -- "Features that enable networking,
/// filesystem expansion, native plugins, dynamic loading, or broad OS
/// capabilities" (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DependencyFeatureCapability {
    Networking,
    FilesystemExpansion,
    NativePluginOrDynamicLoading,
    BroadOsCapability,
}

/// A single enabled dependency feature's capability review.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyFeatureReview {
    pub crate_name: String,
    pub feature_name: String,
    pub capability: DependencyFeatureCapability,
    pub expected: bool,
    pub accepted_exception: bool,
}

/// "Unexpected capability-expanding features SHOULD block release until
/// reviewed."
pub fn reject_unexpected_capability_expanding_feature(
    review: &DependencyFeatureReview,
) -> Result<(), ReleaseSecurityError> {
    if !review.expected && !review.accepted_exception {
        return Err(ReleaseSecurityError::UnexpectedCapabilityExpandingFeature {
            crate_name: review.crate_name.clone(),
            feature: review.feature_name.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Vulnerability Handling
// ---------------------------------------------------------------------

/// "Release SHALL define vulnerability handling policy" -- "Policy SHOULD
/// include" the six fields below (`proposal.md`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VulnerabilityHandlingPolicy {
    pub advisory_severity_handling_defined: bool,
    pub release_blocking_criteria_defined: bool,
    pub mitigation_documentation_required: bool,
    pub exception_approval_defined: bool,
    pub follow_up_tracking_defined: bool,
    pub patch_release_expectation_documented: bool,
}

impl VulnerabilityHandlingPolicy {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        let complete = self.advisory_severity_handling_defined
            && self.release_blocking_criteria_defined
            && self.mitigation_documentation_required
            && self.exception_approval_defined
            && self.follow_up_tracking_defined
            && self.patch_release_expectation_documented;
        if complete {
            Ok(())
        } else {
            Err(ReleaseSecurityError::VulnerabilityHandlingPolicyIncomplete)
        }
    }
}

// ---------------------------------------------------------------------
// Security Notes
// ---------------------------------------------------------------------

/// The full security notes a release SHALL include, implementing "Security
/// Notes" (`proposal.md`). Named `SecurityReleaseNotes`, not
/// `ReleaseSecurityNotes`: [`crate::release_packaging::ReleaseSecurityNotes`]
/// already occupies that name as a deliberately shallow placeholder whose
/// own doc comment defers hardening detail to "a separate release security
/// change" -- this type is that detail, and reuses none of that struct's
/// fields to avoid suggesting the placeholder and this type must stay in
/// sync field-by-field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SecurityReleaseNotes {
    pub v0_1_threat_model: Option<String>,
    pub trusted_native_provider_model: Option<String>,
    pub no_raw_handle_policy: Option<String>,
    pub default_redaction: Option<String>,
    pub source_cache_trust_boundary: Option<String>,
    pub model_artifact_trust_boundary: Option<String>,
    pub component_artifact_trust_boundary: Option<String>,
    pub unsupported_security_features: Vec<String>,
    pub known_risks: Vec<String>,
    pub reporting_process_placeholder: Option<String>,
}

impl SecurityReleaseNotes {
    /// The SHALL-strength minimum: threat model, trusted Provider model,
    /// no-raw-handle policy, default redaction, and a reporting process
    /// placeholder. The remaining fields are `SHOULD`-strength and not
    /// enforced here, matching
    /// [`crate::release_packaging::ReleaseDocumentationChecklist::validate`]'s
    /// judgment call.
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        let required = [
            self.v0_1_threat_model.is_some(),
            self.trusted_native_provider_model.is_some(),
            self.no_raw_handle_policy.is_some(),
            self.default_redaction.is_some(),
            self.reporting_process_placeholder.is_some(),
        ];
        if required.into_iter().all(|present| present) {
            Ok(())
        } else {
            Err(ReleaseSecurityError::SecurityNotesIncomplete {
                reason: "one or more SHALL-strength security note topics missing".into(),
            })
        }
    }
}

// ---------------------------------------------------------------------
// Release Blocking Criteria
// ---------------------------------------------------------------------

/// The ten gate inputs from "Release Blocking Criteria" (`proposal.md`) --
/// "Stable release SHALL be blocked by" each of these.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReleaseSecurityGateInputs {
    pub secrets_detected: bool,
    pub critical_advisory_unmitigated: bool,
    pub incompatible_license_unapproved: bool,
    pub redaction_gate_failed: bool,
    pub raw_handle_exposed: bool,
    pub trust_integrity_failed_in_fixtures: bool,
    pub e2e_conformance_bypassed: bool,
    pub openspec_validation_failed: bool,
    pub checksum_mismatch: bool,
    pub undocumented_security_exception: bool,
}

/// Evaluates every [`ReleaseSecurityGateInputs`] field and, if any are
/// `true`, returns [`ReleaseSecurityError::ReleaseBlocked`] naming every
/// triggered reason (not just the first).
pub fn evaluate_release_security_blocking(
    inputs: &ReleaseSecurityGateInputs,
) -> Result<(), ReleaseSecurityError> {
    let mut reasons = Vec::new();
    if inputs.secrets_detected {
        reasons.push("secrets-detected");
    }
    if inputs.critical_advisory_unmitigated {
        reasons.push("critical-advisory-unmitigated");
    }
    if inputs.incompatible_license_unapproved {
        reasons.push("incompatible-license-unapproved");
    }
    if inputs.redaction_gate_failed {
        reasons.push("redaction-gate-failed");
    }
    if inputs.raw_handle_exposed {
        reasons.push("raw-handle-exposed");
    }
    if inputs.trust_integrity_failed_in_fixtures {
        reasons.push("trust-integrity-failed-in-fixtures");
    }
    if inputs.e2e_conformance_bypassed {
        reasons.push("e2e-conformance-bypassed");
    }
    if inputs.openspec_validation_failed {
        reasons.push("openspec-validation-failed");
    }
    if inputs.checksum_mismatch {
        reasons.push("checksum-mismatch");
    }
    if inputs.undocumented_security_exception {
        reasons.push("undocumented-security-exception");
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(ReleaseSecurityError::ReleaseBlocked {
            reasons: reasons.into_iter().map(String::from).collect(),
        })
    }
}

// ---------------------------------------------------------------------
// Security Exceptions
// ---------------------------------------------------------------------

/// A documented security exception, implementing "Security Exceptions" --
/// "An exception SHOULD include" the eight fields below (`proposal.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecurityException {
    pub issue: String,
    pub affected_component: String,
    pub severity: DependencyAdvisorySeverity,
    pub rationale: String,
    pub mitigation: String,
    pub owner: String,
    pub expiration_or_follow_up: String,
    pub release_note_entry: bool,
}

impl SecurityException {
    pub fn validate(&self) -> Result<(), ReleaseSecurityError> {
        let fields_present = !self.issue.trim().is_empty()
            && !self.affected_component.trim().is_empty()
            && !self.rationale.trim().is_empty()
            && !self.mitigation.trim().is_empty()
            && !self.owner.trim().is_empty()
            && !self.expiration_or_follow_up.trim().is_empty();
        if fields_present && self.release_note_entry {
            Ok(())
        } else {
            Err(ReleaseSecurityError::SecurityExceptionIncomplete)
        }
    }
}

/// "Undocumented exceptions SHALL not be allowed for stable release."
pub fn reject_undocumented_security_exception(
    exception_required: bool,
    exception: Option<&SecurityException>,
) -> Result<(), ReleaseSecurityError> {
    if !exception_required {
        return Ok(());
    }
    match exception {
        Some(exception) => exception.validate(),
        None => Err(ReleaseSecurityError::SecurityExceptionIncomplete),
    }
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// The ten release security observation kinds from "Observability"
/// (`proposal.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReleaseSecurityObservationKind {
    DependencyAuditCompleted,
    LicenseAuditCompleted,
    SbomGenerated,
    ChecksumGenerated,
    SecretScanCompleted,
    RedactionGateCompleted,
    ProvenanceGenerated,
    SecurityExceptionRecorded,
    ReleaseBlocked,
    ReleaseSecurityPassed,
}

/// A single recorded release security observation, always redacted by
/// [`record_release_security_observation`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseSecurityObservation {
    pub kind: ReleaseSecurityObservationKind,
    pub detail: Option<String>,
}

/// "Observability SHALL not expose secrets, credentials, raw prompts, raw
/// weights, raw tensors, raw cache contents, handles, memory pointers, or
/// local paths by default": always redacts `raw_detail` through
/// `redact_release_security_detail` before recording it.
pub fn record_release_security_observation(
    kind: ReleaseSecurityObservationKind,
    raw_detail: &str,
) -> ReleaseSecurityObservation {
    ReleaseSecurityObservation {
        kind,
        detail: Some(redact_release_security_detail(raw_detail)),
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured release security error, covering every failure category this
/// module's validation functions can produce.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReleaseSecurityError {
    HardenedClaimForExcludedFeature {
        feature: String,
    },
    CriticalAdvisoryUnmitigated {
        crate_name: String,
        advisory_id: String,
    },
    IncompatibleLicenseUnapproved {
        crate_name: String,
    },
    SbomMissingOrUndocumented,
    ChecksumMismatch {
        artifact: String,
    },
    SignatureAbsenceUndocumented,
    ProvenanceContainsSecretOrLocalPath {
        field: String,
    },
    ReproducibilityUndocumented,
    LockfileNotCheckedIn,
    LockfileDriftUnreviewed,
    UnexpectedBuildScriptUnreviewed {
        crate_name: String,
    },
    SecretDetected {
        target: String,
    },
    ArtifactIntegrityFailed {
        check: String,
    },
    RedactionGateFailed {
        diagnostic: String,
    },
    DynamicProviderLoadingUnreviewed,
    ProviderRegistrationTrustImplied,
    NativeHandleExposureDenied {
        surface: String,
        reason: String,
    },
    ComponentArtifactUntrusted {
        reason: String,
    },
    UnsignedComponentDeniedInProduction,
    ComponentAuthorityExpansionDenied {
        capability: String,
        reason: String,
    },
    ModelArtifactUntrusted {
        reason: String,
    },
    FixtureTrustPolicyUndocumented,
    CacheEntryTrustNotEstablished {
        artifact: String,
    },
    CacheSignalDoesNotImplyTrust {
        signal: String,
    },
    CliAuthorityDelegatedToRuntime {
        capability: String,
        reason: String,
    },
    RuntimeInferenceApiAuthorityExpansionDenied {
        capability: String,
        reason: String,
    },
    UnsafeCodeUnreviewed {
        location: String,
    },
    UnexpectedCapabilityExpandingFeature {
        crate_name: String,
        feature: String,
    },
    VulnerabilityHandlingPolicyIncomplete,
    SecurityNotesIncomplete {
        reason: String,
    },
    ReleaseBlocked {
        reasons: Vec<String>,
    },
    SecurityExceptionIncomplete,
    InternalReleaseSecurityError {
        reason: String,
    },
}

impl ReleaseSecurityError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::HardenedClaimForExcludedFeature { .. } => "hardened-claim-for-excluded-feature",
            Self::CriticalAdvisoryUnmitigated { .. } => "critical-advisory-unmitigated",
            Self::IncompatibleLicenseUnapproved { .. } => "incompatible-license-unapproved",
            Self::SbomMissingOrUndocumented => "sbom-missing-or-undocumented",
            Self::ChecksumMismatch { .. } => "checksum-mismatch",
            Self::SignatureAbsenceUndocumented => "signature-absence-undocumented",
            Self::ProvenanceContainsSecretOrLocalPath { .. } => {
                "provenance-contains-secret-or-local-path"
            }
            Self::ReproducibilityUndocumented => "reproducibility-undocumented",
            Self::LockfileNotCheckedIn => "lockfile-not-checked-in",
            Self::LockfileDriftUnreviewed => "lockfile-drift-unreviewed",
            Self::UnexpectedBuildScriptUnreviewed { .. } => "unexpected-build-script-unreviewed",
            Self::SecretDetected { .. } => "secret-detected",
            Self::ArtifactIntegrityFailed { .. } => "artifact-integrity-failed",
            Self::RedactionGateFailed { .. } => "redaction-gate-failed",
            Self::DynamicProviderLoadingUnreviewed => "dynamic-provider-loading-unreviewed",
            Self::ProviderRegistrationTrustImplied => "provider-registration-trust-implied",
            Self::NativeHandleExposureDenied { .. } => "native-handle-exposure-denied",
            Self::ComponentArtifactUntrusted { .. } => "component-artifact-untrusted",
            Self::UnsignedComponentDeniedInProduction => "unsigned-component-denied-in-production",
            Self::ComponentAuthorityExpansionDenied { .. } => {
                "component-authority-expansion-denied"
            }
            Self::ModelArtifactUntrusted { .. } => "model-artifact-untrusted",
            Self::FixtureTrustPolicyUndocumented => "fixture-trust-policy-undocumented",
            Self::CacheEntryTrustNotEstablished { .. } => "cache-entry-trust-not-established",
            Self::CacheSignalDoesNotImplyTrust { .. } => "cache-signal-does-not-imply-trust",
            Self::CliAuthorityDelegatedToRuntime { .. } => "cli-authority-delegated-to-runtime",
            Self::RuntimeInferenceApiAuthorityExpansionDenied { .. } => {
                "runtime-inference-api-authority-expansion-denied"
            }
            Self::UnsafeCodeUnreviewed { .. } => "unsafe-code-unreviewed",
            Self::UnexpectedCapabilityExpandingFeature { .. } => {
                "unexpected-capability-expanding-feature"
            }
            Self::VulnerabilityHandlingPolicyIncomplete => {
                "vulnerability-handling-policy-incomplete"
            }
            Self::SecurityNotesIncomplete { .. } => "security-notes-incomplete",
            Self::ReleaseBlocked { .. } => "release-blocked",
            Self::SecurityExceptionIncomplete => "security-exception-incomplete",
            Self::InternalReleaseSecurityError { .. } => "internal-release-security-error",
        }
    }
}

impl fmt::Display for ReleaseSecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HardenedClaimForExcludedFeature { feature } => {
                write!(f, "{}: {feature}", self.id())
            }
            Self::CriticalAdvisoryUnmitigated {
                crate_name,
                advisory_id,
            } => write!(f, "{}: {crate_name} ({advisory_id})", self.id()),
            Self::IncompatibleLicenseUnapproved { crate_name }
            | Self::UnexpectedBuildScriptUnreviewed { crate_name } => {
                write!(f, "{}: {crate_name}", self.id())
            }
            Self::SbomMissingOrUndocumented
            | Self::SignatureAbsenceUndocumented
            | Self::ReproducibilityUndocumented
            | Self::LockfileNotCheckedIn
            | Self::LockfileDriftUnreviewed
            | Self::DynamicProviderLoadingUnreviewed
            | Self::ProviderRegistrationTrustImplied
            | Self::UnsignedComponentDeniedInProduction
            | Self::FixtureTrustPolicyUndocumented
            | Self::VulnerabilityHandlingPolicyIncomplete
            | Self::SecurityExceptionIncomplete => write!(f, "{}", self.id()),
            Self::ChecksumMismatch { artifact } => write!(f, "{}: {artifact}", self.id()),
            Self::ProvenanceContainsSecretOrLocalPath { field } => {
                write!(f, "{}: {field}", self.id())
            }
            Self::SecretDetected { target } => write!(f, "{}: {target}", self.id()),
            Self::ArtifactIntegrityFailed { check } => write!(f, "{}: {check}", self.id()),
            Self::RedactionGateFailed { diagnostic } => write!(f, "{}: {diagnostic}", self.id()),
            Self::NativeHandleExposureDenied { surface, reason } => {
                write!(f, "{}: {surface} ({reason})", self.id())
            }
            Self::ComponentArtifactUntrusted { reason }
            | Self::ModelArtifactUntrusted { reason }
            | Self::SecurityNotesIncomplete { reason }
            | Self::InternalReleaseSecurityError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::ComponentAuthorityExpansionDenied { capability, reason }
            | Self::CliAuthorityDelegatedToRuntime { capability, reason }
            | Self::RuntimeInferenceApiAuthorityExpansionDenied { capability, reason } => {
                write!(f, "{}: {capability} ({reason})", self.id())
            }
            Self::CacheEntryTrustNotEstablished { artifact } => {
                write!(f, "{}: {artifact}", self.id())
            }
            Self::CacheSignalDoesNotImplyTrust { signal } => write!(f, "{}: {signal}", self.id()),
            Self::UnsafeCodeUnreviewed { location } => write!(f, "{}: {location}", self.id()),
            Self::UnexpectedCapabilityExpandingFeature {
                crate_name,
                feature,
            } => {
                write!(f, "{}: {crate_name}/{feature}", self.id())
            }
            Self::ReleaseBlocked { reasons } => {
                write!(f, "{}: {}", self.id(), reasons.join(", "))
            }
        }
    }
}

impl Error for ReleaseSecurityError {}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single release security conformance check result, mirroring
/// [`crate::release_packaging::ReleasePackagingConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseSecurityConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ReleaseSecurityConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseSecurityConformanceReport {
    pub results: Vec<ReleaseSecurityConformanceResult>,
}

impl ReleaseSecurityConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ReleaseSecurityConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ReleaseSecurityConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the release security conformance checks described in this module's
/// doc comment: CUDA cannot be claimed hardened; a critical unmitigated
/// advisory blocks stable release; an unapproved incompatible license
/// blocks stable release; an SBOM manifest with no entries and no
/// limitation note is rejected; a checksum mismatch is rejected; an
/// undocumented signature absence is rejected; provenance containing a
/// secret-shaped or path-shaped field is rejected; undocumented
/// reproducibility is rejected; unreviewed lockfile drift is rejected; an
/// unexpected unreviewed build script is rejected; a detected secret blocks
/// stable release; incomplete artifact integrity is rejected; a diagnostic
/// containing a raw prompt or native handle fails the redaction gate; an
/// unreviewed stable dynamic Provider loading status is rejected; a
/// registration-only Provider trust source is rejected; every native handle
/// surface fragment (including Provider-roadmap-specific and CPU
/// allocation pointer fragments) is denied; an untrusted or unsigned-in-
/// production Component Artifact is denied; a Component/CLI/Runtime
/// authority expansion request is denied; an untrusted Model Artifact is
/// denied regardless of recognized format; an undocumented fixture trust
/// policy is rejected; a cache entry without an explicit Trusted status is
/// rejected regardless of which non-trust signal is present; unreviewed
/// unsafe code is rejected; an unexpected capability-expanding dependency
/// feature is rejected; an incomplete vulnerability handling policy is
/// rejected; incomplete security notes are rejected; every release blocking
/// criterion is evaluated; an incomplete or undocumented security exception
/// is rejected; and a release security observation is always redacted.
pub fn run_release_security_conformance() -> ReleaseSecurityConformanceReport {
    let mut results = Vec::new();

    {
        let outcome = reject_hardened_security_claim_for_excluded_feature("cuda", true);
        record(
            &mut results,
            "CUDA cannot be claimed as hardened in v0.1 security notes",
            matches!(
                outcome,
                Err(ReleaseSecurityError::HardenedClaimForExcludedFeature { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let allowed =
            reject_hardened_security_claim_for_excluded_feature("reference-cpu-provider", true);
        record(
            &mut results,
            "the CPU-local baseline may be claimed hardened",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let report = DependencyAuditReport {
            advisories: vec![DependencyAdvisory {
                crate_name: "example-crate".into(),
                advisory_id: "RUSTSEC-0000-0000".into(),
                severity: DependencyAdvisorySeverity::Critical,
                mitigated: false,
                mitigation: None,
            }],
            ..Default::default()
        };
        let outcome = report.validate_for_stable_release();
        record(
            &mut results,
            "a critical unmitigated advisory blocks stable release",
            matches!(
                outcome,
                Err(ReleaseSecurityError::CriticalAdvisoryUnmitigated { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let mut mitigated = report;
        mitigated.advisories[0].mitigated = true;
        record(
            &mut results,
            "a mitigated critical advisory does not block stable release",
            mitigated.validate_for_stable_release().is_ok(),
            format!(
                "unexpected outcome: {:?}",
                mitigated.validate_for_stable_release()
            ),
        );
    }

    {
        let report = LicenseAuditReport {
            licenses: vec![DependencyLicense {
                crate_name: "example-crate".into(),
                spdx: None,
                status: LicenseAuditStatus::Incompatible,
                exception_approved: false,
            }],
            ..Default::default()
        };
        let outcome = report.validate_for_stable_release();
        record(
            &mut results,
            "an unapproved incompatible license blocks stable release",
            matches!(
                outcome,
                Err(ReleaseSecurityError::IncompatibleLicenseUnapproved { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let mut approved = report;
        approved.licenses[0].exception_approved = true;
        record(
            &mut results,
            "an approved incompatible license does not block stable release",
            approved.validate_for_stable_release().is_ok(),
            format!(
                "unexpected outcome: {:?}",
                approved.validate_for_stable_release()
            ),
        );
    }

    {
        let missing = SbomManifest {
            availability: SbomAvailability::Missing,
            ..Default::default()
        };
        record(
            &mut results,
            "a missing SBOM with no documented limitation is rejected",
            missing.validate().is_err(),
            format!("unexpected outcome: {:?}", missing.validate()),
        );
        let documented = SbomManifest {
            availability: SbomAvailability::PlaceholderDocumented,
            limitation_note: Some("SBOM generation is not implemented for v0.1".into()),
            ..Default::default()
        };
        record(
            &mut results,
            "a documented SBOM limitation is accepted",
            documented.validate().is_ok(),
            format!("unexpected outcome: {:?}", documented.validate()),
        );
    }

    {
        let checksum = ArtifactChecksum::new(
            "magnetar-cli",
            crate::release_packaging::ChecksumAlgorithm::Sha256,
            "deadbeef",
        )
        .expect("non-empty digest");
        let outcome = verify_checksum_matches_final_artifact(&checksum, "different-digest");
        record(
            &mut results,
            "a checksum mismatch is rejected",
            matches!(outcome, Err(ReleaseSecurityError::ChecksumMismatch { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
        let matching = verify_checksum_matches_final_artifact(&checksum, "deadbeef");
        record(
            &mut results,
            "a matching checksum is accepted",
            matching.is_ok(),
            format!("unexpected outcome: {matching:?}"),
        );
    }

    {
        let outcome = validate_signature_status(SignatureStatus::NotImplementedUndocumented);
        record(
            &mut results,
            "an undocumented signature absence is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
        let documented = validate_signature_status(SignatureStatus::NotImplementedDocumented);
        record(
            &mut results,
            "a documented signature absence is accepted",
            documented.is_ok(),
            format!("unexpected outcome: {documented:?}"),
        );
    }

    {
        let leaky = ReleaseProvenance {
            source_commit: Some("/home/user/workspace/magnetar".into()),
            ..Default::default()
        };
        let outcome = leaky.validate();
        record(
            &mut results,
            "provenance containing a local path is rejected",
            matches!(
                outcome,
                Err(ReleaseSecurityError::ProvenanceContainsSecretOrLocalPath { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let clean = ReleaseProvenance {
            source_commit: Some("abc1234".into()),
            release_tag: Some("v0.1.0".into()),
            ..Default::default()
        };
        record(
            &mut results,
            "provenance without secrets or local paths is accepted",
            clean.validate().is_ok(),
            format!("unexpected outcome: {:?}", clean.validate()),
        );
    }

    {
        let undocumented = ReproducibilityReport::default();
        record(
            &mut results,
            "undocumented reproducibility status is rejected",
            undocumented.validate().is_err(),
            format!("unexpected outcome: {:?}", undocumented.validate()),
        );
        let documented = ReproducibilityReport {
            status: ReproducibilityStatus::PartiallyReproducible,
            limitations: vec!["timestamps are not normalized".into()],
        };
        record(
            &mut results,
            "partially reproducible with documented limitations is accepted",
            documented.validate().is_ok(),
            format!("unexpected outcome: {:?}", documented.validate()),
        );
    }

    {
        let unreviewed_drift = LockfileState {
            checked_in: true,
            digest: Some("abc".into()),
            drift_detected: true,
            drift_reviewed: false,
        };
        let outcome = reject_unreviewed_lockfile_drift(&unreviewed_drift);
        record(
            &mut results,
            "unreviewed lockfile drift blocks a release candidate",
            matches!(outcome, Err(ReleaseSecurityError::LockfileDriftUnreviewed)),
            format!("unexpected outcome: {outcome:?}"),
        );
        let reviewed = LockfileState {
            drift_reviewed: true,
            ..unreviewed_drift
        };
        record(
            &mut results,
            "reviewed lockfile drift does not block a release candidate",
            reject_unreviewed_lockfile_drift(&reviewed).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                reject_unreviewed_lockfile_drift(&reviewed)
            ),
        );
    }

    {
        let unreviewed = BuildScriptReview {
            crate_name: "example-native-sys".into(),
            has_build_script: true,
            unexpected: true,
            reviewed: false,
            native_build_documented: false,
        };
        let outcome = flag_unexpected_build_script(&unreviewed);
        record(
            &mut results,
            "an unexpected unreviewed build script is flagged",
            matches!(
                outcome,
                Err(ReleaseSecurityError::UnexpectedBuildScriptUnreviewed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        assert_eq!(SECRET_SCAN_TARGETS.len(), 8);
        let report = SecretScanReport {
            findings: vec![SecretScanFinding {
                target: SecretScanTarget::ReleaseNotes,
                detected: true,
                location: Some("CHANGELOG.md".into()),
            }],
        };
        let outcome = report.validate_for_stable_release();
        record(
            &mut results,
            "a detected secret blocks stable release",
            matches!(outcome, Err(ReleaseSecurityError::SecretDetected { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
        let clean = SecretScanReport {
            findings: vec![SecretScanFinding {
                target: SecretScanTarget::ReleaseNotes,
                detected: false,
                location: None,
            }],
        };
        record(
            &mut results,
            "a clean secret scan does not block stable release",
            clean.validate_for_stable_release().is_ok(),
            format!(
                "unexpected outcome: {:?}",
                clean.validate_for_stable_release()
            ),
        );
    }

    {
        let incomplete = ArtifactIntegrityStatus::default();
        record(
            &mut results,
            "incomplete artifact integrity is rejected",
            incomplete.validate().is_err(),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
        let complete = ArtifactIntegrityStatus {
            source_state_clean_or_ci_controlled: true,
            release_tag_matches_source: true,
            openspec_report_matches_baseline: true,
            conformance_reports_match_commit: true,
            checksums_match_final_artifacts: true,
        };
        record(
            &mut results,
            "complete artifact integrity is accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        assert_eq!(REDACTION_CATEGORIES.len(), 13);
        let outcome = validate_redaction_gate("raw prompt: tell me a secret");
        record(
            &mut results,
            "a diagnostic containing a raw prompt fails the redaction gate",
            matches!(
                outcome,
                Err(ReleaseSecurityError::RedactionGateFailed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let handle_outcome = validate_redaction_gate("provider handle=0xdeadbeef");
        record(
            &mut results,
            "a diagnostic containing a native handle fails the redaction gate",
            handle_outcome.is_err(),
            format!("unexpected outcome: {handle_outcome:?}"),
        );
        let ordinary = validate_redaction_gate("generation completed in 12ms");
        record(
            &mut results,
            "an ordinary diagnostic passes the redaction gate",
            ordinary.is_ok(),
            format!("unexpected outcome: {ordinary:?}"),
        );
    }

    {
        let outcome = validate_dynamic_provider_loading_status(
            ProviderLoadingMode::DynamicLibrary,
            DynamicProviderLoadingStatus::StableUnreviewed,
        );
        record(
            &mut results,
            "unreviewed stable dynamic Provider loading is rejected",
            matches!(
                outcome,
                Err(ReleaseSecurityError::DynamicProviderLoadingUnreviewed)
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let reviewed = validate_dynamic_provider_loading_status(
            ProviderLoadingMode::DynamicLibrary,
            DynamicProviderLoadingStatus::SecurityReviewed,
        );
        record(
            &mut results,
            "a security-reviewed dynamic Provider loading status is accepted",
            reviewed.is_ok(),
            format!("unexpected outcome: {reviewed:?}"),
        );
        let built_in = validate_dynamic_provider_loading_status(
            ProviderLoadingMode::BuiltIn,
            DynamicProviderLoadingStatus::StableUnreviewed,
        );
        record(
            &mut results,
            "a built-in (non-dynamic) Provider is never subject to the dynamic loading check",
            built_in.is_ok(),
            format!("unexpected outcome: {built_in:?}"),
        );
    }

    {
        let outcome =
            reject_provider_registration_implies_trust(ProviderTrustSignalSource::RegistrationOnly);
        record(
            &mut results,
            "Provider registration alone does not establish trust",
            matches!(
                outcome,
                Err(ReleaseSecurityError::ProviderRegistrationTrustImplied)
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let policy_backed =
            reject_provider_registration_implies_trust(ProviderTrustSignalSource::ConfiguredPolicy);
        record(
            &mut results,
            "a configured-policy-backed Provider trust source is accepted",
            policy_backed.is_ok(),
            format!("unexpected outcome: {policy_backed:?}"),
        );
    }

    {
        for surface in [
            "raw-provider-handle",
            "raw-device-handle",
            "raw-kernel-handle",
            "raw-tensor-pointer",
            "raw-memory-pointer",
            "cuda-stream",
            "cuda-device-pointer",
            "metal-buffer",
            "metal-command-queue",
            "openvino-compiled-graph",
            "qnn-native-handle",
            "raw-cpu-allocation-pointer",
        ] {
            let outcome = reject_release_native_handle_exposure(surface);
            record(
                &mut results,
                format!("native handle surface '{surface}' is denied"),
                outcome.is_err(),
                format!("unexpected outcome: {outcome:?}"),
            );
        }
        let allowed = reject_release_native_handle_exposure("generation");
        record(
            &mut results,
            "an ordinary release surface is not denied as a native handle",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let untrusted = ComponentTrustDecision::new(ComponentTrustStatus::Rejected, "test");
        let outcome = validate_component_release_execution_trust(&untrusted, true, true, false);
        record(
            &mut results,
            "an untrusted Component Artifact is denied execution",
            matches!(
                outcome,
                Err(ReleaseSecurityError::ComponentArtifactUntrusted { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let trusted = ComponentTrustDecision::new(ComponentTrustStatus::Trusted, "trusted fixture");
        let unsigned_in_production =
            validate_component_release_execution_trust(&trusted, false, true, false);
        record(
            &mut results,
            "an unsigned Component Artifact is denied under production policy unless explicitly allowed",
            matches!(
                unsigned_in_production,
                Err(ReleaseSecurityError::UnsignedComponentDeniedInProduction)
            ),
            format!("unexpected outcome: {unsigned_in_production:?}"),
        );
        let explicitly_allowed =
            validate_component_release_execution_trust(&trusted, false, true, true);
        record(
            &mut results,
            "an unsigned Component Artifact explicitly allowed under production policy is accepted",
            explicitly_allowed.is_ok(),
            format!("unexpected outcome: {explicitly_allowed:?}"),
        );
    }

    {
        let outcome = reject_component_release_authority_expansion("filesystem");
        record(
            &mut results,
            "Component execution cannot gain filesystem authority",
            matches!(
                outcome,
                Err(ReleaseSecurityError::ComponentAuthorityExpansionDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let handle_outcome = reject_component_release_authority_expansion("raw-provider-handle");
        record(
            &mut results,
            "Component execution cannot gain raw Provider handle authority",
            handle_outcome.is_err(),
            format!("unexpected outcome: {handle_outcome:?}"),
        );
        let allowed = reject_component_release_authority_expansion("generation");
        record(
            &mut results,
            "an inference-scoped Component authority request is accepted",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let untrusted = ModelTrustDecision::new(ModelTrustStatus::Unknown, "no policy matched");
        let outcome = validate_model_artifact_release_trust(&untrusted, true);
        record(
            &mut results,
            "recognized format does not establish Model Artifact trust",
            matches!(
                outcome,
                Err(ReleaseSecurityError::ModelArtifactUntrusted { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let trusted = ModelTrustDecision::new(ModelTrustStatus::Trusted, "digest trusted");
        record(
            &mut results,
            "a policy-trusted Model Artifact is accepted",
            validate_model_artifact_release_trust(&trusted, false).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                validate_model_artifact_release_trust(&trusted, false)
            ),
        );
    }

    {
        let trusted = ModelTrustDecision::new(ModelTrustStatus::Trusted, "fixture trusted");
        let undocumented = FixtureModelTrustPolicy::default();
        let outcome = validate_fixture_model_trust(&trusted, &undocumented);
        record(
            &mut results,
            "a fixture Model Artifact without an explicit test trust policy is rejected",
            matches!(
                outcome,
                Err(ReleaseSecurityError::FixtureTrustPolicyUndocumented)
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let documented = FixtureModelTrustPolicy {
            explicit_test_policy_documented: true,
        };
        record(
            &mut results,
            "a fixture Model Artifact with an explicit test trust policy is accepted",
            validate_fixture_model_trust(&trusted, &documented).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                validate_fixture_model_trust(&trusted, &documented)
            ),
        );
    }

    {
        use crate::{
            CacheIntegrityStatus, CacheLifecycleState, CacheValidationStatus, ModelArtifactId,
            ModelArtifactKind, ModelDigest, ModelName, ModelRevision, ModelSourceKind,
        };

        let mut entry = CacheEntryMetadata::new(
            ModelArtifactId {
                kind: ModelArtifactKind::ModelWeights,
                name: ModelName::new("qwen-test").expect("valid name"),
                revision: ModelRevision::new("v1").expect("valid revision"),
                variant: None,
                digest: ModelDigest {
                    algorithm: "sha256".into(),
                    value: "cachedigest".into(),
                },
                source: None,
                shard: None,
            },
            ModelSourceKind::LocalDirectorySource,
        );
        entry.trust_status = ModelTrustStatus::Unknown;
        entry.integrity_status = CacheIntegrityStatus::Unchecked;
        entry.validation_status = CacheValidationStatus::Unvalidated;
        entry.lifecycle = CacheLifecycleState::Discovered;

        for signal in [
            NonTrustCacheSignal::CacheHit,
            NonTrustCacheSignal::SourceKind,
            NonTrustCacheSignal::Alias,
            NonTrustCacheSignal::LocalFile,
            NonTrustCacheSignal::FixtureStatus,
        ] {
            let outcome = reject_cache_signal_alone_as_trust(signal, &entry);
            record(
                &mut results,
                format!("cache signal {signal:?} alone does not establish trust"),
                matches!(
                    outcome,
                    Err(ReleaseSecurityError::CacheSignalDoesNotImplyTrust { .. })
                ),
                format!("unexpected outcome: {outcome:?}"),
            );
        }

        entry.trust_status = ModelTrustStatus::Trusted;
        record(
            &mut results,
            "a cache entry with an explicit Trusted status passes the release trust check",
            validate_source_cache_release_trust(&entry).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                validate_source_cache_release_trust(&entry)
            ),
        );
    }

    {
        let outcome = validate_cli_authority_not_delegated_to_runtime("filesystem");
        record(
            &mut results,
            "CLI filesystem authority is not delegated to Runtime",
            matches!(
                outcome,
                Err(ReleaseSecurityError::CliAuthorityDelegatedToRuntime { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = validate_runtime_inference_api_security("shell");
        record(
            &mut results,
            "Runtime Inference API rejects shell authority requests",
            matches!(
                outcome,
                Err(ReleaseSecurityError::RuntimeInferenceApiAuthorityExpansionDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let allowed = validate_runtime_inference_api_security("generation");
        record(
            &mut results,
            "Runtime Inference API accepts an inference-scoped capability",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let policy = UnsafeCodePolicy {
            reviews: vec![UnsafeCodeReview {
                location: "compute.rs:100".into(),
                justified: false,
                reviewed: false,
            }],
            deny_unreviewed: true,
        };
        let outcome = policy.validate();
        record(
            &mut results,
            "unreviewed unsafe code is rejected when the policy denies it",
            matches!(
                outcome,
                Err(ReleaseSecurityError::UnsafeCodeUnreviewed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let reviewed_policy = UnsafeCodePolicy {
            reviews: vec![UnsafeCodeReview {
                location: "compute.rs:100".into(),
                justified: true,
                reviewed: true,
            }],
            deny_unreviewed: true,
        };
        record(
            &mut results,
            "reviewed and justified unsafe code is accepted",
            reviewed_policy.validate().is_ok(),
            format!("unexpected outcome: {:?}", reviewed_policy.validate()),
        );

        let real_inventory = magnetar_runtime_unsafe_code_inventory();
        record(
            &mut results,
            "the real magnetar-runtime unsafe code inventory is reviewed and justified",
            real_inventory.validate().is_ok(),
            format!("unexpected outcome: {:?}", real_inventory.validate()),
        );
    }

    {
        let review = DependencyFeatureReview {
            crate_name: "example-crate".into(),
            feature_name: "http-client".into(),
            capability: DependencyFeatureCapability::Networking,
            expected: false,
            accepted_exception: false,
        };
        let outcome = reject_unexpected_capability_expanding_feature(&review);
        record(
            &mut results,
            "an unexpected networking-enabling feature is rejected",
            matches!(
                outcome,
                Err(ReleaseSecurityError::UnexpectedCapabilityExpandingFeature { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let expected = DependencyFeatureReview {
            expected: true,
            ..review
        };
        record(
            &mut results,
            "an expected feature is accepted",
            reject_unexpected_capability_expanding_feature(&expected).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                reject_unexpected_capability_expanding_feature(&expected)
            ),
        );
    }

    {
        let incomplete = VulnerabilityHandlingPolicy::default();
        record(
            &mut results,
            "an incomplete vulnerability handling policy is rejected",
            incomplete.validate().is_err(),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
        let complete = VulnerabilityHandlingPolicy {
            advisory_severity_handling_defined: true,
            release_blocking_criteria_defined: true,
            mitigation_documentation_required: true,
            exception_approval_defined: true,
            follow_up_tracking_defined: true,
            patch_release_expectation_documented: true,
        };
        record(
            &mut results,
            "a complete vulnerability handling policy is accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let incomplete = SecurityReleaseNotes::default();
        record(
            &mut results,
            "incomplete security notes are rejected",
            incomplete.validate().is_err(),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
        let complete = SecurityReleaseNotes {
            v0_1_threat_model: Some("CPU-local baseline".into()),
            trusted_native_provider_model: Some("Providers are trusted native code".into()),
            no_raw_handle_policy: Some("no raw handles in public APIs".into()),
            default_redaction: Some("diagnostics redact by default".into()),
            reporting_process_placeholder: Some("reporting process TBD".into()),
            ..Default::default()
        };
        record(
            &mut results,
            "complete security notes are accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let clean = ReleaseSecurityGateInputs::default();
        record(
            &mut results,
            "no blocking criteria triggered allows stable release",
            evaluate_release_security_blocking(&clean).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                evaluate_release_security_blocking(&clean)
            ),
        );
        let blocked = ReleaseSecurityGateInputs {
            secrets_detected: true,
            checksum_mismatch: true,
            ..Default::default()
        };
        let outcome = evaluate_release_security_blocking(&blocked);
        record(
            &mut results,
            "every triggered blocking criterion is reported",
            matches!(
                &outcome,
                Err(ReleaseSecurityError::ReleaseBlocked { reasons })
                    if reasons.len() == 2
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = reject_undocumented_security_exception(true, None);
        record(
            &mut results,
            "an undocumented required security exception is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
        let exception = SecurityException {
            issue: "advisory RUSTSEC-0000-0000".into(),
            affected_component: "example-crate".into(),
            severity: DependencyAdvisorySeverity::High,
            rationale: "no fixed version available yet".into(),
            mitigation: "vendored patch applied".into(),
            owner: "release-team".into(),
            expiration_or_follow_up: "revisit in v0.2".into(),
            release_note_entry: true,
        };
        let documented = reject_undocumented_security_exception(true, Some(&exception));
        record(
            &mut results,
            "a fully documented security exception is accepted",
            documented.is_ok(),
            format!("unexpected outcome: {documented:?}"),
        );
    }

    {
        let observation = record_release_security_observation(
            ReleaseSecurityObservationKind::SecretScanCompleted,
            "secret scan found credential in build metadata",
        );
        record(
            &mut results,
            "a release security observation is redacted by default",
            observation
                .detail
                .as_deref()
                .is_some_and(|detail| !detail.contains("credential")),
            format!("observation leaked sensitive content: {observation:?}"),
        );
    }

    ReleaseSecurityConformanceReport { results }
}
