//! v0.1 release cutover checklist policy contract (see
//! `openspec/changes/define-v0-1-release-cutover-checklist`).
//!
//! This module does not implement release automation, publish a release,
//! define registry credentials, define a final hosting provider, define a
//! legal approval process, guarantee production security, or include GPU
//! Providers, a server API implementation, model hub downloads, or an
//! agent/tool runtime -- the proposal's "Non-Goals" rule all of that out
//! explicitly. Instead it defines, as executable Rust types and validation
//! functions, the **operational sequence** required to move from release
//! candidate to stable `v0.1`: `freeze -> gates -> reports -> artifacts ->
//! tag -> publish -> verify`. It is deliberately an **orchestrator**: every
//! check below composes an existing policy module
//! ([`crate::release_packaging`], [`crate::release_security`]) rather than
//! re-implementing packaging, versioning, or security logic a sibling
//! module already owns.
//!
//! - [`ReleaseReadinessChecklist`]: "Release Readiness" -- the nine
//!   readiness confirmations before cutover begins.
//! - [`OpenSpecFreezeConfirmation`] / [`reject_semantic_change_after_freeze`]:
//!   "OpenSpec Freeze", composing [`crate::release_packaging::ReleaseFreezeState`]
//!   and [`crate::release_packaging::reject_change_after_freeze`] instead of
//!   a parallel freeze model.
//! - [`validate_v0_1_scope_feature`]: "Scope Confirmation", composing
//!   [`crate::release_packaging::reject_roadmap_feature_as_guarantee`]
//!   rather than a second deferred-roadmap-feature list.
//! - [`WitPackageVersionRecord`] / [`validate_wit_versions_confirmed`] /
//!   [`CutoverVersionConfirmation`] / [`validate_runtime_version_matches_release_tag`]:
//!   "Version Confirmation", composing
//!   [`crate::release_packaging::ReleaseBinaryVersionReport`] and
//!   [`crate::release_packaging::release_wit_contract_versions`].
//! - [`validate_cutover_feature_flag`] / [`validate_cutover_provider_feature_flags`]:
//!   "Feature Flag Confirmation", composing
//!   [`crate::release_packaging::reject_experimental_flag_enabled_by_default`]
//!   and [`crate::release_packaging::validate_provider_feature_flags_for_v0_1`].
//! - [`CutoverCompatibilityDimension`] / [`CutoverCompatibilityStatus`] /
//!   [`CutoverCompatibilityMatrix`] / [`reject_status_misrepresentation`]:
//!   "Compatibility Matrix Completion" -- the twelve-dimension, six-status
//!   matrix this checklist requires (a superset of
//!   [`crate::release_packaging::CompatibilityDimension`]'s eight
//!   dimensions and five statuses, so it is deliberately a distinct type
//!   rather than reusing that narrower one).
//! - [`validate_required_gates_executed`]: "Required Gate Execution",
//!   composing [`crate::release_packaging::release_may_publish_stable`]
//!   directly.
//! - [`GateSkip`] / [`validate_gate_skips`]: "Skip Review".
//! - [`CutoverException`] / [`reject_undocumented_cutover_exception`]:
//!   "Exception Review", composing
//!   [`crate::release_security::SecurityException`] rather than a parallel
//!   exception record.
//! - [`CutoverSecurityVerification`]: "Security Verification", composing
//!   [`crate::release_security::ReleaseSecurityGateInputs`] /
//!   [`crate::release_security::evaluate_release_security_blocking`] and
//!   [`crate::release_security::SecurityReleaseNotes`].
//! - [`validate_cutover_artifacts_generated`]: "Artifact Generation",
//!   composing [`crate::release_packaging::ReleaseArtifactManifest`].
//! - [`CutoverArtifactVerification`] / [`verify_cutover_artifact_checksum`]:
//!   "Artifact Verification", composing
//!   [`crate::release_security::ArtifactIntegrityStatus`] and
//!   [`crate::release_security::verify_checksum_matches_final_artifact`].
//! - [`CutoverChangelogChecklist`]: "Changelog Completion", composing
//!   [`crate::release_packaging::ReleaseChangelog`].
//! - [`CutoverReleaseNotesChecklist`]: "Release Notes Completion".
//! - [`validate_tag_after_gates`]: "Tagging", composing
//!   [`crate::release_packaging::release_may_publish_stable`].
//! - [`validate_publication_scope_preserved`]: "Publication", composing
//!   [`crate::release_packaging::reject_roadmap_feature_as_guarantee`] and
//!   [`reject_status_misrepresentation`].
//! - [`PostPublicationVerification`]: "Post-Publication Verification".
//! - [`RollbackRetractionNotes`]: "Rollback And Retraction Notes".
//! - [`PostV01HandoffItem`] / [`POST_V0_1_HANDOFF_CANDIDATES`] /
//!   [`reject_post_v0_1_item_as_release_claim`]: "Post-v0.1 Handoff".
//! - [`V0_1_FINAL_RELEASE_STATEMENT`] / [`validate_final_release_statement`]:
//!   "Final Release Statement".
//! - [`ReleaseCutoverObservation`] / [`record_release_cutover_observation`]:
//!   observability, composing [`crate::CorrelationId`] (for gate/target/
//!   feature-set/artifact correlation) and
//!   [`crate::release_security::record_release_security_observation`] (for
//!   default redaction) instead of a third redaction implementation.
//! - [`validate_cutover_cli_boundary`] / [`validate_cutover_runtime_scope`]:
//!   the CLI-boundary and Runtime-scope cutover checks, composing
//!   [`crate::release_security::validate_cli_authority_not_delegated_to_runtime`]
//!   and [`crate::release_security::validate_runtime_inference_api_security`]
//!   (which themselves already compose `cli_boundary` and `inference_api`)
//!   rather than importing those crate modules a third time.
//! - [`ReleaseCutoverError`]: structured error categories covering every
//!   failure category above.
//! - [`ReleaseCutoverGateInputs`] / [`evaluate_release_cutover`]: the
//!   top-level blocking-criteria aggregator, in the shape of
//!   [`crate::release_security::ReleaseSecurityGateInputs`] /
//!   [`crate::release_security::evaluate_release_security_blocking`].
//! - [`ReleaseCutoverConformanceReport`] / [`run_release_cutover_conformance`]:
//!   a conformance report, in the shape of
//!   [`crate::ReleasePackagingConformanceReport`] /
//!   [`crate::ReleaseSecurityConformanceReport`], asserting the guarantees
//!   above hold.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::{
    ArtifactChecksum, ArtifactIntegrityStatus, CorrelationId, REQUIRED_RELEASE_GATES,
    ReleaseArtifactManifest, ReleaseBinaryVersionReport, ReleaseChangelog, ReleaseFeatureFlag,
    ReleaseFeatureFlagClass, ReleaseFreezeChangeKind, ReleaseFreezeState, ReleaseGateResult,
    ReleaseSecurityGateInputs, ReleaseSecurityObservation, ReleaseSecurityObservationKind,
    ReleaseVersion, SecurityException, SecurityReleaseNotes, WitInterface,
    evaluate_release_security_blocking, record_release_security_observation,
    reject_change_after_freeze, reject_experimental_flag_enabled_by_default,
    reject_roadmap_feature_as_guarantee, release_may_publish_stable,
    validate_cli_authority_not_delegated_to_runtime, validate_provider_feature_flags_for_v0_1,
    validate_runtime_inference_api_security, verify_checksum_matches_final_artifact,
};

pub const RELEASE_CUTOVER_POLICY_VERSION: &str = "0.1.0";

/// The `v0.1` included baseline named in "Scope Confirmation"
/// (`proposal.md`). Deliberately documentation-only: enforcement is
/// [`validate_v0_1_scope_feature`], which composes
/// [`reject_roadmap_feature_as_guarantee`] rather than a second list of
/// excluded features -- `crate::release_packaging`'s deferred roadmap feature
/// list already owns that.
pub const V0_1_INCLUDED_SCOPE: &[&str] = &[
    "runtime-inference-api-baseline",
    "model-loading-baseline",
    "model-instance-baseline",
    "tokenizer-fixture-path",
    "qwen-like-baseline-fixture-path",
    "generation-and-sampling-baseline",
    "tensor-and-memory-baseline",
    "operator-first-scope",
    "kernel-registry-and-dispatch",
    "reference-cpu-provider",
    "cli-boundary-harness",
    "e2e-local-inference-conformance",
    "release-reports",
];

// ---------------------------------------------------------------------
// Release Readiness
// ---------------------------------------------------------------------

/// The nine release readiness confirmations from "Release Readiness"
/// (`proposal.md`) that SHALL be true before cutover begins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReleaseReadinessChecklist {
    pub release_branch_or_commit_selected: bool,
    pub version_selected: bool,
    pub openspec_baseline_selected: bool,
    pub scope_selected: bool,
    pub gates_selected: bool,
    pub artifacts_selected: bool,
    pub release_notes_draft_exists: bool,
    pub compatibility_matrix_draft_exists: bool,
    pub security_notes_draft_exists: bool,
}

impl ReleaseReadinessChecklist {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        let checks: [(&str, bool); 9] = [
            (
                "release-branch-or-commit-selected",
                self.release_branch_or_commit_selected,
            ),
            ("version-selected", self.version_selected),
            (
                "openspec-baseline-selected",
                self.openspec_baseline_selected,
            ),
            ("scope-selected", self.scope_selected),
            ("gates-selected", self.gates_selected),
            ("artifacts-selected", self.artifacts_selected),
            (
                "release-notes-draft-exists",
                self.release_notes_draft_exists,
            ),
            (
                "compatibility-matrix-draft-exists",
                self.compatibility_matrix_draft_exists,
            ),
            (
                "security-notes-draft-exists",
                self.security_notes_draft_exists,
            ),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::ReleaseReadinessIncomplete {
                    missing: name.into(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// OpenSpec Freeze
// ---------------------------------------------------------------------

/// "OpenSpec Freeze" (`proposal.md`): the six freeze confirmations, plus the
/// [`ReleaseFreezeState`] itself, composed rather than duplicated from
/// [`crate::release_packaging`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenSpecFreezeConfirmation {
    pub freeze_state: ReleaseFreezeState,
    pub accepted_changes_list_final: bool,
    pub pending_changes_excluded: bool,
    pub wit_breaking_changes_have_version_bumps: bool,
    pub checklist_references_correct_changes: bool,
    pub roadmap_items_deferred_unless_included: bool,
}

impl OpenSpecFreezeConfirmation {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        if !matches!(self.freeze_state, ReleaseFreezeState::Frozen) {
            return Err(ReleaseCutoverError::OpenSpecNotFrozen {
                reason: "release is not in a frozen OpenSpec state".into(),
            });
        }
        let checks: [(&str, bool); 5] = [
            (
                "accepted-changes-list-final",
                self.accepted_changes_list_final,
            ),
            ("pending-changes-excluded", self.pending_changes_excluded),
            (
                "wit-breaking-changes-have-version-bumps",
                self.wit_breaking_changes_have_version_bumps,
            ),
            (
                "checklist-references-correct-changes",
                self.checklist_references_correct_changes,
            ),
            (
                "roadmap-items-deferred-unless-included",
                self.roadmap_items_deferred_unless_included,
            ),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::OpenSpecNotFrozen {
                    reason: name.into(),
                });
            }
        }
        Ok(())
    }
}

/// "Late semantic change SHALL require new change proposal or release is
/// delayed": composes [`reject_change_after_freeze`].
pub fn reject_semantic_change_after_freeze(
    state: ReleaseFreezeState,
    kind: ReleaseFreezeChangeKind,
) -> Result<(), ReleaseCutoverError> {
    reject_change_after_freeze(state, kind).map_err(|error| {
        ReleaseCutoverError::SemanticChangeAfterFreeze {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Scope Confirmation
// ---------------------------------------------------------------------

/// "CUDA listed as included without passing required gates SHALL block
/// release" (`specs/v0-1-release-cutover/spec.md`): composes
/// [`reject_roadmap_feature_as_guarantee`] rather than a second deferred-
/// roadmap-feature list.
pub fn validate_v0_1_scope_feature(
    feature: &str,
    presented_as_included: bool,
) -> Result<(), ReleaseCutoverError> {
    reject_roadmap_feature_as_guarantee(feature, presented_as_included).map_err(|error| {
        ReleaseCutoverError::RoadmapFeaturePresentedAsIncluded {
            feature: feature.to_string(),
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Version Confirmation
// ---------------------------------------------------------------------

/// A single WIT package's cutover version confirmation record. Distinct
/// from [`WitInterface`] (whose `version` is always populated): this type
/// exists so "included WIT package lacks version" is representable at all,
/// implementing the "Missing WIT version" scenario
/// (`specs/v0-1-release-cutover/spec.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitPackageVersionRecord {
    pub package: String,
    pub version: Option<String>,
}

impl WitPackageVersionRecord {
    /// Builds a confirmed record from an already-versioned
    /// [`WitInterface`], e.g. from
    /// [`crate::release_packaging::release_wit_contract_versions`].
    pub fn from_interface(interface: &WitInterface) -> Self {
        Self {
            package: interface.name.clone(),
            version: Some(interface.version.clone()),
        }
    }
}

/// "Cutover SHALL confirm ... WIT ... versions": rejects any record missing
/// a version.
pub fn validate_wit_versions_confirmed(
    records: &[WitPackageVersionRecord],
) -> Result<(), ReleaseCutoverError> {
    for record in records {
        if record.version.is_none() {
            return Err(ReleaseCutoverError::WitVersionMissing {
                package: record.package.clone(),
            });
        }
    }
    Ok(())
}

/// "Version Confirmation" (`proposal.md`): the seven version confirmations
/// cutover SHALL make.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverVersionConfirmation {
    pub release_version: ReleaseVersion,
    pub crate_versions_confirmed: bool,
    pub binary_version_confirmed: bool,
    pub wit_packages: Vec<WitPackageVersionRecord>,
    pub conformance_suite_versions_confirmed: bool,
    pub openspec_baseline_version_confirmed: bool,
    /// `None` when no release candidate lineage applies to this release.
    pub release_candidate_lineage_documented: Option<bool>,
}

impl CutoverVersionConfirmation {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        if !self.crate_versions_confirmed {
            return Err(ReleaseCutoverError::VersionConfirmationIncomplete {
                missing: "crate-versions".into(),
            });
        }
        if !self.binary_version_confirmed {
            return Err(ReleaseCutoverError::VersionConfirmationIncomplete {
                missing: "binary-version".into(),
            });
        }
        validate_wit_versions_confirmed(&self.wit_packages)?;
        if !self.conformance_suite_versions_confirmed {
            return Err(ReleaseCutoverError::VersionConfirmationIncomplete {
                missing: "conformance-suite-versions".into(),
            });
        }
        if !self.openspec_baseline_version_confirmed {
            return Err(ReleaseCutoverError::VersionConfirmationIncomplete {
                missing: "openspec-baseline-version".into(),
            });
        }
        if self.release_candidate_lineage_documented == Some(false) {
            return Err(ReleaseCutoverError::VersionConfirmationIncomplete {
                missing: "release-candidate-lineage".into(),
            });
        }
        Ok(())
    }
}

/// "Runtime version metadata SHALL match release tag or documented version
/// mapping" (`specs/runtime/spec.md`).
pub fn validate_runtime_version_matches_release_tag(
    report: &ReleaseBinaryVersionReport,
    release_tag: &str,
) -> Result<(), ReleaseCutoverError> {
    if report.binary_version == release_tag {
        Ok(())
    } else {
        Err(ReleaseCutoverError::RuntimeVersionMismatch {
            reported: report.binary_version.clone(),
            tag: release_tag.to_string(),
        })
    }
}

// ---------------------------------------------------------------------
// Feature Flag Confirmation
// ---------------------------------------------------------------------

/// "Feature Flag Confirmation" (`proposal.md`): composes
/// [`reject_experimental_flag_enabled_by_default`] and additionally denies a
/// test-only or conformance-only flag enabled by default in a release
/// build, which the packaging-level check does not itself enforce.
pub fn validate_cutover_feature_flag(flag: &ReleaseFeatureFlag) -> Result<(), ReleaseCutoverError> {
    reject_experimental_flag_enabled_by_default(flag).map_err(|_| {
        ReleaseCutoverError::ExperimentalFeatureEnabledByDefault {
            flag: flag.name.clone(),
        }
    })?;
    if matches!(
        flag.class,
        ReleaseFeatureFlagClass::TestOnly | ReleaseFeatureFlagClass::ConformanceOnly
    ) && flag.enabled_by_default
    {
        return Err(ReleaseCutoverError::NonBaselineFeatureEnabledInRelease {
            flag: flag.name.clone(),
        });
    }
    Ok(())
}

/// Composes [`validate_provider_feature_flags_for_v0_1`]: only Reference CPU
/// Provider may be enabled by default.
pub fn validate_cutover_provider_feature_flags(
    flags: &[ReleaseFeatureFlag],
) -> Result<(), ReleaseCutoverError> {
    validate_provider_feature_flags_for_v0_1(flags).map_err(|error| {
        ReleaseCutoverError::NonBaselineFeatureEnabledInRelease {
            flag: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Compatibility Matrix Completion
// ---------------------------------------------------------------------

/// The twelve compatibility dimensions from "Compatibility Matrix
/// Completion" (`proposal.md`). A superset of
/// [`crate::release_packaging::CompatibilityDimension`]'s eight dimensions
/// (adding Tokenizer/Adapter Artifact metadata, supported targets, and
/// feature flags), so this is deliberately a distinct type rather than
/// reusing the narrower packaging-level one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CutoverCompatibilityDimension {
    RustPublicApi,
    RuntimeInferenceApi,
    WitPackages,
    ProviderAbi,
    ModelArtifactMetadata,
    TokenizerArtifactMetadata,
    AdapterArtifactMetadata,
    CliCommandSurface,
    ConformanceReportFormat,
    OpenSpecBaseline,
    SupportedTargets,
    FeatureFlags,
}

pub const CUTOVER_COMPATIBILITY_DIMENSIONS: &[CutoverCompatibilityDimension] = &[
    CutoverCompatibilityDimension::RustPublicApi,
    CutoverCompatibilityDimension::RuntimeInferenceApi,
    CutoverCompatibilityDimension::WitPackages,
    CutoverCompatibilityDimension::ProviderAbi,
    CutoverCompatibilityDimension::ModelArtifactMetadata,
    CutoverCompatibilityDimension::TokenizerArtifactMetadata,
    CutoverCompatibilityDimension::AdapterArtifactMetadata,
    CutoverCompatibilityDimension::CliCommandSurface,
    CutoverCompatibilityDimension::ConformanceReportFormat,
    CutoverCompatibilityDimension::OpenSpecBaseline,
    CutoverCompatibilityDimension::SupportedTargets,
    CutoverCompatibilityDimension::FeatureFlags,
];

pub const fn cutover_compatibility_dimension_id(
    dimension: CutoverCompatibilityDimension,
) -> &'static str {
    match dimension {
        CutoverCompatibilityDimension::RustPublicApi => "rust-public-api",
        CutoverCompatibilityDimension::RuntimeInferenceApi => "runtime-inference-api",
        CutoverCompatibilityDimension::WitPackages => "wit-packages",
        CutoverCompatibilityDimension::ProviderAbi => "provider-abi",
        CutoverCompatibilityDimension::ModelArtifactMetadata => "model-artifact-metadata",
        CutoverCompatibilityDimension::TokenizerArtifactMetadata => "tokenizer-artifact-metadata",
        CutoverCompatibilityDimension::AdapterArtifactMetadata => "adapter-artifact-metadata",
        CutoverCompatibilityDimension::CliCommandSurface => "cli-command-surface",
        CutoverCompatibilityDimension::ConformanceReportFormat => "conformance-report-format",
        CutoverCompatibilityDimension::OpenSpecBaseline => "openspec-baseline",
        CutoverCompatibilityDimension::SupportedTargets => "supported-targets",
        CutoverCompatibilityDimension::FeatureFlags => "feature-flags",
    }
}

/// The six-value approved status vocabulary from "Compatibility Matrix
/// Completion" (`proposal.md`): `stable-for-v0.1-baseline`, `preview`,
/// `experimental`, `unstable`, `deferred`, `unsupported`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoverCompatibilityStatus {
    StableForV01Baseline,
    Preview,
    Experimental,
    Unstable,
    Deferred,
    Unsupported,
}

pub const fn cutover_compatibility_status_id(status: CutoverCompatibilityStatus) -> &'static str {
    match status {
        CutoverCompatibilityStatus::StableForV01Baseline => "stable-for-v0.1-baseline",
        CutoverCompatibilityStatus::Preview => "preview",
        CutoverCompatibilityStatus::Experimental => "experimental",
        CutoverCompatibilityStatus::Unstable => "unstable",
        CutoverCompatibilityStatus::Deferred => "deferred",
        CutoverCompatibilityStatus::Unsupported => "unsupported",
    }
}

/// "Compatibility Matrix Complete" (`specs/v0-1-release-cutover/spec.md`):
/// every [`CutoverCompatibilityDimension`] SHALL have an explicit status.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CutoverCompatibilityMatrix {
    pub status: BTreeMap<&'static str, CutoverCompatibilityStatus>,
}

impl CutoverCompatibilityMatrix {
    pub fn set(
        &mut self,
        dimension: CutoverCompatibilityDimension,
        status: CutoverCompatibilityStatus,
    ) {
        self.status
            .insert(cutover_compatibility_dimension_id(dimension), status);
    }

    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
            if !self
                .status
                .contains_key(cutover_compatibility_dimension_id(*dimension))
            {
                return Err(ReleaseCutoverError::CompatibilityDimensionMissing {
                    dimension: cutover_compatibility_dimension_id(*dimension).to_string(),
                });
            }
        }
        Ok(())
    }
}

/// "Experimental API presented stable SHALL block release"
/// (`specs/v0-1-release-cutover/spec.md`): rejects presenting anything less
/// stable than [`CutoverCompatibilityStatus::StableForV01Baseline`] as if it
/// were `StableForV01Baseline`.
pub fn reject_status_misrepresentation(
    actual_status: CutoverCompatibilityStatus,
    presented_status: CutoverCompatibilityStatus,
    subject: &str,
) -> Result<(), ReleaseCutoverError> {
    let actual_not_stable = !matches!(
        actual_status,
        CutoverCompatibilityStatus::StableForV01Baseline
    );
    let presented_stable = matches!(
        presented_status,
        CutoverCompatibilityStatus::StableForV01Baseline
    );
    if actual_not_stable && presented_stable {
        return Err(ReleaseCutoverError::StatusMisrepresented {
            subject: subject.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Required Gate Execution
// ---------------------------------------------------------------------

/// "Required Gates Executed" (`specs/v0-1-release-cutover/spec.md`):
/// composes [`release_may_publish_stable`] directly rather than a parallel
/// gate-completeness check.
pub fn validate_required_gates_executed(
    results: &[ReleaseGateResult],
) -> Result<(), ReleaseCutoverError> {
    release_may_publish_stable(results).map_err(|error| {
        ReleaseCutoverError::RequiredGateFailedOrMissing {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Skip Review
// ---------------------------------------------------------------------

/// A single gate skip record, implementing "Skip Review" (`proposal.md`): a
/// skip is allowed only when every field below holds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GateSkip {
    pub gate: String,
    pub outside_v0_1_scope: bool,
    pub reason: Option<String>,
    pub hides_baseline_failure: bool,
    pub included_in_release_report: bool,
}

impl GateSkip {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        let reason_documented = self
            .reason
            .as_deref()
            .is_some_and(|reason| !reason.trim().is_empty());
        let allowed = self.outside_v0_1_scope
            && reason_documented
            && !self.hides_baseline_failure
            && self.included_in_release_report;
        if allowed {
            Ok(())
        } else {
            Err(ReleaseCutoverError::DisallowedGateSkip {
                gate: self.gate.clone(),
            })
        }
    }
}

/// "Disallowed skips SHALL block release": every [`GateSkip`] SHALL be
/// individually allowed.
pub fn validate_gate_skips(skips: &[GateSkip]) -> Result<(), ReleaseCutoverError> {
    for skip in skips {
        skip.validate()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Exception Review
// ---------------------------------------------------------------------

/// A single cutover exception, implementing "Exception Review"
/// (`proposal.md`). Composes [`SecurityException`] (which already carries
/// issue/component/severity/rationale/mitigation/owner/expiration/release-
/// note fields) rather than a parallel exception record, adding only the
/// `gate` field the cutover checklist additionally names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverException {
    pub gate: String,
    pub exception: SecurityException,
}

impl CutoverException {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        if self.gate.trim().is_empty() {
            return Err(ReleaseCutoverError::UndocumentedException {
                gate: self.gate.clone(),
            });
        }
        self.exception
            .validate()
            .map_err(|_| ReleaseCutoverError::UndocumentedException {
                gate: self.gate.clone(),
            })
    }
}

/// "Undocumented exceptions SHALL block release."
pub fn reject_undocumented_cutover_exception(
    exception_required: bool,
    exception: Option<&CutoverException>,
) -> Result<(), ReleaseCutoverError> {
    if !exception_required {
        return Ok(());
    }
    match exception {
        Some(exception) => exception.validate(),
        None => Err(ReleaseCutoverError::UndocumentedException {
            gate: String::new(),
        }),
    }
}

/// Validates a whole exception list; the first undocumented exception
/// blocks release.
pub fn validate_cutover_exceptions(
    exceptions: &[CutoverException],
) -> Result<(), ReleaseCutoverError> {
    for exception in exceptions {
        exception.validate()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Security Verification
// ---------------------------------------------------------------------

/// "Security Verified" (`specs/v0-1-release-cutover/spec.md`): composes
/// [`ReleaseSecurityGateInputs`] / [`evaluate_release_security_blocking`]
/// and [`SecurityReleaseNotes`] rather than re-checking dependency audit,
/// license audit, secret scan, redaction, native handle, trust/integrity,
/// artifact integrity, checksum, SBOM, or signature status a second time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverSecurityVerification {
    pub gate_inputs: ReleaseSecurityGateInputs,
    pub security_notes: SecurityReleaseNotes,
}

impl CutoverSecurityVerification {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        evaluate_release_security_blocking(&self.gate_inputs).map_err(|error| {
            ReleaseCutoverError::SecurityVerificationFailed {
                reason: error.to_string(),
            }
        })?;
        self.security_notes.validate().map_err(|error| {
            ReleaseCutoverError::SecurityVerificationFailed {
                reason: error.to_string(),
            }
        })
    }
}

// ---------------------------------------------------------------------
// Artifact Generation
// ---------------------------------------------------------------------

/// "Artifacts Generated" (`specs/v0-1-release-cutover/spec.md`): composes
/// [`ReleaseArtifactManifest::validate`] directly.
pub fn validate_cutover_artifacts_generated(
    manifest: &ReleaseArtifactManifest,
) -> Result<(), ReleaseCutoverError> {
    manifest
        .validate()
        .map_err(|error| ReleaseCutoverError::ArtifactGenerationIncomplete {
            reason: error.to_string(),
        })
}

// ---------------------------------------------------------------------
// Artifact Verification
// ---------------------------------------------------------------------

/// "Artifacts Verified" (`specs/v0-1-release-cutover/spec.md`): composes
/// [`ArtifactIntegrityStatus`] (release_security's five integrity checks,
/// which already cover clean source state, tag/source match, OpenSpec
/// report match, conformance-report/commit match, and checksum match) and
/// adds only the two cutover-specific checks that struct does not carry:
/// release notes matching the compatibility matrix, and artifact names
/// including the version where appropriate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverArtifactVerification {
    pub integrity: ArtifactIntegrityStatus,
    pub release_notes_match_compatibility_matrix: bool,
    pub artifact_names_include_version: bool,
}

impl CutoverArtifactVerification {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        self.integrity.validate().map_err(|error| {
            ReleaseCutoverError::ArtifactVerificationFailed {
                reason: error.to_string(),
            }
        })?;
        if !self.release_notes_match_compatibility_matrix {
            return Err(ReleaseCutoverError::ArtifactVerificationFailed {
                reason: "release notes do not match the compatibility matrix".into(),
            });
        }
        if !self.artifact_names_include_version {
            return Err(ReleaseCutoverError::ArtifactVerificationFailed {
                reason: "artifact names do not include the release version".into(),
            });
        }
        Ok(())
    }
}

/// "Checksum mismatch SHALL block release or withdraw it"
/// (`specs/v0-1-release-cutover/spec.md`): composes
/// [`verify_checksum_matches_final_artifact`].
pub fn verify_cutover_artifact_checksum(
    checksum: &ArtifactChecksum,
    recomputed_digest: &str,
) -> Result<(), ReleaseCutoverError> {
    verify_checksum_matches_final_artifact(checksum, recomputed_digest).map_err(|error| {
        ReleaseCutoverError::ArtifactVerificationFailed {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Changelog Completion
// ---------------------------------------------------------------------

/// "Changelog Complete" (`specs/v0-1-release-cutover/spec.md`): composes
/// [`ReleaseChangelog::validate`] (non-empty) and additionally requires the
/// nine cutover-specific categories the checklist names (a superset of
/// [`crate::release_packaging::ChangelogEntryKind`]'s eight categories,
/// adding release scope).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoverChangelogChecklist {
    pub changelog: ReleaseChangelog,
    pub includes_added_contracts: bool,
    pub includes_changed_contracts: bool,
    pub includes_removed_or_deprecated_contracts: bool,
    pub includes_release_scope: bool,
    pub includes_known_limitations: bool,
    pub includes_compatibility_status: bool,
    pub includes_security_notes: bool,
    pub includes_conformance_status: bool,
    pub includes_deferred_roadmap_items: bool,
}

impl CutoverChangelogChecklist {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        self.changelog
            .validate()
            .map_err(|error| ReleaseCutoverError::ChangelogIncomplete {
                missing: error.to_string(),
            })?;
        let checks: [(&str, bool); 8] = [
            ("added-contracts", self.includes_added_contracts),
            ("changed-contracts", self.includes_changed_contracts),
            (
                "removed-or-deprecated-contracts",
                self.includes_removed_or_deprecated_contracts,
            ),
            ("release-scope", self.includes_release_scope),
            ("known-limitations", self.includes_known_limitations),
            ("compatibility-status", self.includes_compatibility_status),
            ("security-notes", self.includes_security_notes),
            ("conformance-status", self.includes_conformance_status),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::ChangelogIncomplete {
                    missing: name.into(),
                });
            }
        }
        if !self.includes_deferred_roadmap_items {
            return Err(ReleaseCutoverError::ChangelogIncomplete {
                missing: "deferred-roadmap-items".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Release Notes Completion
// ---------------------------------------------------------------------

/// "Release Notes Complete" (`specs/v0-1-release-cutover/spec.md`): the
/// thirteen topics release notes SHALL answer or include.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CutoverReleaseNotesChecklist {
    pub explains_what_v0_1_is: bool,
    pub explains_what_users_can_run: bool,
    pub explains_stable_status: bool,
    pub explains_preview_status: bool,
    pub explains_experimental_status: bool,
    pub explains_deferred_status: bool,
    pub explains_unsupported_status: bool,
    pub explains_artifact_verification: bool,
    pub explains_security_limitations: bool,
    pub explains_how_to_run_conformance: bool,
    pub includes_compatibility_matrix: bool,
    pub includes_security_notes: bool,
    pub includes_known_limitations: bool,
}

impl CutoverReleaseNotesChecklist {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        let checks: [(&str, bool); 13] = [
            ("what-v0.1-is", self.explains_what_v0_1_is),
            ("what-users-can-run", self.explains_what_users_can_run),
            ("stable-status", self.explains_stable_status),
            ("preview-status", self.explains_preview_status),
            ("experimental-status", self.explains_experimental_status),
            ("deferred-status", self.explains_deferred_status),
            ("unsupported-status", self.explains_unsupported_status),
            ("artifact-verification", self.explains_artifact_verification),
            ("security-limitations", self.explains_security_limitations),
            (
                "how-to-run-conformance",
                self.explains_how_to_run_conformance,
            ),
            ("compatibility-matrix", self.includes_compatibility_matrix),
            ("security-notes", self.includes_security_notes),
            ("known-limitations", self.includes_known_limitations),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::ReleaseNotesIncomplete {
                    missing: name.into(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Tagging
// ---------------------------------------------------------------------

/// "Tagging After Gates" (`specs/v0-1-release-cutover/spec.md`): "Stable
/// release tag SHALL be created only after required gates pass". Composes
/// [`release_may_publish_stable`] rather than a parallel gate-completeness
/// check.
pub fn validate_tag_after_gates(
    gate_results: &[ReleaseGateResult],
    tag_created: bool,
) -> Result<(), ReleaseCutoverError> {
    if !tag_created {
        return Ok(());
    }
    release_may_publish_stable(gate_results)
        .map_err(|_| ReleaseCutoverError::TagCreatedBeforeGatesPassed)
}

// ---------------------------------------------------------------------
// Publication
// ---------------------------------------------------------------------

/// "Publication Preserves Scope" (`specs/v0-1-release-cutover/spec.md`):
/// "Server API claimed included, but server API gates were skipped as
/// deferred, SHALL block or correct release". Composes
/// [`reject_roadmap_feature_as_guarantee`] (roadmap-feature inclusion) and
/// [`reject_status_misrepresentation`] (stability misrepresentation)
/// instead of a parallel publication-boundary check.
pub fn validate_publication_scope_preserved(
    feature: &str,
    presented_as_included: bool,
    actual_status: CutoverCompatibilityStatus,
    presented_status: CutoverCompatibilityStatus,
) -> Result<(), ReleaseCutoverError> {
    validate_v0_1_scope_feature(feature, presented_as_included).map_err(|error| {
        ReleaseCutoverError::PublicationScopeViolation {
            reason: error.to_string(),
        }
    })?;
    reject_status_misrepresentation(actual_status, presented_status, feature).map_err(|error| {
        ReleaseCutoverError::PublicationScopeViolation {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Post-Publication Verification
// ---------------------------------------------------------------------

/// "Post-Publication Verification" (`proposal.md`): the eight checks
/// post-publication verification SHOULD confirm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PostPublicationVerification {
    pub published_artifacts_match_checksums: bool,
    pub release_notes_visible: bool,
    pub reports_accessible: bool,
    pub version_command_matches_tag: bool,
    pub documentation_links_valid: bool,
    pub compatibility_matrix_visible: bool,
    pub security_notes_visible: bool,
    pub deferred_roadmap_clearly_separated: bool,
}

impl PostPublicationVerification {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        let checks: [(&str, bool); 8] = [
            (
                "published-artifacts-match-checksums",
                self.published_artifacts_match_checksums,
            ),
            ("release-notes-visible", self.release_notes_visible),
            ("reports-accessible", self.reports_accessible),
            (
                "version-command-matches-tag",
                self.version_command_matches_tag,
            ),
            ("documentation-links-valid", self.documentation_links_valid),
            (
                "compatibility-matrix-visible",
                self.compatibility_matrix_visible,
            ),
            ("security-notes-visible", self.security_notes_visible),
            (
                "deferred-roadmap-clearly-separated",
                self.deferred_roadmap_clearly_separated,
            ),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::PostPublicationVerificationFailed {
                    check: name.into(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Rollback And Retraction Notes
// ---------------------------------------------------------------------

/// "Rollback And Retraction Notes" (`proposal.md`): the five steps the
/// release process SHOULD describe for an invalid published release.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RollbackRetractionNotes {
    pub withdrawal_procedure_documented: bool,
    pub advisory_publication_procedure_documented: bool,
    pub patch_release_procedure_documented: bool,
    pub audit_trail_preservation_documented: bool,
    pub release_notes_update_procedure_documented: bool,
}

impl RollbackRetractionNotes {
    pub fn validate(&self) -> Result<(), ReleaseCutoverError> {
        let checks: [(&str, bool); 5] = [
            ("withdrawal-procedure", self.withdrawal_procedure_documented),
            (
                "advisory-publication-procedure",
                self.advisory_publication_procedure_documented,
            ),
            (
                "patch-release-procedure",
                self.patch_release_procedure_documented,
            ),
            (
                "audit-trail-preservation",
                self.audit_trail_preservation_documented,
            ),
            (
                "release-notes-update-procedure",
                self.release_notes_update_procedure_documented,
            ),
        ];
        for (name, ok) in checks {
            if !ok {
                return Err(ReleaseCutoverError::RollbackNotesIncomplete {
                    missing: name.into(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Post-v0.1 Handoff
// ---------------------------------------------------------------------

/// Post-`v0.1` roadmap handoff candidates named in "Post-v0.1 Handoff"
/// (`proposal.md`).
pub const POST_V0_1_HANDOFF_CANDIDATES: &[&str] = &[
    "implementation-hardening",
    "optimized-cpu-provider",
    "model-format-support",
    "source-cache-implementation",
    "server-api-implementation",
    "production-cli-ux",
    "cuda-metal-openvino-qnn-webgpu-exploration",
    "quantized-inference",
    "advanced-attention",
];

/// A single post-`v0.1` handoff item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostV01HandoffItem {
    pub name: String,
    pub presented_as_v0_1_release_claim: bool,
}

/// "Post-v0.1 items SHALL remain separate from `v0.1` release claims."
pub fn reject_post_v0_1_item_as_release_claim(
    item: &PostV01HandoffItem,
) -> Result<(), ReleaseCutoverError> {
    if item.presented_as_v0_1_release_claim {
        return Err(ReleaseCutoverError::PostV01ItemPresentedAsReleaseClaim {
            item: item.name.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Final Release Statement
// ---------------------------------------------------------------------

/// The exact final release statement from "Final Release Statement"
/// (`proposal.md`).
pub const V0_1_FINAL_RELEASE_STATEMENT: &str = "Magnetar v0.1 is a CPU-local inference runtime baseline. It validates the first end-to-end inference path through Runtime Inference API, Reference CPU Provider, and E2E local conformance. Post-baseline roadmap features are not included unless explicitly marked.";

/// "It accurately describes included baseline and excludes roadmap
/// features" (`specs/v0-1-release-cutover/spec.md`): rejects a statement
/// missing any of the four required phrases below.
pub fn validate_final_release_statement(statement: &str) -> Result<(), ReleaseCutoverError> {
    let normalized = statement.to_ascii_lowercase();
    let required_fragments = [
        "cpu-local",
        "runtime inference api",
        "reference cpu provider",
        "e2e local conformance",
    ];
    for fragment in required_fragments {
        if !normalized.contains(fragment) {
            return Err(ReleaseCutoverError::FinalReleaseStatementInvalid {
                reason: format!("missing required phrase: {fragment}"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// A single correlatable cutover observation, implementing "Cutover
/// Observability Is Redacted" and "Cutover Events Are Correlatable"
/// (`specs/observability/spec.md`). Composes [`CorrelationId`] for
/// correlation and [`record_release_security_observation`] for default
/// redaction rather than a third redaction implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseCutoverObservation {
    pub correlation_id: CorrelationId,
    pub gate: Option<String>,
    pub target: Option<String>,
    pub feature_set: Vec<String>,
    pub artifact: Option<String>,
    pub release_metadata: Option<String>,
    pub security_observation: ReleaseSecurityObservation,
}

/// Input to [`record_release_cutover_observation`], bundled into a struct
/// (rather than eight positional parameters) per the recorded fields of
/// [`ReleaseCutoverObservation`].
pub struct ReleaseCutoverObservationInput<'a> {
    pub correlation_id: CorrelationId,
    pub kind: ReleaseSecurityObservationKind,
    pub gate: Option<String>,
    pub target: Option<String>,
    pub feature_set: Vec<String>,
    pub artifact: Option<String>,
    pub release_metadata: Option<String>,
    pub raw_detail: &'a str,
}

/// Records a cutover observation, always redacting `raw_detail` through
/// [`record_release_security_observation`] before attaching it.
pub fn record_release_cutover_observation(
    input: ReleaseCutoverObservationInput<'_>,
) -> ReleaseCutoverObservation {
    ReleaseCutoverObservation {
        correlation_id: input.correlation_id,
        gate: input.gate,
        target: input.target,
        feature_set: input.feature_set,
        artifact: input.artifact,
        release_metadata: input.release_metadata,
        security_observation: record_release_security_observation(input.kind, input.raw_detail),
    }
}

// ---------------------------------------------------------------------
// CLI Boundary / Runtime Scope Cutover Checks
// ---------------------------------------------------------------------

/// "CLI boundary gate failed SHALL block release"
/// (`specs/cli-boundary/spec.md`): composes
/// [`validate_cli_authority_not_delegated_to_runtime`] (which itself already
/// composes `crate::cli_boundary::reject_cli_owned_authority`) rather than
/// importing the CLI boundary module a second time.
pub fn validate_cutover_cli_boundary(capability: &str) -> Result<(), ReleaseCutoverError> {
    validate_cli_authority_not_delegated_to_runtime(capability).map_err(|error| {
        ReleaseCutoverError::RuntimeScopeViolation {
            reason: error.to_string(),
        }
    })
}

/// "Runtime includes tool execution SHALL block release"
/// (`specs/runtime/spec.md`): composes
/// [`validate_runtime_inference_api_security`] (which itself already
/// composes `crate::inference_api::validate_inference_scope`) rather than
/// importing the Runtime Inference API module a second time.
pub fn validate_cutover_runtime_scope(capability: &str) -> Result<(), ReleaseCutoverError> {
    validate_runtime_inference_api_security(capability).map_err(|error| {
        ReleaseCutoverError::RuntimeScopeViolation {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured release cutover error, covering every failure category this
/// module's validation functions can produce.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReleaseCutoverError {
    ReleaseReadinessIncomplete { missing: String },
    OpenSpecNotFrozen { reason: String },
    SemanticChangeAfterFreeze { reason: String },
    RoadmapFeaturePresentedAsIncluded { feature: String, reason: String },
    WitVersionMissing { package: String },
    VersionConfirmationIncomplete { missing: String },
    RuntimeVersionMismatch { reported: String, tag: String },
    ExperimentalFeatureEnabledByDefault { flag: String },
    NonBaselineFeatureEnabledInRelease { flag: String },
    CompatibilityDimensionMissing { dimension: String },
    StatusMisrepresented { subject: String },
    RequiredGateFailedOrMissing { reason: String },
    DisallowedGateSkip { gate: String },
    UndocumentedException { gate: String },
    SecurityVerificationFailed { reason: String },
    ArtifactGenerationIncomplete { reason: String },
    ArtifactVerificationFailed { reason: String },
    ChangelogIncomplete { missing: String },
    ReleaseNotesIncomplete { missing: String },
    TagCreatedBeforeGatesPassed,
    PublicationScopeViolation { reason: String },
    PostPublicationVerificationFailed { check: String },
    RollbackNotesIncomplete { missing: String },
    PostV01ItemPresentedAsReleaseClaim { item: String },
    FinalReleaseStatementInvalid { reason: String },
    RuntimeScopeViolation { reason: String },
    ReleaseCutoverBlocked { reasons: Vec<String> },
    InternalReleaseCutoverError { reason: String },
}

impl ReleaseCutoverError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ReleaseReadinessIncomplete { .. } => "release-readiness-incomplete",
            Self::OpenSpecNotFrozen { .. } => "openspec-not-frozen",
            Self::SemanticChangeAfterFreeze { .. } => "semantic-change-after-freeze",
            Self::RoadmapFeaturePresentedAsIncluded { .. } => {
                "roadmap-feature-presented-as-included"
            }
            Self::WitVersionMissing { .. } => "wit-version-missing",
            Self::VersionConfirmationIncomplete { .. } => "version-confirmation-incomplete",
            Self::RuntimeVersionMismatch { .. } => "runtime-version-mismatch",
            Self::ExperimentalFeatureEnabledByDefault { .. } => {
                "experimental-feature-enabled-by-default"
            }
            Self::NonBaselineFeatureEnabledInRelease { .. } => {
                "non-baseline-feature-enabled-in-release"
            }
            Self::CompatibilityDimensionMissing { .. } => "compatibility-dimension-missing",
            Self::StatusMisrepresented { .. } => "status-misrepresented",
            Self::RequiredGateFailedOrMissing { .. } => "required-gate-failed-or-missing",
            Self::DisallowedGateSkip { .. } => "disallowed-gate-skip",
            Self::UndocumentedException { .. } => "undocumented-exception",
            Self::SecurityVerificationFailed { .. } => "security-verification-failed",
            Self::ArtifactGenerationIncomplete { .. } => "artifact-generation-incomplete",
            Self::ArtifactVerificationFailed { .. } => "artifact-verification-failed",
            Self::ChangelogIncomplete { .. } => "changelog-incomplete",
            Self::ReleaseNotesIncomplete { .. } => "release-notes-incomplete",
            Self::TagCreatedBeforeGatesPassed => "tag-created-before-gates-passed",
            Self::PublicationScopeViolation { .. } => "publication-scope-violation",
            Self::PostPublicationVerificationFailed { .. } => {
                "post-publication-verification-failed"
            }
            Self::RollbackNotesIncomplete { .. } => "rollback-notes-incomplete",
            Self::PostV01ItemPresentedAsReleaseClaim { .. } => {
                "post-v0-1-item-presented-as-release-claim"
            }
            Self::FinalReleaseStatementInvalid { .. } => "final-release-statement-invalid",
            Self::RuntimeScopeViolation { .. } => "runtime-scope-violation",
            Self::ReleaseCutoverBlocked { .. } => "release-cutover-blocked",
            Self::InternalReleaseCutoverError { .. } => "internal-release-cutover-error",
        }
    }
}

impl fmt::Display for ReleaseCutoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReleaseReadinessIncomplete { missing }
            | Self::VersionConfirmationIncomplete { missing }
            | Self::ChangelogIncomplete { missing }
            | Self::ReleaseNotesIncomplete { missing }
            | Self::RollbackNotesIncomplete { missing } => {
                write!(f, "{}: {missing}", self.id())
            }
            Self::OpenSpecNotFrozen { reason }
            | Self::SemanticChangeAfterFreeze { reason }
            | Self::RequiredGateFailedOrMissing { reason }
            | Self::SecurityVerificationFailed { reason }
            | Self::ArtifactGenerationIncomplete { reason }
            | Self::ArtifactVerificationFailed { reason }
            | Self::PublicationScopeViolation { reason }
            | Self::FinalReleaseStatementInvalid { reason }
            | Self::RuntimeScopeViolation { reason }
            | Self::InternalReleaseCutoverError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::RoadmapFeaturePresentedAsIncluded { feature, reason } => {
                write!(f, "{}: {feature} ({reason})", self.id())
            }
            Self::WitVersionMissing { package } => write!(f, "{}: {package}", self.id()),
            Self::RuntimeVersionMismatch { reported, tag } => {
                write!(f, "{}: {reported} != {tag}", self.id())
            }
            Self::ExperimentalFeatureEnabledByDefault { flag }
            | Self::NonBaselineFeatureEnabledInRelease { flag } => {
                write!(f, "{}: {flag}", self.id())
            }
            Self::CompatibilityDimensionMissing { dimension } => {
                write!(f, "{}: {dimension}", self.id())
            }
            Self::StatusMisrepresented { subject } => write!(f, "{}: {subject}", self.id()),
            Self::DisallowedGateSkip { gate } | Self::UndocumentedException { gate } => {
                write!(f, "{}: {gate}", self.id())
            }
            Self::TagCreatedBeforeGatesPassed => write!(f, "{}", self.id()),
            Self::PostPublicationVerificationFailed { check } => {
                write!(f, "{}: {check}", self.id())
            }
            Self::PostV01ItemPresentedAsReleaseClaim { item } => {
                write!(f, "{}: {item}", self.id())
            }
            Self::ReleaseCutoverBlocked { reasons } => {
                write!(f, "{}: {}", self.id(), reasons.join(", "))
            }
        }
    }
}

impl Error for ReleaseCutoverError {}

// ---------------------------------------------------------------------
// Release Blocking Criteria
// ---------------------------------------------------------------------

/// The top-level cutover blocking-criteria inputs, in the shape of
/// [`ReleaseSecurityGateInputs`]: each field is `true` when the
/// corresponding checklist section above returned an error.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReleaseCutoverGateInputs {
    pub readiness_incomplete: bool,
    pub openspec_not_frozen: bool,
    pub scope_violation: bool,
    pub version_confirmation_incomplete: bool,
    pub feature_flag_violation: bool,
    pub compatibility_matrix_incomplete: bool,
    pub required_gate_failed_or_missing: bool,
    pub disallowed_gate_skip: bool,
    pub undocumented_exception: bool,
    pub security_verification_failed: bool,
    pub artifact_generation_incomplete: bool,
    pub artifact_verification_failed: bool,
    pub changelog_incomplete: bool,
    pub release_notes_incomplete: bool,
    pub tag_created_before_gates_passed: bool,
    pub publication_scope_violation: bool,
    pub post_publication_verification_failed: bool,
}

/// Evaluates every [`ReleaseCutoverGateInputs`] field and, if any are
/// `true`, returns [`ReleaseCutoverError::ReleaseCutoverBlocked`] naming
/// every triggered reason (not just the first) -- the "Cutover Principle"
/// (`proposal.md`): "A stable v0.1 release SHALL be cut only after required
/// gates pass and release metadata is complete."
pub fn evaluate_release_cutover(
    inputs: &ReleaseCutoverGateInputs,
) -> Result<(), ReleaseCutoverError> {
    let mut reasons = Vec::new();
    if inputs.readiness_incomplete {
        reasons.push("readiness-incomplete");
    }
    if inputs.openspec_not_frozen {
        reasons.push("openspec-not-frozen");
    }
    if inputs.scope_violation {
        reasons.push("scope-violation");
    }
    if inputs.version_confirmation_incomplete {
        reasons.push("version-confirmation-incomplete");
    }
    if inputs.feature_flag_violation {
        reasons.push("feature-flag-violation");
    }
    if inputs.compatibility_matrix_incomplete {
        reasons.push("compatibility-matrix-incomplete");
    }
    if inputs.required_gate_failed_or_missing {
        reasons.push("required-gate-failed-or-missing");
    }
    if inputs.disallowed_gate_skip {
        reasons.push("disallowed-gate-skip");
    }
    if inputs.undocumented_exception {
        reasons.push("undocumented-exception");
    }
    if inputs.security_verification_failed {
        reasons.push("security-verification-failed");
    }
    if inputs.artifact_generation_incomplete {
        reasons.push("artifact-generation-incomplete");
    }
    if inputs.artifact_verification_failed {
        reasons.push("artifact-verification-failed");
    }
    if inputs.changelog_incomplete {
        reasons.push("changelog-incomplete");
    }
    if inputs.release_notes_incomplete {
        reasons.push("release-notes-incomplete");
    }
    if inputs.tag_created_before_gates_passed {
        reasons.push("tag-created-before-gates-passed");
    }
    if inputs.publication_scope_violation {
        reasons.push("publication-scope-violation");
    }
    if inputs.post_publication_verification_failed {
        reasons.push("post-publication-verification-failed");
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(ReleaseCutoverError::ReleaseCutoverBlocked {
            reasons: reasons.into_iter().map(String::from).collect(),
        })
    }
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single release cutover conformance check result, mirroring
/// [`crate::release_security::ReleaseSecurityConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseCutoverConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ReleaseCutoverConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseCutoverConformanceReport {
    pub results: Vec<ReleaseCutoverConformanceResult>,
}

impl ReleaseCutoverConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ReleaseCutoverConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ReleaseCutoverConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the release cutover conformance checks described in this module's
/// doc comment: incomplete readiness, an unfrozen baseline, a roadmap
/// feature presented as included, a missing WIT version, a runtime version
/// mismatch, an experimental or non-baseline flag enabled by default, a
/// missing compatibility dimension, a misrepresented stability status, a
/// missing/failed required gate, a disallowed gate skip, an undocumented
/// exception, a failed security verification, incomplete artifact
/// generation or verification, an incomplete changelog or release notes, a
/// tag created before gates pass, a publication scope violation, a failed
/// post-publication verification, incomplete rollback notes, a post-`v0.1`
/// item presented as a release claim, an invalid final release statement, a
/// denied CLI/Runtime authority expansion, and a redacted, correlatable
/// cutover observation are all rejected or accepted as documented above.
pub fn run_release_cutover_conformance() -> ReleaseCutoverConformanceReport {
    let mut results = Vec::new();

    {
        let incomplete = ReleaseReadinessChecklist::default();
        record(
            &mut results,
            "missing release notes draft blocks cutover readiness",
            matches!(
                incomplete.validate(),
                Err(ReleaseCutoverError::ReleaseReadinessIncomplete { .. })
            ),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
        let complete = ReleaseReadinessChecklist {
            release_branch_or_commit_selected: true,
            version_selected: true,
            openspec_baseline_selected: true,
            scope_selected: true,
            gates_selected: true,
            artifacts_selected: true,
            release_notes_draft_exists: true,
            compatibility_matrix_draft_exists: true,
            security_notes_draft_exists: true,
        };
        record(
            &mut results,
            "complete release readiness is accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let semantic_change_after_freeze = reject_semantic_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::SemanticContractChange,
        );
        record(
            &mut results,
            "a semantic change after freeze is blocked",
            matches!(
                semantic_change_after_freeze,
                Err(ReleaseCutoverError::SemanticChangeAfterFreeze { .. })
            ),
            format!("unexpected outcome: {semantic_change_after_freeze:?}"),
        );
        let doc_clarification_after_freeze = reject_semantic_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::DocumentationClarification,
        );
        record(
            &mut results,
            "a documentation clarification after freeze is allowed",
            doc_clarification_after_freeze.is_ok(),
            format!("unexpected outcome: {doc_clarification_after_freeze:?}"),
        );
    }

    {
        let cuda_included = validate_v0_1_scope_feature("cuda", true);
        record(
            &mut results,
            "CUDA listed as included is blocked",
            matches!(
                cuda_included,
                Err(ReleaseCutoverError::RoadmapFeaturePresentedAsIncluded { .. })
            ),
            format!("unexpected outcome: {cuda_included:?}"),
        );
        let baseline_included = validate_v0_1_scope_feature("reference-cpu-provider", true);
        record(
            &mut results,
            "the CPU-local baseline can be presented as included",
            baseline_included.is_ok(),
            format!("unexpected outcome: {baseline_included:?}"),
        );
    }

    {
        let missing = validate_wit_versions_confirmed(&[WitPackageVersionRecord {
            package: "magnetar:compute".into(),
            version: None,
        }]);
        record(
            &mut results,
            "a WIT package lacking a version is blocked",
            matches!(missing, Err(ReleaseCutoverError::WitVersionMissing { .. })),
            format!("unexpected outcome: {missing:?}"),
        );
        let present = validate_wit_versions_confirmed(&[WitPackageVersionRecord {
            package: "magnetar:compute".into(),
            version: Some("2.0.0".into()),
        }]);
        record(
            &mut results,
            "a WIT package with a confirmed version is accepted",
            present.is_ok(),
            format!("unexpected outcome: {present:?}"),
        );
    }

    {
        let report = crate::build_release_binary_version_report(
            ReleaseVersion::new(0, 1, 0),
            vec!["reference-cpu-provider".into()],
            "release",
            None,
        );
        let mismatch = validate_runtime_version_matches_release_tag(&report, "0.1.0-rc.1");
        record(
            &mut results,
            "a runtime version that does not match the release tag is blocked",
            matches!(
                mismatch,
                Err(ReleaseCutoverError::RuntimeVersionMismatch { .. })
            ),
            format!("unexpected outcome: {mismatch:?}"),
        );
        let matching = validate_runtime_version_matches_release_tag(&report, "0.1.0");
        record(
            &mut results,
            "a runtime version that matches the release tag is accepted",
            matching.is_ok(),
            format!("unexpected outcome: {matching:?}"),
        );
    }

    {
        let webgpu = ReleaseFeatureFlag {
            name: "webgpu-provider".into(),
            class: ReleaseFeatureFlagClass::Experimental,
            enabled_by_default: true,
        };
        let outcome = validate_cutover_feature_flag(&webgpu);
        record(
            &mut results,
            "experimental WebGPU enabled by default is blocked",
            matches!(
                outcome,
                Err(ReleaseCutoverError::ExperimentalFeatureEnabledByDefault { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let test_only = ReleaseFeatureFlag {
            name: "test-harness".into(),
            class: ReleaseFeatureFlagClass::TestOnly,
            enabled_by_default: true,
        };
        let test_only_outcome = validate_cutover_feature_flag(&test_only);
        record(
            &mut results,
            "a test-only flag enabled by default in a release build is blocked",
            matches!(
                test_only_outcome,
                Err(ReleaseCutoverError::NonBaselineFeatureEnabledInRelease { .. })
            ),
            format!("unexpected outcome: {test_only_outcome:?}"),
        );
    }

    {
        let matrix = CutoverCompatibilityMatrix::default();
        record(
            &mut results,
            "Provider ABI missing from the compatibility matrix is blocked",
            matches!(
                matrix.validate(),
                Err(ReleaseCutoverError::CompatibilityDimensionMissing { .. })
            ),
            format!("unexpected outcome: {:?}", matrix.validate()),
        );
        let mut complete = CutoverCompatibilityMatrix::default();
        for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
            complete.set(*dimension, CutoverCompatibilityStatus::StableForV01Baseline);
        }
        record(
            &mut results,
            "a compatibility matrix with every dimension marked is accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let misrepresented = reject_status_misrepresentation(
            CutoverCompatibilityStatus::Experimental,
            CutoverCompatibilityStatus::StableForV01Baseline,
            "runtime-inference-api",
        );
        record(
            &mut results,
            "an experimental API presented as stable is blocked",
            matches!(
                misrepresented,
                Err(ReleaseCutoverError::StatusMisrepresented { .. })
            ),
            format!("unexpected outcome: {misrepresented:?}"),
        );
        let accurate = reject_status_misrepresentation(
            CutoverCompatibilityStatus::Experimental,
            CutoverCompatibilityStatus::Experimental,
            "runtime-inference-api",
        );
        record(
            &mut results,
            "an experimental API accurately presented as experimental is accepted",
            accurate.is_ok(),
            format!("unexpected outcome: {accurate:?}"),
        );
    }

    {
        let missing_gate: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES[..3]
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        let outcome = validate_required_gates_executed(&missing_gate);
        record(
            &mut results,
            "E2E local conformance not run blocks required gate execution",
            matches!(
                outcome,
                Err(ReleaseCutoverError::RequiredGateFailedOrMissing { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let complete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        record(
            &mut results,
            "every required gate passing is accepted",
            validate_required_gates_executed(&complete).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                validate_required_gates_executed(&complete)
            ),
        );
    }

    {
        let disallowed = GateSkip {
            gate: "reference-cpu-conformance".into(),
            outside_v0_1_scope: false,
            reason: Some("time constraints".into()),
            hides_baseline_failure: false,
            included_in_release_report: true,
        };
        let outcome = validate_gate_skips(&[disallowed]);
        record(
            &mut results,
            "a required-baseline gate skip (Reference CPU) is disallowed",
            matches!(outcome, Err(ReleaseCutoverError::DisallowedGateSkip { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
        let allowed = GateSkip {
            gate: "cuda-conformance".into(),
            outside_v0_1_scope: true,
            reason: Some("CUDA is outside v0.1 scope".into()),
            hides_baseline_failure: false,
            included_in_release_report: true,
        };
        record(
            &mut results,
            "an out-of-scope, documented, reported gate skip is allowed",
            validate_gate_skips(&[allowed]).is_ok(),
            "unexpected outcome",
        );
    }

    {
        let outcome = reject_undocumented_cutover_exception(true, None);
        record(
            &mut results,
            "an undocumented required exception is blocked",
            matches!(
                outcome,
                Err(ReleaseCutoverError::UndocumentedException { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let exception = CutoverException {
            gate: "dependency-audit".into(),
            exception: SecurityException {
                issue: "advisory RUSTSEC-0000-0000".into(),
                affected_component: "example-crate".into(),
                severity: crate::DependencyAdvisorySeverity::High,
                rationale: "no fixed version available yet".into(),
                mitigation: "vendored patch applied".into(),
                owner: "release-team".into(),
                expiration_or_follow_up: "revisit in v0.2".into(),
                release_note_entry: true,
            },
        };
        let documented = reject_undocumented_cutover_exception(true, Some(&exception));
        record(
            &mut results,
            "a fully documented exception is accepted",
            documented.is_ok(),
            format!("unexpected outcome: {documented:?}"),
        );
    }

    {
        let missing_notes = CutoverSecurityVerification {
            gate_inputs: ReleaseSecurityGateInputs::default(),
            security_notes: SecurityReleaseNotes::default(),
        };
        record(
            &mut results,
            "missing security notes block cutover security verification",
            missing_notes.validate().is_err(),
            format!("unexpected outcome: {:?}", missing_notes.validate()),
        );
        let secret_detected = CutoverSecurityVerification {
            gate_inputs: ReleaseSecurityGateInputs {
                secrets_detected: true,
                ..Default::default()
            },
            security_notes: SecurityReleaseNotes {
                v0_1_threat_model: Some("CPU-local baseline".into()),
                trusted_native_provider_model: Some("Providers are trusted native code".into()),
                no_raw_handle_policy: Some("no raw handles in public APIs".into()),
                default_redaction: Some("diagnostics redact by default".into()),
                reporting_process_placeholder: Some("reporting process TBD".into()),
                ..Default::default()
            },
        };
        record(
            &mut results,
            "a detected secret blocks cutover security verification",
            secret_detected.validate().is_err(),
            format!("unexpected outcome: {:?}", secret_detected.validate()),
        );
    }

    {
        let manifest = ReleaseArtifactManifest::default();
        let outcome = validate_cutover_artifacts_generated(&manifest);
        record(
            &mut results,
            "a missing conformance report blocks stable release unless marked not applicable",
            matches!(
                outcome,
                Err(ReleaseCutoverError::ArtifactGenerationIncomplete { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let checksum =
            ArtifactChecksum::new("magnetar-cli", crate::ChecksumAlgorithm::Sha256, "deadbeef")
                .expect("non-empty digest");
        let mismatch = verify_cutover_artifact_checksum(&checksum, "different-digest");
        record(
            &mut results,
            "a published binary checksum mismatch blocks or withdraws release",
            matches!(
                mismatch,
                Err(ReleaseCutoverError::ArtifactVerificationFailed { .. })
            ),
            format!("unexpected outcome: {mismatch:?}"),
        );
        let matching = verify_cutover_artifact_checksum(&checksum, "deadbeef");
        record(
            &mut results,
            "a matching checksum is accepted",
            matching.is_ok(),
            format!("unexpected outcome: {matching:?}"),
        );
    }

    {
        let incomplete = CutoverChangelogChecklist {
            changelog: ReleaseChangelog {
                entries: vec![crate::ChangelogEntry {
                    kind: crate::ChangelogEntryKind::AddedContract,
                    description: "added the Runtime Inference API".into(),
                }],
            },
            includes_added_contracts: true,
            includes_changed_contracts: true,
            includes_removed_or_deprecated_contracts: true,
            includes_release_scope: true,
            includes_known_limitations: false,
            includes_compatibility_status: true,
            includes_security_notes: true,
            includes_conformance_status: true,
            includes_deferred_roadmap_items: true,
        };
        record(
            &mut results,
            "a changelog missing a known limitation is blocked",
            matches!(
                incomplete.validate(),
                Err(ReleaseCutoverError::ChangelogIncomplete { .. })
            ),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
    }

    {
        let experimental_as_stable = CutoverReleaseNotesChecklist {
            explains_what_v0_1_is: true,
            ..Default::default()
        };
        record(
            &mut results,
            "incomplete release notes are blocked",
            experimental_as_stable.validate().is_err(),
            format!(
                "unexpected outcome: {:?}",
                experimental_as_stable.validate()
            ),
        );
        let stability_outcome = reject_status_misrepresentation(
            CutoverCompatibilityStatus::Experimental,
            CutoverCompatibilityStatus::StableForV01Baseline,
            "cli-command-surface",
        );
        record(
            &mut results,
            "release notes presenting an experimental API as stable are blocked",
            stability_outcome.is_err(),
            format!("unexpected outcome: {stability_outcome:?}"),
        );
    }

    {
        let complete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        let mut incomplete = complete.clone();
        incomplete[0].passed = false;
        let early_tag = validate_tag_after_gates(&incomplete, true);
        record(
            &mut results,
            "a stable tag created before gates pass is invalid",
            matches!(
                early_tag,
                Err(ReleaseCutoverError::TagCreatedBeforeGatesPassed)
            ),
            format!("unexpected outcome: {early_tag:?}"),
        );
        let valid_tag = validate_tag_after_gates(&complete, true);
        record(
            &mut results,
            "a stable tag created after every required gate passes is valid",
            valid_tag.is_ok(),
            format!("unexpected outcome: {valid_tag:?}"),
        );
    }

    {
        let server_api_claimed = validate_publication_scope_preserved(
            "server-api-implementation",
            true,
            CutoverCompatibilityStatus::Deferred,
            CutoverCompatibilityStatus::StableForV01Baseline,
        );
        record(
            &mut results,
            "server API claimed included in v0.1 publication is blocked",
            matches!(
                server_api_claimed,
                Err(ReleaseCutoverError::PublicationScopeViolation { .. })
            ),
            format!("unexpected outcome: {server_api_claimed:?}"),
        );
        let accurate_publication = validate_publication_scope_preserved(
            "reference-cpu-provider",
            true,
            CutoverCompatibilityStatus::StableForV01Baseline,
            CutoverCompatibilityStatus::StableForV01Baseline,
        );
        record(
            &mut results,
            "an accurately scoped publication is accepted",
            accurate_publication.is_ok(),
            format!("unexpected outcome: {accurate_publication:?}"),
        );
    }

    {
        let mismatch = PostPublicationVerification {
            published_artifacts_match_checksums: true,
            release_notes_visible: true,
            reports_accessible: true,
            version_command_matches_tag: false,
            documentation_links_valid: true,
            compatibility_matrix_visible: true,
            security_notes_visible: true,
            deferred_roadmap_clearly_separated: true,
        };
        record(
            &mut results,
            "a binary version output differing from the release tag is invalid",
            matches!(
                mismatch.validate(),
                Err(ReleaseCutoverError::PostPublicationVerificationFailed { .. })
            ),
            format!("unexpected outcome: {:?}", mismatch.validate()),
        );
    }

    {
        let incomplete = RollbackRetractionNotes::default();
        record(
            &mut results,
            "incomplete rollback and retraction notes are rejected",
            incomplete.validate().is_err(),
            format!("unexpected outcome: {:?}", incomplete.validate()),
        );
        let complete = RollbackRetractionNotes {
            withdrawal_procedure_documented: true,
            advisory_publication_procedure_documented: true,
            patch_release_procedure_documented: true,
            audit_trail_preservation_documented: true,
            release_notes_update_procedure_documented: true,
        };
        record(
            &mut results,
            "complete rollback and retraction notes are accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let next_work = PostV01HandoffItem {
            name: "optimized-cpu-provider".into(),
            presented_as_v0_1_release_claim: false,
        };
        record(
            &mut results,
            "optimized CPU Provider listed as next work is clearly post-v0.1",
            reject_post_v0_1_item_as_release_claim(&next_work).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                reject_post_v0_1_item_as_release_claim(&next_work)
            ),
        );
        let misclaimed = PostV01HandoffItem {
            name: "server-api-implementation".into(),
            presented_as_v0_1_release_claim: true,
        };
        let outcome = reject_post_v0_1_item_as_release_claim(&misclaimed);
        record(
            &mut results,
            "a post-v0.1 item presented as a v0.1 release claim is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = validate_final_release_statement(V0_1_FINAL_RELEASE_STATEMENT);
        record(
            &mut results,
            "the canonical final release statement is accepted",
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
        let invalid = validate_final_release_statement("Magnetar v0.1 ships GPU support.");
        record(
            &mut results,
            "a release statement missing required phrases is rejected",
            invalid.is_err(),
            format!("unexpected outcome: {invalid:?}"),
        );
    }

    {
        let outcome = validate_cutover_cli_boundary("filesystem");
        record(
            &mut results,
            "Runtime receiving ambient CLI filesystem authority is blocked",
            matches!(
                outcome,
                Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
        let scope_outcome = validate_cutover_runtime_scope("shell");
        record(
            &mut results,
            "Runtime including tool execution is blocked",
            matches!(
                scope_outcome,
                Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
            ),
            format!("unexpected outcome: {scope_outcome:?}"),
        );
        let allowed = validate_cutover_runtime_scope("generation");
        record(
            &mut results,
            "an inference-scoped Runtime capability is accepted",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let observation = record_release_cutover_observation(ReleaseCutoverObservationInput {
            correlation_id: CorrelationId::new("cutover-run-1"),
            kind: ReleaseSecurityObservationKind::SecretScanCompleted,
            gate: Some("secret-scan".into()),
            target: Some("reference-cpu-provider".into()),
            feature_set: vec!["reference-cpu-provider".into()],
            artifact: Some("magnetar-cli".into()),
            release_metadata: Some("v0.1.0-rc.1".into()),
            raw_detail: "found credential abc123 in build.env",
        });
        record(
            &mut results,
            "a cutover observation is redacted by default",
            observation
                .security_observation
                .detail
                .as_deref()
                .is_some_and(|detail| !detail.contains("credential abc123")),
            format!("observation leaked sensitive content: {observation:?}"),
        );
        record(
            &mut results,
            "a cutover observation is correlatable to gate, target, feature set, and artifact",
            observation.gate.is_some()
                && observation.target.is_some()
                && !observation.feature_set.is_empty()
                && observation.artifact.is_some(),
            "observation is missing correlation fields",
        );
    }

    ReleaseCutoverConformanceReport { results }
}
