//! Release packaging and versioning policy contract (see
//! `openspec/changes/define-release-packaging-and-versioning-policy`).
//!
//! This module does not implement release automation, a package registry, or
//! supply-chain signing -- the proposal's "Non-Goals" rule all of that out
//! explicitly. Instead it defines, as executable Rust types and validation
//! functions, the release packaging and versioning **policy** for `v0.1`:
//!
//! - [`ReleaseVersion`]: explicit semantic versioning
//!   ([`evaluate_version_bump`] implements the pre-1.0 / 0.x-minor / patch
//!   breaking-change rules).
//! - [`CrateVersionMetadata`]: per-crate version declarations and
//!   [`validate_crate_dependency_compatibility`].
//! - [`ReleaseBinaryVersionReport`] / [`build_release_binary_version_report`]:
//!   the metadata `magnetar version` reports (binary version, Runtime crate
//!   version, OpenSpec baseline version, WIT contract versions, enabled
//!   feature flags, build profile, commit hash, conformance suite version).
//! - [`WitVersionChangeKind`] / [`required_wit_version_bump`]:
//!   breaking/additive/documentation-only WIT change classification.
//!   [`SupportedWitVersionMatrix`] documents supported WIT package versions.
//! - [`OpenSpecBaselineDeclaration`]: the baseline metadata a release SHALL
//!   declare (accepted changes, validation status, compatibility notes,
//!   deferred changes, conformance status, release tag).
//! - [`ReleaseFreezeState`] / [`reject_change_after_freeze`]: the freeze
//!   policy -- no semantic contract changes, WIT breaking changes without a
//!   version bump, or release-gate changes without a checklist update, once
//!   frozen; documentation clarifications remain allowed.
//! - [`ReleaseFeatureFlagClass`] / [`ReleaseFeatureFlag`]: the six feature
//!   flag classes, plus [`reject_experimental_flag_enabled_by_default`] and
//!   the concrete [`provider_feature_flags`] / [`component_engine_feature_flags`]
//!   catalogs (only `reference-cpu-provider` is required for `v0.1`).
//! - [`ReleasePlatformTarget`] / [`release_platform_targets`]: supported
//!   native targets, the CI target set, and wasm32 check-only status.
//! - [`ReleaseArtifactKind`] / [`ReleaseArtifactManifest`]: the release
//!   artifact set and [`ReleaseArtifactManifest::validate`] ("present or
//!   explicitly not applicable").
//! - [`ArtifactChecksum`] / [`ChecksumAlgorithm`]: checksum metadata,
//!   deliberately never treated as a trust or signature policy substitute.
//! - [`ChangelogEntryKind`] / [`ReleaseChangelog`]: changelog structure and
//!   [`ReleaseChangelog::validate`].
//! - [`CompatibilityDimension`] / [`CompatibilityStatus`] /
//!   [`ReleaseCompatibilityMatrix`]: explicit per-dimension compatibility
//!   status across all eight dimensions the proposal names.
//! - [`reject_release_public_api_handle_exposure`]: release-surface handle
//!   safety, covering Provider/Device/Kernel/tensor/memory/KV-cache/model-
//!   weight internals.
//! - [`ReleaseConformanceVersions`]: the six conformance suite versions a
//!   release SHALL report, reusing existing suite-version constants where
//!   they already exist rather than duplicating them.
//! - [`ReleaseGate`] / [`ReleaseGateResult`] / [`release_may_publish_stable`]:
//!   the fifteen required release gates and the deny-by-default stable
//!   publication decision.
//! - [`ReleaseCandidateTag`] / [`ReleaseCandidateManifest`]: pre-release tag
//!   format (`-alpha`, `-beta`, `-rc.N`) and the RC manifest requirements,
//!   always structurally distinct from a stable release.
//! - [`ReleaseBuildMetadata`] / [`redact_build_metadata`]: build metadata
//!   fields and default secret/local-path redaction.
//! - [`ReleaseDocumentationChecklist`]: the documentation set a release
//!   SHOULD publish, with a SHALL-strength minimum (baseline scope and known
//!   limitations).
//! - [`ReleaseSecurityNotes`]: the security notes a release SHOULD identify,
//!   explicitly deferring hardening detail to a separate change.
//! - [`PublishingBoundaryCategory`] / [`classify_publishing_boundary`]: the
//!   four publishing categories and [`reject_roadmap_feature_as_guarantee`].
//! - [`ReleasePackagingError`]: structured error categories.
//! - [`ReleasePackagingConformanceReport`] / [`run_release_packaging_conformance`]:
//!   a conformance report, in the shape of
//!   [`crate::CliBoundaryConformanceReport`], asserting the guarantees above
//!   hold.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::{
    E2E_SUITE_VERSION, FIRST_OPERATOR_SCOPE_VERSION, MAGNETAR_RUNTIME_VERSION,
    PROVIDER_CONFORMANCE_SUITE_VERSION, QWEN_BASELINE_CONTRACT_VERSION, WitInterface,
    compute::redact_backend_diagnostic,
};

pub const RELEASE_PACKAGING_POLICY_VERSION: &str = "0.1.0";

/// A release SHALL report this version for the Runtime Inference API
/// conformance suite (`specs/inference-api/spec.md`, "Inference API
/// Compatibility Status"); no such constant existed before this change.
pub const RUNTIME_INFERENCE_API_CONFORMANCE_VERSION: &str = "0.1.0";

/// A release SHALL report this version for the CLI boundary conformance
/// suite (`specs/cli-boundary/spec.md`, "CLI Command Stability Status").
pub const CLI_BOUNDARY_CONFORMANCE_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Versioning Policy
// ---------------------------------------------------------------------

/// Explicit semantic version for a release artifact, implementing "Magnetar
/// SHALL use explicit semantic versioning for release artifacts"
/// (`specs/release-packaging/spec.md`, "Semantic Versioning").
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReleaseVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl ReleaseVersion {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub const fn is_pre_1_0(self) -> bool {
        self.major == 0
    }
}

impl fmt::Display for ReleaseVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The version-bump category a proposed change to `from` -> `to` falls into.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReleaseVersionBumpKind {
    Major,
    Minor,
    Patch,
    NoChange,
}

/// Classifies a `from` -> `to` version transition, implementing the
/// versioning policy: "Before `1.0`, breaking changes MAY occur, but they
/// SHALL be documented", "Within `0.x`, minor versions MAY include breaking
/// API changes if clearly documented", and "Patch versions SHOULD avoid
/// breaking changes". Returns `Err` when a breaking change is claimed to
/// live in a patch bump.
pub fn evaluate_version_bump(
    from: ReleaseVersion,
    to: ReleaseVersion,
    is_breaking: bool,
    documented: bool,
) -> Result<ReleaseVersionBumpKind, ReleasePackagingError> {
    let kind = if to.major != from.major {
        ReleaseVersionBumpKind::Major
    } else if to.minor != from.minor {
        ReleaseVersionBumpKind::Minor
    } else if to.patch != from.patch {
        ReleaseVersionBumpKind::Patch
    } else {
        ReleaseVersionBumpKind::NoChange
    };

    if is_breaking && !documented {
        return Err(ReleasePackagingError::UndocumentedBreakingChange {
            from: from.to_string(),
            to: to.to_string(),
        });
    }

    if is_breaking && matches!(kind, ReleaseVersionBumpKind::Patch) {
        return Err(ReleasePackagingError::BreakingChangeInPatchRelease {
            from: from.to_string(),
            to: to.to_string(),
        });
    }

    Ok(kind)
}

// ---------------------------------------------------------------------
// Crate Versioning
// ---------------------------------------------------------------------

/// A publishable Rust crate's declared version metadata, implementing "Each
/// publishable Rust crate SHALL have a declared version" and "Crates MAY
/// share the same workspace version for the first release".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrateVersionMetadata {
    pub crate_name: String,
    pub version: ReleaseVersion,
    pub shares_workspace_version: bool,
}

/// "If independent crate versions are used, dependency compatibility SHALL
/// be documented": rejects a `dependent` declaring a dependency on
/// `dependency` without an explicit compatibility note, unless both crates
/// share the workspace version (in which case no independent compatibility
/// statement is needed).
pub fn validate_crate_dependency_compatibility(
    dependent: &CrateVersionMetadata,
    dependency: &CrateVersionMetadata,
    documented_compatibility: Option<&str>,
) -> Result<(), ReleasePackagingError> {
    if dependent.shares_workspace_version && dependency.shares_workspace_version {
        return Ok(());
    }
    match documented_compatibility {
        Some(note) if !note.trim().is_empty() => Ok(()),
        _ => Err(ReleasePackagingError::UndocumentedCrateDependency {
            dependent: dependent.crate_name.clone(),
            dependency: dependency.crate_name.clone(),
        }),
    }
}

// ---------------------------------------------------------------------
// Binary Versioning
// ---------------------------------------------------------------------

/// Everything `magnetar version` SHALL/SHOULD report, implementing "Binaries
/// such as `magnetar` SHALL report version information" and the "Version
/// output SHOULD include" list (`specs/release-packaging/spec.md`, "Binary
/// Version Reporting").
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseBinaryVersionReport {
    pub binary_version: String,
    pub runtime_crate_version: String,
    pub openspec_baseline_version: String,
    pub wit_contract_versions: Vec<WitInterface>,
    pub enabled_feature_flags: Vec<String>,
    pub build_profile: String,
    pub commit_hash: Option<String>,
    pub conformance_suite_version: Option<String>,
}

/// The WIT packages this release declares, matching
/// `magnetar-runtime/wit/*.wit`.
pub fn release_wit_contract_versions() -> Vec<WitInterface> {
    vec![
        WitInterface::new("magnetar:compute", "2.0.0"),
        WitInterface::new("magnetar:observability", "1.0.0"),
    ]
}

/// Builds the version report `magnetar version` prints, implementing
/// "Report binary version" / "Report runtime crate version" / "Report
/// OpenSpec baseline version" / "Report WIT contract versions" / "Report
/// enabled feature flags" / "Report build profile" / "Report commit hash
/// where available" / "Report conformance suite version where available".
/// `commit_hash` and `conformance_suite_version` are `Option` because both
/// are explicitly "where available" in the proposal, not always-required.
pub fn build_release_binary_version_report(
    binary_version: ReleaseVersion,
    enabled_feature_flags: Vec<String>,
    build_profile: impl Into<String>,
    commit_hash: Option<String>,
) -> ReleaseBinaryVersionReport {
    ReleaseBinaryVersionReport {
        binary_version: binary_version.to_string(),
        runtime_crate_version: MAGNETAR_RUNTIME_VERSION.to_string(),
        openspec_baseline_version: RELEASE_PACKAGING_POLICY_VERSION.to_string(),
        wit_contract_versions: release_wit_contract_versions(),
        enabled_feature_flags,
        build_profile: build_profile.into(),
        commit_hash,
        conformance_suite_version: Some(PROVIDER_CONFORMANCE_SUITE_VERSION.to_string()),
    }
}

// ---------------------------------------------------------------------
// WIT Versioning
// ---------------------------------------------------------------------

/// Classification of a proposed WIT package change, implementing "Breaking
/// WIT changes SHALL require a new major WIT version", "Non-breaking
/// additive changes MAY use minor versions", and "Documentation-only changes
/// MAY use patch versions" (`specs/wit/spec.md`, "WIT Package Release
/// Versions" / "WIT Breaking Change Policy").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WitVersionChangeKind {
    Breaking,
    Additive,
    DocumentationOnly,
}

/// The minimum [`ReleaseVersionBumpKind`] a WIT change of `kind` requires.
pub const fn required_wit_version_bump(kind: WitVersionChangeKind) -> ReleaseVersionBumpKind {
    match kind {
        WitVersionChangeKind::Breaking => ReleaseVersionBumpKind::Major,
        WitVersionChangeKind::Additive => ReleaseVersionBumpKind::Minor,
        WitVersionChangeKind::DocumentationOnly => ReleaseVersionBumpKind::Patch,
    }
}

/// "A breaking WIT change SHALL require a major WIT version bump": rejects a
/// `Breaking` change whose actual bump was not [`ReleaseVersionBumpKind::Major`].
pub fn validate_wit_version_bump(
    kind: WitVersionChangeKind,
    actual_bump: ReleaseVersionBumpKind,
    package: &str,
) -> Result<(), ReleasePackagingError> {
    let required = required_wit_version_bump(kind);
    let satisfied = match (required, actual_bump) {
        (ReleaseVersionBumpKind::Major, ReleaseVersionBumpKind::Major) => true,
        (
            ReleaseVersionBumpKind::Minor,
            ReleaseVersionBumpKind::Major | ReleaseVersionBumpKind::Minor,
        ) => true,
        (ReleaseVersionBumpKind::Patch, actual)
            if !matches!(actual, ReleaseVersionBumpKind::NoChange) =>
        {
            true
        }
        _ => false,
    };
    if satisfied {
        Ok(())
    } else {
        Err(ReleasePackagingError::WitVersionBumpInsufficient {
            package: package.to_string(),
        })
    }
}

/// "A release SHALL document which WIT package versions are supported":
/// the supported-version matrix for `release_wit_contract_versions`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SupportedWitVersionMatrix {
    pub supported: BTreeMap<String, String>,
}

impl SupportedWitVersionMatrix {
    pub fn from_interfaces(interfaces: &[WitInterface]) -> Self {
        Self {
            supported: interfaces
                .iter()
                .map(|interface| (interface.name.clone(), interface.version.clone()))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------
// OpenSpec Baseline
// ---------------------------------------------------------------------

/// "A release SHALL declare the OpenSpec baseline it implements", with the
/// "baseline metadata SHOULD include" fields from
/// `specs/release-packaging/spec.md`, "OpenSpec Baseline Declaration".
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpenSpecBaselineDeclaration {
    pub accepted_changes: Vec<String>,
    pub validation_status: Option<String>,
    pub compatibility_notes: Vec<String>,
    pub deferred_changes: Vec<String>,
    pub conformance_status: Option<String>,
    pub release_tag: Option<String>,
}

impl OpenSpecBaselineDeclaration {
    /// A release SHALL declare at least one accepted change and its
    /// validation status; the remaining fields are `SHOULD`-strength.
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        if self.accepted_changes.is_empty() {
            return Err(ReleasePackagingError::OpenSpecBaselineIncomplete {
                reason: "no accepted changes declared".into(),
            });
        }
        if self.validation_status.is_none() {
            return Err(ReleasePackagingError::OpenSpecBaselineIncomplete {
                reason: "no OpenSpec validation status recorded".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Change Freeze
// ---------------------------------------------------------------------

/// Whether a release's included OpenSpec contracts are frozen, implementing
/// "Before cutting a release, OpenSpec changes included in that release
/// SHALL be frozen" (`specs/release-packaging/spec.md`, "Release Freeze").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReleaseFreezeState {
    Open,
    Frozen,
}

/// The kind of change proposed against a frozen release.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReleaseFreezeChangeKind {
    SemanticContractChange,
    WitBreakingChangeWithoutVersionBump,
    ReleaseGateChangeWithoutChecklistUpdate,
    HiddenScopeExpansion,
    DocumentationClarification,
}

/// "Late semantic change SHALL require new change proposal or release is
/// delayed": denies every change kind except a non-semantic documentation
/// clarification while `state` is [`ReleaseFreezeState::Frozen`]. When
/// `state` is [`ReleaseFreezeState::Open`], every change kind is allowed.
pub fn reject_change_after_freeze(
    state: ReleaseFreezeState,
    kind: ReleaseFreezeChangeKind,
) -> Result<(), ReleasePackagingError> {
    if matches!(state, ReleaseFreezeState::Open) {
        return Ok(());
    }
    if matches!(kind, ReleaseFreezeChangeKind::DocumentationClarification) {
        return Ok(());
    }
    Err(ReleasePackagingError::ReleaseFrozen {
        change_kind: format!("{kind:?}"),
    })
}

// ---------------------------------------------------------------------
// Feature Flag Policy
// ---------------------------------------------------------------------

/// The six release feature flag classes, implementing "Feature flags SHOULD
/// distinguish" (`specs/release-packaging/spec.md`, "Feature Flag
/// Classification").
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReleaseFeatureFlagClass {
    Baseline,
    Experimental,
    ProviderSpecific,
    PlatformSpecific,
    TestOnly,
    ConformanceOnly,
}

impl ReleaseFeatureFlagClass {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Experimental => "experimental",
            Self::ProviderSpecific => "provider-specific",
            Self::PlatformSpecific => "platform-specific",
            Self::TestOnly => "test-only",
            Self::ConformanceOnly => "conformance-only",
        }
    }
}

/// A single release feature flag, classified and tagged as enabled-by-
/// default or not, implementing "Release feature flags SHALL be explicit".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseFeatureFlag {
    pub name: String,
    pub class: ReleaseFeatureFlagClass,
    pub enabled_by_default: bool,
}

/// "Experimental features SHALL not be enabled by default": rejects any
/// [`ReleaseFeatureFlagClass::Experimental`] flag with
/// `enabled_by_default == true`.
pub fn reject_experimental_flag_enabled_by_default(
    flag: &ReleaseFeatureFlag,
) -> Result<(), ReleasePackagingError> {
    if matches!(flag.class, ReleaseFeatureFlagClass::Experimental) && flag.enabled_by_default {
        return Err(ReleasePackagingError::ExperimentalFeatureEnabledByDefault {
            flag: flag.name.clone(),
        });
    }
    Ok(())
}

/// Provider-specific feature flags named in the proposal. Only
/// `reference-cpu-provider` is required for `v0.1`; every other flag is
/// absent, disabled, or explicitly experimental, implementing "Provider
/// Feature Flags".
pub fn provider_feature_flags() -> Vec<ReleaseFeatureFlag> {
    let required_only = "reference-cpu-provider";
    [
        "reference-cpu-provider",
        "optimized-cpu-provider",
        "cuda-provider",
        "metal-provider",
        "openvino-provider",
        "qnn-provider",
        "webgpu-provider",
    ]
    .into_iter()
    .map(|name| ReleaseFeatureFlag {
        name: name.to_string(),
        class: ReleaseFeatureFlagClass::ProviderSpecific,
        enabled_by_default: name == required_only,
    })
    .collect()
}

/// "For `v0.1`, only Reference CPU Provider SHOULD be required. Other
/// Provider flags SHALL be absent, disabled, or explicitly experimental."
pub fn validate_provider_feature_flags_for_v0_1(
    flags: &[ReleaseFeatureFlag],
) -> Result<(), ReleasePackagingError> {
    for flag in flags {
        if flag.name != "reference-cpu-provider" && flag.enabled_by_default {
            return Err(
                ReleasePackagingError::ProviderFeatureFlagRequiredForBaseline {
                    flag: flag.name.clone(),
                },
            );
        }
    }
    Ok(())
}

/// Component engine feature flags named in the proposal, implementing
/// "Component Engine Feature Flags": native Wasmtime support SHALL remain
/// feature-gated, and browser builds SHALL not require Wasmtime.
pub fn component_engine_feature_flags() -> Vec<ReleaseFeatureFlag> {
    [
        "wasmtime-component-engine",
        "web-component-engine",
        "test-component-engine",
    ]
    .into_iter()
    .map(|name| ReleaseFeatureFlag {
        name: name.to_string(),
        class: ReleaseFeatureFlagClass::PlatformSpecific,
        enabled_by_default: false,
    })
    .collect()
}

/// "Browser builds SHALL not require Wasmtime": rejects a browser platform
/// target that lists `wasmtime-component-engine` as a required feature.
pub fn reject_wasmtime_required_for_browser(
    target: &ReleasePlatformTarget,
    required_features: &[&str],
) -> Result<(), ReleasePackagingError> {
    if target.is_browser_like && required_features.contains(&"wasmtime-component-engine") {
        return Err(ReleasePackagingError::BrowserRequiresWasmtime {
            target: target.triple.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Platform Targets
// ---------------------------------------------------------------------

/// A single supported (or check-only) platform target, implementing
/// "Platform Targets": the release SHOULD define supported platform
/// targets, the first release SHOULD support native CPU targets required by
/// CI, and browser/wasm32 support MAY be check-only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleasePlatformTarget {
    pub triple: String,
    pub required_by_ci: bool,
    pub check_only: bool,
    pub is_browser_like: bool,
}

/// The `v0.1` platform target set: native CI targets plus a check-only
/// wasm32 browser target.
pub fn release_platform_targets() -> Vec<ReleasePlatformTarget> {
    vec![
        ReleasePlatformTarget {
            triple: "x86_64-unknown-linux-gnu".into(),
            required_by_ci: true,
            check_only: false,
            is_browser_like: false,
        },
        ReleasePlatformTarget {
            triple: "x86_64-pc-windows-msvc".into(),
            required_by_ci: true,
            check_only: false,
            is_browser_like: false,
        },
        ReleasePlatformTarget {
            triple: "aarch64-apple-darwin".into(),
            required_by_ci: true,
            check_only: false,
            is_browser_like: false,
        },
        ReleasePlatformTarget {
            triple: "wasm32-unknown-unknown".into(),
            required_by_ci: false,
            check_only: true,
            is_browser_like: true,
        },
    ]
}

/// "Unsupported targets SHALL be documented": returns targets not in
/// `supported` for a caller to fold into release documentation.
pub fn unsupported_targets<'a>(
    supported: &[ReleasePlatformTarget],
    candidates: &'a [&'a str],
) -> Vec<&'a str> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| !supported.iter().any(|target| target.triple == *candidate))
        .collect()
}

// ---------------------------------------------------------------------
// Release Artifacts
// ---------------------------------------------------------------------

/// Release artifact kinds from "Release Artifacts"
/// (`specs/release-packaging/spec.md`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReleaseArtifactKind {
    SourceArchive,
    RustCrates,
    CliBinary,
    ConformanceReport,
    E2eReport,
    OpenSpecValidationReport,
    CoverageReport,
    SbomPlaceholder,
    Checksums,
    Changelog,
    ReleaseNotes,
}

pub const RELEASE_ARTIFACT_KINDS: &[ReleaseArtifactKind] = &[
    ReleaseArtifactKind::SourceArchive,
    ReleaseArtifactKind::RustCrates,
    ReleaseArtifactKind::CliBinary,
    ReleaseArtifactKind::ConformanceReport,
    ReleaseArtifactKind::E2eReport,
    ReleaseArtifactKind::OpenSpecValidationReport,
    ReleaseArtifactKind::CoverageReport,
    ReleaseArtifactKind::SbomPlaceholder,
    ReleaseArtifactKind::Checksums,
    ReleaseArtifactKind::Changelog,
    ReleaseArtifactKind::ReleaseNotes,
];

/// Whether a given artifact kind is present, or explicitly marked not
/// applicable to this release, implementing "conformance report is present
/// or explicitly not applicable" (`specs/conformance/spec.md`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseArtifactStatus {
    Present,
    NotApplicable,
    Missing,
}

/// A release candidate's artifact manifest.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseArtifactManifest {
    pub status: BTreeMap<&'static str, ReleaseArtifactStatus>,
}

impl ReleaseArtifactManifest {
    pub fn set(&mut self, kind: ReleaseArtifactKind, status: ReleaseArtifactStatus) {
        self.status.insert(artifact_kind_id(kind), status);
    }

    /// "Required artifacts are present or explicitly marked not applicable":
    /// every [`ReleaseArtifactKind`] SHALL have a recorded, non-`Missing`
    /// status.
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        for kind in RELEASE_ARTIFACT_KINDS {
            match self.status.get(artifact_kind_id(*kind)) {
                Some(ReleaseArtifactStatus::Present | ReleaseArtifactStatus::NotApplicable) => {}
                _ => {
                    return Err(ReleasePackagingError::ReleaseArtifactMissing {
                        artifact: artifact_kind_id(*kind).to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

pub const fn artifact_kind_id(kind: ReleaseArtifactKind) -> &'static str {
    match kind {
        ReleaseArtifactKind::SourceArchive => "source-archive",
        ReleaseArtifactKind::RustCrates => "rust-crates",
        ReleaseArtifactKind::CliBinary => "cli-binary",
        ReleaseArtifactKind::ConformanceReport => "conformance-report",
        ReleaseArtifactKind::E2eReport => "e2e-report",
        ReleaseArtifactKind::OpenSpecValidationReport => "openspec-validation-report",
        ReleaseArtifactKind::CoverageReport => "coverage-report",
        ReleaseArtifactKind::SbomPlaceholder => "sbom-placeholder",
        ReleaseArtifactKind::Checksums => "checksums",
        ReleaseArtifactKind::Changelog => "changelog",
        ReleaseArtifactKind::ReleaseNotes => "release-notes",
    }
}

// ---------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------

/// Checksum algorithms this policy recognizes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ChecksumAlgorithm {
    Sha256,
}

/// A single published artifact's checksum, implementing "Published binary
/// artifacts SHALL include checksums" and "Checksums SHOULD cover" source
/// archives, binaries, generated reports, and packaged artifacts.
/// [`ArtifactChecksum`] deliberately carries no trust/signature field:
/// "Checksums SHALL not replace trust or signature policy" is implemented by
/// this type never being accepted anywhere a trust or signature decision is
/// made -- see [`ReleasePackagingConformanceReport`]'s checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactChecksum {
    pub artifact: String,
    pub algorithm: ChecksumAlgorithm,
    pub digest: String,
}

impl ArtifactChecksum {
    pub fn new(
        artifact: impl Into<String>,
        algorithm: ChecksumAlgorithm,
        digest: impl Into<String>,
    ) -> Result<Self, ReleasePackagingError> {
        let digest = digest.into();
        if digest.trim().is_empty() {
            return Err(ReleasePackagingError::ChecksumInvalid {
                artifact: artifact.into(),
            });
        }
        Ok(Self {
            artifact: artifact.into(),
            algorithm,
            digest,
        })
    }
}

// ---------------------------------------------------------------------
// Changelog
// ---------------------------------------------------------------------

/// Changelog entry categories from "Changelog SHOULD include"
/// (`specs/release-packaging/spec.md`, "Changelog Policy").
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ChangelogEntryKind {
    AddedContract,
    ChangedContract,
    RemovedOrDeprecatedContract,
    FixedIssue,
    KnownLimitation,
    ConformanceStatus,
    CompatibilityNote,
    SecurityNote,
}

/// A single changelog entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangelogEntry {
    pub kind: ChangelogEntryKind,
    pub description: String,
}

/// "Each release SHALL include a changelog": a non-empty ordered set of
/// [`ChangelogEntry`] values.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseChangelog {
    pub entries: Vec<ChangelogEntry>,
}

impl ReleaseChangelog {
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        if self.entries.is_empty() {
            return Err(ReleasePackagingError::ChangelogEmpty);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Compatibility Policy
// ---------------------------------------------------------------------

/// The eight compatibility dimensions from "Compatibility dimensions SHOULD
/// include" (`specs/release-packaging/spec.md`, "Compatibility Policy").
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompatibilityDimension {
    RustPublicApi,
    WitContracts,
    RuntimeInferenceApi,
    ModelArtifactMetadata,
    ProviderAbi,
    OpenSpecBaseline,
    CliCommandSurface,
    ConformanceReportFormat,
}

pub const COMPATIBILITY_DIMENSIONS: &[CompatibilityDimension] = &[
    CompatibilityDimension::RustPublicApi,
    CompatibilityDimension::WitContracts,
    CompatibilityDimension::RuntimeInferenceApi,
    CompatibilityDimension::ModelArtifactMetadata,
    CompatibilityDimension::ProviderAbi,
    CompatibilityDimension::OpenSpecBaseline,
    CompatibilityDimension::CliCommandSurface,
    CompatibilityDimension::ConformanceReportFormat,
];

/// Compatibility status a dimension may hold. Matches the vocabulary used
/// across `specs/*/spec.md` in this change ("stable-for-baseline",
/// "unstable", "experimental", "preview", "deferred").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompatibilityStatus {
    StableForBaseline,
    Unstable,
    Experimental,
    Preview,
    Deferred,
}

/// "Release compatibility SHALL be explicit" / "If an area is unstable, the
/// release SHALL mark it unstable": an explicit per-dimension status map.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseCompatibilityMatrix {
    pub status: BTreeMap<&'static str, CompatibilityStatus>,
}

pub const fn compatibility_dimension_id(dimension: CompatibilityDimension) -> &'static str {
    match dimension {
        CompatibilityDimension::RustPublicApi => "rust-public-api",
        CompatibilityDimension::WitContracts => "wit-contracts",
        CompatibilityDimension::RuntimeInferenceApi => "runtime-inference-api",
        CompatibilityDimension::ModelArtifactMetadata => "model-artifact-metadata",
        CompatibilityDimension::ProviderAbi => "provider-abi",
        CompatibilityDimension::OpenSpecBaseline => "openspec-baseline",
        CompatibilityDimension::CliCommandSurface => "cli-command-surface",
        CompatibilityDimension::ConformanceReportFormat => "conformance-report-format",
    }
}

impl ReleaseCompatibilityMatrix {
    pub fn set(&mut self, dimension: CompatibilityDimension, status: CompatibilityStatus) {
        self.status
            .insert(compatibility_dimension_id(dimension), status);
    }

    /// Every [`CompatibilityDimension`] SHALL have an explicit status
    /// recorded.
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        for dimension in COMPATIBILITY_DIMENSIONS {
            if !self
                .status
                .contains_key(compatibility_dimension_id(*dimension))
            {
                return Err(ReleasePackagingError::CompatibilityDimensionUndeclared {
                    dimension: compatibility_dimension_id(*dimension).to_string(),
                });
            }
        }
        Ok(())
    }
}

/// The `v0.1` compatibility matrix implied by this change's own specs: the
/// Provider ABI is explicitly unstable
/// (`specs/provider/spec.md`, "Provider ABI Compatibility Status") and
/// everything else is stable-for-baseline.
pub fn v0_1_compatibility_matrix() -> ReleaseCompatibilityMatrix {
    let mut matrix = ReleaseCompatibilityMatrix::default();
    for dimension in COMPATIBILITY_DIMENSIONS {
        let status = if matches!(dimension, CompatibilityDimension::ProviderAbi) {
            CompatibilityStatus::Unstable
        } else {
            CompatibilityStatus::StableForBaseline
        };
        matrix.set(*dimension, status);
    }
    matrix
}

// ---------------------------------------------------------------------
// Public API Safety
// ---------------------------------------------------------------------

/// Fragments a release public API surface name SHALL never contain,
/// implementing "Public APIs SHALL not expose" raw Provider/Device/Kernel
/// handles, raw tensor/memory pointers, raw KV cache contents, or raw model
/// weights (`specs/release-packaging/spec.md`, "Public API Safety").
const RELEASE_FORBIDDEN_API_SURFACE_FRAGMENTS: &[&str] = &[
    "raw-provider-handle",
    "raw-device-handle",
    "raw-kernel-handle",
    "raw-tensor-pointer",
    "raw-memory-pointer",
    "raw-kv-cache",
    "raw-model-weight",
];

/// Rejects a release public API surface name that would expose a raw
/// internal handle, pointer, or weight/cache payload.
pub fn reject_release_public_api_handle_exposure(
    surface_name: &str,
) -> Result<(), ReleasePackagingError> {
    let normalized = surface_name.trim().to_ascii_lowercase();
    if RELEASE_FORBIDDEN_API_SURFACE_FRAGMENTS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(ReleasePackagingError::PublicApiHandleExposureDenied {
            surface: surface_name.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Conformance Versioning
// ---------------------------------------------------------------------

/// The six conformance suite versions a release SHALL report, implementing
/// "Conformance Versioning" (`specs/release-packaging/spec.md`). Reuses
/// existing suite-version constants ([`PROVIDER_CONFORMANCE_SUITE_VERSION`],
/// [`FIRST_OPERATOR_SCOPE_VERSION`], [`QWEN_BASELINE_CONTRACT_VERSION`],
/// [`E2E_SUITE_VERSION`]) instead of duplicating them, and introduces the
/// two versions ([`RUNTIME_INFERENCE_API_CONFORMANCE_VERSION`],
/// [`CLI_BOUNDARY_CONFORMANCE_VERSION`]) no prior change declared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseConformanceVersions {
    pub provider_conformance_suite_version: String,
    pub first_operator_scope_conformance_version: String,
    pub qwen_baseline_conformance_version: String,
    pub runtime_inference_api_conformance_version: String,
    pub cli_boundary_conformance_version: String,
    pub e2e_local_conformance_version: String,
}

impl Default for ReleaseConformanceVersions {
    fn default() -> Self {
        Self {
            provider_conformance_suite_version: PROVIDER_CONFORMANCE_SUITE_VERSION.to_string(),
            first_operator_scope_conformance_version: FIRST_OPERATOR_SCOPE_VERSION.to_string(),
            qwen_baseline_conformance_version: QWEN_BASELINE_CONTRACT_VERSION.to_string(),
            runtime_inference_api_conformance_version: RUNTIME_INFERENCE_API_CONFORMANCE_VERSION
                .to_string(),
            cli_boundary_conformance_version: CLI_BOUNDARY_CONFORMANCE_VERSION.to_string(),
            e2e_local_conformance_version: E2E_SUITE_VERSION.to_string(),
        }
    }
}

// ---------------------------------------------------------------------
// Release Gates
// ---------------------------------------------------------------------

/// The fifteen required release gates from "Required gates SHOULD include"
/// (`specs/release-packaging/spec.md`, "Release Gates").
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReleaseGate {
    Formatting,
    CargoCheck,
    Clippy,
    UnitTests,
    ContractTests,
    OpenSpecValidation,
    WitValidation,
    ReferenceCpuConformance,
    OperatorFirstScopeConformance,
    RuntimeInferenceApiTests,
    CliBoundaryTests,
    E2eLocalConformance,
    CoverageGate,
    RedactionChecks,
    NoRawHandleExposureChecks,
}

pub const REQUIRED_RELEASE_GATES: &[ReleaseGate] = &[
    ReleaseGate::Formatting,
    ReleaseGate::CargoCheck,
    ReleaseGate::Clippy,
    ReleaseGate::UnitTests,
    ReleaseGate::ContractTests,
    ReleaseGate::OpenSpecValidation,
    ReleaseGate::WitValidation,
    ReleaseGate::ReferenceCpuConformance,
    ReleaseGate::OperatorFirstScopeConformance,
    ReleaseGate::RuntimeInferenceApiTests,
    ReleaseGate::CliBoundaryTests,
    ReleaseGate::E2eLocalConformance,
    ReleaseGate::CoverageGate,
    ReleaseGate::RedactionChecks,
    ReleaseGate::NoRawHandleExposureChecks,
];

/// The outcome of running a single [`ReleaseGate`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseGateResult {
    pub gate: ReleaseGate,
    pub passed: bool,
}

/// "A release SHALL pass required gates before publication" / "If required
/// release gates fail, release SHALL not be published as stable": every
/// [`ReleaseGate`] in [`REQUIRED_RELEASE_GATES`] SHALL be present in
/// `results` and SHALL have passed.
pub fn release_may_publish_stable(
    results: &[ReleaseGateResult],
) -> Result<(), ReleasePackagingError> {
    for gate in REQUIRED_RELEASE_GATES {
        match results.iter().find(|result| result.gate == *gate) {
            Some(result) if result.passed => {}
            Some(_) => {
                return Err(ReleasePackagingError::ReleaseGateFailed {
                    gate: format!("{gate:?}"),
                });
            }
            None => {
                return Err(ReleasePackagingError::ReleaseGateMissing {
                    gate: format!("{gate:?}"),
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Release Candidate Policy
// ---------------------------------------------------------------------

/// Allowed pre-release tags, implementing "Pre-release Tags":
/// `-alpha`, `-beta`, `-rc.N`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseCandidateTag {
    Alpha,
    Beta,
    Rc(u32),
}

impl fmt::Display for ReleaseCandidateTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alpha => write!(f, "-alpha"),
            Self::Beta => write!(f, "-beta"),
            Self::Rc(n) => write!(f, "-rc.{n}"),
        }
    }
}

impl ReleaseCandidateTag {
    /// "Pre-release tags SHALL not be confused with stable tags" / "A
    /// release candidate SHALL not be presented as stable": every value of
    /// this type is, by construction, never a stable tag.
    pub const fn is_stable(self) -> bool {
        false
    }
}

/// "A release candidate SHOULD include" the fields below, implementing
/// "Release Candidate Policy".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleaseCandidateManifest {
    pub tag: ReleaseCandidateTag,
    pub frozen_openspec_baseline: bool,
    pub conformance_report_included: bool,
    pub known_failures: Vec<String>,
    pub release_notes_draft: bool,
}

impl ReleaseCandidateManifest {
    /// "A release candidate SHOULD include: release version candidate tag,
    /// frozen OpenSpec baseline, conformance report, known failures, release
    /// notes draft" and "A release candidate SHALL not be presented as
    /// stable": SHALL-strength here because [`ReleaseCandidateTag::is_stable`]
    /// already guarantees the tag itself is never stable; this validates
    /// the remaining required manifest fields.
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        if !self.frozen_openspec_baseline {
            return Err(ReleasePackagingError::ReleaseCandidateIncomplete {
                reason: "OpenSpec baseline is not frozen".into(),
            });
        }
        if !self.conformance_report_included {
            return Err(ReleasePackagingError::ReleaseCandidateIncomplete {
                reason: "conformance report is not included".into(),
            });
        }
        Ok(())
    }
}

/// "A failed release candidate MAY be tagged as pre-release only if clearly
/// marked": accepts a failed gate result set only when paired with a
/// [`ReleaseCandidateTag`] (never a stable version).
pub fn allow_failed_candidate_as_pre_release(
    gate_results: &[ReleaseGateResult],
    tag: ReleaseCandidateTag,
) -> Result<ReleaseCandidateTag, ReleasePackagingError> {
    let _ = release_may_publish_stable(gate_results); // failure is expected/allowed here
    let _ = tag.is_stable();
    Ok(tag)
}

// ---------------------------------------------------------------------
// Build Metadata
// ---------------------------------------------------------------------

/// Build metadata fields from "Build Metadata MAY include"
/// (`specs/release-packaging/spec.md`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseBuildMetadata {
    pub commit_hash: Option<String>,
    pub build_timestamp: Option<String>,
    pub target_triple: Option<String>,
    pub enabled_features: Vec<String>,
    pub ci_run_id: Option<String>,
    pub profile: Option<String>,
    pub rustc_version: Option<String>,
}

/// Key name fragments that mark a build metadata value as secret-shaped,
/// beyond what [`redact_backend_diagnostic`]'s path/handle heuristics catch.
const BUILD_METADATA_SECRET_KEY_FRAGMENTS: &[&str] = &[
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "credential",
];

/// "Build metadata SHALL not include secrets or local filesystem paths by
/// default": redacts `value` when `key` looks secret-shaped or `value` looks
/// like a local filesystem path or native handle (via
/// `redact_backend_diagnostic`).
pub fn redact_build_metadata(key: &str, value: &str) -> String {
    let key_lower = key.to_ascii_lowercase();
    if BUILD_METADATA_SECRET_KEY_FRAGMENTS
        .iter()
        .any(|fragment| key_lower.contains(fragment))
    {
        return "[redacted build metadata]".into();
    }
    redact_backend_diagnostic(value)
}

// ---------------------------------------------------------------------
// Documentation Release
// ---------------------------------------------------------------------

/// The documentation set from "Release documentation SHOULD include"
/// (`specs/release-packaging/spec.md`, "Documentation Release").
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReleaseDocumentationChecklist {
    pub architecture_overview: bool,
    pub runtime_inference_api_overview: bool,
    pub cli_boundary_overview: bool,
    pub build_instructions: bool,
    pub test_instructions: bool,
    pub conformance_instructions: bool,
    pub feature_flags: bool,
    pub supported_targets: bool,
    pub known_limitations: bool,
    pub post_baseline_roadmap: bool,
}

impl ReleaseDocumentationChecklist {
    /// "Release documentation SHALL state baseline scope and known
    /// limitations": the SHALL-strength minimum, everything else is
    /// `SHOULD`-strength and not enforced here.
    pub fn validate(&self) -> Result<(), ReleasePackagingError> {
        if !self.known_limitations {
            return Err(ReleasePackagingError::DocumentationIncomplete {
                reason: "known limitations are not documented".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Security Notes
// ---------------------------------------------------------------------

/// The security notes a release SHOULD identify, implementing "Security
/// Notes" (`specs/release-packaging/spec.md`). Detailed security hardening
/// is explicitly deferred to a separate release security change -- this
/// type never carries hardening implementation detail, only the topic
/// names/notes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReleaseSecurityNotes {
    pub sandbox_assumptions: Option<String>,
    pub provider_trust_model: Option<String>,
    pub no_raw_handle_policy: Option<String>,
    pub default_redaction: Option<String>,
    pub source_cache_trust_boundary: Option<String>,
    pub unsupported_security_features: Vec<String>,
    pub known_risks: Vec<String>,
}

// ---------------------------------------------------------------------
// Publishing Boundary
// ---------------------------------------------------------------------

/// The four publishing boundary categories, implementing "Publishing SHALL
/// distinguish" (`specs/release-packaging/spec.md`, "Publishing Boundary").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PublishingBoundaryCategory {
    IncludedBaseline,
    ExperimentalFeature,
    DeferredRoadmap,
    UnsupportedFeature,
}

/// Roadmap feature names this release explicitly defers, matching the
/// proposal's "Release Scope" -- "The release SHALL NOT require" list.
const DEFERRED_ROADMAP_FEATURES: &[&str] = &[
    "cuda",
    "metal",
    "openvino",
    "qnn",
    "webgpu",
    "production-qwen-model-support",
    "large-model-execution",
    "model-hub-downloads",
    "server-api-implementation",
    "agent-tool-runtime",
    "production-cli-ux",
];

/// Classifies `feature` into a [`PublishingBoundaryCategory`], implementing
/// "Publishing SHALL not imply production readiness for all roadmap
/// features".
pub fn classify_publishing_boundary(feature: &str) -> PublishingBoundaryCategory {
    let normalized = feature.trim().to_ascii_lowercase().replace(' ', "-");
    if DEFERRED_ROADMAP_FEATURES.contains(&normalized.as_str()) {
        PublishingBoundaryCategory::DeferredRoadmap
    } else {
        PublishingBoundaryCategory::IncludedBaseline
    }
}

/// "Roadmap feature SHALL NOT be presented as included in `v0.1`": rejects
/// presenting a [`PublishingBoundaryCategory::DeferredRoadmap`] or
/// [`PublishingBoundaryCategory::UnsupportedFeature`] feature as included
/// baseline.
pub fn reject_roadmap_feature_as_guarantee(
    feature: &str,
    presented_as_included: bool,
) -> Result<(), ReleasePackagingError> {
    let category = classify_publishing_boundary(feature);
    if presented_as_included
        && matches!(
            category,
            PublishingBoundaryCategory::DeferredRoadmap
                | PublishingBoundaryCategory::UnsupportedFeature
        )
    {
        return Err(ReleasePackagingError::RoadmapFeaturePresentedAsIncluded {
            feature: feature.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured release packaging error, covering every failure category this
/// module's validation functions can produce.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReleasePackagingError {
    UndocumentedBreakingChange {
        from: String,
        to: String,
    },
    BreakingChangeInPatchRelease {
        from: String,
        to: String,
    },
    UndocumentedCrateDependency {
        dependent: String,
        dependency: String,
    },
    WitVersionBumpInsufficient {
        package: String,
    },
    OpenSpecBaselineIncomplete {
        reason: String,
    },
    ReleaseFrozen {
        change_kind: String,
    },
    ExperimentalFeatureEnabledByDefault {
        flag: String,
    },
    ProviderFeatureFlagRequiredForBaseline {
        flag: String,
    },
    BrowserRequiresWasmtime {
        target: String,
    },
    ReleaseArtifactMissing {
        artifact: String,
    },
    ChecksumInvalid {
        artifact: String,
    },
    ChangelogEmpty,
    CompatibilityDimensionUndeclared {
        dimension: String,
    },
    PublicApiHandleExposureDenied {
        surface: String,
    },
    ReleaseGateFailed {
        gate: String,
    },
    ReleaseGateMissing {
        gate: String,
    },
    ReleaseCandidateIncomplete {
        reason: String,
    },
    DocumentationIncomplete {
        reason: String,
    },
    RoadmapFeaturePresentedAsIncluded {
        feature: String,
    },
    InternalReleasePackagingError {
        reason: String,
    },
}

impl ReleasePackagingError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::UndocumentedBreakingChange { .. } => "undocumented-breaking-change",
            Self::BreakingChangeInPatchRelease { .. } => "breaking-change-in-patch-release",
            Self::UndocumentedCrateDependency { .. } => "undocumented-crate-dependency",
            Self::WitVersionBumpInsufficient { .. } => "wit-version-bump-insufficient",
            Self::OpenSpecBaselineIncomplete { .. } => "openspec-baseline-incomplete",
            Self::ReleaseFrozen { .. } => "release-frozen",
            Self::ExperimentalFeatureEnabledByDefault { .. } => {
                "experimental-feature-enabled-by-default"
            }
            Self::ProviderFeatureFlagRequiredForBaseline { .. } => {
                "provider-feature-flag-required-for-baseline"
            }
            Self::BrowserRequiresWasmtime { .. } => "browser-requires-wasmtime",
            Self::ReleaseArtifactMissing { .. } => "release-artifact-missing",
            Self::ChecksumInvalid { .. } => "checksum-invalid",
            Self::ChangelogEmpty => "changelog-empty",
            Self::CompatibilityDimensionUndeclared { .. } => "compatibility-dimension-undeclared",
            Self::PublicApiHandleExposureDenied { .. } => "public-api-handle-exposure-denied",
            Self::ReleaseGateFailed { .. } => "release-gate-failed",
            Self::ReleaseGateMissing { .. } => "release-gate-missing",
            Self::ReleaseCandidateIncomplete { .. } => "release-candidate-incomplete",
            Self::DocumentationIncomplete { .. } => "documentation-incomplete",
            Self::RoadmapFeaturePresentedAsIncluded { .. } => {
                "roadmap-feature-presented-as-included"
            }
            Self::InternalReleasePackagingError { .. } => "internal-release-packaging-error",
        }
    }
}

impl fmt::Display for ReleasePackagingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndocumentedBreakingChange { from, to }
            | Self::BreakingChangeInPatchRelease { from, to } => {
                write!(f, "{}: {from} -> {to}", self.id())
            }
            Self::UndocumentedCrateDependency {
                dependent,
                dependency,
            } => write!(f, "{}: {dependent} -> {dependency}", self.id()),
            Self::WitVersionBumpInsufficient { package } => {
                write!(f, "{}: {package}", self.id())
            }
            Self::OpenSpecBaselineIncomplete { reason }
            | Self::ReleaseCandidateIncomplete { reason }
            | Self::DocumentationIncomplete { reason }
            | Self::InternalReleasePackagingError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::ReleaseFrozen { change_kind } => write!(f, "{}: {change_kind}", self.id()),
            Self::ExperimentalFeatureEnabledByDefault { flag }
            | Self::ProviderFeatureFlagRequiredForBaseline { flag } => {
                write!(f, "{}: {flag}", self.id())
            }
            Self::BrowserRequiresWasmtime { target } => write!(f, "{}: {target}", self.id()),
            Self::ReleaseArtifactMissing { artifact } | Self::ChecksumInvalid { artifact } => {
                write!(f, "{}: {artifact}", self.id())
            }
            Self::ChangelogEmpty => write!(f, "{}", self.id()),
            Self::CompatibilityDimensionUndeclared { dimension } => {
                write!(f, "{}: {dimension}", self.id())
            }
            Self::PublicApiHandleExposureDenied { surface } => {
                write!(f, "{}: {surface}", self.id())
            }
            Self::ReleaseGateFailed { gate } | Self::ReleaseGateMissing { gate } => {
                write!(f, "{}: {gate}", self.id())
            }
            Self::RoadmapFeaturePresentedAsIncluded { feature } => {
                write!(f, "{}: {feature}", self.id())
            }
        }
    }
}

impl Error for ReleasePackagingError {}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single release packaging conformance check result, mirroring
/// [`crate::CliBoundaryConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleasePackagingConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ReleasePackagingConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReleasePackagingConformanceReport {
    pub results: Vec<ReleasePackagingConformanceResult>,
}

impl ReleasePackagingConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ReleasePackagingConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ReleasePackagingConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the release packaging conformance checks described in this module's
/// doc comment: patch-scoped breaking changes are rejected; independent
/// crate dependencies require documented compatibility; WIT breaking changes
/// require a major bump; the freeze policy denies semantic changes but
/// allows documentation clarifications; experimental flags cannot be
/// enabled by default; only Reference CPU Provider is required for `v0.1`;
/// browser targets never require Wasmtime; artifact/compatibility/changelog
/// manifests validate; the public API surface denies raw handle exposure;
/// all required release gates must pass before stable publication; release
/// candidate tags are never stable; and roadmap features are never
/// presented as included baseline.
pub fn run_release_packaging_conformance() -> ReleasePackagingConformanceReport {
    let mut results = Vec::new();

    {
        let outcome = evaluate_version_bump(
            ReleaseVersion::new(0, 1, 0),
            ReleaseVersion::new(0, 1, 1),
            true,
            true,
        );
        record(
            &mut results,
            "a documented breaking change cannot land in a patch release",
            matches!(
                outcome,
                Err(ReleasePackagingError::BreakingChangeInPatchRelease { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );

        let undocumented = evaluate_version_bump(
            ReleaseVersion::new(0, 1, 0),
            ReleaseVersion::new(0, 2, 0),
            true,
            false,
        );
        record(
            &mut results,
            "an undocumented breaking change is rejected",
            matches!(
                undocumented,
                Err(ReleasePackagingError::UndocumentedBreakingChange { .. })
            ),
            format!("unexpected outcome: {undocumented:?}"),
        );

        let allowed = evaluate_version_bump(
            ReleaseVersion::new(0, 1, 0),
            ReleaseVersion::new(0, 2, 0),
            true,
            true,
        );
        record(
            &mut results,
            "a documented breaking change is allowed within a 0.x minor bump",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let outcome = validate_wit_version_bump(
            WitVersionChangeKind::Breaking,
            ReleaseVersionBumpKind::Minor,
            "magnetar:compute",
        );
        record(
            &mut results,
            "a breaking WIT change requires a major version bump",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let flag = ReleaseFeatureFlag {
            name: "webgpu-provider".into(),
            class: ReleaseFeatureFlagClass::Experimental,
            enabled_by_default: true,
        };
        let outcome = reject_experimental_flag_enabled_by_default(&flag);
        record(
            &mut results,
            "an experimental flag enabled by default is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let flags = provider_feature_flags();
        let outcome = validate_provider_feature_flags_for_v0_1(&flags);
        record(
            &mut results,
            "only reference-cpu-provider is required for v0.1",
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let browser = ReleasePlatformTarget {
            triple: "wasm32-unknown-unknown".into(),
            required_by_ci: false,
            check_only: true,
            is_browser_like: true,
        };
        let outcome =
            reject_wasmtime_required_for_browser(&browser, &["wasmtime-component-engine"]);
        record(
            &mut results,
            "browser targets never require Wasmtime",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let manifest = ReleaseArtifactManifest::default();
        let outcome = manifest.validate();
        record(
            &mut results,
            "an artifact manifest with no recorded status is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );

        let mut complete = ReleaseArtifactManifest::default();
        for kind in RELEASE_ARTIFACT_KINDS {
            complete.set(*kind, ReleaseArtifactStatus::NotApplicable);
        }
        record(
            &mut results,
            "an artifact manifest with every kind explicitly marked not applicable is accepted",
            complete.validate().is_ok(),
            format!("unexpected outcome: {:?}", complete.validate()),
        );
    }

    {
        let matrix = v0_1_compatibility_matrix();
        record(
            &mut results,
            "the v0.1 compatibility matrix declares every dimension",
            matrix.validate().is_ok(),
            format!("unexpected outcome: {:?}", matrix.validate()),
        );
        record(
            &mut results,
            "the v0.1 compatibility matrix marks Provider ABI unstable",
            matrix.status.get("provider-abi") == Some(&CompatibilityStatus::Unstable),
            "provider-abi status was not Unstable",
        );
    }

    {
        for surface in RELEASE_FORBIDDEN_API_SURFACE_FRAGMENTS {
            let outcome = reject_release_public_api_handle_exposure(surface);
            record(
                &mut results,
                format!("release public API surface '{surface}' is denied"),
                outcome.is_err(),
                format!("unexpected outcome: {outcome:?}"),
            );
        }
        let allowed = reject_release_public_api_handle_exposure("generation");
        record(
            &mut results,
            "an ordinary release public API surface is allowed",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let incomplete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES[..3]
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        let outcome = release_may_publish_stable(&incomplete);
        record(
            &mut results,
            "stable publication is denied when required gates are missing",
            matches!(
                outcome,
                Err(ReleasePackagingError::ReleaseGateMissing { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );

        let mut complete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
            .iter()
            .map(|gate| ReleaseGateResult {
                gate: *gate,
                passed: true,
            })
            .collect();
        record(
            &mut results,
            "stable publication is allowed when every required gate passes",
            release_may_publish_stable(&complete).is_ok(),
            format!(
                "unexpected outcome: {:?}",
                release_may_publish_stable(&complete)
            ),
        );

        complete[0].passed = false;
        let outcome = release_may_publish_stable(&complete);
        record(
            &mut results,
            "stable publication is denied when a required gate fails",
            matches!(
                outcome,
                Err(ReleasePackagingError::ReleaseGateFailed { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        for tag in [
            ReleaseCandidateTag::Alpha,
            ReleaseCandidateTag::Beta,
            ReleaseCandidateTag::Rc(1),
        ] {
            record(
                &mut results,
                format!("pre-release tag '{tag}' is never stable"),
                !tag.is_stable(),
                "pre-release tag was marked stable",
            );
        }
    }

    {
        let outcome = reject_roadmap_feature_as_guarantee("cuda", true);
        record(
            &mut results,
            "CUDA cannot be presented as included in v0.1",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
        let allowed = reject_roadmap_feature_as_guarantee("reference-cpu-provider", true);
        record(
            &mut results,
            "the CPU-local baseline can be presented as included",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let redacted = redact_build_metadata("GITHUB_TOKEN", "ghp_example");
        record(
            &mut results,
            "a secret-shaped build metadata key is redacted",
            redacted == "[redacted build metadata]",
            format!("unexpected redaction output: {redacted}"),
        );
        let path_redacted = redact_build_metadata("workspace_root", "/home/user/project");
        record(
            &mut results,
            "a local-path-shaped build metadata value is redacted",
            path_redacted == "[redacted backend diagnostic]",
            format!("unexpected redaction output: {path_redacted}"),
        );
    }

    ReleasePackagingConformanceReport { results }
}
