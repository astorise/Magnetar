# Release Packaging And Versioning Policy

Before Magnetar's first release, the project needs explicit packaging and
versioning rules -- without them it risks publishing unstable contracts as
stable, breaking WIT compatibility accidentally, mixing future roadmap
features into the baseline release, or shipping artifacts that cannot be
reproduced or validated. This document, and the
`magnetar-runtime::release_packaging` module it describes, define that
policy for `v0.1`.

This document and module do **not** implement release automation, a package
registry, or supply-chain signing (see
`openspec/changes/define-release-packaging-and-versioning-policy/proposal.md`'s
"Non-Goals"). They define the release packaging and versioning **policy** as
executable Rust types and validation functions, plus the `magnetar version`
command that reports it.

## Release Scope

`v0.1` is scoped to the CPU-local baseline: Runtime Inference API, Model
Loading, Model Instance, the tokenizer fixture path, the Qwen-like baseline
fixture path, Generation + Sampling, Tensor + Memory, the first Operator
scope, Kernel Registry + Dispatch, Reference CPU Provider, the CLI boundary
harness, and E2E local conformance. It does not require CUDA, Metal,
OpenVINO, QNN, WebGPU, production Qwen model support, large model execution,
model hub downloads, a server API implementation, an agent/tool runtime, or
production CLI UX. `classify_publishing_boundary` and
`reject_roadmap_feature_as_guarantee` make this mechanically enforceable:
every name in `DEFERRED_ROADMAP_FEATURES` (`cuda`, `metal`, `openvino`,
`qnn`, `webgpu`, `server-api-implementation`, ...) is rejected if presented
as included baseline.

## Versioning Policy

`ReleaseVersion` is an explicit `major.minor.patch` semantic version.
`evaluate_version_bump` implements the policy directly: before `1.0`,
breaking changes MAY occur but SHALL be documented
(`ReleasePackagingError::UndocumentedBreakingChange` otherwise); within
`0.x`, minor versions MAY include breaking changes if documented; patch
versions SHALL NOT carry a breaking change
(`ReleasePackagingError::BreakingChangeInPatchRelease`).

## Crate Versioning

`CrateVersionMetadata` records a publishable crate's version and whether it
shares the workspace version. `validate_crate_dependency_compatibility`
requires a documented compatibility note whenever a dependent and its
dependency do not both share the workspace version.

## Binary Versioning

`magnetar version` (also `--version` / `-V`) prints the
`ReleaseBinaryVersionReport` built by `build_release_binary_version_report`:
binary version, Runtime crate version (`MAGNETAR_RUNTIME_VERSION`), OpenSpec
baseline version, WIT contract versions (`release_wit_contract_versions`,
reflecting `magnetar-runtime/wit/*.wit`), enabled feature flags, build
profile, commit hash where available (`MAGNETAR_COMMIT_HASH` at build time),
and conformance suite version where available.

## WIT Versioning

`WitVersionChangeKind` classifies a WIT package change as `Breaking`,
`Additive`, or `DocumentationOnly`. `required_wit_version_bump` maps each
kind to the minimum `ReleaseVersionBumpKind` it requires (major / minor /
patch respectively), and `validate_wit_version_bump` rejects an actual bump
smaller than required -- a breaking change without a major version bump is a
`ReleasePackagingError::WitVersionBumpInsufficient`.
`SupportedWitVersionMatrix` documents the supported package versions.

## OpenSpec Baseline

`OpenSpecBaselineDeclaration` carries the accepted changes, validation
status, compatibility notes, deferred changes, conformance status, and
release tag a release SHALL declare. `validate()` requires at least one
accepted change and a recorded validation status.

## Change Freeze

`ReleaseFreezeState` (`Open` / `Frozen`) and `reject_change_after_freeze`
implement the freeze policy: once frozen, every `ReleaseFreezeChangeKind`
except `DocumentationClarification` is denied
(`ReleasePackagingError::ReleaseFrozen`). While open, every change kind is
allowed.

## Feature Flags

`ReleaseFeatureFlagClass` names the six classes (baseline, experimental,
provider-specific, platform-specific, test-only, conformance-only).
`reject_experimental_flag_enabled_by_default` denies an experimental flag
enabled by default. `provider_feature_flags` lists the seven Provider flags
from the proposal; `validate_provider_feature_flags_for_v0_1` requires that
only `reference-cpu-provider` be enabled by default.
`component_engine_feature_flags` lists the Wasmtime/web/test component
engine flags (all disabled by default), and
`reject_wasmtime_required_for_browser` denies a browser-like platform target
that requires `wasmtime-component-engine`.

## Platform Targets

`release_platform_targets` returns the `v0.1` native CI targets plus a
check-only `wasm32-unknown-unknown` browser target. `unsupported_targets`
reports any candidate target not in the supported list, for release
documentation to record.

## Release Artifacts

`ReleaseArtifactKind` enumerates the eleven artifact kinds (source archive,
Rust crates, CLI binary, conformance report, E2E report, OpenSpec validation
report, coverage report, SBOM placeholder, checksums, changelog, release
notes). `ReleaseArtifactManifest::validate` requires every kind to be
recorded as `Present` or explicitly `NotApplicable` -- a kind with no
recorded status is rejected, matching "conformance report is present or
explicitly not applicable."

## Checksums

`ArtifactChecksum::new` rejects an empty digest. Checksums are deliberately
never treated as a trust or signature policy substitute: no function in this
module accepts a checksum as input to a trust decision.

## Changelog

`ChangelogEntryKind` names the eight changelog categories (added/changed/
removed contracts, fixed issues, known limitations, conformance status,
compatibility notes, security notes). `ReleaseChangelog::validate` requires
at least one entry.

## Compatibility Policy

`CompatibilityDimension` names the eight dimensions a release SHALL report
compatibility status for (Rust public API, WIT contracts, Runtime Inference
API, Model Artifact metadata, Provider ABI, OpenSpec baseline, CLI command
surface, conformance report format). `ReleaseCompatibilityMatrix::validate`
requires every dimension to have an explicit `CompatibilityStatus`.
`v0_1_compatibility_matrix` gives the `v0.1` matrix implied by this change's
own specs: every dimension is `StableForBaseline` except Provider ABI, which
is `Unstable`.

## Public API Safety

`reject_release_public_api_handle_exposure` denies any release public API
surface name containing a raw Provider/Device/Kernel handle, raw tensor or
memory pointer, raw KV cache, or raw model weight fragment.

## Conformance Versioning

`ReleaseConformanceVersions` reports the six conformance suite versions a
release SHALL include, reusing existing suite-version constants
(`PROVIDER_CONFORMANCE_SUITE_VERSION`, `FIRST_OPERATOR_SCOPE_VERSION`,
`QWEN_BASELINE_CONTRACT_VERSION`, `E2E_SUITE_VERSION`) and introducing the
two this change adds (`RUNTIME_INFERENCE_API_CONFORMANCE_VERSION`,
`CLI_BOUNDARY_CONFORMANCE_VERSION`).

## Release Gates

`ReleaseGate` enumerates the fifteen required gates (formatting, `cargo
check`, Clippy, unit tests, contract tests, OpenSpec validation, WIT
validation, Reference CPU conformance, first-operator-scope conformance,
Runtime Inference API tests, CLI boundary tests, E2E local conformance,
coverage gate, redaction checks, no-raw-handle-exposure checks).
`release_may_publish_stable` denies stable publication unless every gate in
`REQUIRED_RELEASE_GATES` is present in the result set and passed.

## Release Candidate Policy

`ReleaseCandidateTag` (`Alpha` / `Beta` / `Rc(N)`) renders as `-alpha`,
`-beta`, `-rc.N`; `is_stable()` is `false` for every value by construction,
so a release candidate can never be presented as stable.
`ReleaseCandidateManifest::validate` requires a frozen OpenSpec baseline and
an included conformance report. `allow_failed_candidate_as_pre_release`
accepts a failed gate result set only when paired with a
`ReleaseCandidateTag`.

## Build Metadata

`ReleaseBuildMetadata` carries commit hash, build timestamp, target triple,
enabled features, CI run identifier, profile, and rustc version.
`redact_build_metadata` redacts a value whose key looks secret-shaped
(`token`, `secret`, `password`, `api_key`, `credential`) or whose value looks
like a local filesystem path or native handle (via the existing
`redact_backend_diagnostic`).

## Documentation Release

`ReleaseDocumentationChecklist` tracks the ten documentation topics a
release SHOULD publish (architecture, Runtime Inference API, CLI boundary,
build/test/conformance instructions, feature flags, supported targets, known
limitations, post-baseline roadmap). `validate()` enforces the SHALL-strength
minimum: known limitations SHALL be documented.

## Security Notes

`ReleaseSecurityNotes` names the topics a release SHOULD identify (sandbox
assumptions, Provider trust model, no-raw-handle policy, default redaction,
source/cache trust boundary, unsupported security features, known risks).
Detailed security hardening is explicitly deferred to a separate release
security change.

## Publishing Boundary

`PublishingBoundaryCategory` names the four categories (included baseline,
experimental feature, deferred roadmap, unsupported feature).
`classify_publishing_boundary` and `reject_roadmap_feature_as_guarantee`
implement "Publishing SHALL not imply production readiness for all roadmap
features" -- see "Release Scope" above.

## Conformance

`run_release_packaging_conformance` asserts the guarantees above hold:
patch-scoped breaking changes rejected, undocumented crate dependencies
rejected, WIT breaking changes require a major bump, the freeze policy
denies semantic changes but allows documentation clarifications,
experimental flags cannot be enabled by default, only Reference CPU Provider
is required for `v0.1`, browser targets never require Wasmtime,
artifact/compatibility/changelog manifests validate, the public API surface
denies raw handle exposure, all required release gates must pass before
stable publication, release candidate tags are never stable, and roadmap
features are never presented as included baseline.
