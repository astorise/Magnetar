# v0.1 Release Cutover Checklist

Magnetar now has architecture contracts, an implementation baseline plan,
post-baseline roadmaps, a release packaging policy, a release security
policy, and release conformance gates. The final step before stabilizing the
first release is a concrete cutover checklist: the operational sequence
required to move from release candidate to stable `v0.1`. This document, and
the `magnetar-runtime::release_cutover` module it describes, define that
checklist for `v0.1`.

This document and module do **not** implement release automation, publish
the release, define registry credentials, define a final hosting provider,
define a legal approval process, guarantee production security, or include
GPU Providers, a server API implementation, model hub downloads, or an
agent/tool runtime (see
`openspec/changes/define-v0-1-release-cutover-checklist/proposal.md`'s
"Non-Goals"). They define the cutover **sequence** as executable Rust types
and validation functions.

`release_cutover` is deliberately an **orchestrator**: every check composes
an existing policy module (`magnetar-runtime::release_packaging`,
`magnetar-runtime::release_security`) rather than re-implementing packaging,
versioning, or security logic a sibling module already owns. The sequence it
enforces is:

```text
freeze -> gates -> reports -> artifacts -> tag -> publish -> verify
```

## Release Readiness

`ReleaseReadinessChecklist` records the nine readiness confirmations from
"Release Readiness" (branch/commit, version, OpenSpec baseline, scope,
gates, artifacts, release notes draft, compatibility matrix draft, security
notes draft). `validate()` requires every field to hold; a missing release
notes draft blocks cutover readiness.

## OpenSpec Freeze

`OpenSpecFreezeConfirmation` composes `release_packaging::ReleaseFreezeState`
directly rather than a parallel freeze model, plus the five additional
freeze confirmations the checklist names (accepted changes list final,
pending changes excluded, WIT-breaking changes have version bumps, checklist
references correct changes, roadmap items deferred unless included).
`reject_semantic_change_after_freeze` composes
`release_packaging::reject_change_after_freeze`: a semantic change after
freeze is blocked or the freeze must be restarted.

## Scope Confirmation

`validate_v0_1_scope_feature` composes
`release_packaging::reject_roadmap_feature_as_guarantee` rather than a
second deferred-roadmap-feature list. `V0_1_INCLUDED_SCOPE` documents the
`v0.1` included baseline; enforcement lives entirely in the composed
function -- CUDA (or any other deferred roadmap feature) listed as included
is rejected the same way `release_packaging` already rejects it.

## Version Confirmation

`WitPackageVersionRecord` is deliberately distinct from
`release_packaging::WitInterface` (whose `version` field is always
populated): it exists so "included WIT package lacks version" is
representable at all. `validate_wit_versions_confirmed` rejects any record
with no version. `CutoverVersionConfirmation` bundles the seven version
confirmations the checklist names (release, crate, binary, WIT packages,
conformance suite, OpenSpec baseline, and optional release candidate
lineage). `validate_runtime_version_matches_release_tag` compares a
`release_packaging::ReleaseBinaryVersionReport`'s reported binary version
against the release tag.

## Feature Flag Confirmation

`validate_cutover_feature_flag` composes
`release_packaging::reject_experimental_flag_enabled_by_default` and
additionally denies a `TestOnly` or `ConformanceOnly` flag enabled by
default in a release build, which the packaging-level check does not itself
enforce. `validate_cutover_provider_feature_flags` composes
`release_packaging::validate_provider_feature_flags_for_v0_1`.

## Compatibility Matrix Completion

`CutoverCompatibilityDimension` names the twelve dimensions the checklist
requires -- a superset of `release_packaging::CompatibilityDimension`'s
eight (adding Tokenizer Artifact metadata, Adapter Artifact metadata,
supported targets, and feature flags), so this is deliberately a distinct
type rather than reusing the narrower packaging-level one.
`CutoverCompatibilityStatus` is the six-value approved status vocabulary
(`stable-for-v0.1-baseline`, `preview`, `experimental`, `unstable`,
`deferred`, `unsupported`). `CutoverCompatibilityMatrix::validate` requires
every dimension to carry an explicit status -- a missing Provider ABI (or
missing WIT packages) entry blocks release. `reject_status_misrepresentation`
rejects presenting anything less stable than `StableForV01Baseline` as if it
were `StableForV01Baseline`, implementing "experimental API presented
stable blocks release."

## Required Gate Execution

`validate_required_gates_executed` composes
`release_packaging::release_may_publish_stable` directly: every gate in
`release_packaging::REQUIRED_RELEASE_GATES` (including E2E local
conformance, OpenSpec validation, and WIT validation) SHALL be present and
passed.

## Skip Review

`GateSkip` records a single gate skip and the four conditions "Skip Review"
requires for it to be allowed (outside `v0.1` scope, documented reason, does
not hide a baseline failure, included in the release report).
`validate_gate_skips` rejects any skip that fails even one condition -- a
Reference CPU Provider skip is disallowed because it is not outside `v0.1`
scope, regardless of how well-documented the reason is.

## Exception Review

`CutoverException` composes `release_security::SecurityException` (which
already carries issue/component/severity/rationale/mitigation/owner/
expiration/release-note fields) rather than a parallel exception record,
adding only the `gate` field "Exception Review" additionally names.
`reject_undocumented_cutover_exception` and `validate_cutover_exceptions`
block release on an incomplete or missing exception.

## Security Verification

`CutoverSecurityVerification` composes
`release_security::ReleaseSecurityGateInputs` /
`release_security::evaluate_release_security_blocking` and
`release_security::SecurityReleaseNotes` rather than re-checking dependency
audit, license audit, secret scan, redaction, native handle, trust/
integrity, artifact integrity, checksum, SBOM, or signature status a second
time.

## Artifact Generation

`validate_cutover_artifacts_generated` composes
`release_packaging::ReleaseArtifactManifest::validate` directly: every
artifact kind SHALL be `Present` or explicitly `NotApplicable` -- a missing
conformance report blocks stable release unless explicitly marked not
applicable and justified.

## Artifact Verification

`CutoverArtifactVerification` composes
`release_security::ArtifactIntegrityStatus` (the five integrity checks:
clean/CI-controlled source state, release tag matching source, OpenSpec
report matching baseline, conformance reports matching commit, and checksums
matching final artifacts) and adds only the two cutover-specific checks that
struct does not carry: release notes matching the compatibility matrix, and
artifact names including the version where appropriate.
`verify_cutover_artifact_checksum` composes
`release_security::verify_checksum_matches_final_artifact`: a checksum
mismatch blocks or withdraws the release.

## Changelog Completion

`CutoverChangelogChecklist` composes `release_packaging::ReleaseChangelog`
(non-empty entries) and additionally requires the nine cutover-specific
categories the checklist names (added/changed/removed contracts, release
scope, known limitations, compatibility status, security notes, conformance
status, deferred roadmap items) -- a known limitation missing from the
changelog blocks release.

## Release Notes Completion

`CutoverReleaseNotesChecklist` records the thirteen topics release notes
SHALL answer or include (what `v0.1` is, what users can run, stable/
preview/experimental/deferred/unsupported status explanations, artifact
verification, security limitations, how to run conformance, and the
compatibility matrix / security notes / known limitations themselves).
`validate()` requires every topic present.

## Tagging

`validate_tag_after_gates` composes `release_packaging::release_may_publish_stable`:
"stable release tag SHALL be created only after required gates pass" --
since `REQUIRED_RELEASE_GATES` includes both WIT validation and Runtime
Inference API tests, this single function also implements "WIT validation
SHALL complete before stable release tag" and "Runtime Inference API
baseline gates SHALL pass before stable tag creation" without a parallel
per-gate ordering check.

## Publication

`validate_publication_scope_preserved` composes
`validate_v0_1_scope_feature` (roadmap-feature inclusion) and
`reject_status_misrepresentation` (stability misrepresentation): "server API
claimed included in `v0.1` publication, but server API gates were skipped
as deferred" is blocked the same way any other deferred-roadmap-feature
inclusion claim is blocked.

## Post-Publication Verification

`PostPublicationVerification` records the eight checks "Post-Publication
Verification" SHOULD confirm (published artifacts match checksums, release
notes visible, reports accessible, version command matches tag,
documentation links valid, compatibility matrix visible, security notes
visible, deferred roadmap clearly separated). A binary version output that
differs from the release tag marks the release invalid pending correction.

## Rollback And Retraction Notes

`RollbackRetractionNotes` records the five procedures the release process
SHOULD describe for an invalid published release (withdrawal, advisory
publication, patch release, audit trail preservation, release notes
update).

## Post-v0.1 Handoff

`POST_V0_1_HANDOFF_CANDIDATES` lists the roadmap candidates named in
"Post-v0.1 Handoff" (implementation hardening, optimized CPU Provider, model
format support, source/cache implementation, server API implementation,
production CLI UX, GPU Provider exploration, quantized inference, advanced
attention). `PostV01HandoffItem` and
`reject_post_v0_1_item_as_release_claim` enforce "post-v0.1 items SHALL
remain separate from `v0.1` release claims."

## Final Release Statement

`V0_1_FINAL_RELEASE_STATEMENT` is the exact statement text from the
proposal. `validate_final_release_statement` rejects any statement text
missing the CPU-local baseline, Runtime Inference API, Reference CPU
Provider, or E2E local conformance phrases.

## Observability

`ReleaseCutoverObservation` and `record_release_cutover_observation` compose
`CorrelationId` (from `magnetar-runtime::observability`, for gate/target/
feature-set/artifact correlation) and
`release_security::record_release_security_observation` (for default
redaction) instead of a third redaction implementation -- a cutover
observation can never leak a secret, credential, prompt, handle, or local
path by default, and always carries enough correlation metadata to trace a
gate failure back to its target, feature set, and artifact.

## CLI Boundary / Runtime Scope

`validate_cutover_cli_boundary` composes
`release_security::validate_cli_authority_not_delegated_to_runtime` (which
itself already composes `cli_boundary::reject_cli_owned_authority`) rather
than importing the CLI boundary module a second time.
`validate_cutover_runtime_scope` composes
`release_security::validate_runtime_inference_api_security` (which itself
already composes `inference_api::validate_inference_scope`). Together these
implement "CLI boundary gate failed blocks release" and "Runtime includes
tool execution blocks release" without a third capability list.

## Release Blocking Criteria

`ReleaseCutoverGateInputs` records seventeen top-level blocking-criteria
inputs, in the shape of `release_security::ReleaseSecurityGateInputs`.
`evaluate_release_cutover` reports every triggered reason, not just the
first, implementing the "Cutover Principle": "a stable `v0.1` release SHALL
be cut only after required gates pass and release metadata is complete."

## Conformance

`run_release_cutover_conformance` asserts the guarantees above hold:
incomplete readiness, an unfrozen baseline, a roadmap feature presented as
included, a missing WIT version, a missing WIT-packages compatibility
dimension, a runtime version mismatch, an experimental or non-baseline flag
enabled by default, a missing Provider ABI compatibility dimension, a
misrepresented stability status, a missing/failed required gate (including
E2E local conformance), a disallowed gate skip (including a required
Reference CPU Provider skip), an undocumented exception, a failed security
verification, incomplete artifact generation or verification, a checksum
mismatch, an incomplete changelog or release notes, a tag created before
gates pass (including before WIT validation), a publication scope violation
(including a server API inclusion claim), a failed post-publication
verification, incomplete rollback notes, a post-`v0.1` item presented as a
release claim, an invalid final release statement, a denied CLI/Runtime
authority expansion, and a redacted, correlatable cutover observation are
all rejected or accepted as documented above.
