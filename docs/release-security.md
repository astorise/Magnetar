# Release Security And Supply-Chain Hardening

Magnetar is an inference runtime that loads external artifacts and trusted
native Providers. Even the first `v0.1` baseline must avoid unsafe release
practices: it must make clear what is trusted, what is untrusted, what is
verified, what is only metadata, what is redacted, and what is explicitly
unsupported. This document, and the `magnetar-runtime::release_security`
module it describes, define that policy for `v0.1`.

This document and module do **not** implement full cryptographic signing,
SLSA compliance, production sandboxing, remote registry authentication,
model hub security, server authentication, or a legal license-approval
process (see
`openspec/changes/define-release-security-and-supply-chain-hardening/proposal.md`'s
"Non-Goals"). They also do not make native Providers untrusted sandboxed
plugins, and do not guarantee all future Providers are secure. They define
the release **security** policy as executable Rust types and validation
functions, composing existing crate contracts (Model/Component trust,
`cli_boundary`, `inference_api`, `provider_roadmap` handle scopes,
`compute::redact_backend_diagnostic`, `release_packaging`) rather than
duplicating them.

## Security Scope For v0.1

`RELEASE_SECURITY_SCOPE_INCLUDED` lists the twelve areas the `v0.1` security
scope covers (Rust source, workspace dependencies, release binaries and
reports, the OpenSpec baseline, WIT packages, Reference CPU Provider, fixture
Model/Tokenizer artifacts, the Runtime Inference API, the CLI boundary
harness, and E2E local conformance).
`RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS` lists the eleven areas
that SHALL not be claimed hardened (CUDA, Metal, OpenVINO, QNN, WebGPU,
server API, model hub downloads, remote registry authentication, production
sandboxing, agent/tool runtime, large third-party model execution).
`reject_hardened_security_claim_for_excluded_feature` mechanically enforces
this: it is deliberately separate from
`release_packaging::classify_publishing_boundary`, which governs whether a
feature may be presented as *included* baseline rather than whether it may be
claimed *hardened*.

## Dependency Audit

`DependencyAdvisorySeverity`, `DependencyAdvisory`, and
`DependencyAuditReport` record known advisories, yanked crates, duplicate
high-risk dependencies, unexpected build scripts, unexpected native
dependencies, and dependency tree drift.
`DependencyAuditReport::validate_for_stable_release` enforces the
SHALL-strength rule: a `Critical`-severity advisory without a recorded
mitigation blocks stable release. The remaining `SHOULD`-strength fields are
recorded but not blocking.

## License Audit

`LicenseAuditStatus`, `DependencyLicense`, and `LicenseAuditReport` record
per-dependency license status and third-party notice generation.
`LicenseAuditReport::validate_for_stable_release` blocks stable release for
an unapproved `Incompatible` or `Unknown` license; `MissingMetadata` is
recorded but not itself blocking, matching the proposal's narrower SHALL
sentence ("Unknown or incompatible licenses ... SHALL block").

## SBOM

`SbomEntry`, `SbomAvailability`, and `SbomManifest` implement "Release
SHOULD produce an SBOM or SBOM placeholder." Each `SbomEntry` carries a
package name, version, license metadata, and source repository metadata
where available; `SbomManifest::build_target` and
`SbomManifest::feature_flags` carry the two manifest-level fields ("build
target metadata", "feature flags where feasible") that apply to the whole
release rather than to a single dependency. `SbomManifest::validate`
requires either a `Generated` manifest with at least one entry or a
`PlaceholderDocumented` manifest with a non-empty limitation note -- silent
absence is always rejected.

## Checksums

`verify_checksum_matches_final_artifact` compares a declared
`release_packaging::ArtifactChecksum` against a digest recomputed from the
final artifact, reusing that type rather than a parallel checksum model.

## Signatures

`SignatureStatus` (`Implemented` / `NotImplementedDocumented` /
`NotImplementedUndocumented`) and `validate_signature_status` implement
"Signature absence SHALL not be hidden": only the undocumented-absence state
is rejected.

## Provenance

`ReleaseProvenance` carries the ten provenance fields the proposal names
(source commit, release tag, CI run identifier, build target/profile, rustc
version, lockfile/OpenSpec-baseline/WIT-package/conformance-report digests).
`ReleaseProvenance::validate` reuses
`release_packaging::redact_build_metadata` (the same helper
`release_packaging::redact_build_metadata` uses for build metadata) to
detect and reject any populated field that redaction would have changed --
"Provenance SHALL not include secrets or local developer paths by default."

## Reproducibility

`ReproducibilityStatus` (`FullyReproducible` / `PartiallyReproducible` /
`NotDocumented`) and `ReproducibilityReport::validate` require either a fully
reproducible status or documented limitations; an undocumented status is
always rejected.

## Lockfile Policy

`LockfileState` and `reject_unreviewed_lockfile_drift` require the lockfile
to be checked in and reject unreviewed drift, implementing "Unreviewed
lockfile drift SHOULD block release candidates."

## Build Script Policy

`BuildScriptReview` and `flag_unexpected_build_script` flag an unexpected,
unreviewed build script in a required dependency.

## Secret Scanning

`SecretScanTarget` names the eight scan targets from the proposal (source
files, generated docs, release notes, conformance reports, E2E reports,
logs, build metadata, packaged artifacts); `SECRET_SCAN_TARGETS` lists all
eight. `SecretScanReport::validate_for_stable_release` blocks stable release
on any detected finding.

## Artifact Integrity

`ArtifactIntegrityStatus` records the five checks from "Artifact Integrity
SHOULD include" (clean/CI-controlled source state, release tag matching
source, OpenSpec report matching baseline, conformance reports matching
commit, checksums matching final artifacts). `validate()` requires every
check to hold.

## Redaction Gates

`RedactionCategory` names the thirteen categories a redaction gate SHALL
verify are absent by default (raw prompts, secrets, credentials, raw file
contents, raw model weights, raw tensor values, raw KV cache contents, raw
Provider/Device/Kernel handles, raw memory pointers, local filesystem paths,
raw cache paths). `validate_redaction_gate` composes
`compute::redact_backend_diagnostic` (native handles, local paths) with an
additional sensitive-content fragment check (prompts, secrets, credentials,
weights, tensors, KV cache, file contents) and rejects any diagnostic that
required redaction.

## Provider Trust Boundary

`ProviderTrustModel` documents that Providers are always trusted native code
in this policy (not a negotiable field). `DynamicProviderLoadingStatus` and
`validate_dynamic_provider_loading_status` compose
`provider::ProviderLoadingMode::is_dynamic` and reject only the
`StableUnreviewed` status for a dynamically loaded Provider -- `Disabled`,
`Experimental`, `MarkedUnstable`, and `SecurityReviewed` are all accepted
ways to present dynamic Provider loading. `ProviderTrustSignalSource` and
`reject_provider_registration_implies_trust` reject a trust decision derived
purely from `RegistrationOnly`, implementing "Provider registration SHALL
not imply Provider trust beyond configured policy."

## Native Handle Boundary

`reject_release_native_handle_exposure` composes
`release_packaging::reject_release_public_api_handle_exposure` (generic
Provider/Device/Kernel/tensor/memory fragments) and
`provider_roadmap::reject_provider_specific_handle_capability`
(CUDA/Metal/OpenVINO/QNN fragments) instead of a third forbidden-fragment
list, adding only the one fragment neither already covers: raw CPU
allocation pointer.

## Component Artifact Trust Boundary

`validate_component_release_execution_trust` composes
`component::ComponentTrustDecision` to require a `Trusted` status before
execution, and additionally denies an unsigned Component Artifact under
production policy unless explicitly allowed.
`reject_component_release_authority_expansion` composes
`inference_api::validate_inference_scope` (OS-capability authority) and
`reject_release_native_handle_exposure` (native handle authority) against a
`component::ComponentAuthorityRequirement`'s `kind` string.

## Model Artifact Trust Boundary

`validate_model_artifact_release_trust` composes `model::ModelTrustDecision`
and requires `ModelTrustStatus::Trusted` regardless of whether the format is
recognized -- "Recognized format SHALL not imply trust."
`FixtureModelTrustPolicy` and `validate_fixture_model_trust` additionally
require `explicit_test_policy_documented` before delegating to the same
trust check -- fixture status alone is never sufficient.

## Source Cache Trust Boundary

`validate_source_cache_release_trust` composes
`model_source_cache_roadmap::CacheEntryMetadata` and requires
`trust_status == ModelTrustStatus::Trusted` regardless of source kind,
alias, lifecycle, or pin state. `NonTrustCacheSignal` names the five signals
that SHALL NOT by themselves imply trust (cache hit, source kind, alias,
local file, fixture status); `reject_cache_signal_alone_as_trust` asserts
that none of them satisfies the trust check on its own.

## CLI Boundary Security

`validate_cli_authority_not_delegated_to_runtime` composes
`cli_boundary::reject_cli_owned_authority` rather than a parallel capability
list, implementing "CLI authority SHALL not become Runtime ambient
authority."

## Runtime Inference API Security

`validate_runtime_inference_api_security` composes
`inference_api::validate_inference_scope`, implementing "Runtime Inference
API SHALL remain inference-only."

## Unsafe Code Policy

`UnsafeCodeReview` and `UnsafeCodePolicy` record per-location review and
justification. `UnsafeCodePolicy::validate` enforces review and
justification only when `deny_unreviewed` is set, implementing "Unsafe code
MAY be denied in release gates unless explicitly allowed."
`magnetar_runtime_unsafe_code_inventory` is the concrete, real audit result
for this crate's `v0.1` required baseline: as of this change the only
`unsafe` surface is `provider::ProviderLoader::load_dynamic`,
`load_dynamic_with_policy`, and `discover_and_load`, each already carrying a
`# Safety` doc comment and gated by `ProviderLoadingPolicy::allows` --
`magnetar_runtime_unsafe_code_inventory().validate()` passes today.

## Dependency Feature Policy

`DependencyFeatureCapability` names the four capability classes (networking,
filesystem expansion, native plugin/dynamic loading, broad OS capability).
`DependencyFeatureReview` and
`reject_unexpected_capability_expanding_feature` block an unexpected,
unaccepted capability-expanding feature.

## Vulnerability Handling

`VulnerabilityHandlingPolicy` records the six policy fields the proposal
names (advisory severity handling, release blocking criteria, mitigation
documentation, exception approval, follow-up tracking, patch release
expectation); `validate()` requires all six to be defined.

## Security Notes

`SecurityReleaseNotes` carries the full security notes a release SHALL
include (`v0.1` threat model, trusted native Provider model, no-raw-handle
policy, default redaction, source/cache and Model/Component Artifact trust
boundaries, unsupported security features, known risks, reporting process
placeholder). `validate()` enforces the SHALL-strength minimum: threat
model, trusted Provider model, no-raw-handle policy, default redaction, and
reporting process placeholder must all be present; the remaining fields are
`SHOULD`-strength.

This type is named `SecurityReleaseNotes`, not `ReleaseSecurityNotes`:
`release_packaging::ReleaseSecurityNotes` already occupies that name as a
deliberately shallow placeholder whose own doc comment defers hardening
detail to "a separate release security change" -- `SecurityReleaseNotes` is
that detail.

## Release Blocking Criteria

`ReleaseSecurityGateInputs` records the ten gate inputs from "Release
Blocking Criteria" (secrets detected, critical advisory unmitigated,
incompatible license unapproved, redaction gate failed, raw handle exposed,
trust/integrity failure in required fixtures, E2E conformance bypass,
OpenSpec validation failure, checksum mismatch, undocumented security
exception). `evaluate_release_security_blocking` reports every triggered
reason, not just the first.

## Security Exceptions

`SecurityException` records the eight fields an exception SHOULD include
(issue, affected component, severity, rationale, mitigation, owner,
expiration/follow-up, release note entry). `validate()` requires every field
populated and a release note entry.
`reject_undocumented_security_exception` denies an undocumented exception
when one is required.

## Observability

`ReleaseSecurityObservationKind` names the ten observation kinds from
"Observability" (dependency audit completed, license audit completed, SBOM
generated, checksum generated, secret scan completed, redaction gate
completed, provenance generated, security exception recorded, release
blocked, release security passed). `record_release_security_observation`
always redacts its detail through the same sensitive-content check
`validate_redaction_gate` uses, so an observation can never leak a secret,
credential, prompt, handle, or local path by default.

## Conformance

`run_release_security_conformance` asserts the guarantees above hold: CUDA
cannot be claimed hardened; a critical unmitigated advisory and an
unapproved incompatible license both block stable release; a missing SBOM
without a documented limitation is rejected; a checksum mismatch is
rejected; an undocumented signature absence is rejected; provenance
containing a secret- or path-shaped field is rejected; undocumented
reproducibility and unreviewed lockfile drift are rejected; an unexpected
unreviewed build script is flagged; a detected secret blocks stable release;
incomplete artifact integrity is rejected; a diagnostic containing a raw
prompt or native handle fails the redaction gate; an unreviewed stable
dynamic Provider loading status and a registration-only Provider trust
source are both rejected; every native handle surface fragment is denied;
an untrusted or unsigned-in-production Component Artifact is denied; a
Component/CLI/Runtime authority expansion request is denied; an untrusted
Model Artifact is denied regardless of recognized format; an undocumented
fixture trust policy is rejected; a cache entry without an explicit
`Trusted` status is rejected regardless of which non-trust signal is
present; unreviewed unsafe code and an unexpected capability-expanding
dependency feature are rejected; an incomplete vulnerability handling policy
and incomplete security notes are rejected; every release blocking
criterion is evaluated; an undocumented security exception is rejected; and
a release security observation is always redacted.
