## ADDED Requirements

### Requirement: v0.1 Release Cutover Checklist

Magnetar SHALL define a cutover checklist before publishing stable `v0.1`.

#### Scenario: Cutover starts

Given release candidate is ready

When cutover begins

Then release readiness, freeze, gates, reports, artifacts, tag, publication, and
verification are covered.

---

### Requirement: Release Readiness Confirmed

Cutover SHALL confirm release branch or commit, version, OpenSpec baseline,
scope, gates, artifacts, release notes, compatibility matrix, and security
notes.

#### Scenario: Missing release notes

Given release notes draft is missing

When cutover readiness runs

Then stable cutover is blocked.

---

### Requirement: OpenSpec Freeze Confirmed

Cutover SHALL confirm included OpenSpec changes are frozen.

#### Scenario: Semantic change after freeze

Given semantic change occurs after freeze

When cutover validates release

Then release is blocked or freeze is restarted.

---

### Requirement: v0.1 Scope Confirmed

Cutover SHALL confirm `v0.1` scope is CPU-local baseline and roadmap features are
deferred unless explicitly included.

#### Scenario: CUDA listed as included

Given CUDA is listed as included without passing required gates

When cutover validates scope

Then release is blocked.

---

### Requirement: Version Confirmation

Cutover SHALL confirm release, crate, binary, WIT, conformance, and OpenSpec
baseline versions.

#### Scenario: Missing WIT version

Given included WIT package lacks version

When cutover runs

Then release is blocked.

---

### Requirement: Feature Flag Confirmation

Cutover SHALL confirm default features are baseline-safe and experimental
features are disabled by default.

#### Scenario: Experimental WebGPU enabled

Given experimental WebGPU is enabled by default

When cutover validates feature flags

Then release is blocked.

---

### Requirement: Compatibility Matrix Complete

Cutover SHALL require complete compatibility matrix using approved status
vocabulary.

#### Scenario: Provider ABI missing

Given Provider ABI status is missing from matrix

When cutover runs

Then release is blocked.

---

### Requirement: Required Gates Executed

Cutover SHALL execute all required gates from release conformance policy.

#### Scenario: E2E not run

Given E2E local inference gate did not run

When cutover runs

Then release is blocked.

---

### Requirement: Skips Reviewed

Cutover SHALL review all skipped gates and require documented allowed skip
reasons.

#### Scenario: Required gate skipped

Given Reference CPU gate is skipped

When cutover reviews skips

Then release is blocked.

---

### Requirement: Exceptions Reviewed

Cutover SHALL review and document all exceptions.

#### Scenario: Undocumented exception

Given exception has no mitigation or owner

When cutover runs

Then release is blocked.

---

### Requirement: Security Verified

Cutover SHALL confirm dependency audit, license audit, secret scan, redaction,
native handle checks, trust/integrity checks, artifact integrity, checksums,
SBOM/signature status, and security notes.

#### Scenario: Secret scan missing

Given secret scan was not run

When cutover validates security

Then release is blocked.

---

### Requirement: Artifacts Generated

Cutover SHALL generate or explicitly mark release artifacts as not applicable.

#### Scenario: Missing conformance report

Given conformance report is not generated

When release artifacts are checked

Then stable release is blocked unless explicitly not applicable and justified.

---

### Requirement: Artifacts Verified

Cutover SHALL verify final artifacts match release commit, reports, baseline,
versions, checksums, and redaction policy.

#### Scenario: Checksum mismatch

Given published binary checksum differs from release metadata

When verification runs

Then release is blocked or withdrawn.

---

### Requirement: Changelog Complete

Cutover SHALL require changelog with contracts, scope, limitations,
compatibility, security, conformance, and deferred roadmap items.

#### Scenario: Known limitation missing

Given known limitation exists but is absent from changelog

When cutover validates changelog

Then release is blocked.

---

### Requirement: Release Notes Complete

Cutover SHALL require release notes explaining what is included, stable,
preview, experimental, deferred, unsupported, verifiable, and limited.

#### Scenario: Experimental API presented stable

Given release notes present experimental API as stable

When cutover validates notes

Then release is blocked.

---

### Requirement: Tagging After Gates

Stable release tag SHALL be created only after required gates pass.

#### Scenario: Tag created early

Given stable tag exists before gates pass

When cutover validates process

Then tag must be removed, replaced, or release marked invalid.

---

### Requirement: Publication Preserves Scope

Publication SHALL not present deferred roadmap features as included or
experimental APIs as stable.

#### Scenario: Server API claimed included

Given server API is claimed in `v0.1` publication

But server API gates were skipped as deferred

When cutover verifies publication

Then release is blocked or corrected.

---

### Requirement: Post-Publication Verification

Cutover SHALL verify published artifacts, notes, reports, version output,
documentation, compatibility matrix, security notes, and roadmap separation.

#### Scenario: Version mismatch

Given binary version output differs from release tag

When post-publication verification runs

Then release is marked invalid pending correction.

---

### Requirement: Rollback And Retraction Notes

Cutover SHALL define rollback and retraction notes.

#### Scenario: Invalid release discovered

Given release is found invalid after publication

When rollback process begins

Then withdrawal, advisory, patch release, audit trail, and release note update
steps are available.

---

### Requirement: Post-v0.1 Handoff

Cutover SHALL define post-v0.1 roadmap handoff without turning roadmap items
into release claims.

#### Scenario: Next roadmap listed

Given optimized CPU Provider is listed as next work

When release notes are read

Then it is clearly post-v0.1 and not included in `v0.1`.

---

### Requirement: Final Release Statement

Cutover SHALL include final release statement describing `v0.1` as CPU-local
inference runtime baseline validated by Runtime Inference API, Reference CPU
Provider, and E2E local conformance.

#### Scenario: Release statement

Given release notes are finalized

When statement is inspected

Then it accurately describes included baseline and excludes roadmap features.