## ADDED Requirements

### Requirement: Release Security Hardening

Magnetar SHALL define release security and supply-chain hardening before stable
release.

#### Scenario: Security hardening exists

Given `v0.1` release candidate is prepared

When release gates run

Then security and supply-chain gates are included.

---

### Requirement: v0.1 Security Scope

`v0.1` security scope SHALL cover the CPU-local baseline and SHALL not claim
hardened production security for deferred roadmap features.

#### Scenario: CUDA deferred

Given CUDA is not part of `v0.1`

When release security notes are generated

Then CUDA is not claimed as hardened.

---

### Requirement: Dependency Audit

Stable release SHALL run dependency audit.

#### Scenario: Critical advisory

Given required dependency has critical advisory without mitigation

When stable release is attempted

Then release is blocked.

---

### Requirement: License Audit

Stable release SHALL run license audit.

#### Scenario: Incompatible license

Given required dependency has incompatible license

When release is attempted

Then release is blocked unless approved exception exists.

---

### Requirement: SBOM Or Documented Limitation

Release SHALL include SBOM or document SBOM limitation.

#### Scenario: No SBOM tooling

Given SBOM is unavailable

When release notes are generated

Then limitation is stated.

---

### Requirement: Release Checksums

Release artifacts SHALL include checksums generated from final artifacts.

#### Scenario: Binary checksum

Given binary artifact is produced

When release metadata is generated

Then checksum is included.

---

### Requirement: Signature Status

Release SHALL document whether signatures are provided.

#### Scenario: Unsigned release

Given signatures are not implemented

When release notes are generated

Then release states checksums exist but signatures are unavailable.

---

### Requirement: Provenance Metadata

Release SHALL include provenance metadata, and provenance SHALL exclude
secrets or local paths.

#### Scenario: Provenance generated

Given release metadata includes CI run

When inspected

Then commit, tag, build target, and lockfile digest may appear without secrets.

---

### Requirement: Lockfile Integrity

Release SHALL use reviewed lockfile state where appropriate.

#### Scenario: Lockfile drift

Given lockfile changed unexpectedly

When release candidate is prepared

Then release is blocked pending review.

---

### Requirement: Secret Scanning

Stable release SHALL run secret scanning on source and release artifacts.

#### Scenario: Secret detected

Given secret scan detects credential

When stable release is attempted

Then release is blocked.

---

### Requirement: Redaction Gate

Stable release SHALL pass redaction gates.

#### Scenario: Raw prompt in diagnostics

Given diagnostics include raw prompt by default

When release gate runs

Then release is blocked.

---

### Requirement: Provider Trust Boundary

Release security notes SHALL state that Providers are trusted native code.

#### Scenario: Provider docs

Given release notes describe Providers

When user reads them

Then native trust boundary is explicit.

---

### Requirement: Native Handle Boundary

Release APIs, diagnostics, and reports SHALL not expose native handles or memory
pointers.

#### Scenario: Device pointer leak

Given diagnostics include raw device pointer

When release gate runs

Then release is blocked.

---

### Requirement: Component Artifact Trust Boundary

Component Artifacts SHALL be validated before execution.

Unsigned Component Artifacts SHALL be denied in production policy unless
explicitly allowed.

#### Scenario: Unsigned production component

Given unsigned Component Artifact is loaded under production policy

When Runtime validates it

Then execution is denied.

---

### Requirement: Model Artifact Trust Boundary

Model Artifacts SHALL pass trust and integrity validation before loading.

#### Scenario: Recognized untrusted format

Given model format is recognized but source is untrusted

When Model Loading runs

Then loading is denied.

---

### Requirement: Source Cache Trust Boundary

Cache hit, source kind, alias, local file, and fixture status SHALL not imply
trust.

#### Scenario: Cached revoked artifact

Given cached artifact is revoked

When loading runs

Then Runtime rejects it.

---

### Requirement: CLI Authority Not Delegated

CLI authority SHALL not become Runtime ambient authority.

#### Scenario: CLI has filesystem access

Given CLI can read workspace files

When Runtime is called

Then Runtime receives explicit prompt data only.

---

### Requirement: Runtime Inference API Security

Runtime Inference API SHALL remain inference-only.

#### Scenario: Runtime network request

Given Runtime request asks for arbitrary network access

When validation runs

Then request is rejected.

---

### Requirement: Unsafe Code Policy

Release SHALL document unsafe code policy and review unsafe usage where present.

#### Scenario: Unsafe block present

Given unsafe block exists in required baseline

When release review runs

Then unsafe rationale is documented.

---

### Requirement: Dependency Feature Review

Release SHALL review enabled dependency features for unexpected capability
expansion.

#### Scenario: Network feature enabled

Given dependency feature enables networking unexpectedly

When release review runs

Then release is blocked or exception is documented.

---

### Requirement: Vulnerability Handling Policy

Release SHALL define vulnerability handling policy.

#### Scenario: High advisory accepted

Given high advisory is accepted with mitigation

When release notes are generated

Then exception and mitigation are documented.

---

### Requirement: Security Notes

Release SHALL include security notes describing threat model, trust boundaries,
redaction, unsupported security features, known risks, and reporting process
placeholder.

#### Scenario: Security section

Given release notes are published

When security section is inspected

Then limitations are explicit.

---

### Requirement: Release Blocking Security Criteria

Stable release SHALL be blocked by secrets, critical unmitigated advisories,
incompatible licenses, redaction failure, raw handle exposure, trust/integrity
failure, E2E bypass, OpenSpec failure, checksum mismatch, or undocumented
security exception.

#### Scenario: Raw handle exposed

Given public API exposes raw Provider handle

When stable release is attempted

Then release is blocked.

---

### Requirement: Security Exceptions Documented

Security exceptions SHALL be documented with issue, component, severity,
rationale, mitigation, owner, follow-up, and release note entry.

#### Scenario: Exception exists

Given exception is required

When release is prepared

Then undocumented exception blocks release.

---

### Requirement: Release Security Observability

Release process SHALL record security hardening observations with default
redaction.

#### Scenario: Secret scan completed

Given secret scan completes

When release metadata is recorded

Then result is recorded without exposing secrets.