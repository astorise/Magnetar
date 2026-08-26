## ADDED Requirements

### Requirement: Release Packaging Policy

Magnetar SHALL define release packaging and versioning policy before publishing
the first release.

#### Scenario: Release policy exists

Given `v0.1` release work begins

When release artifacts are prepared

Then packaging, versioning, compatibility, and gates are defined.

---

### Requirement: v0.1 Scope

`v0.1` SHALL be scoped to CPU-local baseline unless explicitly expanded by
release checklist.

#### Scenario: GPU unavailable

Given CUDA is unavailable

When `v0.1` release gates run

Then required release gates can still pass.

---

### Requirement: Semantic Versioning

Magnetar release artifacts SHALL use explicit semantic versions.

#### Scenario: Patch release

Given patch release is prepared

When compatibility is evaluated

Then breaking changes are avoided or release is not patch.

---

### Requirement: Binary Version Reporting

Magnetar binaries SHALL report version metadata.

#### Scenario: Version command

Given user runs version command

When output is produced

Then binary version, Runtime version, WIT versions, feature flags, and build
metadata are available where applicable.

---

### Requirement: WIT Versioning

WIT packages SHALL be explicitly versioned.

Breaking WIT changes SHALL require major WIT version bump.

#### Scenario: Breaking WIT field removal

Given WIT field is removed

When release is prepared

Then WIT major version is bumped or release is rejected.

---

### Requirement: OpenSpec Baseline Declaration

A release SHALL declare the OpenSpec baseline it implements.

#### Scenario: Release notes

Given release notes are generated

When OpenSpec baseline is listed

Then accepted changes and validation status are included.

---

### Requirement: Release Freeze

Included OpenSpec contracts SHALL be frozen before stable release.

#### Scenario: Late semantic change

Given semantic contract change is proposed after freeze

When release is prepared

Then it requires new change proposal or release is delayed.

---

### Requirement: Feature Flag Classification

Release feature flags SHALL be classified as baseline, experimental,
Provider-specific, platform-specific, test-only, or conformance-only.

#### Scenario: CUDA feature

Given CUDA feature exists

When `v0.1` is prepared

Then it is not required for baseline release.

---

### Requirement: Experimental Features Disabled By Default

Experimental features SHALL not be enabled by default in stable release builds.

#### Scenario: Experimental WebGPU

Given WebGPU feature is experimental

When default build runs

Then WebGPU is disabled.

---

### Requirement: Release Artifacts

Release SHALL define artifacts such as source archive, crates, binary,
conformance report, E2E report, OpenSpec report, coverage report, SBOM
placeholder, checksums, changelog, and release notes where applicable.

#### Scenario: Release candidate

Given release candidate is prepared

When artifacts are listed

Then required artifacts are present or explicitly marked not applicable.

---

### Requirement: Checksums

Published binary artifacts SHALL include checksums.

#### Scenario: Binary artifact

Given CLI binary is published

When release metadata is generated

Then binary checksum is included.

---

### Requirement: Changelog

Each release SHALL include a changelog.

#### Scenario: Changelog generated

Given `v0.1` is prepared

When changelog is inspected

Then added contracts, limitations, compatibility notes, and conformance status
are included.

---

### Requirement: Compatibility Policy

Release SHALL document compatibility status for Rust APIs, WIT contracts,
Runtime Inference API, Model Artifact metadata, Provider ABI, OpenSpec baseline,
CLI command surface, and conformance report format.

#### Scenario: Unstable Provider ABI

Given Provider ABI is not stable

When release notes are generated

Then Provider ABI is marked unstable or experimental.

---

### Requirement: Public API Safety

Release public APIs SHALL not expose raw Provider, Device, Kernel, tensor,
memory, KV cache, or model weight internals.

#### Scenario: API audit

Given public API audit runs

When handles are inspected

Then raw internal handles are absent.

---

### Requirement: Conformance Versioning

Conformance suites SHALL be explicitly versioned in release metadata.

#### Scenario: E2E report

Given E2E conformance report is attached

When metadata is inspected

Then suite version is present.

---

### Requirement: Release Gates

Stable release SHALL pass required release gates.

#### Scenario: E2E failure

Given E2E local conformance fails

When stable release is attempted

Then release is blocked.

---

### Requirement: Release Candidate Marking

Release candidates SHALL be clearly marked as pre-release.

#### Scenario: RC published

Given `v0.1.0-rc.1` is published

When release metadata is read

Then it is not marked stable.

---

### Requirement: Build Metadata Redaction

Build metadata SHALL not include secrets or local filesystem paths by default.

#### Scenario: Build metadata emitted

Given build metadata includes environment variables

When release artifact is generated

Then secrets and local paths are redacted.

---

### Requirement: Documentation Release

Release documentation SHALL state baseline scope and known limitations, and
SHOULD also include architecture, Runtime API, CLI boundary,
build/test/conformance instructions, feature flags, supported targets, and
roadmap.

#### Scenario: Documentation checked

Given release docs are published

When user reads them

Then baseline scope and limitations are clear.

---

### Requirement: Publishing Boundary

Publishing SHALL distinguish included baseline, experimental features, deferred
roadmap, and unsupported features.

#### Scenario: Roadmap feature

Given CUDA appears in roadmap

When release notes are generated

Then CUDA is not presented as included in `v0.1`.