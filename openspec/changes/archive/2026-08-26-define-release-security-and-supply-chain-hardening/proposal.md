# Define Release Security And Supply Chain Hardening

## Why

Magnetar is an inference runtime that loads external artifacts and trusted native
Providers.

Even the first `v0.1` baseline must avoid unsafe release practices.

The release must make clear:

- what is trusted
- what is untrusted
- what is verified
- what is only metadata
- what is redacted
- what is explicitly unsupported

Without release security and supply-chain hardening, Magnetar risks shipping:

- unverifiable binaries
- undeclared dependencies
- unknown licenses
- accidental secrets
- unredacted diagnostics
- cache entries treated as trusted
- native Provider handles exposed through diagnostics
- model artifacts loaded without trust/integrity policy
- confusing security claims

This change defines the release security baseline for `v0.1`.

## What Changes

This change introduces release security and supply-chain hardening requirements.

It defines:

- dependency audit policy
- license audit policy
- SBOM policy
- checksum policy
- signature placeholder policy
- provenance policy
- secret scanning policy
- artifact integrity policy
- release build metadata policy
- Provider trust boundary
- Component artifact trust boundary
- Model artifact trust boundary
- source/cache trust boundary
- redaction release gates
- vulnerability handling policy
- security notes policy
- release-blocking security gates

## Security Scope For v0.1

The `v0.1` security scope SHALL cover the CPU-local baseline.

It SHALL include:

```text
Rust source
workspace dependencies
release binaries
release reports
OpenSpec baseline
WIT packages
Reference CPU Provider
fixture Model Artifacts
fixture Tokenizer Artifacts
Runtime Inference API
CLI boundary harness
E2E local conformance
```

It SHALL not claim hardened production security for:

```text
CUDA Provider
Metal Provider
OpenVINO Provider
QNN Provider
WebGPU Provider
server API
model hub downloads
remote registry authentication
production sandboxing
agent/tool runtime
large third-party model execution
```

## Dependency Audit

Release SHALL run dependency audit.

Dependency audit SHOULD check:

- known advisories
- yanked crates
- duplicate high-risk dependencies
- unexpected build scripts
- unexpected native dependencies
- license metadata
- dependency tree drift

A known critical advisory in a required release dependency SHALL block stable
release unless explicitly accepted with documented mitigation.

## License Audit

Release SHALL include license audit.

License audit SHOULD identify:

- dependency licenses
- incompatible licenses
- unknown licenses
- missing license metadata
- license exceptions
- bundled third-party notices

Unknown or incompatible licenses in required release dependencies SHALL block
stable release unless explicitly approved.

## SBOM

Release SHOULD produce an SBOM or SBOM placeholder.

SBOM SHOULD include:

- package name
- package version
- dependency list
- dependency versions
- license metadata
- source repository metadata where available
- build target metadata
- feature flags where feasible

If full SBOM generation is not implemented for `v0.1`, release notes SHALL state
the limitation.

## Checksums

Release artifacts SHOULD include checksums.

Checksums SHOULD cover:

- source archive
- binaries
- conformance report
- E2E report
- OpenSpec validation report
- coverage report
- SBOM where present

Checksums SHALL be generated from final release artifacts.

## Signatures

Release MAY include cryptographic signatures.

If signatures are not implemented for `v0.1`, release notes SHALL state that
checksums are provided but signatures are not yet available.

Signature absence SHALL not be hidden.

## Provenance

Release SHOULD include provenance metadata.

Provenance metadata MAY include:

- source commit
- release tag
- CI run identifier
- build target
- build profile
- rustc version
- dependency lockfile digest
- OpenSpec baseline digest
- WIT package digest
- conformance report digest

Provenance SHALL not include secrets or local developer paths by default.

## Reproducibility

Release SHOULD document reproducibility status.

If builds are not fully reproducible, release notes SHALL state limitations.

The release process SHOULD preserve enough metadata to allow independent
verification where feasible.

## Lockfile Policy

Release SHALL use a checked-in dependency lockfile where appropriate.

The lockfile digest SHOULD be included in release provenance.

Unreviewed lockfile drift SHOULD block release candidates.

## Build Script Policy

Release SHALL review build scripts in required dependencies where feasible.

Unexpected build scripts in new dependencies SHOULD be flagged.

Native code build steps SHOULD be documented.

## Secret Scanning

Release SHALL run secret scanning on source and release artifacts.

Secret scanning SHOULD check:

- source files
- generated docs
- release notes
- conformance reports
- E2E reports
- logs where included
- build metadata
- packaged artifacts

Detected secrets SHALL block stable release.

## Artifact Integrity

Release artifacts SHALL be produced from validated source state.

Artifact integrity SHOULD include:

- source state clean or CI-controlled
- release tag matches source
- OpenSpec validation report matches release baseline
- conformance reports match release commit
- checksums match final artifacts

## Redaction Gates

Release SHALL include redaction gates.

Redaction gates SHALL verify diagnostics, reports, logs, and observations do not
include by default:

- raw prompts
- secrets
- credentials
- raw file contents
- raw model weights
- raw tensor values
- raw KV cache contents
- raw Provider handles
- raw Device handles
- raw Kernel handles
- raw memory pointers
- local filesystem paths where not explicitly allowed
- raw cache paths by default

## Provider Trust Boundary

Providers are trusted native code.

Release security notes SHALL state this clearly.

Reference CPU Provider may be included in `v0.1`.

Dynamic Provider loading, if present, SHALL be disabled, experimental, or
clearly marked unstable unless security reviewed.

Provider registration SHALL not imply Provider trust beyond configured policy.

## Native Handle Boundary

Release public APIs, diagnostics, and reports SHALL not expose native Provider,
Device, Kernel, tensor, or memory handles.

### Native handle examples

```text
CUDA device pointer
CUDA stream
Metal buffer
Metal command queue
OpenVINO compiled model pointer
QNN graph handle
raw CPU allocation pointer
```

These SHALL remain internal.

## Component Artifact Trust Boundary

Component Artifacts SHALL be validated before execution.

Unsigned Component Artifacts SHALL be denied in production policy unless
explicitly allowed for development/test.

Release notes SHALL describe Component trust status for `v0.1`.

## Model Artifact Trust Boundary

Model Artifacts SHALL pass trust and integrity validation before loading.

Fixture artifacts SHALL have explicit test trust policy.

Recognized format SHALL not imply trust.

Cache presence SHALL not imply trust.

## Source Cache Trust Boundary

Source/cache security SHALL preserve:

```text
cache hit != trusted
source kind != trusted
alias != trusted
local file != trusted
fixture != trusted unless test policy
```

Cache metadata MAY store trust status, but current policy SHALL decide whether
artifact can load.

## CLI Boundary Security

`magnetar-cli` may have broader authority than Runtime.

Release security SHALL verify CLI authority is not delegated to Runtime.

CLI-side secrets, filesystem, Git, network, shell, and tool authority SHALL not
become Runtime ambient authority.

## Runtime Inference API Security

Runtime Inference API SHALL remain inference-only.

It SHALL not gain:

- arbitrary filesystem authority
- arbitrary network authority
- secret authority
- shell/process authority
- Git authority
- tool execution authority
- agent orchestration authority

## Unsafe Code Policy

Release SHALL document unsafe Rust policy.

If unsafe code exists, it SHOULD be reviewed and justified.

Unsafe code in required baseline should be minimized.

Unsafe code MAY be denied in release gates unless explicitly allowed.

## Dependency Feature Policy

Release SHALL review enabled dependency features.

Features that enable networking, filesystem expansion, native plugins, dynamic
loading, or broad OS capabilities SHOULD be explicit.

Unexpected capability-expanding features SHOULD block release until reviewed.

## Vulnerability Handling

Release SHALL define vulnerability handling policy.

Policy SHOULD include:

- advisory severity handling
- release blocking criteria
- documented mitigation
- exception approval
- follow-up tracking
- patch release expectation

## Security Notes

Release SHALL include security notes.

Security notes SHOULD include:

- v0.1 threat model
- trusted native Provider model
- no raw handle policy
- default redaction
- source/cache trust boundary
- model artifact trust boundary
- Component artifact trust boundary
- unsupported security features
- known risks
- reporting process placeholder

## Release Blocking Criteria

Stable release SHALL be blocked by:

- detected secrets
- critical dependency advisory without mitigation
- incompatible required dependency license
- failing redaction gate
- raw internal handle exposure
- failed trust/integrity validation in required fixtures
- E2E conformance bypass
- OpenSpec validation failure
- release artifact checksum mismatch
- undocumented security exception

## Security Exceptions

Security exceptions SHALL be documented.

An exception SHOULD include:

- issue
- affected component
- severity
- rationale
- mitigation
- owner
- expiration or follow-up
- release note entry

Undocumented exceptions SHALL not be allowed for stable release.

## Observability

Security hardening SHOULD emit or record release observations for:

- dependency audit completed
- license audit completed
- SBOM generated
- checksum generated
- secret scan completed
- redaction gate completed
- provenance generated
- security exception recorded
- release blocked
- release security passed

Observability SHALL not expose secrets, credentials, raw prompts, raw weights,
raw tensors, raw cache contents, handles, memory pointers, or local paths by
default.

## Non-Goals

This change does not:

- implement full cryptographic signing
- implement SLSA compliance
- implement production sandboxing
- implement remote registry authentication
- implement model hub security
- implement server authentication
- make native Providers untrusted sandboxed plugins
- guarantee all future Providers are secure
- define legal approval process
- replace separate release conformance gates

## Impact

Magnetar gains a release security baseline.

The first release can be described honestly:

```text
v0.1 = CPU-local inference baseline
     + explicit trust boundaries
     + dependency/license/secret/redaction gates
     + checksums/provenance where available
     + documented limitations
```

without overstating production hardening.