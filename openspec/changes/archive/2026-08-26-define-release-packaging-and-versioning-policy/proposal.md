# Define Release Packaging And Versioning Policy

## Why

Magnetar now has a baseline architecture and post-baseline roadmaps for:

- Runtime inference
- CLI boundary
- E2E local conformance
- Provider roadmap
- model format roadmap
- model source/cache roadmap
- server API roadmap

Before a first release, Magnetar needs explicit packaging and versioning rules.

Without them, the project risks publishing unstable contracts as stable,
breaking WIT compatibility accidentally, mixing future roadmap features into the
baseline release, or shipping artifacts that cannot be reproduced or validated.

This change defines the release packaging and versioning policy for `v0.1`.

## What Changes

This change defines:

- release versioning policy
- crate/package versioning
- binary versioning
- WIT package versioning
- OpenSpec change freeze policy
- feature flag policy
- experimental feature policy
- release artifact policy
- changelog policy
- compatibility policy
- conformance gate policy
- release CI gate policy
- publishing boundaries

The exact package manager, registry, and release automation details are
implementation-defined.

## Release Scope

The first release SHOULD be named `v0.1`.

The `v0.1` release SHOULD include only the CPU-local baseline:

```text
Runtime Inference API
Model Loading baseline
Model Instance baseline
Tokenizer fixture path
Qwen-like baseline fixture path
Generation + Sampling baseline
Tensor + Memory baseline
Operator first scope
Kernel Registry + Dispatch
Reference CPU Provider
CLI boundary harness
E2E local conformance
```

The release SHALL NOT require:

```text
CUDA
Metal
OpenVINO
QNN
WebGPU
production Qwen model support
large model execution
model hub downloads
server API implementation
agent/tool runtime
production CLI UX
```

## Versioning Policy

Magnetar SHALL use explicit semantic versioning for release artifacts.

Before `1.0`, breaking changes MAY occur, but they SHALL be documented.

Within `0.x`, minor versions MAY include breaking API changes if clearly
documented.

Patch versions SHOULD avoid breaking changes.

## Crate Versioning

Each publishable Rust crate SHALL have a declared version.

Crates MAY share the same workspace version for the first release.

If independent crate versions are used, dependency compatibility SHALL be
documented.

## Binary Versioning

Binaries such as `magnetar` SHALL report version information.

Version output SHOULD include:

- binary version
- runtime crate version
- OpenSpec baseline version
- WIT contract versions
- enabled feature flags
- build profile
- commit hash where available
- conformance suite version where available

## WIT Versioning

WIT packages SHALL use explicit versions.

Breaking WIT changes SHALL require a new major WIT version.

Non-breaking additive changes MAY use minor versions.

Documentation-only changes MAY use patch versions.

A release SHALL document which WIT package versions are supported.

## OpenSpec Baseline Version

A release SHALL declare the OpenSpec baseline it implements.

The baseline metadata SHOULD include:

- list of accepted changes
- OpenSpec validation status
- compatibility notes
- deferred changes
- conformance status
- release tag

## Change Freeze

Before cutting a release, OpenSpec changes included in that release SHALL be
frozen.

Frozen means:

- no semantic contract changes without new change proposal
- no WIT breaking change without version bump
- no release gate changes without explicit release checklist update
- no hidden scope expansion

Documentation clarifications MAY be allowed if they do not change semantics.

## Feature Flag Policy

Release feature flags SHALL be explicit.

Feature flags SHOULD distinguish:

```text
baseline
experimental
provider-specific
platform-specific
test-only
conformance-only
```

Baseline release features SHOULD be minimal and CPU-only.

Experimental features SHALL not be enabled by default.

## Experimental Features

Experimental features MAY exist but SHALL be clearly marked.

Experimental features SHALL not be required for release conformance.

Experimental APIs SHALL not be documented as stable.

Experimental features SHOULD be excluded from default build unless explicitly
enabled.

## Provider Feature Flags

Provider-specific feature flags MAY include:

```text
reference-cpu-provider
optimized-cpu-provider
cuda-provider
metal-provider
openvino-provider
qnn-provider
webgpu-provider
```

For `v0.1`, only Reference CPU Provider SHOULD be required.

Other Provider flags SHALL be absent, disabled, or explicitly experimental.

## Component Engine Feature Flags

Component engine feature flags MAY include:

```text
wasmtime-component-engine
web-component-engine
test-component-engine
```

Native Wasmtime support SHALL remain feature-gated if not universally supported.

Browser builds SHALL not require Wasmtime.

## Platform Targets

The release SHOULD define supported platform targets.

The first release SHOULD support native CPU targets required by CI.

Browser or wasm32 support MAY be check-only if not fully implemented.

Unsupported targets SHALL be documented.

## Release Artifacts

Release artifacts MAY include:

- source archive
- Rust crates
- CLI binary
- conformance report
- E2E report
- OpenSpec validation report
- coverage report
- SBOM placeholder
- checksums
- changelog
- release notes

All release artifacts SHOULD be reproducible where feasible.

## Checksums

Published release artifacts SHOULD have checksums.

Checksums SHOULD cover:

- source archives
- binaries
- generated reports
- packaged artifacts where applicable

Checksums SHALL not replace trust or signature policy.

## Changelog Policy

Each release SHALL include a changelog.

Changelog SHOULD include:

- added contracts
- changed contracts
- removed/deprecated contracts
- fixed issues
- known limitations
- conformance status
- compatibility notes
- security notes where applicable

## Compatibility Policy

Release compatibility SHALL be explicit.

Compatibility dimensions SHOULD include:

- Rust public API
- WIT contracts
- Runtime Inference API
- Model Artifact metadata
- Provider ABI
- OpenSpec baseline
- CLI command surface
- conformance report format

If an area is unstable, the release SHALL mark it unstable.

## Public API Stability

`v0.1` SHALL distinguish stable baseline APIs from internal or experimental APIs.

Public APIs SHALL not expose:

- raw Provider handles
- raw Device handles
- raw Kernel handles
- raw tensor pointers
- raw memory pointers
- raw KV cache contents
- raw model weights

## Conformance Versioning

Conformance suites SHALL have explicit versions.

The release SHALL report:

- provider conformance suite version
- first operator scope conformance version
- Qwen baseline conformance version
- Runtime Inference API conformance version
- CLI boundary conformance version
- E2E local conformance version

## Release Gates

A release SHALL pass required gates before publication.

Required gates SHOULD include:

```text
formatting
cargo check
clippy
unit tests
contract tests
OpenSpec validation
WIT validation
Reference CPU conformance
Operator first scope conformance
Runtime Inference API tests
CLI boundary tests
E2E local conformance
coverage gate
redaction checks
no raw handle exposure checks
```

## Release Failure Policy

If required release gates fail, release SHALL not be published as stable.

A failed release candidate MAY be tagged as pre-release only if clearly marked.

## Release Candidate Policy

Release candidates MAY be used.

A release candidate SHOULD include:

- release version candidate tag
- frozen OpenSpec baseline
- conformance report
- known failures
- release notes draft

A release candidate SHALL not be presented as stable.

## Pre-release Tags

Allowed pre-release tags MAY include:

```text
-alpha
-beta
-rc.N
```

Pre-release tags SHALL not be confused with stable tags.

## Build Metadata

Build metadata MAY include:

- commit hash
- build timestamp
- target triple
- enabled features
- CI run identifier
- profile
- rustc version

Build metadata SHALL not include secrets or local filesystem paths by default.

## Documentation Release

Release documentation SHOULD include:

- architecture overview
- Runtime Inference API overview
- CLI boundary overview
- build instructions
- test instructions
- conformance instructions
- feature flags
- supported targets
- known limitations
- post-baseline roadmap

## Security Notes

Release packaging SHALL include security notes where applicable.

Security notes SHOULD identify:

- sandbox assumptions
- Provider trust model
- no raw handle policy
- default redaction
- source/cache trust boundary
- unsupported security features
- known risks

Detailed security hardening is defined by a separate release security change.

## Publishing Boundary

Publishing SHALL not imply production readiness for all roadmap features.

The release SHALL clearly distinguish:

```text
included baseline
experimental feature
deferred roadmap
unsupported feature
```

## Non-Goals

This change does not:

- implement release automation
- define final crate names
- publish packages
- define full security hardening
- define supply-chain signing
- finalize production CLI UX
- finalize server API
- stabilize CUDA/Metal/OpenVINO/QNN/WebGPU
- make all APIs 1.0-stable

## Impact

Magnetar gains release discipline.

The first release can be cut as a constrained and verifiable baseline:

```text
v0.1 = CPU-local inference baseline + conformance reports + explicit limitations
```

without accidentally promising the full post-baseline roadmap.