# Define v0.1 Release Cutover Checklist

## Why

Magnetar now has:

- architecture contracts
- implementation baseline plan
- post-baseline roadmaps
- release packaging policy
- release security policy
- release conformance gates

The final step before stabilizing the first release is a concrete cutover
checklist.

This checklist defines the operational sequence required to move from release
candidate to stable `v0.1`.

It prevents accidental release with:

- unfrozen OpenSpec changes
- incomplete conformance reports
- missing compatibility matrix
- undocumented skips
- undocumented exceptions
- missing checksums
- unclear security notes
- unstable APIs presented as stable
- roadmap features presented as included

## What Changes

This change defines the `v0.1` release cutover checklist.

It covers:

- release readiness
- OpenSpec freeze
- scope confirmation
- version confirmation
- feature flag confirmation
- conformance execution
- compatibility matrix completion
- security verification
- release artifact generation
- changelog and release notes
- tagging
- publication
- post-release verification
- rollback notes
- post-v0.1 roadmap handoff

## Cutover Principle

A stable `v0.1` release SHALL be cut only after required gates pass and release
metadata is complete.

The release manager SHALL be able to answer:

```text
What is included?
What is excluded?
Which contracts are stable-for-v0.1?
Which APIs are preview/experimental?
Which gates passed?
Which gates were skipped and why?
Which exceptions exist?
Which artifacts were published?
How can users verify them?
```

## Release Readiness

Before cutover, release readiness SHALL confirm:

- release branch or commit selected
- version selected
- OpenSpec baseline selected
- release scope selected
- release gates selected
- release artifacts selected
- release notes draft exists
- compatibility matrix draft exists
- security notes draft exists

## OpenSpec Freeze

Before stable release, included OpenSpec changes SHALL be frozen.

Freeze SHALL confirm:

- accepted changes list is final
- pending changes are not included
- no semantic change after freeze
- WIT-breaking changes have version bumps
- release checklist references correct changes
- roadmap items remain deferred unless explicitly included

## Scope Confirmation

`v0.1` scope SHALL be confirmed as CPU-local baseline.

Included baseline SHOULD be:

```text
Runtime Inference API baseline
Model Loading baseline
Model Instance baseline
Tokenizer fixture path
Qwen-like baseline fixture path
Generation and Sampling baseline
Tensor and Memory baseline
Operator first scope
Kernel Registry and Dispatch
Reference CPU Provider
CLI boundary harness
E2E local inference conformance
release reports
```

Excluded or deferred items SHOULD include:

```text
CUDA Provider
Metal Provider
OpenVINO Provider
QNN Provider
WebGPU Provider
production large model support
model hub download UX
server API implementation
agent/tool runtime
production CLI UX
full quantized inference
advanced attention kernels
```

## Version Confirmation

Cutover SHALL confirm:

- release version
- crate versions
- binary version
- WIT package versions
- conformance suite versions
- OpenSpec baseline version
- release candidate lineage where applicable

## Feature Flag Confirmation

Cutover SHALL confirm default feature flags.

Default `v0.1` features SHOULD be CPU-local baseline only.

Experimental features SHALL be disabled by default.

Provider-specific features outside Reference CPU SHALL be disabled, absent, or
marked experimental.

## Compatibility Matrix Completion

Before stable release, compatibility matrix SHALL be complete.

The matrix SHALL include:

- Rust public API
- Runtime Inference API
- WIT packages
- Provider ABI
- Model Artifact metadata
- Tokenizer Artifact metadata
- Adapter Artifact metadata
- CLI command surface
- conformance report format
- OpenSpec baseline
- supported targets
- feature flags

Each item SHALL be marked:

```text
stable-for-v0.1-baseline
preview
experimental
unstable
deferred
unsupported
```

## Required Gate Execution

Cutover SHALL execute required gates from release conformance policy.

Required gates include:

- source/build gates
- OpenSpec gates
- WIT gates
- Rust public API gates
- Runtime contract gates
- Runtime Inference API gates
- Provider gates
- Reference CPU gates
- Operator first scope gates
- Tensor/Memory gates
- Kernel Registry/Dispatch gates
- Model Loading gates
- Qwen baseline gates
- Generation/Sampling gates
- CLI boundary gates
- E2E local inference gates
- observability/redaction gates
- security/supply-chain gates
- release artifact gates

## Skip Review

Cutover SHALL review all skipped gates.

A skip is allowed only when:

- feature is outside `v0.1` scope
- skip reason is documented
- skip does not hide baseline failure
- release report includes skip metadata

Disallowed skips SHALL block release.

## Exception Review

Cutover SHALL review all exceptions.

Each exception SHALL include:

- gate name
- issue
- severity
- affected component
- rationale
- mitigation
- owner
- expiration or follow-up
- release note entry

Undocumented exceptions SHALL block release.

## Security Verification

Cutover SHALL confirm:

- dependency audit completed
- license audit completed
- secret scan completed
- redaction gates passed
- native handle exposure checks passed
- trust/integrity checks passed
- artifact integrity checks passed
- checksums generated
- SBOM generated or limitation documented
- signature status documented
- security notes complete

## Artifact Generation

Cutover SHALL generate final release artifacts.

Artifacts MAY include:

- source archive
- Rust crate artifacts
- CLI binary
- OpenSpec validation report
- WIT validation report
- conformance report
- E2E local inference report
- release security report
- compatibility matrix
- coverage report
- SBOM or SBOM limitation
- checksums
- changelog
- release notes

If an artifact is not applicable, it SHALL be marked explicitly.

## Artifact Verification

Cutover SHALL verify final artifacts.

Verification SHOULD include:

- checksums generated from final artifacts
- reports match release commit
- release notes match compatibility matrix
- OpenSpec baseline matches tag
- conformance versions match reports
- no local paths or secrets in artifacts
- artifact names include version where appropriate

## Changelog Completion

Cutover SHALL finalize changelog.

Changelog SHOULD include:

- added contracts
- changed contracts
- removed/deprecated contracts
- release scope
- known limitations
- compatibility status
- security notes
- conformance status
- deferred roadmap items

## Release Notes Completion

Cutover SHALL finalize release notes.

Release notes SHOULD answer:

```text
What is v0.1?
What can users run?
What is stable-for-v0.1?
What is preview?
What is experimental?
What is deferred?
What is unsupported?
How were artifacts verified?
What security limitations exist?
How to run conformance?
```

## Tagging

Stable release tag SHALL be created only after required gates pass.

Tagging SHOULD include:

- semantic version tag
- release commit
- OpenSpec baseline reference
- artifact checksum reference
- release notes reference

If using signed tags, signature status SHALL be documented.

## Publication

Publication MAY include:

- source release
- crates
- binaries
- reports
- documentation
- release notes

Publication SHALL not present deferred roadmap features as included.

Publication SHALL not present preview/experimental APIs as stable.

## Post-Publication Verification

After publication, release verification SHOULD confirm:

- published artifacts match checksums
- release notes visible
- reports accessible
- version command reports expected version
- documentation links valid
- compatibility matrix visible
- security notes visible
- deferred roadmap clearly separated

## Rollback And Retraction Notes

Release process SHOULD define rollback or retraction notes.

If a release is found invalid after publication, process SHOULD describe:

- how to mark release as withdrawn
- how to publish advisory
- how to cut patch release
- how to preserve audit trail
- how to update release notes

## Post-v0.1 Handoff

After `v0.1`, roadmap handoff SHOULD identify next work.

Post-v0.1 candidates MAY include:

- implementation hardening
- optimized CPU Provider
- model format support
- source/cache implementation
- server API implementation
- production CLI UX
- CUDA/Metal/OpenVINO/QNN/WebGPU exploration
- quantized inference
- advanced attention

Post-v0.1 items SHALL remain separate from `v0.1` release claims.

## Final Release Statement

The stable release statement SHOULD be:

```text
Magnetar v0.1 is a CPU-local inference runtime baseline.
It validates the first end-to-end inference path through Runtime Inference API,
Reference CPU Provider, and E2E local conformance.
Post-baseline roadmap features are not included unless explicitly marked.
```

## Non-Goals

This change does not:

- implement release automation
- publish the release
- define registry credentials
- define final hosting provider
- define legal approval process
- guarantee production security
- include GPU Providers
- include server API implementation
- include model hub downloads
- include agent/tool runtime

## Impact

Magnetar gains a final operational release cutover process.

After this change, the OpenSpec release stabilization sequence is complete.

A first release can proceed only by following:

```text
freeze
  -> gates
  -> reports
  -> artifacts
  -> tag
  -> publish
  -> verify
```