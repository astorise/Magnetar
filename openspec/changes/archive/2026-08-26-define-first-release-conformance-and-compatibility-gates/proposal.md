# Define First Release Conformance And Compatibility Gates

## Why

Magnetar now has release packaging, versioning, and security hardening policy.

Before a first stable `v0.1` release can be cut, the project needs a precise
definition of the required release gates.

A release must not be accepted simply because the repository builds.

It must demonstrate that the implemented baseline respects the contracts that
make Magnetar a safe inference runtime:

- inference-only Runtime
- Runtime Inference API boundary
- CLI/Runtime authority separation
- Component/Provider separation
- Tensor and Memory ownership
- Kernel Registry and Dispatch
- Reference CPU correctness baseline
- Model Artifact and Model Loading validation
- Qwen-like fixture path
- E2E local inference
- redaction
- release security
- documented compatibility

This change defines the first release conformance and compatibility gates.

## What Changes

This change introduces release-blocking validation gates for `v0.1`.

It defines:

- required conformance gates
- required compatibility gates
- required security gates
- required redaction gates
- required reporting artifacts
- release-blocking failures
- allowed skips
- documented exceptions
- compatibility matrix
- release candidate validation process

## Release Gate Principle

A `v0.1` stable release SHALL pass all required gates.

Required gates SHALL be deterministic where feasible.

Optional gates MAY be skipped only when:

- the feature is explicitly out of `v0.1` scope
- the skip is reported
- the skip does not hide a failure in baseline functionality

A skipped required baseline gate SHALL block release.

## Required Gate Categories

The `v0.1` release SHOULD include these gate categories:

```text
source/build gates
OpenSpec gates
WIT gates
Rust API gates
Runtime contract gates
Provider gates
Operator gates
Tensor/Memory gates
Model gates
Inference API gates
CLI boundary gates
E2E gates
observability/redaction gates
security/supply-chain gates
compatibility gates
documentation gates
release artifact gates
```

## Source And Build Gates

Release SHALL pass source and build gates.

Required gates SHOULD include:

- repository state clean or CI-controlled
- release tag matches source
- lockfile reviewed
- dependency tree resolved
- formatting
- compilation
- Clippy or equivalent linting
- unit tests
- doc tests where applicable
- feature matrix checks
- supported target checks

## OpenSpec Gates

Release SHALL pass OpenSpec validation.

OpenSpec gate SHOULD verify:

- accepted changes are listed
- pending changes are not accidentally included
- release baseline is declared
- release checklist references correct changes
- no semantic change is made after freeze
- no hidden scope expansion exists
- specs are internally consistent

OpenSpec validation failure SHALL block release.

## WIT Gates

Release SHALL pass WIT validation for included WIT packages.

WIT gate SHOULD verify:

- WIT package versions are declared
- breaking changes have major version bump
- supported WIT version matrix is documented
- generated bindings compile where applicable
- WIT world boundaries match Runtime contracts
- Provider/Device targeting is not exposed to Components
- no raw handles are exposed through WIT

WIT validation failure SHALL block release.

## Rust Public API Gates

Release SHALL audit Rust public APIs.

Rust API gate SHOULD verify:

- exported types are intentional
- public IDs are opaque
- internal modules are not accidentally exported
- experimental APIs are feature-gated
- unstable APIs are marked
- no raw Provider handles are exposed
- no raw Device handles are exposed
- no raw Kernel handles are exposed
- no raw tensor pointers are exposed
- no raw memory pointers are exposed
- no raw KV cache contents are exposed
- no raw model weights are exposed

Public API safety failure SHALL block release.

## Runtime Contract Gates

Runtime contract gates SHALL verify the Runtime remains inference-only.

They SHOULD validate:

- no arbitrary filesystem authority
- no arbitrary network authority
- no secret authority
- no shell/process execution
- no Git execution
- no tool execution
- no agent orchestration
- Runtime calls stay within inference contracts
- Runtime rejects boundary violations

Runtime boundary failure SHALL block release.

## Runtime Inference API Gates

Runtime Inference API gates SHALL validate:

- model resolution request behavior
- model loading request behavior
- session creation
- session close
- one-shot inference if included
- tokenization API
- generation API
- streaming API
- cancellation API
- diagnostics API
- usage reporting
- structured error preservation
- redaction by default
- no internal handle exposure

## Provider Gates

Provider gates SHALL validate Provider contracts.

For `v0.1`, Reference CPU Provider is required.

Required Provider gates SHOULD include:

- Provider identity valid
- Provider status snapshot valid
- Device metadata valid
- Kernel advertisements valid
- readiness/health/pressure separated
- Provider does not expose native handles
- Provider execution goes through Kernel Dispatch
- Provider errors are structured

Non-Reference Providers MAY be skipped if out of scope.

## Reference CPU Gates

Reference CPU gates SHALL validate correctness baseline behavior.

They SHOULD verify:

- Reference CPU Provider registers correctly
- CPU Device metadata exists
- required-now Kernels are advertised
- host contiguous f32 path works
- deterministic fixtures pass
- correctness is prioritized over performance
- Reference CPU remains baseline despite optimized roadmap

Failure of Reference CPU gates SHALL block release.

## Operator First Scope Gates

Operator gates SHALL validate first operator scope.

They SHOULD verify required-now coverage for:

```text
embedding
matmul
rmsnorm
rope
attention
softmax
silu
add
mul
residual-add
dtype-conversion where included
layout-conversion where included
```

Missing required-now operator coverage SHALL block release unless explicitly
deferred by release checklist and not required by E2E path.

## Tensor And Memory Gates

Tensor/Memory gates SHALL validate:

- TensorDescriptor behavior
- TensorResource lifecycle
- TensorLayout metadata
- dtype metadata
- shape metadata
- host contiguous layout
- Memory Manager allocation tracking
- size accounting
- readiness state
- cleanup
- no raw pointer exposure
- cache storage is distinct from memory residency

## Kernel Registry And Dispatch Gates

Kernel Registry gates SHALL validate:

- Kernel advertisement validation
- Kernel candidate lookup
- candidate filtering
- Resource Affinity validation
- Memory Manager feasibility
- Provider/Device readiness checks
- dispatch revalidation
- structured missing-kernel errors
- no direct Provider bypass
- no silent fallback

## Model Artifact And Loading Gates

Model gates SHALL validate:

- fixture Model Artifact manifest
- artifact identity
- trust validation
- integrity validation
- tensor inventory validation
- tokenizer compatibility
- Model Loading lifecycle
- Model Instance readiness
- unload cleanup
- cache hit does not imply trust
- recognized format does not imply trust

## Qwen Baseline Gates

Qwen-like baseline gates SHALL validate:

- Qwen config validation
- tensor inventory validation
- tokenizer compatibility metadata
- target module metadata
- prefill graph production
- decode graph production
- required operator use
- no QwenProvider
- no direct Provider/Kernel access

## Generation And Sampling Gates

Generation/Sampling gates SHALL validate:

- prefill orchestration
- decode orchestration
- greedy sampling
- stop conditions
- max new tokens
- max total tokens
- finish reason
- usage accounting
- cancellation checkpoints
- streaming events
- structured generation errors

## CLI Boundary Gates

CLI boundary gates SHALL validate:

- CLI calls Runtime Inference API
- CLI sends explicit prompt/context
- Runtime receives no ambient filesystem authority
- Runtime receives no ambient Git authority
- Runtime receives no ambient network authority
- Runtime receives no ambient secret authority
- Runtime receives no ambient tool authority
- Runtime receives no ambient shell/process authority
- Runtime structured errors are preserved
- CLI diagnostics are redacted

## E2E Local Inference Gates

E2E local inference gates SHALL validate full baseline path.

Required E2E path:

```text
Runtime Inference API
  -> model resolution
  -> model loading
  -> Model Instance
  -> session
  -> tokenizer
  -> generation
  -> graph production
  -> operator validation
  -> Kernel Registry
  -> Reference CPU dispatch
  -> streaming/result
  -> cleanup
```

E2E success path SHALL be CPU-only, local, deterministic, and no-shortcut.

E2E failure SHALL block release.

## Observability And Redaction Gates

Observability gates SHALL validate redaction by default.

They SHALL verify no default diagnostics, reports, logs, or observations expose:

- raw prompts
- secrets
- credentials
- raw file contents
- raw model weights
- raw tensor values
- raw KV cache contents
- Provider handles
- Device handles
- Kernel handles
- memory pointers
- raw cache paths by default

Failure SHALL block release.

## Security And Supply-Chain Gates

Security gates SHALL include release security hardening requirements.

They SHOULD include:

- dependency audit
- license audit
- secret scanning
- SBOM or documented limitation
- checksums
- provenance
- lockfile review
- unsafe code review
- artifact integrity
- security notes
- documented exceptions

Security-blocking failures SHALL block release.

## Compatibility Matrix

Release SHALL publish a compatibility matrix.

The matrix SHOULD include status for:

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

Each area SHALL be marked as one of:

```text
stable-for-v0.1-baseline
preview
experimental
unstable
deferred
unsupported
```

## Allowed Skips

A gate MAY be skipped only if:

- it covers a feature outside `v0.1` scope
- the skip is listed in the release report
- the skip reason is structured
- the skip does not hide a baseline failure

Examples of allowed skips:

```text
CUDA Provider conformance
Metal Provider conformance
OpenVINO Provider conformance
QNN Provider conformance
WebGPU Provider conformance
server API conformance
model hub source conformance
production model format conformance
```

## Disallowed Skips

The following SHALL NOT be skipped for stable `v0.1`:

- OpenSpec validation
- WIT validation for included WIT
- formatting/check/lint
- required unit/contract tests
- Reference CPU conformance
- Runtime Inference API baseline tests
- CLI boundary tests
- E2E local inference conformance
- redaction gates
- release security gates
- artifact integrity gates
- release documentation checklist

## Exception Policy

Release exceptions SHALL be documented.

Each exception SHOULD include:

- gate name
- failure or deviation
- severity
- affected component
- rationale
- mitigation
- owner
- expiration/follow-up
- release note entry

Undocumented exceptions SHALL block release.

## Release Reports

Release SHALL produce machine-readable and human-readable reports.

Reports SHOULD include:

- gate name
- gate category
- status
- duration
- version
- target
- feature set
- skip reason
- failure reason
- exception reference
- artifact checksum
- redaction status

Reports SHALL be redacted by default.

## Release Candidate Validation

Release candidates SHALL run the same required gates as stable release.

A release candidate MAY contain known failures only if:

- failures are documented
- release is clearly marked pre-release
- stable publication is blocked until resolved or exception accepted

## Stable Release Cutover

Stable release SHALL occur only after:

- all required gates pass
- allowed skips are documented
- exceptions are documented
- release reports are generated
- compatibility matrix is complete
- release notes are complete
- release artifacts are checksummed
- security notes are complete

## Non-Goals

This change does not:

- implement conformance suites
- implement release automation
- define final CI provider
- define benchmark thresholds
- require GPU Providers
- require server API
- require model hub support
- require production large model support
- make all APIs 1.0-stable

## Impact

Magnetar gains concrete release gates.

`v0.1` becomes releasable only when the CPU-local baseline is proven by:

```text
contracts
  + conformance
  + compatibility
  + redaction
  + security
  + release reports
```