# Expand Runtime Contract and Failure Test Coverage

## Why

Magnetar now has a clearer architecture:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Resource Affinity
    | Resolution Policy
    | planning
    | scheduling
    v
Provider
    |
    v
Device
```

Recent changes clarified:

- Backend and Plugin removal
- Compute Component boundary
- Runtime modularization
- Component Runtime boundary
- WASM Component execution
- Component Artifact trust
- inference-scoped Component authority
- Component distribution
- Provider health/readiness/pressure
- Provider loading and ABI policy

These contracts are only useful if they are enforced by tests.

The current project needs a dedicated test-coverage expansion focused on
architectural failure modes.

This change is different from CI gate setup.

The CI gate measures and protects quality.

This change adds the tests that should make those gates meaningful.

It covers:

- Runtime contract tests
- Provider failure tests
- Provider status tests
- Provider loading tests
- Resource Affinity tests
- Resolution Policy tests
- Compute boundary tests
- Component Runtime tests
- WASM trap and cancellation tests
- Artifact trust tests
- distribution validation tests
- no broad authority tests
- observability failure isolation tests

## What Changes

This change expands contract and failure-mode coverage across Magnetar Runtime.

The target is not only line coverage.

The target is architectural confidence.

Tests SHALL verify that Magnetar preserves its core invariants under normal and
failure conditions.

### Core Invariants

The test suite SHALL protect these invariants:

```text
Components import Capabilities, not Providers.
Runtime resolves Providers and Devices.
Providers own Devices.
Resource Affinity is authoritative.
Resolution Policy cannot override hard affinity.
Compute WIT does not expose Provider or Device targeting.
Component Artifacts must be validated and trusted before preparation.
Component authority is inference-scoped.
Distribution sources do not imply trust.
Dynamic Provider ABI is explicit and versioned.
Provider health, readiness, and pressure are distinct.
Scheduler consumes validated plans.
Observability must not alter execution correctness.
```

### Test Types

The repository SHALL include several categories of tests:

- unit tests
- integration tests
- contract tests
- failure injection tests
- fixture-based tests
- WIT validation tests
- artifact validation tests
- ABI-loading tests
- concurrency/cancellation tests
- observability isolation tests

Where property tests are useful, they MAY be introduced.

### Failure Injection

The Runtime SHALL include test utilities or mocks capable of simulating:

- Provider initialization failure
- Provider not-ready
- Provider degraded
- Provider saturated
- Provider draining
- stale Provider status
- Device unavailable
- Capability unavailable
- execution rejection
- execution failure
- cancellation unsupported
- cancellation failure
- Component trap
- Component interruption
- invalid Component artifact
- digest mismatch
- WIT mismatch
- forbidden authority
- revoked artifact
- invalid Provider ABI descriptor
- unsupported Provider ABI version
- observability sink failure

### Provider Tests

Provider tests SHALL cover:

- successful registration
- metadata validation
- Capability advertisement validation
- Device metadata validation
- status reporting
- readiness filtering
- pressure ranking
- saturation rejection
- draining behavior
- pinned resources during drain
- Provider refusal versus execution failure
- dynamic ABI loading failure cases

### Resource Affinity Tests

Resource Affinity tests SHALL prove that:

- Provider-bound resources remain bound
- Device-bound resources remain bound
- Resolution Policy cannot override mandatory affinity
- draining does not silently migrate resources
- explicit movement is required for placement change
- incompatible affinity returns structured errors
- cache/session/resource handles cannot forge affinity

### Compute Boundary Tests

Compute tests SHALL prove that:

- Component-facing Compute descriptors contain no Provider selector
- Component-facing Compute descriptors contain no Device selector
- placement intent remains portable
- host staging policy is respected
- diagnostics may report resolved Provider/Device as output metadata
- diagnostics do not become routing input
- Compute v1 and v2 compatibility rules are enforced where applicable

### Component Runtime Tests

Component Runtime tests SHALL cover:

- import validation
- unauthorized import failure
- no ambient WASI
- no broad filesystem/network/secrets/Git authority
- Link Plan construction
- Capability linking without Provider pinning
- multiple instance isolation
- trap normalization
- interruption/cancellation
- destruction behavior
- invocation after destruction failure

### Component Artifact and Distribution Tests

Artifact and distribution tests SHALL cover:

- digest validation
- manifest validation
- WIT consistency
- Runtime compatibility
- Capability compatibility
- trust policy
- revocation
- quarantine
- development mode
- local source
- client-provided source
- cache integrity
- Tachyon-source metadata not implying trust
- forbidden broad authority rejection

### Provider ABI Tests

Provider ABI tests SHALL cover:

- missing factory symbol
- unsupported ABI version
- malformed ABI descriptor
- missing required function
- invalid metadata
- invalid Capability advertisement
- invalid Device metadata
- invalid status report
- memory release protocol
- no Rust trait object dynamic ABI as stable path
- dynamic library loading policy

### Scheduler and Runtime Tests

Scheduler and Runtime tests SHALL cover:

- selected Provider becomes not-ready before submission
- selected Provider becomes saturated before submission
- retry versus fail behavior according to policy
- Scheduler does not independently select Provider
- cancellation propagation
- shutdown behavior
- active operation cleanup
- Provider drain lifecycle

### Observability Failure Isolation

Observability tests SHALL prove that:

- observability sink failure does not fail Compute execution
- observability exporter saturation follows policy
- observations do not grant authority
- observations do not become source of truth
- sensitive data is redacted

### Coverage Expectations

This change SHALL increase meaningful coverage of critical Runtime behavior.

Coverage goals SHALL focus on:

- Runtime orchestration
- Provider Registry
- Resolution
- Resource Affinity
- Compute planning
- Component Runtime
- Artifact validation
- Provider loading
- failure paths

A numeric ratchet MAY be applied by existing CI quality gates.

This change SHALL not lower existing coverage thresholds.

### Test Fixtures

The repository SHALL include deterministic fixtures where needed.

Fixtures MAY include:

- mock Providers
- mock Devices
- mock ComponentEngine
- WASM Component fixtures
- invalid artifact manifests
- invalid WIT fixtures
- ABI-shaped Provider loading fixtures
- fake distribution sources
- fake observability sinks

Fixtures SHALL avoid external network dependencies.

### CI Integration

All tests added by this change SHALL run in CI unless explicitly feature-gated.

Feature-gated tests SHALL have at least one CI job enabling the relevant feature.

Tests SHALL be deterministic and suitable for local development.

## Non-Goals

This change does not:

- introduce new architecture
- implement a new Provider ABI
- implement model inference
- implement Tachyon integration
- implement magnetar-cli
- implement broad agent tools
- define new Component authority
- increase coverage by testing trivial getters only
- rely on external network services
- require real GPUs in CI
- replace conformance suite work

## Impact

Magnetar gains a stronger safety net.

Future refactors and feature work become safer because architecture invariants
are executable.

The project moves from:

```text
contracts documented
```

to:

```text
contracts enforced by tests
```

This prepares the final recadrage change:

```text
define-provider-conformance-suite
```

which will focus specifically on validating Provider implementations against
Magnetar's Provider contract.