# Test Coverage Inventory

This inventory supports the OpenSpec change
`expand-runtime-contract-and-failure-test-coverage`.

## Existing Coverage

- Unit tests live primarily in `magnetar-runtime/src/tests.rs`.
- Feature-gated Wasmtime Component Engine tests live in
  `magnetar-runtime/src/component_wasmtime.rs`.
- Observability exporter unit tests live in
  `magnetar-runtime/src/observability/exporter.rs`.
- Component fixtures live under `magnetar-runtime/fixtures/components`.
- WIT contracts live under `magnetar-runtime/wit`.
- CI runs formatting, Clippy, workspace tests, Wasmtime feature tests, WIT
  validation, OpenSpec validation, and coverage ratchet checks.

## Unit Test Inventory

The main runtime unit suite currently covers:

- Provider registration, duplicate rejection, failed initialization isolation,
  shutdown, dynamic loading policy, Provider ABI descriptor validation, and
  Provider execution API behavior.
- Device registration and Provider ownership checks.
- Capability registration, compatibility, dependency validation, compute
  capability identity, and Compute v1/v2 compatibility boundaries.
- Compute WIT surface checks, including absence of direct Provider, Device, and
  AffinityGroup targeting fields.
- Compute graph validation, operation schema validation, dtype/layout/provider
  support, data movement validation, and memory planning.
- Execution planning, validated scheduler admission, FIFO scheduling,
  cancellation, completion, and Provider status checks before submission.
- Resolution policy behavior, Provider status rejection diagnostics, readiness,
  stale status handling, pressure-related policy behavior, and phase-aware
  restart rejection.
- Resource Affinity preservation and conflict reporting for Provider, Device,
  Capability, Artifact, execution context, and groups.
- Component Runtime import authorization, link-plan behavior, no ambient broad
  authority, instance isolation, invocation limits, destruction, shutdown,
  observation redaction, trap normalization, interruption, and resource limits.
- Component artifact manifest, digest, WIT, compatibility, trust, revocation,
  quarantine, development mode, cache integrity, authority validation, and
  distribution-source handling.
- Observability correlation, redaction, exporter component classification, sink
  dependencies, queue overflow, filtering, metrics snapshots, and exporter
  failure reporting.

## Integration Test Inventory

The repository does not currently have a separate `tests/` integration-test
tree for `magnetar-runtime`. Cross-module behavior is covered through crate
unit tests and feature-gated module tests, which can access internal Runtime
contracts. Future organization work should split the oversized unit test module
into focused test modules before adding a broad integration-test tree.

## WIT Validation Inventory

CI validates all `*/wit/*.wit` files with `wasm-tools component wit`. The main
runtime suite also asserts key `compute.wit` contract strings, including the
portable placement intent surface and absence of direct Provider or Device
targeting.

## Component Runtime Inventory

Component Runtime tests already cover the mock engine path and, behind
`wasmtime-component-engine`, real Wasmtime fixture loading, linking,
instantiation, invocation, host import failure, trap normalization,
interruption, unauthorized WASI/resource imports, and memory limits.

## Provider Inventory

Provider tests cover registry behavior, Provider metadata and capability
registration, Device ownership, lifecycle initialization/shutdown, loading
policy, stable ABI descriptor validation, unsupported dynamic ABI rejection,
Provider execution submission, output completion, cancellation, stale status,
and status/admission errors.

## Resolution Inventory

Resolution tests cover compatibility ordering, selected Provider diagnostics,
policy rejection, healthy candidate preference, not-ready and stale rejection,
Capability health rejection, phase-aware fallback, and Resource Affinity-aware
resolution.

## Resource Affinity Inventory

Resource Affinity tests cover immutable binding preservation, compatible merges,
fallback precedence, Provider/Device/Capability/Artifact/context/group
conflicts, exact live Capability version preservation, Provider-local
compatible versions, Device ownership reconciliation, and shutdown-runtime
rejection.

## Artifact Trust Inventory

Artifact trust tests cover manifest discovery, missing manifest, digest
mismatch, WIT mismatch, runtime compatibility, Capability compatibility,
publisher trust, signatures, revocation, quarantine, explicit development mode,
cache corruption, broad authority rejection, and source metadata not implying
trust.

## Provider Loading Inventory

Provider loading tests cover default path denial, explicit dynamic and
development loading policy, unsupported legacy Rust trait-object factory
contract rejection, ABI version validation, descriptor layout validation,
function-table validation, ownership-rule validation, lifecycle transitions,
and internal ABI error codes.

## Critical Untested Or Under-Organized Failure Paths

- Tests are concentrated in one oversized unit-test file, which makes contract
  ownership hard to scan.
- Provider and Runtime failure injection utilities exist only as local test
  structs, not as reusable focused fixtures.
- There is no dedicated integration-test module tree organized by Provider,
  Runtime, Component, Artifact, Distribution, Resolution, and Affinity.
- Some Provider ABI failure cases are descriptor-policy tests rather than real
  dynamic library fixture loading tests.
- Coverage is measured in CI, but this change has not yet added a before/after
  coverage comparison.
