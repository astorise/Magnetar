# Test Coverage Traceability

This document records the implementation evidence for the OpenSpec change
`expand-runtime-contract-and-failure-test-coverage`.

## Evidence Sources

- `magnetar-runtime/src/tests.rs` contains the main Runtime contract and failure
  suite.
- `magnetar-runtime/src/component_wasmtime.rs` contains feature-gated concrete
  WASM Component Engine fixture tests.
- `magnetar-runtime/src/observability/exporter.rs` contains observability
  exporter and sink isolation tests.
- `magnetar-runtime/tests/contract_tests/` contains public integration
  contract tests grouped by Provider, Runtime, Component, Artifact,
  Distribution, Resolution, and Affinity.
- `.github/workflows/quality.yml` runs formatting, check, Clippy, workspace
  tests, Wasmtime feature tests, WIT and fixture validation, OpenSpec
  validation, and coverage ratchet validation.

## Runtime And Provider

Provider registry, metadata, Capability advertisement, Device metadata,
initialization failure, duplicate Provider IDs, status dimensions, readiness,
pressure, drain, stale status, execution submission, execution completion,
cancellation, output affinity, release, ABI descriptor validation, dynamic
loading policy, and stable ABI rejection are covered by the Provider,
Scheduler, Resolution, and ABI tests in `src/tests.rs` and the Provider
integration tests.

The configurable Provider and Device fixtures are represented by the internal
`TestProvider`, `TestProviderExecutionApi`, `DeviceDescriptor`, and public
integration `ContractProvider` test fixture. They support metadata,
Capability advertisement, Device metadata, health/status snapshots,
initialization failure, execution success, cancellation outcomes, stale status,
drain/status combinations, and Provider-owned resource checks.

## Resolution And Affinity

Resolution Policy coverage includes healthy versus not-ready selection,
unavailable and stale rejection, Capability health rejection, status rejection
diagnostics, phase-aware fallback, Provider/Device/Capability selection
diagnostics, and deterministic preference behavior.

Resource Affinity coverage includes Provider, Device, Capability, Artifact,
execution-context, and group binding preservation; incompatible affinity
errors; Provider-local compatible versions; exact live Capability versions;
Device ownership reconciliation; shutdown rejection; and explicit transfer or
materialization diagnostics.

## Compute Boundary And Planning

Compute boundary coverage includes the WIT surface, absence of Provider,
Device, and AffinityGroup targeting fields, portable placement intents,
host-staging policy, diagnostics as output metadata, Compute v1/v2
compatibility, operation schema validation, unsupported operation families,
unsupported dtype/layout, memory planning, data movement planning, resolved
execution plans, and Scheduler consumption of validated plans.

## Component Runtime And WASM Fixtures

The mock `MockComponentEngine` supports preparation success/failure,
instantiation success/failure, trap simulation, interruption simulation,
resource-limit failure, multiple isolated instances, destroyed instances, and
contract mismatch behavior.

Feature-gated Wasmtime tests cover real fixture preparation, instantiation,
invocation, missing artifacts, invalid bytes, malformed WAT, unauthorized WASI
and resource imports, host import roundtrip, host failure versus Component
trap, deadline interruption, isolated instance state, and memory limits.

## Artifact, Authority, And Distribution

Component artifact coverage includes digest success and mismatch, manifest
failure and missing fields, unsupported runtime compatibility, WIT import and
export mismatch, Capability compatibility, untrusted artifacts, revoked and
quarantined artifacts, explicit development mode, cache verification and
corruption rejection, publisher/signature policy, and structured observations.

Inference authority coverage includes trusted model/tokenizer artifact reads,
compute Capability authority, observability emit authority, broad authority
rejection for filesystem, network, secrets, Git, workspace, and process, and
trust/development-mode precedence over forbidden authority.

Distribution coverage includes local-directory and client-provided packages,
Tachyon-labelled source metadata not implying trust, source digest mismatch,
distributed manifest/WIT/authority/revocation failures, cache hits with digest
verification, cache corruption rejection, and offline local distribution.

## Observability, Cancellation, Shutdown, And Concurrency

Observability coverage includes execution correlation, exporter failure
isolation, sink dependency authorization, queue saturation/overflow policy,
trust/status non-authority, diagnostic redaction, and absence of native handles
in public observations.

Cancellation and shutdown coverage includes Component invocation cancellation,
Compute operation cancellation, Provider cancellation unsupported/failure
mapping, Runtime shutdown preventing new lifecycle operations, Component
instance destruction on shutdown, Provider shutdown/drain behavior, and
resource release according to ownership.

Concurrency-sensitive coverage is deterministic and focused on Scheduler queue
admission, cancellation/completion terminal states, same-instance invocation
limits, multiple Component instances, and Provider status checks at admission.

## Table-Driven And Property Decisions

Provider status combinations, authority validation, Compute compatibility, and
trust precedence are implemented with compact loop/table-style tests where the
current API shape makes that useful. Property-test crates were not introduced:
version compatibility and digest normalization are deterministic and covered by
focused examples without adding a new dependency.

## Fixture And CI Policy

Fixtures are checked in under `magnetar-runtime/fixtures/components`, are WAT
text fixtures, do not require external network, Tachyon, or GPU hardware, and
are validated by the CI WIT job with `wasm-tools`.

Coverage before this change is the accepted baseline in
`quality/coverage-baseline.json`. Coverage after this change is generated by
`cargo llvm-cov` and checked against the ratchet; test source remains excluded
from production coverage by policy.
