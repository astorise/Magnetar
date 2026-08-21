# Tasks

## 1. Correct the OpenSpec Change

- [x] 1.1 Add the required Capability declarations and align the proposal with the current repository.
- [x] 1.2 Replace the invalid delta headings and add the phase-aware Provider fallback delta.
- [x] 1.3 Add `design.md` covering scope, invariants, resolution, errors, groups, fallback, and exclusions.
- [x] 1.4 Replace non-verifiable resource tasks with numbered tasks scoped to existing runtime APIs.

## 2. Resource Affinity Foundation

- [x] 2.1 Define runtime-local IDs, Provider/Device/Capability/artifact bindings, and fallback classification.
- [x] 2.2 Implement immutable `ResourceAffinity` facts and conflict-detecting `AffinityConstraints` aggregation.
- [x] 2.3 Implement structured affinity validation and resolution errors with stable diagnostic values.
- [x] 2.4 Add a reusable `AffinityResource<T>` host-side envelope for opaque resources.

## 3. Affinity-Aware Resolution

- [x] 3.1 Assign every built Runtime a process-local execution-context identity.
- [x] 3.2 Add Provider-specific compatible-version lookup without changing stateless resolution.
- [x] 3.3 Add single-Provider affinity-aware resolution that preserves all selected bindings.
- [x] 3.4 Reject unavailable or inconsistent bound Providers and Devices without implicit fallback.

## 4. Documentation

- [x] 4.1 Publish the Resource Affinity rules and compatibility matrix in `docs/architecture/resource-affinity.md`.
- [x] 4.2 Document host-adapter examples for Compute tensor, graph, and operation resources.
- [x] 4.3 Document model/tokenizer/template and generation-session examples as future integration guidance.

## 5. Verification

- [x] 5.1 Add unit tests for every binding conflict, compatible aggregation, artifact roles, and fallback precedence.
- [x] 5.2 Add resolver tests for Provider/Device pinning, exact live versions, Provider-local version selection, context isolation, and structured failures.
- [x] 5.3 Run Rust formatting, Clippy with warnings denied, and all workspace tests.
- [x] 5.4 Run strict OpenSpec validation for `define-resource-affinity-model`.
