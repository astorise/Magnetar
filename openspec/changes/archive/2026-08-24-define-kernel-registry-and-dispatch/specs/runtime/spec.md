## ADDED Requirements
### Requirement: Runtime Owns Kernel Registry

Runtime SHALL own Kernel Registry, advertisement validation, indexing,
candidate lookup, invalidation, and conformance gating.

#### Scenario: Provider registration

Given a Provider registers Kernel advertisements

When Runtime accepts the Provider

Then Runtime validates and indexes eligible Kernels.

---

### Requirement: Runtime Owns Kernel Dispatch

Runtime SHALL own Kernel Dispatch planning, revalidation, invocation creation,
fallback, result handling, and cleanup.

#### Scenario: Dispatch selected Kernel

Given Kernel selection succeeds

When execution starts

Then Runtime creates a Dispatch Plan and Provider Kernel Invocation.

---

### Requirement: Runtime Prevents Raw Kernel Access

Runtime SHALL not expose raw native Kernel function pointers or Provider handles
through registry or dispatch APIs.

#### Scenario: Kernel list

Given a caller lists available Kernels

When Runtime returns metadata

Then no function pointers or raw handles are present.

---

### Requirement: Runtime Applies Dispatch Policy

Runtime SHALL apply policy during candidate selection, ranking, fallback,
revalidation, dispatch, cancellation, timeout, and result handling.

#### Scenario: Determinism required

Given deterministic execution is required

When candidate Kernels are ranked

Then nondeterministic candidates are rejected or deprioritized according to
policy.

---

### Requirement: Runtime Observes Registry And Dispatch

Runtime SHALL support structured observations for Kernel Registry and Dispatch
without exposing raw data or handles.

#### Scenario: Dispatch failed

Given Kernel dispatch fails

When Runtime emits observability

Then it records redacted Kernel, Provider, Device, and error metadata.