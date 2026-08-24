## ADDED Requirements
### Requirement: Runtime Owns Kernel Validation

Runtime SHALL validate Kernel metadata against Operator invocation, graph plan,
Memory Manager, Provider, Device, Resource Affinity, and policy before dispatch.

#### Scenario: Validate candidate kernel

Given a candidate Kernel is advertised

When Runtime plans execution

Then Runtime validates compatibility before dispatch.

---

### Requirement: Runtime Creates Kernel Invocations

Runtime SHALL create Kernel Invocations.

Components SHALL NOT create direct Provider Kernel invocations.

#### Scenario: Component direct call

Given a Component attempts to invoke a native Kernel directly

When Runtime validates the request

Then the request is denied.

---

### Requirement: Runtime Prevents Raw Kernel Handle Exposure

Runtime SHALL not expose raw Kernel function pointers, Provider handles, Device
handles, or memory pointers through Kernel APIs.

#### Scenario: Kernel metadata request

Given Kernel metadata is requested

When Runtime returns it

Then only stable metadata is exposed.

---

### Requirement: Runtime Applies Kernel Fallback Policy

Runtime SHALL apply explicit fallback policy when a Kernel is unavailable or
incompatible.

#### Scenario: Kernel unavailable

Given preferred Kernel is unavailable

When fallback is permitted

Then Runtime validates alternate Kernel, Provider, Device, memory, dtype, layout,
and Resource Affinity before fallback.

---

### Requirement: Runtime Observes Kernel Execution

Runtime SHALL emit structured observations for Kernel validation, invocation,
dispatch, completion, failure, fallback, conformance, and diagnostics without
exposing raw data or handles.

#### Scenario: Kernel completed

Given a Kernel completes successfully

When Runtime emits observability

Then it records redacted Kernel execution metadata.
