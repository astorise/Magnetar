## ADDED Requirements

### Requirement: Reference CPU Validates Operator Semantics

Reference CPU Provider SHALL provide baseline behavior for supported Operators.

#### Scenario: Operator conformance

Given an Operator has Reference CPU implementation

When conformance runs

Then outputs are compared against expected Operator semantics.

---

### Requirement: Unsupported Operators Are Not Assumed

Runtime SHALL not assume Reference CPU supports an Operator unless a Kernel is
advertised.

#### Scenario: GELU unsupported

Given no Reference CPU GELU Kernel is advertised

When graph requires GELU

Then Runtime reports missing Kernel or uses explicit fallback according to
policy.