## ADDED Requirements

### Requirement: Registry Handles Optimized Provider Candidates

Kernel Registry SHALL support optimized Provider candidates without bypassing
normal validation and ranking.

#### Scenario: CUDA and CPU candidates

Given CUDA and Reference CPU kernels exist for matmul

When Kernel Registry ranks candidates

Then it validates compatibility, readiness, memory, policy, and Resource
Affinity before selection.

---

### Requirement: Registry Requires Conformance Metadata For Advanced Features

Kernel Registry SHALL NOT select an advanced-feature Kernel candidate that lacks required conformance metadata.
Kernel Registry SHOULD consider conformance metadata for advanced Provider
features.

#### Scenario: Unconformant flash attention

Given flash attention Kernel lacks required conformance

When Registry selects candidates

Then the Kernel is rejected or ranked unavailable according to policy.