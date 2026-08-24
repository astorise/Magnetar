## ADDED Requirements

### Requirement: Runtime May Register Reference CPU Provider

Reference CPU Provider registration decisions SHALL be explicit and SHALL not depend on implicit defaults.
Runtime MAY register a Reference CPU Provider when policy and build features allow it.

#### Scenario: CPU provider disabled

Given Runtime policy disables Reference CPU Provider

When Runtime initializes Providers

Then Reference CPU Provider is not registered or is unavailable.

---

### Requirement: Runtime Treats Reference CPU As Normal Provider

Runtime SHALL route Reference CPU execution through Provider, Kernel Registry,
Kernel Dispatch, Memory Manager, and observability contracts.

#### Scenario: CPU dispatch

Given Reference CPU Kernel is selected

When execution runs

Then Runtime creates normal Kernel Dispatch Plan and Invocation.

---

### Requirement: Runtime Prevents Silent CPU Fallback

Runtime SHALL not use Reference CPU fallback silently.

#### Scenario: CUDA unavailable

Given CUDA Kernel is unavailable

When CPU fallback is not permitted

Then Runtime reports failure instead of using Reference CPU.

---

### Requirement: Runtime Observes Reference CPU Use

Runtime SHALL emit observations when Reference CPU Provider is registered,
selected, used, failed, or used as fallback.

#### Scenario: CPU fallback denied

Given fallback to CPU is denied

When Runtime rejects fallback

Then observability records redacted fallback denial metadata.