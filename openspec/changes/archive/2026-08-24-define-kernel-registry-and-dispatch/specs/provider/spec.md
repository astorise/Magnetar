## ADDED Requirements
### Requirement: Provider Supplies Kernel Advertisements

Providers SHALL supply Kernel advertisements to Runtime when they expose Kernel
implementations.

#### Scenario: Provider with Kernels

Given a Provider supports matmul and attention

When Runtime initializes the Provider

Then the Provider supplies Kernel advertisements.

---

### Requirement: Provider Does Not Bypass Runtime Dispatch

Providers SHALL execute Kernel Invocations created by Runtime and SHALL NOT
accept direct unvalidated Component Kernel calls.

#### Scenario: Direct Component call

Given a Component attempts to call Provider Kernel directly

When Provider boundary is enforced

Then the call is denied or impossible.

---

### Requirement: Provider Status Affects Registry Eligibility

Provider health, readiness, pressure, admission, lifecycle, and failure SHALL
affect Kernel Registry eligibility and Dispatch revalidation.

#### Scenario: Provider draining

Given Provider enters draining

When Runtime updates Kernel Registry

Then its Kernels are not selected for new work unless policy allows draining use.

---

### Requirement: Provider Dispatch Errors Are Mapped

Provider dispatch errors SHALL map to stable Kernel Dispatch errors.

#### Scenario: Provider execution failure

Given Provider reports native execution failure

When Runtime receives it

Then Runtime maps it to kernel-dispatch-failed or a more specific error.