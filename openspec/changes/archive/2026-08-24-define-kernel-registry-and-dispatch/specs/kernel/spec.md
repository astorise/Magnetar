## ADDED Requirements
### Requirement: Kernels Are Registered Through Kernel Registry

Kernel advertisements SHALL enter Runtime use through Kernel Registry
validation.

#### Scenario: Provider advertises Kernel

Given Provider advertises a Kernel

When Runtime accepts the advertisement

Then the Kernel becomes available as a registry candidate.

---

### Requirement: Kernels Are Dispatched Through Runtime

Kernels SHALL be dispatched through Runtime-created Kernel Dispatch Plans and
Kernel Invocations.

#### Scenario: Dispatch Kernel

Given a Kernel is selected

When execution begins

Then Runtime dispatches it through the owning Provider.

---

### Requirement: Kernel Metadata Supports Registry Selection

Kernel metadata SHALL be sufficient for Registry filtering, ranking, fallback,
dispatch, and conformance gating.

#### Scenario: Missing shape metadata

Given a Kernel advertisement lacks required shape constraints

When Runtime validates it

Then the advertisement is rejected or marked unusable.

---

### Requirement: Kernel Dispatch Is Revalidated

A selected Kernel SHALL be revalidated before dispatch.

#### Scenario: Provider saturated

Given Kernel was selected while Provider pressure was low

But Provider becomes saturated before dispatch

When revalidation runs

Then Runtime delays, falls back, or rejects according to policy.