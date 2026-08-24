## ADDED Requirements
### Requirement: Memory Manager Participates In Kernel Selection

Memory Manager SHALL participate in Kernel candidate feasibility checks for
inputs, outputs, workspace, staging, movement, dtype conversion, and layout
conversion.

#### Scenario: Workspace infeasible

Given a Kernel requires workspace larger than allowed

When selection runs

Then Memory Manager rejects the candidate.

---

### Requirement: Memory Reservations Are Revalidated Before Dispatch

Memory reservations required by a Dispatch Plan SHALL be revalidated before
Kernel dispatch.

#### Scenario: Reservation expired

Given workspace reservation expires before dispatch

When revalidation runs

Then dispatch fails stale or replans according to policy.

---

### Requirement: Dispatch Results Update Memory Metadata

Kernel Dispatch results SHALL update Memory Manager metadata for output
readiness, residency, Resource Affinity, workspace release, and provider-owned
memory accounting.

#### Scenario: Kernel output ready

Given Kernel writes output tensor

When dispatch completes

Then Memory Manager records output readiness and residency metadata.