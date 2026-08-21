# resolution-policy Specification

## Purpose
Define how the Runtime chooses among compatible Provider implementations for a
requested Capability, how it records policy decisions, and when fallback or
re-resolution is permitted without violating Resource Affinity.

## Requirements
### Requirement: Resolution Policy

The Runtime SHALL use a Resolution Policy when choosing among compatible
Providers for a requested Capability.

A Resolution Policy SHALL produce a Resolution Decision.

#### Scenario: Resolve compatible Providers

Given multiple Providers implement the same Capability

When a Component requests that Capability

Then the Runtime evaluates the Providers using the active Resolution Policy.

---

### Requirement: Resolution Context

The Runtime SHALL provide a Resolution Context to the Resolution Policy.

The Resolution Context SHALL include:

- requested Capability identifier
- requested Capability version
- compatible Provider candidates
- available Device metadata when applicable
- Resource Affinity constraints when applicable
- fallback classification when applicable
- execution phase when applicable

#### Scenario: Evaluate resolution context

Given a Component requests a Capability

When the Runtime evaluates candidates

Then the Resolution Policy receives enough context to make a deterministic
selection.

---

### Requirement: Deterministic Selection

Resolution Policies SHALL produce deterministic decisions for identical inputs.

#### Scenario: Repeat resolution

Given the same Providers

And the same Capability request

And the same Resource Affinity constraints

When resolution is executed multiple times

Then the same Provider is selected.

---

### Requirement: Candidate Rejection

The Runtime SHALL allow a Resolution Policy to reject a compatible Provider
candidate.

The Runtime SHALL expose a structured reason when all candidates are rejected.

#### Scenario: Policy rejects candidate

Given a Provider is technically compatible

And the active policy rejects it

When no other candidate is available

Then the Runtime returns a policy rejection error.

---

### Requirement: Resource Affinity Preservation

The Runtime SHALL preserve Resource Affinity for dependent calls.

Provider-pinned resources SHALL NOT be silently re-resolved to another Provider.

#### Scenario: Provider-pinned resource

Given a generation session is bound to a Provider

When a dependent call is made on that session

Then the Runtime uses the existing Provider affinity

And it does not invoke a fresh Provider selection.

---

### Requirement: Transparent Re-resolution

The Runtime SHALL permit transparent re-resolution of a Capability before
observable work has started.

Transparent re-resolution SHALL only occur when Resource Affinity permits it.

#### Scenario: Transparent fallback

Given a Capability request has no created state

And the preferred Provider is unavailable

And another compatible Provider exists

When resolution occurs

Then the Runtime may select the alternative Provider.

---

### Requirement: Restartable Re-resolution

The Runtime SHALL permit restartable re-resolution only when the complete input
can be replayed safely.

Restartable re-resolution SHALL NOT duplicate observable output.

#### Scenario: Restartable request

Given an operation fails before returning observable output

And the input is replayable

And another compatible Provider exists

When the Resolution Policy allows restart

Then the Runtime may restart the operation on another Provider.

---

### Requirement: Provider-pinned Execution

Provider-pinned resources SHALL remain bound to their owning Provider and Device
until explicitly released, cancelled, interrupted or failed.

#### Scenario: Provider unavailable after state creation

Given a Provider-pinned session has emitted output

When the owning Provider becomes unavailable

Then the Runtime reports interruption or failure

And it does not silently continue on another Provider.

---

### Requirement: Execution Phase Awareness

The Resolution Policy SHALL consider the execution phase when fallback or
re-resolution is evaluated.

Execution phases include:

- before resource creation
- after resource creation
- after submitted work
- after observable output

#### Scenario: Phase-sensitive fallback

Given a candidate supports restart before output

When observable output has already been emitted

Then the Runtime rejects transparent fallback.

---

### Requirement: Decision Diagnostics

The Runtime SHALL record diagnostics for Resolution Decisions.

Diagnostics MAY include selected Provider, selected Device, rejected candidates
and policy reasons.

Diagnostics SHALL NOT expose native handles, raw backend errors or unstable
Provider internals as part of the stable contract.

#### Scenario: Inspect decision

Given a Resolution Decision

When diagnostics are requested

Then the Runtime returns stable identifiers and policy reasons only.
