## ADDED Requirements

### Requirement: Runtime Failure Paths Are Tested

Runtime orchestration SHALL have tests for critical failure paths.

#### Scenario: Provider changes status before submission

Given Resolution selected a Provider

And the Provider becomes not-ready before Scheduler submission

When the Scheduler checks admission

Then tests verify Runtime policy decides retry, queue, or failure.

---

### Requirement: Scheduler Does Not Resolve Providers

Tests SHALL verify that Scheduler consumes validated execution plans rather than
independently selecting Providers.

#### Scenario: Alternate Provider available

Given an execution plan selects Provider A

And Provider B is also compatible

When Scheduler receives the plan

Then Scheduler does not silently switch to Provider B.

---

### Requirement: Resource Affinity Test Coverage

Runtime tests SHALL verify Resource Affinity precedence over policy preference.

#### Scenario: Policy prefers different Provider

Given Resource Affinity requires Provider A

And policy prefers Provider B

When dependent work is resolved

Then tests verify Provider B is not selected without explicit movement.

---

### Requirement: Runtime Shutdown Test Coverage

Runtime shutdown behavior SHALL be covered by tests.

#### Scenario: Shutdown with active work

Given active Component or Provider work exists

When Runtime shutdown begins

Then tests verify new work is prevented and existing work is drained,
interrupted, or failed according to policy.

---

### Requirement: Observability Failure Isolation Tests

Runtime tests SHALL verify that observability failures do not alter execution
correctness.

#### Scenario: Observation sink fails

Given a Provider execution succeeds

And observability delivery fails

When Runtime completes the operation

Then tests verify execution result remains successful.
