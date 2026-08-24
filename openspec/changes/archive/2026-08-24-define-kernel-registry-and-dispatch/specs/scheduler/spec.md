## ADDED Requirements
### Requirement: Scheduler Uses Kernel Registry Metadata

Scheduler SHALL be allowed to use Kernel Registry metadata for planning, batching, deadlines,
backpressure, and pressure-aware scheduling.

#### Scenario: Batch planning

Given Scheduler forms a batch

When it estimates feasibility

Then it may use Kernel metadata such as max batch size and workspace.

---

### Requirement: Scheduler Delegates Final Dispatch To Runtime

Scheduler SHALL delegate final Kernel Dispatch validation and invocation
creation to Runtime.

#### Scenario: Scheduled work ready

Given Scheduler selects work to run

When Kernel execution is needed

Then Runtime Kernel Dispatch performs final revalidation and Provider
invocation.

---

### Requirement: Scheduler Handles Dispatch Outcomes

Scheduler SHALL handle Kernel Dispatch outcomes such as completion, failure,
timeout, cancellation, fallback, and backpressure.

#### Scenario: Dispatch timeout

Given Kernel Dispatch times out

When Scheduler receives the result

Then Scheduler updates operation state according to policy.