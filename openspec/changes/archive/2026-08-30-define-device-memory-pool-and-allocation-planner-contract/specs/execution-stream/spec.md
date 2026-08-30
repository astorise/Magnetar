## ADDED Requirements
### Requirement: Allocation Reuse Is Completion Token Aware

Execution completion SHALL govern when pool-backed slots SHALL be reused.

#### Scenario: Workspace reused across decode steps

Given step N workspace is still referenced by pending Kernel

When step N+1 begins

Then same slot is reused only after ordering guarantees safety.

### Requirement: Deferred Reclaim Is Observable To Allocator

A logically released AllocationLease SHALL remain pending until associated
CompletionTokens terminate.

#### Scenario: Cancelled request

Given Device work continues

When request resources are released logically

Then allocator does not treat storage as free yet.

### Requirement: Transfer Staging Pool Lifetime Is Synchronized

Transfer-staging allocation SHALL remain valid until asynchronous transfer
completes.

#### Scenario: Host-to-Device transfer

Given staging lease is used by transfer stream

When caller releases input

Then staging storage is reclaimed only after transfer completion.