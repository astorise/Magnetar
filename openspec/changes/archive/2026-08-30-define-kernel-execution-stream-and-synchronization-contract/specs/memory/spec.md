## ADDED Requirements

### Requirement: Allocation Reuse Waits For Completion

Memory Manager SHALL not reuse storage while unfinished execution may access it.

#### Scenario: Temporary buffer released logically

Given GPU Kernel still uses buffer

When logical Tensor lifetime ends

Then physical storage remains unavailable for reuse until completion.

### Requirement: Completion Fences Resource Lifetime

Memory Manager SHALL be able to associate allocation lifetime with one or more
CompletionTokens.

#### Scenario: Two readers

Given two streams read shared allocation

When one completes

Then allocation remains retained until second relevant completion occurs.

### Requirement: Workspace Reuse Is Synchronization-Aware

Prepared Plan workspace reuse SHALL honor asynchronous completion.

#### Scenario: Attention workspace reused next step

Given previous Attention execution still pending

When next step requests same workspace

Then reuse is ordered or separate workspace is provided.

### Requirement: Cancellation Does Not Release Memory Early

Cancelled request SHALL retain memory needed by in-flight Provider work.

#### Scenario: User cancels generation

Given current GPU Kernel cannot be interrupted

When request result is cancelled

Then its resources remain alive until Device work completes.

### Requirement: Lost Completion Fails Safe

Memory Manager SHALL not assume storage is safely reusable if completion state
cannot be determined after Provider/Device failure.

#### Scenario: Device disappears

Given Tensor write was pending

When completion becomes lost

Then allocation follows Device/Provider recovery or invalidation policy rather
than immediate reuse.
