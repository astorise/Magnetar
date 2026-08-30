## ADDED Requirements
### Requirement: Data Movement Uses Execution Synchronization

Asynchronous residency-changing movement SHALL integrate with ExecutionStream
and CompletionToken semantics.

#### Scenario: Host to GPU upload

Given transfer occurs asynchronously

When movement is submitted

Then destination readiness depends on transfer CompletionToken.

### Requirement: Source Remains Alive During Transfer

Source Resource SHALL not be destroyed/reused before asynchronous movement
finishes.

#### Scenario: Upload input buffer

Given host input upload remains pending

When logical input owner releases it

Then physical source storage remains alive until transfer completion.

### Requirement: Destination Is Not Ready Before Transfer Completion

Destination Resource SHALL not be consumed before transfer completion or
equivalent dependency ordering.

#### Scenario: MatMul after upload

Given GPU upload pending

When MatMul is submitted on another stream

Then explicit dependency orders MatMul after transfer.

### Requirement: Mapping Conflicts Respect Execution Dependencies

Host mapping SHALL synchronize with incompatible Device accesses.

#### Scenario: Host write during GPU read

Given GPU Kernel reads Resource

When host requests write mapping

Then mapping waits or fails according to policy.

### Requirement: Zero Copy Still Uses Readiness

Directly accessible shared Resource SHALL still follow ResourceReadiness.

#### Scenario: Shared host/GPU memory

Given GPU writes shared allocation

When CPU reads it zero-copy

Then host visibility/completion is established first.