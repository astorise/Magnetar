## ADDED Requirements
### Requirement: First Profile Uses Runtime Memory Manager

All first-profile model Tensor backing SHALL be governed by Runtime Memory
Manager.

#### Scenario: Intermediate activation

Given Qwen MatMul creates output

When storage is allocated

Then allocation occurs through Memory Manager rather than Qwen Component native
allocation.

### Requirement: Simple Pool Is Sufficient

First profile SHALL allow a simple CPU-compatible pool/arena to satisfy pool
semantics.

#### Scenario: Only one physical arena

Given implementation logically manages all CPU allocations in one pool

When lifetime/alignment/resource rules are preserved

Then implementation is conformant.

### Requirement: Persistent And Temporary Lifetimes Are Distinguished

Memory Manager SHALL preserve sufficient lifetime classes for weights, KV, and
temporary execution resources.

#### Scenario: Temporary activation released

Given weight remains Model Instance-owned

When activation ends

Then weight storage is not reclaimed.

### Requirement: Planned Storage May Be Reused

Non-overlapping temporary resources SHALL be allowed to reuse backing according to
AllocationPlan and CompletionToken semantics.

#### Scenario: Sequential CPU operations

Given prior Resource is completed and dead

When later compatible Resource needs storage

Then planned slot may be reused safely.
