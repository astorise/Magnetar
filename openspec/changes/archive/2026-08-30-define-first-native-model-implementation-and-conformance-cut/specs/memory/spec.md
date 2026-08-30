## ADDED Requirements
### Requirement: Memory Cut Precedes Model Integration

Mandatory model integration SHALL not allocate Tensor payloads through
Qwen-specific unmanaged storage.

#### Scenario: Qwen projection output

Given Tensor output is created

When storage is needed

Then Memory Manager provides backing.

### Requirement: Reuse Gate Is Completion Safe

Any pool-backed temporal reuse demonstrated by the first cut SHALL respect
ResourceReadiness/CompletionToken.

#### Scenario: Synchronous CPU

Given prior Kernel completion is terminal

When Resource lifetime ends

Then backing may be reused.

### Requirement: Allocation Failure Is Tested

The first implementation cut SHALL include deterministic allocation-failure
handling.

#### Scenario: Pool capacity constrained

Given allocation cannot be satisfied

When Plan/model executes

Then structured memory failure occurs without memory corruption.

### Requirement: Native Pointer Is Not Public Identity

First implementation SHALL not shortcut Tensor identity to a raw host pointer.

#### Scenario: CPU memory

Given backing uses Rust/host allocation

When Tensor Resource is inspected

Then logical Resource identity remains separate from pointer.