## ADDED Requirements
### Requirement: Prepared Plan May Encode Residency Assumptions

Prepared Execution Plan SHALL be able to bind Kernel inputs/outputs to required or preferred
MemoryDomains.

#### Scenario: Decode Plan

Given all decode Kernels run on GPU0

When Plan is built

Then weights, KV, intermediates, and workspace may be planned as GPU0 resident.

### Requirement: Prepared Plan May Elide Redundant Transfers

Plan construction SHALL remove movement that is unnecessary under validated
residency.

#### Scenario: Already-resident Tensor

Given previous node output is GPU0-local

And next Kernel executes on GPU0

When Plan is prepared

Then no GPU0-to-GPU0 staging copy is emitted.

### Requirement: Residency Guard

Prepared Plan SHALL not execute against Resources violating hard residency
assumptions.

#### Scenario: KV spilled to host

Given Plan requires GPU-resident KV

When decode starts

Then Runtime rebinds/transfers/replans according to policy before Kernel
execution.

### Requirement: Prepared Plan Does Not Store Native Addresses

Plan SHALL refer to logical Resource bindings rather than Device pointers.

#### Scenario: CUDA graph-related Plan

Given Provider has native addresses internally

When Plan metadata is inspected

Then raw addresses are absent.

### Requirement: Host Mapping Is An Explicit Boundary

Prepared Plan SHALL designate host-visible output boundary when host access is
required.

#### Scenario: Final logits

Given sampler executes on host

When Plan completes Device logits

Then explicit map/transfer step makes required data host-visible.
