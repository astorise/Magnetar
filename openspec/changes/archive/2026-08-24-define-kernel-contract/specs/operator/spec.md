## ADDED Requirements
### Requirement: Operators Are Implemented By Kernels

Operators SHALL be implementable by one or more Kernels.

#### Scenario: Multiple matmul kernels

Given CPU and CUDA Providers both advertise matmul Kernels

When Runtime plans matmul execution

Then each Kernel is considered an implementation of the matmul Operator.

---

### Requirement: Operator Semantics Constrain Kernels

A Kernel implementing an Operator SHALL preserve the Operator's declared
semantics.

#### Scenario: Approximate operator behavior

Given a Kernel changes observable Operator behavior beyond allowed tolerance

When conformance validates it

Then the Kernel fails conformance.

---

### Requirement: Operator Metadata Feeds Kernel Compatibility

Operator metadata SHALL be used to validate Kernel compatibility, including
attributes, shape rules, dtype rules, layout rules, memory behavior, and
determinism metadata.

#### Scenario: Attention attributes

Given attention Operator requires causal mode

When Runtime selects a Kernel

Then candidate Kernel must support that causal mode.
