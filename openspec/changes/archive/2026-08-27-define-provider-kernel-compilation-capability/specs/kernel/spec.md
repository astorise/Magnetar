## ADDED Requirements

### Requirement: Executable Kernel Uses Prepared State

Artifact-backed Kernel SHALL execute only through previously prepared Provider
state.

#### Scenario: Kernel dispatch

Given compatible Kernel has no PreparedKernelId

When dispatch runs

Then Runtime does not invoke source compilation through execution path.

---

### Requirement: Compilation Capability Is Not Kernel Semantics

Kernel compilation mechanism SHALL NOT change portable Operator semantics.

#### Scenario: Triton MatMul

Given MatMul is implemented through generated Triton

When Kernel is registered

Then it still implements the existing portable MatMul Operator contract.