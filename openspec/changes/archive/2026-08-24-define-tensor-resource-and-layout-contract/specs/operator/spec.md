## ADDED Requirements

### Requirement: Operators Use Tensor Contracts

Operators SHALL consume and produce Tensor Descriptors or Tensor Resources using the Tensor Resource and Layout Contract.

#### Scenario: Operator input validation

Given Operator receives Tensor Resource input

When validation runs

Then shape, dtype, layout, memory behavior, aliasing, and Resource Affinity are validated.

---

### Requirement: Operators Do Not Require Raw Tensor Pointers

Operator semantics SHALL be expressed without raw memory pointers.

#### Scenario: Matmul semantics

Given matmul Operator is defined

When Component inspects it

Then it sees Tensor metadata requirements, not raw pointer requirements.

---

### Requirement: Operator Layout Requirements Are Explicit

Operator layout requirements SHALL be explicit and validated.

#### Scenario: Layout mismatch

Given Operator requires contiguous input

When input is strided

Then Runtime rejects or plans explicit layout conversion.