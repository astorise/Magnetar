## ADDED Requirements
### Requirement: First Profile Operator Set

First profile SHALL provide the Operator semantics required by the Qwen fixture.

Required Operator set SHALL include at minimum:

- embedding
- matmul
- rmsnorm
- rope
- attention
- softmax
- silu
- add
- mul
- residual-add
- dtype conversion
- layout conversion

#### Scenario: Qwen graph validation

Given fixture graph is constructed

When Operator catalog validates it

Then every required node has defined Operator semantics.

### Requirement: Operators Remain Portable

Required Operators SHALL not include Reference CPU implementation details.

#### Scenario: MatMul Operator

Given MatMul is inspected

Then it contains no CPU function pointer or Provider ID.

### Requirement: Operator Validation Is Enforced

Malformed Tensor shapes or attributes SHALL fail before unsafe Kernel
execution.

#### Scenario: Invalid MatMul dimensions

Given inner dimensions do not match

When graph/dispatch validates

Then structured Operator compatibility failure occurs.