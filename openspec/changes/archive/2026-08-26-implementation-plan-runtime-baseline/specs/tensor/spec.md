## ADDED Requirements

### Requirement: Tensor Implementation Precedes Operator Execution

Tensor Resource and Layout baseline SHALL be implemented before executable
Operator dispatch.

#### Scenario: Operator output

Given matmul Kernel produces output

When execution is implemented

Then Tensor Resource metadata exists to represent the output.

---

### Requirement: Tensor Baseline Is Host Contiguous First

The first Tensor implementation SHALL support host contiguous tensors before
advanced layouts.

#### Scenario: CPU fixture

Given Reference CPU executes fixture graph

When tensors are allocated

Then host contiguous layout is supported.