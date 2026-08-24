## ADDED Requirements
### Requirement: Kernels Declare First Scope Coverage

Kernel metadata SHALL indicate whether the Kernel participates in first
operator implementation scope.

#### Scenario: CPU kernel in scope

Given Reference CPU matmul kernel is advertised

When Registry records it

Then metadata may mark it as first-scope capable.

---

### Requirement: Placeholder Kernels Require Explicit Status

If a Kernel advertisement corresponds to a placeholder Operator, Runtime SHALL
require explicit implemented status and conformance before use.

#### Scenario: Placeholder kernel

Given Provider advertises paged-attention

When first scope validates it

Then Runtime requires concrete support metadata and conformance status.
