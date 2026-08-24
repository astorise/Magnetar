## ADDED Requirements
### Requirement: Model Component Declares Operator Requirements

Model Components SHALL declare portable Operator requirements for architecture
execution.

#### Scenario: Qwen operators

Given Qwen Model Component supports decode

When Runtime inspects requirements

Then it sees portable Operator IDs such as attention, matmul, rope, rmsnorm, and
activation.

---

### Requirement: Operator Requirements Are Not Kernel Requirements

Model Component Operator requirements SHALL not be authoritative Provider Kernel
selection.

#### Scenario: Kernel name declared

Given a Model Component declares a Provider-specific Kernel name as required

When Runtime validates it

Then Runtime rejects it or treats it as non-authoritative invalid metadata.