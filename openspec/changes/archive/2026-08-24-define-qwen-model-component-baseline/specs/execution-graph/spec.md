## ADDED Requirements

### Requirement: Qwen Component Produces Portable Graphs

Qwen Model Component SHALL produce portable Execution Graphs using Operator
identities.

#### Scenario: Decode graph

Given Qwen decode graph is produced

When Runtime validates it

Then graph nodes reference portable Operators only.

---

### Requirement: Qwen Graphs Are Unfused-Compatible

Qwen baseline graphs SHALL be executable without fused kernels.

#### Scenario: Fused MLP unavailable

Given fused MLP Kernel is unavailable

When Qwen graph production runs

Then graph uses unfused matmul, SiLU, mul, and matmul.

---

### Requirement: Qwen Graphs Preserve Generation Boundary

Qwen graphs SHALL produce model outputs such as hidden states or logits and
SHALL not own Sampling or streaming semantics.

#### Scenario: Logits produced

Given Qwen decode graph produces logits

When Runtime receives logits

Then Sampling Contract remains responsible for token selection.