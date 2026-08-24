## ADDED Requirements
### Requirement: Graph Planning Applies First Operator Scope

Execution Graph planning SHALL apply first operator implementation scope when
running the initial executable baseline.

#### Scenario: Unsupported graph node

Given graph contains unsupported MoE dispatch

When first baseline planning runs

Then graph planning fails with operator-explicitly-unsupported.

---

### Requirement: Graph Planning Avoids Hidden Substitutions

Graph planning SHALL not silently replace unsupported operators with unrelated
operators.

#### Scenario: Quantized matmul

Given graph requires quantized-matmul

When no implementation exists

Then planning rejects it instead of silently using f32 matmul.

---

### Requirement: First Decoder Graph Is In Scope

A decoder-only graph using the required-now operator set SHALL be considered
valid for first baseline planning if all metadata is compatible.

#### Scenario: Decoder graph valid

Given graph uses embedding, RMSNorm, matmul, RoPE, attention, softmax, SiLU,
add, mul, residual-add, and logits matmul

When first scope validation runs

Then graph operators pass scope validation.