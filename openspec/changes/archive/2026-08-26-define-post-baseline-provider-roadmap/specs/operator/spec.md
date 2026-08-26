## ADDED Requirements

### Requirement: Optimized Providers Preserve Operator Semantics

Optimized Providers SHALL preserve portable Operator semantics.

#### Scenario: Optimized softmax

Given optimized softmax Kernel is selected

When output is compared to reference fixture

Then output matches within declared tolerance.

---

### Requirement: Advanced Operator Variants Are Explicit

Advanced Operator variants such as flash attention or paged attention SHALL be
explicit variants or graph fragments, not hidden substitutions.

#### Scenario: Hidden flash attention

Given Provider silently replaces attention with unsupported flash semantics

When conformance validates execution

Then conformance fails.