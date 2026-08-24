## ADDED Requirements

### Requirement: Reference CPU May Execute Qwen Baseline Graphs

Reference CPU Provider SHALL be permitted to execute Qwen baseline graphs when
all required Operators have compatible CPU kernels.

#### Scenario: CPU Qwen smoke path

Given Qwen graph uses required-now operators

And Reference CPU kernels are available

When Runtime dispatches the graph

Then execution may proceed on Reference CPU.

---

### Requirement: Reference CPU Missing Coverage Fails Explicitly

If Reference CPU lacks a required Qwen baseline operator, Runtime SHALL fail
explicitly.

#### Scenario: Missing RoPE

Given Qwen graph requires RoPE

And Reference CPU lacks RoPE kernel

When Kernel Registry selection runs

Then Runtime returns structured missing kernel error.