## ADDED Requirements
### Requirement: Runtime Implementation Cut Owns Execution

By the end of the implementation cut, Runtime SHALL own model execution from
loaded Model Instance to produced logits.

#### Scenario: Generation requests logits

Given active Session

When next logits are needed

Then Runtime executes the Model Instance rather than asking caller.

### Requirement: Runtime Cut Uses Production Paths

Mandatory E2E SHALL use normal production Runtime paths rather than `cfg(test)`
replacement execution.

#### Scenario: Test build

Given fixture data is test-only

When inference executes

Then Runtime orchestration code is the same production implementation.

### Requirement: Runtime Cut Has Synthetic Graph Gate

Before Qwen integration is considered complete, Runtime SHALL demonstrate a
synthetic multi-node graph executing through Registry and Provider.

#### Scenario: Synthetic graph succeeds

Given simple Operators and Tensor inputs

When graph runs

Then output is produced through normal dispatch.

### Requirement: Runtime Cut Reuses Prepared Plan

Compatible repeated execution SHALL use PreparedExecutionPlan baseline.

#### Scenario: Decode loop

Given Plan remains valid

When several tokens decode

Then Runtime does not rebuild all Kernel bindings every step.

### Requirement: Runtime Cut Exposes Structured Failures

Integration errors SHALL propagate through structured Runtime error categories.

#### Scenario: Required Kernel unavailable

Given graph is valid

When Plan preparation fails to find Kernel

Then structured failure reaches caller rather than panic or silent fallback.