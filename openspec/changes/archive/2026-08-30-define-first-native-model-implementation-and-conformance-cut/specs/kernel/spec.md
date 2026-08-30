## ADDED Requirements
### Requirement: Every Mandatory Kernel Has Unit Gate

Each first-profile Reference CPU Kernel SHALL pass focused correctness tests
before final model conformance.

#### Scenario: MatMul implementation exists

Given no independent numerical test exists

When implementation gate is evaluated

Then Kernel task is incomplete.

### Requirement: Kernel Unit Tests May Use Direct Kernel Boundary

Focused Kernel tests SHALL be allowed to invoke Kernel implementation directly at the Kernel
unit-test layer.

#### Scenario: Softmax golden test

Given test specifically validates Softmax implementation

When it invokes Kernel implementation directly

Then this does not violate E2E bypass prohibition.

### Requirement: E2E Shall Not Use Direct Kernel Boundary

Direct Kernel invocation SHALL not be used as the System Under Test model path.

#### Scenario: Qwen Attention

Given native E2E runs

When Attention executes

Then Kernel is reached through Registry and Provider.

### Requirement: Kernel Correctness Is Independent From Model Fixture

Kernel correctness tests SHALL include data independent from Qwen fixture
weights.

#### Scenario: MatMul test

Given model fixture changes

When Kernel unit suite runs

Then core mathematical tests remain valid.
