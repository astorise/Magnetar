## ADDED Requirements
### Requirement: Provider Does Not Own Model Architecture

Provider SHALL not be the primary abstraction for model architecture families.

#### Scenario: Qwen support

Given a Provider can execute Qwen graph operators

When Runtime reports support

Then Provider advertises Operator/Kernel capabilities, not QwenProvider identity.

---

### Requirement: Provider Executes Model Component Graphs Through Runtime

Provider SHALL execute only Runtime-validated Operators or Kernel Invocations
derived from Model Component graphs.

#### Scenario: Component graph execution

Given Model Component emits graph

When Provider executes work

Then execution occurs only after Runtime validation and dispatch.

---

### Requirement: Provider Does Not Receive Component Authority

Provider support for a Model Component SHALL not grant that Component Provider
authority.

#### Scenario: Compatible Provider

Given CUDA Provider can execute graph from Component C

When Component C runs

Then C still does not receive CUDA Provider handles.