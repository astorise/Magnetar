## ADDED Requirements
### Requirement: Model Component May Produce Execution Graph

Execution Graphs SHALL support production by Model Components.

Runtime SHALL validate all Component-produced graphs before planning or
execution.

#### Scenario: Component graph

Given Model Component emits prefill graph

When Runtime receives it

Then graph validation runs before graph planning.

---

### Requirement: Model Component Graphs Must Use Portable Operators

Model Component-produced graphs SHALL use portable Operator identities.

#### Scenario: Provider-specific graph node

Given a Model Component graph contains `cuda.flash_attention`

When Runtime validates the graph

Then validation rejects Provider-specific node identity as portable Operator.

---

### Requirement: Model Component Graphs Do Not Embed Kernel Handles

Model Component-produced graphs SHALL not embed raw Kernel handles, Provider
handles, Device handles, or function pointers.

#### Scenario: Raw kernel pointer

Given a graph includes native function pointer metadata

When Runtime validates it

Then validation fails.