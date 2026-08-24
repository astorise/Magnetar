## ADDED Requirements

### Requirement: Execution Graph May Run On Reference CPU

When it executes through Reference CPU Kernels, a validated Execution Graph SHALL be treated as a normal graph subject to standard validation and dispatch.
A validated Execution Graph MAY execute through Reference CPU Kernels where compatible and policy allows.

#### Scenario: CPU graph execution

Given graph operators are supported by Reference CPU Kernels

When Runtime dispatches the graph

Then graph execution may complete on Reference CPU.

---

### Requirement: CPU Execution Does Not Bypass Graph Planning

Execution through Reference CPU SHALL still use graph validation, planning,
Kernel Registry, Kernel Dispatch, and Memory Manager.

#### Scenario: Direct CPU path

Given a graph is valid

When CPU execution is requested

Then Runtime does not bypass normal planning and dispatch.