## ADDED Requirements
### Requirement: Graph Planning Uses Kernel Registry

Execution Graph planning SHALL use Kernel Registry to resolve operator
implementation candidates.

#### Scenario: Plan attention

Given graph contains attention Operator

When Runtime plans execution

Then Kernel Registry provides compatible Kernel Candidates.

---

### Requirement: Graph Plan Contains Dispatch Requirements Not Raw Kernels

Execution Graph plans SHALL contain Runtime-managed Kernel requirements or
Dispatch Plans, not raw native function pointers.

#### Scenario: Inspect graph plan

Given graph plan is inspected

When metadata is returned

Then raw native Kernel function pointers are absent.

---

### Requirement: Graph Execution Uses Runtime Dispatch

Graph execution SHALL execute operators through Runtime Kernel Dispatch.

#### Scenario: Execute planned graph

Given graph planning selects Kernel dispatch plans

When execution runs

Then Runtime dispatches Kernel Invocations through owning Providers.