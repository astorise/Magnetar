## ADDED Requirements
### Requirement: Runtime Owns Graph Validation And Planning

Runtime SHALL own Execution Graph validation and planning.

#### Scenario: Graph submitted

Given a graph is produced by a Model Component

When Runtime receives it

Then Runtime validates and plans it before execution.

---

### Requirement: Runtime Prevents Direct Component-To-Provider Graph Execution

Runtime SHALL prevent Components from using graphs to call Providers directly.

#### Scenario: Component graph

Given a Component emits a graph

When graph execution starts

Then Provider interaction occurs only through Runtime-managed dispatch.

---

### Requirement: Runtime Preserves Affinity In Graph Planning

Runtime SHALL preserve Resource Affinity during graph planning and require
explicit movement or conversion when needed.

#### Scenario: Affinity conflict

Given graph edge is Device-bound

When planned operator placement is incompatible

Then Runtime rejects or inserts explicit authorized movement.

---

### Requirement: Runtime Observes Graph Execution

Runtime SHALL define graph and operator observations without exposing raw tensors,
weights, prompts, cache contents, or native handles.

#### Scenario: Operator planned

Given an operator is planned

When observability emits metadata

Then only redacted operator planning metadata is included.
