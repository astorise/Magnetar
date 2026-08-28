## ADDED Requirements

### Requirement: Execution Graph Remains Portable

Execution Graph SHALL express portable Operator semantics rather than executable
Kernel Artifact content.

#### Scenario: Generated attention kernel exists

Given Registry has generated attention implementation

When Model Component builds graph

Then graph still contains portable Attention Operator.

---

### Requirement: Graph Does Not Embed Native Kernel Handles

Portable graph SHALL NOT embed PreparedKernelId or native handles.

#### Scenario: Graph serialized

Given graph is inspected or transported

When representation is produced

Then Provider-native execution state is absent.

---

### Requirement: Graph Does Not Embed Arbitrary Source

Portable graph SHALL NOT include arbitrary executable source as part of normal
Operator nodes.

#### Scenario: Triton source available

Given generated Triton kernel exists

When graph is built

Then graph references Operator semantics rather than Triton code.