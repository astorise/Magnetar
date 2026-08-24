## ADDED Requirements

### Requirement: Graph Edges Use Tensor Descriptors

Execution Graph edges SHALL use Tensor Descriptors before materialization and Tensor Resources after planning where applicable.

#### Scenario: Graph edge

Given graph operator A produces edge E

When graph is validated

Then E has Tensor Descriptor metadata.

---

### Requirement: Graph Planning Materializes Tensor Resources

Graph planning SHALL materialize Tensor Resources through Runtime and Memory Manager where execution requires storage.

#### Scenario: Planned output

Given graph output requires storage

When planning runs

Then Runtime asks Memory Manager to plan Tensor Resource allocation.

---

### Requirement: Graph Planning Makes Tensor Conversion Explicit

Graph planning SHALL make dtype conversion, layout conversion, memory movement, and host staging explicit.

#### Scenario: DType mismatch

Given producer outputs f16

And consumer requires f32

When graph planning runs

Then explicit dtype conversion is inserted or planning fails.