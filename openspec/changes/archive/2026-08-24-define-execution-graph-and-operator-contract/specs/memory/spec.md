## ADDED Requirements
### Requirement: Memory Manager Plans Graph Memory

Memory Manager SHALL participate in Execution Graph memory planning.

Graph memory MAY include tensor edges, operator outputs, workspace, layout
conversions, dtype conversions, KV cache inputs/outputs, adapter paths, and
temporary buffers.

#### Scenario: Operator workspace

Given an operator requires workspace

When Runtime plans the graph

Then Memory Manager admits, queues, or rejects workspace allocation.

---

### Requirement: Memory Manager Tracks Tensor Edge Residency

Memory Manager SHALL track residency for graph tensor edges where those edges
correspond to Runtime-managed allocations.

#### Scenario: Tensor output

Given an operator writes tensor T to Device memory

When execution completes

Then Memory Manager tracks T residency and Resource Affinity.

---

### Requirement: Memory Manager Prevents Silent Movement

Memory Manager SHALL require explicit Runtime-planned data movement, dtype
conversion, or layout conversion.

#### Scenario: Host staging forbidden

Given graph planning would require host staging

And policy forbids host staging

When planning runs

Then planning fails instead of silently staging.