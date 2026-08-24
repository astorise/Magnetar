## ADDED Requirements
### Requirement: Execution Graphs Represent KV Cache Use

Execution Graphs that use KV cache SHALL represent cache inputs, outputs,
append behavior, layout metadata, and compatibility requirements.

#### Scenario: Decode appends KV cache

Given decode graph appends KV state

When Runtime validates the graph

Then KV cache behavior is explicit.

---

### Requirement: Graph Planning Preserves KV Cache Affinity

Graph planning SHALL preserve KV cache Resource Affinity.

#### Scenario: KV cache on Device A

Given graph consumes KV cache bound to Device A

When planning selects execution

Then compatible placement is used or explicit movement/rebuild is required.