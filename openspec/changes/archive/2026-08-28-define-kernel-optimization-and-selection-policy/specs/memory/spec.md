## ADDED Requirements

### Requirement: Memory Manager Is Authoritative For Feasibility

Kernel selection SHALL respect Memory Manager feasibility decisions.

#### Scenario: Fast Kernel workspace exceeds capacity

Given Kernel A is faster

But its workspace cannot be admitted

When selection runs

Then Kernel A is excluded.

---

### Requirement: Memory Cost May Influence Ranking

Runtime SHALL NOT use memory cost to override Memory Manager feasibility decisions, though among feasible candidates memory/workspace cost MAY influence optimization.

#### Scenario: Memory profile

Given two feasible Kernels

When memory profile is active

Then lower memory candidate may rank higher.

---

### Requirement: Selection Does Not Allocate Hidden Memory

Kernel ranking SHALL NOT perform hidden inference allocations.

#### Scenario: Candidate evaluated

Given candidate requires workspace

When eligibility is evaluated

Then feasibility is checked without silently committing unmanaged allocation.