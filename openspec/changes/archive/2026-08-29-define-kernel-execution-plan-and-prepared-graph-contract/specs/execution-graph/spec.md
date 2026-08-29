## ADDED Requirements
### Requirement: Execution Graph Remains Semantic Source

Prepared planning SHALL not replace Execution Graph as portable semantic
contract.

#### Scenario: Plan inspected

Given Plan contains Provider-specific execution decisions

When semantics are validated

Then original Execution Graph remains authoritative.

---

### Requirement: Graph Fingerprint Is Stable

Runtime SHALL be able to derive deterministic graph identity from portable
semantics.

#### Scenario: Same graph rebuilt

Given equivalent topology and portable attributes

When fingerprint is calculated

Then identity remains stable according to graph-fingerprint version.

---

### Requirement: Provider Preparation Does Not Mutate Portable Graph

Provider graph capture/preparation SHALL not write Provider-native state into
portable Execution Graph.

#### Scenario: CUDA Graph captured

Given CUDA Provider prepares segment

When original graph is inspected

Then CUDA graph handles are absent.