## ADDED Requirements
### Requirement: Model Instance May Produce Execution Graphs

A Model Instance or its architecture implementation SHALL be able to produce Execution Graphs
for inference phases.

#### Scenario: Decode graph from instance

Given a Model Instance is ready

When decode begins

Then Runtime may obtain a decode Execution Graph compatible with that instance.

---

### Requirement: Graph Identity Depends On Instance Semantics

Execution Graph identity SHALL reflect Model Instance semantic state where
relevant.

#### Scenario: Adapter merge changes semantics

Given a Model Instance changes due to adapter merge

When a graph is built

Then graph identity reflects the changed semantic state.
