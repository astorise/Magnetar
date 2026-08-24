## ADDED Requirements
### Requirement: Generation Uses Execution Graphs Where Available

Generation SHALL be able to use Execution Graphs for prefill and decode execution.

#### Scenario: Prefill graph

Given a ready Model Instance exposes a prefill graph

When Generation runs prefill

Then Runtime validates, plans, and executes that graph.

---

### Requirement: Generation Semantics Remain Separate From Graph

Execution Graphs SHALL not redefine Generation stop conditions, streaming
semantics, or request lifecycle.

#### Scenario: Decode graph returns logits

Given decode graph produces logits

When next token is needed

Then Generation continues to use Sampling and stop condition contracts.
