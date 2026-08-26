## ADDED Requirements

### Requirement: Qwen Baseline Implemented After Operator Scope

Qwen baseline implementation SHALL depend on first operator scope metadata.

#### Scenario: Produce graph

Given Qwen baseline produces decode graph

When graph is validated

Then all operator requirements are checked against first scope.

---

### Requirement: Qwen Baseline Does Not Introduce QwenProvider

Implementation SHALL not add QwenProvider.

#### Scenario: Provider registry

Given Qwen baseline is implemented

When Provider registry is inspected

Then no QwenProvider appears.