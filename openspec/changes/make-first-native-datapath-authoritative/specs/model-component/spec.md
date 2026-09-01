## ADDED Requirements

### Requirement: Component Produces Graph Semantics
The Qwen Model Component SHALL produce or describe the portable graph semantics that Runtime validates and prepares for strict first-native execution.

#### Scenario: Component absent in strict profile
- **WHEN** strict first-native inference requires the component and the component engine or artifact is unavailable
- **THEN** Runtime fails closed with a structured component error.

#### Scenario: Node count matches but semantics differ
- **WHEN** the component output has the expected node count but invalid or wrong graph semantics
- **THEN** Runtime rejects the component output before execution.
