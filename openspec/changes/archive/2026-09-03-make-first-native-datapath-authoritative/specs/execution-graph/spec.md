## ADDED Requirements

### Requirement: Graph Is Execution Recipe
The ExecutionGraph SHALL be the authoritative numerical recipe for first-native model compute.

#### Scenario: Graph node is missing a binding
- **WHEN** execution reaches a graph node without a matching prepared plan binding
- **THEN** Runtime rejects execution before provider submission.

#### Scenario: Graph output feeds logits
- **WHEN** graph execution completes
- **THEN** logits used by sampling come from the declared graph output resource.
