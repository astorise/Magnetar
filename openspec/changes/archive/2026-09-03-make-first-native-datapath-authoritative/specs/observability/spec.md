## ADDED Requirements

### Requirement: Causal Execution Evidence
Runtime observations SHALL be emitted by the layer that performed the observed action and SHALL correlate request, session, model instance, graph node, plan generation, kernel, provider, device, submission, completion, KV, sampling, and token commit identities when applicable.

#### Scenario: Provider operation succeeds
- **WHEN** a provider completes execution for a graph node
- **THEN** the observation identifies the graph node, plan binding, provider binding, submission, and completion involved.

#### Scenario: Operation is not executed
- **WHEN** a graph node is skipped or rejected before provider submission
- **THEN** Runtime emits no successful provider completion observation for that node.

#### Scenario: Observation buffer is inspected
- **WHEN** observations are collected
- **THEN** prompt text, raw tensor contents, raw KV bytes, native handles, and pointers are absent and the buffer remains bounded.
