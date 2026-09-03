## ADDED Requirements

### Requirement: ModelInstance Owns Executed Resource Bindings
An active ModelInstance SHALL expose the stable resource bindings for weights, constants, and adapters used by its prepared graph execution.

#### Scenario: Two instances are loaded
- **WHEN** two ModelInstances for different artifacts execute
- **THEN** each execution uses only the resource bindings owned by its active instance.

#### Scenario: Instance unloads
- **WHEN** a ModelInstance is unloaded
- **THEN** its resource bindings are released according to Runtime policy and cannot be used for new execution.
