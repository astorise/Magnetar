## ADDED Requirements

### Requirement: Causal First-Native Datapath
The first-native execution profile SHALL prove that local inference is caused by the Model Component, Runtime-validated ExecutionGraph, PreparedExecutionPlan, Runtime ProviderLoader, Runtime MemoryManager, ModelInstance resources, Runtime-owned KV cache, and sampling path.

#### Scenario: Baseline path is complete
- **WHEN** first-native inference completes successfully
- **THEN** the emitted evidence identifies each required datapath layer as causally used for the produced token.

#### Scenario: Shortcut fails conformance
- **WHEN** any required datapath layer is bypassed in the first-native profile
- **THEN** the first-native conformance check fails.
