## ADDED Requirements

### Requirement: Authoritative E2E Uses Production Runtime Path
The authoritative first-native E2E suite SHALL instantiate production Runtime APIs and SHALL NOT execute the model through a separate harness engine.

#### Scenario: Same path as CLI
- **WHEN** authoritative first-native E2E runs generation
- **THEN** it exercises the same RuntimeInferenceApi path used by CLI generation.

### Requirement: E2E Evidence Is Observational
First-native E2E evidence SHALL be collected from bounded observations emitted by the Runtime layers that actually executed work.

#### Scenario: Provider evidence comes from provider submission
- **WHEN** E2E asserts Provider execution
- **THEN** the assertion is based on Provider submission and completion observations emitted by the Provider/dispatch path.

#### Scenario: Evidence cannot be self-declared
- **WHEN** a test helper sets only a boolean claiming a layer executed
- **THEN** that value is insufficient for authoritative E2E conformance.

### Requirement: E2E Proves No First-Native Shortcuts
The authoritative first-native E2E suite SHALL fail if model execution bypasses Component validation, graph validation, PreparedExecutionPlan, Kernel Registry, Provider dispatch, Runtime-owned KV, or Sampling.

#### Scenario: Shortcut removed
- **WHEN** any required layer is disabled or unavailable
- **THEN** E2E fails with a structured error rather than silently falling back.
