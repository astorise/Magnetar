## ADDED Requirements
### Requirement: Qwen Component Participation Is Mandatory

The final native E2E SHALL prove the WASM Qwen Model Component was instantiated
and used.

#### Scenario: Component Artifact unavailable

Given Model Artifact exists

When Qwen Component cannot load

Then E2E fails instead of falling back to native model-specific implementation.

### Requirement: Component Test Precedes Full E2E

Qwen Component SHALL have focused architecture/config tests independent from
full generation.

#### Scenario: Invalid KV head count

Given incompatible Qwen config

When Component validates architecture

Then structured component/model error is returned.

### Requirement: Component Does Not Gain Temporary Provider Authority

Implementation convenience SHALL not permit temporary direct Provider/Device
selection inside Qwen Component.

#### Scenario: CPU is only Provider

Given first profile always runs CPU

When Component graph is inspected

Then concrete CPU selection is still absent.