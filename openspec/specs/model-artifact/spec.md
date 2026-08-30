# model-artifact Specification

## Purpose
TBD - created by archiving change define-first-native-model-execution-profile. Update Purpose after archive.
## Requirements
### Requirement: First Profile Uses External Model Data

Qwen Model Component SHALL obtain model data from Model Artifact resources
rather than embedding all weights into WASM Component.

#### Scenario: Component loaded

Given Qwen Component Artifact is instantiated

When Model Instance loads

Then weights originate from external Model Artifact.

### Requirement: Fixture Model Artifact Is Deterministic

The first-profile fixture SHALL have stable versioned configuration and weights.

#### Scenario: CI execution

Given same fixture version

When loaded on two runs

Then model weight bytes and configuration identity are identical.

### Requirement: Minimal Physical Format Is Allowed

First profile SHALL allow a constrained model package when it normalizes into
Model Artifact contracts.

#### Scenario: Single safetensors fixture

Given fixture has one weight file

When Model Loader parses it

Then sharded safetensors support is not required.

### Requirement: Required Weights Are Validated

Model Loading SHALL fail if mandatory Qwen tensor is absent or incompatible.

#### Scenario: Missing projection weight

Given fixture lacks required tensor

When model loads

Then structured model-loading failure is returned.

