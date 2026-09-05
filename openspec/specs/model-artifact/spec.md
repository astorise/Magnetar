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

### Requirement: Fixture Is Versioned Before Final E2E

The deterministic Qwen fixture SHALL have explicit version identity before
final golden conformance is accepted.

#### Scenario: Weight bytes change

Given fixture weights are regenerated differently

When E2E golden is evaluated

Then fixture version/digest mismatch is detected.

### Requirement: Model Artifact Loading Is In System Under Test

Final E2E SHALL load model data through Model Artifact/Loading contracts.

#### Scenario: Test already has weight arrays

Given fixture helper can construct weights in memory

When native E2E runs

Then it SHALL NOT bypass the required Model Artifact loading path.

### Requirement: Malformed Artifact Failure Is Required

At least one malformed/missing model-data case SHALL be exercised.

#### Scenario: Required tensor missing

Given Model Artifact lacks Q projection weight

When load occurs

Then structured failure is returned.

### Requirement: Tensor Content Digest Binding

When a tensor's inventory entry declares a content digest, that digest SHALL identify the specific bytes that count as that tensor's content for the artifact it belongs to.

A Model Artifact's tensor inventory MAY declare a per-tensor content digest; declaring one is optional per tensor. A tensor entry that declares no digest is not covered by this requirement; whether its content is verified is left to whatever mechanism does declare a digest for it, if any.

#### Scenario: Declared digest identifies real content

Given a tensor inventory entry declares a content digest

When the artifact's real bytes for that tensor are hashed with the digest's algorithm

Then the computed digest matches the declared one

#### Scenario: Tensor without a declared digest is unconstrained by this requirement

Given a tensor inventory entry declares no content digest

When that tensor's content is later supplied for materialization

Then this requirement imposes no constraint on it

