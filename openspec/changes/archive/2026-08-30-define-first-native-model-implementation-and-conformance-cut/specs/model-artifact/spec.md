## ADDED Requirements
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