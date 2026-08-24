## ADDED Requirements

### Requirement: Model Artifact May Have Multiple Instances

Runtime SHALL allow a single Model Artifact to have multiple Model Instances.

#### Scenario: Same artifact CPU and GPU

Given one Model Artifact is loaded for CPU and GPU execution

When Runtime records instances

Then each instance has distinct lifecycle, readiness, and residency.

---

### Requirement: Model Artifact Alone Is Not Executable

A valid and trusted Model Artifact SHALL not be treated as executable inference
state.

#### Scenario: Generate from artifact

Given a Model Artifact is valid

But no ready Model Instance exists

When generation is requested

Then Runtime rejects or performs explicit policy-controlled loading.
