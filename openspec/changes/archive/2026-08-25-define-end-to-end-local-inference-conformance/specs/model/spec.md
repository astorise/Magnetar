## ADDED Requirements

### Requirement: E2E Fixture Model Artifact

E2E conformance SHALL use a fixture Model Artifact that still passes normal
Model Artifact validation.

#### Scenario: Fixture artifact

Given fixture model artifact is loaded

When validation runs

Then normal artifact identity, manifest, config, tensor inventory, and trust
checks are applied.

---

### Requirement: E2E Fixture Does Not Bypass Model Artifact Contract

Fixture models SHALL not bypass Model Artifact validation.

#### Scenario: Invalid fixture manifest

Given fixture manifest is invalid

When E2E loading runs

Then Model Loading fails before Model Instance creation.