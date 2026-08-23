## ADDED Requirements

### Requirement: Model Component Is Separate From Model Artifact

A Model Component, when used, SHALL remain an executable Component Artifact
separate from Model Artifact data.

#### Scenario: Model architecture Component

Given a model architecture implementation is packaged as a Component

And model weights are packaged as Model Artifacts

When Runtime validates them

Then each has separate identity, manifest, trust, compatibility, and caching.

---

### Requirement: Component May Request Model Artifact Authority

A Component SHALL be allowed to request inference-scoped model artifact access authority.

Such authority SHALL grant access only to Runtime-registered Model Artifacts
authorized for the inference context.

#### Scenario: Component requests model artifact read

Given a Component has `model-artifact-read`

When it accesses model data

Then access is mediated through Runtime Model Artifact records

And not arbitrary filesystem paths.
