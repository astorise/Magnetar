## ADDED Requirements
### Requirement: Model Artifact May Reference Adapter Compatibility

Model Artifact metadata SHALL preserve adapter compatibility metadata when declared.

#### Scenario: Adapter compatible

Given a base model declares compatible adapter target modules

When adapter validation runs

Then Runtime may use that metadata to validate adapter targets.

---

### Requirement: Adapter Artifact Is Distinct From Base Model Artifact

Adapter Artifact identity SHALL remain distinct from base Model Artifact
identity.

#### Scenario: Adapter and base model

Given adapter A targets base model M

When Runtime records them

Then A and M remain separate artifacts with separate trust decisions.