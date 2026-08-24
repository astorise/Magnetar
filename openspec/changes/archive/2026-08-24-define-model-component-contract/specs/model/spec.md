## ADDED Requirements
### Requirement: Model Artifact May Require Model Component

A Model Artifact SHALL be allowed to declare compatible Model Component requirements.

#### Scenario: Required component missing

Given a Model Artifact requires architecture family `qwen`

And no compatible Model Component or native implementation exists

When Model Loading validates it

Then loading fails with model-component-not-found or architecture-unsupported.

---

### Requirement: Model Artifact Remains Data

Model Artifact SHALL remain data and SHALL NOT embed Provider or Kernel
selection authority.

#### Scenario: Artifact requests CUDA kernel

Given Model Artifact metadata names a CUDA Kernel

When Runtime validates it

Then the metadata is rejected or treated as non-authoritative.