## ADDED Requirements

### Requirement: Model Artifact Must Be Loaded Before Execution

A Model Artifact SHALL be loaded or materialized into Runtime-owned model
residency before it is used for inference execution.

#### Scenario: Generate from unloaded artifact

Given a Model Artifact is valid and trusted

But not loaded

When generation requires a loaded context

Then Runtime rejects the request or performs explicit policy-controlled loading.

---

### Requirement: Model Residency Is Distinct From Artifact Identity

Model Residency SHALL be distinct from Model Artifact identity.

#### Scenario: Same artifact multiple placements

Given the same artifact is loaded on CPU and GPU

When Runtime reports residency

Then the artifact identity is the same

But residency and Resource Affinity differ.

---

### Requirement: Model Loading Does Not Trust Artifacts

Loading a Model Artifact SHALL NOT make untrusted content trusted.

#### Scenario: Untrusted artifact loading

Given a Model Artifact is untrusted

When loading is requested

Then loading fails before materialization.