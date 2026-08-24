## ADDED Requirements

### Requirement: Model Loading Is Exposed Through Inference API

Runtime Inference API SHALL expose explicit or policy-controlled implicit model loading.

#### Scenario: Load model through API

Given caller requests model load

When Runtime accepts it

Then Model Loading Contract performs validation and materialization.

---

### Requirement: Inference API Loading Does Not Bypass Trust

Model loading through Runtime Inference API SHALL not bypass Model Artifact trust, Component authority, Memory Manager admission, Provider readiness, or policy validation.

#### Scenario: Untrusted artifact

Given artifact is untrusted

When load request is submitted through API

Then Runtime rejects it before ready Model Instance publication.