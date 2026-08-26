## ADDED Requirements

### Requirement: Inference API Accepts Normalized Model References

Runtime Inference API MAY accept model references that resolve to normalized Model Artifacts from supported formats, and Model Loading SHALL apply standard validation to every such reference.

#### Scenario: safetensors model reference

Given caller references a safetensors-based model

When Runtime resolves it

Then Runtime loads the normalized Model Artifact through standard loading.

---

### Requirement: Inference API Does Not Download Formats Arbitrarily

Runtime Inference API SHALL not perform arbitrary model downloads during
inference.

#### Scenario: Remote URL inference

Given inference request contains remote model URL

When Runtime validates it

Then Runtime uses authorized source contracts or rejects arbitrary network
access.