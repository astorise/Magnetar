## ADDED Requirements

### Requirement: Model References Are Inference API Inputs

Runtime Inference API SHALL support model references as inputs to model resolution, loading, session creation, or one-shot inference.

#### Scenario: Model reference

Given caller submits model reference `qwen-test`

When Runtime resolves it

Then Runtime maps it to validated Model Artifact metadata or reports resolution
failure.

---

### Requirement: Model Reference Does Not Grant File Access

A model reference SHALL not grant arbitrary filesystem access.

#### Scenario: Path-like reference

Given model reference looks like a filesystem path

When Runtime validates it

Then Runtime uses only authorized model source contracts or rejects it.