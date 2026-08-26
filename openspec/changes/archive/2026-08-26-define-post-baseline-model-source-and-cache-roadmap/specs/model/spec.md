## ADDED Requirements

### Requirement: Model Source Candidate

Model Artifact workflow SHALL support source candidates that normalize into
Model Artifact metadata.

#### Scenario: Client-provided artifact

Given client provides artifact source

When normalization succeeds

Then Model Artifact metadata is available for loading.

---

### Requirement: Model Artifact Identity Uses Digest

Model Artifact identity SHALL use digest-based identity where possible.

#### Scenario: Name collision

Given two artifacts named `qwen-local`

When their digests differ

Then Runtime treats them as distinct artifacts.

---

### Requirement: Cached Model Still Validates

Cached Model Artifacts SHALL still pass validation before loading.

#### Scenario: Cached untrusted artifact

Given cached artifact is untrusted

When Model Loading runs

Then Runtime rejects it.