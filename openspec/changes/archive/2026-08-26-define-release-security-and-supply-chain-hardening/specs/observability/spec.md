## ADDED Requirements

### Requirement: Release Observability Security Redaction

Release observability SHALL be redacted by default.

#### Scenario: Release observation

Given inference request includes prompt and model artifact

When observation is emitted

Then raw prompt, weights, tensors, cache contents, secrets, credentials, handles,
pointers, and raw file contents are absent.

---

### Requirement: Release Security Event Recording

Release process SHALL record security gate status, and recording SHOULD avoid
exposing sensitive content.

#### Scenario: Secret scan failed

Given secret scan fails

When release metadata is recorded

Then failure status is recorded without printing the secret.