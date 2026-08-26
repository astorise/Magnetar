## ADDED Requirements

### Requirement: Release Observability Redaction

Release builds SHALL preserve default observability redaction.

#### Scenario: Release diagnostics

Given release binary emits diagnostics

When diagnostics are inspected

Then raw prompts, secrets, file contents, model weights, tensor values, KV cache
contents, handles, and memory pointers are absent by default.

---

### Requirement: Release Build Metadata Observability

Release observability MAY include build metadata, but included metadata SHALL
exclude secrets and local filesystem paths.

#### Scenario: Version observation

Given Runtime emits version observation

When metadata is inspected

Then version and feature flags may be included without secrets or local paths.