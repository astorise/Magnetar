## ADDED Requirements

### Requirement: Model Formats Integrate With Source Cache

Model format normalization SHALL integrate with source/cache workflow.

#### Scenario: Safetensors from cache

Given safetensors artifact is found in cache

When normalization runs

Then normalized Model Artifact metadata is produced before loading.

---

### Requirement: Format Parser Does Not Own Source Policy

Format parsers SHALL not decide whether a source is allowed or trusted.

#### Scenario: Valid GGUF denied

Given GGUF metadata is parseable

But source policy denies it

When loading runs

Then loading is rejected by policy.