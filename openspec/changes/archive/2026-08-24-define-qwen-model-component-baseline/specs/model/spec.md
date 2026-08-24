## ADDED Requirements

### Requirement: Qwen-Compatible Model Artifact

Model Artifact metadata SHALL be permitted to declare a Qwen-compatible
architecture family for Qwen baseline validation.

#### Scenario: Qwen artifact

Given artifact metadata declares Qwen-compatible architecture

When Model Loading validates it

Then Runtime resolves Qwen Model Component compatibility.

---

### Requirement: Model Artifact Does Not Select Qwen Provider

Qwen-compatible Model Artifact metadata SHALL NOT select a Qwen Provider.

#### Scenario: Artifact requests QwenProvider

Given artifact metadata references `QwenProvider`

When Runtime validates metadata

Then Runtime rejects it or treats it as non-authoritative invalid metadata.