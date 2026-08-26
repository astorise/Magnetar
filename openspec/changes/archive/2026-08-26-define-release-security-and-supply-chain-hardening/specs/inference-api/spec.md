## ADDED Requirements

### Requirement: Inference API Release Security

Runtime Inference API release SHALL remain inference-only and redacted by
default.

#### Scenario: Release inference diagnostics

Given diagnostics are requested after inference

When response is returned

Then raw prompt, raw weights, raw tensors, raw KV cache, secrets, credentials,
handles, and memory pointers are absent by default.

---

### Requirement: Inference API Rejects Non-Inference Authority

Runtime Inference API SHALL reject requests for filesystem, network, secret,
shell, process, Git, tool, or agent authority.

#### Scenario: Tool execution request

Given inference request asks Runtime to execute tool

When Runtime validates it

Then request is rejected.