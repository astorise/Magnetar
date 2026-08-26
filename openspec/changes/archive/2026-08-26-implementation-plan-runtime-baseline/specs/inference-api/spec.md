## ADDED Requirements

### Requirement: Runtime Inference API Implemented After Core Baseline

Runtime Inference API baseline SHALL be implemented after Tensor, Memory,
Operators, Reference CPU, Kernel Registry, Model Loading, Tokenizer, Qwen
baseline, Generation, and Sampling are sufficiently available.

#### Scenario: API success path

Given Runtime Inference API accepts request

When generation completes

Then it uses the implemented core baseline instead of fake responses.

---

### Requirement: Inference API Baseline Is Inference-Only

Runtime Inference API implementation SHALL not add workspace, Git, tool, shell,
network, secret, or agent responsibilities.

#### Scenario: Tool execution request

Given API request asks Runtime to execute a tool

When validation runs

Then Runtime rejects it.