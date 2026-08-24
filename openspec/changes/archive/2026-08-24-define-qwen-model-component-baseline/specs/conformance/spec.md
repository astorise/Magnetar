## ADDED Requirements

### Requirement: Qwen Baseline Conformance

Conformance SHALL include Qwen baseline fixtures for config validation, tensor
inventory, graph production, operator scope, tokenizer compatibility, KV cache
metadata, adapter metadata, quantization rejection, authority, and handle
safety.

#### Scenario: Qwen conformance

Given Qwen Component claims baseline support

When conformance runs

Then it must pass Qwen baseline fixtures.

---

### Requirement: Qwen Baseline CPU Smoke Conformance

Conformance SHALL define a CPU smoke path requirement for a minimal Qwen-like
graph, which SHOULD run where all required Reference CPU kernels exist.

#### Scenario: CPU smoke graph

Given minimal Qwen-like fixture graph

When Reference CPU executes it

Then conformance validates graph planning, dispatch, and output metadata.