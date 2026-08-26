## ADDED Requirements

### Requirement: Inference API Compatibility Status

Release metadata SHALL declare Runtime Inference API compatibility status.

#### Scenario: API status

Given `v0.1` is released

When compatibility notes are inspected

Then Runtime Inference API is marked stable-for-baseline or unstable as
appropriate.

---

### Requirement: Inference API Release Safety

Runtime Inference API release SHALL not expose internal handles.

#### Scenario: Release API audit

Given release API docs are generated

When responses are inspected

Then Provider, Device, Kernel, tensor pointer, memory pointer, KV cache, and raw
weight internals are absent.