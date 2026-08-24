## ADDED Requirements

### Requirement: Qwen Baseline Uses Tensor Contract

Qwen baseline graphs SHALL use Tensor Descriptors and Tensor Resources through
the Tensor Resource and Layout Contract.

#### Scenario: Qwen graph edge

Given Qwen prefill graph contains hidden state edge

When graph is validated

Then the edge has explicit shape, dtype, layout, and semantic role metadata.

---

### Requirement: Qwen Baseline Tensor Layout Is Explicit

Qwen baseline SHALL target explicit layouts and SHALL not assume hidden tensor
layout.

#### Scenario: Unknown layout

Given model tensor layout is unknown

When Qwen loading validates tensors

Then Runtime rejects or requires explicit materialization metadata.