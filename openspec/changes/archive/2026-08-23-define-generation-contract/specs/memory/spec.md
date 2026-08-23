## ADDED Requirements

### Requirement: Memory Manager Supports Generation Admission

Memory Manager SHALL support generation memory admission.

Generation memory MAY include input token buffers, output token buffers, logits
buffers, sampling workspace, prefill workspace, decode workspace, and future KV
cache memory.

#### Scenario: Generation memory denied

Given generation requires more memory than policy permits

When Memory Manager evaluates admission

Then generation is rejected, queued, or delayed according to policy.

---

### Requirement: Memory Manager Prepares KV Cache Memory Boundary

Memory Manager SHALL prepare for KV cache memory requirements without requiring
full KV cache semantics in this change.

#### Scenario: Future KV cache estimate

Given generation estimates KV cache memory needs

When Memory Manager evaluates feasibility

Then it treats the memory as KV-cache-related allocation class or placeholder.