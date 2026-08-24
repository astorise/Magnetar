## ADDED Requirements

### Requirement: Memory Manager Supports Batch Admission

Memory Manager SHALL support memory admission for continuous batching.

Batching memory MAY include input buffers, output buffers, logits buffers,
attention masks, position buffers, sampling workspace, KV cache blocks, Prefix
Cache lookup workspace, temporary staging, and Provider-specific workspace.

#### Scenario: Batch memory denied

Given a planned batch exceeds memory policy

When Memory Manager evaluates admission

Then Runtime reduces, queues, or rejects the batch according to policy.

---

### Requirement: Memory Pressure Influences Batch Size

Memory pressure SHALL influence batch sizing and admission.

#### Scenario: High memory pressure

Given memory pressure is high

When Scheduler forms the next batch

Then it may reduce batch size or delay prefill according to policy.
