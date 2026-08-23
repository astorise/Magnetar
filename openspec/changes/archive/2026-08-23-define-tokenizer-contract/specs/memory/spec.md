## ADDED Requirements

### Requirement: Memory Manager Supports Tokenizer Residency

Memory Manager SHALL support tokenizer artifact and vocabulary residency where
needed.

#### Scenario: Tokenizer vocabulary loaded

Given tokenizer vocabulary data is loaded

When Runtime records residency

Then Memory Manager may track vocabulary memory usage.

---

### Requirement: Memory Manager Supports Token Buffers

Memory Manager SHALL support tokenization-related buffers.

Token buffers MAY include encode output buffers, batch token buffers, attention
masks, token type IDs, and streaming decode state.

#### Scenario: Batch encoding memory

Given batch encoding requires token output buffers

When Memory Manager evaluates the request

Then it admits, queues, or rejects according to memory policy.
