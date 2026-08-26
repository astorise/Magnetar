## ADDED Requirements

### Requirement: Model Formats Feed Memory Planning

Normalized model format metadata SHALL provide enough size, dtype, layout, and
shard metadata for Memory Manager planning.

#### Scenario: Sharded model

Given sharded artifact metadata is normalized

When Memory Manager plans loading

Then it can estimate or compute required memory.

---

### Requirement: Memory Mapping Is Policy-Controlled

If memory mapping is supported for model formats, it SHALL be policy-controlled
and SHALL not expose raw mmap pointers through public APIs.

#### Scenario: mmap requested

Given safetensors loading uses memory mapping

When Runtime reports metadata

Then no raw memory pointer is exposed.