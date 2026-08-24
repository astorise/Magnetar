## ADDED Requirements

### Requirement: Memory Manager Accounts For Prefix Cache

Memory Manager SHALL account for Prefix Cache metadata, index memory, lookup
workspace, and backing KV cache references.

#### Scenario: Prefix metadata allocation

Given Prefix Cache creates an entry

When metadata is allocated

Then Memory Manager accounts for that memory.

---

### Requirement: Memory Pressure May Evict Prefix Cache

Runtime SHALL allow Memory Manager pressure to trigger Prefix Cache eviction according to Runtime
policy.

#### Scenario: Memory pressure

Given prefix cache memory pressure is high

When Runtime applies eviction policy

Then Prefix Cache entries may be evicted.
