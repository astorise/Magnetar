## ADDED Requirements

### Requirement: KV Cache Supports Batch Slots

KV cache SHALL support association with batch slots where batching is enabled.

#### Scenario: Slot cache

Given a decode operation is assigned to a batch slot

When it uses KV cache

Then the slot references Runtime-managed KV cache state.

---

### Requirement: KV Cache Resource Affinity Constrains Batching

KV cache Resource Affinity SHALL constrain batch placement.

#### Scenario: Cache Device mismatch

Given operation A depends on KV cache on Device A

When Scheduler attempts to batch it on Device B

Then Runtime rejects, moves explicitly, or rebuilds according to policy.
