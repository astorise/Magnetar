## ADDED Requirements

### Requirement: Generation Uses KV Cache

Generation SHALL use KV cache through Runtime-managed cache references where
cache is enabled.

#### Scenario: Decode after prefill

Given prefill has populated a ready KV cache

When decode runs

Then Generation uses the cache to continue token production.

---

### Requirement: Prefill May Populate KV Cache

Prefill SHALL route KV cache creation or population through Runtime policy when
KV cache is enabled.

#### Scenario: Prefill prompt

Given a generation request includes prompt tokens

When prefill executes

Then Runtime may allocate and populate KV cache entries for those tokens.

---

### Requirement: Decode May Append KV Cache

Decode SHALL route key/value state append for newly generated tokens through
Runtime policy when KV cache is enabled.

#### Scenario: Append generated token

Given decode produces a token

When model state advances

Then Runtime may append corresponding KV state to the cache.

---

### Requirement: Generation Validates Cache Compatibility

Generation SHALL validate KV cache compatibility before reuse.

#### Scenario: Prompt mismatch

Given a cache was created for prompt prefix A

When generation for prompt prefix B attempts reuse

Then Runtime rejects reuse or rebuilds according to policy.

---

### Requirement: Generation Handles Cache Invalidation

If KV cache becomes invalid during generation, Runtime policy SHALL determine
whether generation fails, rebuilds, retries, or cancels.

#### Scenario: Device reset

Given Device reset invalidates cache residency

When generation continues

Then Runtime handles the invalid cache according to policy.
