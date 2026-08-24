## ADDED Requirements

### Requirement: KV Cache May Back Prefix Cache

A sealed KV cache SHALL be representable as backing state referenced by a Prefix Cache entry.

#### Scenario: Sealed cache referenced

Given a KV cache is sealed and compatible

When Prefix Cache stores a reusable prefix

Then the entry may reference that sealed KV cache.

---

### Requirement: Mutable KV Cache Is Not Shared By Default

Mutable KV cache SHALL not be reused through Prefix Cache across unrelated
operations by default.

#### Scenario: Active cache

Given a KV cache is active and mutable

When Prefix Cache considers sharing it

Then Runtime denies reuse unless explicit safe sharing policy exists.

---

### Requirement: KV Cache Invalidation Affects Prefix Cache

Runtime SHALL make dependent Prefix Cache entries stale, invalid, or evicted
according to policy when a KV cache is invalidated, evicted, or released.

#### Scenario: KV cache released

Given a prefix entry references a KV cache

When that KV cache is released

Then the prefix entry is no longer reusable.
