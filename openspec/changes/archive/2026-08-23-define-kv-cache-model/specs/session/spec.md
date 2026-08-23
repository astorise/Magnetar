## ADDED Requirements

### Requirement: Session May Own KV Cache

An Inference Session SHALL define how it may own or reference KV cache resources according to
session policy.

#### Scenario: Session cache

Given session policy enables KV cache

When generation prefill completes

Then the session may reference the created KV cache.

---

### Requirement: Session KV Cache Policy

A session SHALL define or reference policy for KV cache usage, budget, reuse,
sharing, persistence, and eviction.

#### Scenario: Cache budget exceeded

Given a session KV cache budget is exceeded

When generation attempts to append cache state

Then Runtime rejects, evicts, rebuilds, or fails according to policy.

---

### Requirement: Session Close Handles KV Cache

When a session closes, session-owned KV cache resources SHALL be released,
evicted, retained, or transferred to Runtime cache according to policy.

#### Scenario: Close with cache

Given a session owns a KV cache

When the session closes

Then Runtime applies session KV cache cleanup policy.

---

### Requirement: Session Status Redacts KV Cache

Session status SHALL not expose raw KV cache contents.

#### Scenario: Inspect session cache status

Given a session has KV cache state

When status is inspected

Then Runtime may report cache metadata such as size or lifecycle

And not raw key/value tensors.
