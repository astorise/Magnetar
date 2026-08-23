## ADDED Requirements

### Requirement: Runtime Owns KV Cache Lifecycle

Runtime SHALL own KV cache creation, lookup, compatibility validation,
invalidation, eviction, and release.

#### Scenario: Cache lookup

Given a generation operation references a cache

When Runtime resolves it

Then Runtime validates lifecycle, compatibility, authority, and residency.

---

### Requirement: Runtime Prevents KV Cache Forgery

Runtime SHALL reject client- or Component-forged KV cache identities and
affinity metadata.

#### Scenario: Forged cache affinity

Given a request attempts to claim a cache is on Provider A

When Runtime validates the request

Then Runtime ignores or rejects the claim unless it comes from Runtime-owned
state.

---

### Requirement: Runtime Protects KV Cache Privacy

Runtime SHALL not expose raw KV cache content, raw prompt text, or raw cache
handles by default.

#### Scenario: Cache diagnostics

Given cache diagnostics are requested

When Runtime returns diagnostics

Then diagnostics are redacted and do not include raw cache tensors.

---

### Requirement: Runtime Applies KV Cache Policy

Runtime SHALL apply policy to cache creation, reuse, sharing, sealing,
invalidation, eviction, and retention.

#### Scenario: Sharing disabled

Given cache sharing is disabled

When another session attempts reuse

Then Runtime rejects reuse with cache-sharing-denied.
