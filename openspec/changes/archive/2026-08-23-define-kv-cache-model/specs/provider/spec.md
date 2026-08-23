## ADDED Requirements

### Requirement: Provider KV Cache Resources Are Opaque

Providers SHALL keep any native KV cache resources opaque when they are used.

Provider-owned KV cache resources SHALL remain opaque to Components and public
portable APIs.

#### Scenario: Native cache handle

Given Provider returns a native cache handle

When Runtime records the cache

Then Runtime stores it internally and exposes only Runtime KV cache identity.

---

### Requirement: Provider KV Cache Errors Are Structured

Provider KV cache failures SHALL map to stable Runtime KV cache errors.

#### Scenario: Provider cache failure

Given Provider fails to append KV cache state

When Runtime receives the failure

Then it maps the failure to a stable cache-provider-failure or related error.

---

### Requirement: Provider Status Influences KV Cache Use

Provider health, readiness, pressure, Device status, and memory pressure SHALL
influence KV cache admission and reuse.

#### Scenario: Provider draining

Given a cache resides on a Provider that is draining

When generation attempts to reuse the cache

Then Runtime applies Provider, Resource Affinity, and cache policy.
