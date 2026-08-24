## ADDED Requirements

### Requirement: Runtime Owns Model Loading

Runtime SHALL own model loading coordination.

Runtime SHALL coordinate artifact validation, architecture compatibility, Memory
Manager feasibility, Provider/Device compatibility, materialization, residency,
unload, and failure cleanup.

#### Scenario: Runtime loads model

Given a valid Model Loading Request

When Runtime processes it

Then Runtime coordinates all loading phases through Runtime-owned services.

---

### Requirement: Runtime Prevents Direct Provider Selection During Loading

Runtime SHALL prevent Model Artifacts and loading requests from directly
selecting Providers as authoritative execution targets.

#### Scenario: Loading request names Provider

Given a loading request attempts to force Provider `cuda`

When Runtime validates the request

Then Runtime rejects it or treats it as non-authoritative policy input.

---

### Requirement: Runtime Prevents Direct Device Selection During Loading

Runtime SHALL prevent Model Artifacts and loading requests from directly
selecting Devices as authoritative placement targets.

#### Scenario: Loading request names Device

Given a loading request attempts to force Device `gpu0`

When Runtime validates the request

Then Runtime preserves Runtime-owned placement resolution.

---

### Requirement: Runtime Cleans Up Failed Loads

Runtime SHALL clean up or invalidate resources after failed loading.

#### Scenario: Allocation then failure

Given loading allocated memory

And materialization fails

When Runtime reports loading failure

Then allocated memory is released or marked invalid according to policy.

---

### Requirement: Runtime Invalidates Dependent State On Unload

Runtime SHALL invalidate or release dependent state when a loaded model context
is unloaded according to policy.

Dependent state MAY include sessions, KV caches, residency records, and Provider
resources.

#### Scenario: Unload with KV cache

Given a KV cache depends on a loaded model context

When the model is unloaded

Then Runtime invalidates or releases the cache according to policy.

---

### Requirement: Runtime Observes Model Loading

Runtime SHOULD emit observations for model loading lifecycle events without
exposing raw weights or native handles.

#### Scenario: Loading failed

Given loading fails during materialization

When Runtime emits observability

Then it includes stable phase and error metadata.