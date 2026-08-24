## ADDED Requirements
### Requirement: Runtime Owns Adapter Loading

Runtime SHALL coordinate adapter artifact validation, base model compatibility,
Memory Manager feasibility, Provider compatibility, materialization, activation,
deactivation, unload, and failure cleanup.

#### Scenario: Load adapter

Given a valid AdapterLoadingRequest

When Runtime processes it

Then Runtime coordinates all adapter loading phases.

---

### Requirement: Runtime Prevents Direct Provider Selection For Adapters

Runtime SHALL prevent Adapter Artifacts and adapter loading requests from
directly selecting Providers as authoritative execution targets.

#### Scenario: Adapter requests Provider

Given an adapter loading request attempts to force Provider `cuda`

When Runtime validates it

Then Runtime rejects it or treats it as non-authoritative policy input.

---

### Requirement: Runtime Prevents Silent Adapter Activation

Runtime SHALL not activate adapters without explicit request or explicit policy.

#### Scenario: Adapter loaded

Given adapter A is loaded and ready

When generation runs without adapter activation

Then Runtime does not apply A.

---

### Requirement: Runtime Cleans Up Failed Adapter Loads

Runtime SHALL clean up or invalidate resources after failed adapter loading,
activation, merge, or materialization.

#### Scenario: Materialization failure

Given adapter memory was allocated

And materialization fails

When Runtime reports failure

Then allocated memory is released or marked invalid according to policy.

---

### Requirement: Runtime Observes Adapter Lifecycle

Runtime SHALL define observations for adapter loading, activation, deactivation,
merge, unload, failure, cache invalidation, and batching compatibility without
exposing raw tensors or handles.

#### Scenario: Adapter load failed

Given adapter loading fails during validation

When Runtime emits observability

Then it includes stable phase and error metadata.