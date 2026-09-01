## ADDED Requirements

### Requirement: Runtime-Owned KV Resources
KV cache data used by first-native decode SHALL be represented by Runtime-owned resources with session, model instance, layer, affinity, lifecycle, and memory accounting metadata.

#### Scenario: Decode reads prior KV
- **WHEN** decode executes after prefill or a prior decode step
- **THEN** attention reads historical KV through Runtime-owned cache resource bindings.

#### Scenario: Wrong session attempts access
- **WHEN** a decode step references KV resources owned by another session
- **THEN** Runtime rejects the access.

### Requirement: Transactional KV Updates
KV cache updates SHALL commit only after the corresponding generation step succeeds and SHALL abort on failure, cancellation, timeout, or session close.

#### Scenario: Sampling fails
- **WHEN** sampling fails after compute prepared a KV update
- **THEN** committed KV state remains unchanged and pending KV state is cleared.

#### Scenario: Commit succeeds once
- **WHEN** a generation step is successfully sampled and token-committed
- **THEN** its KV update is committed exactly once.
