## ADDED Requirements
### Requirement: Prepared Plan Lifecycle Is Observable

Runtime SHALL expose redacted Plan lifecycle observations.

#### Scenario: Plan becomes ready

Given Plan preparation succeeds

When event is emitted

Then Plan ID, generation, graph fingerprint and workload scope may be reported.

---

### Requirement: Plan Staleness And Invalidation Are Distinguished

Observability SHALL distinguish optimization staleness from hard invalidation.

#### Scenario: Better Kernel available

Given old Plan remains safe

When replacement is requested

Then event says stale rather than revoked/invalid.

---

### Requirement: Plan Replacement Is Observable

Atomic Plan generation replacement SHALL emit old/new generation information.

#### Scenario: Generation 12 replaced by 13

Given promotion completes

When event is emitted

Then both logical Plan generations may be correlated.

---

### Requirement: Plan Guard Failure Is Observable

Runtime SHALL report structured guard-failure reason without exposing Runtime
native resources.

#### Scenario: Sequence too long

Given invocation exceeds Plan envelope

When guard fails

Then workload incompatibility is observable.

---

### Requirement: Plan Observability Redacts Native State

Observability SHALL not expose PreparedKernel native handles,
ProviderPreparedSegment native objects, tensor addresses, model weights, KV
contents, prompts, secrets or credentials.

#### Scenario: Provider segment fails

Given native graph object becomes invalid

When diagnostic is emitted

Then only opaque stable Plan/segment identifiers are reported.
