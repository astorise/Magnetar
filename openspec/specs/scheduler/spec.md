# scheduler Specification

## Purpose
TBD - created by archiving change define-continuous-batching-contract. Update Purpose after archive.
## Requirements
### Requirement: Scheduler Owns Continuous Batching Policy

Scheduler SHALL own continuous batching policy execution under Runtime control.

#### Scenario: Scheduler forms batch

Given admitted operations are waiting

When Scheduler evaluates policy

Then it forms compatible prefill or decode work.

---

### Requirement: Scheduler Does Not Allocate Memory Directly

Scheduler SHALL request Memory Manager admission and reservations rather than
allocating memory directly.

#### Scenario: Batch workspace needed

Given a decode batch requires workspace

When Scheduler plans it

Then it requests Memory Manager feasibility.

---

### Requirement: Scheduler Uses Provider Pressure

Scheduler SHALL consider Provider readiness, admission, and pressure when
forming batches.

#### Scenario: Provider saturated

Given Provider status reports saturated pressure

When Scheduler forms a batch

Then it reduces, delays, or rejects work according to policy.

---

### Requirement: Scheduler Preserves Resource Affinity

Scheduler SHALL preserve Resource Affinity from model residency, KV cache,
Prefix Cache, tensors, and Provider-owned resources.

#### Scenario: KV cache on Device A

Given a batch operation depends on KV cache on Device A

When Scheduler selects placement

Then it preserves compatible placement or requests explicit Runtime movement or
rebuild.

---

### Requirement: Scheduler Maintains Operation State

Scheduler SHALL maintain per-operation state independently within a batch.

#### Scenario: Operation B cancelled

Given operation B is cancelled

When operation A remains active

Then Scheduler does not corrupt operation A state.

