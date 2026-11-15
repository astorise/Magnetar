## ADDED Requirements

### Requirement: Bucketed prefill

When prefill replay is enabled and a prompt of `n` tokens fits the largest configured bucket, the Runtime SHALL run its prefill as a bucket of `b >= n` rows, the smallest configured, right-padded with in-vocabulary ids. It SHALL commit only the `n` real rows to the KV history, and SHALL take the next token's distribution from row `n - 1`. A prompt longer than the largest bucket SHALL run the exact-length per-node prefill.

#### Scenario: Short prompt bucketed

- **WHEN** a 5-token prompt is prefilled with buckets 16, 32, 64 and 128
- **THEN** the prefill computes 16 rows, the committed history holds 5 rows, and the logits come from row 4

#### Scenario: Equivalence with the unpadded prefill

- **WHEN** the same prompt is prefilled bucketed and with prefill replay disabled
- **THEN** the greedy tokens are identical and the logits agree within the f32 qualification tolerance

### Requirement: Logits row selection without contract changes

The Runtime SHALL select the logits row through a Runtime-inserted `embedding` row gather fed by a Runtime-provided input. It SHALL NOT require any WIT, Component or Operator change.

#### Scenario: Unchanged Component

- **WHEN** an existing Component's prefill graph runs bucketed
- **THEN** it links unchanged, and only Runtime-inserted nodes and inputs differ

### Requirement: Pooled KV stores

The Runtime SHALL keep the KV stores of a released bucketed request in a per-Model-Instance pool keyed by capacity bucket. The pool SHALL remain admitted in the Memory Manager under a pool owner, SHALL be bounded by a configured byte budget and set count, and SHALL be reclaimed before an admission fails for lack of memory. A pooled store set SHALL be used by at most one live request, and its valid row count SHALL be reset before reuse.

#### Scenario: Store reuse

- **WHEN** two sequential requests fall in the same capacity bucket
- **THEN** the second request's KV stores have the same Device addresses as the first's

#### Scenario: Memory pressure

- **WHEN** an admission would fail while the pool holds store sets
- **THEN** pooled sets are released, least recently used first, before the admission is refused

### Requirement: Prefill capture and replay

For a key (Model Instance, graph fingerprint, prompt bucket, capacity bucket, store set), the Runtime SHALL record a bucketed prefill through the existing step capture contract, after one per-node prefill of that key. It SHALL replay that recording with offset 0 for later prefills of the same key, writing only the token ids and the logits row index. A failed capture or an invalidated replay SHALL fall back to the per-node prefill without failing the request.

#### Scenario: Third same-shape request replays

- **WHEN** three sequential short requests share a key on a Provider that supports step capture
- **THEN** the first prefill runs per node, the second is captured, and the third is replayed

#### Scenario: Invalidated recording

- **WHEN** a pooled store set was released and reallocated at different addresses
- **THEN** its recording is not replayed, and the prefill runs per node

### Requirement: Controls and qualification

Prefill bucketing, the KV store pool and prefill capture SHALL be disabled together by `ProductionLoadedModel::set_prefill_replay(false)` and by `MAGNETAR_PREFILL_REPLAY=0`. With them disabled, the engine SHALL run the exact-length per-node prefill. Promotion SHALL require a real-checkpoint campaign on real hardware that reports warm TTFT with and without replay in the same run, logits and token agreement with the unpadded prefill, and the bytes the pool holds.

#### Scenario: Disable control

- **WHEN** `MAGNETAR_PREFILL_REPLAY=0`
- **THEN** no prompt is padded, no store is pooled, and no prefill is captured
