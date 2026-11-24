# pooled-decode-replay Specification

## Purpose
This specification defines decode step recordings shared by every request of a pooled KV store set: capture, capture again at an earlier position, fallback and release.
## Requirements
### Requirement: Pooled decode recordings

When a generation request's KV stores are a pooled store set and step replay is enabled, the Runtime SHALL key that request's decode recording by the decode graph's semantic fingerprint and the store set, not by the request. It SHALL run the set's decode steps in a resource namespace of the set and the graph, whose intermediates, token input and logits outlive the request. A later request served by the same set SHALL replay that recording from its first decode step. Every replayed step SHALL produce the logits and tokens of a per-node step.

#### Scenario: Requests of one prompt length

- **WHEN** four sequential requests with the same 3-token prompt each run 4 decode steps on one pooled store set
- **THEN** the first request captures its second decode step, the second request captures its first decode step, every other decode step of the four requests replays, and each request's logits match the per-node run

#### Scenario: What a recording writes

- **WHEN** a pooled set's decode step is recorded
- **THEN** every resource it writes is the set's decode namespace's, a prefill bucket's, or one of the set's KV stores

### Requirement: Capture again at an earlier position

The Runtime SHALL replay a pooled set's decode recording only at a position at or after the recorded step's. At an earlier position, it SHALL capture the step again and replace the recording.

#### Scenario: A shorter prompt

- **WHEN** a set's decode step was recorded at position 6, and the set's next request has a 3-token prompt
- **THEN** that request's first decode step, at position 3, is captured, and its later steps replay

### Requirement: Fallback and release

The Runtime SHALL decline a pooled decode key whose capture fails or whose Provider cannot capture, without keeping a namespace for the Provider that cannot capture. A request whose key is declined SHALL record its own decode steps. When the KV store pool drops a set, the Runtime SHALL release the set's decode recordings and namespace resources together with its stores.

#### Scenario: Failed capture

- **WHEN** every capture of the Provider fails
- **THEN** each request's steps run per node with correct results, and no pooled decode recording is kept

#### Scenario: A dropped set

- **WHEN** the pool drops a set whose decode step was recorded
- **THEN** its recording is released, no `decode{set}-` resource remains, and the next request starts its sequence again with correct results
