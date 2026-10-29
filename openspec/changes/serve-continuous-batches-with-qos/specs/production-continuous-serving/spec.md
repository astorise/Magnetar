## ADDED Requirements

### Requirement: E01 Production continuous batching loop

Runtime SHALL admit real production sessions into mixed decode steps with isolated KV, stop conditions, token commits and stream attribution through existing batching contracts.

#### Scenario: E01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** requests enter and leave a running batch
- **THEN** each result matches its per-session reference and released slots are reused only after completion

#### Scenario: E01 Ineligible or failed execution

- **WHEN** one session cancels or errors during a submitted step
- **THEN** only its future work stops and other sessions retain their correct state and outputs

### Requirement: E02 Fair chunked prefill

The scheduler SHALL split supported long prefills by token/time budgets while preserving absolute positions and fair access for active decode work.

#### Scenario: E02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** long prompts arrive during decode traffic
- **THEN** bounded prefill chunks produce the reference logits and report TTFT/TPOT fairness

#### Scenario: E02 Ineligible or failed execution

- **WHEN** a chunk would exceed memory or the model does not support segmentation
- **THEN** admission defers or explicitly rejects chunking without publishing partial invalid KV

### Requirement: E05 Local cost aware request routing

Runtime SHALL route locally among eligible Model Instances using context length, occupied KV, FLOP quota, pressure and TTFT/TPOT objectives after affinity and admission validation.

#### Scenario: E05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** several compatible local execution candidates are available
- **THEN** routing records its costs and preserves session-affine placement unless movement is explicitly authorized

#### Scenario: E05 Ineligible or failed execution

- **WHEN** the lowest-cost candidate conflicts with active KV affinity
- **THEN** it is excluded rather than causing implicit session migration

### Requirement: E06 Serving QoS and bounded streaming

Serving SHALL enforce profile-defined priority, deadlines, fairness, starvation protection, backpressure, per-session cancellation and request budgets.

#### Scenario: E06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** sessions at concurrency 1, 4, 16 or 32 run when capacity permits
- **THEN** tail latency, starvation and cancellation behavior are measured under the declared policy

#### Scenario: E06 Ineligible or failed execution

- **WHEN** a slow stream consumer or expired deadline exhausts its budget
- **THEN** bounded buffering and policy-driven cancellation protect other sessions and preserve completion safety

### Requirement: production-continuous-serving promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
