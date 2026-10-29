## ADDED Requirements

### Requirement: D06 Operational paged KV pool

KV storage SHALL implement page leases, reference counts, copy-on-write, fragmentation accounting and garbage collection, recycling pages only after all CompletionTokens and owners release them.

#### Scenario: D06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** two sequences share sealed pages and one appends
- **THEN** copy-on-write isolates mutations and only unreferenced completed pages return to the pool

#### Scenario: D06 Ineligible or failed execution

- **WHEN** a sequence is cancelled while attention still reads its pages
- **THEN** future work stops but pages remain leased until completion; pool exhaustion returns a structured admission result

### Requirement: paged-kv-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
