## ADDED Requirements

### Requirement: Bounded memory bookkeeping

The Memory Manager SHALL keep the bookkeeping it scans during admission proportional to live allocations. Released, evicted, cancelled and failed allocations SHALL leave the working set and MAY be retained only in a bounded history for inspection. Observation logs SHALL be bounded. A Provider SHALL NOT overwrite a caller's residency record for a pre-admitted output, and a resident Model Instance SHALL release each finished request's KV state, session and cache records, so serving many requests does not grow per-step cost or memory with the number of requests served.

#### Scenario: Many requests on a resident instance

- **WHEN** one resident Model Instance serves request after request
- **THEN** its live allocations, residency records, Provider storage and tracking, sessions and KV caches stop growing after the first requests, and per-token latency stays flat

#### Scenario: Inspecting a recent release

- **WHEN** an allocation is released
- **THEN** it is no longer live, `allocations()` still reports it as released while it remains in the bounded history, and releasing it again fails with an invalid handle
