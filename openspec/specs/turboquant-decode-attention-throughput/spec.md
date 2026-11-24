# turboquant-decode-attention-throughput Specification

## Purpose
This specification defines the throughput requirements of decode attention over TurboQuant codes: codes read once per chunk and a bounded number of splits.
## Requirements
### Requirement: Codes Read Once Per Chunk

The CUDA Provider's decode attention over TurboQuant stores SHALL read each chunk's codes and norms from global memory once, with coalesced loads, and SHALL produce the attention `turboquant-kv-history` and `keep-turboquant-kv-head-exact` specify, within f32 rounding.

#### Scenario: Same results
- **WHEN** a decode step attends over a TurboQuant history, with or without an exact head
- **THEN** its output matches the CPU semantics within the existing conformance tolerances

### Requirement: Bounded Splits

The CUDA Provider SHALL split a decode step's history over a number of blocks per KV head chosen from the Device's multiprocessors and the history, at most 32, and SHALL merge one partial per block, whatever the history's length.

#### Scenario: Long history
- **WHEN** a decode step attends over 32,768 keys
- **THEN** each query head's partials number at most 32

#### Scenario: Captured step
- **WHEN** a captured decode step is replayed with a longer history
- **THEN** the same blocks cover the whole longer history and the output matches an uncaptured step's
