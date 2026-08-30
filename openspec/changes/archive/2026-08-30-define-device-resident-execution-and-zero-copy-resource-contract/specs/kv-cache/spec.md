## ADDED Requirements
### Requirement: KV Cache Supports Device Residency

KV-cache entries/pages SHALL support remaining resident on execution Device
across decode steps.

#### Scenario: Long decode

Given Session performs 1000 decode steps on GPU0

When capacity permits

Then KV does not require host round-trip between steps.

### Requirement: KV Residency Preserves Completion Ordering

Device-resident KV SHALL still follow asynchronous write/read readiness.

#### Scenario: Append followed by Attention

Given K/V append is pending

When next decode step starts

Then ResourceReadiness orders use correctly.

### Requirement: KV Spill Is Explicit

Moving KV page from Device to another memory domain SHALL be explicit and
policy-controlled.

#### Scenario: Memory pressure

Given inactive sequence may spill

When host staging is permitted

Then transfer is visible and completion-safe.

### Requirement: Active KV Cannot Be Evicted Unsafely

KV page referenced by in-flight Kernel SHALL not be physically reused or moved
incompatibly.

#### Scenario: Cancelled Session

Given Attention still reads page

When Session is cancelled

Then page remains until quiescence.

### Requirement: Peer KV Access Is Capability-Gated

A Device SHALL consume KV owned by another Device directly only when explicit
peer capability and policy permit it.

#### Scenario: GPU1 reads GPU0 KV

Given peer-read unsupported

When Plan attempts direct access

Then zero-copy peer path is denied.