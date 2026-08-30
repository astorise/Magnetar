## ADDED Requirements
### Requirement: KV Cache SHALL Use Dedicated Page Pool

KV-cache implementation SHALL support logically dedicated page-pool
allocation.

#### Scenario: Long-running server

Given many Sessions create and retire KV pages

When decode operates

Then pages can be leased/recycled without one native allocation per token.

### Requirement: KV Page Size Is Runtime Defined

KV page geometry SHALL follow KV-cache format and model execution requirements.

#### Scenario: Paged Attention

Given page represents fixed token block

When KVPagePool initializes

Then page size derives from dtype/head/layout requirements, not arbitrary native
allocator bucket.

### Requirement: KV Page Lease Is Session Or Sequence Scoped

Leased pages SHALL have explicit ownership.

#### Scenario: Sequence A

Given A grows context

When pages are allocated

Then leases are associated with A/session state.

### Requirement: KV Page Recycle Is Completion Safe

A page SHALL return to free list only after no in-flight execution can access
it.

#### Scenario: Sequence cancelled

Given last Attention Kernel still reads page

When Session ownership ends

Then page becomes pending reclaim until completion.

### Requirement: KV Pool Exhaustion Is Structured

Failure to acquire KV page SHALL not silently corrupt or overwrite another
Session's cache.

#### Scenario: No free pages

Given pool has no reclaimable page

When decode needs growth

Then Runtime applies backpressure/spill/failure policy explicitly.

### Requirement: Prefix Cache SHALL Retain KV Page Ownership

A page shared by Prefix Cache SHALL not recycle merely because original Session
ended.

#### Scenario: Prefix retained

Given shared prefix references pages

When source Session closes

Then pages remain leased until Prefix Cache releases them.