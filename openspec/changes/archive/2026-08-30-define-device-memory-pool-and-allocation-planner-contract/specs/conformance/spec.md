## ADDED Requirements
### Requirement: Memory Authority Separation Conformance

Conformance SHALL prove Memory Manager owns pool/allocation policy while
Provider owns native realization.

#### Scenario: Provider native pool

Given Provider supports native memory pool

When Runtime policy reserves KV capacity

Then Provider cannot override reservation.

### Requirement: Device Allocation Boundary Conformance

Conformance SHALL prove Device abstraction exposes no native allocation API.

#### Scenario: Device API inspection

Given GPU Device exists

When public Device contract is inspected

Then allocate/free/compact methods are absent.

### Requirement: Memory Pool Native Pointer Isolation Conformance

Conformance SHALL prove pools, blocks, leases, and Plans contain no native
pointer semantics.

#### Scenario: Allocation lease logged

Given Device allocation exists internally

When logical state is inspected

Then native address is absent.

### Requirement: Temporal Reuse Conformance

Conformance SHALL prove non-overlapping Tensor lifetimes SHALL share storage.

#### Scenario: A then C

Given A completion precedes C start

When AllocationPlan executes

Then same slot SHALL back both safely.

### Requirement: Asynchronous Reuse Safety Conformance

Conformance SHALL prove logical lifetime analysis does not free/reuse storage
before actual asynchronous completion.

#### Scenario: A graph node logically ended

Given Device execution still pending

When C wants same slot

Then reuse waits for CompletionToken.

### Requirement: Alignment Conformance

Conformance SHALL reject slot binding that violates Kernel/Provider alignment.

#### Scenario: 256-byte requirement

Given slot only guarantees 64-byte alignment

When Plan binds it

Then binding fails.

### Requirement: Hard Reservation Conformance

Conformance SHALL prove hard-reserved memory cannot be silently borrowed.

#### Scenario: KV reservation

Given optional tuning needs protected bytes

When pool is full outside reservation

Then tuning allocation fails.

### Requirement: Soft Borrowing Conformance

Conformance SHALL prove soft borrowing occurs only according to policy and
remains accounted.

#### Scenario: Borrowed workspace capacity

Given KV borrows soft capacity

When accounting is queried

Then borrowed amount is represented.

### Requirement: Watermark Reclamation Conformance

Conformance SHALL prove high-watermark policy can trigger reclamation without
freeing active memory.

#### Scenario: Pressure

Given pool exceeds high watermark

When reclamation runs

Then only eligible Resources are reclaimed.

### Requirement: Pending Reclaim Accounting Conformance

Conformance SHALL prove in-flight released memory is not counted as free.

#### Scenario: 1 GiB pending

Given Kernel still references released Resource

When admission checks capacity

Then 1 GiB is unavailable.

### Requirement: Fragmentation Classification Conformance

Conformance SHALL distinguish inability to realize large allocation from simple
aggregate free-byte exhaustion when allocator reports fragmentation.

#### Scenario: Fragmented pool

Given total free >= requested bytes

But no compatible region exists

When allocation fails

Then fragmentation error SHALL be emitted.

### Requirement: Compaction Safety Conformance

Conformance SHALL prove compaction does not relocate pinned, mapped, or
in-flight Resource unsafely.

#### Scenario: Active graph buffer

Given Resource requires stable address

When compaction runs

Then Resource is not moved.

### Requirement: Prepared Plan Memory Reservation Conformance

Conformance SHALL prove Plan cannot become ready under policy requiring
reservation when mandatory memory is unavailable.

#### Scenario: Workspace reservation fails

Given no compatible capacity

When Plan preparation completes

Then Plan is not marked ready.

### Requirement: KV Page Recycle Conformance

Conformance SHALL prove KV page is not recycled while any in-flight or shared
reference exists.

#### Scenario: Cancelled sequence

Given final Kernel still reads page

When Session ends

Then page remains pending reclaim.

### Requirement: Batch Memory Isolation Conformance

Conformance SHALL prove batch workspace cannot violate protected KV/weight pool
policy.

#### Scenario: Large batch spike

Given workspace request exceeds unreserved capacity

When batch is admitted

Then protected allocation class remains safe.

### Requirement: OOM Fallback Conformance

Conformance SHALL prove OOM fallback still obeys hard policy.

#### Scenario: Host spill forbidden

Given Device memory exhausted

When fallback evaluates spill

Then host-staging prohibition is respected.

### Requirement: Allocation Plan Cache Revalidation Conformance

Conformance SHALL prove cached AllocationPlan cannot bypass current pool
capacity/reservation/compatibility.

#### Scenario: Pool policy changed

Given cached Plan assumes old capacity

When reused

Then current policy is checked.

### Requirement: Memory Pool Observability Redaction Conformance

Conformance SHALL prove allocation diagnostics contain no native pointers,
native allocator handles, Tensor payloads, weights, KV contents, prompts,
secrets, or credentials.

#### Scenario: Pool OOM trace

Given Provider exposes native allocator details internally

When trace is exported

Then only safe logical/aggregate metadata remains.
