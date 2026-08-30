## ADDED Requirements
### Requirement: Runtime Coordinates Allocation Planning

Runtime Memory Manager SHALL coordinate logical AllocationPlan construction for
Prepared execution.

#### Scenario: Model Instance preparation

Given graph, Kernels, and workspace requirements are known

When Runtime prepares Plan

Then compatible allocation slots and reservations SHALL be computed.

### Requirement: Runtime Avoids Native Allocate Free On Normal Hot Path

Where compatible pool-backed slots exist, normal token execution SHALL reuse
them rather than repeatedly invoking Provider native allocation/free.

#### Scenario: Decode loop

Given decode Plan remains stable

When 100 tokens execute

Then transient/workspace resources SHALL reuse pool storage.

### Requirement: Runtime Applies Memory Admission Before Unsafe Partial Work

Predictable memory infeasibility SHALL be detected before starting work that
cannot complete safely.

#### Scenario: KV growth impossible

Given required KV page cannot be reserved

When next decode step is admitted

Then Runtime SHALL fail/backpressure before launching dependent Kernel.

### Requirement: Runtime Handles Pool Pressure Explicitly

Runtime SHALL react to pool pressure through reclaim, fallback, or admission.

#### Scenario: High watermark

Given transient pool exceeds high watermark

When pressure policy runs

Then reclaimable Resources SHALL be trimmed.

### Requirement: Runtime Does Not Treat Pending Reclaim As Free

Asynchronously released memory SHALL remain unavailable until physically safe
for reuse.

#### Scenario: Cancelled generation

Given 256 MiB becomes pending reclaim

When another request arrives

Then Runtime capacity accounting does not promise those bytes immediately.

### Requirement: Runtime SHALL Choose Lower Workspace Kernel

Memory pressure SHALL affect Kernel selection among otherwise eligible candidates.

#### Scenario: Fast Kernel needs 1 GiB workspace

Given only 256 MiB compatible workspace remains

When alternative Kernel needs 128 MiB

Then selection SHALL choose the feasible candidate.

### Requirement: Runtime Keeps Pool Policy Outside Inference Request

Normal generation request SHALL not choose allocator implementation or native
pool.

#### Scenario: Client requests CUDA pool handle

Given request reaches Runtime Inference API

When validated

Then such native memory authority is outside API scope.