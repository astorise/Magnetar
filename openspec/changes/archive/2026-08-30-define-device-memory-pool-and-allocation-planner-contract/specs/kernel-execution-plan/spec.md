## ADDED Requirements
### Requirement: Prepared Execution Plan SHALL Reference Allocation Plan

PreparedExecutionPlan SHALL reference a validated AllocationPlan generation.

#### Scenario: Decode plan

Given Kernel bindings and memory lifetimes are known

When Plan becomes ready

Then it SHALL reference precomputed Device allocation slots.

### Requirement: Plan Readiness SHALL Require Memory Reservation

A Plan SHALL NOT be marked READY if mandatory memory reservation cannot be
satisfied when policy requires pre-reservation.

#### Scenario: Required attention workspace unavailable

Given Plan requires 512 MiB protected workspace

When reservation fails

Then Plan remains not-ready or fails preparation.

### Requirement: Plan Resource Slot Is Logical

Prepared Plan SHALL refer to logical allocation slots rather than native
addresses.

#### Scenario: Runtime restart

Given Provider backing addresses change

When Plan is reconstructed

Then logical slot relationships can remain while native backing is recreated.

### Requirement: Allocation Plan Change SHALL Stale Plan

A compatible optimization of allocation strategy SHALL mark Prepared Plan stale
without changing semantics.

#### Scenario: Better reuse plan available

Given current Plan remains memory-safe

When new AllocationPlan reduces workspace

Then Runtime SHALL build replacement Plan generation.

### Requirement: Hard Memory Incompatibility Invalidates Plan

A Plan whose mandatory memory assumptions cannot be satisfied SHALL not accept
new work.

#### Scenario: Required pool removed

Given decode Plan requires dedicated Device-local KV pool

When pool becomes unavailable

Then Plan is invalidated or rebuilt before execution.

### Requirement: Address-Stable Prepared Segment Pins Required Slots

If Provider-prepared segment requires stable native addresses, Plan SHALL
declare corresponding logical slots non-relocatable for segment lifetime.

#### Scenario: Native graph capture

Given Provider says buffers must retain address

When AllocationPlan is generated

Then those slots are pinned.