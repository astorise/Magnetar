## ADDED Requirements

### Requirement: Runtime Owns Memory Manager

Runtime SHALL instantiate and own the Memory Manager as a first-class subsystem.

#### Scenario: Runtime initialization

Given Runtime starts

When subsystems are initialized

Then Memory Manager is initialized as an explicit Runtime service.

---

### Requirement: Runtime Does Not Hide Memory In Compute

Runtime SHALL NOT hide allocation, residency, staging, pinned memory, or
zero-copy decisions inside Compute contract definitions.

#### Scenario: Compute request planned

Given a Compute request is validated

When memory feasibility is required

Then Runtime uses Memory Manager rather than Compute owning allocator state.

---

### Requirement: Runtime Does Not Hide Memory In Device

Runtime SHALL NOT hide global allocation policy inside Device metadata or Device
implementations.

#### Scenario: Device memory metadata available

Given a Device reports memory capacity

When Runtime admits work

Then Memory Manager consumes Device metadata

And Device does not own global memory policy.

---

### Requirement: Runtime Does Not Hide Memory In Provider

Runtime SHALL NOT hide Runtime-wide memory allocation policy inside Provider
execution code.

Providers may own native resources, but Runtime memory policy SHALL remain in
Memory Manager.

#### Scenario: Provider allocation required

Given Provider execution requires native memory

When Runtime plans execution

Then Memory Manager coordinates feasibility and pressure before Provider
submission where applicable.

---

### Requirement: Runtime Uses Memory Admission Before Scheduling

Runtime SHALL use Memory Manager admission before scheduling memory-dependent
work.

#### Scenario: Saturated memory pressure

Given Memory Manager reports saturated pressure

When Scheduler evaluates a new memory-heavy operation

Then Runtime policy decides queue, retry, or failure before Provider execution.

---

### Requirement: Runtime Separates Memory Pressure From Provider Failure

Runtime SHALL not classify memory pressure alone as Provider failure.

#### Scenario: Allocation pressure

Given memory pressure is saturated

When new work is rejected

Then Runtime reports memory admission failure

And does not mark the Provider failed solely for pressure.

---

### Requirement: Runtime Tracks Runtime-Owned Residency

Runtime SHALL track residency for Runtime-owned tensor and inference resources
through Memory Manager.

#### Scenario: Tensor materialized

Given a tensor is materialized in Device memory

When dependent work is planned

Then Runtime uses Memory Manager residency state.

---

### Requirement: Runtime Applies Memory Policy To Host Staging

Runtime SHALL apply Memory Manager policy before inserting host staging.

#### Scenario: Host staging permitted by request

Given host staging is permitted by Compute request

But Runtime memory policy denies staging due to pressure

When data movement is planned

Then Runtime rejects or delays staging according to policy.

---

### Requirement: Runtime Exposes Memory Diagnostics

Runtime SHALL expose stable diagnostics for memory decisions.

Diagnostics MAY explain allocation rejection, pending queue placement,
zero-copy rejection, staging denial, pressure, or residency conflict.

#### Scenario: Zero-copy rejected

Given zero-copy is unavailable

When Runtime reports planning diagnostics

Then diagnostics include a stable zero-copy rejection reason.
