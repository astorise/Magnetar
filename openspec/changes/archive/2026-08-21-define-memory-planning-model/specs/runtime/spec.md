## ADDED Requirements

### Requirement: Memory Planning

The Runtime SHALL define a Memory Planning Model.

Memory planning SHALL be a Runtime responsibility.

Memory planning SHALL NOT be exposed as a direct WIT allocation Capability for
portable Components.

#### Scenario: Plan compute graph memory

Given a Compute Graph

When the Runtime validates it before execution

Then the Runtime creates or validates a Memory Plan for required tensor
resources, intermediate values, temporary buffers and outputs.

---

### Requirement: Opaque Memory

Memory used by Tensor Resources SHALL remain opaque to Components.

Components SHALL NOT receive raw pointers, native buffers, backend storage,
allocators, GPU pointers, queues, streams or Provider handles.

#### Scenario: Component receives tensor output

Given a Compute Graph produces a tensor output

When the output is returned to the Component

Then the Component receives an opaque Tensor Resource and portable metadata only.

---

### Requirement: Tensor Resource Lifetime

The Memory Planning Model SHALL track Tensor Resource lifetimes.

A Tensor Resource SHALL remain live while it may be observed, passed to another
operation, downloaded, transferred, materialized or used by a dependent
resource.

#### Scenario: Prevent premature release

Given a Tensor Resource is still referenced by a Compute Graph output

When the Runtime considers releasing memory

Then the Runtime keeps the resource live.

---

### Requirement: Intermediate Lifetime

The Memory Planning Model SHALL track intermediate value lifetimes inside a
Compute Graph.

Intermediate buffers MAY be reused only after their last required use.

#### Scenario: Reuse intermediate buffer

Given an intermediate tensor is no longer needed

When another compatible intermediate allocation is required

Then the Runtime may reuse the memory only if affinity and layout constraints
permit it.

---

### Requirement: Affinity-Aware Memory Planning

Memory planning SHALL respect Resource Affinity.

Provider-pinned or Device-bound resources SHALL NOT be planned for incompatible
Providers or Devices without an explicit transfer, copy, materialization or
reload step.

#### Scenario: Incompatible memory placement

Given a Tensor Resource is bound to one Device

When a Compute Graph is planned for another Device

Then the Runtime requires explicit data movement before planning execution.

---

### Requirement: Provider Advertisement Awareness

Memory planning SHALL use Provider Compute Advertisements.

Provider advertisements MAY define:

- Device memory limits
- maximum tensor byte size
- maximum element count
- supported layouts
- view consumption support
- materialization requirements
- data movement constraints

#### Scenario: Provider memory limit

Given a Provider advertises a maximum tensor byte size

When the Memory Plan contains a larger tensor

Then the Runtime rejects the plan before Provider execution.

---

### Requirement: Materialization Planning

Memory planning SHALL identify materialization requirements.

A tensor view SHALL NOT be silently materialized unless an explicit
materialization operation exists in the compute or data movement model.

#### Scenario: View requires materialization

Given a Tensor Resource is a view

And the selected Provider cannot consume the view directly

When memory planning occurs

Then the Runtime requires explicit materialization or rejects execution.

---

### Requirement: Data Movement Memory Planning

Memory planning SHALL include memory required for explicit data movement
operations.

Data movement operations include:

- upload
- download
- copy
- materialize
- transfer
- dtype conversion
- layout conversion

#### Scenario: Transfer requires temporary buffer

Given a Transfer operation requires an intermediate staging buffer

When memory planning occurs

Then the Runtime accounts for the staging buffer explicitly.

---

### Requirement: No Hidden CPU Staging

The Runtime SHALL NOT hide CPU staging as an invisible memory planning detail
when it changes placement, cost or synchronization behavior.

#### Scenario: Host-staged transfer

Given a transfer between Providers requires host staging

When the Runtime plans the transfer

Then host staging is represented in the Memory Plan or the transfer is rejected.

---

### Requirement: Memory Feasibility Validation

The Runtime SHALL validate memory feasibility before Provider execution when
sufficient metadata is available.

Validation SHALL consider:

- tensor descriptors
- byte-size calculations
- intermediate lifetimes
- output resources
- Device memory limits
- Provider memory limits
- transfer buffers
- materialization buffers

#### Scenario: Insufficient memory

Given a Compute Graph requires more memory than the selected Device advertises

When the Runtime validates the Memory Plan

Then the Runtime rejects execution with a structured out-of-memory or
memory-planning error.

---

### Requirement: Byte Size Validation

Memory planning SHALL validate tensor byte-size calculations.

The Runtime SHALL detect:

- element count overflow
- dtype byte-size overflow
- total byte-size overflow
- Provider maximum-size violation
- Device maximum-size violation

#### Scenario: Byte-size overflow

Given a Tensor Descriptor would overflow byte-size calculation

When memory planning occurs

Then the Runtime rejects the plan with a structured size-overflow error.

---

### Requirement: Memory Reuse Safety

The Runtime SHALL preserve correctness when reusing memory for intermediate values.

Memory reuse SHALL preserve correctness, Resource Affinity and observable
resource lifetimes.

#### Scenario: Unsafe reuse

Given an intermediate buffer is still required by a later operation

When the Runtime considers reusing it

Then the Runtime rejects reuse until the buffer is no longer live.

---

### Requirement: Output Resource Ownership

Memory planning SHALL define ownership for produced Tensor Resources.

Produced Tensor Resources SHALL carry Resource Affinity metadata.

#### Scenario: Produce output resource

Given a Compute Graph completes successfully

When output Tensor Resources are returned

Then each output records Provider and Device affinity when applicable.

---

### Requirement: Memory Pressure Reporting

The Runtime SHALL support stable memory pressure diagnostics when reporting memory planning failures.

Memory pressure diagnostics MAY include:

- estimated required bytes
- estimated peak bytes
- selected Provider identifier
- selected Device identifier
- rejected Device memory limits
- materialization cost
- transfer buffer cost

Diagnostics SHALL NOT expose native memory addresses, pointers, allocator
internals or backend storage handles.

#### Scenario: Report memory pressure

Given memory planning fails

When diagnostics are available

Then the Runtime may report stable memory pressure metadata.

---

### Requirement: Provider-Owned Native Allocation

Providers SHALL own native allocation implementation details.

Provider-owned allocation details include:

- allocator implementation
- memory pools
- backend storage representation
- native buffer handles
- queue-specific allocation behavior
- device-specific memory APIs

#### Scenario: Provider executes planned graph

Given the Runtime has validated a Memory Plan

When the Provider executes the graph

Then the Provider may allocate and optimize memory internally without exposing
native allocation details.

---

### Requirement: No Live Resource Migration

The Memory Planning Model SHALL NOT imply live resource migration.

Moving Provider-pinned resources requires explicit transfer, copy,
materialization, replay, reload or a future migration contract.

#### Scenario: Provider-pinned resource unavailable

Given a Provider-pinned Tensor Resource becomes unavailable

When execution depends on it

Then the Runtime reports interruption or failure instead of silently migrating
the resource.

---

### Requirement: Structured Memory Planning Errors

The Runtime SHALL return stable structured errors for memory planning failures.

Structured errors SHALL include categories for:

- memory planning failed
- out of memory
- resource exhausted
- size overflow
- incompatible resource affinity
- unsupported layout
- materialization required
- transfer required
- Provider memory limit exceeded
- Device memory limit exceeded

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report memory planning failure

Given memory planning fails

When the Runtime reports the failure

Then the error uses a stable structured memory planning error variant.
