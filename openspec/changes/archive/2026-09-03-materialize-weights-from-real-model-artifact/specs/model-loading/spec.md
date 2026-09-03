## ADDED Requirements

### Requirement: Weight Materialization Sources Real Artifact Bytes

Model Loading's weight-materialization phase SHALL be able to construct materialized tensor data from real Model Artifact bytes, using a format parser's generic tensor inventory to locate each tensor's byte range, not only from a pre-materialized in-memory source.

The construction step SHALL depend only on generic Model Artifact types, never on a concrete format parser crate.

#### Scenario: Materialize from a real Safetensors file

Given a real `.safetensors` file's bytes and its parsed generic tensor inventory

When weight materialization runs

Then it reads each tensor's declared byte range from the file bytes and produces the same materialized tensor data structure the existing in-memory materialization path produces

And no format-specific type crosses into the materialization step itself.

#### Scenario: Unsupported storage dtype is rejected structurally

Given a tensor's declared storage dtype is not one the Runtime's host tensor representation supports

When weight materialization attempts to read it

Then it returns a structured error rather than silently reinterpreting the bytes.

#### Scenario: Real and in-memory materialization agree

Given the same logical weights are available both as an in-memory source and as real artifact bytes

When both are materialized independently

Then they produce equal tensor data.

---

### Requirement: Weight Resource Completeness Gates Generation And Instance Lifecycle

A Model Instance SHALL NOT report Ready, and SHALL NOT be usable for generation, while any of its mandatory weight resources have not been materialized, admitted through the Memory Manager, and bound into `resource_bindings.weights`.

**Correction, not the original wording:** an earlier version of this requirement said Model Instance creation "MAY report an instance as structurally ready before" materialization, relying only on a deeper graph-dispatch-time check to fail closed. An external audit correctly identified that this was an incomplete guarantee: `acquire_usage`-style readiness checks that inspect only the instance's coarse lifecycle/readiness flag (not weight bindings) would incorrectly accept a not-yet-materialized instance as usable. The instance's own reported readiness SHALL be trustworthy on its own, not merely "safe in practice because something deeper happens to also check." `ModelLoadingCoordinator::load()` itself stays separate from materialization (the Lazy Loading Policy requirement is unaffected: `load()` still succeeds without weight bytes ready), but Model Instance creation SHALL leave the instance in a non-Ready lifecycle state until a subsequent, explicit weight-materialization step completes successfully and itself transitions the instance to Ready.

#### Scenario: An instance is not Ready until its weights are materialized

Given a Model Instance has just been created from a successfully loaded artifact

When no weight-materialization step has run yet for it

Then the instance's lifecycle and readiness both report a non-Ready state, and generation against it is rejected before any Kernel dispatches

#### Scenario: Weight materialization is what makes the instance Ready

Given a Model Instance's mandatory weight resources have all been materialized, admitted through the Memory Manager, and bound

When that materialization step completes

Then the instance transitions to Ready, and only then does generation against it become possible

#### Scenario: A failed or partial materialization never produces a Ready instance

Given weight materialization fails partway through, for any reason (memory admission denied, Provider write failure, residency registration failure)

When the failure is handled

Then every resource staged during that attempt is rolled back, and the instance is left in a Failed lifecycle state, never Ready

#### Scenario: A later, distinct materialization step remains architecturally valid

Given `load()` completed successfully under the Lazy Loading Policy, with weight materialization intentionally deferred to a distinct, later step

When that later step subsequently materializes, admits, and binds every mandatory weight

Then the instance becomes genuinely usable for generation at that point, and no change to `load()`'s own signature or contract was required to reach it.

### Requirement: Weight Materialization Is Transactional

Weight materialization SHALL admit each resource through the Memory Manager before writing it into Provider-owned storage, SHALL propagate every step's errors rather than discarding them, and SHALL roll back every resource staged during a failed attempt rather than leaving partial state behind.

#### Scenario: Memory admission precedes Provider materialization

Given a weight is about to be materialized

When its resource is staged

Then Memory Manager admission is attempted first, and Provider-owned storage is written to only after admission succeeds

#### Scenario: A residency registration failure is not silently discarded

Given a weight's Memory Manager admission and Provider write both succeed

When residency registration for that weight fails

Then the failure is propagated as a real error, not discarded, and triggers rollback of that weight and every weight staged before it in the same attempt

#### Scenario: A failure partway through rolls back every already-staged weight

Given weights 1 through N-1 were staged successfully in one materialization attempt

When weight N fails to stage, for any reason

Then weights 1 through N-1's Provider-owned storage and Memory Manager allocations are released, and none of them remain bound to the Model Instance

### Requirement: Unloading A Model Instance Releases Its Provider-Owned Weight Storage

Unloading a Model Instance SHALL release both its Memory Manager allocations and its Provider-owned weight Tensor Resources, not the allocations alone.

#### Scenario: Unload leaves no orphaned Provider-owned weight storage

Given a Model Instance whose weights were materialized into Provider-owned storage

When that instance is unloaded

Then every weight Tensor Resource bound to it is released from Provider-owned storage, in addition to its Memory Manager allocations being released

#### Scenario: Repeated load and unload does not accumulate Provider-owned storage

Given a Model Instance is repeatedly loaded and unloaded with no other instance created in between

When this is repeated many times

Then Provider-owned storage returns to its prior baseline after each unload, not growing unboundedly across cycles
