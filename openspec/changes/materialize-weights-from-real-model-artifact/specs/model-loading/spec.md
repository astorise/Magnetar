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

### Requirement: Weight Resource Completeness Gates Generation, Not Merely Instance Lifecycle

A Model Instance SHALL NOT be usable for generation while any of its mandatory weight resources have not been materialized, admitted through the Memory Manager, and bound into `resource_bindings.weights`, regardless of what the instance's coarse lifecycle/readiness flag currently reports.

This gate SHALL be enforced at the graph-dispatch boundary that resolves a weight edge to a bound Tensor Resource, not only by the instance's lifecycle state at creation time: `ModelLoadingCoordinator::load()` and Model Instance creation MAY report an instance as structurally ready before a later, distinct weight-materialization step runs (the Lazy Loading Policy requirement already permits `load()` to succeed without weight bytes ready), but generation against that instance SHALL fail closed, naming the missing weight, until materialization for it has actually completed.

#### Scenario: Generation against a not-yet-materialized weight fails closed

Given a Model Instance whose lifecycle already reports Ready but a required weight has not been bound into `resource_bindings.weights`

When graph execution resolves that weight's edge

Then execution fails with a structured error naming the missing weight, before any Kernel dispatches

And the failure does not depend on the instance's lifecycle/readiness flag having already been demoted.

#### Scenario: A later, distinct materialization step remains architecturally valid

Given `load()` completed successfully under the Lazy Loading Policy, with weight materialization intentionally deferred to a distinct, later step

When that later step subsequently materializes, admits, and binds every mandatory weight

Then the instance becomes genuinely usable for generation at that point, and no change to `load()`'s own signature or contract was required to reach it.
