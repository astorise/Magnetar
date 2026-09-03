## ADDED Requirements

### Requirement: Model Loading Materializes Weight Resources

Model Loading SHALL provide a generic, artifact-source-agnostic weight
materialization phase, distinct from the aggregate residency allocation
`load()` performs, that creates one Tensor Resource per declared weight
through a registered Provider and records each resulting Tensor Resource
identity against the Model Instance it belongs to.

This phase SHALL NOT assume a specific Model Artifact format or a specific
model family: any weight source that can supply named tensors SHALL be able
to invoke it, whether the source is a test fixture or a real Model Artifact
parser.

#### Scenario: Fixture-sourced weights are materialized generically

Given a Model Instance is created from a fixture's in-memory weight tensors

When Model Loading materializes its weights

Then each weight is written into Provider storage and admitted through
Runtime's Memory Manager via the same generic materialization phase a real
Model Artifact loader would use.

#### Scenario: Materialization does not require load() to accept a Provider

Given a Model Instance whose weights are not yet materialized (a
lazily-loaded instance)

When `load()` completes for that instance

Then `load()` itself succeeds without requiring a Provider or weight byte
source, and materialization remains a distinct, later step.

---

### Requirement: Missing Weight Materialization Is Structurally Detectable

Runtime SHALL be able to determine, before first Kernel dispatch, whether a
Model Instance's declared weights were successfully materialized, rather
than that failure surfacing only as an opaque missing-resource error deep
inside generation.

#### Scenario: Weight materialization failure is visible at the boundary

Given weight materialization fails or was never attempted for a Model
Instance

When Runtime checks whether that instance can accept generation

Then Runtime can determine the materialization failure at that boundary,
not only after a Kernel fails to find a resource it needs.
