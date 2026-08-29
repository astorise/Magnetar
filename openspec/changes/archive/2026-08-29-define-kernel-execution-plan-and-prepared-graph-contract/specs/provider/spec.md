## ADDED Requirements
### Requirement: Provider May Prepare Execution Segment

Provider SHALL be able to support preparation of a Runtime-defined compatible execution
segment.

#### Scenario: CUDA graph capture

Given segment is compatible with CUDA graph capture

When Provider prepares it

Then Provider may return opaque ProviderPreparedSegmentId.

---

### Requirement: Prepared Segment State Is Provider Private

Provider SHALL retain native graph/pipeline/command state privately.

#### Scenario: QNN graph object

Given QNN Provider creates native graph

When Runtime stores segment binding

Then native object is not exposed.

---

### Requirement: Provider Prepared Segment Is Optional Capability

Provider without prepared-segment support SHALL remain conformant.

#### Scenario: Scalar CPU Provider

Given only individual Kernel execute exists

When Plan executes

Then Runtime dispatches prepared Kernels individually.

---

### Requirement: Provider Execute Does Not Replan

Provider execution SHALL not invoke Runtime Kernel selection/replanning as
hidden side effect.

#### Scenario: Prepared segment executes

Given binding is valid

When Provider executes it

Then Provider follows prepared contract or returns structured incompatibility
failure.

---

### Requirement: Provider Segment Invalidity Is Explicit

Provider SHALL report when prepared native segment state is no longer usable.

#### Scenario: Device context lost

Given native prepared graph becomes invalid

When Runtime tries to use/revalidate it

Then structured failure causes Plan invalidation/rebuild.
