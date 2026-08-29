## ADDED Requirements
### Requirement: Prepared Execution Plan

Runtime SHALL support a Prepared Execution Plan representing validated execution
decisions for a compatible Execution Graph context.

#### Scenario: Qwen decode graph prepared

Given Qwen decode Execution Graph is validated

And required Kernels are selected and prepared

When Plan preparation completes

Then Runtime may expose a ready Prepared Execution Plan for compatible decode
workload.

---

### Requirement: Plan References Semantic Graph

Prepared Execution Plan SHALL identify the Execution Graph from which it was
derived.

#### Scenario: Graph changes

Given Model Component emits a different semantic graph

When old Plan is considered

Then graph fingerprint mismatch prevents reuse.

---

### Requirement: Plan Does Not Redefine Semantics

Prepared Execution Plan SHALL preserve portable Operator semantics.

#### Scenario: MatMul node binding

Given graph requires MatMul

When Plan binds generated Kernel

Then binding implements validated MatMul semantics rather than alternate hidden
operation.

---

### Requirement: Plan Has Exact Kernel Bindings

Plan SHALL identify concrete selected Kernel implementation and specialization.

#### Scenario: Two qualified Attention Kernels

Given selection chose Kernel B specialization S4

When Plan becomes ready

Then execution does not silently switch to Kernel A without a new Plan
generation.

---

### Requirement: Prepared Kernel Handle Is Opaque

PreparedKernelId in Plan SHALL remain opaque Provider-owned state.

#### Scenario: Numeric ID

Given Plan contains PreparedKernelId 812

When Runtime executes

Then 812 is not interpreted as native pointer.

---

### Requirement: Plan Supports Logical Resource Slots

Dynamic Runtime resources SHALL be bound through logical Plan slots rather than
portable native addresses.

#### Scenario: Session KV cache

Given two Sessions execute same decode Plan

When resources are bound

Then each Session supplies its own compatible KV resources to Plan slots.

---

### Requirement: Plan Guard

Prepared Execution Plan SHALL define guards required for safe execution.

#### Scenario: Sequence outside range

Given Plan supports sequence <=4096

When sequence is 8192

Then guard fails before Kernel dispatch.

---

### Requirement: Guard Evaluation Is Bounded

Plan guard SHALL not invoke expensive planning operations.

#### Scenario: Guard fails

Given shape is incompatible

When guard checks

Then Runtime may choose alternate Plan/request re-plan but does not synchronously
compile or benchmark.

---

### Requirement: Plan May Be Stale

Plan SHALL be able to remain safe but suboptimal after optimization evidence changes.

#### Scenario: New faster Kernel promoted

Given existing Plan still references qualified/trusted compatible Kernel

When new Kernel becomes preferred

Then old Plan may become stale rather than immediately invalid.

---

### Requirement: Invalidated Plan Cannot Receive New Work

Hard-invalidated Plan SHALL not be used for new executions.

#### Scenario: Kernel revoked

Given Plan depends on revoked Kernel

When next execution begins

Then Plan is rejected.

---

### Requirement: Plan Replacement Is Atomic

Runtime SHALL publish replacement Plan generation atomically for new work.

#### Scenario: Generation replacement

Given generation 5 is active and 6 becomes ready

When generation 6 is published

Then new invocation sees complete generation 5 or 6, never mixed node bindings.

---

### Requirement: In-Flight Plan Remains Valid Until Safe Retirement

Retiring Plan generation SHALL remain alive while referenced by active
invocation when safe-completion policy permits.

#### Scenario: Decode operation in-flight

Given operation uses generation 5

When generation 6 is promoted

Then generation 5 native prepared state remains alive until operation completes.

---

### Requirement: Plan Build Is Outside Normal Hot Path

Normal Kernel execution SHALL not construct complete Prepared Execution Plan.

#### Scenario: Decode executes ready Plan

Given compatible Plan exists

When token step executes

Then Runtime uses guards/bindings instead of repeating full Registry selection.

---

### Requirement: Cached Plan Requires Revalidation

Cached Plan metadata SHALL not bypass current hard eligibility.

#### Scenario: Kernel revoked while Runtime stopped

Given persisted Plan references now-revoked Kernel

When Runtime restarts

Then Plan does not become ready.

---

### Requirement: Provider Prepared Segment Is Optional

A Plan SHALL function even when Provider does not support native graph capture.

#### Scenario: Reference CPU

Given Provider only executes individual Kernels

When Plan is prepared

Then valid Plan can dispatch node bindings without ProviderPreparedSegmentId.

---

### Requirement: Performance Feedback Does Not Mutate Plan In Place

Adaptive feedback SHALL request replacement/replanning rather than changing
active Plan bindings in place.

#### Scenario: Different Kernel becomes preferable

Given Performance Model changes ranking

When Runtime adapts

Then a new Plan generation is prepared and safely published.
